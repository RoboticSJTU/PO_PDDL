"""Small synchronous client for persistent Codex app-server model workers."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_WORKER_INSTRUCTIONS = (
    "Answer this model task directly and return only the requested response. "
    "Do not inspect the repository, load skills, call tools, or explain your process."
)


@dataclass(frozen=True)
class AppServerTask:
    model: str
    base_instructions: str
    input_items: list[dict[str, Any]]


def request_to_app_server_task(request: dict[str, Any]) -> AppServerTask:
    """Convert a materialized OpenAI-compatible request to app-server inputs."""

    instruction_parts: list[str] = []
    input_items: list[dict[str, Any]] = []
    for message in request.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower()
        content = message.get("content", "")
        if role in {"system", "developer"}:
            instruction_parts.extend(_extract_text_blocks(content))
            continue
        if role not in {"user", "assistant"}:
            continue
        role_prefix = "" if role == "user" else "ASSISTANT CONTEXT:\n"
        if isinstance(content, str):
            input_items.append({"type": "text", "text": role_prefix + content})
            continue
        text_prefix_pending = bool(role_prefix)
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "")
            if block_type == "text":
                text = str(block.get("text") or "")
                if text_prefix_pending:
                    text = role_prefix + text
                    text_prefix_pending = False
                input_items.append({"type": "text", "text": text})
            elif block_type == "image_url":
                input_items.append(_convert_image_block(block))
            elif block_type in {"video_url", "audio_url"}:
                raise ValueError(f"Persistent Codex workers do not support {block_type} inputs yet")

    if not input_items:
        raise ValueError("Codex app-server task has no user input")
    base_instructions = "\n\n".join(part for part in instruction_parts if part.strip())
    base_instructions = f"{base_instructions}\n\n{_WORKER_INSTRUCTIONS}".strip()
    return AppServerTask(
        model=str(request.get("model") or "gpt-5.6-sol"),
        base_instructions=base_instructions,
        input_items=input_items,
    )


def _extract_text_blocks(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    return [
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]


def _convert_image_block(block: dict[str, Any]) -> dict[str, Any]:
    image = block.get("image_url")
    if not isinstance(image, dict):
        raise ValueError("image_url block must contain an object")
    value = image.get("url")
    detail = str(image.get("detail") or "high")
    if isinstance(value, dict):
        media_path = str(value.get("agent_media_path") or "").strip()
        if not media_path:
            raise ValueError("Materialized image is missing agent_media_path")
        return {"type": "localImage", "path": media_path, "detail": detail}
    url = str(value or "").strip()
    if not url:
        raise ValueError("image_url block is missing a URL")
    return {"type": "image", "url": url, "detail": detail}


class CodexAppServerClient:
    """Reuse one app-server process while isolating every task in a fresh thread."""

    def __init__(
        self,
        *,
        codex_executable: str = "codex",
        cwd: str | Path,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.cwd = Path(cwd).resolve()
        self.timeout_seconds = float(timeout_seconds)
        self._next_request_id = 1
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._deferred_messages: deque[dict[str, Any]] = deque()
        self._stderr_lines: list[str] = []
        self._process = subprocess.Popen(
            [codex_executable, "app-server", "--stdio"],
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._request(
            "initialize",
            {"clientInfo": {"name": "po-pddl", "version": "0.1"}},
        )
        self._send({"method": "initialized", "params": {}})

    def complete(self, request: dict[str, Any]) -> str:
        task = request_to_app_server_task(request)
        thread_response = self._request(
            "thread/start",
            {
                "model": task.model,
                "cwd": str(self.cwd),
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "ephemeral": True,
                "baseInstructions": task.base_instructions,
            },
        )
        thread_id = str(thread_response["thread"]["id"])
        turn_response = self._request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": task.input_items,
                "effort": "low",
                "summary": "none",
            },
        )
        turn_id = str(turn_response["turn"]["id"])
        answer = ""
        while True:
            message = self._receive()
            if "id" in message:
                self._reject_server_request(message)
                continue
            method = message.get("method")
            params = message.get("params") or {}
            if method == "item/completed" and params.get("threadId") == thread_id:
                item = params.get("item") or {}
                if item.get("type") == "agentMessage":
                    answer = str(item.get("text") or "").strip()
            if method == "turn/completed" and params.get("threadId") == thread_id:
                turn = params.get("turn") or {}
                if str(turn.get("id")) != turn_id:
                    continue
                if str(turn.get("status")) != "completed":
                    raise RuntimeError(f"Codex turn ended with status={turn.get('status')!r}")
                if not answer:
                    raise RuntimeError("Codex turn completed without an agent message")
                return answer

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)

    def __enter__(self) -> CodexAppServerClient:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _send(self, payload: dict[str, Any]) -> None:
        if self._process.stdin is None:
            raise RuntimeError("Codex app-server stdin is unavailable")
        self._process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_request_id
        self._next_request_id += 1
        self._send({"id": request_id, "method": method, "params": params})
        deferred: list[dict[str, Any]] = []
        while True:
            message = self._receive()
            if message.get("id") == request_id:
                self._deferred_messages.extendleft(reversed(deferred))
                if "error" in message:
                    raise RuntimeError(f"Codex app-server {method} failed: {message['error']}")
                return dict(message.get("result") or {})
            if "id" in message:
                self._reject_server_request(message)
            else:
                deferred.append(message)

    def _receive(self) -> dict[str, Any]:
        if self._deferred_messages:
            return self._deferred_messages.popleft()
        try:
            return self._messages.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            stderr_tail = "".join(self._stderr_lines[-20:]).strip()
            raise TimeoutError(
                f"Timed out waiting for Codex app-server after {self.timeout_seconds:g}s. {stderr_tail}"
            ) from exc

    def _reject_server_request(self, message: dict[str, Any]) -> None:
        if "method" not in message:
            return
        self._send(
            {
                "id": message["id"],
                "error": {"code": -32601, "message": "Model tools are disabled for PO-PDDL semantic tasks"},
            }
        )

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                self._messages.put(payload)

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr_lines.append(line)
            if len(self._stderr_lines) > 200:
                del self._stderr_lines[:100]


__all__ = [
    "AppServerTask",
    "CodexAppServerClient",
    "request_to_app_server_task",
]
