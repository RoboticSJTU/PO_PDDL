"""OpenAI-compatible adapter backed by the locally authenticated Codex CLI."""

from __future__ import annotations

import base64
import mimetypes
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote_to_bytes, urlparse

CODEX_CLI_BASE_URL = "codex-cli://local"


class CodexCLIError(RuntimeError):
    """Raised when a non-interactive Codex invocation fails."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class _PromptInput:
    text: str
    image_paths: tuple[Path, ...]


def is_codex_cli_base_url(base_url: str | None) -> bool:
    return bool(base_url and base_url.rstrip("/").lower() == CODEX_CLI_BASE_URL)


def _resolve_executable(explicit: str | None = None) -> str:
    configured = explicit or os.getenv("PO_PDDL_CODEX_EXECUTABLE")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path.resolve())
        discovered = shutil.which(configured)
        if discovered:
            return discovered
        raise FileNotFoundError(f"Configured Codex executable was not found: {configured}")

    standalone = Path.home() / ".local" / "bin" / "codex"
    if standalone.is_file():
        return str(standalone)
    discovered = shutil.which("codex")
    if discovered:
        return discovered
    raise FileNotFoundError(
        "Codex CLI was not found. Install it with "
        "`curl -fsSL https://chatgpt.com/codex/install.sh | sh`."
    )


def _decode_data_url(url: str, target_dir: Path, index: int) -> Path:
    header, separator, payload = url.partition(",")
    if not separator or not header.startswith("data:"):
        raise ValueError("Malformed image data URL.")
    media_type = header[5:].split(";", 1)[0] or "image/jpeg"
    if not media_type.startswith("image/"):
        raise ValueError(f"Codex CLI adapter only supports image attachments, got {media_type!r}.")
    raw = base64.b64decode(payload, validate=True) if ";base64" in header else unquote_to_bytes(payload)
    suffix = mimetypes.guess_extension(media_type) or ".jpg"
    target = target_dir / f"input_image_{index:03d}{suffix}"
    target.write_bytes(raw)
    return target


def _materialize_image(url: str, target_dir: Path, index: int) -> Path:
    if url.startswith("data:"):
        return _decode_data_url(url, target_dir, index)
    parsed = urlparse(url)
    if parsed.scheme == "file":
        path = Path(parsed.path)
    elif not parsed.scheme:
        path = Path(url)
    else:
        raise ValueError(
            "Codex CLI accepts local image files. Remote image URLs must be downloaded "
            "before calling the pipeline."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Codex image input does not exist: {path}")
    return path.resolve()


def _render_messages(messages: list[dict[str, Any]], target_dir: Path) -> _PromptInput:
    sections: list[str] = [
        "Act as a stateless language-model backend. Do not inspect the workspace, run tools, "
        "or modify files. Follow the supplied instructions and return only the requested answer."
    ]
    images: list[Path] = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = message.get("content", "")
        text_parts: list[str] = []
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                block_type = str(block.get("type", "")).lower()
                if block_type in {"text", "input_text"}:
                    text_parts.append(str(block.get("text", "")))
                elif block_type in {"image_url", "input_image"}:
                    value = block.get("image_url")
                    if isinstance(value, dict):
                        value = value.get("url")
                    if not isinstance(value, str):
                        raise ValueError("Image content block is missing an image URL.")
                    images.append(_materialize_image(value, target_dir, len(images)))
                elif block_type == "video_url":
                    raise ValueError(
                        "Codex CLI does not accept video attachments. Configure the pipeline "
                        "to sample video frames and pass those images instead."
                    )
        sections.append(f"\n--- {role} ---\n" + "\n".join(text_parts))
    return _PromptInput(text="\n".join(sections).strip(), image_paths=tuple(images))


def _is_retryable_failure(message: str) -> bool:
    normalized = message.lower()
    return any(
        marker in normalized
        for marker in (
            "connection",
            "temporarily unavailable",
            "timed out",
            "timeout",
            "rate limit",
            "429",
            "server error",
        )
    )


class _CodexCompletions:
    def __init__(self, owner: "CodexCLIClient") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        return self._owner._create_completion(**kwargs)


class CodexCLIClient:
    """Expose ``chat.completions.create`` while invoking ``codex exec`` locally."""

    def __init__(self, *, executable: str | None = None, timeout_seconds: float | None = None) -> None:
        self.executable = _resolve_executable(executable)
        self.timeout_seconds = timeout_seconds or float(os.getenv("PO_PDDL_LLM_TIMEOUT_SECONDS", "180"))
        if self.timeout_seconds <= 0:
            raise ValueError("PO_PDDL_LLM_TIMEOUT_SECONDS must be positive")
        self.chat = SimpleNamespace(completions=_CodexCompletions(self))

    def _create_completion(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Codex CLI completion requires a non-empty messages list.")
        model = str(kwargs.get("model") or "").strip()
        max_tokens = kwargs.get("max_completion_tokens", kwargs.get("max_tokens"))

        with tempfile.TemporaryDirectory(prefix="po_pddl_codex_") as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            prompt = _render_messages(messages, tmpdir)
            if max_tokens:
                prompt_text = f"{prompt.text}\n\nKeep the final answer within approximately {int(max_tokens)} tokens."
            else:
                prompt_text = prompt.text
            output_path = tmpdir / "last_message.txt"
            command = [
                self.executable,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "--cd",
                str(tmpdir),
                "--output-last-message",
                str(output_path),
            ]
            if model:
                command.extend(["--model", model])
            reasoning_effort = os.getenv("PO_PDDL_CODEX_REASONING_EFFORT")
            if reasoning_effort:
                command.extend(["--config", f'model_reasoning_effort="{reasoning_effort}"'])
            for image_path in prompt.image_paths:
                command.extend(["--image", str(image_path)])
            command.append("-")

            try:
                completed = subprocess.run(
                    command,
                    input=prompt_text,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise CodexCLIError(
                    f"Codex CLI timed out after {self.timeout_seconds:.0f} seconds.",
                    retryable=True,
                ) from exc

            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "unknown Codex CLI error").strip()
                raise CodexCLIError(
                    f"Codex CLI exited with status {completed.returncode}: {detail}",
                    retryable=_is_retryable_failure(detail),
                )
            reply = output_path.read_text(encoding="utf-8").strip() if output_path.is_file() else ""
            if not reply:
                raise CodexCLIError("Codex CLI returned an empty final message.", retryable=True)
            usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0)
            message = SimpleNamespace(content=reply)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)
