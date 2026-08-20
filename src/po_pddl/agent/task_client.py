"""OpenAI-compatible client that exchanges model tasks through the filesystem."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote_to_bytes, urlparse

AGENT_RUN_DIR_ENV = "PO_PDDL_AGENT_RUN_DIR"
_TASK_LOCK = threading.RLock()

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows falls back to the process-local lock.
    fcntl = None


class AgentTaskPending(RuntimeError):
    """Raised when the external agent must answer the active model task."""

    def __init__(self, task_id: str, task_dir: Path) -> None:
        super().__init__(f"Agent response required for {task_id}: {task_dir}")
        self.task_id = task_id
        self.task_dir = task_dir


def configured_agent_run_dir() -> Path | None:
    raw = os.getenv(AGENT_RUN_DIR_ENV)
    return Path(raw).expanduser().resolve() if raw else None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path, default: object) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _media_suffix(media_type: str) -> str:
    return mimetypes.guess_extension(media_type) or {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "video/mp4": ".mp4",
    }.get(media_type, ".bin")


def _decode_data_url(url: str) -> tuple[str, bytes]:
    header, separator, payload = url.partition(",")
    if not separator or not header.startswith("data:"):
        raise ValueError("Malformed media data URL.")
    media_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    raw = base64.b64decode(payload, validate=True) if ";base64" in header else unquote_to_bytes(payload)
    return media_type, raw


class AgentTaskStore:
    """Persistent request/response journal shared by all client instances."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.tasks_dir = self.run_dir / "tasks"
        self.media_dir = self.run_dir / "media"
        self.active_file = self.run_dir / "active_task.json"
        self.last_consumed_file = self.run_dir / "last_consumed_task.json"
        self.lock_file = self.run_dir / ".task_store.lock"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self):
        """Serialize journal updates across threads and worker shell processes."""

        with _TASK_LOCK:
            with self.lock_file.open("a+", encoding="utf-8") as lock_handle:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _materialize_media(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        media_type: str | None = None
        raw: bytes | None = None
        source_path: Path | None = None
        if value.startswith("data:"):
            media_type, raw = _decode_data_url(value)
        else:
            parsed = urlparse(value)
            if parsed.scheme == "file":
                source_path = Path(parsed.path)
            elif not parsed.scheme:
                candidate = Path(value).expanduser()
                if candidate.is_file():
                    source_path = candidate
            if source_path is not None:
                raw = source_path.read_bytes()
                media_type = mimetypes.guess_type(str(source_path))[0] or "application/octet-stream"
        if raw is None or media_type is None:
            return value
        digest = hashlib.sha256(raw).hexdigest()
        target = self.media_dir / f"{digest}{_media_suffix(media_type)}"
        if not target.exists():
            target.write_bytes(raw)
        return {
            "agent_media_path": str(target),
            "media_type": media_type,
            "sha256": digest,
            "source_path": str(source_path.resolve()) if source_path is not None else None,
        }

    def _normalize_value(self, value: Any, *, parent_key: str | None = None) -> Any:
        if isinstance(value, list):
            return [self._normalize_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._normalize_value(item) for item in value]
        if isinstance(value, dict):
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if key == "url" and parent_key in {"image_url", "video_url"}:
                    normalized[key] = self._materialize_media(item)
                else:
                    normalized[key] = self._normalize_value(item, parent_key=key)
            return normalized
        return value

    @staticmethod
    def _request_digest(request: dict[str, Any]) -> str:
        def portable(value: Any) -> Any:
            if isinstance(value, list):
                return [portable(item) for item in value]
            if isinstance(value, dict):
                if "agent_media_path" in value and "sha256" in value:
                    return {
                        "media_type": value.get("media_type"),
                        "sha256": value["sha256"],
                    }
                return {key: portable(item) for key, item in value.items()}
            return value

        canonical = json.dumps(portable(request), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _render_prompt(request: dict[str, Any]) -> str:
        sections = [
            "# External Agent Model Task",
            "",
            "Return only the response requested by the prompts. Do not edit pipeline artifacts directly.",
        ]
        for message in request.get("messages", []):
            role = str(message.get("role", "user")).upper()
            sections.extend(["", f"## {role}", ""])
            content = message.get("content", "")
            if isinstance(content, str):
                sections.append(content)
            else:
                sections.extend(["```json", json.dumps(content, indent=2, ensure_ascii=False), "```"])
        return "\n".join(sections).rstrip() + "\n"

    def _active_payload(self) -> dict[str, Any] | None:
        payload = _read_json(self.active_file, None)
        return payload if isinstance(payload, dict) else None

    def _task_dir(self, task_id: str) -> Path:
        return self.tasks_dir / task_id

    def _pending_task_dirs(self) -> list[Path]:
        pending = [
            path
            for path in self.tasks_dir.iterdir()
            if path.is_dir() and (path / "request.json").exists() and not (path / "response.txt").exists()
        ]
        return sorted(pending, key=lambda path: ((path / "request.json").stat().st_mtime_ns, path.name))

    def _refresh_active_file(self) -> list[Path]:
        pending = self._pending_task_dirs()
        if not pending:
            self.active_file.unlink(missing_ok=True)
        else:
            first = pending[0]
            _write_json(self.active_file, {"task_id": first.name, "task_dir": str(first)})
        return pending

    def request(self, raw_request: dict[str, Any]) -> str:
        normalized_request = self._normalize_value(raw_request)
        digest = self._request_digest(normalized_request)
        task_id = f"task_{digest[:20]}"
        task_dir = self._task_dir(task_id)
        response_file = task_dir / "response.txt"

        with self._locked():
            if response_file.exists():
                response = response_file.read_text(encoding="utf-8").strip()
                if response:
                    _write_json(self.last_consumed_file, {"task_id": task_id, "task_dir": str(task_dir)})
                    active = self._active_payload()
                    if active and active.get("task_id") == task_id:
                        self.active_file.unlink(missing_ok=True)
                    return response

            task_dir.mkdir(parents=True, exist_ok=True)
            request_file = task_dir / "request.json"
            if not request_file.exists():
                _write_json(
                    request_file,
                    {
                        "task_id": task_id,
                        "digest": digest,
                        **normalized_request,
                    },
                )
                (task_dir / "prompt.md").write_text(self._render_prompt(normalized_request), encoding="utf-8")
            self._refresh_active_file()
            raise AgentTaskPending(task_id, task_dir)

    def submit(self, response: str, *, task_id: str | None = None) -> Path:
        cleaned = response.strip()
        if not cleaned:
            raise ValueError("Agent response must not be empty.")
        with self._locked():
            pending = self._pending_task_dirs()
            if task_id is None and len(pending) > 1:
                raise ValueError("Multiple agent tasks are pending; pass --task-id explicitly.")
            active = self._active_payload()
            resolved_task_id = task_id or (pending[0].name if pending else "")
            if not resolved_task_id and active is not None:
                resolved_task_id = str(active.get("task_id") or "")
            if not resolved_task_id:
                raise ValueError("There is no active agent task.")
            task_dir = self._task_dir(resolved_task_id)
            if not (task_dir / "request.json").exists():
                raise FileNotFoundError(f"Unknown agent task: {resolved_task_id}")
            response_file = task_dir / "response.txt"
            response_file.write_text(cleaned + "\n", encoding="utf-8")
            _write_json(task_dir / "submission.json", {"task_id": resolved_task_id, "status": "submitted"})
            self._refresh_active_file()
            return response_file

    def reject_last_consumed(self, error: BaseException) -> dict[str, Any] | None:
        with self._locked():
            payload = _read_json(self.last_consumed_file, None)
            if not isinstance(payload, dict):
                return None
            task_id = str(payload.get("task_id") or "")
            if not task_id:
                return None
            task_dir = self._task_dir(task_id)
            response_file = task_dir / "response.txt"
            if not response_file.exists():
                return None
            attempts_dir = task_dir / "rejected_attempts"
            attempts_dir.mkdir(parents=True, exist_ok=True)
            attempt_index = len(list(attempts_dir.glob("response_*.txt"))) + 1
            rejected_response = attempts_dir / f"response_{attempt_index:03d}.txt"
            response_file.replace(rejected_response)
            error_payload = {
                "task_id": task_id,
                "status": "response_rejected",
                "error_type": type(error).__name__,
                "error": str(error),
                "rejected_response": str(rejected_response),
            }
            _write_json(task_dir / "validation.json", error_payload)
            self._refresh_active_file()
            self.last_consumed_file.unlink(missing_ok=True)
            return error_payload

    def reopen(self, task_id: str, reason: str) -> dict[str, Any]:
        """Reject a cached response after semantic review and reactivate its task."""
        with self._locked():
            task_dir = self._task_dir(task_id)
            if not (task_dir / "request.json").exists():
                raise FileNotFoundError(f"Unknown agent task: {task_id}")
            response_file = task_dir / "response.txt"
            if not response_file.exists():
                raise FileNotFoundError(f"Task {task_id} has no submitted response to reopen.")
            attempts_dir = task_dir / "rejected_attempts"
            attempts_dir.mkdir(parents=True, exist_ok=True)
            attempt_index = len(list(attempts_dir.glob("response_*.txt"))) + 1
            rejected_response = attempts_dir / f"response_{attempt_index:03d}.txt"
            response_file.replace(rejected_response)
            payload = {
                "task_id": task_id,
                "status": "response_rejected",
                "error_type": "SemanticReview",
                "error": reason.strip() or "Rejected by semantic review.",
                "rejected_response": str(rejected_response),
            }
            _write_json(task_dir / "validation.json", payload)
            self._refresh_active_file()
            self.last_consumed_file.unlink(missing_ok=True)
            return payload

    def begin_execution(self) -> None:
        """Start a fresh pipeline attempt for response-validation tracking."""
        with self._locked():
            self.last_consumed_file.unlink(missing_ok=True)

    def status(self) -> dict[str, Any]:
        with self._locked():
            pending = self._refresh_active_file()
            if not pending:
                return {"status": "idle", "run_dir": str(self.run_dir)}
            pending_payloads = []
            for task_dir in pending:
                pending_payloads.append(
                    {
                        "task_id": task_dir.name,
                        "task_dir": str(task_dir),
                        "prompt_file": str(task_dir / "prompt.md"),
                        "request_file": str(task_dir / "request.json"),
                        "validation_file": (
                            str(task_dir / "validation.json")
                            if (task_dir / "validation.json").exists()
                            else None
                        ),
                    }
                )
            first = pending_payloads[0]
            return {
                "status": "pending_response",
                "run_dir": str(self.run_dir),
                "pending_count": len(pending_payloads),
                **first,
                "pending_tasks": pending_payloads,
            }


class _AgentCompletions:
    def __init__(self, store: AgentTaskStore) -> None:
        self._store = store

    def create(self, **kwargs: Any) -> Any:
        reply = self._store.request(kwargs)
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0)
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class AgentTaskClient:
    """Expose ``chat.completions.create`` through a resumable task journal."""

    def __init__(self, run_dir: str | Path) -> None:
        self.store = AgentTaskStore(Path(run_dir))
        self.chat = SimpleNamespace(completions=_AgentCompletions(self.store))


__all__ = [
    "AGENT_RUN_DIR_ENV",
    "AgentTaskClient",
    "AgentTaskPending",
    "AgentTaskStore",
    "configured_agent_run_dir",
]
