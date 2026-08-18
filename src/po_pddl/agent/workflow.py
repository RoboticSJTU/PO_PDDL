"""Resumable domain and problem workflows driven by an external coding agent."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from po_pddl.config import (
    DEFAULT_MODEL,
    DomainExtensionConfig,
    DomainGenerationConfig,
    LLMSettings,
    ProblemGenerationConfig,
)
from po_pddl.core.linter import lint_domain_text, lint_texts
from po_pddl.domain_generation.service import extend_domain, generate_domain
from po_pddl.problem_generation.service import generate_problem

from .task_client import AGENT_RUN_DIR_ENV, AgentTaskPending, AgentTaskStore
from .worker_pool import (
    DEFAULT_POOL_SIZE,
    DEFAULT_TASKS_PER_WORKER,
    write_assignment_manifests,
)

MANIFEST_NAME = "workflow.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _absolute(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())


def _existing_path(value: str | None, *, label: str, directory: bool | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if directory is True and not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    if directory is False and not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _lint_payload(result: Any) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "diagnostics": [asdict(item) for item in result.diagnostics],
    }


def _require_valid_pddl(result: Any, *, label: str) -> dict[str, Any]:
    payload = _lint_payload(result)
    if not result.ok:
        summary = "; ".join(f"{item.code}: {item.message}" for item in result.errors[:10])
        raise ValueError(f"Generated {label} failed PDDL validation: {summary}")
    return payload


@contextmanager
def _agent_environment(run_dir: Path) -> Iterator[None]:
    previous = os.environ.get(AGENT_RUN_DIR_ENV)
    os.environ[AGENT_RUN_DIR_ENV] = str(run_dir)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(AGENT_RUN_DIR_ENV, None)
        else:
            os.environ[AGENT_RUN_DIR_ENV] = previous


class AgentWorkflow:
    """Persist workflow configuration and advance it one dependency-safe task batch at a time."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.manifest_file = self.run_dir / MANIFEST_NAME
        self.store = AgentTaskStore(self.run_dir)

    def initialize(self, workflow: str, arguments: dict[str, Any], *, replace: bool = False) -> dict[str, Any]:
        if self.manifest_file.exists() and not replace:
            raise FileExistsError(
                f"Workflow already exists at {self.manifest_file}. Use --replace only to start it over."
            )
        if workflow not in {"domain", "extension", "problem"}:
            raise ValueError(f"Unsupported workflow: {workflow}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "workflow": workflow,
            "status": "initialized",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "arguments": arguments,
            "result": None,
            "error": None,
        }
        _atomic_json(self.manifest_file, manifest)
        return manifest

    def load(self) -> dict[str, Any]:
        if not self.manifest_file.exists():
            raise FileNotFoundError(f"Agent workflow is not initialized: {self.manifest_file}")
        payload = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid workflow manifest: {self.manifest_file}")
        return payload

    def _save(self, manifest: dict[str, Any]) -> dict[str, Any]:
        manifest["updated_at"] = _utc_now()
        _atomic_json(self.manifest_file, manifest)
        return manifest

    @staticmethod
    def _llm(arguments: dict[str, Any], *, default_max_tokens: int) -> LLMSettings:
        return LLMSettings(
            model=str(arguments.get("model") or DEFAULT_MODEL),
            temperature=arguments.get("temperature"),
            max_tokens=int(arguments.get("max_tokens", default_max_tokens)),
        )

    def _run_domain(self, arguments: dict[str, Any]) -> dict[str, Any]:
        input_dir = _existing_path(arguments["input_dir"], label="input_dir", directory=True)
        assert input_dir is not None
        result = generate_domain(
            DomainGenerationConfig(
                input_dir=input_dir,
                output_dir=Path(arguments["output_dir"]),
                llm=self._llm(arguments, default_max_tokens=5000),
                max_workers=int(arguments.get("max_workers", 1)),
                max_iterations=int(arguments.get("max_iterations", 3)),
                smoothing=float(arguments.get("smoothing", 0.0)),
                annotation_fps=float(arguments.get("annotation_fps", 2.0)),
                annotation_video_types=tuple(arguments.get("annotation_video_types") or ()),
                run_stages=tuple(arguments["run_stages"]) if arguments.get("run_stages") else None,
                keep_all_intersection_preconditions=bool(arguments.get("keep_all_intersection_preconditions", False)),
                verbose=bool(arguments.get("verbose", False)),
            )
        )
        payload = result.to_dict()
        domain_file = Path(result.merged_domain_file)
        payload["pddl_validation"] = _require_valid_pddl(
            lint_domain_text(domain_file.read_text(encoding="utf-8")),
            label="domain",
        )
        return payload

    def _run_extension(self, arguments: dict[str, Any]) -> dict[str, Any]:
        bundle_dir = _existing_path(arguments["bundle_dir"], label="bundle_dir", directory=True)
        input_dir = _existing_path(arguments["input_dir"], label="input_dir", directory=True)
        assert bundle_dir is not None and input_dir is not None
        result = extend_domain(
            DomainExtensionConfig(
                bundle_dir=bundle_dir,
                input_dir=input_dir,
                output_dir=Path(arguments["output_dir"]),
                llm=self._llm(arguments, default_max_tokens=5000),
                max_workers=int(arguments.get("max_workers", 1)),
                max_iterations=int(arguments.get("max_iterations", 3)),
                smoothing=float(arguments.get("smoothing", 0.0)),
                annotation_fps=float(arguments.get("annotation_fps", 2.0)),
                verbose=bool(arguments.get("verbose", False)),
            )
        )
        payload = result.to_dict()
        domain_file = Path(result.merged_domain_file)
        payload["pddl_validation"] = _require_valid_pddl(
            lint_domain_text(domain_file.read_text(encoding="utf-8")),
            label="extended domain",
        )
        return payload

    def _run_problem(self, arguments: dict[str, Any]) -> dict[str, Any]:
        domain_file = _existing_path(arguments["domain_file"], label="domain_file", directory=False)
        image_path = _existing_path(arguments["image_path"], label="image_path")
        final_bundle_dir = _existing_path(
            arguments.get("final_bundle_dir"), label="final_bundle_dir", directory=True
        )
        objects_file = _existing_path(arguments.get("objects_file"), label="objects_file", directory=False)
        reuse_problem_file = _existing_path(
            arguments.get("reuse_problem_file"), label="reuse_problem_file", directory=False
        )
        assert domain_file is not None and image_path is not None
        output_file = Path(arguments["output_file"])
        result = generate_problem(
            ProblemGenerationConfig(
                domain_file=domain_file,
                image_path=image_path,
                instruction=str(arguments["instruction"]),
                initial_state_hint=arguments.get("initial_state_hint"),
                output_file=output_file,
                final_bundle_dir=final_bundle_dir,
                objects_file=objects_file,
                reuse_problem_file=reuse_problem_file,
                problem_name=arguments.get("problem_name"),
                llm=self._llm(arguments, default_max_tokens=4096),
                max_workers=int(arguments.get("max_workers", 8)),
                inference_strategy=str(arguments.get("inference_strategy", "batch")),
                inference_batch_size=int(arguments.get("inference_batch_size", 20)),
                prior_data_confidence=float(arguments.get("prior_data_confidence", 0.0)),
                close_domain=bool(arguments.get("close_domain", False)),
                skip_init_observation=bool(arguments.get("skip_init_observation", False)),
                verbose=bool(arguments.get("verbose", False)),
            )
        )
        validation = _require_valid_pddl(
            lint_texts(
                domain_file.read_text(encoding="utf-8"),
                output_file.read_text(encoding="utf-8"),
            ),
            label="domain/problem pair",
        )
        return {
            "output_file": str(output_file),
            "problem_name": result.spec.problem_name,
            "visible_object_count": len(result.visible_objects),
            "predicate_judgment_count": len(result.predicate_judgments),
            "has_observation_module": result.spec.has_observation_module,
            "reused_problem_file": result.reused_problem_file,
            "diagnostics": result.diagnostics,
            "pddl_validation": validation,
        }

    def advance(self) -> dict[str, Any]:
        manifest = self.load()
        if manifest.get("status") == "complete":
            return self.status()
        self.store.begin_execution()
        manifest["status"] = "running"
        manifest["error"] = None
        self._save(manifest)
        try:
            with _agent_environment(self.run_dir):
                workflow = manifest["workflow"]
                if workflow == "domain":
                    result = self._run_domain(manifest["arguments"])
                elif workflow == "extension":
                    result = self._run_extension(manifest["arguments"])
                else:
                    result = self._run_problem(manifest["arguments"])
        except AgentTaskPending:
            manifest["status"] = "awaiting_response"
            self._save(manifest)
            return self.status()
        except Exception as error:
            rejection = self.store.reject_last_consumed(error)
            if rejection is not None:
                manifest["status"] = "awaiting_revision"
                manifest["error"] = rejection
                self._save(manifest)
                return self.status()
            manifest["status"] = "failed"
            manifest["error"] = {"error_type": type(error).__name__, "error": str(error)}
            self._save(manifest)
            raise
        manifest["status"] = "complete"
        manifest["result"] = result
        self._save(manifest)
        return self.status()

    def submit(self, response: str, *, task_id: str | None = None) -> dict[str, Any]:
        response_file = self.store.submit(response, task_id=task_id)
        manifest = self.load()
        manifest["status"] = "response_submitted"
        manifest["error"] = None
        self._save(manifest)
        task_status = self.store.status()
        return {
            "workflow": manifest["workflow"],
            "status": "response_submitted",
            "run_dir": str(self.run_dir),
            "task_id": response_file.parent.name,
            "response_file": str(response_file),
            "pending_count": int(task_status.get("pending_count", 0)),
        }

    def reopen(self, task_id: str, reason: str) -> dict[str, Any]:
        rejection = self.store.reopen(task_id, reason)
        manifest = self.load()
        manifest["status"] = "awaiting_revision"
        manifest["result"] = None
        manifest["error"] = rejection
        self._save(manifest)
        return self.status()

    def dispatch(
        self,
        *,
        worker_count: int = DEFAULT_POOL_SIZE,
        tasks_per_worker: int = DEFAULT_TASKS_PER_WORKER,
    ) -> dict[str, Any]:
        """Advance when possible, then emit compact manifests for persistent workers."""

        status = self.status()
        if status["active_task"] is None and status["status"] != "complete":
            status = self.advance()
        active_task = status.get("active_task")
        if active_task is None:
            result = status.get("result") or {}
            result_keys = (
                "merged_domain_file",
                "final_bundle_dir",
                "total_episode_count",
                "output_file",
                "problem_name",
                "visible_object_count",
                "predicate_judgment_count",
                "has_observation_module",
                "pddl_validation",
            )
            return {
                "workflow": status["workflow"],
                "status": status["status"],
                "run_dir": status["run_dir"],
                "result": {key: result[key] for key in result_keys if key in result},
                "error": status.get("error"),
            }

        configured_workers = max(1, int(self.load()["arguments"].get("max_workers", worker_count)))
        assignments = write_assignment_manifests(
            self.run_dir,
            active_task["pending_tasks"],
            worker_count=min(worker_count, configured_workers),
            tasks_per_worker=tasks_per_worker,
        )
        return {
            "workflow": status["workflow"],
            "workflow_status": status["status"],
            **assignments,
        }

    def status(self) -> dict[str, Any]:
        manifest = self.load()
        task_status = self.store.status()
        return {
            "workflow": manifest["workflow"],
            "status": manifest["status"],
            "run_dir": str(self.run_dir),
            "active_task": task_status if task_status.get("status") != "idle" else None,
            "result": manifest.get("result"),
            "error": manifest.get("error"),
        }


def domain_arguments(**kwargs: Any) -> dict[str, Any]:
    payload = dict(kwargs)
    for key in ("input_dir", "output_dir"):
        payload[key] = _absolute(payload[key])
    return payload


def extension_arguments(**kwargs: Any) -> dict[str, Any]:
    payload = dict(kwargs)
    for key in ("bundle_dir", "input_dir", "output_dir"):
        payload[key] = _absolute(payload[key])
    return payload


def problem_arguments(**kwargs: Any) -> dict[str, Any]:
    payload = dict(kwargs)
    for key in (
        "domain_file",
        "image_path",
        "output_file",
        "final_bundle_dir",
        "objects_file",
        "reuse_problem_file",
    ):
        payload[key] = _absolute(payload.get(key))
    return payload


__all__ = [
    "AgentWorkflow",
    "domain_arguments",
    "extension_arguments",
    "problem_arguments",
]
