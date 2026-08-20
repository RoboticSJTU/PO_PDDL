"""Automated persistent Codex worker pool for resumable agent workflows."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .app_server import CodexAppServerClient
from .task_client import AgentTaskStore
from .workflow import AgentWorkflow


def run_persistent_pool(
    workflow: AgentWorkflow,
    *,
    worker_count: int,
    tasks_per_worker: int,
    codex_executable: str = "codex",
    timeout_seconds: float = 300.0,
    client_factory: Callable[..., CodexAppServerClient] = CodexAppServerClient,
) -> dict[str, Any]:
    """Drive a workflow to completion with app-server processes reused across waves."""

    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    clients: dict[int, CodexAppServerClient] = {}
    started_at = time.perf_counter()
    completed_task_count = 0
    wave_count = 0
    try:
        while True:
            dispatch = workflow.dispatch(
                worker_count=worker_count,
                tasks_per_worker=tasks_per_worker,
            )
            if dispatch.get("status") == "complete":
                return {
                    **dispatch,
                    "worker_mode": "persistent_app_server",
                    "worker_count": len(clients),
                    "wave_count": wave_count,
                    "completed_task_count": completed_task_count,
                    "elapsed_seconds": time.perf_counter() - started_at,
                }
            if dispatch.get("status") != "tasks_assigned":
                raise RuntimeError(f"Unexpected workflow dispatch status: {dispatch}")
            assignments = list(dispatch.get("assignments") or [])
            wave_count += 1
            for assignment in assignments:
                worker_index = int(assignment["worker_index"])
                if worker_index not in clients:
                    clients[worker_index] = client_factory(
                        codex_executable=codex_executable,
                        cwd=Path.cwd(),
                        timeout_seconds=timeout_seconds,
                    )
            with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
                futures = {
                    executor.submit(
                        _process_assignment,
                        clients[int(assignment["worker_index"])],
                        Path(assignment["manifest_file"]),
                        workflow.store,
                    ): assignment
                    for assignment in assignments
                }
                for future in as_completed(futures):
                    completed_task_count += future.result()
    finally:
        for client in clients.values():
            client.close()


def _process_assignment(
    client: CodexAppServerClient,
    manifest_file: Path,
    store: AgentTaskStore,
) -> int:
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks") or []
    completed = 0
    for task in tasks:
        request = json.loads(Path(task["request_file"]).read_text(encoding="utf-8"))
        response = client.complete(request)
        store.submit(response, task_id=str(task["task_id"]))
        completed += 1
    return completed


__all__ = ["run_persistent_pool"]
