"""Balanced task manifests for persistent Codex worker pools."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_POOL_SIZE = 10
DEFAULT_TASKS_PER_WORKER = 4


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _walk_media(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if "agent_media_path" in value:
            yield value
        for item in value.values():
            yield from _walk_media(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_media(item)


def _estimated_cost(task: dict[str, Any]) -> int:
    """Estimate relative model work without interpreting task semantics."""

    prompt_file = Path(task["prompt_file"])
    prompt_size = prompt_file.stat().st_size
    request = json.loads(Path(task["request_file"]).read_text(encoding="utf-8"))
    media_cost = 0
    seen: set[str] = set()
    for media in _walk_media(request):
        media_path = str(media.get("agent_media_path") or "")
        if not media_path or media_path in seen:
            continue
        seen.add(media_path)
        media_type = str(media.get("media_type") or "")
        media_cost += 50_000 if media_type.startswith("video/") else 12_000
    if task.get("validation_file"):
        prompt_size += Path(task["validation_file"]).stat().st_size
    return max(1, prompt_size + media_cost)


@dataclass(frozen=True)
class WorkerAssignment:
    worker_index: int
    estimated_load: int
    tasks: tuple[dict[str, Any], ...]


def balance_tasks(
    pending_tasks: list[dict[str, Any]],
    *,
    worker_count: int,
    tasks_per_worker: int,
) -> list[WorkerAssignment]:
    """Create deterministic, bounded LPT assignments to reduce wave tail latency."""

    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    if tasks_per_worker < 1:
        raise ValueError("tasks_per_worker must be at least 1")
    selected = pending_tasks[: worker_count * tasks_per_worker]
    if not selected:
        return []

    actual_workers = min(worker_count, len(selected))
    bins: list[list[dict[str, Any]]] = [[] for _ in range(actual_workers)]
    loads = [0] * actual_workers
    weighted = sorted(
        ((_estimated_cost(task), task) for task in selected),
        key=lambda item: (-item[0], str(item[1]["task_id"])),
    )
    for cost, task in weighted:
        candidates = [index for index, tasks in enumerate(bins) if len(tasks) < tasks_per_worker]
        worker_index = min(candidates, key=lambda index: (loads[index], len(bins[index]), index))
        bins[worker_index].append({**task, "estimated_cost": cost})
        loads[worker_index] += cost

    return [
        WorkerAssignment(index, loads[index], tuple(bins[index]))
        for index in range(actual_workers)
        if bins[index]
    ]


def write_assignment_manifests(
    run_dir: Path,
    pending_tasks: list[dict[str, Any]],
    *,
    worker_count: int = DEFAULT_POOL_SIZE,
    tasks_per_worker: int = DEFAULT_TASKS_PER_WORKER,
) -> dict[str, Any]:
    assignments = balance_tasks(
        pending_tasks,
        worker_count=worker_count,
        tasks_per_worker=tasks_per_worker,
    )
    fingerprint_rows = []
    for assignment in assignments:
        for task in assignment.tasks:
            validation_file = task.get("validation_file")
            validation_digest = ""
            if validation_file:
                validation_digest = hashlib.sha256(Path(validation_file).read_bytes()).hexdigest()
            fingerprint_rows.append(f"{task['task_id']}:{validation_digest}")
    fingerprint = "\n".join(fingerprint_rows)
    batch_id = f"batch_{hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()[:16]}"
    batch_dir = run_dir / "worker_assignments" / batch_id
    manifests = []
    for assignment in assignments:
        manifest_file = batch_dir / f"worker_{assignment.worker_index:02d}.json"
        payload = {
            "schema_version": 1,
            "batch_id": batch_id,
            "run_dir": str(run_dir),
            "worker_index": assignment.worker_index,
            "task_count": len(assignment.tasks),
            "estimated_load": assignment.estimated_load,
            "tasks": list(assignment.tasks),
        }
        _atomic_json(manifest_file, payload)
        manifests.append(
            {
                "worker_index": assignment.worker_index,
                "task_count": len(assignment.tasks),
                "estimated_load": assignment.estimated_load,
                "manifest_file": str(manifest_file),
            }
        )
    assigned_count = sum(item["task_count"] for item in manifests)
    return {
        "status": "tasks_assigned",
        "run_dir": str(run_dir),
        "batch_id": batch_id,
        "pending_count": len(pending_tasks),
        "assigned_count": assigned_count,
        "remaining_count": len(pending_tasks) - assigned_count,
        "worker_count": len(manifests),
        "tasks_per_worker": tasks_per_worker,
        "assignments": manifests,
    }


__all__ = [
    "DEFAULT_POOL_SIZE",
    "DEFAULT_TASKS_PER_WORKER",
    "WorkerAssignment",
    "balance_tasks",
    "write_assignment_manifests",
]
