from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DomainLearningArtifactRows:
    action_schemas: list[dict[str, Any]]
    taxonomy_records: list[dict[str, Any]]
    manipulation_records: list[dict[str, Any]]


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object line in {path}, got {type(payload).__name__}")
        rows.append(payload)
    return rows


def load_optional_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    if not resolved.exists():
        return []
    return load_jsonl(resolved)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def discover_episode_files(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    annotated_episode_files = sorted(root.rglob("annotated_episode.json"))
    if annotated_episode_files:
        return annotated_episode_files
    episode_files = sorted([path for pattern in ("episode.json", "annotations.json") for path in root.rglob(pattern)])
    if not episode_files:
        raise ValueError(f"No episode.json or annotations.json files found under {root}")
    return episode_files


def load_episode_payload(path: str | Path) -> dict[str, Any]:
    payload = load_json_object(path)
    steps = payload.get("steps", [])
    if steps is not None and not isinstance(steps, list):
        raise ValueError(f"{path} has non-list 'steps'")
    return payload


def load_episode_name(path: str | Path) -> str:
    episode_path = Path(path)
    payload = load_episode_payload(episode_path)
    return str(payload.get("episode_name", episode_path.parent.name))


def load_domain_learning_artifact_rows(path: str | Path) -> DomainLearningArtifactRows:
    root = Path(path)
    action_schemas = load_json(root / "action_schemas.json")
    if not isinstance(action_schemas, list):
        raise ValueError(f"Expected JSON array at {root / 'action_schemas.json'}")
    action_schema_rows = [item for item in action_schemas if isinstance(item, dict)]
    if len(action_schema_rows) != len(action_schemas):
        raise ValueError(f"Expected all action schema rows to be JSON objects in {root / 'action_schemas.json'}")
    return DomainLearningArtifactRows(
        action_schemas=action_schema_rows,
        taxonomy_records=load_jsonl(root / "action_taxonomy.jsonl"),
        manipulation_records=load_jsonl(root / "manipulation_records.jsonl"),
    )
