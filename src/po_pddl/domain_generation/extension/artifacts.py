"""Resolve reusable artifacts from a canonical final bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BundleArtifacts:
    root: Path
    manipulation_dir: Path
    final_domain_file: Path
    final_manipulation_domain_file: Path
    action_schemas_file: Path
    passive_observation_dir: Path | None
    init_observation_dir: Path | None
    active_observation_dir: Path | None
    passive_observation_summary_file: Path | None
    init_observation_summary_file: Path | None
    active_observation_summary_file: Path | None
    problem_grounding_dir: Path | None


def _load_manifest(root: Path) -> dict[str, str]:
    path = root / "bundle_manifest.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Bundle manifest must contain a JSON object: {path}")
    return {str(key): str(value) for key, value in payload.items() if str(value).strip()}


def _first_existing(*paths: Path | None) -> Path | None:
    return next((path for path in paths if path is not None and path.exists()), None)


def _manifest_path(manifest: dict[str, str], key: str) -> Path | None:
    value = manifest.get(key)
    return Path(value).expanduser() if value else None


def _required(path: Path | None, *, description: str, root: Path) -> Path:
    if path is None:
        raise FileNotFoundError(f"Could not locate {description} in reusable bundle {root}.")
    return path


def resolve_bundle_artifacts(bundle_dir: str | Path) -> BundleArtifacts:
    """Resolve bundle files while preferring self-contained bundle paths."""

    root = Path(bundle_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Reusable bundle directory does not exist: {root}")
    manifest = _load_manifest(root)
    manipulation_dir = _first_existing(root / "manipulation_domain", root)
    assert manipulation_dir is not None

    final_domain_file = _required(
        _first_existing(root / "final_merged_domain.pddl", _manifest_path(manifest, "final_domain_file")),
        description="final merged domain",
        root=root,
    )
    final_manipulation_domain_file = _required(
        _first_existing(
            manipulation_dir / "final_manipulation_domain.pddl",
            _manifest_path(manifest, "final_manipulation_domain_file"),
        ),
        description="final manipulation domain",
        root=root,
    )
    action_schemas_file = _required(
        _first_existing(
            manipulation_dir / "precondition_action_schemas.json",
            _manifest_path(manifest, "precondition_action_schemas_json"),
            manipulation_dir / "action_schemas.json",
            _manifest_path(manifest, "action_schemas_json"),
        ),
        description="action schemas",
        root=root,
    )

    passive_module = _first_existing(
        root / "passive_observation" / "passive_observation_module.pddl",
        _manifest_path(manifest, "passive_observation_module_pddl"),
    )
    init_module = _first_existing(
        root / "init_observation" / "init_observation_module.pddl",
        _manifest_path(manifest, "init_observation_module_pddl"),
    )
    active_module = _first_existing(
        root / "active_observation" / "active_observation_module.pddl",
        _manifest_path(manifest, "active_observation_module_pddl"),
    )
    passive_summary = _first_existing(
        root / "passive_observation" / "passive_observation_learning_summary.json",
        _manifest_path(manifest, "passive_observation_learning_summary_json"),
    )
    init_summary = _first_existing(
        root / "init_observation" / "init_observation_learning_summary.json",
        _manifest_path(manifest, "init_observation_learning_summary_json"),
    )
    active_summary = _first_existing(
        root / "active_observation" / "active_observation_learning_summary.json",
        _manifest_path(manifest, "active_observation_learning_summary_json"),
    )
    problem_grounding_dir = _first_existing(
        root / "problem_grounding_all",
        _manifest_path(manifest, "problem_grounding_all_dir"),
    )

    return BundleArtifacts(
        root=root,
        manipulation_dir=manipulation_dir,
        final_domain_file=final_domain_file,
        final_manipulation_domain_file=final_manipulation_domain_file,
        action_schemas_file=action_schemas_file,
        passive_observation_dir=passive_module.parent if passive_module else None,
        init_observation_dir=init_module.parent if init_module else None,
        active_observation_dir=active_module.parent if active_module else None,
        passive_observation_summary_file=passive_summary,
        init_observation_summary_file=init_summary,
        active_observation_summary_file=active_summary,
        problem_grounding_dir=problem_grounding_dir,
    )


__all__ = ["BundleArtifacts", "resolve_bundle_artifacts"]
