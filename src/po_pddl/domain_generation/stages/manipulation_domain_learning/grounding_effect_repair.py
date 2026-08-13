from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from po_pddl.domain_generation.infrastructure.artifact_io import write_jsonl

from .effect_merge import rewrite_records_using_action_schemas
from .grounding_update import load_action_schemas, load_manipulation_records
from .models import ActionEffectStatistic, ActionSchema, ManipulationEffectRecord
from .renderer import collect_action_effect_statistics


@dataclass(frozen=True)
class GroundingEffectRepairResult:
    action_schemas: list[ActionSchema]
    original_records: list[ManipulationEffectRecord]
    repaired_records: list[ManipulationEffectRecord]
    action_statistics: dict[str, list[ActionEffectStatistic]]
    changed_step_count: int

    @property
    def changed(self) -> bool:
        return self.changed_step_count > 0

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "changed": self.changed,
            "changed_step_count": self.changed_step_count,
            "record_count": len(self.repaired_records),
        }


def repair_grounding_effects_from_artifacts(
    artifact_dir: str | Path,
) -> GroundingEffectRepairResult:
    artifact_path = Path(artifact_dir)
    action_schemas = load_action_schemas(artifact_path / "action_schemas.json")
    original_records = load_manipulation_records(artifact_path / "manipulation_records.jsonl")
    repaired_records = rewrite_records_using_action_schemas(original_records, action_schemas)
    action_statistics = collect_action_effect_statistics(repaired_records)
    changed_step_count = sum(
        1 for original, repaired in zip(original_records, repaired_records) if original.to_dict() != repaired.to_dict()
    )
    return GroundingEffectRepairResult(
        action_schemas=action_schemas,
        original_records=original_records,
        repaired_records=repaired_records,
        action_statistics=action_statistics,
        changed_step_count=changed_step_count,
    )


def write_grounding_effect_repair_artifacts(
    *,
    result: GroundingEffectRepairResult,
    base_artifact_dir: str | Path,
    output_dir: str | Path,
) -> None:
    base_path = Path(base_artifact_dir)
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.copytree(base_path, output_path)

    write_jsonl(
        output_path / "manipulation_records.jsonl",
        [record.to_dict() for record in result.repaired_records],
    )
    (output_path / "manipulation_effect_statistics.json").write_text(
        json.dumps(
            {
                action_name: [item.to_dict() for item in stats]
                for action_name, stats in result.action_statistics.items()
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_path / "grounding_effect_repair_summary.json").write_text(
        json.dumps(result.to_summary_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
