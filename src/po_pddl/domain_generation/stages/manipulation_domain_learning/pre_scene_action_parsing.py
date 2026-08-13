from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .learner import load_raw_trajectory_steps
from .models import ActionTaxonomyRecord, EpisodeObjectInventory
from .object_name_normalization import coarsen_object_identifiers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreSceneActionParsingResult:
    taxonomy_records: list[ActionTaxonomyRecord]
    episode_object_inventories: list[EpisodeObjectInventory]
    action_templates: list[dict[str, Any]]
    action_name_map: dict[str, Any]
    action_text_normalization_records: list[dict[str, Any]]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "taxonomy_record_count": len(self.taxonomy_records),
            "episode_count": len(self.episode_object_inventories),
            "action_template_count": len(self.action_templates),
            "normalized_action_count": len(self.action_text_normalization_records),
        }


def _build_episode_object_inventories_from_taxonomy(
    taxonomy_records: list[ActionTaxonomyRecord],
) -> list[EpisodeObjectInventory]:
    objects_by_episode: dict[str, list[str]] = {}
    seen_by_episode: dict[str, set[str]] = {}
    for record in taxonomy_records:
        for argument in coarsen_object_identifiers(list(record.action_arguments)):
            seen = seen_by_episode.setdefault(record.episode_name, set())
            if argument in seen:
                continue
            seen.add(argument)
            objects_by_episode.setdefault(record.episode_name, []).append(argument)
    return [
        EpisodeObjectInventory(
            episode_name=episode_name,
            object_names=sorted(object_names),
        )
        for episode_name, object_names in sorted(objects_by_episode.items())
    ]


@dataclass
class PreSceneActionParsingRunner:
    action_taxonomy_module: Any
    action_text_preprocessing_module: Any | None = None
    action_template_builder: (
        Callable[[list[ActionTaxonomyRecord]], tuple[list[dict[str, Any]], dict[str, Any]]] | None
    ) = None

    def run(self, input_dir: str | Path) -> PreSceneActionParsingResult:
        logger.info("Pre-scene action parsing: loading trajectory steps from %s", input_dir)
        steps = load_raw_trajectory_steps(input_dir)
        normalization_records: list[dict[str, Any]] = []
        if self.action_text_preprocessing_module is not None:
            preprocessing_result = self.action_text_preprocessing_module.preprocess_steps(steps)
            steps = preprocessing_result.normalized_steps
            normalization_records = [
                item.to_dict() for item in getattr(preprocessing_result, "normalization_records", [])
            ]
        taxonomy_records = self.action_taxonomy_module.classify_actions(steps)
        episode_object_inventories = _build_episode_object_inventories_from_taxonomy(taxonomy_records)
        action_templates: list[dict[str, Any]] = []
        action_name_map: dict[str, Any] = {}
        if self.action_template_builder is not None:
            action_templates, action_name_map = self.action_template_builder(taxonomy_records)
        return PreSceneActionParsingResult(
            taxonomy_records=taxonomy_records,
            episode_object_inventories=episode_object_inventories,
            action_templates=action_templates,
            action_name_map=action_name_map,
            action_text_normalization_records=normalization_records,
        )

    def write_outputs(self, result: PreSceneActionParsingResult, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "action_taxonomy.jsonl").open("w", encoding="utf-8") as handle:
            for record in result.taxonomy_records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        with (output_path / "episode_object_inventory.jsonl").open("w", encoding="utf-8") as handle:
            for record in result.episode_object_inventories:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        if result.action_templates:
            (output_path / "action_templates.json").write_text(
                json.dumps(result.action_templates, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        if result.action_name_map:
            (output_path / "action_name_map.json").write_text(
                json.dumps(result.action_name_map, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        if result.action_text_normalization_records:
            with (output_path / "action_text_normalization.jsonl").open("w", encoding="utf-8") as handle:
                for row in result.action_text_normalization_records:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (output_path / "pre_scene_action_parsing_summary.json").write_text(
            json.dumps(result.to_summary_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


__all__ = [
    "PreSceneActionParsingResult",
    "PreSceneActionParsingRunner",
]
