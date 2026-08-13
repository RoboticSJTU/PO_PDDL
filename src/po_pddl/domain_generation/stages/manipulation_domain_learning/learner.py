from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from po_pddl.domain_generation.infrastructure.artifact_io import load_json, load_json_object, load_optional_jsonl

from . import modules as manipulation_domain_learning_modules
from .effect_completeness_review import EffectCompletenessRepairPlan, VLMEffectCompletenessReviewModule
from .effect_merge import LLMManipulationEffectMergeModule
from .models import (
    ActionEffectStatistic,
    ActionSchema,
    ActionTaxonomyRecord,
    EpisodeObjectInventory,
    ManipulationEffectRecord,
    ObjectTypeDefinition,
    PredicateInventoryResult,
    PredicateSchema,
)
from .modules import (
    ActionSchemaConsolidationModule,
    ActionTaxonomyModule,
    ManipulationEffectLearningModule,
    validate_manipulation_records_against_predicate_inventory,
)
from .object_name_normalization import coarsen_object_identifier, coarsen_object_identifiers
from .predicate_comments import LLMPredicateCommentModule
from .predicate_inventory import LLMPredicateInventoryModule
from .renderer import (
    attach_action_effects_to_schemas,
    classify_records_by_effect_statistics,
    collect_action_effect_statistics,
    render_action_schema_fragment,
    render_manipulation_domain_fragment,
)
from .structured_action_templates import compile_template_regex, induced_template_from_dict, normalize_argument_value


@dataclass(frozen=True)
class ManipulationDomainLearningResult:
    episode_object_inventories: list[EpisodeObjectInventory]
    taxonomy_records: list[ActionTaxonomyRecord]
    action_schemas: list[ActionSchema]
    manipulation_records: list[ManipulationEffectRecord]
    action_statistics: dict[str, list[ActionEffectStatistic]]
    rendered_action_schema_pddl: str
    rendered_manipulation_pddl: str
    observation_action_learning_status: dict[str, Any]
    action_templates: list[dict[str, Any]]
    action_name_map: dict[str, Any]
    predicate_inventory: list[PredicateSchema]
    predicate_comments: dict[str, str]
    object_types: list[ObjectTypeDefinition] | None = None
    object_type_map: dict[str, str] | None = None
    action_text_normalization_records: list[dict[str, Any]] | None = None

    def action_statistics_dict(self) -> dict[str, Any]:
        return {
            action_name: [item.to_dict() for item in stats] for action_name, stats in self.action_statistics.items()
        }


def load_raw_trajectory_steps(input_dir: str | Path) -> list[Any]:
    return manipulation_domain_learning_modules.load_raw_trajectory_steps(input_dir)


def load_pre_scene_action_parsing_artifacts(artifact_dir: str | Path) -> dict[str, Any]:
    artifact_path = Path(artifact_dir)
    taxonomy_rows = load_optional_jsonl(artifact_path / "action_taxonomy.jsonl")
    inventory_rows = load_optional_jsonl(artifact_path / "episode_object_inventory.jsonl")
    action_templates: list[dict[str, Any]] = []
    action_templates_file = artifact_path / "action_templates.json"
    if action_templates_file.exists():
        loaded_templates = load_json(action_templates_file)
        if isinstance(loaded_templates, list):
            action_templates = list(loaded_templates)
    action_name_map: dict[str, Any] = {}
    action_name_map_file = artifact_path / "action_name_map.json"
    if action_name_map_file.exists():
        loaded_action_map = load_json_object(action_name_map_file)
        if isinstance(loaded_action_map, dict):
            action_name_map = dict(loaded_action_map)
    normalization_records = [
        row for row in load_optional_jsonl(artifact_path / "action_text_normalization.jsonl") if isinstance(row, dict)
    ]
    return {
        "taxonomy_records": [ActionTaxonomyRecord(**row) for row in taxonomy_rows if isinstance(row, dict)],
        "episode_object_inventories": [
            EpisodeObjectInventory(**row) for row in inventory_rows if isinstance(row, dict)
        ],
        "action_templates": action_templates,
        "action_name_map": action_name_map,
        "action_text_normalization_records": normalization_records,
    }


_WHITESPACE_RE = re.compile(r"\s+")


def _default_placeholder_name(index: int) -> str:
    return "object" if index == 0 else f"object_{index + 1}"


def _surface_from_argument(argument: str) -> str:
    return argument.replace("_", " ")


def _templated_action_text(
    raw_action_text: str | None,
    action_arguments: list[str],
) -> str | None:
    text = str(raw_action_text).strip() if raw_action_text is not None else None
    if text == "":
        text = None
    if text is None:
        return None
    templated = text
    for index, argument in sorted(
        enumerate(action_arguments),
        key=lambda item: len(_surface_from_argument(item[1])),
        reverse=True,
    ):
        surface = _surface_from_argument(argument).strip()
        if not surface:
            continue
        placeholder = "{" + _default_placeholder_name(index) + "}"
        pattern = re.compile(rf"(?i)\b{re.escape(surface)}\b")
        templated, _count = pattern.subn(placeholder, templated, count=1)
    templated = _WHITESPACE_RE.sub(" ", templated).strip()
    return templated or None


def _build_generic_action_name_map(
    *,
    taxonomy_records: list[ActionTaxonomyRecord],
    action_schemas: list[ActionSchema],
) -> dict[str, Any]:
    records_by_action: dict[str, list[ActionTaxonomyRecord]] = {}
    for record in taxonomy_records:
        if not record.canonical_action_name:
            continue
        records_by_action.setdefault(record.canonical_action_name, []).append(record)

    schema_by_action = {schema.canonical_action_name: schema for schema in action_schemas}
    actions_by_name: dict[str, dict[str, Any]] = {}
    template_id_to_action: dict[str, str] = {}
    template_text_to_action: dict[str, str] = {}

    for action_name, records in sorted(records_by_action.items()):
        schema = schema_by_action.get(action_name)
        action_category = schema.action_category if schema is not None else records[0].action_category
        role_count = max((len(record.action_arguments) for record in records), default=0)
        parameter_roles = list(schema.parameter_roles) if schema is not None else ["object"] * role_count
        placeholder_candidates = [
            tuple(record.parameter_placeholders) for record in records if record.parameter_placeholders
        ]
        if placeholder_candidates:
            chosen_placeholder_tuple = sorted(
                {
                    candidate: placeholder_candidates.count(candidate) for candidate in set(placeholder_candidates)
                }.items(),
                key=lambda pair: (-pair[1], pair[0]),
            )[0][0]
            parameter_placeholders = list(chosen_placeholder_tuple)
        else:
            parameter_placeholders = [_default_placeholder_name(index) for index in range(len(parameter_roles))]
        template_candidates = [
            record.template_text or _templated_action_text(record.raw_action_text, record.action_arguments)
            for record in records
        ]
        normalized_candidates = [item for item in template_candidates if item]
        chosen_template = (
            sorted(
                {template: normalized_candidates.count(template) for template in set(normalized_candidates)}.items(),
                key=lambda pair: (-pair[1], pair[0]),
            )[0][0]
            if normalized_candidates
            else action_name
        )
        success_bucket = None
        failure_bucket = None
        if schema is not None:
            for branch in schema.effect_branches:
                if branch.success and success_bucket is None:
                    success_bucket = branch.effect_bucket
                if not branch.success and failure_bucket is None:
                    failure_bucket = branch.effect_bucket
        actions_by_name[action_name] = {
            "action_name": action_name,
            "action_category": action_category,
            "template_id": action_name,
            "template_text": chosen_template,
            "parameter_roles": parameter_roles,
            "parameter_placeholders": parameter_placeholders,
            "effect_buckets": {
                "success": success_bucket,
                "failure": failure_bucket,
            },
            "source_action_name": None,
            "ground_truth_positive_literal": None,
        }
        template_id_to_action[action_name] = action_name
        template_text_to_action[chosen_template] = action_name

    return {
        "schema_version": 2,
        "actions_by_name": actions_by_name,
        "lookup": {
            "template_id_to_action": template_id_to_action,
            "template_text_to_action": template_text_to_action,
        },
    }


def _apply_predicate_type_hierarchy(
    object_types: list[ObjectTypeDefinition],
    result: PredicateInventoryResult,
) -> list[ObjectTypeDefinition]:
    if not object_types:
        return []
    special_supertypes_by_type: dict[str, list[str]] = {}
    for type_name in result.containable_item_member_types:
        special_supertypes_by_type.setdefault(type_name, []).append("containable_item")
    for type_name in result.movable_item_member_types:
        special_supertypes_by_type.setdefault(type_name, []).append("movable_item")
    for type_name in result.fixed_item_member_types:
        special_supertypes_by_type.setdefault(type_name, []).append("fixed_item")

    def choose_parent_type(type_name: str) -> str | None:
        memberships = set(special_supertypes_by_type.get(type_name, []))
        if "movable_item" in memberships:
            return "movable_item"
        if "fixed_item" in memberships:
            return "fixed_item"
        if "containable_item" in memberships:
            return "containable_item"
        return None

    return [
        ObjectTypeDefinition(
            type_name=item.type_name,
            member_object_names=list(item.member_object_names),
            parent_type=choose_parent_type(item.type_name),
            special_supertypes=sorted(set(special_supertypes_by_type.get(item.type_name, []))),
        )
        for item in object_types
    ]


def _effect_learning_predicate_inventory(
    predicate_inventory: list[PredicateSchema],
) -> list[PredicateSchema]:
    return list(predicate_inventory)


def _effect_guidance_signature_key(
    canonical_action_name: str,
    effect_bucket: str,
    variant_rank: int | None,
) -> tuple[str, str, int | None]:
    return (str(canonical_action_name), str(effect_bucket), int(variant_rank) if variant_rank is not None else None)


def _bucket_variant_rank(effect_bucket: str | None) -> int | None:
    bucket_text = str(effect_bucket or "").strip()
    if not bucket_text:
        return None
    parts = bucket_text.rsplit("_bucket_", 1)
    if len(parts) != 2:
        return None
    suffix = parts[1]
    return int(suffix) if suffix.isdigit() else None


def _merge_effect_completeness_guidance(
    current_guidance: dict[tuple[str, str, int | None], dict[str, object]],
    plan: EffectCompletenessRepairPlan,
) -> tuple[dict[tuple[str, str, int | None], dict[str, object]], bool]:
    if not plan.should_apply_repair or plan.canonical_action_name is None or plan.effect_bucket is None:
        return dict(current_guidance), False
    key = _effect_guidance_signature_key(
        plan.canonical_action_name,
        plan.effect_bucket,
        plan.variant_rank,
    )
    next_guidance = {item_key: dict(item_value) for item_key, item_value in current_guidance.items()}
    existing = dict(next_guidance.get(key, {}))
    required_add = {str(item.grounded_fact) for item in plan.predicate_additions if item.value_after_action}
    required_del = {str(item.grounded_fact) for item in plan.predicate_additions if not item.value_after_action}
    merged_add = sorted(set(existing.get("required_effect_facts_add", [])) | required_add)
    merged_del = sorted(set(existing.get("required_effect_facts_del", [])) | required_del)
    existing["required_effect_facts_add"] = merged_add
    existing["required_effect_facts_del"] = merged_del
    existing["repair_summary"] = plan.repair_summary
    existing["predicate_additions"] = [item.to_dict() for item in plan.predicate_additions]
    existing["canonical_action_name"] = plan.canonical_action_name
    existing["effect_bucket"] = plan.effect_bucket
    existing["variant_rank"] = plan.variant_rank
    changed = existing != next_guidance.get(key, {})
    next_guidance[key] = existing
    return next_guidance, changed


def _expand_effect_guidance_to_episode_steps(
    records: list[ManipulationEffectRecord],
    guidance_by_signature: dict[tuple[str, str, int | None], dict[str, object]],
) -> dict[tuple[str, int], dict[str, object]]:
    if not records or not guidance_by_signature:
        return {}
    expanded: dict[tuple[str, int], dict[str, object]] = {}
    for record in records:
        record_variant_rank = _bucket_variant_rank(record.effect_bucket)
        candidate_keys = [
            _effect_guidance_signature_key(
                record.canonical_action_name,
                record.effect_bucket,
                record_variant_rank,
            )
        ]
        if record_variant_rank is None:
            candidate_keys.append(
                _effect_guidance_signature_key(
                    record.canonical_action_name,
                    record.effect_bucket,
                    None,
                )
            )
        for key in candidate_keys:
            guidance = guidance_by_signature.get(key)
            if not guidance:
                continue
            expanded[(record.episode_name, record.step_index)] = dict(guidance)
            break
    return expanded


def _action_outcomes_for_target_signatures(
    records: list[ManipulationEffectRecord],
    target_signatures: set[tuple[str, str, int | None]],
) -> set[tuple[str, bool]]:
    outputs: set[tuple[str, bool]] = set()
    if not target_signatures:
        return outputs
    for record in records:
        record_variant_rank = _bucket_variant_rank(record.effect_bucket)
        signature = _effect_guidance_signature_key(
            record.canonical_action_name,
            record.effect_bucket,
            record_variant_rank,
        )
        if signature in target_signatures:
            outputs.add((record.canonical_action_name, bool(record.success)))
    return outputs


def _merge_predicate_additions_into_inventory(
    predicate_inventory: list[PredicateSchema],
    plan: EffectCompletenessRepairPlan,
) -> tuple[list[PredicateSchema], bool]:
    if not plan.should_apply_repair or not plan.predicate_additions:
        return list(predicate_inventory), False
    existing_by_signature = {(item.predicate_name, tuple(item.parameter_types)): item for item in predicate_inventory}
    existing_names = {item.predicate_name: tuple(item.parameter_types) for item in predicate_inventory}
    merged = list(predicate_inventory)
    changed = False
    for item in plan.predicate_additions:
        signature = (item.predicate_name, tuple(item.parameter_types))
        if signature in existing_by_signature:
            continue
        existing_signature = existing_names.get(item.predicate_name)
        if existing_signature is not None and existing_signature != tuple(item.parameter_types):
            raise ValueError(
                f"Completeness review proposed conflicting predicate signature for {item.predicate_name!r}: "
                f"existing={list(existing_signature)} proposed={item.parameter_types}"
            )
        merged.append(
            PredicateSchema(
                predicate_name=item.predicate_name,
                parameter_types=list(item.parameter_types),
                comment=item.comment,
            )
        )
        existing_by_signature[signature] = merged[-1]
        existing_names[item.predicate_name] = tuple(item.parameter_types)
        changed = True
    return merged, changed


def _match_action_text_with_action_map(
    *,
    action_text: str,
    action_name: str,
    action_name_map: dict[str, Any],
) -> list[str]:
    actions_by_name = action_name_map.get("actions_by_name", {}) if isinstance(action_name_map, dict) else {}
    action_info = actions_by_name.get(action_name, {})
    template_text = str(action_info.get("template_text") or "").strip()
    parameter_placeholders = [str(item) for item in action_info.get("parameter_placeholders", []) if str(item).strip()]
    if not template_text or not parameter_placeholders:
        return []
    match = compile_template_regex(template_text).match(action_text.strip())
    if match is None:
        return []
    return [
        normalize_argument_value(match.group(placeholder))
        for placeholder in parameter_placeholders
        if match.groupdict().get(placeholder)
    ]


def _build_episode_object_inventories_from_action_map(
    *,
    steps: list[Any],
    taxonomy_records: list[ActionTaxonomyRecord],
    action_name_map: dict[str, Any],
) -> list[EpisodeObjectInventory]:
    taxonomy_by_key = {(record.episode_name, record.step_index): record for record in taxonomy_records}
    objects_by_episode: dict[str, set[str]] = {}
    for step in steps:
        if not getattr(step, "action_text", None):
            continue
        episode_name = str(step.episode_name)
        taxonomy = taxonomy_by_key.get((episode_name, int(step.step_index)))
        if taxonomy is None:
            continue
        parsed_arguments = _match_action_text_with_action_map(
            action_text=str(step.action_text),
            action_name=taxonomy.canonical_action_name,
            action_name_map=action_name_map,
        )
        if not parsed_arguments:
            parsed_arguments = list(taxonomy.action_arguments)
        if not parsed_arguments:
            continue
        objects_by_episode.setdefault(episode_name, set()).update(
            coarsen_object_identifier(argument) for argument in parsed_arguments
        )
    return [
        EpisodeObjectInventory(
            episode_name=episode_name,
            object_names=sorted(object_names),
        )
        for episode_name, object_names in sorted(objects_by_episode.items())
    ]


def _normalize_taxonomy_records_with_action_map(
    *,
    taxonomy_records: list[ActionTaxonomyRecord],
    action_name_map: dict[str, Any],
    episode_object_inventories: list[EpisodeObjectInventory],
) -> list[ActionTaxonomyRecord]:
    allowed_by_episode = {item.episode_name: set(item.object_names) for item in episode_object_inventories}
    normalized_records: list[ActionTaxonomyRecord] = []
    for record in taxonomy_records:
        parsed_arguments = _match_action_text_with_action_map(
            action_text=record.raw_action_text,
            action_name=record.canonical_action_name,
            action_name_map=action_name_map,
        )
        action_arguments = coarsen_object_identifiers(parsed_arguments or list(record.action_arguments))
        object_mentions = list(action_arguments or record.object_mentions)
        allowed_names = allowed_by_episode.get(record.episode_name)
        if allowed_names:
            action_arguments = [argument for argument in action_arguments if argument in allowed_names]
            object_mentions = [argument for argument in object_mentions if argument in allowed_names]
        action_info = (
            action_name_map.get("actions_by_name", {}).get(record.canonical_action_name, {})
            if isinstance(action_name_map, dict)
            else {}
        )
        normalized_records.append(
            ActionTaxonomyRecord(
                episode_name=record.episode_name,
                step_index=record.step_index,
                raw_action_text=record.raw_action_text,
                proposed_action_name=record.proposed_action_name,
                canonical_action_name=record.canonical_action_name,
                action_category=record.action_category,
                action_arguments=action_arguments,
                object_mentions=object_mentions,
                observation_text=record.observation_text,
                extra_info=record.extra_info,
                template_text=str(action_info.get("template_text") or record.template_text or "").strip() or None,
                parameter_placeholders=[
                    str(item)
                    for item in action_info.get("parameter_placeholders", record.parameter_placeholders)
                    if str(item).strip()
                ],
            )
        )
    return normalized_records


class ManipulationDomainLearningLearner:
    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        episode_object_inventory_builder: Callable[[list[Any]], list[EpisodeObjectInventory]] | None = None,
        action_taxonomy_module: ActionTaxonomyModule,
        action_schema_consolidation_module: ActionSchemaConsolidationModule,
        manipulation_effect_module: ManipulationEffectLearningModule,
        action_text_preprocessing_module: Any | None = None,
        object_typing_module: Any | None = None,
        effect_merge_module: LLMManipulationEffectMergeModule | None = None,
        effect_variant_review_module: Any | None = None,
        effect_completeness_review_module: VLMEffectCompletenessReviewModule | None = None,
        action_template_builder: Callable[[list[ActionTaxonomyRecord]], tuple[list[dict[str, Any]], dict[str, Any]]]
        | None = None,
        predicate_inventory_module: LLMPredicateInventoryModule | None = None,
        predicate_comment_module: LLMPredicateCommentModule | None = None,
    ) -> None:
        self._action_taxonomy_module = action_taxonomy_module
        self._episode_object_inventory_builder = episode_object_inventory_builder
        self._action_schema_consolidation_module = action_schema_consolidation_module
        self._manipulation_effect_module = manipulation_effect_module
        self._action_text_preprocessing_module = action_text_preprocessing_module
        self._object_typing_module = object_typing_module
        self._effect_merge_module = effect_merge_module
        self._effect_variant_review_module = effect_variant_review_module
        self._effect_completeness_review_module = effect_completeness_review_module
        self._action_template_builder = action_template_builder
        self._predicate_inventory_module = predicate_inventory_module
        self._predicate_comment_module = predicate_comment_module
        self._logger = logging.getLogger(__name__)
        self._output_dir = Path(output_dir) if output_dir is not None else None

    def _write_intermediate_typing_and_predicate_outputs(
        self,
        *,
        object_types: list[ObjectTypeDefinition],
        object_type_map: dict[str, str],
        predicate_inventory: list[PredicateSchema],
    ) -> None:
        if self._output_dir is None:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if object_types:
            object_types_path = self._output_dir / "object_types.json"
            object_types_path.write_text(
                json.dumps([item.to_dict() for item in object_types], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self._logger.info("Wrote intermediate object types: %s", object_types_path)
        if object_type_map:
            object_type_map_path = self._output_dir / "object_type_map.json"
            object_type_map_path.write_text(
                json.dumps(object_type_map, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self._logger.info("Wrote intermediate object type map: %s", object_type_map_path)
        if predicate_inventory:
            predicate_inventory_path = self._output_dir / "predicate_inventory.json"
            predicate_inventory_path.write_text(
                json.dumps([item.to_dict() for item in predicate_inventory], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self._logger.info("Wrote intermediate predicate inventory: %s", predicate_inventory_path)

    def learn_from_directory(self, input_dir: str | Path) -> ManipulationDomainLearningResult:
        return self._learn_from_directory(input_dir, allowed_action_schemas=None)

    def learn_from_directory_with_preparsed_artifacts(
        self,
        input_dir: str | Path,
        *,
        preparsed_artifact_dir: str | Path,
    ) -> ManipulationDomainLearningResult:
        preparsed = load_pre_scene_action_parsing_artifacts(preparsed_artifact_dir)
        return self._learn_from_directory(
            input_dir,
            allowed_action_schemas=None,
            preparsed_artifacts=preparsed,
        )

    def learn_from_directory_with_fixed_action_schemas(
        self,
        input_dir: str | Path,
        allowed_action_schemas: list[ActionSchema],
    ) -> ManipulationDomainLearningResult:
        return self._learn_from_directory(input_dir, allowed_action_schemas=allowed_action_schemas)

    def _learn_from_directory(
        self,
        input_dir: str | Path,
        *,
        allowed_action_schemas: list[ActionSchema] | None,
        preparsed_artifacts: dict[str, Any] | None = None,
    ) -> ManipulationDomainLearningResult:
        self._logger.info("Stage 0/4: loading full-chain trajectory steps from %s", input_dir)
        steps = load_raw_trajectory_steps(input_dir)
        self._logger.info("Loaded %d raw trajectory steps", len(steps))
        normalization_records: list[dict[str, Any]] = []
        taxonomy_records: list[ActionTaxonomyRecord] = []
        episode_object_inventories: list[EpisodeObjectInventory] = []
        object_types: list[ObjectTypeDefinition] = []
        object_type_map: dict[str, str] = {}
        typed_action_schemas: list[ActionSchema] | None = None
        action_templates: list[dict[str, Any]] = []
        action_name_map: dict[str, Any] = {}

        if preparsed_artifacts is not None:
            taxonomy_records = list(preparsed_artifacts.get("taxonomy_records", []))
            episode_object_inventories = list(preparsed_artifacts.get("episode_object_inventories", []))
            action_templates = list(preparsed_artifacts.get("action_templates", []))
            action_name_map = dict(preparsed_artifacts.get("action_name_map", {}))
            normalization_records = list(preparsed_artifacts.get("action_text_normalization_records", []))
            self._restore_template_registries(action_templates)
            self._logger.info(
                "Reusing pre-scene action parsing artifacts: %d taxonomy records, %d episode inventories, %d action templates.",
                len(taxonomy_records),
                len(episode_object_inventories),
                len(action_templates),
            )
        elif allowed_action_schemas is None and self._action_text_preprocessing_module is not None:
            self._logger.info("Stage 0.5/5: semantic action-text normalization")
            preprocessing_result = self._action_text_preprocessing_module.preprocess_steps(steps)
            steps = preprocessing_result.normalized_steps
            normalization_records = [
                item.to_dict() for item in getattr(preprocessing_result, "normalization_records", [])
            ]
            self._logger.info(
                "Normalized %d action texts into template form",
                len(normalization_records),
            )

        if preparsed_artifacts is None:
            self._logger.info("Stage 1/4: action taxonomy learning")
            if allowed_action_schemas and hasattr(
                self._action_taxonomy_module, "classify_actions_with_allowed_schemas"
            ):
                try:
                    taxonomy_records = self._action_taxonomy_module.classify_actions_with_allowed_schemas(
                        steps,
                        allowed_action_schemas,
                        episode_object_inventories=episode_object_inventories,
                    )
                except TypeError:
                    taxonomy_records = self._action_taxonomy_module.classify_actions_with_allowed_schemas(
                        steps,
                        allowed_action_schemas,
                    )
            elif hasattr(self._action_taxonomy_module, "classify_actions_with_episode_objects"):
                taxonomy_records = self._action_taxonomy_module.classify_actions_with_episode_objects(
                    steps,
                    episode_object_inventories,
                )
            else:
                taxonomy_records = self._action_taxonomy_module.classify_actions(steps)
            self._logger.info("Classified %d action steps", len(taxonomy_records))

        if allowed_action_schemas is None and self._object_typing_module is not None:
            self._logger.info("Stage 1.5/5: object typing and typed action-map synthesis")
            base_action_templates: list[dict[str, Any]] = list(action_templates)
            if not base_action_templates and self._action_template_builder is not None:
                base_action_templates, _unused_action_name_map = self._action_template_builder(taxonomy_records)
            typing_result = self._object_typing_module.build_typed_artifacts(
                taxonomy_records,
                base_action_templates=base_action_templates,
            )
            object_types = list(typing_result.object_types)
            object_type_map = dict(typing_result.object_type_map)
            episode_object_inventories = list(typing_result.episode_object_inventories)
            taxonomy_records = list(typing_result.taxonomy_records)
            action_templates = list(typing_result.action_templates)
            action_name_map = dict(typing_result.action_name_map)
            typed_action_schemas = list(typing_result.action_schemas)
            self._logger.info(
                "Typed %d unique objects into %d types and synthesized %d typed actions",
                len(object_type_map),
                len(object_types),
                len(action_name_map.get("actions_by_name", {})),
            )
            self._write_intermediate_typing_and_predicate_outputs(
                object_types=object_types,
                object_type_map=object_type_map,
                predicate_inventory=[],
            )
        elif self._action_template_builder is not None:
            action_templates, action_name_map = self._action_template_builder(taxonomy_records)
        elif not action_name_map:
            action_name_map = _build_generic_action_name_map(
                taxonomy_records=taxonomy_records,
                action_schemas=allowed_action_schemas or [],
            )

        if (
            self._object_typing_module is None
            and self._episode_object_inventory_builder is not None
            and action_name_map
        ):
            self._logger.info("Stage 1.5/4: episode object inventory parsing from action templates")
            episode_object_inventories = _build_episode_object_inventories_from_action_map(
                steps=steps,
                taxonomy_records=taxonomy_records,
                action_name_map=action_name_map,
            )
            taxonomy_records = _normalize_taxonomy_records_with_action_map(
                taxonomy_records=taxonomy_records,
                action_name_map=action_name_map,
                episode_object_inventories=episode_object_inventories,
            )
            self._logger.info(
                "Parsed object inventories for %d episodes",
                len(episode_object_inventories),
            )

        predicate_inventory: list[PredicateSchema] = []
        if self._predicate_inventory_module is not None:
            self._logger.info("Stage 2/5: global predicate inventory induction")
            available_types = sorted(item.type_name for item in object_types if str(item.type_name).strip())
            try:
                predicate_inventory_result = self._predicate_inventory_module.generate_predicate_inventory(
                    steps=steps,
                    taxonomy_records=taxonomy_records,
                    action_templates=action_templates,
                    available_types=available_types,
                )
            except TypeError:
                predicate_inventory_result = self._predicate_inventory_module.generate_predicate_inventory(
                    steps=steps,
                    taxonomy_records=taxonomy_records,
                    action_templates=action_templates,
                )
            if isinstance(predicate_inventory_result, list):
                predicate_inventory = list(predicate_inventory_result)
            else:
                predicate_inventory = list(predicate_inventory_result.predicate_inventory)
                object_types = _apply_predicate_type_hierarchy(object_types, predicate_inventory_result)
                if action_name_map:
                    typing_payload = action_name_map.setdefault("typing", {})
                    if isinstance(typing_payload, dict):
                        typing_payload["type_to_parent_type"] = {
                            item.type_name: item.parent_type for item in object_types if item.parent_type
                        }
                        typing_payload["type_to_special_supertypes"] = {
                            item.type_name: list(item.special_supertypes)
                            for item in object_types
                            if item.special_supertypes
                        }
            self._write_intermediate_typing_and_predicate_outputs(
                object_types=object_types,
                object_type_map=object_type_map,
                predicate_inventory=predicate_inventory,
            )

        self._logger.info("Stage 3/5: action schema consolidation")
        effective_allowed_action_schemas = allowed_action_schemas or typed_action_schemas
        if effective_allowed_action_schemas and hasattr(
            self._action_schema_consolidation_module, "consolidate_with_allowed_schemas"
        ):
            action_schemas, taxonomy_records = (
                self._action_schema_consolidation_module.consolidate_with_allowed_schemas(
                    steps,
                    taxonomy_records,
                    effective_allowed_action_schemas,
                    predicate_inventory=predicate_inventory,
                )
            )
        else:
            action_schemas, taxonomy_records = self._action_schema_consolidation_module.consolidate(
                steps,
                taxonomy_records,
                predicate_inventory=predicate_inventory,
            )
        self._logger.info("Consolidated to %d action schemas", len(action_schemas))
        base_action_schemas = list(action_schemas)
        active_predicate_inventory = _effect_learning_predicate_inventory(list(predicate_inventory))
        effect_review_guidance_by_signature: dict[tuple[str, str, int | None], dict[str, object]] = {}
        outer_max_iterations = max(
            0,
            int(getattr(self._manipulation_effect_module, "max_iterations", 0)),
        )
        manipulation_records: list[ManipulationEffectRecord] = []
        action_statistics: dict[str, list[ActionEffectStatistic]] = {}
        for outer_iteration in range(outer_max_iterations + 1):
            self._logger.info(
                "Stage 4/5 [outer iteration %d/%d]: manipulation effect learning",
                outer_iteration,
                outer_max_iterations,
            )
            effect_review_guidance_by_episode_step = _expand_effect_guidance_to_episode_steps(
                manipulation_records,
                effect_review_guidance_by_signature,
            )
            try:
                manipulation_records = self._manipulation_effect_module.learn_effects(
                    steps,
                    base_action_schemas,
                    taxonomy_records,
                    predicate_inventory=active_predicate_inventory,
                    object_types=object_types,
                    review_guidance_by_episode_step=effect_review_guidance_by_episode_step,
                )
            except TypeError as exc:
                if "review_guidance_by_episode_step" not in str(exc) and "object_types" not in str(exc):
                    raise
                try:
                    manipulation_records = self._manipulation_effect_module.learn_effects(
                        steps,
                        base_action_schemas,
                        taxonomy_records,
                        predicate_inventory=active_predicate_inventory,
                        object_types=object_types,
                    )
                except TypeError as inner_exc:
                    if "object_types" not in str(inner_exc):
                        raise
                    manipulation_records = self._manipulation_effect_module.learn_effects(
                        steps,
                        base_action_schemas,
                        taxonomy_records,
                        predicate_inventory=active_predicate_inventory,
                    )
            if self._effect_merge_module is not None and manipulation_records:
                self._logger.info("Stage 4.5/5: manipulation effect merge")
                manipulation_records = self._effect_merge_module.merge_records(manipulation_records)
            validate_manipulation_records_against_predicate_inventory(
                records=manipulation_records,
                predicate_inventory=active_predicate_inventory,
            )
            self._logger.info("Learned %d manipulation effect records", len(manipulation_records))

            self._logger.info("Stage 5/5: statistics and PDDL rendering")
            manipulation_records = classify_records_by_effect_statistics(manipulation_records)
            action_statistics = collect_action_effect_statistics(manipulation_records)
            if self._effect_variant_review_module is not None and manipulation_records:
                self._logger.info("Stage 5.1/5: effect-variant review and merge")
                manipulation_records, action_statistics = self._effect_variant_review_module.review_and_merge_variants(
                    steps=steps,
                    taxonomy_records=taxonomy_records,
                    records=manipulation_records,
                )
            if manipulation_records and hasattr(self._manipulation_effect_module, "revalidate_and_repair_records"):
                self._logger.info("Stage 5.2/5: post-merge goal validation and episode repair")
                manipulation_records, records_changed_after_revalidation = (
                    self._manipulation_effect_module.revalidate_and_repair_records(
                        steps=steps,
                        action_schemas=base_action_schemas,
                        taxonomy_records=taxonomy_records,
                        records=manipulation_records,
                        predicate_inventory=active_predicate_inventory,
                    )
                )
                if records_changed_after_revalidation:
                    self._logger.info("Stage 5.3/5: recomputing effect statistics once after post-merge repairs")
                records_changed_after_noop_pruning = False
                if hasattr(self._manipulation_effect_module, "prune_replay_noop_effects"):
                    manipulation_records, records_changed_after_noop_pruning = (
                        self._manipulation_effect_module.prune_replay_noop_effects(
                            manipulation_records,
                        )
                    )
                if records_changed_after_revalidation or records_changed_after_noop_pruning:
                    manipulation_records = classify_records_by_effect_statistics(manipulation_records)
                    action_statistics = collect_action_effect_statistics(manipulation_records)

            if self._effect_completeness_review_module is None or not manipulation_records:
                break

            episode_effect_results = getattr(
                self._manipulation_effect_module,
                "last_post_statistics_repair_results",
                None,
            )
            if not isinstance(episode_effect_results, dict) or not episode_effect_results:
                episode_effect_results = getattr(self._manipulation_effect_module, "last_episode_results", None)
            if not isinstance(episode_effect_results, dict) or not episode_effect_results:
                self._logger.info(
                    "Stage 5.35/5: skipping effect completeness review because no episode grounding results are available."
                )
                break

            review_predicate_comments = {
                item.predicate_name: item.comment for item in active_predicate_inventory if item.comment
            }
            review_action_schemas = attach_action_effects_to_schemas(
                base_action_schemas,
                manipulation_records,
                action_statistics,
            )
            rendered_review_domain_pddl = render_manipulation_domain_fragment(
                review_action_schemas,
                manipulation_records,
                action_statistics,
                predicate_inventory=active_predicate_inventory,
                predicate_comments=review_predicate_comments,
                object_types=object_types,
            )
            self._logger.info("Stage 5.35/5: serial VLM effect completeness review")
            completeness_plan = self._effect_completeness_review_module.review_and_suggest_repairs(
                steps=steps,
                taxonomy_records=taxonomy_records,
                records=manipulation_records,
                action_statistics=action_statistics,
                predicate_inventory=active_predicate_inventory,
                predicate_comments=review_predicate_comments,
                object_types=object_types,
                episode_results=episode_effect_results,
                rendered_domain_pddl=rendered_review_domain_pddl,
            )
            restart_outer_iteration = False
            while True:
                if not completeness_plan.should_apply_repair:
                    break
                if outer_iteration >= outer_max_iterations:
                    self._logger.info(
                        "Stage 5.35/5: completeness review suggested repair but outer max_iterations=%d is exhausted; keeping current artifacts.",
                        outer_max_iterations,
                    )
                    break
                active_predicate_inventory, predicates_changed = _merge_predicate_additions_into_inventory(
                    active_predicate_inventory,
                    completeness_plan,
                )
                effect_review_guidance_by_signature, guidance_changed = _merge_effect_completeness_guidance(
                    effect_review_guidance_by_signature,
                    completeness_plan,
                )
                self._logger.info(
                    "Stage 5.35/5: completeness review requested repair for [%s step %s], predicates_changed=%s, guidance_changed=%s.",
                    completeness_plan.episode_name,
                    completeness_plan.step_index,
                    predicates_changed,
                    guidance_changed,
                )
                if not predicates_changed and not guidance_changed:
                    self._logger.info(
                        "Stage 5.35/5: completeness review produced no effective change; stopping to avoid a no-op rerun."
                    )
                    break
                if (
                    guidance_changed
                    and not predicates_changed
                    and hasattr(self._manipulation_effect_module, "rerun_records_from_effect_guidance")
                ):
                    self._logger.info(
                        "Stage 5.35/5: applying local effect rerun for guidance-only repair on matching bucket/variant episodes."
                    )
                    manipulation_records, locally_changed = (
                        self._manipulation_effect_module.rerun_records_from_effect_guidance(
                            steps=steps,
                            action_schemas=base_action_schemas,
                            taxonomy_records=taxonomy_records,
                            records=manipulation_records,
                            predicate_inventory=active_predicate_inventory,
                            review_guidance_by_signature=effect_review_guidance_by_signature,
                        )
                    )
                    if not locally_changed:
                        self._logger.info(
                            "Stage 5.35/5: local effect rerun found no matching records to update; stopping to avoid a no-op rerun."
                        )
                        break
                    manipulation_records = classify_records_by_effect_statistics(manipulation_records)
                    action_statistics = collect_action_effect_statistics(manipulation_records)
                    target_bucket_variants = {
                        _effect_guidance_signature_key(
                            completeness_plan.canonical_action_name or "",
                            completeness_plan.effect_bucket or "",
                            completeness_plan.variant_rank,
                        )
                    }
                    target_action_outcomes = _action_outcomes_for_target_signatures(
                        manipulation_records,
                        target_bucket_variants,
                    )
                    if self._effect_variant_review_module is not None and target_action_outcomes:
                        self._logger.info(
                            "Stage 5.35/5: re-running effect-variant review only for the modified action/outcome group."
                        )
                        manipulation_records, action_statistics = (
                            self._effect_variant_review_module.review_and_merge_variants(
                                steps=steps,
                                taxonomy_records=taxonomy_records,
                                records=manipulation_records,
                                target_action_outcomes=target_action_outcomes,
                            )
                        )
                    review_action_schemas = attach_action_effects_to_schemas(
                        base_action_schemas,
                        manipulation_records,
                        action_statistics,
                    )
                    rendered_review_domain_pddl = render_manipulation_domain_fragment(
                        review_action_schemas,
                        manipulation_records,
                        action_statistics,
                        predicate_inventory=active_predicate_inventory,
                        predicate_comments=review_predicate_comments,
                        object_types=object_types,
                    )
                    self._logger.info(
                        "Stage 5.35/5: re-running completeness review only for the modified bucket/variant sample."
                    )
                    completeness_plan = self._effect_completeness_review_module.review_and_suggest_repairs(
                        steps=steps,
                        taxonomy_records=taxonomy_records,
                        records=manipulation_records,
                        action_statistics=action_statistics,
                        predicate_inventory=active_predicate_inventory,
                        predicate_comments=review_predicate_comments,
                        object_types=object_types,
                        episode_results=episode_effect_results,
                        rendered_domain_pddl=rendered_review_domain_pddl,
                        target_bucket_variants=target_bucket_variants,
                    )
                    if completeness_plan.should_apply_repair:
                        restart_outer_iteration = True
                        break
                    self._logger.info(
                        "Stage 5.35/5: modified bucket/variant now passes completeness review; continuing later completeness-review jobs."
                    )
                    completeness_plan = self._effect_completeness_review_module.review_and_suggest_repairs(
                        steps=steps,
                        taxonomy_records=taxonomy_records,
                        records=manipulation_records,
                        action_statistics=action_statistics,
                        predicate_inventory=active_predicate_inventory,
                        predicate_comments=review_predicate_comments,
                        object_types=object_types,
                        episode_results=episode_effect_results,
                        rendered_domain_pddl=rendered_review_domain_pddl,
                        start_after_bucket_variant=next(iter(target_bucket_variants)),
                    )
                    continue
                restart_outer_iteration = True
                break
            if restart_outer_iteration:
                continue
            break

        predicate_inventory = active_predicate_inventory
        action_schemas = attach_action_effects_to_schemas(
            base_action_schemas,
            manipulation_records,
            action_statistics,
        )
        if self._action_template_builder is None:
            action_name_map = _build_generic_action_name_map(
                taxonomy_records=taxonomy_records,
                action_schemas=action_schemas,
            )
        elif not action_name_map:
            action_name_map = _build_generic_action_name_map(
                taxonomy_records=taxonomy_records,
                action_schemas=action_schemas,
            )
        predicate_comments: dict[str, str] = {}
        if predicate_inventory:
            predicate_comments.update(
                {item.predicate_name: item.comment for item in predicate_inventory if item.comment}
            )
        if self._predicate_comment_module is not None:
            self._logger.info("Stage 5.5/5: predicate comment generation")
            generated_comments = self._predicate_comment_module.generate_predicate_comments(
                action_schemas=action_schemas,
                manipulation_records=manipulation_records,
            )
            predicate_comments.update(generated_comments)
        rendered_action_schema_pddl = render_action_schema_fragment(
            action_schemas,
            predicate_inventory=predicate_inventory,
            predicate_comments=predicate_comments,
            object_types=object_types,
        )
        rendered_manipulation_pddl = render_manipulation_domain_fragment(
            action_schemas,
            manipulation_records,
            action_statistics,
            predicate_inventory=predicate_inventory,
            predicate_comments=predicate_comments,
            object_types=object_types,
        )
        observation_status = {
            "status": "not_implemented",
            "message": "Observation action learning is reserved for a later phase.",
        }
        return ManipulationDomainLearningResult(
            episode_object_inventories=episode_object_inventories,
            taxonomy_records=taxonomy_records,
            action_schemas=action_schemas,
            manipulation_records=manipulation_records,
            action_statistics=action_statistics,
            rendered_action_schema_pddl=rendered_action_schema_pddl,
            rendered_manipulation_pddl=rendered_manipulation_pddl,
            observation_action_learning_status=observation_status,
            action_templates=action_templates,
            action_name_map=action_name_map,
            predicate_inventory=predicate_inventory,
            predicate_comments=predicate_comments,
            object_types=object_types,
            object_type_map=object_type_map,
            action_text_normalization_records=normalization_records,
        )

    def _restore_template_registries(self, action_templates: list[dict[str, Any]]) -> None:
        if not action_templates:
            return
        restored_templates = [induced_template_from_dict(item) for item in action_templates if isinstance(item, dict)]
        if not restored_templates:
            return
        for module in (
            self._action_text_preprocessing_module,
            self._action_taxonomy_module,
            self._action_schema_consolidation_module,
        ):
            registry = getattr(module, "registry", None)
            if registry is not None and hasattr(registry, "set_templates"):
                registry.set_templates(restored_templates)

    def write_outputs(self, result: ManipulationDomainLearningResult, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with (output_path / "action_taxonomy.jsonl").open("w", encoding="utf-8") as handle:
            for record in result.taxonomy_records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        if result.episode_object_inventories:
            with (output_path / "episode_object_inventory.jsonl").open("w", encoding="utf-8") as handle:
                for record in result.episode_object_inventories:
                    handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        (output_path / "action_schemas.json").write_text(
            json.dumps([schema.to_dict() for schema in result.action_schemas], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_path / "action_schemas.pddl").write_text(
            result.rendered_action_schema_pddl,
            encoding="utf-8",
        )
        with (output_path / "manipulation_records.jsonl").open("w", encoding="utf-8") as handle:
            for record in result.manipulation_records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        (output_path / "manipulation_effect_statistics.json").write_text(
            json.dumps(result.action_statistics_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_path / "manipulation_actions.pddl").write_text(
            result.rendered_manipulation_pddl,
            encoding="utf-8",
        )
        (output_path / "observation_action_learning_status.json").write_text(
            json.dumps(result.observation_action_learning_status, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
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
        if result.object_types:
            (output_path / "object_types.json").write_text(
                json.dumps([item.to_dict() for item in result.object_types], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        if result.object_type_map:
            (output_path / "object_type_map.json").write_text(
                json.dumps(result.object_type_map, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        episode_effect_results = getattr(
            self._manipulation_effect_module,
            "last_post_statistics_repair_results",
            None,
        )
        if not isinstance(episode_effect_results, dict) or not episode_effect_results:
            episode_effect_results = getattr(self._manipulation_effect_module, "last_episode_results", None)
        if isinstance(episode_effect_results, dict) and episode_effect_results:
            serialized_episode_results = {
                episode_name: episode_result.to_dict()
                for episode_name, episode_result in sorted(episode_effect_results.items())
            }
            (output_path / "episode_grounded_effect_learning_summary.json").write_text(
                json.dumps(
                    serialized_episode_results,
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            self._write_episode_effect_debug_outputs(
                output_path=output_path,
                episode_effect_results=episode_effect_results,
                action_schemas=result.action_schemas,
                predicate_inventory=result.predicate_inventory,
            )
            (output_path / "episode_problem_grounding_results.json").write_text(
                json.dumps(
                    serialized_episode_results,
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        effect_variant_review_summary = getattr(self._effect_variant_review_module, "last_review_summary", None)
        if isinstance(effect_variant_review_summary, dict) and effect_variant_review_summary:
            (output_path / "effect_variant_review_summary.json").write_text(
                json.dumps(effect_variant_review_summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        effect_completeness_review_summary = getattr(
            self._effect_completeness_review_module,
            "last_review_summary",
            None,
        )
        if isinstance(effect_completeness_review_summary, dict) and effect_completeness_review_summary:
            (output_path / "effect_completeness_review_summary.json").write_text(
                json.dumps(effect_completeness_review_summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        post_statistics_repair_summary = getattr(
            self._manipulation_effect_module,
            "last_post_statistics_repair_summary",
            None,
        )
        if isinstance(post_statistics_repair_summary, dict) and post_statistics_repair_summary:
            (output_path / "post_statistics_episode_repair_summary.json").write_text(
                json.dumps(post_statistics_repair_summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        if result.predicate_inventory:
            (output_path / "predicate_inventory.json").write_text(
                json.dumps([item.to_dict() for item in result.predicate_inventory], indent=2, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
        if result.predicate_comments:
            (output_path / "predicate_comments.json").write_text(
                json.dumps(result.predicate_comments, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    def _write_episode_effect_debug_outputs(
        self,
        *,
        output_path: Path,
        episode_effect_results: dict[str, Any],
        action_schemas: list[ActionSchema],
        predicate_inventory: list[PredicateSchema],
    ) -> None:
        debug_root = output_path / "episode_grounded_effect_learning_debug"
        debug_root.mkdir(parents=True, exist_ok=True)
        for episode_name, episode_result in sorted(episode_effect_results.items()):
            episode_dir = debug_root / episode_name
            episode_dir.mkdir(parents=True, exist_ok=True)
            domain_text = render_action_schema_fragment(
                action_schemas,
                predicate_inventory=predicate_inventory,
            )
            (episode_dir / "domain_file.pddl").write_text(domain_text, encoding="utf-8")
            (episode_dir / "problem_context.json").write_text(
                json.dumps(episode_result.to_dict().get("problem_context", {}), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            problem_context = getattr(episode_result, "problem_context", None)
            if problem_context is not None:
                for name, text in sorted(getattr(problem_context, "object_init_raw_llm_outputs", {}).items()):
                    if str(text).strip():
                        (episode_dir / f"object_init_{name}_raw_output.txt").write_text(str(text), encoding="utf-8")
                goal_raw_output = getattr(problem_context, "goal_inference_raw_output", None)
                if isinstance(goal_raw_output, str) and goal_raw_output.strip():
                    (episode_dir / "goal_inference_raw_output.txt").write_text(goal_raw_output, encoding="utf-8")

            final_dir = episode_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            (final_dir / "problem.pddl").write_text(str(getattr(episode_result, "problem_pddl", "")), encoding="utf-8")
            (final_dir / "problem_spec.json").write_text(
                json.dumps(getattr(episode_result, "problem_spec").to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self._write_jsonl(
                final_dir / "manipulation_records.jsonl",
                [item.to_dict() for item in getattr(episode_result, "manipulation_records", [])],
            )
            self._write_jsonl(
                final_dir / "grounded_steps.jsonl",
                [item.to_dict() for item in getattr(episode_result, "grounded_steps", [])],
            )
            (final_dir / "validation_report.json").write_text(
                json.dumps(
                    {
                        "validation_steps": [
                            item.to_dict() for item in getattr(episode_result, "validation_steps", [])
                        ],
                        "validation_issues": [
                            item.to_dict() for item in getattr(episode_result, "validation_issues", [])
                        ],
                        "goal_satisfied": bool(getattr(episode_result, "goal_satisfied", False)),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            self._write_raw_outputs_for_records(
                final_dir,
                getattr(episode_result, "manipulation_records", []),
            )

            iterations_dir = episode_dir / "iterations"
            iterations_dir.mkdir(parents=True, exist_ok=True)
            for iteration in getattr(episode_result, "iterations", []):
                iteration_dir = iterations_dir / f"iteration_{int(getattr(iteration, 'iteration_index', 0)):02d}"
                iteration_dir.mkdir(parents=True, exist_ok=True)
                (iteration_dir / "problem.pddl").write_text(
                    str(getattr(iteration, "problem_pddl", "")), encoding="utf-8"
                )
                (iteration_dir / "problem_spec.json").write_text(
                    json.dumps(getattr(iteration, "problem_spec").to_dict(), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                self._write_jsonl(
                    iteration_dir / "manipulation_records.jsonl",
                    [item.to_dict() for item in getattr(iteration, "manipulation_records", [])],
                )
                self._write_jsonl(
                    iteration_dir / "grounded_steps.jsonl",
                    [item.to_dict() for item in getattr(iteration, "grounded_steps", [])],
                )
                (iteration_dir / "validation_report.json").write_text(
                    json.dumps(
                        {
                            "validation_steps": [item.to_dict() for item in getattr(iteration, "validation_steps", [])],
                            "validation_issues": [
                                item.to_dict() for item in getattr(iteration, "validation_issues", [])
                            ],
                            "goal_satisfied": bool(getattr(iteration, "goal_satisfied", False)),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                self._write_raw_outputs_for_records(
                    iteration_dir,
                    getattr(iteration, "manipulation_records", []),
                )
                repair_plan = getattr(iteration, "repair_plan", None)
                if repair_plan is not None:
                    (iteration_dir / "repair_plan.json").write_text(
                        json.dumps(repair_plan.to_dict(), indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    raw_llm_output = getattr(repair_plan, "raw_llm_output", None)
                    if isinstance(raw_llm_output, str) and raw_llm_output.strip():
                        (iteration_dir / "repair_raw_output.txt").write_text(raw_llm_output, encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_raw_outputs_for_records(directory: Path, records: list[ManipulationEffectRecord]) -> None:
        for record in records:
            raw_output = getattr(record, "raw_llm_output", None)
            if not isinstance(raw_output, str) or not raw_output.strip():
                continue
            step_index = int(getattr(record, "step_index", -1))
            (directory / f"step_{step_index:03d}_effect_raw_output.txt").write_text(raw_output, encoding="utf-8")
