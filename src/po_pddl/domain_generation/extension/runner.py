from __future__ import annotations

import json
import logging
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from po_pddl.config import DEFAULT_MODEL
from po_pddl.domain_generation.infrastructure.artifact_io import (
    load_episode_name as _load_episode_name,
)
from po_pddl.domain_generation.infrastructure.artifact_io import (
    load_json,
    load_jsonl,
)
from po_pddl.domain_generation.infrastructure.artifact_io import (
    load_optional_jsonl as _read_optional_jsonl,
)
from po_pddl.domain_generation.infrastructure.artifact_io import (
    write_jsonl as _write_jsonl_dicts,
)
from po_pddl.domain_generation.pipeline.runner import LearningPipelineRunner, discover_episode_files
from po_pddl.domain_generation.stages.active_observation_learning.factory import (
    build_learner_from_args as build_active_observation_learner_from_args,
)
from po_pddl.domain_generation.stages.domain_merge import (
    annotate_rewards_and_apply_to_domain,
    merge_domain_with_observation_modules,
)
from po_pddl.domain_generation.stages.init_observation_learning.factory import (
    build_learner_from_args as build_init_observation_learner_from_args,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.effect_merge import (
    rewrite_records_using_action_schemas,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.factory import (
    build_manipulation_domain_learner_from_args,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.grounding_update import (
    collect_grounding_episode_record_updates,
    load_object_types,
    load_predicate_inventory,
    refresh_domain_learning_artifacts,
    write_refreshed_domain_learning_artifacts,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.models import (
    ActionEffectBranch,
    ActionSchema,
    ActionTaxonomyRecord,
    ManipulationEffectRecord,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.renderer import (
    collect_action_effect_statistics,
    render_manipulation_domain_fragment,
)
from po_pddl.domain_generation.stages.observation_postprocessing import (
    prune_passive_and_active_observation_outputs_by_action_preconditions,
)
from po_pddl.domain_generation.stages.passive_observation_learning.factory import (
    build_learner_from_args as build_passive_observation_learner_from_args,
)
from po_pddl.domain_generation.stages.precondition_learning.factory import (
    build_learner_from_args as build_precondition_learner_from_args,
)

from .artifacts import BundleArtifacts, resolve_bundle_artifacts
from .models import BundleUpdateResult, ExtensionReviewResult
from .modules import (
    LLMActionExtensionReviewModule,
)
from .observation_learning import (
    load_active_result,
    load_init_result,
    load_passive_result,
    merge_active_results,
    merge_init_results,
    merge_passive_results,
)
from .shared import load_llm_config

logger = logging.getLogger(__name__)
_PREDICATE_NAME_PATTERN = re.compile(r"^(?:not\s+)?([a-z][a-z0-9_]*)\(")


def _load_action_schemas(path: Path) -> list[ActionSchema]:
    payload = load_json(path)
    return [
        ActionSchema(
            canonical_action_name=str(item["canonical_action_name"]),
            action_category=str(item["action_category"]),
            parameter_count=int(item["parameter_count"]),
            parameter_roles=[str(role) for role in item.get("parameter_roles", [])],
            precondition_literals=[str(literal) for literal in item.get("precondition_literals", [])],
            schema_description=item.get("schema_description"),
            effect_branches=[
                ActionEffectBranch(
                    effect_bucket=str(branch.get("effect_bucket") or ""),
                    probability=float(branch.get("probability", 0.0)),
                    success=bool(branch.get("success")),
                    delta_add=[str(fact) for fact in branch.get("delta_add", [])],
                    delta_del=[str(fact) for fact in branch.get("delta_del", [])],
                )
                for branch in item.get("effect_branches", [])
            ],
        )
        for item in payload
    ]


def _load_manipulation_records(path: Path) -> list[ManipulationEffectRecord]:
    records: list[ManipulationEffectRecord] = []
    for item in load_jsonl(path):
        records.append(
            ManipulationEffectRecord(
                episode_name=str(item["episode_name"]),
                step_index=int(item["step_index"]),
                raw_action_text=str(item.get("raw_action_text", "")),
                canonical_action_name=str(item["canonical_action_name"]),
                action_arguments=[str(arg) for arg in item.get("action_arguments", [])],
                pre_observation_text=item.get("pre_observation_text"),
                post_observation_text=item.get("post_observation_text"),
                extra_info=item.get("extra_info"),
                delta_add=[str(fact) for fact in item.get("delta_add", [])],
                delta_del=[str(fact) for fact in item.get("delta_del", [])],
                effect_bucket=str(item["effect_bucket"]),
                success=bool(item.get("success")),
                execution_time_sec=(
                    float(item["execution_time_sec"])
                    if item.get("execution_time_sec") is not None
                    else None
                ),
            )
        )
    return records


def _predicate_names(literals: list[str]) -> set[str]:
    names: set[str] = set()
    for literal in literals:
        match = _PREDICATE_NAME_PATTERN.search(literal.strip())
        if match:
            names.add(match.group(1))
    return names


def _action_name_tokens(name: str) -> set[str]:
    tokens = {token for token in name.split("_") if token and token not in {"item", "object"}}
    return tokens or {name}


def _select_existing_action_schema_name(
    new_schema: ActionSchema,
    existing_schemas: list[ActionSchema],
) -> str | None:
    existing_names = {schema.canonical_action_name for schema in existing_schemas}
    if new_schema.canonical_action_name in existing_names:
        return new_schema.canonical_action_name

    new_predicates = _predicate_names(new_schema.precondition_literals)
    new_tokens = _action_name_tokens(new_schema.canonical_action_name)
    candidates: list[tuple[float, str]] = []
    for old_schema in existing_schemas:
        if old_schema.action_category != new_schema.action_category:
            continue
        if old_schema.parameter_count != new_schema.parameter_count:
            continue
        old_tokens = _action_name_tokens(old_schema.canonical_action_name)
        name_overlap = len(new_tokens & old_tokens) / max(1, len(new_tokens | old_tokens))
        name_similarity = SequenceMatcher(
            None, new_schema.canonical_action_name, old_schema.canonical_action_name
        ).ratio()
        role_overlap = len(set(new_schema.parameter_roles) & set(old_schema.parameter_roles)) / max(
            1,
            len(set(new_schema.parameter_roles) | set(old_schema.parameter_roles)),
        )
        old_predicates = _predicate_names(old_schema.precondition_literals)
        predicate_overlap = (
            len(new_predicates & old_predicates) / max(1, len(new_predicates | old_predicates))
            if (new_predicates or old_predicates)
            else 1.0
        )
        score = (0.45 * name_overlap) + (0.25 * name_similarity) + (0.15 * role_overlap) + (0.15 * predicate_overlap)
        if score >= 0.55:
            candidates.append((score, old_schema.canonical_action_name))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < 0.05:
        return None
    return candidates[0][1]


def _rewrite_effect_bucket_prefix(effect_bucket: str, old_action_name: str, new_action_name: str) -> str:
    if effect_bucket.startswith(f"{new_action_name}_"):
        return f"{old_action_name}_{effect_bucket[len(new_action_name) + 1 :]}"
    return effect_bucket


def _schema_roles_are_reusable(new_schema: ActionSchema, old_schema: ActionSchema) -> bool:
    if new_schema.parameter_count != old_schema.parameter_count:
        return False
    broad_roles = {"object", "movable_item", "fixed_item", "containable_item"}
    return all(
        old_role == new_role or old_role in broad_roles
        for new_role, old_role in zip(new_schema.parameter_roles, old_schema.parameter_roles)
    )


def _disambiguate_incompatible_schema_names(
    *,
    old_schemas: list[ActionSchema],
    new_schemas: list[ActionSchema],
    taxonomy_records: list[ActionTaxonomyRecord],
    manipulation_records: list[ManipulationEffectRecord],
) -> tuple[list[ActionSchema], list[ActionTaxonomyRecord], list[ManipulationEffectRecord], dict[str, str]]:
    old_by_name = {schema.canonical_action_name: schema for schema in old_schemas}
    occupied_names = set(old_by_name) | {schema.canonical_action_name for schema in new_schemas}
    rename_map: dict[str, str] = {}
    for schema in new_schemas:
        old_schema = old_by_name.get(schema.canonical_action_name)
        if old_schema is None or _schema_roles_are_reusable(schema, old_schema):
            continue
        role_suffix = "_".join(schema.parameter_roles) or "no_args"
        base_name = f"{schema.canonical_action_name}_typed_{role_suffix}"
        candidate = base_name
        index = 2
        while candidate in occupied_names:
            candidate = f"{base_name}_{index}"
            index += 1
        rename_map[schema.canonical_action_name] = candidate
        occupied_names.add(candidate)

    if not rename_map:
        return new_schemas, taxonomy_records, manipulation_records, {}

    rewritten_schemas = [
        ActionSchema(
            canonical_action_name=rename_map.get(schema.canonical_action_name, schema.canonical_action_name),
            action_category=schema.action_category,
            parameter_count=schema.parameter_count,
            parameter_roles=list(schema.parameter_roles),
            precondition_literals=list(schema.precondition_literals),
            schema_description=schema.schema_description,
            effect_branches=list(schema.effect_branches),
        )
        for schema in new_schemas
    ]
    rewritten_taxonomy = [
        ActionTaxonomyRecord(
            episode_name=record.episode_name,
            step_index=record.step_index,
            raw_action_text=record.raw_action_text,
            proposed_action_name=rename_map.get(record.proposed_action_name, record.proposed_action_name),
            canonical_action_name=rename_map.get(record.canonical_action_name, record.canonical_action_name),
            action_category=record.action_category,
            action_arguments=list(record.action_arguments),
            object_mentions=list(record.object_mentions),
            observation_text=record.observation_text,
            extra_info=record.extra_info,
            template_text=record.template_text,
            parameter_placeholders=list(record.parameter_placeholders),
            action_argument_types=list(record.action_argument_types),
        )
        for record in taxonomy_records
    ]
    rewritten_records: list[ManipulationEffectRecord] = []
    for record in manipulation_records:
        renamed = rename_map.get(record.canonical_action_name, record.canonical_action_name)
        rewritten_records.append(
            ManipulationEffectRecord(
                episode_name=record.episode_name,
                step_index=record.step_index,
                raw_action_text=record.raw_action_text,
                canonical_action_name=renamed,
                action_arguments=list(record.action_arguments),
                pre_observation_text=record.pre_observation_text,
                post_observation_text=record.post_observation_text,
                extra_info=record.extra_info,
                delta_add=list(record.delta_add),
                delta_del=list(record.delta_del),
                effect_bucket=_rewrite_effect_bucket_prefix(
                    record.effect_bucket,
                    renamed,
                    record.canonical_action_name,
                ),
                success=record.success,
                raw_llm_output=record.raw_llm_output,
                execution_time_sec=record.execution_time_sec,
            )
        )
    return rewritten_schemas, rewritten_taxonomy, rewritten_records, rename_map


def _rewrite_action_metadata_artifacts(root: Path, rename_map: dict[str, str]) -> None:
    if not rename_map:
        return
    action_map_path = root / "action_name_map.json"
    if action_map_path.exists():
        payload = load_json(action_map_path)
        actions = payload.get("actions_by_name") if isinstance(payload, dict) else None
        if isinstance(actions, dict):
            for old_name, new_name in rename_map.items():
                item = actions.pop(old_name, None)
                if not isinstance(item, dict):
                    continue
                item["action_name"] = new_name
                item["template_id"] = new_name
                buckets = item.get("effect_buckets")
                if isinstance(buckets, dict):
                    item["effect_buckets"] = {
                        key: _rewrite_effect_bucket_prefix(str(value), new_name, old_name)
                        for key, value in buckets.items()
                    }
                actions[new_name] = item
            _write_json(action_map_path, payload)
    templates_path = root / "action_templates.json"
    if templates_path.exists():
        templates = load_json(templates_path)
        if isinstance(templates, list):
            for item in templates:
                if not isinstance(item, dict):
                    continue
                old_name = str(item.get("canonical_action_name") or "")
                new_name = rename_map.get(old_name)
                if not new_name:
                    continue
                item["canonical_action_name"] = new_name
                item["template_id"] = new_name
                for bucket_key in ("success_effect_bucket", "failure_effect_bucket"):
                    if item.get(bucket_key):
                        item[bucket_key] = _rewrite_effect_bucket_prefix(
                            str(item[bucket_key]), new_name, old_name
                        )
            _write_json(templates_path, templates)


def _load_jsonl_taxonomy_records(path: Path) -> list[ActionTaxonomyRecord]:
    rows = _read_optional_jsonl(path)
    return [
        ActionTaxonomyRecord(
            episode_name=str(item["episode_name"]),
            step_index=int(item["step_index"]),
            raw_action_text=str(item["raw_action_text"]),
            proposed_action_name=str(item["proposed_action_name"]),
            canonical_action_name=str(item["canonical_action_name"]),
            action_category=str(item["action_category"]),
            action_arguments=[str(value) for value in item.get("action_arguments", [])],
            object_mentions=[str(value) for value in item.get("object_mentions", [])],
            observation_text=item.get("observation_text"),
            extra_info=item.get("extra_info"),
        )
        for item in rows
    ]


def _copy_text_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _merge_episode_rows(
    existing_rows: list[dict[str, Any]], replacement_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    replacement_episode_names = {
        str(item.get("episode_name") or "").strip()
        for item in replacement_rows
        if str(item.get("episode_name") or "").strip()
    }
    merged = [
        item for item in existing_rows if str(item.get("episode_name") or "").strip() not in replacement_episode_names
    ]
    merged.extend(replacement_rows)
    return sorted(
        merged,
        key=lambda item: (
            str(item.get("episode_name") or ""),
            int(item.get("step_index", -1)),
        ),
    )


def _merge_named_json_lists(
    existing_path: Path,
    current_path: Path,
    *,
    key: str,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in (existing_path, current_path):
        if not path.exists():
            continue
        payload = load_json(path)
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict) or not str(item.get(key) or "").strip():
                continue
            name = str(item[key])
            if name not in merged:
                merged[name] = dict(item)
                continue
            previous = merged[name]
            combined = dict(previous)
            combined.update(item)
            for list_key in ("member_object_names", "special_supertypes"):
                values = list(previous.get(list_key, [])) + list(item.get(list_key, []))
                if values:
                    combined[list_key] = sorted({str(value) for value in values})
            merged[name] = combined
    return [merged[name] for name in sorted(merged)]


def _merge_mapping_json(existing_path: Path, current_path: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in (existing_path, current_path):
        if not path.exists():
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    return merged


def _prepare_combined_manipulation_base_artifacts(
    *,
    bundle_artifacts: BundleArtifacts,
    current_domain_learning_dir: Path,
    output_dir: Path,
) -> None:
    shutil.copytree(bundle_artifacts.manipulation_dir, output_dir, dirs_exist_ok=True)

    shutil.copy2(current_domain_learning_dir / "action_schemas.json", output_dir / "action_schemas.json")

    current_taxonomy_rows = _read_optional_jsonl(current_domain_learning_dir / "action_taxonomy.jsonl")
    existing_taxonomy_rows = _read_optional_jsonl(bundle_artifacts.manipulation_dir / "action_taxonomy.jsonl")
    merged_taxonomy_rows = _merge_episode_rows(existing_taxonomy_rows, current_taxonomy_rows)
    if merged_taxonomy_rows:
        _write_jsonl_dicts(output_dir / "action_taxonomy.jsonl", merged_taxonomy_rows)

    current_manipulation_rows = _read_optional_jsonl(current_domain_learning_dir / "manipulation_records.jsonl")
    existing_manipulation_rows = _read_optional_jsonl(bundle_artifacts.manipulation_dir / "manipulation_records.jsonl")
    merged_manipulation_rows = _merge_episode_rows(existing_manipulation_rows, current_manipulation_rows)
    execution_times = {
        (
            str(item.get("episode_name") or ""),
            int(item.get("step_index", -1)),
            str(item.get("canonical_action_name") or ""),
        ): item.get("execution_time_sec")
        for item in merged_manipulation_rows
        if item.get("execution_time_sec") is not None
    }
    merged_action_schemas = _load_action_schemas(current_domain_learning_dir / "action_schemas.json")
    merged_manipulation_rows = [
        record.to_dict()
        for record in rewrite_records_using_action_schemas(
            [
                ManipulationEffectRecord(
                    episode_name=str(item["episode_name"]),
                    step_index=int(item["step_index"]),
                    raw_action_text=str(item.get("raw_action_text", "")),
                    canonical_action_name=str(item["canonical_action_name"]),
                    action_arguments=[str(arg) for arg in item.get("action_arguments", [])],
                    pre_observation_text=item.get("pre_observation_text"),
                    post_observation_text=item.get("post_observation_text"),
                    extra_info=item.get("extra_info"),
                    delta_add=[str(fact) for fact in item.get("delta_add", [])],
                    delta_del=[str(fact) for fact in item.get("delta_del", [])],
                    effect_bucket=str(item["effect_bucket"]),
                    success=bool(item.get("success")),
                    execution_time_sec=(
                        float(item["execution_time_sec"])
                        if item.get("execution_time_sec") is not None
                        else None
                    ),
                )
                for item in merged_manipulation_rows
            ],
            merged_action_schemas,
        )
    ]
    for item in merged_manipulation_rows:
        key = (
            str(item.get("episode_name") or ""),
            int(item.get("step_index", -1)),
            str(item.get("canonical_action_name") or ""),
        )
        if key in execution_times:
            item["execution_time_sec"] = execution_times[key]
    _write_jsonl_dicts(output_dir / "manipulation_records.jsonl", merged_manipulation_rows)

    for filename, key in (
        ("object_types.json", "type_name"),
        ("predicate_inventory.json", "predicate_name"),
        ("action_templates.json", "canonical_action_name"),
    ):
        rows = _merge_named_json_lists(
            bundle_artifacts.manipulation_dir / filename,
            current_domain_learning_dir / filename,
            key=key,
        )
        if rows:
            _write_json(output_dir / filename, rows)

    for filename in ("object_type_map.json", "action_name_map.json", "predicate_comments.json"):
        payload = _merge_mapping_json(
            bundle_artifacts.manipulation_dir / filename,
            current_domain_learning_dir / filename,
        )
        if payload:
            _write_json(output_dir / filename, payload)

    grounding_context = _merge_mapping_json(
        bundle_artifacts.manipulation_dir / "episode_problem_grounding_results.json",
        current_domain_learning_dir / "episode_problem_grounding_results.json",
    )
    if grounding_context:
        _write_json(output_dir / "episode_problem_grounding_results.json", grounding_context)

    for filename in ("action_text_normalization.jsonl", "episode_object_inventory.jsonl"):
        existing_rows = _read_optional_jsonl(bundle_artifacts.manipulation_dir / filename)
        current_rows = _read_optional_jsonl(current_domain_learning_dir / filename)
        rows = _merge_episode_rows(existing_rows, current_rows)
        if rows:
            _write_jsonl_dicts(output_dir / filename, rows)

    observation_status_source = current_domain_learning_dir / "observation_action_learning_status.json"
    if not observation_status_source.exists():
        observation_status_source = bundle_artifacts.manipulation_dir / "observation_action_learning_status.json"
    if observation_status_source.exists():
        shutil.copy2(observation_status_source, output_dir / "observation_action_learning_status.json")


def _identity_or_reuse_aliases(
    *,
    new_schemas: list[ActionSchema],
    old_schemas: list[ActionSchema],
    review_result: ExtensionReviewResult,
    allow_new: bool,
) -> tuple[dict[str, str], list[str], list[ActionSchema]]:
    old_names = {schema.canonical_action_name for schema in old_schemas}
    alias_map: dict[str, str] = {}
    approved_new_names = set(review_result.approved_new_schema_names if allow_new else [])
    for schema in new_schemas:
        name = schema.canonical_action_name
        if name in old_names:
            alias_map[name] = name
            continue
        if name in review_result.reusable_aliases:
            alias_map[name] = review_result.reusable_aliases[name]
            continue
        if not allow_new:
            heuristic = _select_existing_action_schema_name(schema, old_schemas)
            if heuristic is not None:
                alias_map[name] = heuristic
                continue
        if allow_new and name in approved_new_names:
            alias_map[name] = name
    approved_new_schemas = [
        schema
        for schema in new_schemas
        if schema.canonical_action_name in approved_new_names and schema.canonical_action_name not in old_names
    ]
    unmapped = sorted(
        schema.canonical_action_name for schema in new_schemas if schema.canonical_action_name not in alias_map
    )
    return alias_map, unmapped, approved_new_schemas


class BundleUpdateRunner:
    def __init__(
        self,
        *,
        config: str | None = None,
        config_name: str = "openai_config",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 5000,
        max_workers: int = 1,
        max_iterations: int = 3,
        smoothing: float = 0.0,
        verbose: bool = False,
        annotation_fps: float = 2.0,
    ) -> None:
        self.config = config
        self.config_name = config_name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.max_iterations = max_iterations
        self.smoothing = smoothing
        self.verbose = verbose
        self.annotation_fps = annotation_fps

    def _shared_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            config=self.config,
            config_name=self.config_name,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_workers=self.max_workers,
            max_iterations=self.max_iterations,
            smoothing=self.smoothing,
            verbose=self.verbose,
            annotation_fps=self.annotation_fps,
        )

    @staticmethod
    def _stage_layout(output_dir: str | Path) -> dict[str, Path]:
        root = Path(output_dir)
        return {
            "pre_scene_action_parsing_dir": root / "0_pre_scene_action_parsing",
            "scene_description_dir": root / "1_scene_description",
            "manipulation_learning_dir": root / "2_manipulation_learning",
            "action_review_dir": root / "3_action_review",
            "manipulation_merge_dir": root / "4_manipulation_merge",
            "problem_grounding_root": root / "5_problem_grounding",
            "passive_observation_learning_dir": root / "6_passive_observation_learning",
            "init_observation_learning_dir": root / "7_init_observation_learning",
            "active_observation_learning_dir": root / "8_active_observation_learning",
            "merged_dir": root / "9_merged_domain",
            "final_bundle_dir": root / "10_final_bundle",
        }

    def _build_action_review_module(self) -> LLMActionExtensionReviewModule:
        config = load_llm_config(self.config, config_name=self.config_name)
        model = self.model or config.get("model") or DEFAULT_MODEL
        api_key = self.api_key or config.get("api_key")
        base_url = self.base_url or config.get("base_url")
        temperature = (
            self.temperature
            if self.temperature is not None
            else (config.get("temperature") if config.get("temperature") is not None else 0.0)
        )
        return LLMActionExtensionReviewModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )

    def run(
        self,
        *,
        bundle_dir: str | Path,
        input_dir: str | Path,
        output_dir: str | Path,
    ) -> BundleUpdateResult:
        bundle_artifacts = resolve_bundle_artifacts(bundle_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stage = self._stage_layout(output_path)
        scene_description_dir = stage["scene_description_dir"]

        episode_files = discover_episode_files(input_dir)
        logger.info("Discovered %d new episodes under %s", len(episode_files), input_dir)
        annotation_helper_runner = LearningPipelineRunner(
            config=self.config,
            config_name=self.config_name,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_workers=self.max_workers,
            max_iterations=self.max_iterations,
            smoothing=self.smoothing,
            verbose=self.verbose,
            annotation_fps=self.annotation_fps,
        )
        logger.info("Stage 0/10: pre-scene action and object parsing on new episodes")
        pre_scene_action_parsing_dir = stage["pre_scene_action_parsing_dir"]
        pre_scene_runner = annotation_helper_runner._build_pre_scene_action_parsing_runner()
        pre_scene_result = pre_scene_runner.run(input_dir)
        pre_scene_runner.write_outputs(pre_scene_result, pre_scene_action_parsing_dir)
        allowed_object_names_by_episode = {
            item.episode_name: list(item.object_names) for item in pre_scene_result.episode_object_inventories
        }
        focus_object_names_by_episode_step: dict[str, dict[int, list[str]]] = {}
        for record in pre_scene_result.taxonomy_records:
            focus_object_names_by_episode_step.setdefault(record.episode_name, {})[record.step_index] = list(
                record.action_arguments
            )
        logger.info("Stage 1/10: scene description on new episodes")
        annotated_episode_files = annotation_helper_runner._prepare_scene_described_episode_files(
            episode_files=episode_files,
            scene_description_root=scene_description_dir,
            allowed_object_names_by_episode=allowed_object_names_by_episode,
            focus_object_names_by_episode_step=focus_object_names_by_episode_step,
        )
        annotated_input_dir = scene_description_dir

        old_action_schemas = _load_action_schemas(bundle_artifacts.action_schemas_file)
        old_manipulation_records = _load_manipulation_records(
            bundle_artifacts.manipulation_dir / "manipulation_records.jsonl"
        )
        old_final_manipulation_domain_file = bundle_artifacts.final_manipulation_domain_file

        logger.info("Stage 2/10: manipulation domain learning on new episodes")
        new_domain_learning_dir = stage["manipulation_learning_dir"]
        reusable_domain_files = (
            new_domain_learning_dir / "action_schemas.json",
            new_domain_learning_dir / "action_taxonomy.jsonl",
            new_domain_learning_dir / "manipulation_records.jsonl",
            new_domain_learning_dir / "episode_problem_grounding_results.json",
        )
        if all(path.exists() for path in reusable_domain_files):
            logger.info("Reusing complete manipulation-domain learning artifacts from %s", new_domain_learning_dir)
            loaded_new_action_schemas = _load_action_schemas(new_domain_learning_dir / "action_schemas.json")
            loaded_new_taxonomy_records = _load_jsonl_taxonomy_records(
                new_domain_learning_dir / "action_taxonomy.jsonl"
            )
            loaded_new_manipulation_records = _load_manipulation_records(
                new_domain_learning_dir / "manipulation_records.jsonl"
            )
        else:
            domain_learner, _ = build_manipulation_domain_learner_from_args(
                SimpleNamespace(
                    **vars(self._shared_args()),
                    input_dir=annotated_input_dir,
                    output_dir=new_domain_learning_dir,
                )
            )
            new_domain_learning_result = domain_learner.learn_from_directory_with_preparsed_artifacts(
                annotated_input_dir,
                preparsed_artifact_dir=pre_scene_action_parsing_dir,
            )
            domain_learner.write_outputs(new_domain_learning_result, new_domain_learning_dir)
            loaded_new_action_schemas = new_domain_learning_result.action_schemas
            loaded_new_taxonomy_records = new_domain_learning_result.taxonomy_records
            loaded_new_manipulation_records = new_domain_learning_result.manipulation_records
        (
            new_action_schemas,
            new_taxonomy_records,
            new_manipulation_records,
            incompatible_name_renames,
        ) = _disambiguate_incompatible_schema_names(
            old_schemas=old_action_schemas,
            new_schemas=loaded_new_action_schemas,
            taxonomy_records=loaded_new_taxonomy_records,
            manipulation_records=loaded_new_manipulation_records,
        )
        if incompatible_name_renames:
            logger.info("Disambiguated incompatible same-name schemas: %s", incompatible_name_renames)

        logger.info("Stage 3/10: action extension review")
        action_review_dir = stage["action_review_dir"]
        action_review_dir.mkdir(parents=True, exist_ok=True)
        action_review = self._build_action_review_module().review(
            existing_action_schemas=old_action_schemas,
            new_action_schemas=new_action_schemas,
            new_taxonomy_records=new_taxonomy_records,
            new_effect_records=new_manipulation_records,
        )
        _write_json(action_review_dir / "review_result.json", action_review.to_dict())
        if action_review.raw_llm_output:
            (action_review_dir / "review_raw_output.txt").write_text(
                action_review.raw_llm_output,
                encoding="utf-8",
            )

        action_alias_map, unmapped_new_action_schemas, approved_new_action_schemas = (
            _identity_or_reuse_aliases(
                new_schemas=new_action_schemas,
                old_schemas=old_action_schemas,
                review_result=action_review,
                allow_new=action_review.should_extend,
            )
        )
        combined_action_schemas = list(old_action_schemas) + [
            schema
            for schema in approved_new_action_schemas
            if schema.canonical_action_name not in {item.canonical_action_name for item in old_action_schemas}
        ]

        manipulation_merge_dir = stage["manipulation_merge_dir"]
        grounding_artifacts_dir = manipulation_merge_dir / "grounding_artifacts"
        shutil.copytree(
            stage["manipulation_learning_dir"],
            grounding_artifacts_dir,
            dirs_exist_ok=True,
        )
        _rewrite_action_metadata_artifacts(grounding_artifacts_dir, incompatible_name_renames)
        (grounding_artifacts_dir / "action_schemas.json").write_text(
            json.dumps([schema.to_dict() for schema in combined_action_schemas], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        rewritten_taxonomy_rows: list[dict] = []
        for record in new_taxonomy_records:
            row = record.to_dict()
            mapped_name = action_alias_map.get(record.canonical_action_name)
            if mapped_name is None:
                continue
            row["canonical_action_name"] = mapped_name
            row["proposed_action_name"] = mapped_name
            rewritten_taxonomy_rows.append(row)
        _write_jsonl_dicts(grounding_artifacts_dir / "action_taxonomy.jsonl", rewritten_taxonomy_rows)

        rewritten_manipulation_rows: list[dict] = []
        for record in new_manipulation_records:
            mapped_name = action_alias_map.get(record.canonical_action_name)
            if mapped_name is None:
                continue
            row = record.to_dict()
            row["effect_bucket"] = _rewrite_effect_bucket_prefix(
                str(row["effect_bucket"]),
                mapped_name,
                record.canonical_action_name,
            )
            row["canonical_action_name"] = mapped_name
            rewritten_manipulation_rows.append(row)
        rewritten_manipulation_rows = [
            record.to_dict()
            for record in rewrite_records_using_action_schemas(
                [
                    ManipulationEffectRecord(
                        episode_name=str(item["episode_name"]),
                        step_index=int(item["step_index"]),
                        raw_action_text=str(item["raw_action_text"]),
                        canonical_action_name=str(item["canonical_action_name"]),
                        action_arguments=[str(value) for value in item.get("action_arguments", [])],
                        pre_observation_text=item.get("pre_observation_text"),
                        post_observation_text=item.get("post_observation_text"),
                        extra_info=item.get("extra_info"),
                        delta_add=[str(value) for value in item.get("delta_add", [])],
                        delta_del=[str(value) for value in item.get("delta_del", [])],
                        effect_bucket=str(item["effect_bucket"]),
                        success=bool(item.get("success")),
                        execution_time_sec=(
                            float(item["execution_time_sec"])
                            if item.get("execution_time_sec") is not None
                            else None
                        ),
                    )
                    for item in rewritten_manipulation_rows
                ],
                combined_action_schemas,
            )
        ]
        _write_jsonl_dicts(grounding_artifacts_dir / "manipulation_records.jsonl", rewritten_manipulation_rows)
        _write_json(
            grounding_artifacts_dir / "grounding_alias_map.json",
            {
                "action_alias_map": action_alias_map,
                "unmapped_new_action_schemas": unmapped_new_action_schemas,
                "approved_new_action_schemas": [
                    schema.canonical_action_name for schema in approved_new_action_schemas
                ],
            },
        )

        grounding_seed_artifacts_dir = manipulation_merge_dir / "grounding_seed_artifacts"
        _prepare_combined_manipulation_base_artifacts(
            bundle_artifacts=bundle_artifacts,
            current_domain_learning_dir=grounding_artifacts_dir,
            output_dir=grounding_seed_artifacts_dir,
        )

        logger.info("Stage 4/10: merge manipulation artifacts")
        current_domain_learning_dir = grounding_seed_artifacts_dir
        current_domain_file = manipulation_merge_dir / "final_manipulation_domain.pddl"
        if rewritten_manipulation_rows:
            renamed_new_manipulation_records = [
                ManipulationEffectRecord(
                    episode_name=str(item["episode_name"]),
                    step_index=int(item["step_index"]),
                    raw_action_text=str(item["raw_action_text"]),
                    canonical_action_name=str(item["canonical_action_name"]),
                    action_arguments=[str(value) for value in item.get("action_arguments", [])],
                    pre_observation_text=item.get("pre_observation_text"),
                    post_observation_text=item.get("post_observation_text"),
                    extra_info=item.get("extra_info"),
                    delta_add=[str(value) for value in item.get("delta_add", [])],
                    delta_del=[str(value) for value in item.get("delta_del", [])],
                    effect_bucket=str(item["effect_bucket"]),
                    success=bool(item["success"]),
                    execution_time_sec=(
                        float(item["execution_time_sec"])
                        if item.get("execution_time_sec") is not None
                        else None
                    ),
                )
                for item in rewritten_manipulation_rows
            ]
            candidate_records = rewrite_records_using_action_schemas(
                old_manipulation_records + renamed_new_manipulation_records,
                combined_action_schemas,
            )
            candidate_statistics = collect_action_effect_statistics(candidate_records)
            predicate_comments: dict[str, str] = {}
            predicate_comments_path = current_domain_learning_dir / "predicate_comments.json"
            if predicate_comments_path.exists():
                loaded_comments = load_json(predicate_comments_path)
                if isinstance(loaded_comments, dict):
                    predicate_comments = {
                        str(key): str(value)
                        for key, value in loaded_comments.items()
                        if str(key).strip() and str(value).strip()
                    }
            candidate_domain_text = render_manipulation_domain_fragment(
                combined_action_schemas,
                candidate_records,
                candidate_statistics,
                predicate_inventory=load_predicate_inventory(
                    current_domain_learning_dir / "predicate_inventory.json"
                ),
                predicate_comments=predicate_comments,
                object_types=load_object_types(current_domain_learning_dir / "object_types.json"),
            )
            candidate_domain_file = manipulation_merge_dir / "candidate_domain.pddl"
            candidate_domain_file.write_text(candidate_domain_text, encoding="utf-8")
            current_domain_file.write_text(candidate_domain_text, encoding="utf-8")
        else:
            _copy_text_file(old_final_manipulation_domain_file, current_domain_file)

        logger.info("Stage 5/10: grounding all new episodes with selected manipulation domain")
        grounding_helper_runner = LearningPipelineRunner(
            config=self.config,
            config_name=self.config_name,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_workers=self.max_workers,
            max_iterations=self.max_iterations,
            smoothing=self.smoothing,
            verbose=self.verbose,
            annotation_fps=self.annotation_fps,
        )
        episode_grounding_pairs: list[tuple[Path, Path]] = []
        shutil.rmtree(stage["problem_grounding_root"], ignore_errors=True)
        stage["problem_grounding_root"].mkdir(parents=True, exist_ok=True)
        for episode_file in annotated_episode_files:
            episode_name = _load_episode_name(episode_file)
            result, pair = grounding_helper_runner._run_grounding_loop(
                domain_file=current_domain_file,
                episode_file=episode_file,
                domain_learning_dir=current_domain_learning_dir,
                grounding_root=stage["problem_grounding_root"],
            )
            if result.issue_count > 0:
                logger.info(
                    "Grounding for %s completed with %d issue(s); no domain-repair loop will be applied.",
                    episode_name,
                    result.issue_count,
                )
            episode_grounding_pairs.append(pair)

        manipulation_update_dir = manipulation_merge_dir / "manipulation_update"
        manipulation_update_dir.mkdir(parents=True, exist_ok=True)
        combined_base_artifact_dir = manipulation_merge_dir / "combined_base_artifacts"
        _prepare_combined_manipulation_base_artifacts(
            bundle_artifacts=bundle_artifacts,
            current_domain_learning_dir=current_domain_learning_dir,
            output_dir=combined_base_artifact_dir,
        )
        grounding_update_collection = collect_grounding_episode_record_updates(
            episode_grounding_pairs,
            require_zero_issues=False,
        )
        refreshed_domain_learning = refresh_domain_learning_artifacts(
            base_artifact_dir=combined_base_artifact_dir,
            repaired_domain_file=current_domain_file,
            grounding_updates=grounding_update_collection.updates,
            include_non_converged_updates=True,
        )
        shutil.copytree(combined_base_artifact_dir, manipulation_update_dir, dirs_exist_ok=True)
        write_refreshed_domain_learning_artifacts(
            refreshed_domain_learning,
            base_artifact_dir=combined_base_artifact_dir,
            output_dir=manipulation_update_dir,
        )
        logger.info("Stage 5.5/10: learning action preconditions from grounded state-before candidates")
        precondition_learner = build_precondition_learner_from_args(self._shared_args())
        precondition_summary = precondition_learner.learn_from_groundings(
            artifact_dir=manipulation_update_dir,
            domain_file=current_domain_file,
            episode_grounding_pairs=episode_grounding_pairs,
        )
        precondition_learning_dir = manipulation_merge_dir / "precondition_learning"
        LearningPipelineRunner._write_precondition_learning_snapshot(
            precondition_learning_dir=precondition_learning_dir,
            precondition_summary=precondition_summary,
            source_domain_learning_dir=manipulation_update_dir,
        )
        current_domain_file = precondition_learning_dir / "domain_with_observation_actions.pddl"
        (manipulation_update_dir / "final_manipulation_domain.pddl").write_text(
            current_domain_file.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        _write_json(
            manipulation_update_dir / "update_diagnostics.json",
            {
                "action_alias_map": action_alias_map,
                "approved_new_action_schemas": [
                    schema.canonical_action_name for schema in approved_new_action_schemas
                ],
                "unmapped_new_action_schemas": unmapped_new_action_schemas,
                "replaced_episode_names": refreshed_domain_learning.manipulation_update.replaced_episode_names,
                "skipped_episode_reasons": {
                    **grounding_update_collection.skipped_episode_reasons,
                    **refreshed_domain_learning.manipulation_update.skipped_episode_reasons,
                },
            },
        )

        has_observation_inputs = LearningPipelineRunner._has_observation_learning_inputs(
            domain_learning_dir=manipulation_update_dir,
            scene_description_dir=scene_description_dir,
            episode_grounding_pairs=episode_grounding_pairs,
        )
        passive_observation_dir = bundle_artifacts.passive_observation_dir
        init_observation_dir = bundle_artifacts.init_observation_dir
        active_observation_dir = bundle_artifacts.active_observation_dir
        observation_updated = False
        if has_observation_inputs:
            logger.info("Stage 6/10: update passive observation model")
            passive_observation_dir = stage["passive_observation_learning_dir"]
            passive_learner, _ = build_passive_observation_learner_from_args(
                SimpleNamespace(
                    **vars(self._shared_args()),
                    domain_file=current_domain_file,
                    output_dir=passive_observation_dir,
                    manipulation_artifact_dir=manipulation_update_dir,
                    scene_description_dir=scene_description_dir,
                )
            )
            new_passive_result = passive_learner.learn_from_pairs(
                domain_file=current_domain_file,
                episode_grounding_pairs=episode_grounding_pairs,
            )
            passive_result = merge_passive_results(
                load_passive_result(bundle_artifacts.passive_observation_summary_file),
                new_passive_result,
            )
            passive_learner.write_outputs(passive_result, passive_observation_dir)

            logger.info("Stage 7/10: update initial observation model")
            init_observation_dir = stage["init_observation_learning_dir"]
            init_learner, _ = build_init_observation_learner_from_args(
                SimpleNamespace(
                    **vars(self._shared_args()),
                    domain_file=current_domain_file,
                    output_dir=init_observation_dir,
                    manipulation_artifact_dir=manipulation_update_dir,
                    scene_description_dir=scene_description_dir,
                )
            )
            new_init_result = init_learner.learn_from_pairs(
                domain_file=current_domain_file,
                episode_grounding_pairs=episode_grounding_pairs,
            )
            init_result = merge_init_results(
                load_init_result(bundle_artifacts.init_observation_summary_file),
                new_init_result,
            )
            init_learner.write_outputs(init_result, init_observation_dir)
            observation_updated = True

        has_active_observation_inputs = (
            has_observation_inputs
            and LearningPipelineRunner._has_active_observation_learning_inputs(
                domain_learning_dir=manipulation_update_dir,
                episode_grounding_pairs=episode_grounding_pairs,
            )
        )
        if has_active_observation_inputs:
            logger.info("Stage 8/10: update active observation model")
            active_observation_dir = stage["active_observation_learning_dir"]
            active_learner, _ = build_active_observation_learner_from_args(
                SimpleNamespace(
                    **vars(self._shared_args()),
                    domain_file=current_domain_file,
                    output_dir=active_observation_dir,
                    manipulation_artifact_dir=manipulation_update_dir,
                    scene_description_dir=scene_description_dir,
                    passive_observation_learning_dir=passive_observation_dir,
                    init_observation_learning_dir=init_observation_dir,
                )
            )
            new_active_result = active_learner.learn_from_pairs(
                domain_file=current_domain_file,
                episode_grounding_pairs=episode_grounding_pairs,
            )
            active_result = merge_active_results(
                load_active_result(bundle_artifacts.active_observation_summary_file),
                new_active_result,
            )
            active_learner.write_outputs(active_result, active_observation_dir)
            observation_updated = True

        if passive_observation_dir is not None and active_observation_dir is not None:
            prune_passive_and_active_observation_outputs_by_action_preconditions(
                precondition_learning_dir=precondition_learning_dir,
                passive_observation_learning_dir=passive_observation_dir,
                active_observation_learning_dir=active_observation_dir,
            )

        logger.info("Stage 9/10: merge updated manipulation and observation domains")
        merged_dir = stage["merged_dir"]
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_domain_file = merged_dir / "final_merged_domain.pddl"
        observation_module_files = [
            path
            for path in (
                (
                    passive_observation_dir / "passive_observation_module.pddl"
                    if passive_observation_dir
                    else None
                ),
                (
                    init_observation_dir / "init_observation_module.pddl"
                    if init_observation_dir
                    else None
                ),
                (
                    active_observation_dir / "active_observation_module.pddl"
                    if active_observation_dir
                    else None
                ),
            )
            if path is not None and path.exists()
        ]
        observation_module_texts = [path.read_text(encoding="utf-8") for path in observation_module_files]
        merged_domain_text = merge_domain_with_observation_modules(
            current_domain_file.read_text(encoding="utf-8"),
            observation_module_texts,
        )
        merged_domain_file.write_text(merged_domain_text, encoding="utf-8")
        LearningPipelineRunner._write_combined_observation_learning_outputs(
            merged_dir=merged_dir,
            passive_observation_learning_dir=passive_observation_dir,
            init_observation_learning_dir=init_observation_dir,
            active_observation_learning_dir=active_observation_dir,
        )
        reward_summary = annotate_rewards_and_apply_to_domain(
            episode_files=annotated_episode_files,
            domain_learning_dir=manipulation_update_dir,
            observation_learning_dir=merged_dir,
            merged_domain_file=merged_domain_file,
        )
        _write_json(merged_dir / "reward_annotation_summary.json", reward_summary.to_dict())

        logger.info("Stage 10/10: write final reusable bundle")
        combined_grounding_dir = merged_dir / "problem_grounding_all"
        if bundle_artifacts.problem_grounding_dir is not None:
            shutil.copytree(bundle_artifacts.problem_grounding_dir, combined_grounding_dir, dirs_exist_ok=True)
        shutil.copytree(stage["problem_grounding_root"], combined_grounding_dir, dirs_exist_ok=True)
        LearningPipelineRunner._write_final_bundle(
            bundle_dir=stage["final_bundle_dir"],
            final_manipulation_domain_file=current_domain_file,
            domain_learning_dir=manipulation_update_dir,
            precondition_learning_dir=precondition_learning_dir,
            passive_observation_learning_dir=passive_observation_dir,
            init_observation_learning_dir=init_observation_dir,
            active_observation_learning_dir=active_observation_dir,
            problem_grounding_root=combined_grounding_dir,
            merged_domain_file=merged_domain_file,
        )

        result = BundleUpdateResult(
            scene_description_dir=str(scene_description_dir),
            manipulation_learning_dir=str(stage["manipulation_learning_dir"]),
            action_review_dir=str(action_review_dir),
            manipulation_merge_dir=str(manipulation_merge_dir),
            problem_grounding_root=str(stage["problem_grounding_root"]),
            passive_observation_learning_dir=(str(passive_observation_dir) if passive_observation_dir else None),
            init_observation_learning_dir=(str(init_observation_dir) if init_observation_dir else None),
            active_observation_learning_dir=(str(active_observation_dir) if active_observation_dir else None),
            merged_domain_file=str(merged_domain_file),
            final_bundle_dir=str(stage["final_bundle_dir"]),
            updated_episode_count=len(annotated_episode_files),
            action_schema_extended=action_review.should_extend,
            observation_updated=observation_updated,
        )
        _write_json(output_path / "bundle_update_summary.json", result.to_dict())
        return result
