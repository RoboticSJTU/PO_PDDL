from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from po_pddl.core.parser import ParsedDomain, SExpr, parse_domain
from po_pddl.domain_generation.infrastructure.artifact_io import (
    load_episode_payload,
    load_json,
    load_json_object,
    load_jsonl,
    write_jsonl,
)
from po_pddl.domain_generation.stages.domain_comments import (
    extract_action_comments,
    extract_predicate_comments,
)

from .effect_merge import rewrite_records_using_action_schemas
from .models import (
    ActionEffectBranch,
    ActionEffectStatistic,
    ActionSchema,
    ManipulationEffectRecord,
    ObjectTypeDefinition,
    PredicateSchema,
)
from .renderer import (
    attach_action_effects_to_schemas,
    classify_records_by_effect_statistics,
    collect_action_effect_statistics,
    render_action_schema_fragment,
    render_manipulation_domain_fragment,
)


@dataclass(frozen=True)
class GroundingEpisodeRecordUpdate:
    episode_name: str
    source_episode_file: str
    source_grounding_dir: str
    issue_count: int
    manipulation_records: list[ManipulationEffectRecord]


@dataclass(frozen=True)
class GroundingEpisodeRecordCollection:
    updates: list[GroundingEpisodeRecordUpdate]
    skipped_episode_reasons: dict[str, str]


@dataclass(frozen=True)
class ManipulationArtifactsUpdate:
    manipulation_records: list[ManipulationEffectRecord]
    action_statistics: dict[str, list[ActionEffectStatistic]]
    replaced_episode_names: list[str]
    skipped_episode_reasons: dict[str, str]

    def action_statistics_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            action_name: [item.to_dict() for item in stats] for action_name, stats in self.action_statistics.items()
        }


@dataclass(frozen=True)
class DomainLearningArtifactsRefresh:
    action_schemas: list[ActionSchema]
    manipulation_update: ManipulationArtifactsUpdate
    predicate_inventory: list[PredicateSchema] = field(default_factory=list)
    predicate_comments: dict[str, str] = field(default_factory=dict)

    def action_schemas_dicts(self) -> list[dict[str, Any]]:
        return [schema.to_dict() for schema in self.action_schemas]


def load_manipulation_records(path: str | Path) -> list[ManipulationEffectRecord]:
    return [_manipulation_record_from_row(item) for item in load_jsonl(path)]


def load_object_types(path: str | Path) -> list[ObjectTypeDefinition]:
    rows = load_json(path)
    if isinstance(rows, dict):
        rows = rows.get("object_types", [])
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array at {path}")
    object_types: list[ObjectTypeDefinition] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        type_name = str(item.get("type_name") or "").strip()
        if not type_name:
            continue
        member_object_names = [
            str(value).strip() for value in item.get("member_object_names", []) if str(value).strip()
        ]
        parent_type = str(item.get("parent_type") or "").strip() or None
        special_supertypes = [str(value).strip() for value in item.get("special_supertypes", []) if str(value).strip()]
        object_types.append(
            ObjectTypeDefinition(
                type_name=type_name,
                member_object_names=member_object_names,
                parent_type=parent_type,
                special_supertypes=special_supertypes,
            )
        )
    return object_types


def load_predicate_inventory(path: str | Path) -> list[PredicateSchema]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array at {path}")
    inventory: list[PredicateSchema] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        predicate_name = str(item.get("predicate_name") or "").strip()
        if not predicate_name:
            continue
        inventory.append(
            PredicateSchema(
                predicate_name=predicate_name,
                parameter_types=[str(value).strip() for value in item.get("parameter_types", []) if str(value).strip()],
                comment=(str(item.get("comment")).strip() if item.get("comment") is not None else None),
                predicate_kind=item.get("predicate_kind"),
                is_static_feature=bool(item.get("is_static_feature", False)),
            )
        )
    return inventory


def grounded_step_to_manipulation_record(
    step: dict[str, Any],
    episode_payload: dict[str, Any],
) -> ManipulationEffectRecord:
    if str(step.get("action_category") or "").strip() != "manipulation":
        raise ValueError("grounded_step_to_manipulation_record expects a manipulation step.")

    effect_bucket = str(step.get("effect_bucket") or "").strip()
    canonical_action_name = str(step.get("canonical_action_name") or "").strip()
    if not canonical_action_name:
        raise ValueError("Grounded manipulation step is missing canonical_action_name.")
    if not effect_bucket:
        raise ValueError("Grounded manipulation step is missing effect_bucket.")

    steps_payload = {
        int(item["step_index"]): item
        for item in episode_payload.get("steps", [])
        if isinstance(item, dict) and "step_index" in item
    }
    step_index = int(step["step_index"])
    current_step = steps_payload.get(step_index, {})
    previous_step = steps_payload.get(step_index - 1, {})

    return ManipulationEffectRecord(
        episode_name=str(step.get("episode_name") or episode_payload.get("episode_name") or ""),
        step_index=step_index,
        raw_action_text=str(step.get("raw_action_text") or ""),
        canonical_action_name=canonical_action_name,
        action_arguments=[str(arg) for arg in step.get("ground_arguments", [])],
        pre_observation_text=_optional_text(previous_step.get("observation_text")),
        post_observation_text=_optional_text(current_step.get("observation_text")),
        extra_info=_optional_text(current_step.get("extra_info")),
        delta_add=[str(fact) for fact in step.get("delta_add", [])],
        delta_del=[str(fact) for fact in step.get("delta_del", [])],
        effect_bucket=effect_bucket,
        success=bool(step.get("success")),
    )


def load_grounding_episode_record_update(
    episode_file: str | Path,
    grounding_dir: str | Path,
    *,
    require_zero_issues: bool = True,
) -> GroundingEpisodeRecordUpdate | None:
    episode_path = Path(episode_file)
    grounding_path = Path(grounding_dir)
    episode_payload = load_episode_payload(episode_path)
    episode_name = str(episode_payload.get("episode_name", episode_path.parent.name))
    validation_payload = load_json_object(grounding_path / "validation_report.json")
    issue_count = int(validation_payload.get("issue_count", 0))
    if require_zero_issues and issue_count != 0:
        return None

    grounded_rows = load_jsonl(grounding_path / "grounded_trajectory.jsonl")
    manipulation_records = [
        grounded_step_to_manipulation_record(item, episode_payload)
        for item in grounded_rows
        if str(item.get("action_category") or "").strip() == "manipulation"
    ]
    return GroundingEpisodeRecordUpdate(
        episode_name=episode_name,
        source_episode_file=str(episode_path),
        source_grounding_dir=str(grounding_path),
        issue_count=issue_count,
        manipulation_records=manipulation_records,
    )


def collect_grounding_episode_record_updates(
    episode_grounding_pairs: Iterable[tuple[str | Path, str | Path]],
    *,
    require_zero_issues: bool = True,
) -> GroundingEpisodeRecordCollection:
    updates: list[GroundingEpisodeRecordUpdate] = []
    skipped_episode_reasons: dict[str, str] = {}
    for episode_file, grounding_dir in episode_grounding_pairs:
        episode_path = Path(episode_file)
        grounding_path = Path(grounding_dir)
        episode_payload = load_episode_payload(episode_path)
        episode_name = str(episode_payload.get("episode_name", episode_path.parent.name))
        validation_payload = load_json_object(grounding_path / "validation_report.json")
        issue_count = int(validation_payload.get("issue_count", 0))
        if require_zero_issues and issue_count != 0:
            skipped_episode_reasons[episode_name] = (
                f"validation issue_count={issue_count} at {grounding_path / 'validation_report.json'}"
            )
            continue
        update = load_grounding_episode_record_update(
            episode_path,
            grounding_path,
            require_zero_issues=False,
        )
        if update is None:
            skipped_episode_reasons[episode_name] = f"failed to load grounding update from {grounding_path}"
            continue
        updates.append(update)
    return GroundingEpisodeRecordCollection(
        updates=updates,
        skipped_episode_reasons=skipped_episode_reasons,
    )


def rebuild_manipulation_artifacts(
    existing_records: list[ManipulationEffectRecord],
    grounding_updates: Iterable[GroundingEpisodeRecordUpdate],
    *,
    skipped_episode_reasons: dict[str, str] | None = None,
) -> ManipulationArtifactsUpdate:
    updates_by_episode = {
        item.episode_name: {
            record.step_index: record
            for record in sorted(item.manipulation_records, key=lambda record: record.step_index)
        }
        for item in grounding_updates
    }
    replaced_episode_names = sorted(updates_by_episode)

    preserved_records_by_episode: dict[str, list[ManipulationEffectRecord]] = {}
    episode_order: list[str] = []
    for record in existing_records:
        bucket = preserved_records_by_episode.setdefault(record.episode_name, [])
        if not bucket:
            episode_order.append(record.episode_name)
        bucket.append(record)

    for episode_name in replaced_episode_names:
        if episode_name not in episode_order:
            episode_order.append(episode_name)

    updated_records: list[ManipulationEffectRecord] = []
    for episode_name in episode_order:
        existing_episode_records = preserved_records_by_episode.get(episode_name, [])
        update_map = updates_by_episode.get(episode_name)
        if not update_map:
            updated_records.extend(existing_episode_records)
            continue

        kept_records = [update_map.get(record.step_index, record) for record in existing_episode_records]
        seen_step_indices = {record.step_index for record in existing_episode_records}
        appended_records = [
            record for step_index, record in sorted(update_map.items()) if step_index not in seen_step_indices
        ]
        merged_episode_records = sorted(
            kept_records + appended_records,
            key=lambda record: record.step_index,
        )
        updated_records.extend(merged_episode_records)

    updated_records = classify_records_by_effect_statistics(updated_records)
    action_statistics = collect_action_effect_statistics(updated_records)
    return ManipulationArtifactsUpdate(
        manipulation_records=updated_records,
        action_statistics=action_statistics,
        replaced_episode_names=replaced_episode_names,
        skipped_episode_reasons=dict(skipped_episode_reasons or {}),
    )


def write_manipulation_artifacts_update(
    result: ManipulationArtifactsUpdate,
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output_path / "manipulation_records.jsonl",
        [record.to_dict() for record in result.manipulation_records],
    )
    (output_path / "manipulation_effect_statistics.json").write_text(
        json.dumps(result.action_statistics_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def refresh_domain_learning_artifacts(
    *,
    base_artifact_dir: str | Path,
    repaired_domain_file: str | Path,
    grounding_updates: Iterable[GroundingEpisodeRecordUpdate],
    include_non_converged_updates: bool = False,
) -> DomainLearningArtifactsRefresh:
    base_dir = Path(base_artifact_dir)
    repaired_domain_path = Path(repaired_domain_file)
    existing_action_schemas = load_action_schemas(base_dir / "action_schemas.json")
    existing_records = load_manipulation_records(base_dir / "manipulation_records.jsonl")
    filtered_updates = list(grounding_updates)
    if not include_non_converged_updates:
        filtered_updates = [item for item in filtered_updates if item.issue_count == 0]
    manipulation_update = rebuild_manipulation_artifacts(existing_records, filtered_updates)
    action_schemas = action_schemas_from_domain_file(
        repaired_domain_path,
        existing_action_schemas=existing_action_schemas,
    )
    predicate_inventory: list[PredicateSchema] = []
    predicate_inventory_path = base_dir / "predicate_inventory.json"
    if predicate_inventory_path.exists():
        predicate_inventory = load_predicate_inventory(predicate_inventory_path)
    repaired_domain_text = repaired_domain_path.read_text(encoding="utf-8")
    predicate_comments = extract_predicate_comments(repaired_domain_text)
    return DomainLearningArtifactsRefresh(
        action_schemas=action_schemas,
        manipulation_update=manipulation_update,
        predicate_inventory=predicate_inventory,
        predicate_comments=predicate_comments,
    )


def write_refreshed_domain_learning_artifacts(
    refresh: DomainLearningArtifactsRefresh,
    *,
    base_artifact_dir: str | Path,
    output_dir: str | Path,
) -> None:
    base_dir = Path(base_artifact_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    action_taxonomy_path = base_dir / "action_taxonomy.jsonl"
    if action_taxonomy_path.exists():
        target_path = output_path / "action_taxonomy.jsonl"
        if action_taxonomy_path.resolve() != target_path.resolve():
            shutil.copy2(action_taxonomy_path, target_path)

    observation_status_path = base_dir / "observation_action_learning_status.json"
    if observation_status_path.exists():
        target_path = output_path / "observation_action_learning_status.json"
        if observation_status_path.resolve() != target_path.resolve():
            shutil.copy2(observation_status_path, target_path)

    rewritten_records = rewrite_records_using_action_schemas(
        refresh.manipulation_update.manipulation_records,
        refresh.action_schemas,
    )
    rewritten_records = classify_records_by_effect_statistics(rewritten_records)
    rewritten_statistics = collect_action_effect_statistics(rewritten_records)
    action_schemas = attach_action_effects_to_schemas(
        refresh.action_schemas,
        rewritten_records,
        rewritten_statistics,
    )
    predicate_comments: dict[str, str] = {}
    predicate_comments_path = base_dir / "predicate_comments.json"
    if predicate_comments_path.exists():
        loaded_comments = load_json(predicate_comments_path)
        if isinstance(loaded_comments, dict):
            predicate_comments = {
                str(key): str(value)
                for key, value in loaded_comments.items()
                if str(key).strip() and str(value).strip()
            }
    predicate_comments.update(
        {
            str(key): str(value)
            for key, value in refresh.predicate_comments.items()
            if str(key).strip() and str(value).strip()
        }
    )
    object_types: list[ObjectTypeDefinition] = []
    object_types_path = base_dir / "object_types.json"
    if object_types_path.exists():
        object_types = load_object_types(object_types_path)

    (output_path / "action_schemas.json").write_text(
        json.dumps([schema.to_dict() for schema in action_schemas], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_path / "action_schemas.pddl").write_text(
        render_action_schema_fragment(
            action_schemas,
            predicate_inventory=refresh.predicate_inventory,
            predicate_comments=predicate_comments,
            object_types=object_types,
        ),
        encoding="utf-8",
    )
    write_manipulation_artifacts_update(
        ManipulationArtifactsUpdate(
            manipulation_records=rewritten_records,
            action_statistics=rewritten_statistics,
            replaced_episode_names=list(refresh.manipulation_update.replaced_episode_names),
            skipped_episode_reasons=dict(refresh.manipulation_update.skipped_episode_reasons),
        ),
        output_path,
    )
    (output_path / "manipulation_actions.pddl").write_text(
        render_manipulation_domain_fragment(
            action_schemas,
            rewritten_records,
            rewritten_statistics,
            predicate_inventory=refresh.predicate_inventory,
            predicate_comments=predicate_comments,
            object_types=object_types,
        ),
        encoding="utf-8",
    )
    if predicate_comments:
        (output_path / "predicate_comments.json").write_text(
            json.dumps(predicate_comments, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def load_action_schemas(path: str | Path) -> list[ActionSchema]:
    rows = load_json_object_or_array(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array at {path}")
    return [_action_schema_from_row(item) for item in rows]


def action_schemas_from_domain_file(
    path: str | Path,
    *,
    existing_action_schemas: list[ActionSchema] | None = None,
) -> list[ActionSchema]:
    return action_schemas_from_domain_text(
        Path(path).read_text(encoding="utf-8"),
        existing_action_schemas=existing_action_schemas,
    )


def action_schemas_from_domain_text(
    domain_text: str,
    *,
    existing_action_schemas: list[ActionSchema] | None = None,
) -> list[ActionSchema]:
    parsed_domain = parse_domain(domain_text)
    action_comments = extract_action_comments(domain_text)
    return action_schemas_from_parsed_domain(
        parsed_domain,
        existing_action_schemas=existing_action_schemas,
        action_comments=action_comments,
    )


def action_schemas_from_parsed_domain(
    parsed_domain: ParsedDomain,
    *,
    existing_action_schemas: list[ActionSchema] | None = None,
    action_comments: dict[str, str] | None = None,
) -> list[ActionSchema]:
    existing_by_name = {schema.canonical_action_name: schema for schema in (existing_action_schemas or [])}
    parsed_action_names = {schema.action.name for schema in parsed_domain.actions}
    refreshed_schemas: list[ActionSchema] = []

    for parsed_action in parsed_domain.actions:
        existing = existing_by_name.get(parsed_action.action.name)
        parameter_roles = [type_name.strip() or "object" for _, type_name in parsed_action.parameter_types]
        parameter_count = len(parsed_action.action.params)
        if not parameter_roles:
            parameter_roles = (
                list(existing.parameter_roles)
                if existing is not None and existing.parameter_roles
                else ["object"] * parameter_count
            )
        if len(parameter_roles) < parameter_count:
            parameter_roles.extend(["object"] * (parameter_count - len(parameter_roles)))
        refreshed_schemas.append(
            ActionSchema(
                canonical_action_name=parsed_action.action.name,
                action_category=(existing.action_category if existing is not None else "manipulation"),
                parameter_count=parameter_count,
                parameter_roles=parameter_roles[:parameter_count],
                precondition_literals=_precondition_expr_to_literals(parsed_action.precondition),
                schema_description=(
                    (action_comments or {}).get(parsed_action.action.name)
                    or (existing.schema_description if existing is not None else None)
                ),
                effect_branches=(list(existing.effect_branches) if existing is not None else []),
            )
        )

    preserved_non_parsed = [
        schema for schema in (existing_action_schemas or []) if schema.canonical_action_name not in parsed_action_names
    ]
    return sorted(
        refreshed_schemas + preserved_non_parsed,
        key=lambda item: item.canonical_action_name,
    )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def load_json_object_or_array(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _manipulation_record_from_row(item: dict[str, Any]) -> ManipulationEffectRecord:
    return ManipulationEffectRecord(
        episode_name=str(item["episode_name"]),
        step_index=int(item["step_index"]),
        raw_action_text=str(item.get("raw_action_text") or ""),
        canonical_action_name=str(item["canonical_action_name"]),
        action_arguments=[str(arg) for arg in item.get("action_arguments", [])],
        pre_observation_text=_optional_text(item.get("pre_observation_text")),
        post_observation_text=_optional_text(item.get("post_observation_text")),
        extra_info=_optional_text(item.get("extra_info")),
        delta_add=[str(fact) for fact in item.get("delta_add", [])],
        delta_del=[str(fact) for fact in item.get("delta_del", [])],
        effect_bucket=str(item["effect_bucket"]),
        success=bool(item.get("success")),
        execution_time_sec=(
            float(item["execution_time_sec"]) if item.get("execution_time_sec") is not None else None
        ),
    )


def _action_schema_from_row(item: dict[str, Any]) -> ActionSchema:
    return ActionSchema(
        canonical_action_name=str(item["canonical_action_name"]),
        action_category=str(item.get("action_category") or "manipulation"),
        parameter_count=int(item.get("parameter_count", 0)),
        parameter_roles=[str(role) for role in item.get("parameter_roles", [])],
        precondition_literals=[str(literal) for literal in item.get("precondition_literals", [])],
        schema_description=_optional_text(item.get("schema_description")),
        effect_branches=[
            ActionEffectBranch(
                effect_bucket=str(branch.get("effect_bucket") or ""),
                probability=float(branch.get("probability", 0.0)),
                success=bool(branch.get("success")),
                delta_add=[str(fact) for fact in branch.get("delta_add", [])],
                delta_del=[str(fact) for fact in branch.get("delta_del", [])],
                variant_rank=(int(branch.get("variant_rank")) if branch.get("variant_rank") is not None else None),
                fixed_delta_add=[str(fact) for fact in branch.get("fixed_delta_add", [])],
                fixed_delta_del=[str(fact) for fact in branch.get("fixed_delta_del", [])],
                residual_delta_add=[str(fact) for fact in branch.get("residual_delta_add", [])],
                residual_delta_del=[str(fact) for fact in branch.get("residual_delta_del", [])],
            )
            for branch in item.get("effect_branches", [])
        ],
    )


def _precondition_expr_to_literals(expr: SExpr | None) -> list[str]:
    if expr is None:
        return []
    if isinstance(expr, str):
        return [f"{expr}()"]
    if not expr:
        return []
    head = expr[0]
    if head == "and":
        literals: list[str] = []
        for item in expr[1:]:
            literals.extend(_precondition_expr_to_literals(item))
        return literals
    if head == "not":
        if len(expr) != 2 or not isinstance(expr[1], list) or not expr[1]:
            raise ValueError(f"Unsupported negated precondition expression: {expr!r}")
        inner = expr[1]
        inner_head = inner[0]
        if not isinstance(inner_head, str):
            raise ValueError(f"Unsupported negated precondition expression: {expr!r}")
        arguments = [str(item) for item in inner[1:]]
        return [f"not {_fact_text(inner_head, arguments)}"]
    if isinstance(head, str):
        return [_fact_text(head, [str(item) for item in expr[1:]])]
    raise ValueError(f"Unsupported precondition expression: {expr!r}")


def _fact_text(predicate: str, arguments: list[str]) -> str:
    if arguments:
        return f"{predicate}({','.join(arguments)})"
    return f"{predicate}()"


__all__ = [
    "DomainLearningArtifactsRefresh",
    "GroundingEpisodeRecordCollection",
    "GroundingEpisodeRecordUpdate",
    "ManipulationArtifactsUpdate",
    "action_schemas_from_domain_file",
    "action_schemas_from_domain_text",
    "action_schemas_from_parsed_domain",
    "collect_grounding_episode_record_updates",
    "refresh_domain_learning_artifacts",
    "grounded_step_to_manipulation_record",
    "load_action_schemas",
    "load_grounding_episode_record_update",
    "load_manipulation_records",
    "rebuild_manipulation_artifacts",
    "write_refreshed_domain_learning_artifacts",
    "write_manipulation_artifacts_update",
]
