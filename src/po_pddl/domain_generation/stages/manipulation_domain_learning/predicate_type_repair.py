from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from po_pddl.core.parser import parse_domain
from po_pddl.domain_generation.infrastructure.artifact_io import load_json, write_jsonl
from po_pddl.domain_generation.infrastructure.fact_utils import (
    format_symbolic_literal,
    parse_positive_symbolic_fact,
    parse_symbolic_literal,
)
from po_pddl.domain_generation.stages.problem_grounding.models import (
    ObjectDeclaration,
    ProblemSpec,
)
from po_pddl.domain_generation.stages.problem_grounding.renderer import (
    render_problem_pddl,
)

from .grounding_update import load_action_schemas, load_manipulation_records, load_object_types
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
    collect_action_effect_statistics,
    render_action_schema_fragment,
    render_manipulation_domain_fragment,
)


@dataclass(frozen=True)
class PredicateTypeRepairResult:
    action_schemas: list[ActionSchema]
    manipulation_records: list[ManipulationEffectRecord]
    predicate_inventory: list[PredicateSchema]
    predicate_comments: dict[str, str]
    action_statistics: dict[str, list[ActionEffectStatistic]]
    renamed_predicates: dict[str, dict[str, str]]
    changed: bool

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "changed": self.changed,
            "renamed_predicates": dict(self.renamed_predicates),
            "predicate_inventory": [item.to_dict() for item in self.predicate_inventory],
        }


def repair_predicate_types_from_artifacts(
    artifact_dir: str | Path,
) -> PredicateTypeRepairResult:
    artifact_path = Path(artifact_dir)
    action_schemas = load_action_schemas(artifact_path / "action_schemas.json")
    manipulation_records = load_manipulation_records(artifact_path / "manipulation_records.jsonl")
    predicate_inventory = _load_predicate_inventory(artifact_path / "predicate_inventory.json")
    predicate_comments = _load_predicate_comments(artifact_path / "predicate_comments.json")
    object_types = (
        load_object_types(artifact_path / "object_types.json") if (artifact_path / "object_types.json").exists() else []
    )
    return repair_predicate_types(
        action_schemas=action_schemas,
        manipulation_records=manipulation_records,
        predicate_inventory=predicate_inventory,
        predicate_comments=predicate_comments,
        object_types=object_types,
    )


def write_predicate_type_repair_artifacts(
    *,
    result: PredicateTypeRepairResult,
    base_artifact_dir: str | Path,
    output_dir: str | Path,
) -> None:
    base_path = Path(base_artifact_dir)
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.copytree(base_path, output_path)

    object_types_path = output_path / "object_types.json"
    object_types = load_object_types(object_types_path) if object_types_path.exists() else []

    (output_path / "action_schemas.json").write_text(
        json.dumps([schema.to_dict() for schema in result.action_schemas], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_jsonl(
        output_path / "manipulation_records.jsonl",
        [record.to_dict() for record in result.manipulation_records],
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
    (output_path / "predicate_inventory.json").write_text(
        json.dumps([item.to_dict() for item in result.predicate_inventory], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_path / "predicate_comments.json").write_text(
        json.dumps(result.predicate_comments, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    action_schema_text = render_action_schema_fragment(
        result.action_schemas,
        predicate_inventory=result.predicate_inventory,
        predicate_comments=result.predicate_comments,
        object_types=object_types,
    )
    manipulation_domain_text = render_manipulation_domain_fragment(
        result.action_schemas,
        result.manipulation_records,
        result.action_statistics,
        predicate_inventory=result.predicate_inventory,
        predicate_comments=result.predicate_comments,
        object_types=object_types,
    )
    parse_domain(action_schema_text)
    parse_domain(manipulation_domain_text)
    (output_path / "action_schemas.pddl").write_text(action_schema_text, encoding="utf-8")
    (output_path / "manipulation_actions.pddl").write_text(manipulation_domain_text, encoding="utf-8")
    (output_path / "predicate_type_repair_summary.json").write_text(
        json.dumps(result.to_summary_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _rewrite_cached_grounding_artifacts(
        artifact_dir=output_path,
        predicate_inventory=result.predicate_inventory,
        object_types=object_types,
    )


def repair_predicate_types(
    *,
    action_schemas: list[ActionSchema],
    manipulation_records: list[ManipulationEffectRecord],
    predicate_inventory: list[PredicateSchema],
    predicate_comments: dict[str, str],
    object_types: list[ObjectTypeDefinition] | None = None,
) -> PredicateTypeRepairResult:
    inventory_by_name = {item.predicate_name: item for item in predicate_inventory}
    type_parents, type_special_supertypes = _build_type_hierarchy_maps(object_types or [])
    observed_signatures = _collect_observed_predicate_signatures(
        action_schemas,
        inventory_by_name,
        type_parents=type_parents,
        type_special_supertypes=type_special_supertypes,
    )
    rename_by_predicate = _build_predicate_rename_map(
        observed_signatures,
        inventory_by_name=inventory_by_name,
        type_parents=type_parents,
        type_special_supertypes=type_special_supertypes,
    )

    rewritten_action_schemas = [
        ActionSchema(
            canonical_action_name=schema.canonical_action_name,
            action_category=schema.action_category,
            parameter_count=schema.parameter_count,
            parameter_roles=list(schema.parameter_roles),
            precondition_literals=[
                _rewrite_literal(
                    literal,
                    parameter_roles=schema.parameter_roles,
                    rename_by_predicate=rename_by_predicate,
                    inventory_by_name=inventory_by_name,
                    type_parents=type_parents,
                    type_special_supertypes=type_special_supertypes,
                )
                for literal in schema.precondition_literals
            ],
            schema_description=schema.schema_description,
            effect_branches=[
                ActionEffectBranch(
                    effect_bucket=branch.effect_bucket,
                    probability=branch.probability,
                    success=branch.success,
                    delta_add=[
                        _rewrite_fact(
                            fact,
                            parameter_roles=schema.parameter_roles,
                            rename_by_predicate=rename_by_predicate,
                            inventory_by_name=inventory_by_name,
                            type_parents=type_parents,
                            type_special_supertypes=type_special_supertypes,
                        )
                        for fact in branch.delta_add
                    ],
                    delta_del=[
                        _rewrite_fact(
                            fact,
                            parameter_roles=schema.parameter_roles,
                            rename_by_predicate=rename_by_predicate,
                            inventory_by_name=inventory_by_name,
                            type_parents=type_parents,
                            type_special_supertypes=type_special_supertypes,
                        )
                        for fact in branch.delta_del
                    ],
                    variant_rank=branch.variant_rank,
                    fixed_delta_add=[
                        _rewrite_fact(
                            fact,
                            parameter_roles=schema.parameter_roles,
                            rename_by_predicate=rename_by_predicate,
                            inventory_by_name=inventory_by_name,
                            type_parents=type_parents,
                            type_special_supertypes=type_special_supertypes,
                        )
                        for fact in branch.fixed_delta_add
                    ],
                    fixed_delta_del=[
                        _rewrite_fact(
                            fact,
                            parameter_roles=schema.parameter_roles,
                            rename_by_predicate=rename_by_predicate,
                            inventory_by_name=inventory_by_name,
                            type_parents=type_parents,
                            type_special_supertypes=type_special_supertypes,
                        )
                        for fact in branch.fixed_delta_del
                    ],
                    residual_delta_add=[
                        _rewrite_fact(
                            fact,
                            parameter_roles=schema.parameter_roles,
                            rename_by_predicate=rename_by_predicate,
                            inventory_by_name=inventory_by_name,
                            type_parents=type_parents,
                            type_special_supertypes=type_special_supertypes,
                        )
                        for fact in branch.residual_delta_add
                    ],
                    residual_delta_del=[
                        _rewrite_fact(
                            fact,
                            parameter_roles=schema.parameter_roles,
                            rename_by_predicate=rename_by_predicate,
                            inventory_by_name=inventory_by_name,
                            type_parents=type_parents,
                            type_special_supertypes=type_special_supertypes,
                        )
                        for fact in branch.residual_delta_del
                    ],
                    extra_pddl_effect_conjuncts=list(branch.extra_pddl_effect_conjuncts),
                )
                for branch in schema.effect_branches
            ],
        )
        for schema in action_schemas
    ]

    schema_by_action = {schema.canonical_action_name: schema for schema in action_schemas}
    rewritten_records = [
        ManipulationEffectRecord(
            episode_name=record.episode_name,
            step_index=record.step_index,
            raw_action_text=record.raw_action_text,
            canonical_action_name=record.canonical_action_name,
            action_arguments=list(record.action_arguments),
            pre_observation_text=record.pre_observation_text,
            post_observation_text=record.post_observation_text,
            extra_info=record.extra_info,
            delta_add=[
                _rewrite_fact_for_record(
                    fact,
                    record=record,
                    schema=schema_by_action.get(record.canonical_action_name),
                    rename_by_predicate=rename_by_predicate,
                    inventory_by_name=inventory_by_name,
                    type_parents=type_parents,
                    type_special_supertypes=type_special_supertypes,
                )
                for fact in record.delta_add
            ],
            delta_del=[
                _rewrite_fact_for_record(
                    fact,
                    record=record,
                    schema=schema_by_action.get(record.canonical_action_name),
                    rename_by_predicate=rename_by_predicate,
                    inventory_by_name=inventory_by_name,
                    type_parents=type_parents,
                    type_special_supertypes=type_special_supertypes,
                )
                for fact in record.delta_del
            ],
            effect_bucket=record.effect_bucket,
            success=record.success,
            raw_llm_output=record.raw_llm_output,
        )
        for record in manipulation_records
    ]

    rewritten_inventory = _rewrite_predicate_inventory(
        predicate_inventory=predicate_inventory,
        predicate_comments=predicate_comments,
        observed_signatures=observed_signatures,
        rename_by_predicate=rename_by_predicate,
        type_parents=type_parents,
        type_special_supertypes=type_special_supertypes,
    )
    rewritten_comments = _rewrite_predicate_comments(
        predicate_inventory=rewritten_inventory,
        predicate_comments=predicate_comments,
        rename_by_predicate=rename_by_predicate,
    )
    action_statistics = collect_action_effect_statistics(rewritten_records)
    rewritten_action_schemas = attach_action_effects_to_schemas(
        rewritten_action_schemas,
        rewritten_records,
        action_statistics,
    )

    changed = (
        any(rename_by_predicate.values())
        or [schema.to_dict() for schema in rewritten_action_schemas] != [schema.to_dict() for schema in action_schemas]
        or [record.to_dict() for record in rewritten_records] != [record.to_dict() for record in manipulation_records]
        or [item.to_dict() for item in rewritten_inventory] != [item.to_dict() for item in predicate_inventory]
        or rewritten_comments != predicate_comments
    )
    renamed_summary = {
        predicate_name: {"|".join(signature): renamed for signature, renamed in sorted(signature_map.items())}
        for predicate_name, signature_map in sorted(rename_by_predicate.items())
        if signature_map
    }
    return PredicateTypeRepairResult(
        action_schemas=rewritten_action_schemas,
        manipulation_records=rewritten_records,
        predicate_inventory=rewritten_inventory,
        predicate_comments=rewritten_comments,
        action_statistics=action_statistics,
        renamed_predicates=renamed_summary,
        changed=changed,
    )


def _load_predicate_inventory(path: Path) -> list[PredicateSchema]:
    if not path.exists():
        return []
    payload = load_json(path)
    if not isinstance(payload, list):
        return []
    results: list[PredicateSchema] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("predicate_name") or "").strip()
        if not name:
            continue
        results.append(
            PredicateSchema(
                predicate_name=name,
                parameter_types=[str(type_name).strip() or "object" for type_name in item.get("parameter_types", [])],
                comment=item.get("comment"),
                predicate_kind=item.get("predicate_kind"),
                is_static_feature=bool(item.get("is_static_feature", False)),
            )
        )
    return results


def _load_predicate_comments(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items() if str(key).strip() and str(value).strip()}


def _collect_observed_predicate_signatures(
    action_schemas: list[ActionSchema],
    inventory_by_name: dict[str, PredicateSchema],
    *,
    type_parents: dict[str, str],
    type_special_supertypes: dict[str, set[str]],
) -> dict[str, set[tuple[str, ...]]]:
    observed: dict[str, set[tuple[str, ...]]] = {}
    for schema in action_schemas:
        for literal in schema.precondition_literals:
            _negated, predicate, arguments = parse_symbolic_literal(literal)
            resolved_signature = _resolve_signature(
                arguments,
                parameter_roles=schema.parameter_roles,
                argument_name_to_type=None,
                inventory_by_name=inventory_by_name,
                predicate_name=predicate,
            )
            normalized_arguments, normalized_signature = _normalize_binary_argument_order(
                predicate_name=predicate,
                arguments=arguments,
                resolved_signature=resolved_signature,
                inventory_by_name=inventory_by_name,
                type_parents=type_parents,
                type_special_supertypes=type_special_supertypes,
            )
            del normalized_arguments
            observed.setdefault(predicate, set()).add(normalized_signature)
        for branch in schema.effect_branches:
            for fact in (
                list(branch.delta_add)
                + list(branch.delta_del)
                + list(branch.fixed_delta_add)
                + list(branch.fixed_delta_del)
                + list(branch.residual_delta_add)
                + list(branch.residual_delta_del)
            ):
                predicate, arguments = parse_positive_symbolic_fact(fact)
                resolved_signature = _resolve_signature(
                    arguments,
                    parameter_roles=schema.parameter_roles,
                    argument_name_to_type=None,
                    inventory_by_name=inventory_by_name,
                    predicate_name=predicate,
                )
                normalized_arguments, normalized_signature = _normalize_binary_argument_order(
                    predicate_name=predicate,
                    arguments=arguments,
                    resolved_signature=resolved_signature,
                    inventory_by_name=inventory_by_name,
                    type_parents=type_parents,
                    type_special_supertypes=type_special_supertypes,
                )
                del normalized_arguments
                observed.setdefault(predicate, set()).add(normalized_signature)
    return observed


def _build_predicate_rename_map(
    observed_signatures: dict[str, set[tuple[str, ...]]],
    *,
    inventory_by_name: dict[str, PredicateSchema],
    type_parents: dict[str, str],
    type_special_supertypes: dict[str, set[str]],
) -> dict[str, dict[tuple[str, ...], str]]:
    rename_map: dict[str, dict[tuple[str, ...], str]] = {}
    for predicate_name, signatures in observed_signatures.items():
        if len(signatures) <= 1:
            continue
        existing = inventory_by_name.get(predicate_name)
        if existing is not None:
            expected_signature = tuple(_normalize_type_name(type_name) for type_name in existing.parameter_types)
            if all(
                _signature_compatible_with_expected(
                    signature=signature,
                    expected_signature=expected_signature,
                    type_parents=type_parents,
                    type_special_supertypes=type_special_supertypes,
                )
                for signature in signatures
            ):
                continue
        rename_map[predicate_name] = {
            signature: _split_predicate_name(predicate_name, signature) for signature in sorted(signatures)
        }
    return rename_map


def _split_predicate_name(predicate_name: str, signature: tuple[str, ...]) -> str:
    if not signature:
        return f"{predicate_name}_zero_arity"
    suffix = "_".join(_sanitize_identifier(type_name) for type_name in signature)
    return f"{predicate_name}_{suffix}"


def _sanitize_identifier(text: str) -> str:
    cleaned = str(text).strip().replace("-", "_")
    return cleaned or "object"


def _parameter_index(argument: str) -> int | None:
    if argument.startswith("?arg") and argument[4:].isdigit():
        return int(argument[4:])
    if argument.startswith("?param_") and argument[7:].isdigit():
        return int(argument[7:]) - 1
    return None


def _normalize_type_name(type_name: str | None) -> str:
    cleaned = str(type_name or "").strip()
    return cleaned or "object"


def _resolve_signature(
    arguments: list[str],
    *,
    parameter_roles: list[str],
    argument_name_to_type: dict[str, str] | None,
    inventory_by_name: dict[str, PredicateSchema],
    predicate_name: str,
) -> tuple[str, ...]:
    existing = inventory_by_name.get(predicate_name)
    existing_types = list(existing.parameter_types) if existing is not None else []
    resolved: list[str] = []
    argument_type_map = argument_name_to_type or {}
    for index, argument in enumerate(arguments):
        parameter_index = _parameter_index(argument)
        if parameter_index is not None and parameter_index < len(parameter_roles):
            resolved.append(_normalize_type_name(parameter_roles[parameter_index]))
            continue
        if argument in argument_type_map:
            resolved.append(_normalize_type_name(argument_type_map[argument]))
            continue
        if index < len(existing_types):
            resolved.append(_normalize_type_name(existing_types[index]))
            continue
        resolved.append("object")
    return tuple(resolved)


def _rename_predicate(
    predicate_name: str,
    signature: tuple[str, ...],
    rename_by_predicate: dict[str, dict[tuple[str, ...], str]],
) -> str:
    signature_map = rename_by_predicate.get(predicate_name)
    if not signature_map:
        return predicate_name
    return signature_map.get(signature, predicate_name)


def _rewrite_literal(
    literal: str,
    *,
    parameter_roles: list[str],
    rename_by_predicate: dict[str, dict[tuple[str, ...], str]],
    inventory_by_name: dict[str, PredicateSchema],
    type_parents: dict[str, str],
    type_special_supertypes: dict[str, set[str]],
) -> str:
    negated, predicate_name, arguments = parse_symbolic_literal(literal)
    signature = _resolve_signature(
        arguments,
        parameter_roles=parameter_roles,
        argument_name_to_type=None,
        inventory_by_name=inventory_by_name,
        predicate_name=predicate_name,
    )
    normalized_arguments, normalized_signature = _normalize_binary_argument_order(
        predicate_name=predicate_name,
        arguments=arguments,
        resolved_signature=signature,
        inventory_by_name=inventory_by_name,
        type_parents=type_parents,
        type_special_supertypes=type_special_supertypes,
    )
    renamed = _rename_predicate(predicate_name, normalized_signature, rename_by_predicate)
    return format_symbolic_literal(renamed, normalized_arguments, negated=negated)


def _rewrite_fact(
    fact: str,
    *,
    parameter_roles: list[str],
    rename_by_predicate: dict[str, dict[tuple[str, ...], str]],
    inventory_by_name: dict[str, PredicateSchema],
    type_parents: dict[str, str],
    type_special_supertypes: dict[str, set[str]],
) -> str:
    predicate_name, arguments = parse_positive_symbolic_fact(fact)
    signature = _resolve_signature(
        arguments,
        parameter_roles=parameter_roles,
        argument_name_to_type=None,
        inventory_by_name=inventory_by_name,
        predicate_name=predicate_name,
    )
    normalized_arguments, normalized_signature = _normalize_binary_argument_order(
        predicate_name=predicate_name,
        arguments=arguments,
        resolved_signature=signature,
        inventory_by_name=inventory_by_name,
        type_parents=type_parents,
        type_special_supertypes=type_special_supertypes,
    )
    renamed = _rename_predicate(predicate_name, normalized_signature, rename_by_predicate)
    return format_symbolic_literal(renamed, normalized_arguments)


def _rewrite_fact_for_record(
    fact: str,
    *,
    record: ManipulationEffectRecord,
    schema: ActionSchema | None,
    rename_by_predicate: dict[str, dict[tuple[str, ...], str]],
    inventory_by_name: dict[str, PredicateSchema],
    type_parents: dict[str, str],
    type_special_supertypes: dict[str, set[str]],
) -> str:
    predicate_name, arguments = parse_positive_symbolic_fact(fact)
    parameter_roles = list(schema.parameter_roles) if schema is not None else []
    argument_type_map = {
        argument: _normalize_type_name(parameter_roles[index]) if index < len(parameter_roles) else "object"
        for index, argument in enumerate(record.action_arguments)
    }
    signature = _resolve_signature(
        arguments,
        parameter_roles=parameter_roles,
        argument_name_to_type=argument_type_map,
        inventory_by_name=inventory_by_name,
        predicate_name=predicate_name,
    )
    normalized_arguments, normalized_signature = _normalize_binary_argument_order(
        predicate_name=predicate_name,
        arguments=arguments,
        resolved_signature=signature,
        inventory_by_name=inventory_by_name,
        type_parents=type_parents,
        type_special_supertypes=type_special_supertypes,
    )
    renamed = _rename_predicate(predicate_name, normalized_signature, rename_by_predicate)
    return format_symbolic_literal(renamed, normalized_arguments)


def _normalize_binary_argument_order(
    *,
    predicate_name: str,
    arguments: list[str],
    resolved_signature: tuple[str, ...],
    inventory_by_name: dict[str, PredicateSchema],
    type_parents: dict[str, str],
    type_special_supertypes: dict[str, set[str]],
) -> tuple[list[str], tuple[str, ...]]:
    existing = inventory_by_name.get(predicate_name)
    expected_signature = tuple(
        _normalize_type_name(type_name) for type_name in (existing.parameter_types if existing is not None else [])
    )
    if len(arguments) == 2 and len(resolved_signature) == 2 and len(expected_signature) == 2:
        if _signature_compatible_with_expected(
            signature=resolved_signature,
            expected_signature=expected_signature,
            type_parents=type_parents,
            type_special_supertypes=type_special_supertypes,
        ):
            return list(arguments), resolved_signature
        reversed_signature = tuple(reversed(resolved_signature))
        if _signature_compatible_with_expected(
            signature=reversed_signature,
            expected_signature=expected_signature,
            type_parents=type_parents,
            type_special_supertypes=type_special_supertypes,
        ):
            return [arguments[1], arguments[0]], reversed_signature
    if (
        len(arguments) == 2
        and len(resolved_signature) == 2
        and len(expected_signature) == 2
        and resolved_signature != expected_signature
        and tuple(reversed(resolved_signature)) == expected_signature
    ):
        return [arguments[1], arguments[0]], expected_signature
    return list(arguments), resolved_signature


def _build_type_hierarchy_maps(
    object_types: list[ObjectTypeDefinition],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    type_parents: dict[str, str] = {}
    type_special_supertypes: dict[str, set[str]] = {}
    for item in object_types:
        type_name = _normalize_type_name(item.type_name)
        parent_type = _normalize_type_name(item.parent_type) if item.parent_type else ""
        if type_name and parent_type:
            type_parents[type_name] = parent_type
        if item.special_supertypes:
            type_special_supertypes[type_name] = {
                _normalize_type_name(value) for value in item.special_supertypes if _normalize_type_name(value)
            }
    return type_parents, type_special_supertypes


def _type_matches_expected(
    concrete_type: str,
    expected_type: str,
    *,
    type_parents: dict[str, str],
    type_special_supertypes: dict[str, set[str]],
) -> bool:
    normalized_concrete = _normalize_type_name(concrete_type)
    normalized_expected = _normalize_type_name(expected_type)
    if normalized_concrete == normalized_expected:
        return True
    if normalized_expected in type_special_supertypes.get(normalized_concrete, set()):
        return True
    current = type_parents.get(normalized_concrete)
    while current:
        if current == normalized_expected:
            return True
        if normalized_expected in type_special_supertypes.get(current, set()):
            return True
        current = type_parents.get(current)
    return False


def _signature_compatible_with_expected(
    *,
    signature: tuple[str, ...],
    expected_signature: tuple[str, ...],
    type_parents: dict[str, str],
    type_special_supertypes: dict[str, set[str]],
) -> bool:
    if len(signature) != len(expected_signature):
        return False
    return all(
        _type_matches_expected(
            concrete_type=concrete_type,
            expected_type=expected_type,
            type_parents=type_parents,
            type_special_supertypes=type_special_supertypes,
        )
        for concrete_type, expected_type in zip(signature, expected_signature, strict=True)
    )


def _rewrite_predicate_inventory(
    *,
    predicate_inventory: list[PredicateSchema],
    predicate_comments: dict[str, str],
    observed_signatures: dict[str, set[tuple[str, ...]]],
    rename_by_predicate: dict[str, dict[tuple[str, ...], str]],
    type_parents: dict[str, str],
    type_special_supertypes: dict[str, set[str]],
) -> list[PredicateSchema]:
    inventory_by_name = {item.predicate_name: item for item in predicate_inventory}
    rewritten: list[PredicateSchema] = []
    emitted_names: set[str] = set()

    for item in predicate_inventory:
        observed = sorted(observed_signatures.get(item.predicate_name, set()))
        if not observed:
            if item.predicate_name not in emitted_names:
                rewritten.append(item)
                emitted_names.add(item.predicate_name)
            continue
        expected_signature = tuple(_normalize_type_name(type_name) for type_name in item.parameter_types)
        if all(
            _signature_compatible_with_expected(
                signature=signature,
                expected_signature=expected_signature,
                type_parents=type_parents,
                type_special_supertypes=type_special_supertypes,
            )
            for signature in observed
        ):
            compiled_signature = _compile_signature_for_pddl(
                expected_signature=expected_signature,
                observed_signatures=observed,
                type_parents=type_parents,
                type_special_supertypes=type_special_supertypes,
            )
            rewritten.append(
                item
                if compiled_signature == expected_signature
                else PredicateSchema(
                    predicate_name=item.predicate_name,
                    parameter_types=list(compiled_signature),
                    comment=item.comment,
                    predicate_kind=item.predicate_kind,
                    is_static_feature=item.is_static_feature,
                )
            )
            emitted_names.add(item.predicate_name)
            continue
        if len(observed) == 1:
            signature = observed[0]
            rewritten.append(
                PredicateSchema(
                    predicate_name=item.predicate_name,
                    parameter_types=list(signature),
                    comment=item.comment,
                    predicate_kind=item.predicate_kind,
                    is_static_feature=item.is_static_feature,
                )
            )
            emitted_names.add(item.predicate_name)
            continue
        for signature in observed:
            renamed = rename_by_predicate[item.predicate_name][signature]
            if renamed in emitted_names:
                continue
            rewritten.append(
                PredicateSchema(
                    predicate_name=renamed,
                    parameter_types=list(signature),
                    comment=item.comment,
                    predicate_kind=item.predicate_kind,
                    is_static_feature=item.is_static_feature,
                )
            )
            emitted_names.add(renamed)

    for predicate_name, observed in sorted(observed_signatures.items()):
        if predicate_name in inventory_by_name:
            continue
        if len(observed) == 1:
            signature = next(iter(observed))
            rewritten.append(
                PredicateSchema(
                    predicate_name=predicate_name,
                    parameter_types=list(signature),
                    comment=predicate_comments.get(predicate_name),
                )
            )
            emitted_names.add(predicate_name)
            continue
        for signature in sorted(observed):
            renamed = rename_by_predicate[predicate_name][signature]
            if renamed in emitted_names:
                continue
            rewritten.append(
                PredicateSchema(
                    predicate_name=renamed,
                    parameter_types=list(signature),
                    comment=predicate_comments.get(predicate_name),
                )
            )
            emitted_names.add(renamed)

    return rewritten


def _compile_signature_for_pddl(
    *,
    expected_signature: tuple[str, ...],
    observed_signatures: list[tuple[str, ...]],
    type_parents: dict[str, str],
    type_special_supertypes: dict[str, set[str]],
) -> tuple[str, ...]:
    compiled: list[str] = []
    for index, expected_type in enumerate(expected_signature):
        observed_types = [signature[index] for signature in observed_signatures]
        if all(
            _type_matches_expected(
                concrete_type=concrete_type,
                expected_type=expected_type,
                type_parents=type_parents,
                type_special_supertypes={},
            )
            for concrete_type in observed_types
        ):
            compiled.append(expected_type)
            continue

        if _representable_special_supertype_parent(
            special_type=expected_type,
            type_parents=type_parents,
            type_special_supertypes=type_special_supertypes,
        ) is not None:
            compiled.append(expected_type)
            continue

        # Special supertypes model cross-cutting memberships that PDDL's
        # single-inheritance type tree cannot always declare directly. Mixed
        # movable/fixed memberships still fall back to their common ancestor.
        member_types = [
            concrete_type
            for concrete_type, memberships in type_special_supertypes.items()
            if expected_type in memberships
        ]
        compiled.append(
            _nearest_common_pddl_type(
                [*observed_types, *member_types],
                type_parents=type_parents,
            )
        )
    return tuple(compiled)


def _representable_special_supertype_parent(
    *,
    special_type: str,
    type_parents: dict[str, str],
    type_special_supertypes: dict[str, set[str]],
) -> str | None:
    member_types = [
        concrete_type
        for concrete_type, memberships in type_special_supertypes.items()
        if special_type in memberships
    ]
    if not member_types:
        return None
    common_parent = _nearest_common_pddl_type(member_types, type_parents=type_parents)
    return None if common_parent == "object" else common_parent


def _nearest_common_pddl_type(
    type_names: list[str],
    *,
    type_parents: dict[str, str],
) -> str:
    normalized = [_normalize_type_name(type_name) for type_name in type_names if _normalize_type_name(type_name)]
    if not normalized:
        return "object"

    def ancestry(type_name: str) -> list[str]:
        result = [type_name]
        current = type_name
        seen = {type_name}
        while current in type_parents:
            current = type_parents[current]
            if current in seen:
                break
            result.append(current)
            seen.add(current)
        if "object" not in seen:
            result.append("object")
        return result

    chains = [ancestry(type_name) for type_name in normalized]
    shared = set(chains[0]).intersection(*(set(chain) for chain in chains[1:]))
    return next((type_name for type_name in chains[0] if type_name in shared), "object")


def _rewrite_predicate_comments(
    *,
    predicate_inventory: list[PredicateSchema],
    predicate_comments: dict[str, str],
    rename_by_predicate: dict[str, dict[tuple[str, ...], str]],
) -> dict[str, str]:
    reverse_name_map = {
        renamed: original
        for original, signature_map in rename_by_predicate.items()
        for renamed in signature_map.values()
    }
    rewritten: dict[str, str] = {}
    for item in predicate_inventory:
        source_name = reverse_name_map.get(item.predicate_name, item.predicate_name)
        comment = str(
            predicate_comments.get(item.predicate_name) or predicate_comments.get(source_name) or item.comment or ""
        ).strip()
        if comment:
            rewritten[item.predicate_name] = comment
    return rewritten


def _rewrite_cached_grounding_artifacts(
    *,
    artifact_dir: Path,
    predicate_inventory: list[PredicateSchema],
    object_types: list[ObjectTypeDefinition],
) -> None:
    inventory_by_name = {item.predicate_name: item for item in predicate_inventory}
    global_object_type_map = _build_global_object_type_map(object_types)

    json_files = [
        artifact_dir / "episode_problem_grounding_results.json",
        artifact_dir / "episode_grounded_effect_learning_summary.json",
        artifact_dir / "post_statistics_episode_repair_summary.json",
    ]
    for path in json_files:
        if not path.exists():
            continue
        payload = load_json(path)
        rewritten = _rewrite_cached_payload(
            payload,
            inventory_by_name=inventory_by_name,
            fallback_object_type_map=global_object_type_map,
        )
        path.write_text(
            json.dumps(rewritten, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    debug_root = artifact_dir / "episode_grounded_effect_learning_debug"
    if not debug_root.exists():
        return

    for path in sorted(debug_root.rglob("problem_context.json")):
        payload = load_json(path)
        rewritten = _rewrite_cached_payload(
            payload,
            inventory_by_name=inventory_by_name,
            fallback_object_type_map=global_object_type_map,
        )
        path.write_text(
            json.dumps(rewritten, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    for path in sorted(debug_root.rglob("problem_spec.json")):
        payload = load_json(path)
        rewritten = _rewrite_cached_payload(
            payload,
            inventory_by_name=inventory_by_name,
            fallback_object_type_map=global_object_type_map,
        )
        path.write_text(
            json.dumps(rewritten, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        problem_pddl_path = path.with_name("problem.pddl")
        if problem_pddl_path.exists() and isinstance(rewritten, dict):
            problem_spec = _problem_spec_from_dict(rewritten)
            if problem_spec is not None:
                problem_pddl_path.write_text(render_problem_pddl(problem_spec), encoding="utf-8")

    for path in sorted(debug_root.rglob("validation_report.json")):
        payload = load_json(path)
        rewritten = _rewrite_cached_payload(
            payload,
            inventory_by_name=inventory_by_name,
            fallback_object_type_map=global_object_type_map,
        )
        path.write_text(
            json.dumps(rewritten, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    for path in sorted(debug_root.rglob("manipulation_records.jsonl")):
        _rewrite_jsonl_records_file(
            path,
            inventory_by_name=inventory_by_name,
            fallback_object_type_map=global_object_type_map,
        )

    for path in sorted(debug_root.rglob("grounded_steps.jsonl")):
        _rewrite_jsonl_records_file(
            path,
            inventory_by_name=inventory_by_name,
            fallback_object_type_map=global_object_type_map,
        )


def _rewrite_jsonl_records_file(
    path: Path,
    *,
    inventory_by_name: dict[str, PredicateSchema],
    fallback_object_type_map: dict[str, str],
) -> None:
    rows: list[object] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    rewritten_rows = [
        _rewrite_cached_payload(
            row,
            inventory_by_name=inventory_by_name,
            fallback_object_type_map=fallback_object_type_map,
        )
        for row in rows
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rewritten_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rewrite_cached_payload(
    payload: object,
    *,
    inventory_by_name: dict[str, PredicateSchema],
    fallback_object_type_map: dict[str, str],
    current_object_type_map: dict[str, str] | None = None,
) -> object:
    object_type_map = dict(fallback_object_type_map)
    if current_object_type_map:
        object_type_map.update(current_object_type_map)

    if isinstance(payload, list):
        return [
            _rewrite_cached_payload(
                item,
                inventory_by_name=inventory_by_name,
                fallback_object_type_map=fallback_object_type_map,
                current_object_type_map=object_type_map,
            )
            for item in payload
        ]
    if not isinstance(payload, dict):
        return payload

    rewritten = dict(payload)
    local_object_type_map = dict(object_type_map)

    if isinstance(rewritten.get("problem_spec"), dict):
        rewritten_problem_spec = _rewrite_problem_spec_dict(
            rewritten["problem_spec"],
            inventory_by_name=inventory_by_name,
            fallback_object_type_map=local_object_type_map,
        )
        rewritten["problem_spec"] = rewritten_problem_spec
        local_object_type_map.update(_object_type_map_from_problem_spec_dict(rewritten_problem_spec))
        if "problem_pddl" in rewritten:
            problem_spec = _problem_spec_from_dict(rewritten_problem_spec)
            rewritten["problem_pddl"] = render_problem_pddl(problem_spec) if problem_spec is not None else ""
    if isinstance(rewritten.get("problem_spec_without_goal"), dict):
        rewritten_problem_spec_without_goal = _rewrite_problem_spec_dict(
            rewritten["problem_spec_without_goal"],
            inventory_by_name=inventory_by_name,
            fallback_object_type_map=local_object_type_map,
        )
        rewritten["problem_spec_without_goal"] = rewritten_problem_spec_without_goal
        local_object_type_map.update(_object_type_map_from_problem_spec_dict(rewritten_problem_spec_without_goal))

    for key in (
        "init_facts",
        "goal_facts",
        "delta_add",
        "delta_del",
        "state_before",
        "state_after",
        "failed_preconditions",
    ):
        value = rewritten.get(key)
        if isinstance(value, list):
            rewritten[key] = _rewrite_fact_list(
                value,
                inventory_by_name=inventory_by_name,
                object_type_map=local_object_type_map,
            )

    for key, value in list(rewritten.items()):
        if key in {
            "problem_spec",
            "problem_spec_without_goal",
            "init_facts",
            "goal_facts",
            "delta_add",
            "delta_del",
            "state_before",
            "state_after",
            "failed_preconditions",
        }:
            continue
        rewritten[key] = _rewrite_cached_payload(
            value,
            inventory_by_name=inventory_by_name,
            fallback_object_type_map=fallback_object_type_map,
            current_object_type_map=local_object_type_map,
        )
    return rewritten


def _rewrite_problem_spec_dict(
    payload: dict[str, object],
    *,
    inventory_by_name: dict[str, PredicateSchema],
    fallback_object_type_map: dict[str, str],
) -> dict[str, object]:
    rewritten = dict(payload)
    object_type_map = dict(fallback_object_type_map)
    object_type_map.update(_object_type_map_from_problem_spec_dict(rewritten))
    for key in ("init_facts", "goal_facts"):
        value = rewritten.get(key)
        if isinstance(value, list):
            rewritten[key] = _rewrite_fact_list(
                value,
                inventory_by_name=inventory_by_name,
                object_type_map=object_type_map,
            )
    return rewritten


def _problem_spec_from_dict(payload: dict[str, object]) -> ProblemSpec | None:
    problem_name = str(payload.get("problem_name") or "").strip()
    domain_name = str(payload.get("domain_name") or "").strip()
    if not problem_name or not domain_name:
        return None
    objects: list[ObjectDeclaration] = []
    for item in payload.get("objects", []):
        if not isinstance(item, dict):
            continue
        object_name = str(item.get("name") or "").strip()
        if not object_name:
            continue
        objects.append(
            ObjectDeclaration(
                name=object_name,
                type_name=_normalize_type_name(str(item.get("type_name") or "object")),
            )
        )
    return ProblemSpec(
        problem_name=problem_name,
        domain_name=domain_name,
        objects=objects,
        init_facts=[str(item).strip() for item in payload.get("init_facts", []) if str(item).strip()],
        goal_facts=[str(item).strip() for item in payload.get("goal_facts", []) if str(item).strip()],
        canonical_object_map={
            str(key): str(value)
            for key, value in dict(payload.get("canonical_object_map", {})).items()
            if str(key).strip() and str(value).strip()
        },
    )


def _build_global_object_type_map(object_types: list[ObjectTypeDefinition]) -> dict[str, str]:
    object_type_map: dict[str, str] = {}
    for definition in object_types:
        type_name = _normalize_type_name(definition.type_name)
        for object_name in definition.member_object_names:
            cleaned = str(object_name).strip()
            if cleaned:
                object_type_map[cleaned] = type_name
    return object_type_map


def _object_type_map_from_problem_spec_dict(payload: dict[str, object]) -> dict[str, str]:
    object_type_map: dict[str, str] = {}
    for item in payload.get("objects", []):
        if not isinstance(item, dict):
            continue
        object_name = str(item.get("name") or "").strip()
        if not object_name:
            continue
        object_type_map[object_name] = _normalize_type_name(str(item.get("type_name") or "object"))
    return object_type_map


def _rewrite_fact_list(
    facts: list[object],
    *,
    inventory_by_name: dict[str, PredicateSchema],
    object_type_map: dict[str, str],
) -> list[str]:
    rewritten: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        text = str(fact).strip()
        if not text:
            continue
        rewritten_fact = _rewrite_grounded_literal(
            text,
            inventory_by_name=inventory_by_name,
            object_type_map=object_type_map,
        )
        if rewritten_fact not in seen:
            rewritten.append(rewritten_fact)
            seen.add(rewritten_fact)
    return rewritten


def _rewrite_grounded_literal(
    literal: str,
    *,
    inventory_by_name: dict[str, PredicateSchema],
    object_type_map: dict[str, str],
) -> str:
    try:
        negated, predicate_name, arguments = parse_symbolic_literal(literal)
    except ValueError:
        return literal
    signature = tuple(
        object_type_map.get(argument)
        or (
            inventory_by_name[predicate_name].parameter_types[index]
            if predicate_name in inventory_by_name and index < len(inventory_by_name[predicate_name].parameter_types)
            else "object"
        )
        for index, argument in enumerate(arguments)
    )
    rewritten_predicate = predicate_name
    split_candidate = _split_predicate_name(predicate_name, signature)
    if split_candidate in inventory_by_name:
        rewritten_predicate = split_candidate
    return format_symbolic_literal(rewritten_predicate, arguments, negated=negated)
