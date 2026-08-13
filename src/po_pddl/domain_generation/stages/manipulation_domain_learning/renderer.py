from __future__ import annotations

import ast
import logging
import textwrap
from collections import defaultdict

from po_pddl.domain_generation.infrastructure.fact_utils import (
    parse_positive_symbolic_fact as _parse_fact,
)
from po_pddl.domain_generation.infrastructure.fact_utils import (
    parse_symbolic_literal as _parse_literal,
)
from po_pddl.domain_generation.infrastructure.fact_utils import (
    remap_symbolic_literal_arguments,
    render_symbolic_literal_to_pddl,
)

from .models import (
    ActionEffectBranch,
    ActionEffectStatistic,
    ActionSchema,
    ManipulationEffectRecord,
    ObjectTypeDefinition,
    PredicateSchema,
)

logger = logging.getLogger(__name__)


_fact_to_pddl = render_symbolic_literal_to_pddl
_literal_to_pddl = render_symbolic_literal_to_pddl


def _schema_parameter_symbol(index: int) -> str:
    return f"?param_{index + 1}"


def _schema_action_argument_symbol(index: int) -> str:
    return f"?arg{index}"


def _schema_parameter_render_mapping(parameter_count: int) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for index in range(parameter_count):
        target = _schema_action_argument_symbol(index)
        mapping[f"?arg{index}"] = target
        mapping[_schema_parameter_symbol(index)] = target
    return mapping


def _parameter_index(argument: str) -> int | None:
    if argument.startswith("?arg") and argument[4:].isdigit():
        return int(argument[4:])
    if argument.startswith("?param_") and argument[7:].isdigit():
        return int(argument[7:]) - 1
    return None


def _normalize_declared_type_name(type_entry: object) -> str:
    if isinstance(type_entry, (list, tuple)):
        if len(type_entry) >= 2 and str(type_entry[1]).strip():
            return str(type_entry[1]).strip()
        if len(type_entry) >= 1 and str(type_entry[0]).strip():
            return str(type_entry[0]).strip()
        return "object"
    text = str(type_entry).strip()
    if text.startswith("(") and text.endswith(")"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return _normalize_declared_type_name(parsed)
    return text or "object"


def _infer_existential_variable_types(
    literals: list[str],
    *,
    predicate_type_lookup: dict[str, list[str]],
    action_parameter_symbols: set[str],
) -> dict[str, str]:
    inferred_types: dict[str, str] = {}
    for literal in literals:
        _negated, predicate, arguments = _parse_literal(literal)
        declared_types = predicate_type_lookup.get(predicate, [])
        for index, argument in enumerate(arguments):
            if not argument.startswith("?") or argument in action_parameter_symbols:
                continue
            candidate_type = (
                _normalize_declared_type_name(declared_types[index]) if index < len(declared_types) else "object"
            )
            existing_type = inferred_types.get(argument)
            if existing_type is None or existing_type == candidate_type:
                inferred_types[argument] = candidate_type
            elif existing_type == "object":
                inferred_types[argument] = candidate_type
            elif candidate_type != "object":
                inferred_types[argument] = "object"
    return inferred_types


def _rename_quantified_literal_variables(
    literal: str,
    *,
    variable_mapping: dict[str, str],
) -> str:
    return remap_symbolic_literal_arguments(literal, variable_mapping) if variable_mapping else literal


def _render_precondition_literals(
    literals: list[str],
    *,
    parameter_mapping: dict[str, str],
    predicate_type_lookup: dict[str, list[str]],
) -> tuple[str, bool, bool]:
    if not literals:
        return "(and)", False, False
    remapped_literals = [remap_symbolic_literal_arguments(literal, parameter_mapping) for literal in literals]
    conjuncts = [render_symbolic_literal_to_pddl(literal) for literal in remapped_literals]
    action_parameter_symbols = set(parameter_mapping.values())
    existential_types = _infer_existential_variable_types(
        remapped_literals,
        predicate_type_lookup=predicate_type_lookup,
        action_parameter_symbols=action_parameter_symbols,
    )
    if not existential_types:
        lines = ["(and"]
        lines.extend(f"  {conjunct}" for conjunct in conjuncts)
        lines.append(")")
        return "\n".join(lines), False, False
    base_literals: list[str] = []
    existential_literals: list[str] = []
    extra_variable_literals: list[tuple[str, bool, list[str]]] = []
    for literal in remapped_literals:
        negated, predicate, arguments = _parse_literal(literal)
        del predicate
        extra_variables = [
            argument for argument in arguments if argument.startswith("?") and argument not in action_parameter_symbols
        ]
        if extra_variables:
            existential_literals.append(literal)
            extra_variable_literals.append((literal, negated, extra_variables))
        else:
            base_literals.append(literal)
    rendered_base_conjuncts = [render_symbolic_literal_to_pddl(literal) for literal in base_literals]
    if extra_variable_literals and all(negated for _literal, negated, _extra_variables in extra_variable_literals):
        rendered_universal_conjuncts: list[str] = []
        uses_universal = False
        for index, literal in enumerate(existential_literals):
            negated, _predicate, arguments = _parse_literal(literal)
            del arguments
            per_literal_existential_types = _infer_existential_variable_types(
                [literal],
                predicate_type_lookup=predicate_type_lookup,
                action_parameter_symbols=action_parameter_symbols,
            )
            variable_mapping = {
                variable: f"?dep{index}_{position}"
                for position, variable in enumerate(sorted(per_literal_existential_types))
            }
            renamed_literal = _rename_quantified_literal_variables(
                literal,
                variable_mapping=variable_mapping,
            )
            negated, _predicate, _arguments = _parse_literal(renamed_literal)
            existential_parameters = " ".join(
                f"{variable_mapping[variable]} - {per_literal_existential_types[variable]}"
                for variable in sorted(per_literal_existential_types)
            )
            positive_literal = renamed_literal[4:].strip() if negated else renamed_literal
            rendered_positive_literal = render_symbolic_literal_to_pddl(positive_literal)
            rendered_universal_conjuncts.append(
                _render_quantified_precondition_block(
                    "forall",
                    existential_parameters,
                    [f"(not {rendered_positive_literal})"],
                )
            )
            uses_universal = True
        rendered_all_conjuncts = rendered_base_conjuncts + rendered_universal_conjuncts
        lines = ["(and"]
        for conjunct in rendered_all_conjuncts:
            lines.append(_indent_multiline(conjunct, "  "))
        lines.append(")")
        return "\n".join(lines), False, uses_universal
    existential_variable_mapping = {
        variable: f"?dep_exists_{index}" for index, variable in enumerate(sorted(existential_types))
    }
    existential_parameters = " ".join(
        f"{existential_variable_mapping[variable]} - {existential_types[variable]}"
        for variable in sorted(existential_types)
    )
    rendered_existential_block = _render_quantified_precondition_block(
        "exists",
        existential_parameters,
        [
            render_symbolic_literal_to_pddl(
                _rename_quantified_literal_variables(
                    literal,
                    variable_mapping=existential_variable_mapping,
                )
            )
            for literal in existential_literals
        ],
    )
    rendered_all_conjuncts = rendered_base_conjuncts + [rendered_existential_block]
    lines = ["(and"]
    for conjunct in rendered_all_conjuncts:
        lines.append(_indent_multiline(conjunct, "  "))
    lines.append(")")
    return "\n".join(lines), True, False


def _abstract_record_fact(fact: str, action_arguments: list[str]) -> str:
    predicate, arguments = _parse_fact(fact)
    argument_mapping = {argument: _schema_parameter_symbol(index) for index, argument in enumerate(action_arguments)}
    abstract_arguments = [argument_mapping.get(argument, argument) for argument in arguments]
    if abstract_arguments:
        return f"{predicate}({','.join(abstract_arguments)})"
    return f"{predicate}()"


def _abstract_record_literals(literals: list[str], action_arguments: list[str]) -> list[str]:
    return sorted({_abstract_record_fact(literal, action_arguments) for literal in literals})


def _statistical_effect_bucket_name(action_name: str, success: bool, variant_rank: int) -> str:
    outcome = "success" if success else "failure"
    return f"{action_name}_{outcome}_bucket_{variant_rank}"


def _outcome_effect_bucket_name(action_name: str, success: bool) -> str:
    outcome = "success" if success else "failure"
    return f"{action_name}_{outcome}"


def _remap_stat_literal_to_action_args(literal: str, parameter_count: int) -> str:
    mapping = {
        _schema_parameter_symbol(index): _schema_action_argument_symbol(index) for index in range(parameter_count)
    }
    return remap_symbolic_literal_arguments(literal, mapping) if mapping else literal


def _render_comment_lines(text: str | None, *, indent: str = "  ", prefix: str = ";; ") -> list[str]:
    if text is None:
        return []
    cleaned = " ".join(part.strip() for part in str(text).splitlines() if part.strip()).strip()
    if not cleaned:
        return []
    wrapped = textwrap.wrap(cleaned, width=max(20, 100 - len(indent) - len(prefix)))
    return [f"{indent}{prefix}{line}" for line in wrapped]


def _render_predicate_comment(
    predicate: str,
    predicate_comments: dict[str, str] | None,
    *,
    indent: str = "    ",
) -> list[str]:
    if not predicate_comments:
        return []
    comment = str(predicate_comments.get(predicate) or "").strip()
    if not comment:
        return []
    return [f"{indent};; {comment}"]


def _indent_multiline(text: str, indent: str) -> str:
    return "\n".join(f"{indent}{line}" if line else indent.rstrip() for line in text.splitlines())


def _render_requirements_block(requirements: list[str]) -> list[str]:
    lines = ["  (:requirements"]
    lines.extend(f"    {requirement}" for requirement in requirements)
    lines.append("  )")
    return lines


def _render_quantified_precondition_block(
    quantifier: str,
    parameters: str,
    body_literals: list[str],
) -> str:
    lines = [f"({quantifier} ({parameters})", "  (and"]
    lines.extend(f"    {literal}" for literal in body_literals)
    lines.append("  )")
    lines.append(")")
    return "\n".join(lines)


def _merge_type(existing: str | None, candidate: str | None) -> str:
    if candidate is None or not candidate:
        return existing or "object"
    if existing in (None, "", "object"):
        return candidate
    if candidate == existing:
        return existing
    return "object"


def _normalize_role_type(role: str | None) -> str:
    if role is None:
        return "object"
    cleaned = role.strip()
    return cleaned if cleaned else "object"


def _infer_predicate_argument_types(
    action_schemas: list[ActionSchema],
    records: list[ManipulationEffectRecord],
) -> dict[str, list[str]]:
    predicate_types: dict[str, list[str]] = {}
    schema_by_action = {schema.canonical_action_name: schema for schema in action_schemas}

    def ensure(predicate: str, arity: int) -> list[str]:
        current = predicate_types.setdefault(predicate, ["object"] * arity)
        if len(current) < arity:
            current.extend(["object"] * (arity - len(current)))
        return current

    for schema in action_schemas:
        role_types = [_normalize_role_type(role) for role in schema.parameter_roles]
        for literal in schema.precondition_literals:
            _negated, predicate, arguments = _parse_literal(literal)
            current = ensure(predicate, len(arguments))
            for index, argument in enumerate(arguments):
                role_index = _parameter_index(argument)
                if role_index is not None:
                    candidate = role_types[role_index] if role_index < len(role_types) else "object"
                    current[index] = _merge_type(current[index], candidate)
        for branch in schema.effect_branches:
            for literal in branch.delta_add + branch.delta_del:
                predicate, arguments = _parse_fact(literal)
                current = ensure(predicate, len(arguments))
                for index, argument in enumerate(arguments):
                    role_index = _parameter_index(argument)
                    if role_index is not None:
                        candidate = role_types[role_index] if role_index < len(role_types) else "object"
                        current[index] = _merge_type(current[index], candidate)

    for record in records:
        schema = schema_by_action.get(record.canonical_action_name)
        role_types = [_normalize_role_type(role) for role in (schema.parameter_roles if schema else [])]
        argument_type_map = {
            argument: role_types[index] if index < len(role_types) else "object"
            for index, argument in enumerate(record.action_arguments)
        }
        for fact in record.delta_add + record.delta_del:
            predicate, arguments = _parse_fact(fact)
            current = ensure(predicate, len(arguments))
            for index, argument in enumerate(arguments):
                current[index] = _merge_type(current[index], argument_type_map.get(argument, "object"))

    return predicate_types


def _render_types_block(
    action_schemas: list[ActionSchema],
    *,
    predicate_inventory: list[PredicateSchema] | None = None,
    object_types: list[ObjectTypeDefinition] | None = None,
) -> list[str]:
    special_type_order = ["movable_item", "fixed_item", "containable_item"]
    special_type_set = set(special_type_order)

    parent_by_type = {
        _normalize_role_type(item.type_name): _normalize_role_type(item.parent_type)
        for item in (object_types or [])
        if _normalize_role_type(item.type_name) != "object" and item.parent_type
    }
    special_members_by_type: dict[str, list[str]] = defaultdict(list)
    for item in object_types or []:
        concrete_type = _normalize_role_type(item.type_name)
        for special_type in item.special_supertypes:
            normalized_special = _normalize_role_type(special_type)
            if normalized_special == "containable_item":
                special_members_by_type[normalized_special].append(concrete_type)

    special_parent_by_type: dict[str, str] = {}
    for special_type, member_types in special_members_by_type.items():
        member_parents = {parent_by_type.get(member_type, "object") for member_type in member_types}
        if len(member_parents) != 1:
            continue
        common_parent = next(iter(member_parents))
        if common_parent == "object" or common_parent == special_type:
            continue
        special_parent_by_type[special_type] = common_parent
        for member_type in member_types:
            parent_by_type[member_type] = special_type

    discovered_concrete_types: set[str] = {
        _normalize_role_type(item.type_name)
        for item in (object_types or [])
        if _normalize_role_type(item.type_name) not in {"object", *special_type_set}
    }
    discovered_concrete_types.update(
        _normalize_role_type(role)
        for schema in action_schemas
        for role in schema.parameter_roles
        if _normalize_role_type(role) not in {"object", *special_type_set}
    )
    discovered_concrete_types.update(
        _normalize_role_type(type_name)
        for predicate in (predicate_inventory or [])
        for type_name in predicate.parameter_types
        if _normalize_role_type(type_name) not in {"object", *special_type_set}
    )

    used_special_types: list[str] = [
        type_name
        for type_name in special_type_order
        if any(parent == type_name for parent in parent_by_type.values())
        or type_name in special_parent_by_type.values()
        or any(_normalize_role_type(role) == type_name for schema in action_schemas for role in schema.parameter_roles)
        or any(
            _normalize_role_type(param_type) == type_name
            for predicate in (predicate_inventory or [])
            for param_type in predicate.parameter_types
        )
    ]

    grouped_types: dict[str, list[str]] = defaultdict(list)
    for type_name in sorted(discovered_concrete_types):
        parent_name = parent_by_type.get(type_name, "object")
        grouped_types[parent_name].append(type_name)

    if not used_special_types and not grouped_types.get("object"):
        return []

    lines = ["  (:types"]
    root_special_types = [
        type_name for type_name in used_special_types if type_name not in special_parent_by_type
    ]
    if root_special_types:
        lines.append(f"    {' '.join(root_special_types)} - object")
    for parent_name in special_type_order:
        members = [
            type_name
            for type_name in used_special_types
            if special_parent_by_type.get(type_name) == parent_name
        ]
        if members:
            lines.append(f"    {' '.join(members)} - {parent_name}")
    for parent_name in special_type_order + ["object"]:
        members = grouped_types.get(parent_name, [])
        if members:
            lines.append(f"    {' '.join(sorted(members))} - {parent_name}")
    lines.append("  )")
    return lines


def classify_records_by_effect_statistics(
    records: list[ManipulationEffectRecord],
) -> list[ManipulationEffectRecord]:
    if not records:
        return []

    records_with_index = list(enumerate(records))
    grouped_by_action: dict[str, list[tuple[int, ManipulationEffectRecord]]] = defaultdict(list)
    for index, record in records_with_index:
        grouped_by_action[record.canonical_action_name].append((index, record))

    rewritten_by_index: dict[int, ManipulationEffectRecord] = {}
    for action_name, action_rows in grouped_by_action.items():
        grouped_by_outcome: dict[bool, list[tuple[int, ManipulationEffectRecord]]] = defaultdict(list)
        for index, record in action_rows:
            grouped_by_outcome[record.success].append((index, record))

        for success_value, outcome_rows in grouped_by_outcome.items():
            abstracted_rows: list[tuple[int, ManipulationEffectRecord, tuple[str, ...], tuple[str, ...]]] = []
            for index, record in outcome_rows:
                abstracted_rows.append(
                    (
                        index,
                        record,
                        tuple(_abstract_record_literals(record.delta_add, record.action_arguments)),
                        tuple(_abstract_record_literals(record.delta_del, record.action_arguments)),
                    )
                )
            fixed_delta_add = set(abstracted_rows[0][2]) if abstracted_rows else set()
            fixed_delta_del = set(abstracted_rows[0][3]) if abstracted_rows else set()
            for _index, _record, delta_add, delta_del in abstracted_rows[1:]:
                fixed_delta_add.intersection_update(delta_add)
                fixed_delta_del.intersection_update(delta_del)

            residual_groups: dict[
                tuple[tuple[str, ...], tuple[str, ...]], list[tuple[int, ManipulationEffectRecord]]
            ] = defaultdict(list)
            for index, record, delta_add, delta_del in abstracted_rows:
                residual_signature = (
                    tuple(sorted(set(delta_add) - fixed_delta_add)),
                    tuple(sorted(set(delta_del) - fixed_delta_del)),
                )
                residual_groups[residual_signature].append((index, record))

            ordered_groups = sorted(
                residual_groups.items(),
                key=lambda item: (
                    -len(item[1]),
                    item[0][0],
                    item[0][1],
                ),
            )
            use_ranked_bucket_names = len(ordered_groups) > 1
            for variant_rank, (_signature, grouped_rows) in enumerate(ordered_groups, start=1):
                bucket_name = (
                    _statistical_effect_bucket_name(action_name, success_value, variant_rank)
                    if use_ranked_bucket_names
                    else _outcome_effect_bucket_name(action_name, success_value)
                )
                for index, record in grouped_rows:
                    rewritten_by_index[index] = ManipulationEffectRecord(
                        episode_name=record.episode_name,
                        step_index=record.step_index,
                        raw_action_text=record.raw_action_text,
                        canonical_action_name=record.canonical_action_name,
                        action_arguments=list(record.action_arguments),
                        pre_observation_text=record.pre_observation_text,
                        post_observation_text=record.post_observation_text,
                        extra_info=record.extra_info,
                        delta_add=list(record.delta_add),
                        delta_del=list(record.delta_del),
                        effect_bucket=bucket_name,
                        success=record.success,
                        raw_llm_output=record.raw_llm_output,
                        execution_time_sec=record.execution_time_sec,
                    )

    return [rewritten_by_index.get(index, record) for index, record in records_with_index]


def collect_action_effect_statistics(
    records: list[ManipulationEffectRecord],
) -> dict[str, list[ActionEffectStatistic]]:
    if not records:
        return {}

    bucketized_records = classify_records_by_effect_statistics(records)
    grouped_by_action: dict[str, list[ManipulationEffectRecord]] = defaultdict(list)
    for record in bucketized_records:
        grouped_by_action[record.canonical_action_name].append(record)

    statistics: dict[str, list[ActionEffectStatistic]] = {}
    for action_name, action_records in grouped_by_action.items():
        total = len(action_records)
        grouped_by_outcome: dict[bool, list[ManipulationEffectRecord]] = defaultdict(list)
        for record in action_records:
            grouped_by_outcome[record.success].append(record)

        action_stats: list[ActionEffectStatistic] = []
        for success_value, outcome_records in sorted(grouped_by_outcome.items(), key=lambda item: (item[0] is False,)):
            abstracted_by_bucket: dict[str, list[tuple[tuple[str, ...], tuple[str, ...]]]] = defaultdict(list)
            for record in outcome_records:
                abstracted_by_bucket[record.effect_bucket].append(
                    (
                        tuple(_abstract_record_literals(record.delta_add, record.action_arguments)),
                        tuple(_abstract_record_literals(record.delta_del, record.action_arguments)),
                    )
                )
            fixed_delta_add = set(next(iter(abstracted_by_bucket.values()))[0][0]) if abstracted_by_bucket else set()
            fixed_delta_del = set(next(iter(abstracted_by_bucket.values()))[0][1]) if abstracted_by_bucket else set()
            all_abstracted_rows = [item for bucket_rows in abstracted_by_bucket.values() for item in bucket_rows]
            if all_abstracted_rows:
                fixed_delta_add = set(all_abstracted_rows[0][0])
                fixed_delta_del = set(all_abstracted_rows[0][1])
                for delta_add, delta_del in all_abstracted_rows[1:]:
                    fixed_delta_add.intersection_update(delta_add)
                    fixed_delta_del.intersection_update(delta_del)

            bucket_groups: dict[str, list[ManipulationEffectRecord]] = defaultdict(list)
            for record in outcome_records:
                bucket_groups[record.effect_bucket].append(record)
            ordered_bucket_names = sorted(
                bucket_groups,
                key=lambda bucket_name: (
                    -len(bucket_groups[bucket_name]),
                    bucket_name,
                ),
            )
            for bucket_name in ordered_bucket_names:
                bucket_records = bucket_groups[bucket_name]
                exemplar = bucket_records[0]
                full_delta_add = tuple(_abstract_record_literals(exemplar.delta_add, exemplar.action_arguments))
                full_delta_del = tuple(_abstract_record_literals(exemplar.delta_del, exemplar.action_arguments))
                residual_delta_add = tuple(sorted(set(full_delta_add) - fixed_delta_add))
                residual_delta_del = tuple(sorted(set(full_delta_del) - fixed_delta_del))
                variant_rank = None
                suffix = bucket_name.rsplit("_bucket_", 1)
                if len(suffix) == 2 and suffix[1].isdigit():
                    variant_rank = int(suffix[1])
                action_stats.append(
                    ActionEffectStatistic(
                        canonical_action_name=action_name,
                        effect_bucket=bucket_name,
                        count=len(bucket_records),
                        probability=len(bucket_records) / total if total else 0.0,
                        delta_add=list(full_delta_add),
                        delta_del=list(full_delta_del),
                        success=success_value,
                        variant_rank=variant_rank,
                        fixed_delta_add=sorted(fixed_delta_add),
                        fixed_delta_del=sorted(fixed_delta_del),
                        residual_delta_add=list(residual_delta_add),
                        residual_delta_del=list(residual_delta_del),
                    )
                )
        statistics[action_name] = action_stats
    return statistics


def attach_action_effects_to_schemas(
    action_schemas: list[ActionSchema],
    records: list[ManipulationEffectRecord],
    statistics: dict[str, list[ActionEffectStatistic]],
) -> list[ActionSchema]:
    def _abstract_schema_literal(
        literal: str,
        argument_mapping: dict[str, str],
        *,
        action_name: str,
        effect_bucket: str,
    ) -> str | None:
        remapped = remap_symbolic_literal_arguments(literal, argument_mapping) if argument_mapping else literal
        _negated, _predicate, arguments = _parse_literal(remapped)
        if any(not argument.startswith("?") for argument in arguments):
            logger.warning(
                "Dropping non-abstract effect literal %r for action %s bucket %s because it still contains concrete arguments.",
                remapped,
                action_name,
                effect_bucket,
            )
            return None
        return remapped

    records_by_bucket: dict[tuple[str, str], list[ManipulationEffectRecord]] = defaultdict(list)
    for record in records:
        records_by_bucket[(record.canonical_action_name, record.effect_bucket)].append(record)

    enriched_schemas: list[ActionSchema] = []
    for schema in action_schemas:
        effect_branches: list[ActionEffectBranch] = []
        for stat in statistics.get(schema.canonical_action_name, []):
            matching_records = records_by_bucket.get((schema.canonical_action_name, stat.effect_bucket), [])
            exemplar = matching_records[0] if matching_records else None
            argument_mapping = _schema_parameter_render_mapping(schema.parameter_count)
            if exemplar is not None:
                argument_mapping.update(
                    {
                        argument: _schema_action_argument_symbol(index)
                        for index, argument in enumerate(exemplar.action_arguments)
                    }
                )
            delta_add = [
                candidate
                for candidate in (
                    _abstract_schema_literal(
                        fact,
                        argument_mapping,
                        action_name=schema.canonical_action_name,
                        effect_bucket=stat.effect_bucket,
                    )
                    for fact in list(getattr(stat, "delta_add", []))
                )
                if candidate is not None
            ]
            delta_del = [
                candidate
                for candidate in (
                    _abstract_schema_literal(
                        fact,
                        argument_mapping,
                        action_name=schema.canonical_action_name,
                        effect_bucket=stat.effect_bucket,
                    )
                    for fact in list(getattr(stat, "delta_del", []))
                )
                if candidate is not None
            ]
            fixed_delta_add = [
                candidate
                for candidate in (
                    _abstract_schema_literal(
                        fact,
                        argument_mapping,
                        action_name=schema.canonical_action_name,
                        effect_bucket=stat.effect_bucket,
                    )
                    for fact in list(getattr(stat, "fixed_delta_add", []))
                )
                if candidate is not None
            ]
            fixed_delta_del = [
                candidate
                for candidate in (
                    _abstract_schema_literal(
                        fact,
                        argument_mapping,
                        action_name=schema.canonical_action_name,
                        effect_bucket=stat.effect_bucket,
                    )
                    for fact in list(getattr(stat, "fixed_delta_del", []))
                )
                if candidate is not None
            ]
            residual_delta_add = [
                candidate
                for candidate in (
                    _abstract_schema_literal(
                        fact,
                        argument_mapping,
                        action_name=schema.canonical_action_name,
                        effect_bucket=stat.effect_bucket,
                    )
                    for fact in list(getattr(stat, "residual_delta_add", []))
                )
                if candidate is not None
            ]
            residual_delta_del = [
                candidate
                for candidate in (
                    _abstract_schema_literal(
                        fact,
                        argument_mapping,
                        action_name=schema.canonical_action_name,
                        effect_bucket=stat.effect_bucket,
                    )
                    for fact in list(getattr(stat, "residual_delta_del", []))
                )
                if candidate is not None
            ]
            success = getattr(stat, "success", None)
            if success is None:
                success = exemplar.success if exemplar is not None else stat.effect_bucket.endswith("_success")
            variant_rank = getattr(stat, "variant_rank", None)
            if variant_rank is None:
                suffix = str(stat.effect_bucket).rsplit("_bucket_", 1)
                if len(suffix) == 2 and suffix[1].isdigit():
                    variant_rank = int(suffix[1])
            effect_branches.append(
                ActionEffectBranch(
                    effect_bucket=stat.effect_bucket,
                    probability=stat.probability,
                    success=bool(success),
                    delta_add=delta_add,
                    delta_del=delta_del,
                    variant_rank=variant_rank,
                    fixed_delta_add=fixed_delta_add,
                    fixed_delta_del=fixed_delta_del,
                    residual_delta_add=residual_delta_add,
                    residual_delta_del=residual_delta_del,
                )
            )
        enriched_schemas.append(
            ActionSchema(
                canonical_action_name=schema.canonical_action_name,
                action_category=schema.action_category,
                parameter_count=schema.parameter_count,
                parameter_roles=list(schema.parameter_roles),
                precondition_literals=list(schema.precondition_literals),
                schema_description=schema.schema_description,
                effect_branches=effect_branches,
            )
        )
    return enriched_schemas


def render_manipulation_domain_fragment(
    action_schemas: list[ActionSchema],
    records: list[ManipulationEffectRecord],
    statistics: dict[str, list[ActionEffectStatistic]],
    *,
    predicate_inventory: list[PredicateSchema] | None = None,
    predicate_comments: dict[str, str] | None = None,
    object_types: list[ObjectTypeDefinition] | None = None,
) -> str:
    predicate_signatures: dict[str, int] = {}
    predicate_argument_types = _infer_predicate_argument_types(action_schemas, records)
    uses_negative_preconditions = False
    rendered_preconditions_by_action: dict[str, tuple[str, bool, bool]] = {}

    for record in records:
        for fact in record.delta_add + record.delta_del:
            predicate, arguments = _parse_fact(fact)
            predicate_signatures[predicate] = max(predicate_signatures.get(predicate, 0), len(arguments))
    for schema in action_schemas:
        for literal in schema.precondition_literals:
            negated, predicate, arguments = _parse_literal(literal)
            predicate_signatures[predicate] = max(predicate_signatures.get(predicate, 0), len(arguments))
            uses_negative_preconditions = uses_negative_preconditions or negated
        parameter_mapping = _schema_parameter_render_mapping(schema.parameter_count)
        rendered_preconditions_by_action[schema.canonical_action_name] = _render_precondition_literals(
            schema.precondition_literals,
            parameter_mapping=parameter_mapping,
            predicate_type_lookup=predicate_argument_types,
        )
    uses_existential_preconditions = any(
        uses_existential
        for _precondition_text, uses_existential, _uses_universal in rendered_preconditions_by_action.values()
    )
    uses_universal_preconditions = any(
        uses_universal
        for _precondition_text, _uses_existential, uses_universal in rendered_preconditions_by_action.values()
    )

    requirements = [":strips", ":typing", ":probabilistic-effects"]
    if uses_negative_preconditions:
        requirements.append(":negative-preconditions")
    if uses_existential_preconditions:
        requirements.append(":existential-preconditions")
    if uses_universal_preconditions:
        requirements.append(":universal-preconditions")
    lines = ["(define (domain learned_manipulation_fragment)"]
    lines.extend(_render_requirements_block(requirements))
    lines.extend(
        _render_types_block(action_schemas, predicate_inventory=predicate_inventory, object_types=object_types)
    )
    if len(lines) > 2:
        lines.append("")
    inventory_by_name = {item.predicate_name: item for item in predicate_inventory or []}
    all_predicates = sorted(set(predicate_signatures) | set(inventory_by_name))
    lines.append("  (:predicates")
    for predicate in all_predicates:
        arity = predicate_signatures.get(
            predicate, len(inventory_by_name[predicate].parameter_types) if predicate in inventory_by_name else 0
        )
        lines.extend(_render_predicate_comment(predicate, predicate_comments))
        if predicate in inventory_by_name:
            declared_types = list(inventory_by_name[predicate].parameter_types)
        else:
            declared_types = predicate_argument_types.get(predicate, ["object"] * arity)
        if arity == 0:
            lines.append(f"    ({predicate})")
        else:
            params = " ".join(
                f"?x{i} - {declared_types[i] if i < len(declared_types) else 'object'}" for i in range(arity)
            )
            lines.append(f"    ({predicate} {params})")
    lines.append("  )")
    lines.append("")
    lines.append("  ;; Manipulation actions learned from trajectories.")

    grouped_by_action: dict[str, list[ManipulationEffectRecord]] = defaultdict(list)
    for record in records:
        grouped_by_action[record.canonical_action_name].append(record)

    manipulation_schemas = sorted(
        [schema for schema in action_schemas if schema.action_category == "manipulation"],
        key=lambda schema: schema.canonical_action_name,
    )
    for schema in manipulation_schemas:
        action_name = schema.canonical_action_name
        parameter_types = [
            _normalize_role_type(schema.parameter_roles[i]) if i < len(schema.parameter_roles) else "object"
            for i in range(schema.parameter_count)
        ]
        parameter_mapping = _schema_parameter_render_mapping(schema.parameter_count)
        parameters = [
            f"{_schema_action_argument_symbol(i)} - {parameter_types[i]}" for i in range(schema.parameter_count)
        ]
        lines.extend(_render_comment_lines(schema.schema_description))
        lines.append(f"  (:action {action_name}")
        lines.append(f"    :parameters ({' '.join(parameters)})")
        if schema.precondition_literals:
            precondition_text, _action_uses_existential, _action_uses_universal = rendered_preconditions_by_action[
                action_name
            ]
            lines.append("    :precondition")
            lines.append(_indent_multiline(precondition_text, "      "))
        else:
            lines.append("    :precondition (and)")
        lines.append("    :effect")
        lines.append("      (probabilistic")
        action_stats = statistics.get(action_name, [])
        if not action_stats:
            lines.append("        1.000000 (and)")
        else:
            for stat in action_stats:
                if stat.fixed_delta_add or stat.fixed_delta_del:
                    fixed_comment_add = [
                        _remap_stat_literal_to_action_args(fact, schema.parameter_count)
                        for fact in list(stat.fixed_delta_add)
                    ]
                    fixed_comment_del = [
                        _remap_stat_literal_to_action_args(fact, schema.parameter_count)
                        for fact in list(stat.fixed_delta_del)
                    ]
                    lines.append(f"        ; fixed: add={fixed_comment_add}, del={fixed_comment_del}")
                lines.append(
                    f"        ; bucket: {stat.effect_bucket}, success: {'true' if stat.success else 'false'}, "
                    f"variant_rank: {stat.variant_rank if stat.variant_rank is not None else 0}"
                )
                if stat.residual_delta_add or stat.residual_delta_del:
                    residual_comment_add = [
                        _remap_stat_literal_to_action_args(fact, schema.parameter_count)
                        for fact in list(stat.residual_delta_add)
                    ]
                    residual_comment_del = [
                        _remap_stat_literal_to_action_args(fact, schema.parameter_count)
                        for fact in list(stat.residual_delta_del)
                    ]
                    lines.append(f"        ; residual: add={residual_comment_add}, del={residual_comment_del}")
                conjuncts = [
                    _fact_to_pddl(_remap_stat_literal_to_action_args(fact, schema.parameter_count))
                    for fact in stat.delta_add
                ]
                conjuncts.extend(
                    f"(not {_fact_to_pddl(_remap_stat_literal_to_action_args(fact, schema.parameter_count))})"
                    for fact in stat.delta_del
                )
                extra_conjuncts = [
                    str(item).strip() for item in getattr(stat, "extra_pddl_effect_conjuncts", []) if str(item).strip()
                ]
                conjuncts.extend(extra_conjuncts)
                if conjuncts:
                    effect_body = f"(and {' '.join(conjuncts)})"
                else:
                    effect_body = "(and)"
                lines.append(f"        {stat.probability:.6f} {effect_body}")
        lines.append("      )")
        lines.append("  )")
        lines.append("")

    lines.append("  ;; Observation-action learning is not implemented yet.")
    lines.append(")")
    return "\n".join(lines) + "\n"


def render_action_schema_fragment(
    action_schemas: list[ActionSchema],
    *,
    predicate_inventory: list[PredicateSchema] | None = None,
    predicate_comments: dict[str, str] | None = None,
    object_types: list[ObjectTypeDefinition] | None = None,
) -> str:
    predicate_signatures: dict[str, int] = {}
    predicate_argument_types = _infer_predicate_argument_types(action_schemas, [])
    uses_negative_preconditions = False
    rendered_preconditions_by_action: dict[str, tuple[str, bool, bool]] = {}
    uses_probabilistic_effects = any(schema.effect_branches for schema in action_schemas)
    for schema in action_schemas:
        for literal in schema.precondition_literals:
            negated, predicate, arguments = _parse_literal(literal)
            predicate_signatures[predicate] = max(predicate_signatures.get(predicate, 0), len(arguments))
            uses_negative_preconditions = uses_negative_preconditions or negated
        parameter_mapping = _schema_parameter_render_mapping(schema.parameter_count)
        rendered_preconditions_by_action[schema.canonical_action_name] = _render_precondition_literals(
            schema.precondition_literals,
            parameter_mapping=parameter_mapping,
            predicate_type_lookup=predicate_argument_types,
        )
        for branch in schema.effect_branches:
            for literal in branch.delta_add + branch.delta_del:
                predicate, arguments = _parse_fact(literal)
                predicate_signatures[predicate] = max(predicate_signatures.get(predicate, 0), len(arguments))
    uses_existential_preconditions = any(
        uses_existential
        for _precondition_text, uses_existential, _uses_universal in rendered_preconditions_by_action.values()
    )
    uses_universal_preconditions = any(
        uses_universal
        for _precondition_text, _uses_existential, uses_universal in rendered_preconditions_by_action.values()
    )

    requirements = [":strips", ":typing"]
    if uses_probabilistic_effects:
        requirements.append(":probabilistic-effects")
    if uses_negative_preconditions:
        requirements.append(":negative-preconditions")
    if uses_existential_preconditions:
        requirements.append(":existential-preconditions")
    if uses_universal_preconditions:
        requirements.append(":universal-preconditions")
    lines = ["(define (domain learned_action_schema_fragment)"]
    lines.extend(_render_requirements_block(requirements))
    lines.extend(
        _render_types_block(action_schemas, predicate_inventory=predicate_inventory, object_types=object_types)
    )
    if len(lines) > 2:
        lines.append("")
    inventory_by_name = {item.predicate_name: item for item in predicate_inventory or []}
    all_predicates = sorted(set(predicate_signatures) | set(inventory_by_name))
    lines.append("  (:predicates")
    for predicate in all_predicates:
        arity = predicate_signatures.get(
            predicate, len(inventory_by_name[predicate].parameter_types) if predicate in inventory_by_name else 0
        )
        lines.extend(_render_predicate_comment(predicate, predicate_comments))
        if predicate in inventory_by_name:
            declared_types = list(inventory_by_name[predicate].parameter_types)
        else:
            declared_types = predicate_argument_types.get(predicate, ["object"] * arity)
        if arity == 0:
            lines.append(f"    ({predicate})")
        else:
            params = " ".join(
                f"?x{i} - {declared_types[i] if i < len(declared_types) else 'object'}" for i in range(arity)
            )
            lines.append(f"    ({predicate} {params})")
    lines.append("  )")
    lines.append("")
    lines.append("  ;; Action schemas learned during consolidation.")

    for schema in sorted(action_schemas, key=lambda item: item.canonical_action_name):
        parameter_types = [
            _normalize_role_type(schema.parameter_roles[i]) if i < len(schema.parameter_roles) else "object"
            for i in range(schema.parameter_count)
        ]
        parameter_mapping = _schema_parameter_render_mapping(schema.parameter_count)
        parameters = [
            f"{_schema_action_argument_symbol(i)} - {parameter_types[i]}" for i in range(schema.parameter_count)
        ]
        lines.extend(_render_comment_lines(schema.schema_description))
        lines.append(f"  (:action {schema.canonical_action_name}")
        lines.append(f"    :parameters ({' '.join(parameters)})")
        if schema.precondition_literals:
            precondition_text, _action_uses_existential, _action_uses_universal = rendered_preconditions_by_action[
                schema.canonical_action_name
            ]
            lines.append("    :precondition")
            lines.append(_indent_multiline(precondition_text, "      "))
        else:
            lines.append("    :precondition (and)")
        if schema.effect_branches:
            lines.append("    :effect")
            lines.append("      (probabilistic")
            for branch in schema.effect_branches:
                if branch.fixed_delta_add or branch.fixed_delta_del:
                    lines.append(
                        f"        ; fixed: add={list(branch.fixed_delta_add)}, del={list(branch.fixed_delta_del)}"
                    )
                lines.append(
                    f"        ; bucket: {branch.effect_bucket}, success: {'true' if branch.success else 'false'}, "
                    f"variant_rank: {branch.variant_rank if branch.variant_rank is not None else 0}"
                )
                if branch.residual_delta_add or branch.residual_delta_del:
                    lines.append(
                        "        ; residual: "
                        f"add={list(branch.residual_delta_add)}, del={list(branch.residual_delta_del)}"
                    )
                conjuncts = [_fact_to_pddl(fact) for fact in branch.delta_add]
                conjuncts.extend(f"(not {_fact_to_pddl(fact)})" for fact in branch.delta_del)
                conjuncts.extend(str(item).strip() for item in branch.extra_pddl_effect_conjuncts if str(item).strip())
                effect_body = f"(and {' '.join(conjuncts)})" if conjuncts else "(and)"
                lines.append(f"        {branch.probability:.6f} {effect_body}")
            lines.append("      )")
        else:
            lines.append("    :effect (and)")
        lines.append("  )")
        lines.append("")

    lines.append(")")
    return "\n".join(lines) + "\n"
