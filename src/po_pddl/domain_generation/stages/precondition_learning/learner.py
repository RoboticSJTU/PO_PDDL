from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

from po_pddl.core.parser import parse_domain
from po_pddl.domain_generation.infrastructure.artifact_io import load_json, load_json_object, load_jsonl
from po_pddl.domain_generation.infrastructure.fact_utils import parse_symbolic_literal
from po_pddl.domain_generation.stages.domain_comments import (
    extract_action_comments,
    extract_predicate_comments,
)
from po_pddl.domain_generation.stages.domain_patching import (
    DomainActionPatch,
    apply_domain_repair_patch,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.grounding_update import (
    load_action_schemas,
    load_object_types,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.models import ActionSchema, PredicateSchema
from po_pddl.domain_generation.stages.manipulation_domain_learning.renderer import render_action_schema_fragment

from .models import (
    ActionPreconditionCandidateBundle,
    GroundedPreconditionExample,
    LearnedPreconditionSummary,
    PreconditionCandidateStat,
)
from .modules import (
    LLMPreconditionSelectionModule,
    prune_directional_variant_contradictions,
)


def _literal_uses_dependency_variable(literal: str) -> bool:
    _negated, _predicate, arguments = parse_symbolic_literal(literal)
    return any(argument.startswith("?dep") for argument in arguments)


def _literal_predicate_name(literal: str) -> str:
    _negated, predicate, _arguments = parse_symbolic_literal(literal)
    return predicate


def _dependency_literal_has_prior_deletion(
    *,
    literal: str,
    ground_arguments: list[str],
    step_index: int,
    validation_rows: dict[int, dict],
    parsed_domain,
    object_type_by_name: dict[str, str],
) -> bool:
    negated, predicate, abstract_arguments = parse_symbolic_literal(literal)
    if not negated or not any(argument.startswith("?dep") for argument in abstract_arguments):
        return False

    for prior_step_index, prior_row in validation_rows.items():
        if prior_step_index >= step_index:
            continue
        state_before = set(
            _normalize_state_facts(
                parsed_domain=parsed_domain,
                state_before=prior_row.get("state_before", []),
                object_type_by_name=object_type_by_name,
            )
        )
        state_after = set(
            _normalize_state_facts(
                parsed_domain=parsed_domain,
                state_before=prior_row.get("state_after", []),
                object_type_by_name=object_type_by_name,
            )
        )
        for deleted_fact in state_before - state_after:
            fact_negated, fact_predicate, fact_arguments = parse_symbolic_literal(deleted_fact)
            if fact_negated or fact_predicate != predicate or len(fact_arguments) != len(abstract_arguments):
                continue
            matched = True
            for abstract_argument, fact_argument in zip(abstract_arguments, fact_arguments):
                if abstract_argument.startswith("?arg"):
                    try:
                        argument_index = int(abstract_argument.removeprefix("?arg"))
                    except ValueError:
                        matched = False
                        break
                    if argument_index >= len(ground_arguments) or ground_arguments[argument_index] != fact_argument:
                        matched = False
                        break
                elif abstract_argument.startswith("?dep"):
                    if fact_argument in ground_arguments:
                        matched = False
                        break
                elif abstract_argument != fact_argument:
                    matched = False
                    break
            if matched:
                return True
    return False


def _load_predicate_comments(artifact_path: Path) -> dict[str, str]:
    predicate_comments_path = artifact_path / "predicate_comments.json"
    if not predicate_comments_path.exists():
        return {}
    loaded_comments = load_json(predicate_comments_path)
    if not isinstance(loaded_comments, dict):
        return {}
    return {str(key): str(value) for key, value in loaded_comments.items() if str(key).strip() and str(value).strip()}


def _load_predicate_inventory(artifact_path: Path) -> list[PredicateSchema]:
    predicate_inventory_path = artifact_path / "predicate_inventory.json"
    if not predicate_inventory_path.exists():
        return []
    loaded_inventory = load_json(predicate_inventory_path)
    if not isinstance(loaded_inventory, list):
        return []
    inventory: list[PredicateSchema] = []
    for row in loaded_inventory:
        if not isinstance(row, dict):
            continue
        predicate_name = str(row.get("predicate_name") or "").strip()
        if not predicate_name:
            continue
        inventory.append(
            PredicateSchema(
                predicate_name=predicate_name,
                parameter_types=[str(item) for item in row.get("parameter_types", [])],
                comment=(str(row.get("comment")).strip() if row.get("comment") is not None else None),
                predicate_kind=row.get("predicate_kind"),
                is_static_feature=bool(row.get("is_static_feature", False)),
            )
        )
    return inventory


def _zero_arity_predicate_names(domain_file: str | Path) -> list[str]:
    parsed_domain = parse_domain(Path(domain_file).read_text(encoding="utf-8"))
    zero_arity = [
        predicate.name
        for predicate in parsed_domain.predicates
        if len(parsed_domain.predicate_parameter_types.get(predicate, [])) == 0
    ]
    return sorted(set(zero_arity))


def _object_matches_type(parsed_domain, *, object_type_name: str, expected_type_name: str) -> bool:
    object_type = parsed_domain.types.get(object_type_name)
    if object_type is None:
        return object_type_name == expected_type_name or expected_type_name == "object"
    return object_type.is_subtype_of(expected_type_name)


def _missing_ground_facts_for_objects(
    *,
    parsed_domain,
    known_objects: list[tuple[str, str]],
    state_set: set[str],
    zero_arity_predicates: set[str],
) -> list[str]:
    missing_facts: list[str] = []
    for predicate in parsed_domain.predicates:
        predicate_name = predicate.name
        parameter_types = parsed_domain.predicate_parameter_types.get(predicate, [])
        if not parameter_types:
            literal = f"{predicate_name}()"
            if predicate_name not in zero_arity_predicates and literal not in state_set:
                missing_facts.append(literal)
            continue
        candidate_object_names_per_param: list[list[str]] = []
        for _param_name, expected_type_name in parameter_types:
            compatible_names = [
                object_name
                for object_name, object_type_name in known_objects
                if _object_matches_type(
                    parsed_domain,
                    object_type_name=object_type_name,
                    expected_type_name=expected_type_name,
                )
            ]
            if not compatible_names:
                candidate_object_names_per_param = []
                break
            candidate_object_names_per_param.append(compatible_names)
        if not candidate_object_names_per_param:
            continue
        for argument_tuple in product(*candidate_object_names_per_param):
            literal = f"{predicate_name}({','.join(argument_tuple)})"
            if literal not in state_set:
                missing_facts.append(literal)
    return sorted(dict.fromkeys(missing_facts))


def _normalize_state_facts(
    *,
    parsed_domain,
    state_before: Iterable[str],
    object_type_by_name: dict[str, str],
) -> list[str]:
    normalized_facts: list[str] = []
    available_predicate_names = {predicate.name for predicate in parsed_domain.predicates}
    predicate_param_types = {
        predicate.name: [
            expected_type_name
            for _param_name, expected_type_name in parsed_domain.predicate_parameter_types.get(predicate, [])
        ]
        for predicate in parsed_domain.predicates
    }
    for raw_fact in state_before:
        fact = str(raw_fact).strip()
        if not fact:
            continue
        negated, predicate, arguments = parse_symbolic_literal(fact)
        if not negated and len(arguments) == 2:
            expected_types = predicate_param_types.get(predicate, [])
            if len(expected_types) == 2:
                left_type = object_type_by_name.get(arguments[0], "").strip()
                right_type = object_type_by_name.get(arguments[1], "").strip()
                direct_matches = (
                    left_type
                    and right_type
                    and _object_matches_type(
                        parsed_domain,
                        object_type_name=left_type,
                        expected_type_name=expected_types[0],
                    )
                    and _object_matches_type(
                        parsed_domain,
                        object_type_name=right_type,
                        expected_type_name=expected_types[1],
                    )
                )
                swapped_matches = (
                    left_type
                    and right_type
                    and _object_matches_type(
                        parsed_domain,
                        object_type_name=right_type,
                        expected_type_name=expected_types[0],
                    )
                    and _object_matches_type(
                        parsed_domain,
                        object_type_name=left_type,
                        expected_type_name=expected_types[1],
                    )
                )
                if not direct_matches and swapped_matches:
                    fact = f"{predicate}({arguments[1]},{arguments[0]})"
        normalized_facts.append(fact)
        if negated:
            continue
        if predicate == "on_top_of" and len(arguments) == 2:
            object_name, support_name = arguments
            object_type_name = object_type_by_name.get(object_name, "").strip()
            support_type_name = object_type_by_name.get(support_name, "").strip()
            if object_type_name and support_type_name:
                typed_predicate = f"on_top_of_{object_type_name}_{support_type_name}"
                if typed_predicate in available_predicate_names:
                    normalized_facts.append(f"{typed_predicate}({object_name},{support_name})")
    return sorted(dict.fromkeys(normalized_facts))


def _collect_universal_negative_literals(
    *,
    parsed_domain,
    known_objects: list[tuple[str, str]],
    state_set: set[str],
    ground_arguments: list[str],
    object_type_by_name: dict[str, str],
    observed_argument_types_by_predicate: dict[str, list[set[str]]] | None = None,
) -> tuple[set[str], set[str]]:
    action_argument_types = {
        object_type_by_name.get(argument, "").strip()
        for argument in ground_arguments
        if object_type_by_name.get(argument, "").strip()
    }
    argument_mapping = {argument: f"?arg{index}" for index, argument in enumerate(ground_arguments)}
    candidate_literals: set[str] = set()
    eligible_literals: set[str] = set()

    for predicate in parsed_domain.predicates:
        predicate_name = predicate.name
        parameter_types = parsed_domain.predicate_parameter_types.get(predicate, [])
        observed_parameter_types = (observed_argument_types_by_predicate or {}).get(predicate_name, [])
        if not parameter_types:
            continue
        for dep_index, (_dep_param_name, dep_expected_type) in enumerate(parameter_types):
            observed_dep_types = (
                observed_parameter_types[dep_index] if dep_index < len(observed_parameter_types) else set()
            )
            dep_objects = [
                object_name
                for object_name, object_type_name in known_objects
                if object_name not in argument_mapping
                and object_type_name not in action_argument_types
                and (not observed_dep_types or object_type_name in observed_dep_types)
                and _object_matches_type(
                    parsed_domain,
                    object_type_name=object_type_name,
                    expected_type_name=dep_expected_type,
                )
            ]
            if not dep_objects:
                continue

            candidate_names_per_param: list[list[str] | None] = []
            valid_shape = True
            for index, (_param_name, expected_type_name) in enumerate(parameter_types):
                if index == dep_index:
                    candidate_names_per_param.append(None)
                    continue
                compatible_action_arguments = [
                    argument
                    for argument in ground_arguments
                    if _object_matches_type(
                        parsed_domain,
                        object_type_name=object_type_by_name.get(argument, ""),
                        expected_type_name=expected_type_name,
                    )
                    and (
                        index >= len(observed_parameter_types)
                        or not observed_parameter_types[index]
                        or object_type_by_name.get(argument, "") in observed_parameter_types[index]
                    )
                ]
                if not compatible_action_arguments:
                    valid_shape = False
                    break
                candidate_names_per_param.append(compatible_action_arguments)
            if not valid_shape:
                continue

            other_param_options = [names for names in candidate_names_per_param if names is not None]
            for chosen_arguments in product(*other_param_options):
                chosen_iter = iter(chosen_arguments)
                grounded_arguments: list[str] = []
                for index in range(len(parameter_types)):
                    if index == dep_index:
                        grounded_arguments.append("?dep0")
                    else:
                        grounded_arguments.append(next(chosen_iter))
                abstract_arguments = [argument_mapping.get(argument, argument) for argument in grounded_arguments]
                if not any(argument.startswith("?arg") for argument in abstract_arguments):
                    continue
                abstract_literal = f"not {predicate_name}({','.join(abstract_arguments)})"
                eligible_literals.add(abstract_literal)

                all_missing = True
                for dep_object in dep_objects:
                    concrete_arguments = list(grounded_arguments)
                    concrete_arguments[dep_index] = dep_object
                    grounded_literal = f"{predicate_name}({','.join(concrete_arguments)})"
                    if grounded_literal in state_set:
                        all_missing = False
                        break
                if all_missing:
                    candidate_literals.add(abstract_literal)

        # Some demonstrated ordering constraints clear a relation whose objects are
        # entirely external to the later action. Represent those as a fully
        # quantified negative; the caller retains them only when an earlier step in
        # the same trajectory actually deleted a matching fact.
        if len(parameter_types) < 2:
            continue
        dep_objects_per_param: list[list[str]] = []
        for index, (_param_name, expected_type_name) in enumerate(parameter_types):
            observed_types = observed_parameter_types[index] if index < len(observed_parameter_types) else set()
            compatible_objects = [
                object_name
                for object_name, object_type_name in known_objects
                if object_name not in argument_mapping
                and (not observed_types or object_type_name in observed_types)
                and _object_matches_type(
                    parsed_domain,
                    object_type_name=object_type_name,
                    expected_type_name=expected_type_name,
                )
            ]
            if not compatible_objects:
                dep_objects_per_param = []
                break
            dep_objects_per_param.append(compatible_objects)
        if not dep_objects_per_param:
            continue

        abstract_arguments = [f"?dep{index}" for index in range(len(parameter_types))]
        abstract_literal = f"not {predicate_name}({','.join(abstract_arguments)})"
        eligible_literals.add(abstract_literal)
        if all(
            f"{predicate_name}({','.join(concrete_arguments)})" not in state_set
            for concrete_arguments in product(*dep_objects_per_param)
        ):
            candidate_literals.add(abstract_literal)

    return candidate_literals, eligible_literals


def _abstract_state_before_candidates(
    *,
    parsed_domain,
    known_objects: list[tuple[str, str]],
    state_before: Iterable[str],
    ground_arguments: list[str],
    object_type_by_name: dict[str, str] | None = None,
    missing_ground_facts: Iterable[str] | None = None,
    zero_arity_predicates: list[str] | None = None,
    observed_argument_types_by_predicate: dict[str, list[set[str]]] | None = None,
) -> tuple[set[str], set[str]]:
    object_type_by_name = {str(key): str(value) for key, value in (object_type_by_name or {}).items()}
    normalized_state_before = _normalize_state_facts(
        parsed_domain=parsed_domain,
        state_before=state_before,
        object_type_by_name=object_type_by_name,
    )
    state_set = {str(item).strip() for item in normalized_state_before if str(item).strip()}
    argument_mapping = {argument: f"?arg{index}" for index, argument in enumerate(ground_arguments)}

    parsed_positive_facts: list[tuple[str, list[str]]] = []
    zero_arity_literals: set[str] = set()
    for fact in sorted(state_set):
        negated, predicate, arguments = parse_symbolic_literal(fact)
        if negated:
            continue
        if not arguments:
            zero_arity_literals.add(f"{predicate}()")
            continue
        parsed_positive_facts.append((predicate, arguments))

    candidate_literals: set[str] = set(zero_arity_literals)
    eligible_literals: set[str] = set(zero_arity_literals)
    eligible_negative_literals: list[tuple[str, list[str]]] = []
    for fact in sorted(str(item).strip() for item in (missing_ground_facts or []) if str(item).strip()):
        negated, predicate, arguments = parse_symbolic_literal(fact)
        if negated:
            continue
        if not arguments:
            if predicate in set(zero_arity_predicates or []):
                candidate_literals.add(f"not {predicate}()")
            continue
        if not all(argument in argument_mapping for argument in arguments):
            continue
        observed_parameter_types = (observed_argument_types_by_predicate or {}).get(predicate, [])
        if any(
            index < len(observed_parameter_types)
            and observed_parameter_types[index]
            and object_type_by_name.get(argument, "") not in observed_parameter_types[index]
            for index, argument in enumerate(arguments)
        ):
            continue
        remapped_arguments = [argument_mapping[argument] for argument in arguments]
        candidate_literals.add(f"not {predicate}({','.join(remapped_arguments)})")
        eligible_literals.add(f"not {predicate}({','.join(remapped_arguments)})")

    for predicate, arguments in parsed_positive_facts:
        if not all(argument in argument_mapping for argument in arguments):
            continue
        remapped_arguments = [argument_mapping[argument] for argument in arguments]
        candidate_literals.add(f"{predicate}({','.join(remapped_arguments)})")
        eligible_literals.add(f"{predicate}({','.join(remapped_arguments)})")
        eligible_negative_literals.append((predicate, arguments))
    for predicate, arguments in eligible_negative_literals:
        remapped_arguments = [argument_mapping[argument] for argument in arguments]
        eligible_literals.add(f"not {predicate}({','.join(remapped_arguments)})")
    universal_candidates, universal_eligible = _collect_universal_negative_literals(
        parsed_domain=parsed_domain,
        known_objects=known_objects,
        state_set=state_set,
        ground_arguments=ground_arguments,
        object_type_by_name=object_type_by_name,
        observed_argument_types_by_predicate=observed_argument_types_by_predicate,
    )
    candidate_literals.update(universal_candidates)
    eligible_literals.update(universal_eligible)
    return candidate_literals, eligible_literals


def _collect_candidate_bundles(
    *,
    episode_grounding_pairs: Iterable[tuple[str | Path, str | Path]],
    parsed_domain,
    zero_arity_predicates: list[str],
    excluded_predicate_names: set[str] | None = None,
) -> dict[str, ActionPreconditionCandidateBundle]:
    episode_grounding_pairs = list(episode_grounding_pairs)
    examples_by_action: dict[str, list[GroundedPreconditionExample]] = defaultdict(list)
    witnessed_dependencies: set[tuple[str, str]] = set()
    zero_arity_predicate_set = set(zero_arity_predicates)
    excluded_predicate_names = set(excluded_predicate_names or set())

    observed_argument_types_by_predicate: dict[str, list[set[str]]] = {}
    for _episode_file, grounding_dir in episode_grounding_pairs:
        grounding_path = Path(grounding_dir)
        problem_summary = load_json_object(grounding_path / "problem_summary.json")
        object_type_by_name = {
            str(item.get("name") or "").strip(): str(item.get("type_name") or "").strip()
            for item in problem_summary.get("objects", [])
            if isinstance(item, dict)
            and str(item.get("name") or "").strip()
            and str(item.get("type_name") or "").strip()
        }
        validation_payload = load_json_object(grounding_path / "validation_report.json")
        for validation_row in validation_payload.get("steps", []):
            if not isinstance(validation_row, dict):
                continue
            facts = [
                *validation_row.get("state_before", []),
                *validation_row.get("state_after", []),
            ]
            for fact in _normalize_state_facts(
                parsed_domain=parsed_domain,
                state_before=facts,
                object_type_by_name=object_type_by_name,
            ):
                negated, predicate, arguments = parse_symbolic_literal(fact)
                if negated or not arguments:
                    continue
                position_types = observed_argument_types_by_predicate.setdefault(
                    predicate,
                    [set() for _argument in arguments],
                )
                if len(position_types) < len(arguments):
                    position_types.extend(set() for _ in range(len(arguments) - len(position_types)))
                for index, argument in enumerate(arguments):
                    object_type = object_type_by_name.get(argument, "")
                    if object_type:
                        position_types[index].add(object_type)

    for _episode_file, grounding_dir in episode_grounding_pairs:
        grounding_path = Path(grounding_dir)
        grounded_rows = [
            row
            for row in load_jsonl(grounding_path / "grounded_trajectory.jsonl")
            if str(row.get("action_category") or "").strip() in {"manipulation", "active_observation", "observation"}
        ]
        validation_payload = load_json_object(grounding_path / "validation_report.json")
        problem_summary = load_json_object(grounding_path / "problem_summary.json")
        known_objects = [
            (
                str(item.get("name") or "").strip(),
                str(item.get("type_name") or "").strip(),
            )
            for item in problem_summary.get("objects", [])
            if isinstance(item, dict)
            and str(item.get("name") or "").strip()
            and str(item.get("type_name") or "").strip()
        ]
        validation_rows = {
            int(item["step_index"]): item
            for item in validation_payload.get("steps", [])
            if isinstance(item, dict) and "step_index" in item
        }
        for row in grounded_rows:
            action_name = str(row.get("canonical_action_name") or "").strip()
            if not action_name:
                continue
            step_index = int(row["step_index"])
            validation_row = validation_rows.get(step_index)
            if not isinstance(validation_row, dict):
                continue
            action_category = str(row.get("action_category") or "").strip()
            validation_status = str(validation_row.get("status") or "").strip()
            if action_category == "manipulation":
                if validation_status != "applied":
                    continue
            elif action_category in {"active_observation", "observation"}:
                if validation_status not in {"applied", "skipped_non_manipulation"}:
                    continue
            else:
                continue
            ground_arguments = [str(arg) for arg in row.get("ground_arguments", [])]
            object_type_by_name = {object_name: object_type_name for object_name, object_type_name in known_objects}
            raw_state_before = [str(fact) for fact in validation_row.get("state_before", [])]
            state_before = _normalize_state_facts(
                parsed_domain=parsed_domain,
                state_before=raw_state_before,
                object_type_by_name=object_type_by_name,
            )
            state_set = {fact for fact in state_before if fact}
            missing_ground_facts = _missing_ground_facts_for_objects(
                parsed_domain=parsed_domain,
                known_objects=known_objects,
                state_set=state_set,
                zero_arity_predicates=zero_arity_predicate_set,
            )
            candidate_literals, eligible_literals = _abstract_state_before_candidates(
                parsed_domain=parsed_domain,
                known_objects=known_objects,
                state_before=state_before,
                ground_arguments=ground_arguments,
                object_type_by_name=object_type_by_name,
                missing_ground_facts=missing_ground_facts,
                zero_arity_predicates=zero_arity_predicates,
                observed_argument_types_by_predicate=(observed_argument_types_by_predicate),
            )
            for zero_arity_predicate in zero_arity_predicates:
                zero_literal = f"{zero_arity_predicate}()"
                if zero_literal in state_before:
                    candidate_literals.add(zero_literal)
                else:
                    candidate_literals.add(f"not {zero_literal}")
                eligible_literals.add(zero_literal)
                eligible_literals.add(f"not {zero_literal}")
            candidate_literals = {
                literal
                for literal in candidate_literals
                if _literal_predicate_name(literal) not in excluded_predicate_names
            }
            eligible_literals = {
                literal
                for literal in eligible_literals
                if _literal_predicate_name(literal) not in excluded_predicate_names
            }
            for literal in candidate_literals:
                if _literal_uses_dependency_variable(literal) and _dependency_literal_has_prior_deletion(
                    literal=literal,
                    ground_arguments=ground_arguments,
                    step_index=step_index,
                    validation_rows=validation_rows,
                    parsed_domain=parsed_domain,
                    object_type_by_name=object_type_by_name,
                ):
                    witnessed_dependencies.add((action_name, literal))
            example = GroundedPreconditionExample(
                episode_name=str(row.get("episode_name") or ""),
                step_index=step_index,
                action_name=action_name,
                ground_arguments=ground_arguments,
                state_before=state_before,
                candidate_literals=sorted(candidate_literals),
                eligible_literals=sorted(eligible_literals),
            )
            examples_by_action[action_name].append(example)

    bundles: dict[str, ActionPreconditionCandidateBundle] = {}
    for action_name, examples in examples_by_action.items():
        filtered_examples = [
            GroundedPreconditionExample(
                episode_name=example.episode_name,
                step_index=example.step_index,
                action_name=example.action_name,
                ground_arguments=list(example.ground_arguments),
                state_before=list(example.state_before),
                candidate_literals=[
                    literal
                    for literal in example.candidate_literals
                    if not _literal_uses_dependency_variable(literal)
                    or (action_name, literal) in witnessed_dependencies
                ],
                eligible_literals=[
                    literal
                    for literal in example.eligible_literals
                    if not _literal_uses_dependency_variable(literal)
                    or (action_name, literal) in witnessed_dependencies
                ],
            )
            for example in examples
        ]
        literal_counts: dict[str, int] = defaultdict(int)
        literal_eligible_counts: dict[str, int] = defaultdict(int)
        for example in filtered_examples:
            for literal in example.candidate_literals:
                literal_counts[literal] += 1
                if not _literal_uses_dependency_variable(literal):
                    literal_eligible_counts[literal] += 1
            for literal in example.eligible_literals:
                if _literal_uses_dependency_variable(literal):
                    literal_eligible_counts[literal] += 1
        example_count = len(examples)
        candidate_stats = [
            PreconditionCandidateStat(
                literal=literal,
                occurrence_count=count,
                eligible_example_count=(
                    example_count
                    if not _literal_uses_dependency_variable(literal)
                    else max(literal_eligible_counts.get(literal, 0), count)
                ),
                support=(
                    count
                    / (
                        example_count
                        if not _literal_uses_dependency_variable(literal)
                        else max(literal_eligible_counts.get(literal, 0), count)
                    )
                    if (
                        example_count
                        if not _literal_uses_dependency_variable(literal)
                        else max(literal_eligible_counts.get(literal, 0), count)
                    )
                    else 0.0
                ),
            )
            for literal, count in sorted(literal_counts.items())
        ]
        bundles[action_name] = ActionPreconditionCandidateBundle(
            action_name=action_name,
            example_count=example_count,
            candidate_stats=candidate_stats,
            examples=sorted(
                filtered_examples,
                key=lambda item: (item.episode_name, item.step_index),
            ),
        )
    return bundles


@dataclass
class PreconditionLearningLearner:
    selection_module: LLMPreconditionSelectionModule | None
    max_workers: int = 1
    keep_all_intersection_preconditions: bool = False

    def learn_from_groundings(
        self,
        *,
        artifact_dir: str | Path,
        domain_file: str | Path,
        episode_grounding_pairs: Iterable[tuple[str | Path, str | Path]],
    ) -> LearnedPreconditionSummary:
        artifact_path = Path(artifact_dir)
        domain_path = Path(domain_file)
        domain_text = domain_path.read_text(encoding="utf-8")
        parsed_domain = parse_domain(domain_text)
        action_schemas = load_action_schemas(artifact_path / "action_schemas.json")
        predicate_inventory = _load_predicate_inventory(artifact_path)
        feature_predicate_names = {
            item.predicate_name
            for item in predicate_inventory
            if str(item.predicate_kind or "").strip().lower() == "feature" or item.is_static_feature
        }
        zero_arity_predicates = sorted(
            predicate.name
            for predicate in parsed_domain.predicates
            if len(parsed_domain.predicate_parameter_types.get(predicate, [])) == 0
        )
        bundles = _collect_candidate_bundles(
            episode_grounding_pairs=episode_grounding_pairs,
            parsed_domain=parsed_domain,
            zero_arity_predicates=zero_arity_predicates,
            excluded_predicate_names=feature_predicate_names,
        )
        schema_map = {schema.canonical_action_name: schema for schema in action_schemas}

        selected_preconditions_by_action: dict[str, list[str]] = {}
        selection_summaries_by_action: dict[str, str] = {}

        def _select_for_action(action_name: str) -> tuple[str, list[str], str]:
            bundle = bundles[action_name]
            schema = schema_map.get(action_name)
            if schema is None:
                return action_name, [], ""
            if self.keep_all_intersection_preconditions:
                selected = sorted(
                    item.literal
                    for item in bundle.candidate_stats
                    if item.occurrence_count == item.eligible_example_count
                )
                return (
                    action_name,
                    selected,
                    "Kept all candidate literals present in every eligible grounded example; skipped final LLM selection.",
                )
            if self.selection_module is None:
                raise ValueError(
                    "PreconditionLearningLearner requires a selection_module when "
                    "keep_all_intersection_preconditions is disabled."
                )
            selected, summary = self.selection_module.select_preconditions(
                domain_text=domain_text,
                action_schema=schema,
                bundle=bundle,
            )
            return action_name, sorted(dict.fromkeys(selected)), summary

        action_names = sorted(name for name in bundles if name in schema_map)
        if self.max_workers > 1 and len(action_names) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for action_name, selected, summary in executor.map(_select_for_action, action_names):
                    selected_preconditions_by_action[action_name] = selected
                    selection_summaries_by_action[action_name] = summary
        else:
            for action_name in action_names:
                selected_action_name, selected, summary = _select_for_action(action_name)
                selected_preconditions_by_action[selected_action_name] = selected
                selection_summaries_by_action[selected_action_name] = summary

        universally_supported_by_action = {
            action_name: [
                item.literal
                for item in bundle.candidate_stats
                if item.eligible_example_count > 0 and item.occurrence_count == item.eligible_example_count
            ]
            for action_name, bundle in bundles.items()
        }
        selected_preconditions_by_action = prune_directional_variant_contradictions(
            selected_preconditions_by_action=selected_preconditions_by_action,
            universally_supported_by_action=universally_supported_by_action,
            parameter_types_by_action={
                schema.canonical_action_name: tuple(schema.parameter_roles) for schema in action_schemas
            },
        )

        updated_action_schemas: list[ActionSchema] = []
        for schema in action_schemas:
            if schema.canonical_action_name not in selected_preconditions_by_action:
                updated_action_schemas.append(schema)
                continue
            updated_action_schemas.append(
                ActionSchema(
                    canonical_action_name=schema.canonical_action_name,
                    action_category=schema.action_category,
                    parameter_count=schema.parameter_count,
                    parameter_roles=list(schema.parameter_roles),
                    precondition_literals=list(selected_preconditions_by_action[schema.canonical_action_name]),
                    schema_description=schema.schema_description,
                    effect_branches=list(schema.effect_branches),
                )
            )

        action_comments = extract_action_comments(domain_text)
        if action_comments:
            updated_action_schemas = [
                ActionSchema(
                    canonical_action_name=schema.canonical_action_name,
                    action_category=schema.action_category,
                    parameter_count=schema.parameter_count,
                    parameter_roles=list(schema.parameter_roles),
                    precondition_literals=list(schema.precondition_literals),
                    schema_description=action_comments.get(schema.canonical_action_name) or schema.schema_description,
                    effect_branches=list(schema.effect_branches),
                )
                for schema in updated_action_schemas
            ]
        predicate_comments = _load_predicate_comments(artifact_path)
        predicate_comments.update(extract_predicate_comments(domain_text))
        object_types_path = artifact_path / "object_types.json"
        object_types = load_object_types(object_types_path) if object_types_path.exists() else []
        rendered_action_schema_pddl = render_action_schema_fragment(
            updated_action_schemas,
            predicate_inventory=predicate_inventory,
            predicate_comments=predicate_comments,
            object_types=object_types,
        )
        rendered_domain_pddl = apply_domain_repair_patch(
            domain_text=domain_text,
            predicate_comment_updates=None,
            predicate_parameter_type_updates=None,
            predicate_additions=None,
            predicate_removals=None,
            action_patches=[
                DomainActionPatch(
                    action_name=schema.canonical_action_name,
                    precondition_literals=list(schema.precondition_literals),
                )
                for schema in updated_action_schemas
            ],
        )

        summary = LearnedPreconditionSummary(
            zero_arity_predicates=zero_arity_predicates,
            example_counts_by_action={
                action_name: bundle.example_count for action_name, bundle in sorted(bundles.items())
            },
            selected_preconditions_by_action=selected_preconditions_by_action,
            candidate_stats_by_action={
                action_name: [item.to_dict() for item in bundle.candidate_stats]
                for action_name, bundle in sorted(bundles.items())
            },
            selection_summaries_by_action=selection_summaries_by_action,
            updated_action_schemas=[schema.to_dict() for schema in updated_action_schemas],
            rendered_action_schema_pddl=rendered_action_schema_pddl,
            rendered_domain_pddl=rendered_domain_pddl,
            predicate_comments=predicate_comments,
        )
        return summary
