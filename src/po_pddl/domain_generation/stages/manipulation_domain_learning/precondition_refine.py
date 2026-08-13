from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from po_pddl.core.parser import parse_domain
from po_pddl.domain_generation.infrastructure.artifact_io import load_json, load_json_object, load_jsonl
from po_pddl.domain_generation.infrastructure.fact_utils import parse_symbolic_literal
from po_pddl.domain_generation.stages.domain_comments import (
    extract_action_comments,
    extract_predicate_comments,
)

from .grounding_update import load_action_schemas, load_manipulation_records, load_object_types
from .models import ActionSchema
from .renderer import (
    collect_action_effect_statistics,
    render_action_schema_fragment,
    render_manipulation_domain_fragment,
)


@dataclass(frozen=True)
class PreconditionRefinementSummary:
    zero_arity_predicates: list[str]
    candidate_counts_by_action: dict[str, int]
    refined_preconditions_by_action: dict[str, list[str]]

    def to_dict(self) -> dict[str, object]:
        return {
            "zero_arity_predicates": list(self.zero_arity_predicates),
            "candidate_counts_by_action": dict(self.candidate_counts_by_action),
            "refined_preconditions_by_action": {
                key: list(value) for key, value in self.refined_preconditions_by_action.items()
            },
        }


def _zero_arity_predicate_names(domain_file: str | Path) -> list[str]:
    parsed_domain = parse_domain(Path(domain_file).read_text(encoding="utf-8"))
    zero_arity = [
        predicate.name
        for predicate in parsed_domain.predicates
        if len(parsed_domain.predicate_parameter_types.get(predicate, [])) == 0
    ]
    return sorted(set(zero_arity))


def _abstract_state_before_candidates(
    *,
    state_before: Iterable[str],
    ground_arguments: list[str],
    zero_arity_predicates: list[str],
) -> set[str]:
    state_set = {str(item).strip() for item in state_before if str(item).strip()}
    argument_mapping = {argument: f"?arg{index}" for index, argument in enumerate(ground_arguments)}
    candidate_literals: set[str] = set()

    for fact in sorted(state_set):
        negated, predicate, arguments = parse_symbolic_literal(fact)
        if negated:
            continue
        if not arguments:
            candidate_literals.add(f"{predicate}()")
            continue
        if all(argument in argument_mapping for argument in arguments):
            remapped_arguments = [argument_mapping[argument] for argument in arguments]
            candidate_literals.add(f"{predicate}({','.join(remapped_arguments)})")

    present_zero_arity = {parse_symbolic_literal(fact)[1] for fact in state_set if not parse_symbolic_literal(fact)[2]}
    for predicate_name in zero_arity_predicates:
        if predicate_name in present_zero_arity:
            candidate_literals.add(f"{predicate_name}()")
        else:
            candidate_literals.add(f"not {predicate_name}()")
    return candidate_literals


def _collect_refined_preconditions(
    *,
    episode_grounding_pairs: Iterable[tuple[str | Path, str | Path]],
    zero_arity_predicates: list[str],
) -> tuple[dict[str, list[set[str]]], dict[str, int]]:
    candidates_by_action: dict[str, list[set[str]]] = {}
    candidate_counts_by_action: dict[str, int] = {}

    for _episode_file, grounding_dir in episode_grounding_pairs:
        grounding_path = Path(grounding_dir)
        grounded_rows = load_jsonl(grounding_path / "grounded_trajectory.jsonl")
        validation_payload = load_json_object(grounding_path / "validation_report.json")
        state_before_by_step = {
            int(item["step_index"]): [str(fact) for fact in item.get("state_before", [])]
            for item in validation_payload.get("steps", [])
            if isinstance(item, dict) and "step_index" in item
        }
        for row in grounded_rows:
            if str(row.get("action_category") or "").strip() != "manipulation":
                continue
            action_name = str(row.get("canonical_action_name") or "").strip()
            if not action_name:
                continue
            step_index = int(row["step_index"])
            ground_arguments = [str(arg) for arg in row.get("ground_arguments", [])]
            state_before = state_before_by_step.get(step_index)
            if state_before is None:
                continue
            candidate_set = _abstract_state_before_candidates(
                state_before=state_before,
                ground_arguments=ground_arguments,
                zero_arity_predicates=zero_arity_predicates,
            )
            candidates_by_action.setdefault(action_name, []).append(candidate_set)
            candidate_counts_by_action[action_name] = candidate_counts_by_action.get(action_name, 0) + 1
    return candidates_by_action, candidate_counts_by_action


def _intersect_candidate_sets(candidates_by_action: dict[str, list[set[str]]]) -> dict[str, list[str]]:
    refined: dict[str, list[str]] = {}
    for action_name, candidate_sets in candidates_by_action.items():
        if not candidate_sets:
            continue
        intersection = set(candidate_sets[0])
        for candidate_set in candidate_sets[1:]:
            intersection &= candidate_set
        refined[action_name] = sorted(intersection)
    return refined


def refine_action_schemas_preconditions(
    action_schemas: list[ActionSchema],
    refined_preconditions_by_action: dict[str, list[str]],
) -> list[ActionSchema]:
    refined_schemas: list[ActionSchema] = []
    for schema in action_schemas:
        refined_preconditions = refined_preconditions_by_action.get(schema.canonical_action_name)
        if refined_preconditions is None:
            refined_schemas.append(schema)
            continue
        merged_preconditions = sorted(set(schema.precondition_literals) | set(refined_preconditions))
        refined_schemas.append(
            ActionSchema(
                canonical_action_name=schema.canonical_action_name,
                action_category=schema.action_category,
                parameter_count=schema.parameter_count,
                parameter_roles=list(schema.parameter_roles),
                precondition_literals=merged_preconditions,
                schema_description=schema.schema_description,
                effect_branches=list(schema.effect_branches),
            )
        )
    return refined_schemas


def refine_domain_learning_preconditions_from_groundings(
    *,
    artifact_dir: str | Path,
    domain_file: str | Path,
    episode_grounding_pairs: Iterable[tuple[str | Path, str | Path]],
) -> PreconditionRefinementSummary:
    artifact_path = Path(artifact_dir)
    domain_text = Path(domain_file).read_text(encoding="utf-8")
    action_schemas = load_action_schemas(artifact_path / "action_schemas.json")
    manipulation_records = load_manipulation_records(artifact_path / "manipulation_records.jsonl")
    action_statistics = collect_action_effect_statistics(manipulation_records)
    zero_arity_predicates = _zero_arity_predicate_names(domain_file)
    candidates_by_action, candidate_counts_by_action = _collect_refined_preconditions(
        episode_grounding_pairs=episode_grounding_pairs,
        zero_arity_predicates=zero_arity_predicates,
    )
    refined_preconditions_by_action = _intersect_candidate_sets(candidates_by_action)
    refined_action_schemas = refine_action_schemas_preconditions(
        action_schemas,
        refined_preconditions_by_action,
    )
    action_comments = extract_action_comments(domain_text)
    if action_comments:
        refined_action_schemas = [
            ActionSchema(
                canonical_action_name=schema.canonical_action_name,
                action_category=schema.action_category,
                parameter_count=schema.parameter_count,
                parameter_roles=list(schema.parameter_roles),
                precondition_literals=list(schema.precondition_literals),
                schema_description=action_comments.get(schema.canonical_action_name) or schema.schema_description,
                effect_branches=list(schema.effect_branches),
            )
            for schema in refined_action_schemas
        ]
    predicate_comments: dict[str, str] = {}
    predicate_comments_path = artifact_path / "predicate_comments.json"
    if predicate_comments_path.exists():
        loaded_comments = load_json(predicate_comments_path)
        if isinstance(loaded_comments, dict):
            predicate_comments = {
                str(key): str(value)
                for key, value in loaded_comments.items()
                if str(key).strip() and str(value).strip()
            }
    predicate_comments.update(extract_predicate_comments(domain_text))
    object_types = []
    object_types_path = artifact_path / "object_types.json"
    if object_types_path.exists():
        object_types = load_object_types(object_types_path)

    (artifact_path / "action_schemas.json").write_text(
        json.dumps([schema.to_dict() for schema in refined_action_schemas], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (artifact_path / "action_schemas.pddl").write_text(
        render_action_schema_fragment(
            refined_action_schemas,
            predicate_comments=predicate_comments,
            object_types=object_types,
        ),
        encoding="utf-8",
    )
    (artifact_path / "manipulation_actions.pddl").write_text(
        render_manipulation_domain_fragment(
            refined_action_schemas,
            manipulation_records,
            action_statistics,
            predicate_comments=predicate_comments,
            object_types=object_types,
        ),
        encoding="utf-8",
    )
    summary = PreconditionRefinementSummary(
        zero_arity_predicates=zero_arity_predicates,
        candidate_counts_by_action=candidate_counts_by_action,
        refined_preconditions_by_action=refined_preconditions_by_action,
    )
    (artifact_path / "precondition_refinement_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "PreconditionRefinementSummary",
    "refine_action_schemas_preconditions",
    "refine_domain_learning_preconditions_from_groundings",
]
