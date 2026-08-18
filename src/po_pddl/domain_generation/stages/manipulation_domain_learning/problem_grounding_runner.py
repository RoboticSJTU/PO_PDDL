from __future__ import annotations

import json
import logging
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from po_pddl.core.conventions import containment_arguments
from po_pddl.core.parser import parse_domain, parse_problem
from po_pddl.domain_generation.infrastructure.artifact_io import load_episode_payload, load_json, load_jsonl
from po_pddl.domain_generation.infrastructure.fact_utils import (
    format_symbolic_literal,
    parse_symbolic_literal,
)
from po_pddl.domain_generation.stages.problem_grounding.models import (
    GroundedTrajectoryStep,
    ObjectDeclaration,
    ProblemGroundingResult,
    ProblemSpec,
    ValidationIssue,
    ValidationStepReport,
)
from po_pddl.domain_generation.stages.problem_grounding.renderer import render_problem_pddl
from po_pddl.domain_generation.stages.problem_grounding.validator import execute_grounded_step

from .grounding_update import load_manipulation_records
from .models import ActionTaxonomyRecord, PredicateSchema, RawTrajectoryStep

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModuleModeSummary:
    object_init_module: str
    goal_inference_module: str
    assembly_module: str
    trajectory_grounding_module: str
    validator_module: str


def _load_episode_steps_from_file(episode_file: str | Path) -> list[RawTrajectoryStep]:
    payload = load_episode_payload(episode_file)
    episode_name = str(payload.get("episode_name", Path(episode_file).parent.name))
    instruction = str(payload.get("instruction", "")).strip()
    previous_observation_text: str | None = None
    previous_known_observation_text: str | None = None
    outputs: list[RawTrajectoryStep] = []
    for step in payload.get("steps", []):
        observation_text = step.get("observation_text")
        action_text = step.get("action_text")
        extra_info = step.get("extra_info")
        raw_step = RawTrajectoryStep(
            episode_name=episode_name,
            instruction=instruction,
            step_index=int(step["step_index"]),
            start_time_sec=step.get("start_time_sec"),
            end_time_sec=step.get("end_time_sec"),
            action_text=str(action_text).strip() if action_text is not None and str(action_text).strip() else None,
            observation_text=str(observation_text).strip()
            if observation_text is not None and str(observation_text).strip()
            else None,
            extra_info=str(extra_info).strip() if extra_info is not None and str(extra_info).strip() else None,
            previous_observation_text=previous_observation_text,
            previous_known_observation_text=previous_known_observation_text,
        )
        outputs.append(raw_step)
        previous_observation_text = raw_step.observation_text
        if raw_step.observation_text is not None:
            previous_known_observation_text = raw_step.observation_text
    return outputs


def _load_taxonomy_records(artifact_dir: str | Path, *, episode_name: str) -> list[ActionTaxonomyRecord]:
    rows = [
        row
        for row in load_jsonl(Path(artifact_dir) / "action_taxonomy.jsonl")
        if str(row.get("episode_name") or "").strip() == episode_name
    ]
    return [
        ActionTaxonomyRecord(
            episode_name=str(row["episode_name"]),
            step_index=int(row["step_index"]),
            raw_action_text=str(row["raw_action_text"]),
            proposed_action_name=str(row["proposed_action_name"]),
            canonical_action_name=str(row["canonical_action_name"]),
            action_category=str(row["action_category"]),
            action_arguments=[str(item) for item in row.get("action_arguments", [])],
            object_mentions=[str(item) for item in row.get("object_mentions", [])],
            observation_text=row.get("observation_text"),
            extra_info=row.get("extra_info"),
            template_text=row.get("template_text"),
            parameter_placeholders=[str(item) for item in row.get("parameter_placeholders", [])],
            action_argument_types=[str(item) for item in row.get("action_argument_types", [])],
        )
        for row in rows
    ]


def _load_predicate_inventory(artifact_dir: str | Path) -> list[PredicateSchema]:
    path = Path(artifact_dir) / "predicate_inventory.json"
    if not path.exists():
        return []
    rows = load_json(path)
    if not isinstance(rows, list):
        return []
    outputs: list[PredicateSchema] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("predicate_name") or "").strip()
        if not name:
            continue
        outputs.append(
            PredicateSchema(
                predicate_name=name,
                parameter_types=[str(item) for item in row.get("parameter_types", [])],
                comment=row.get("comment"),
                predicate_kind=row.get("predicate_kind"),
                is_static_feature=bool(row.get("is_static_feature", False)),
            )
        )
    return outputs


def _result_summary(result: ProblemGroundingResult) -> dict[str, Any]:
    return {
        "goal_satisfied": result.goal_satisfied,
        "issue_count": len(result.validation_issues),
        "issues": [issue.to_dict() for issue in result.validation_issues],
        "steps": [step.to_dict() for step in result.validation_steps],
    }


def _render_ground_action_pddl(action_name: str, arguments: list[str]) -> str:
    if arguments:
        return f"({action_name} {' '.join(arguments)})"
    return f"({action_name})"


def _object_declaration_from_row(row: dict[str, Any]) -> ObjectDeclaration:
    return ObjectDeclaration(
        name=str(row.get("name") or "").strip(),
        type_name=str(row.get("type_name") or "object").strip() or "object",
    )


def _problem_spec_from_row(row: dict[str, Any]) -> ProblemSpec:
    return ProblemSpec(
        problem_name=str(row.get("problem_name") or "").strip(),
        domain_name=str(row.get("domain_name") or "").strip(),
        objects=[
            _object_declaration_from_row(item)
            for item in row.get("objects", [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ],
        init_facts=[str(item) for item in row.get("init_facts", [])],
        goal_facts=[str(item) for item in row.get("goal_facts", [])],
        canonical_object_map={
            str(key): str(value)
            for key, value in dict(row.get("canonical_object_map", {})).items()
            if str(key).strip() and str(value).strip()
        },
    )


def _grounded_step_from_row(row: dict[str, Any]) -> GroundedTrajectoryStep:
    return GroundedTrajectoryStep(
        episode_name=str(row.get("episode_name") or "").strip(),
        step_index=int(row.get("step_index", 0)),
        raw_action_text=row.get("raw_action_text"),
        action_category=row.get("action_category"),
        canonical_action_name=row.get("canonical_action_name"),
        ground_arguments=[str(item) for item in row.get("ground_arguments", [])],
        ground_action_pddl=row.get("ground_action_pddl"),
        effect_bucket=row.get("effect_bucket"),
        delta_add=[str(item) for item in row.get("delta_add", [])],
        delta_del=[str(item) for item in row.get("delta_del", [])],
        success=row.get("success"),
        observation_text=row.get("observation_text"),
        extra_info=row.get("extra_info"),
        requested_effect_bucket=row.get("requested_effect_bucket"),
        branch_expectation=row.get("branch_expectation"),
        missing_effect_branch=bool(row.get("missing_effect_branch", False)),
        available_effect_buckets=[str(item) for item in row.get("available_effect_buckets", [])],
    )


def _validation_step_from_row(row: dict[str, Any]) -> ValidationStepReport:
    return ValidationStepReport(
        step_index=int(row.get("step_index", 0)),
        action_name=row.get("action_name"),
        effect_bucket=row.get("effect_bucket"),
        status=str(row.get("status") or "").strip(),
        state_before=[str(item) for item in row.get("state_before", [])],
        state_after=[str(item) for item in row.get("state_after", [])],
        failed_preconditions=[str(item) for item in row.get("failed_preconditions", [])],
    )


def _validation_issue_from_row(row: dict[str, Any]) -> ValidationIssue:
    return ValidationIssue(
        step_index=int(row.get("step_index", 0)),
        error_code=str(row.get("error_code") or "").strip(),
        message=str(row.get("message") or "").strip(),
        action_name=row.get("action_name"),
        requested_effect_bucket=row.get("requested_effect_bucket"),
        expected_branch=row.get("expected_branch"),
        available_effect_buckets=[str(item) for item in row.get("available_effect_buckets", [])],
        failed_preconditions=[str(item) for item in row.get("failed_preconditions", [])],
        state_before=[str(item) for item in row.get("state_before", [])],
    )


def _load_episode_grounding_result_rows(artifact_dir: str | Path) -> dict[str, dict[str, Any]]:
    artifact_path = Path(artifact_dir)
    candidate_files = [
        artifact_path / "episode_problem_grounding_results.json",
        artifact_path / "post_statistics_episode_repair_summary.json",
        artifact_path / "episode_grounded_effect_learning_summary.json",
    ]
    for path in candidate_files:
        if not path.exists():
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("episode_results"), dict):
            return {
                str(key): value
                for key, value in payload["episode_results"].items()
                if isinstance(key, str) and isinstance(value, dict)
            }
        return {str(key): value for key, value in payload.items() if isinstance(key, str) and isinstance(value, dict)}
    return {}


def _load_stored_episode_grounding_row(
    artifact_dir: str | Path,
    *,
    episode_name: str,
) -> dict[str, Any] | None:
    artifact_path = Path(artifact_dir)
    candidate_dirs = [
        artifact_path,
        artifact_path.parent / "3c_grounding_effect_repair",
        artifact_path.parent / "3b_predicate_type_repair",
        artifact_path.parent / "2_manipulation_domain_learning",
    ]
    seen_dirs: set[Path] = set()
    for candidate_dir in candidate_dirs:
        candidate_dir = candidate_dir.resolve()
        if candidate_dir in seen_dirs or not candidate_dir.exists():
            continue
        seen_dirs.add(candidate_dir)
        stored_row = _load_episode_grounding_result_rows(candidate_dir).get(episode_name)
        if stored_row is not None:
            return stored_row
    return None


def _fact_predicate_name(fact: str) -> str:
    _negated, predicate_name, _arguments = parse_symbolic_literal(str(fact).strip())
    return predicate_name


def _split_fact_arguments(fact: str) -> tuple[str, list[str]]:
    stripped = str(fact).strip()
    if "(" not in stripped or not stripped.endswith(")"):
        return stripped, []
    predicate_name, remainder = stripped.split("(", 1)
    arguments_text = remainder[:-1].strip()
    arguments = [item.strip() for item in arguments_text.split(",")] if arguments_text else []
    return predicate_name.strip(), arguments


def _canonicalize_fact(
    fact: str,
    *,
    object_type_by_name: dict[str, str],
) -> str:
    negated, predicate_name, arguments = parse_symbolic_literal(str(fact).strip())
    if predicate_name != "in" or len(arguments) != 2:
        return format_symbolic_literal(predicate_name, arguments, negated=negated)
    left_name, right_name = arguments
    left_type = object_type_by_name.get(left_name, "").strip()
    right_type = object_type_by_name.get(right_name, "").strip()
    if left_type == "drawer" and right_type and right_type != "drawer":
        movable_name, container_name = containment_arguments(right_name, left_name)
        return format_symbolic_literal("in", [movable_name, container_name], negated=negated)
    if right_type == "drawer" and left_type and left_type != "drawer":
        movable_name, container_name = containment_arguments(left_name, right_name)
        return format_symbolic_literal("in", [movable_name, container_name], negated=negated)
    return format_symbolic_literal(predicate_name, arguments, negated=negated)


def _canonicalize_fact_set(
    facts: set[str] | list[str],
    *,
    object_type_by_name: dict[str, str],
) -> set[str]:
    return {_canonicalize_fact(fact, object_type_by_name=object_type_by_name) for fact in facts if str(fact).strip()}


def _overlay_predicate_facts(
    *,
    base_facts: set[str],
    replacement_facts: set[str],
    predicate_names: set[str],
) -> set[str]:
    if not predicate_names:
        return set(base_facts)
    overlaid = set(base_facts)
    for replacement in sorted(replacement_facts):
        negated, predicate_name, arguments = parse_symbolic_literal(replacement)
        if predicate_name not in predicate_names:
            continue
        positive = format_symbolic_literal(predicate_name, arguments)
        negative = format_symbolic_literal(predicate_name, arguments, negated=True)
        overlaid.discard(positive)
        overlaid.discard(negative)
        overlaid.add(negative if negated else positive)
    return overlaid


def _set_fact_value(state: set[str], fact: str, *, value: bool) -> None:
    _negated, predicate_name, arguments = parse_symbolic_literal(fact)
    positive = format_symbolic_literal(predicate_name, arguments)
    negative = format_symbolic_literal(predicate_name, arguments, negated=True)
    state.discard(positive)
    state.discard(negative)
    state.add(positive if value else negative)


def _goal_fact_holds(state: set[str], goal_fact: str) -> bool:
    negated, predicate_name, arguments = parse_symbolic_literal(goal_fact)
    positive = format_symbolic_literal(predicate_name, arguments)
    return positive not in state if negated else positive in state


def _problem_grounding_result_from_summary_row(row: dict[str, Any]) -> ProblemGroundingResult:
    problem_spec = _problem_spec_from_row(dict(row.get("problem_spec", {})))
    problem_pddl = str(row.get("problem_pddl") or "").strip() or render_problem_pddl(problem_spec)
    return ProblemGroundingResult(
        problem_spec=problem_spec,
        problem_pddl=problem_pddl,
        grounded_steps=[
            _grounded_step_from_row(item) for item in row.get("grounded_steps", []) if isinstance(item, dict)
        ],
        validation_steps=[
            _validation_step_from_row(item) for item in row.get("validation_steps", []) if isinstance(item, dict)
        ],
        validation_issues=[
            _validation_issue_from_row(item) for item in row.get("validation_issues", []) if isinstance(item, dict)
        ],
        goal_satisfied=bool(row.get("goal_satisfied", False)),
        object_init_raw_llm_outputs={
            str(key): str(value)
            for key, value in dict(row.get("object_init_raw_llm_outputs", {})).items()
            if str(key).strip() and isinstance(value, str) and value.strip()
        },
        goal_inference_raw_output=(
            str(row.get("goal_inference_raw_output")).strip()
            if row.get("goal_inference_raw_output") is not None and str(row.get("goal_inference_raw_output")).strip()
            else None
        ),
    )


def load_problem_grounding_result_from_output_dir(output_dir: str | Path) -> ProblemGroundingResult:
    output_path = Path(output_dir)
    problem_summary_payload = load_json(output_path / "problem_summary.json")
    if not isinstance(problem_summary_payload, dict):
        raise ValueError(f"Expected JSON object in {output_path / 'problem_summary.json'}")
    validation_payload = load_json(output_path / "validation_report.json")
    if not isinstance(validation_payload, dict):
        raise ValueError(f"Expected JSON object in {output_path / 'validation_report.json'}")
    grounded_rows = load_jsonl(output_path / "grounded_trajectory.jsonl")
    problem_spec = _problem_spec_from_row(problem_summary_payload)
    problem_pddl = (output_path / "problem.pddl").read_text(encoding="utf-8")
    return ProblemGroundingResult(
        problem_spec=problem_spec,
        problem_pddl=problem_pddl,
        grounded_steps=[_grounded_step_from_row(item) for item in grounded_rows if isinstance(item, dict)],
        validation_steps=[
            _validation_step_from_row(item) for item in validation_payload.get("steps", []) if isinstance(item, dict)
        ],
        validation_issues=[
            _validation_issue_from_row(item) for item in validation_payload.get("issues", []) if isinstance(item, dict)
        ],
        goal_satisfied=bool(validation_payload.get("goal_satisfied", False)),
        object_init_raw_llm_outputs={
            str(key): str(value)
            for key, value in dict(problem_summary_payload.get("object_init_raw_llm_outputs", {})).items()
            if str(key).strip() and isinstance(value, str)
        },
        goal_inference_raw_output=(
            str(problem_summary_payload.get("goal_inference_raw_output")).strip()
            if problem_summary_payload.get("goal_inference_raw_output") is not None
            and str(problem_summary_payload.get("goal_inference_raw_output")).strip()
            else None
        ),
    )


def reverse_effects_to_reconstruct_states(
    result: ProblemGroundingResult,
) -> tuple[ProblemGroundingResult, dict[str, Any]]:
    object_type_by_name = {
        str(item.name).strip(): str(item.type_name).strip()
        for item in result.problem_spec.objects
        if str(item.name).strip()
    }
    validation_by_step = {item.step_index: item for item in result.validation_steps}
    grounded_by_step = {item.step_index: item for item in result.grounded_steps}
    ordered_step_indices = sorted(validation_by_step)
    if not ordered_step_indices:
        return result, {
            "changed": False,
            "reason": "no_validation_steps",
            "added_init_facts": [],
            "removed_init_facts": [],
        }

    all_tracked_predicate_names = {_fact_predicate_name(fact) for fact in result.problem_spec.goal_facts}
    for step in result.grounded_steps:
        all_tracked_predicate_names.update(_fact_predicate_name(fact) for fact in step.delta_add)
        all_tracked_predicate_names.update(_fact_predicate_name(fact) for fact in step.delta_del)

    last_step = validation_by_step[ordered_step_indices[-1]]
    current_state = _canonicalize_fact_set(
        set(last_step.state_after),
        object_type_by_name=object_type_by_name,
    )
    current_state = _overlay_predicate_facts(
        base_facts=current_state,
        replacement_facts=_canonicalize_fact_set(
            set(result.problem_spec.goal_facts),
            object_type_by_name=object_type_by_name,
        ),
        predicate_names={_fact_predicate_name(fact) for fact in result.problem_spec.goal_facts},
    )

    rewritten_validation_steps_desc: list[ValidationStepReport] = []
    for step_index in reversed(ordered_step_indices):
        report = validation_by_step[step_index]
        grounded_step = grounded_by_step.get(step_index)
        touched_predicate_names: set[str] = set()
        if grounded_step is not None:
            touched_predicate_names.update(_fact_predicate_name(fact) for fact in grounded_step.delta_add)
            touched_predicate_names.update(_fact_predicate_name(fact) for fact in grounded_step.delta_del)
        original_state_after = _canonicalize_fact_set(
            set(report.state_after),
            object_type_by_name=object_type_by_name,
        )
        state_after_set = _overlay_predicate_facts(
            base_facts=original_state_after,
            replacement_facts=current_state,
            predicate_names=all_tracked_predicate_names,
        )
        state_after = sorted(_canonicalize_fact_set(state_after_set, object_type_by_name=object_type_by_name))
        previous_state = _canonicalize_fact_set(
            set(current_state),
            object_type_by_name=object_type_by_name,
        )
        if grounded_step is not None:
            for fact in grounded_step.delta_add:
                canonical_fact = _canonicalize_fact(str(fact), object_type_by_name=object_type_by_name)
                _set_fact_value(previous_state, canonical_fact, value=False)
            for fact in grounded_step.delta_del:
                canonical_fact = _canonicalize_fact(str(fact), object_type_by_name=object_type_by_name)
                _set_fact_value(previous_state, canonical_fact, value=True)
        original_state_before = _canonicalize_fact_set(
            set(report.state_before),
            object_type_by_name=object_type_by_name,
        )
        state_before_set = _overlay_predicate_facts(
            base_facts=original_state_before,
            replacement_facts=previous_state,
            predicate_names=all_tracked_predicate_names,
        )
        rewritten_validation_steps_desc.append(
            ValidationStepReport(
                step_index=report.step_index,
                action_name=report.action_name,
                effect_bucket=report.effect_bucket,
                status=report.status,
                state_before=sorted(
                    _canonicalize_fact_set(
                        state_before_set,
                        object_type_by_name=object_type_by_name,
                    )
                ),
                state_after=state_after,
                failed_preconditions=list(report.failed_preconditions),
            )
        )
        current_state = _canonicalize_fact_set(
            set(report.state_before),
            object_type_by_name=object_type_by_name,
        )
        current_state = _overlay_predicate_facts(
            base_facts=current_state,
            replacement_facts=previous_state,
            predicate_names=all_tracked_predicate_names,
        )
        current_state = _canonicalize_fact_set(
            current_state,
            object_type_by_name=object_type_by_name,
        )

    rewritten_validation_steps = list(reversed(rewritten_validation_steps_desc))
    rewritten_problem_spec = ProblemSpec(
        problem_name=result.problem_spec.problem_name,
        domain_name=result.problem_spec.domain_name,
        objects=list(result.problem_spec.objects),
        init_facts=sorted(
            _canonicalize_fact_set(
                _overlay_predicate_facts(
                    base_facts=_canonicalize_fact_set(
                        set(result.problem_spec.init_facts),
                        object_type_by_name=object_type_by_name,
                    ),
                    replacement_facts=current_state,
                    predicate_names=all_tracked_predicate_names,
                ),
                object_type_by_name=object_type_by_name,
            )
        ),
        goal_facts=list(result.problem_spec.goal_facts),
        canonical_object_map=dict(result.problem_spec.canonical_object_map),
    )
    rewritten_result = ProblemGroundingResult(
        problem_spec=rewritten_problem_spec,
        problem_pddl=render_problem_pddl(rewritten_problem_spec),
        grounded_steps=list(result.grounded_steps),
        validation_steps=rewritten_validation_steps,
        validation_issues=list(result.validation_issues),
        goal_satisfied=result.goal_satisfied,
        object_init_raw_llm_outputs=dict(result.object_init_raw_llm_outputs),
        goal_inference_raw_output=result.goal_inference_raw_output,
    )
    original_init = set(result.problem_spec.init_facts)
    new_init = set(rewritten_problem_spec.init_facts)
    return rewritten_result, {
        "changed": rewritten_problem_spec.init_facts != result.problem_spec.init_facts
        or any(
            old.state_before != new.state_before or old.state_after != new.state_after
            for old, new in zip(result.validation_steps, rewritten_validation_steps, strict=False)
        ),
        "reason": "reverse_replay_from_final_state_and_goal",
        "original_init_fact_count": len(result.problem_spec.init_facts),
        "rewritten_init_fact_count": len(rewritten_problem_spec.init_facts),
        "added_init_facts": sorted(new_init - original_init),
        "removed_init_facts": sorted(original_init - new_init),
        "goal_facts_seeded_into_reverse_pass": sorted(set(result.problem_spec.goal_facts)),
    }


def write_problem_grounding_outputs(result: ProblemGroundingResult, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    problem_summary_payload = result.problem_spec.to_dict()
    problem_summary_payload["object_init_raw_llm_outputs"] = dict(result.object_init_raw_llm_outputs)
    problem_summary_payload["goal_inference_raw_output"] = result.goal_inference_raw_output
    (output_path / "problem_summary.json").write_text(
        json.dumps(problem_summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_path / "problem.pddl").write_text(result.problem_pddl, encoding="utf-8")
    with (output_path / "grounded_trajectory.jsonl").open("w", encoding="utf-8") as handle:
        for step in result.grounded_steps:
            handle.write(json.dumps(step.to_dict(), ensure_ascii=False) + "\n")
    (output_path / "validation_report.json").write_text(
        json.dumps(_result_summary(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for name, text in sorted(result.object_init_raw_llm_outputs.items()):
        (output_path / f"object_init_{name}_raw_output.txt").write_text(text, encoding="utf-8")
    if result.goal_inference_raw_output:
        (output_path / "goal_inference_raw_output.txt").write_text(
            result.goal_inference_raw_output,
            encoding="utf-8",
        )


def _augment_with_non_state_changing_observation_steps(
    *,
    result: ProblemGroundingResult,
    episode_steps: list[RawTrajectoryStep],
    taxonomy_records: list[ActionTaxonomyRecord],
) -> ProblemGroundingResult:
    taxonomy_by_step = {item.step_index: item for item in taxonomy_records}
    grounded_by_step = {item.step_index: item for item in result.grounded_steps}
    validation_by_step = {item.step_index: item for item in result.validation_steps}
    first_issue_step = min((item.step_index for item in result.validation_issues), default=None)

    current_state = list(result.problem_spec.init_facts)
    augmented_grounded_steps: list[GroundedTrajectoryStep] = []
    augmented_validation_steps: list[ValidationStepReport] = []
    inserted_any = False

    for step in sorted((item for item in episode_steps if item.step_index > 0), key=lambda item: item.step_index):
        if first_issue_step is not None and step.step_index > first_issue_step:
            break
        existing_grounded = grounded_by_step.get(step.step_index)
        existing_validation = validation_by_step.get(step.step_index)
        if existing_grounded is not None and existing_validation is not None:
            augmented_grounded_steps.append(existing_grounded)
            augmented_validation_steps.append(existing_validation)
            current_state = list(existing_validation.state_after)
            continue

        taxonomy = taxonomy_by_step.get(step.step_index)
        if taxonomy is None or taxonomy.action_category not in {"active_observation", "observation"}:
            continue

        grounded_step = GroundedTrajectoryStep(
            episode_name=step.episode_name,
            step_index=step.step_index,
            raw_action_text=step.action_text,
            action_category=taxonomy.action_category,
            canonical_action_name=taxonomy.canonical_action_name,
            ground_arguments=list(taxonomy.action_arguments),
            ground_action_pddl=(
                _render_ground_action_pddl(taxonomy.canonical_action_name, taxonomy.action_arguments)
                if taxonomy.canonical_action_name
                else None
            ),
            effect_bucket=None,
            delta_add=[],
            delta_del=[],
            success=None,
            observation_text=step.observation_text,
            extra_info=step.extra_info,
        )
        validation_step = ValidationStepReport(
            step_index=step.step_index,
            action_name=taxonomy.canonical_action_name,
            effect_bucket=None,
            status="applied",
            state_before=list(current_state),
            state_after=list(current_state),
            failed_preconditions=[],
        )
        augmented_grounded_steps.append(grounded_step)
        augmented_validation_steps.append(validation_step)
        inserted_any = True

    if not inserted_any:
        return result

    return ProblemGroundingResult(
        problem_spec=result.problem_spec,
        problem_pddl=result.problem_pddl,
        grounded_steps=augmented_grounded_steps,
        validation_steps=augmented_validation_steps,
        validation_issues=list(result.validation_issues),
        goal_satisfied=result.goal_satisfied,
        object_init_raw_llm_outputs=dict(result.object_init_raw_llm_outputs),
        goal_inference_raw_output=result.goal_inference_raw_output,
    )


@dataclass
class ManipulationIntegratedProblemGroundingLearner:
    max_workers: int = 1

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {self.max_workers}")

    def learn_from_files(
        self,
        *,
        domain_file: str | Path,
        episode_file: str | Path,
        domain_learning_dir: str | Path,
        review_guidance: dict[str, object] | None = None,
    ) -> ProblemGroundingResult:
        del review_guidance
        episode_steps = _load_episode_steps_from_file(episode_file)
        if not episode_steps:
            raise ValueError(f"No steps found in episode file {episode_file}")
        episode_name = episode_steps[0].episode_name
        stored_row = _load_stored_episode_grounding_row(domain_learning_dir, episode_name=episode_name)
        if stored_row is not None:
            base_result = _problem_grounding_result_from_summary_row(stored_row)
            taxonomy_records = _load_taxonomy_records(domain_learning_dir, episode_name=episode_name)
            manipulation_records = [
                item
                for item in load_manipulation_records(Path(domain_learning_dir) / "manipulation_records.jsonl")
                if item.episode_name == episode_name
            ]
            grounded_result = self._ground_episode(
                domain_file=domain_file,
                problem_spec=base_result.problem_spec,
                episode_steps=episode_steps,
                taxonomy_records=taxonomy_records,
                manipulation_records=manipulation_records,
                object_init_raw_llm_outputs=base_result.object_init_raw_llm_outputs,
                goal_inference_raw_output=base_result.goal_inference_raw_output,
            )
            return _augment_with_non_state_changing_observation_steps(
                result=grounded_result,
                episode_steps=episode_steps,
                taxonomy_records=taxonomy_records,
            )
        raise ValueError(
            f"No stored episode grounding result was found for {episode_name} under {domain_learning_dir}. "
            "Please rerun manipulation_domain_learning so final episode grounding artifacts are available."
        )

    def replay_with_existing_problem(
        self,
        *,
        domain_file: str | Path,
        episode_file: str | Path,
        domain_learning_dir: str | Path,
        base_result: ProblemGroundingResult,
        review_guidance: dict[str, object] | None = None,
    ) -> ProblemGroundingResult:
        del review_guidance
        episode_steps = _load_episode_steps_from_file(episode_file)
        if not episode_steps:
            raise ValueError(f"No steps found in episode file {episode_file}")
        episode_name = episode_steps[0].episode_name
        stored_row = _load_stored_episode_grounding_row(domain_learning_dir, episode_name=episode_name)
        if stored_row is not None:
            base_result = _problem_grounding_result_from_summary_row(stored_row)
        else:
            base_result = base_result
        taxonomy_records = _load_taxonomy_records(domain_learning_dir, episode_name=episode_name)
        manipulation_records = [
            item
            for item in load_manipulation_records(Path(domain_learning_dir) / "manipulation_records.jsonl")
            if item.episode_name == episode_name
        ]
        grounded_result = self._ground_episode(
            domain_file=domain_file,
            problem_spec=base_result.problem_spec,
            episode_steps=episode_steps,
            taxonomy_records=taxonomy_records,
            manipulation_records=manipulation_records,
            object_init_raw_llm_outputs=base_result.object_init_raw_llm_outputs,
            goal_inference_raw_output=base_result.goal_inference_raw_output,
        )
        return _augment_with_non_state_changing_observation_steps(
            result=grounded_result,
            episode_steps=episode_steps,
            taxonomy_records=taxonomy_records,
        )

    def write_outputs(self, result: ProblemGroundingResult, output_dir: str | Path) -> None:
        write_problem_grounding_outputs(result, output_dir)

    def _ground_episode(
        self,
        *,
        domain_file: str | Path,
        problem_spec,
        episode_steps: list[RawTrajectoryStep],
        taxonomy_records: list[ActionTaxonomyRecord],
        manipulation_records,
        object_init_raw_llm_outputs: dict[str, str],
        goal_inference_raw_output: str | None,
    ) -> ProblemGroundingResult:
        parsed_domain = parse_domain(Path(domain_file).read_text(encoding="utf-8"))
        parsed_problem = parse_problem(render_problem_pddl(problem_spec))
        object_names = set(parsed_problem.objects)
        action_map = {schema.action.name: schema for schema in parsed_domain.actions}
        taxonomy_by_step = {item.step_index: item for item in taxonomy_records}
        record_by_step = {item.step_index: item for item in manipulation_records}
        state = {predicate for predicate, value in parsed_problem.init_state.items() if value}
        state = {
            f"{predicate.name}({','.join(predicate.params)})" if predicate.params else f"{predicate.name}()"
            for predicate in state
        }
        grounded_steps: list[GroundedTrajectoryStep] = []
        validation_steps: list[ValidationStepReport] = []
        validation_issues: list[ValidationIssue] = []

        for step in sorted((item for item in episode_steps if item.step_index > 0), key=lambda item: item.step_index):
            taxonomy = taxonomy_by_step.get(step.step_index)
            record = record_by_step.get(step.step_index)
            if record is not None:
                grounded_step = GroundedTrajectoryStep(
                    episode_name=record.episode_name,
                    step_index=record.step_index,
                    raw_action_text=record.raw_action_text,
                    action_category="manipulation",
                    canonical_action_name=record.canonical_action_name,
                    ground_arguments=list(record.action_arguments),
                    ground_action_pddl=_render_ground_action_pddl(
                        record.canonical_action_name, record.action_arguments
                    ),
                    effect_bucket=record.effect_bucket,
                    delta_add=list(record.delta_add),
                    delta_del=list(record.delta_del),
                    success=record.success,
                    observation_text=record.post_observation_text,
                    extra_info=record.extra_info,
                )
            else:
                grounded_step = GroundedTrajectoryStep(
                    episode_name=step.episode_name,
                    step_index=step.step_index,
                    raw_action_text=step.action_text,
                    action_category=taxonomy.action_category if taxonomy is not None else None,
                    canonical_action_name=taxonomy.canonical_action_name if taxonomy is not None else None,
                    ground_arguments=list(taxonomy.action_arguments) if taxonomy is not None else [],
                    ground_action_pddl=(
                        _render_ground_action_pddl(taxonomy.canonical_action_name, taxonomy.action_arguments)
                        if taxonomy is not None and taxonomy.canonical_action_name
                        else None
                    ),
                    effect_bucket=None,
                    delta_add=[],
                    delta_del=[],
                    success=None,
                    observation_text=step.observation_text,
                    extra_info=step.extra_info,
                )
            grounded_steps.append(grounded_step)
            report, issue, next_state = execute_grounded_step(
                parsed_domain=parsed_domain,
                object_names=object_names,
                state=state,
                step=grounded_step,
                action_map=action_map,
                check_preconditions=True,
            )
            validation_steps.append(report)
            if issue is not None:
                validation_issues.append(issue)
                break
            state = next_state

        goal_satisfied = all(_goal_fact_holds(state, goal_fact) for goal_fact in problem_spec.goal_facts)
        return ProblemGroundingResult(
            problem_spec=problem_spec,
            problem_pddl=render_problem_pddl(problem_spec),
            grounded_steps=grounded_steps,
            validation_steps=validation_steps,
            validation_issues=validation_issues,
            goal_satisfied=goal_satisfied,
            object_init_raw_llm_outputs=dict(object_init_raw_llm_outputs),
            goal_inference_raw_output=goal_inference_raw_output,
        )


def build_problem_grounding_runner_from_args(
    args: Namespace,
) -> tuple[ManipulationIntegratedProblemGroundingLearner, ModuleModeSummary]:
    logger.debug("Preparing manipulation-integrated problem grounding from stored manipulation artifacts")
    max_workers = int(getattr(args, "max_workers", 1))
    runner = ManipulationIntegratedProblemGroundingLearner(
        max_workers=max_workers,
    )
    return runner, ModuleModeSummary(
        object_init_module="reused_from_manipulation_episode_grounding",
        goal_inference_module="reused_from_manipulation_episode_grounding",
        assembly_module="reused_from_manipulation_episode_grounding",
        trajectory_grounding_module="reused_from_manipulation_episode_grounding",
        validator_module="reused_from_manipulation_episode_grounding",
    )
