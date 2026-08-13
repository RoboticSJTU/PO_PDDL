from __future__ import annotations

import re
from dataclasses import dataclass

from po_pddl.core.parser import ParsedActionSchema, ParsedDomain, ParsedProblem, SExpr

from .models import GroundedTrajectoryStep, ValidationIssue, ValidationStepReport


@dataclass(frozen=True)
class _Literal:
    negated: bool
    predicate: str
    arguments: list[str]

    def ground(self, binding: dict[str, str]) -> str:
        rendered_arguments = [binding.get(argument, argument) for argument in self.arguments]
        if rendered_arguments:
            body = f"{self.predicate}({','.join(rendered_arguments)})"
        else:
            body = f"{self.predicate}()"
        if self.negated:
            return f"not {body}"
        return body

    def positive_key(self, binding: dict[str, str]) -> str:
        rendered_arguments = [binding.get(argument, argument) for argument in self.arguments]
        if rendered_arguments:
            return f"{self.predicate}({','.join(rendered_arguments)})"
        return f"{self.predicate}()"


def validate_grounded_trajectory(
    parsed_domain: ParsedDomain,
    parsed_problem: ParsedProblem,
    grounded_steps: list[GroundedTrajectoryStep],
    *,
    check_preconditions: bool = True,
) -> tuple[list[ValidationStepReport], list[ValidationIssue], bool]:
    action_map = {schema.action.name: schema for schema in parsed_domain.actions}
    object_names = set(parsed_problem.objects)
    state = build_initial_state(parsed_problem)

    reports: list[ValidationStepReport] = []
    issues: list[ValidationIssue] = []

    for step in grounded_steps:
        report, issue, state = execute_grounded_step(
            parsed_domain=parsed_domain,
            object_names=object_names,
            state=state,
            step=step,
            action_map=action_map,
            check_preconditions=check_preconditions,
        )
        reports.append(report)
        if issue is not None:
            issues.append(issue)
            break

    goal_satisfied = _goal_satisfied(parsed_problem.goal, state)
    return reports, issues, goal_satisfied


def build_initial_state(parsed_problem: ParsedProblem) -> set[str]:
    return {
        _predicate_to_fact(predicate.name, list(predicate.params))
        for predicate, value in parsed_problem.init_state.items()
        if value
    }


def normalize_fact_key(fact: str) -> str:
    """Normalize a symbolic fact before comparing replay state entries."""
    text = str(fact).strip()
    negated = text.lower().startswith("not ")
    if negated:
        text = text[4:].strip()
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_-]*)\s*\((.*)\)", text)
    if match is None:
        return str(fact).strip()
    predicate = match.group(1)
    raw_arguments = match.group(2).strip()
    arguments = [item.strip() for item in raw_arguments.split(",")] if raw_arguments else []
    normalized = _predicate_to_fact(predicate, arguments)
    return f"not {normalized}" if negated else normalized


def execute_grounded_step(
    *,
    parsed_domain: ParsedDomain,
    object_names: set[str],
    state: set[str],
    step: GroundedTrajectoryStep,
    action_map: dict[str, ParsedActionSchema] | None = None,
    check_preconditions: bool = True,
) -> tuple[ValidationStepReport, ValidationIssue | None, set[str]]:
    if action_map is None:
        action_map = {schema.action.name: schema for schema in parsed_domain.actions}
    state = {normalize_fact_key(fact) for fact in state}
    state_before = sorted(state)
    if step.action_category != "manipulation":
        return (
            ValidationStepReport(
                step_index=step.step_index,
                action_name=step.canonical_action_name,
                effect_bucket=step.effect_bucket,
                status="skipped_non_manipulation",
                state_before=state_before,
                state_after=state_before,
            ),
            None,
            set(state),
        )

    action_name = step.canonical_action_name
    if action_name is None:
        issue = ValidationIssue(
            step_index=step.step_index,
            error_code="unknown_action",
            message="No grounded action was available for this step.",
            action_name=None,
            state_before=state_before,
        )
        return (
            ValidationStepReport(
                step_index=step.step_index,
                action_name=None,
                effect_bucket=step.effect_bucket,
                status="unknown_action",
                state_before=state_before,
                state_after=state_before,
            ),
            issue,
            set(state),
        )

    schema = action_map.get(action_name)
    if schema is None:
        issue = ValidationIssue(
            step_index=step.step_index,
            error_code="unknown_action",
            message=f"Action `{action_name}` is not defined in the domain.",
            action_name=action_name,
            state_before=state_before,
        )
        return (
            ValidationStepReport(
                step_index=step.step_index,
                action_name=action_name,
                effect_bucket=step.effect_bucket,
                status="unknown_action",
                state_before=state_before,
                state_after=state_before,
            ),
            issue,
            set(state),
        )

    if step.missing_effect_branch:
        expected_branch = step.branch_expectation
        branch_label = f"{expected_branch} " if expected_branch else ""
        issue = ValidationIssue(
            step_index=step.step_index,
            error_code="missing_effect_branch",
            message=(f"Action `{action_name}` is missing a {branch_label}effect branch needed for this grounded step."),
            action_name=action_name,
            requested_effect_bucket=step.requested_effect_bucket,
            expected_branch=expected_branch,
            available_effect_buckets=list(step.available_effect_buckets),
            state_before=state_before,
        )
        return (
            ValidationStepReport(
                step_index=step.step_index,
                action_name=action_name,
                effect_bucket=step.effect_bucket,
                status="missing_effect_branch",
                state_before=state_before,
                state_after=state_before,
            ),
            issue,
            set(state),
        )

    if len(step.ground_arguments) != len(schema.action.params):
        issue = ValidationIssue(
            step_index=step.step_index,
            error_code="arity_mismatch",
            message=(
                f"Action `{action_name}` expects {len(schema.action.params)} arguments but "
                f"received {len(step.ground_arguments)}."
            ),
            action_name=action_name,
            state_before=state_before,
        )
        return (
            ValidationStepReport(
                step_index=step.step_index,
                action_name=action_name,
                effect_bucket=step.effect_bucket,
                status="arity_mismatch",
                state_before=state_before,
                state_after=state_before,
            ),
            issue,
            set(state),
        )

    unknown_objects = [name for name in step.ground_arguments if name not in object_names]
    if unknown_objects:
        issue = ValidationIssue(
            step_index=step.step_index,
            error_code="unknown_object",
            message=f"Action `{action_name}` references unknown objects: {', '.join(unknown_objects)}.",
            action_name=action_name,
            state_before=state_before,
        )
        return (
            ValidationStepReport(
                step_index=step.step_index,
                action_name=action_name,
                effect_bucket=step.effect_bucket,
                status="unknown_object",
                state_before=state_before,
                state_after=state_before,
            ),
            issue,
            set(state),
        )

    binding = {parameter: argument for parameter, argument in zip(schema.action.params, step.ground_arguments)}
    failed_preconditions = _evaluate_precondition_failures(schema, state, binding) if check_preconditions else []
    if failed_preconditions:
        issue = ValidationIssue(
            step_index=step.step_index,
            error_code="precondition_failed",
            message=f"Action `{action_name}` failed its precondition check.",
            action_name=action_name,
            failed_preconditions=failed_preconditions,
            state_before=state_before,
        )
        return (
            ValidationStepReport(
                step_index=step.step_index,
                action_name=action_name,
                effect_bucket=step.effect_bucket,
                status="precondition_failed",
                state_before=state_before,
                state_after=state_before,
                failed_preconditions=failed_preconditions,
            ),
            issue,
            set(state),
        )

    next_state = set(state)
    for fact in step.delta_del:
        next_state.discard(normalize_fact_key(fact))
    for fact in step.delta_add:
        next_state.add(normalize_fact_key(fact))
    return (
        ValidationStepReport(
            step_index=step.step_index,
            action_name=action_name,
            effect_bucket=step.effect_bucket,
            status="applied",
            state_before=state_before,
            state_after=sorted(next_state),
        ),
        None,
        next_state,
    )


def _predicate_to_fact(name: str, arguments: list[str]) -> str:
    if arguments:
        return f"{name}({','.join(arguments)})"
    return f"{name}()"


def _evaluate_precondition_failures(
    schema: ParsedActionSchema,
    state: set[str],
    binding: dict[str, str],
) -> list[str]:
    literals = _flatten_conjunctive_literals(schema.precondition)
    failures: list[str] = []
    for literal in literals:
        positive_fact = literal.positive_key(binding)
        rendered = literal.ground(binding)
        if literal.negated:
            if positive_fact in state:
                failures.append(rendered)
        else:
            if positive_fact not in state:
                failures.append(rendered)
    return failures


def _flatten_conjunctive_literals(expr: SExpr | None) -> list[_Literal]:
    if expr is None:
        return []
    if isinstance(expr, str):
        return [_Literal(negated=False, predicate=expr, arguments=[])]
    if not expr:
        return []
    head = expr[0]
    if head == "and":
        literals: list[_Literal] = []
        for item in expr[1:]:
            literals.extend(_flatten_conjunctive_literals(item))
        return literals
    if head == "not":
        if len(expr) != 2 or not isinstance(expr[1], list) or not expr[1]:
            raise ValueError(f"Unsupported negated precondition expression: {expr!r}")
        inner_head = expr[1][0]
        if not isinstance(inner_head, str):
            raise ValueError(f"Unsupported negated precondition expression: {expr!r}")
        args = [str(item) for item in expr[1][1:]]
        return [_Literal(negated=True, predicate=inner_head, arguments=args)]
    if isinstance(head, str):
        return [_Literal(negated=False, predicate=head, arguments=[str(item) for item in expr[1:]])]
    raise ValueError(f"Unsupported precondition expression: {expr!r}")


def _goal_satisfied(goal_expr: SExpr | None, state: set[str]) -> bool:
    if goal_expr is None:
        return True
    for literal in _flatten_conjunctive_literals(goal_expr):
        fact = literal.positive_key({})
        if literal.negated:
            if fact in state:
                return False
        else:
            if fact not in state:
                return False
    return True
