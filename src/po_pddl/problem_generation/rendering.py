from __future__ import annotations

from collections import defaultdict

from ..core.models.belief_factor import BeliefFactor
from ..core.models.factorized_belief import FactorizedBelief
from ..core.models.predicate import Predicate
from ..core.parser.sexpr import SExpr
from .models import OnlinePlanningProblemSpec


def _render_atom(predicate: Predicate) -> str:
    return predicate.to_pddl_str()


def _render_negated_atom(predicate: Predicate) -> str:
    return f"(not {_render_atom(predicate)})"


def _is_last_action_predicate(predicate: Predicate) -> bool:
    return predicate.name.startswith("last_action_")


def _render_belief_factor_case(factor: BeliefFactor, true_predicates: list[Predicate]) -> str:
    true_set = set(true_predicates)
    assignments: list[str] = []
    for predicate in factor.scope:
        if predicate in true_set:
            assignments.append(_render_atom(predicate))
        else:
            assignments.append(_render_negated_atom(predicate))
    if not assignments:
        return "(and)"
    if len(assignments) == 1:
        return assignments[0]
    return f"(and {' '.join(assignments)})"


def render_init_belief_pddl(init_belief: FactorizedBelief) -> list[str]:
    init_belief.validate()
    lines: list[str] = ["  (:init-belief"]
    for predicate in sorted(init_belief.known_true, key=lambda item: item.to_pddl_str()):
        lines.append(f"    (prob {_render_atom(predicate)} 1.0)")
    for predicate in sorted(init_belief.known_false, key=lambda item: item.to_pddl_str()):
        if _is_last_action_predicate(predicate):
            continue
        lines.append(f"    (prob {_render_atom(predicate)} 0.0)")
    for factor in init_belief.factors:
        if len(factor.scope) == 1:
            predicate = factor.scope[0]
            positive_probability = sum(
                probability for probability, true_predicates in factor.cases if predicate in true_predicates
            )
            lines.append(f"    (prob {_render_atom(predicate)} {positive_probability:g})")
            continue
        for probability, true_predicates in factor.cases:
            case_expr = _render_belief_factor_case(factor, true_predicates)
            lines.append(f"    (joint {probability:g} {case_expr})")
    lines.append("  )")
    return lines


def _render_goal_expr(expr: SExpr, indent: int = 0) -> str:
    if not isinstance(expr, list):
        return str(expr)
    if not expr:
        return "()"
    if all(not isinstance(item, list) for item in expr):
        return "(" + " ".join(str(item) for item in expr) + ")"

    head = str(expr[0])
    child_indent = indent + 2
    pieces = ["(" + head]
    for item in expr[1:]:
        rendered = _render_goal_expr(item, child_indent)
        pieces.append(" " * child_indent + rendered)
    pieces.append(" " * indent + ")")
    return "\n".join(pieces)


def render_online_problem_pddl(problem_spec: OnlinePlanningProblemSpec) -> str:
    grouped_objects: dict[str, list[str]] = defaultdict(list)
    for item in sorted(problem_spec.objects, key=lambda obj: (obj.type_name, obj.name)):
        grouped_objects[item.type_name].append(item.name)

    lines = [
        f"(define (problem {problem_spec.problem_name})",
        f"  (:domain {problem_spec.domain_name})",
    ]
    if grouped_objects:
        lines.append("  (:objects")
        for type_name, names in sorted(grouped_objects.items()):
            lines.append(f"    {' '.join(names)} - {type_name}")
        lines.append("  )")

    lines.append("  (:init")
    for predicate, value in sorted(problem_spec.init_state.items(), key=lambda item: item[0].to_pddl_str()):
        if value:
            lines.append(f"    {_render_atom(predicate)}")
    lines.append("  )")

    lines.extend(render_init_belief_pddl(problem_spec.init_belief))

    lines.append("  (:goal")
    if problem_spec.goal_expr is None:
        lines.append("    (and)")
    else:
        rendered_goal = _render_goal_expr(problem_spec.goal_expr, indent=4)
        lines.extend(f"    {line}" if index == 0 else line for index, line in enumerate(rendered_goal.splitlines()))
    lines.append("  )")
    lines.append("  (:metric maximize (total-reward))")
    lines.append(")")
    return "\n".join(lines) + "\n"
