from __future__ import annotations

from collections import defaultdict

from po_pddl.domain_generation.infrastructure.fact_utils import render_symbolic_literal_to_pddl as _fact_to_pddl

from .models import ProblemSpec


def render_problem_pddl(problem_spec: ProblemSpec) -> str:
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
    for fact in problem_spec.init_facts:
        lines.append(f"    {_fact_to_pddl(fact)}")
    lines.append("  )")
    lines.append("  (:goal")
    if problem_spec.goal_facts:
        goal_body = " ".join(_fact_to_pddl(fact) for fact in problem_spec.goal_facts)
        lines.append(f"    (and {goal_body})")
    else:
        lines.append("    (and)")
    lines.append("  )")
    lines.append(")")
    return "\n".join(lines) + "\n"
