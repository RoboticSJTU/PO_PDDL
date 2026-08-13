from __future__ import annotations

from .models import InitObservationRuleSchema


def render_init_observation_module(
    schemas: list[InitObservationRuleSchema],
) -> str:
    if not schemas:
        return ";; No init observation rules were learned.\n"

    lines: list[str] = []
    lines.append(";; Init observation domain module learned from initial scene descriptions")
    lines.append("(:observables")
    lines.append("  (obs-nothing)")
    seen_observables: set[tuple[str, tuple[str, ...]]] = set()
    for schema in schemas:
        observable_signature = (schema.observable_name, tuple(schema.parameter_types))
        if observable_signature not in seen_observables:
            seen_observables.add(observable_signature)
            lines.append("  " + _render_predicate_signature(schema.observable_name, schema.parameter_types))
    lines.append(")")
    lines.append("")
    for schema in schemas:
        lines.extend(_render_rule_pair(schema))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_predicate_signature(predicate_name: str, parameter_types: list[str]) -> str:
    parameters = [f"?arg{index} - {type_name}" for index, type_name in enumerate(parameter_types)]
    return f"({predicate_name} {' '.join(parameters)})".rstrip()


def _render_rule_pair(schema: InitObservationRuleSchema) -> list[str]:
    lines = [f";; init observation / predicate={schema.predicate_name}"]
    lines.extend(_render_single_rule(schema, truth_value=True))
    lines.extend(_render_single_rule(schema, truth_value=False))
    return lines


def _render_single_rule(schema: InitObservationRuleSchema, *, truth_value: bool) -> list[str]:
    rule_name = schema.true_rule_name if truth_value else schema.false_rule_name
    condition_literals = []
    target_literal = schema.target_literal_template
    if not truth_value:
        target_literal = f"not {target_literal}"
    condition_literals.append(target_literal)
    condition_literals.extend(schema.condition_literals)
    condition_literals.extend(schema.last_action_false_conditions)
    if truth_value:
        prob_true = schema.prob_observable_true_given_ground_truth_true
        prob_false = schema.prob_observable_false_given_ground_truth_true
    else:
        prob_true = schema.prob_observable_true_given_ground_truth_false
        prob_false = schema.prob_observable_false_given_ground_truth_false
    lines = [
        f"(:observation {rule_name}",
        f"  :parameters ({_render_rule_parameters(schema)})",
        "  :condition (and " + " ".join(_condition_expr_to_pddl(item) for item in condition_literals) + ")",
        "  :distribution (probabilistic",
        f"      {prob_true:.6f} {_condition_expr_to_pddl(_lifted_observable_literal(schema, observed_true=True))}",
        f"      {prob_false:.6f} {_condition_expr_to_pddl(_lifted_observable_literal(schema, observed_true=False))}",
        "    )",
        ")",
    ]
    return lines


def _render_rule_parameters(schema: InitObservationRuleSchema) -> str:
    return " ".join(f"?obs{index} - {type_name}" for index, type_name in enumerate(schema.parameter_types))


def _lifted_observable_literal(schema: InitObservationRuleSchema, *, observed_true: bool) -> str:
    positive_literal = schema.target_literal_template.replace(f"{schema.predicate_name}(", f"{schema.observable_name}(")
    if observed_true:
        return positive_literal
    return f"not {positive_literal}"


def _condition_expr_to_pddl(text: str) -> str:
    literal = str(text).strip()
    if literal.startswith("("):
        return literal
    if literal.startswith("not "):
        return f"(not {_condition_expr_to_pddl(literal[4:])})"
    if "(" not in literal or not literal.endswith(")"):
        return literal
    predicate, rest = literal.split("(", 1)
    arguments = [item.strip() for item in rest[:-1].split(",") if item.strip()]
    if arguments:
        return f"({predicate} {' '.join(arguments)})"
    return f"({predicate})"
