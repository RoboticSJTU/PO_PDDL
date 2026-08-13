from __future__ import annotations

from .models import ActiveObservationRuleSchema


def render_active_observation_module(
    schemas: list[ActiveObservationRuleSchema],
) -> str:
    if not schemas:
        return ";; No active observation rules were learned.\n"

    lines: list[str] = []
    lines.append(";; Active observation domain module learned from active perception actions")
    lines.append("(:observables")
    lines.append("  (obs-nothing)")
    seen_observables: set[tuple[str, tuple[str, ...]]] = set()
    for schema in schemas:
        signature = (schema.observable_name, tuple(schema.target_predicate_parameter_types))
        if signature in seen_observables:
            continue
        seen_observables.add(signature)
        lines.append(
            "  "
            + _render_predicate_signature(
                schema.observable_name,
                schema.target_predicate_parameter_types,
            )
        )
    lines.append(")")
    lines.append("")
    for schema in schemas:
        lines.extend(_render_rule_pair(schema))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_predicate_signature(predicate_name: str, parameter_types: list[str]) -> str:
    parameters = [f"?arg{index} - {type_name}" for index, type_name in enumerate(parameter_types)]
    return f"({predicate_name} {' '.join(parameters)})".rstrip()


def _render_rule_pair(schema: ActiveObservationRuleSchema) -> list[str]:
    lines = [
        f";; {schema.canonical_action_name} / {schema.effect_bucket} / predicate={schema.predicate_name}",
    ]
    lines.extend(_render_single_rule(schema, truth_value=True))
    lines.extend(_render_single_rule(schema, truth_value=False))
    return lines


def _render_single_rule(schema: ActiveObservationRuleSchema, *, truth_value: bool) -> list[str]:
    rule_name = schema.true_rule_name if truth_value else schema.false_rule_name
    target_literal = schema.target_literal_template
    if not truth_value:
        target_literal = f"not {target_literal}"
    condition_literals = [
        f"({schema.last_action_predicate_name} {schema.last_action_constant} {_render_action_argument_terms(schema)})".rstrip(),
        target_literal,
        *schema.condition_literals,
    ]
    if truth_value:
        prob_true = schema.prob_observable_true_given_ground_truth_true
        prob_false = schema.prob_observable_false_given_ground_truth_true
    else:
        prob_true = schema.prob_observable_true_given_ground_truth_false
        prob_false = schema.prob_observable_false_given_ground_truth_false
    lines = [
        f"(:observation {rule_name}",
        f"  :parameters ({_render_rule_parameters(schema)})",
        "  :condition (and " + " ".join(_literal_to_pddl(item) for item in condition_literals) + ")",
        "  :distribution (probabilistic",
        f"      {prob_true:.6f} {_literal_to_pddl(_lifted_observable_literal(schema, observed_true=True))}",
        f"      {prob_false:.6f} {_literal_to_pddl(_lifted_observable_literal(schema, observed_true=False))}",
        "    )",
        ")",
    ]
    return lines


def _render_rule_parameters(schema: ActiveObservationRuleSchema) -> str:
    parameters: list[str] = []
    for index, type_name in enumerate(schema.action_argument_types):
        parameters.append(f"?arg{index} - {type_name}")
    for index, type_name in enumerate(schema.extra_argument_types):
        parameters.append(f"?obs{index} - {type_name}")
    return " ".join(parameters)


def _render_action_argument_terms(schema: ActiveObservationRuleSchema) -> str:
    return " ".join(f"?arg{index}" for index, _ in enumerate(schema.action_argument_types))


def _lifted_observable_literal(schema: ActiveObservationRuleSchema, *, observed_true: bool) -> str:
    positive_literal = schema.target_literal_template.replace(f"{schema.predicate_name}(", f"{schema.observable_name}(")
    if observed_true:
        return positive_literal
    return f"not {positive_literal}"


def _literal_to_pddl(literal: str) -> str:
    literal = str(literal).strip()
    if literal.startswith("not "):
        return f"(not {_literal_to_pddl(literal[4:])})"
    if "(" not in literal or not literal.endswith(")"):
        return literal
    predicate, rest = literal.split("(", 1)
    arguments = [item.strip() for item in rest[:-1].split(",") if item.strip()]
    if arguments:
        return f"({predicate} {' '.join(arguments)})"
    return f"({predicate})"
