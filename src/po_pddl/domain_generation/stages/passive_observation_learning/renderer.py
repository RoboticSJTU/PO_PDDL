from __future__ import annotations

from .models import PassiveObservationRuleSchema


def render_passive_observation_module(
    schemas: list[PassiveObservationRuleSchema],
) -> str:
    if not schemas:
        return ";; No passive observation rules were learned.\n"

    lines: list[str] = []
    lines.append(";; Passive observation domain module learned from manipulation effect variants")
    lines.append("(:observables")
    lines.append("  (obs-nothing)")
    seen_observables: set[tuple[str, tuple[str, ...]]] = set()
    for schema in schemas:
        observable_signature = (schema.observable_name, tuple(schema.target_predicate_parameter_types))
        if observable_signature not in seen_observables:
            seen_observables.add(observable_signature)
            lines.append(
                "  "
                + _render_predicate_signature(
                    schema.observable_name,
                    schema.target_predicate_parameter_types,
                    first_parameter_prefix="obs",
                )
            )
    lines.append(")")
    lines.append("")

    for schema in schemas:
        lines.extend(_render_rule_pair(schema))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_predicate_signature(predicate_name: str, parameter_types: list[str], *, first_parameter_prefix: str) -> str:
    parameters: list[str] = []
    for index, type_name in enumerate(parameter_types):
        if index == 0 and type_name == "last_action_marker":
            parameter_name = f"?{first_parameter_prefix}"
        elif index < 4:
            parameter_name = f"?arg{index if first_parameter_prefix == 'last' else index}"
        else:
            parameter_name = f"?p{index}"
        parameters.append(f"{parameter_name} - {type_name}")
    return f"({predicate_name} {' '.join(parameters)})".rstrip()


def _render_rule_pair(schema: PassiveObservationRuleSchema) -> list[str]:
    lines: list[str] = []
    lines.append(f";; {schema.canonical_action_name} / {schema.effect_bucket} / predicate={schema.predicate_name}")
    lines.extend(_render_single_rule(schema, truth_value=True))
    lines.extend(_render_single_rule(schema, truth_value=False))
    return lines


def _render_single_rule(schema: PassiveObservationRuleSchema, *, truth_value: bool) -> list[str]:
    rule_name = schema.true_rule_name if truth_value else schema.false_rule_name
    parameters = _render_rule_parameters(schema)
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
    positive_observable_literal = _lifted_observable_literal(schema, observed_true=True)
    negative_observable_literal = _lifted_observable_literal(schema, observed_true=False)
    lines = [
        f"(:observation {rule_name}",
        f"  :parameters ({parameters})",
        "  :condition (and " + " ".join(_literal_to_pddl(item) for item in condition_literals) + ")",
        "  :distribution (probabilistic",
        f"      {prob_true:.6f} {_literal_to_pddl(positive_observable_literal)}",
        f"      {prob_false:.6f} {_literal_to_pddl(negative_observable_literal)}",
        "    )",
        ")",
    ]
    return lines


def _render_rule_parameters(schema: PassiveObservationRuleSchema) -> str:
    parameters: list[str] = []
    for index, type_name in enumerate(schema.action_argument_types):
        parameters.append(f"?arg{index} - {type_name}")
    for index, type_name in enumerate(schema.extra_argument_types):
        parameters.append(f"?obs{index} - {type_name}")
    return " ".join(parameters)


def _render_action_argument_terms(schema: PassiveObservationRuleSchema) -> str:
    return " ".join(f"?arg{index}" for index, _ in enumerate(schema.action_argument_types))


def _lifted_observable_literal(schema: PassiveObservationRuleSchema, *, observed_true: bool) -> str:
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
