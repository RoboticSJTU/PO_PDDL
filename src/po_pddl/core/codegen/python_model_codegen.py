"""Build grounded code-generation plans for explicit Python model emission."""

from __future__ import annotations

from itertools import product

from ..models import Action, DefaultPolicyRule, Observable, ObservationRule, Predicate
from ..parser import parse_default_policy, parse_domain, parse_problem
from ..parser.schemas import ParsedDefaultPolicy, ParsedDomain, ParsedProblem
from .schemas import (
    GroundedActionCase,
    GroundedDefaultPolicyRuleCase,
    GroundedObservationRuleCase,
    PythonModelCodegenPlan,
)


def build_codegen_plan_from_texts(
    domain_text: str,
    problem_text: str,
    default_policy_text: str | None = None,
) -> PythonModelCodegenPlan:
    """Parse text inputs and build a grounded plan for explicit Python code generation."""
    parsed_domain = parse_domain(domain_text)
    parsed_problem = parse_problem(problem_text)
    parsed_default_policy = (
        parse_default_policy(default_policy_text)
        if default_policy_text is not None
        else ParsedDefaultPolicy(policy_name="default-policy")
    )
    return build_codegen_plan_from_parsed(parsed_domain, parsed_problem, parsed_default_policy)


def build_codegen_plan_from_parsed(
    parsed_domain: ParsedDomain,
    parsed_problem: ParsedProblem,
    parsed_default_policy: ParsedDefaultPolicy | None = None,
) -> PythonModelCodegenPlan:
    """Ground parser output into a code-generation-oriented plan."""
    if parsed_problem.domain_name is not None and parsed_problem.domain_name != parsed_domain.domain_name:
        raise ValueError(
            "Problem references domain "
            f"`{parsed_problem.domain_name}` but parsed domain is "
            f"`{parsed_domain.domain_name}`."
        )

    default_policy = parsed_default_policy or ParsedDefaultPolicy(policy_name="default-policy")
    universe = dict(parsed_domain.constants)
    universe.update(parsed_problem.objects)

    grounded_observables = _ground_observables(parsed_domain, universe)
    grounded_observable_set = set(grounded_observables)
    grounded_observation_rules = [
        case
        for case in _ground_observation_rules(parsed_domain, universe)
        if all(observable in grounded_observable_set for observable in case.rule.distribution)
    ]

    return PythonModelCodegenPlan(
        domain_name=parsed_domain.domain_name,
        problem_name=parsed_problem.problem_name,
        types=parsed_domain.types,
        constants=dict(parsed_domain.constants),
        objects=dict(parsed_problem.objects),
        predicates=_ground_predicates(parsed_domain, universe),
        observables=grounded_observables,
        grounded_actions=_ground_actions(parsed_domain, universe),
        grounded_observation_rules=grounded_observation_rules,
        grounded_default_policy_rules=_ground_default_policy_rules(default_policy, universe, parsed_domain.types),
        init_belief=parsed_problem.init_belief,
        goal_expr=parsed_problem.goal,
        maximize_reward=parsed_problem.maximize_reward,
        goal_reward=parsed_problem.goal_reward,
    )


def _ground_predicates(parsed_domain: ParsedDomain, universe: dict[str, str]) -> list[Predicate]:
    grounded: list[Predicate] = []
    for predicate in parsed_domain.predicates:
        parameter_types = parsed_domain.predicate_parameter_types.get(predicate, [])
        if not parameter_types:
            grounded.append(Predicate(predicate.name, list(predicate.params)))
            continue
        for bindings in _iter_object_bindings(parameter_types, universe, parsed_domain.types):
            grounded.append(Predicate(predicate.name, [bindings[var_name] for var_name, _ in parameter_types]))
    return grounded


def _ground_observables(parsed_domain: ParsedDomain, universe: dict[str, str]) -> list[Observable]:
    grounded: list[Observable] = []
    for observable in parsed_domain.observables:
        parameter_types = parsed_domain.observable_parameter_types.get(observable, [])
        if not parameter_types:
            grounded.append(Observable(observable.name, list(observable.params)))
            continue
        for bindings in _iter_object_bindings(
            parameter_types,
            universe,
            parsed_domain.types,
            exclude_internal_markers=True,
        ):
            grounded.append(Observable(observable.name, [bindings[var_name] for var_name, _ in parameter_types]))
    return grounded


def _ground_actions(parsed_domain: ParsedDomain, universe: dict[str, str]) -> list[GroundedActionCase]:
    grounded: list[GroundedActionCase] = []
    for schema in parsed_domain.actions:
        if not schema.parameter_types:
            grounded.append(
                GroundedActionCase(
                    action=Action(schema.action.name, list(schema.action.params)),
                    schema=schema,
                    bindings={},
                )
            )
            continue
        for bindings in _iter_object_bindings(schema.parameter_types, universe, parsed_domain.types):
            grounded_action = Action(
                schema.action.name,
                [bindings[var_name] for var_name, _ in schema.parameter_types],
            )
            grounded.append(
                GroundedActionCase(
                    action=grounded_action,
                    schema=schema,
                    bindings=dict(bindings),
                )
            )
    return grounded


def _ground_observation_rules(
    parsed_domain: ParsedDomain,
    universe: dict[str, str],
) -> list[GroundedObservationRuleCase]:
    grounded: list[GroundedObservationRuleCase] = []
    for schema in parsed_domain.observation_rules:
        if not schema.parameter_types:
            grounded.append(
                GroundedObservationRuleCase(
                    rule=ObservationRule(schema.rule.name, list(schema.rule.distribution)),
                    schema=schema,
                    bindings={},
                )
            )
            continue
        for bindings in _iter_object_bindings(schema.parameter_types, universe, parsed_domain.types):
            grounded_distribution = [
                Observable(
                    observable.name,
                    [bindings.get(param, param) for param in observable.params],
                )
                for observable in schema.rule.distribution
            ]
            grounded_rule = ObservationRule(
                _build_grounded_name(schema.rule.name, schema.parameter_types, bindings),
                grounded_distribution,
            )
            grounded.append(
                GroundedObservationRuleCase(
                    rule=grounded_rule,
                    schema=schema,
                    bindings=dict(bindings),
                )
            )
    return grounded


def _ground_default_policy_rules(
    parsed_default_policy: ParsedDefaultPolicy,
    universe: dict[str, str],
    types: dict[str, object],
) -> list[GroundedDefaultPolicyRuleCase]:
    grounded: list[GroundedDefaultPolicyRuleCase] = []
    for schema in parsed_default_policy.rules:
        if not schema.parameter_types:
            grounded.append(
                GroundedDefaultPolicyRuleCase(
                    rule=DefaultPolicyRule(schema.rule.name, list(schema.rule.params)),
                    schema=schema,
                    bindings={},
                )
            )
            continue
        for bindings in _iter_object_bindings(schema.parameter_types, universe, types):
            grounded_rule = DefaultPolicyRule(
                schema.rule.name,
                [bindings[var_name] for var_name, _ in schema.parameter_types],
            )
            grounded.append(
                GroundedDefaultPolicyRuleCase(
                    rule=grounded_rule,
                    schema=schema,
                    bindings=dict(bindings),
                )
            )
    return grounded


def _iter_object_bindings(
    parameter_types: list[tuple[str, str]],
    universe: dict[str, str],
    types: dict[str, object],
    *,
    exclude_internal_markers: bool = False,
):
    domains: list[list[str]] = []
    for _var_name, type_name in parameter_types:
        domains.append(
            _objects_of_type(
                type_name,
                universe,
                types,
                exclude_internal_markers=exclude_internal_markers,
            )
        )
    for values in product(*domains):
        yield {var_name: value for (var_name, _), value in zip(parameter_types, values)}


def _objects_of_type(
    type_name: str,
    universe: dict[str, str],
    types: dict[str, object],
    *,
    exclude_internal_markers: bool = False,
) -> list[str]:
    def is_internal_marker(object_type: str) -> bool:
        return _is_subtype_name(object_type, "last_action_marker", types)

    if type_name == "object":
        return [
            object_name
            for object_name, object_type in universe.items()
            if not exclude_internal_markers or not is_internal_marker(object_type)
        ]
    results: list[str] = []
    for obj_name, obj_type in universe.items():
        if (
            exclude_internal_markers
            and is_internal_marker(obj_type)
            and not _is_subtype_name(type_name, "last_action_marker", types)
        ):
            continue
        if _is_subtype_name(obj_type, type_name, types):
            results.append(obj_name)
    return results


def _is_subtype_name(child_type: str, target_type: str, types: dict[str, object]) -> bool:
    current_name: str | None = child_type
    while current_name is not None:
        if current_name == target_type:
            return True
        node = types.get(current_name)
        if node is None or getattr(node, "parent", None) is None:
            current_name = None
        else:
            current_name = node.parent.name
    return False


def _build_grounded_name(
    base_name: str,
    parameter_types: list[tuple[str, str]],
    bindings: dict[str, str],
) -> str:
    if not parameter_types:
        return base_name
    args = ",".join(bindings[var_name] for var_name, _ in parameter_types)
    return f"{base_name}[{args}]"
