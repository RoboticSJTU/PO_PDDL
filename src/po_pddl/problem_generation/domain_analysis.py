from __future__ import annotations

from dataclasses import dataclass

from ..core.codegen.python_model_codegen import build_codegen_plan_from_parsed
from ..core.models.factorized_belief import FactorizedBelief
from ..core.models.predicate import Predicate
from ..core.parser import ParsedDomain, ParsedProblem, parse_domain
from ..domain_generation.stages.problem_grounding.models import ObjectDeclaration


@dataclass(frozen=True)
class DomainAnalysisResult:
    parsed_domain: ParsedDomain
    has_observation_module: bool

    def render_summary(self) -> str:
        predicate_lines = []
        for predicate in self.parsed_domain.predicates:
            parameter_types = self.parsed_domain.predicate_parameter_types.get(predicate, [])
            if parameter_types:
                rendered_params = ", ".join(f"{name}: {type_name}" for name, type_name in parameter_types)
            else:
                rendered_params = "(no parameters)"
            predicate_lines.append(f"- {predicate.name}: {rendered_params}")
        if not predicate_lines:
            predicate_lines.append("- (no predicates declared)")

        action_names = ", ".join(schema.action.name for schema in self.parsed_domain.actions) or "(no actions declared)"
        constant_names = ", ".join(sorted(self.parsed_domain.constants)) or "(no constants)"
        type_names = ", ".join(sorted(self.parsed_domain.types)) or "(no types)"
        return "\n".join(
            [
                f"Domain: {self.parsed_domain.domain_name}",
                f"Types: {type_names}",
                f"Constants: {constant_names}",
                f"Actions: {action_names}",
                "Predicates:",
                *predicate_lines,
                f"Observation module present: {'yes' if self.has_observation_module else 'no'}",
            ]
        )


def analyze_domain(domain_text: str) -> DomainAnalysisResult:
    parsed_domain = parse_domain(domain_text)
    has_observation_module = bool(parsed_domain.observables or parsed_domain.observation_rules)
    return DomainAnalysisResult(
        parsed_domain=parsed_domain,
        has_observation_module=has_observation_module,
    )


def _collect_operational_predicate_type_signatures(
    parsed_domain: ParsedDomain,
) -> dict[str, set[tuple[str, ...]]]:
    predicate_arities = {predicate.name: len(predicate.params) for predicate in parsed_domain.predicates}
    signatures: dict[str, set[tuple[str, ...]]] = {}

    def _visit(expr: object, parameter_types: dict[str, str]) -> None:
        if not isinstance(expr, list) or not expr:
            return
        head = expr[0]
        if isinstance(head, str) and head in predicate_arities and len(expr) - 1 == predicate_arities[head]:
            argument_types: list[str] = []
            for argument in expr[1:]:
                if not isinstance(argument, str):
                    break
                if argument in parameter_types:
                    argument_types.append(parameter_types[argument])
                elif argument in parsed_domain.constants:
                    argument_types.append(parsed_domain.constants[argument])
                else:
                    break
            else:
                signatures.setdefault(head, set()).add(tuple(argument_types))
        for child in expr[1:]:
            _visit(child, parameter_types)

    for action_schema in parsed_domain.actions:
        parameter_types = dict(action_schema.parameter_types)
        _visit(action_schema.precondition, parameter_types)
        _visit(action_schema.effect, parameter_types)
    for observation_schema in parsed_domain.observation_rules:
        parameter_types = dict(observation_schema.parameter_types)
        _visit(observation_schema.condition, parameter_types)
        _visit(observation_schema.distribution_expr, parameter_types)
    return signatures


def _type_matches(
    parsed_domain: ParsedDomain,
    actual_type: str,
    required_type: str,
) -> bool:
    if actual_type == required_type:
        return True
    node = parsed_domain.types.get(actual_type)
    return node is not None and node.is_subtype_of(required_type)


def build_grounded_predicates_for_objects(
    parsed_domain: ParsedDomain,
    objects: list[ObjectDeclaration],
    *,
    problem_name: str | None = None,
) -> list[Predicate]:
    parsed_problem = ParsedProblem(
        problem_name=problem_name or f"{parsed_domain.domain_name}_online_problem",
        domain_name=parsed_domain.domain_name,
        objects={item.name: item.type_name for item in objects},
        init_state={},
        goal=None,
        init_belief=FactorizedBelief(),
    )
    plan = build_codegen_plan_from_parsed(parsed_domain, parsed_problem, None)
    action_marker_constants = {
        name
        for name, type_name in parsed_domain.constants.items()
        if type_name == "last_action_marker"
        or (type_name in parsed_domain.types and parsed_domain.types[type_name].is_subtype_of("last_action_marker"))
    }
    object_types = {
        **parsed_domain.constants,
        **{item.name: item.type_name for item in objects},
    }
    operational_signatures = _collect_operational_predicate_type_signatures(parsed_domain)

    def _matches_operational_signature(predicate: Predicate) -> bool:
        signatures = operational_signatures.get(predicate.name)
        if not signatures:
            return True
        actual_types = [object_types.get(argument) for argument in predicate.params]
        if any(type_name is None for type_name in actual_types):
            return False
        return any(
            len(signature) == len(actual_types)
            and all(
                _type_matches(parsed_domain, actual_type, required_type)
                for actual_type, required_type in zip(actual_types, signature)
                if actual_type is not None
            )
            for signature in signatures
        )

    # Action-history markers are false before the first action and are omitted
    # from rendered online beliefs. They also must not fill broad `object`
    # parameters in ordinary state predicates.
    return [
        predicate
        for predicate in plan.predicates
        if not predicate.name.startswith("last_action_")
        and not action_marker_constants.intersection(predicate.params)
        and _matches_operational_signature(predicate)
    ]
