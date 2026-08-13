"""Parser for the supported subset of `:init-belief`."""

from __future__ import annotations

from .sexpr import ParseError, SExpr
from ..data_structures.belief_factor import BeliefFactor
from ..data_structures.factorized_belief import FactorizedBelief
from ..data_structures.predicate import Predicate


def parse_init_belief(section: list[SExpr] | None) -> FactorizedBelief:
    """Parse a supported `:init-belief` section into a factorized belief.

    Currently supported top-level items:
    - `(pred ...)`
    - `(not (pred ...))`
    - `(prob (pred ...) p)`
    - `(joint p <assignment-expr>)`
    - `(oneof <assignment-expr> ...)`

    Supported assignment expressions:
    - `(pred ...)`
    - `(not (pred ...))`
    - `(and <assignment-expr> ...)`
    """

    if section is None:
        return FactorizedBelief()

    known_true: list[Predicate] = []
    known_false: list[Predicate] = []
    factors: list[BeliefFactor] = []

    pending_joint_scope: list[Predicate] | None = None
    pending_joint_cases: list[tuple[float, list[Predicate]]] = []
    factor_index = 0

    def flush_joint_group() -> None:
        nonlocal pending_joint_scope, pending_joint_cases, factor_index
        if pending_joint_scope is None:
            return
        factors.append(
            BeliefFactor(
                name=f"joint_factor_{factor_index}",
                scope=list(pending_joint_scope),
                cases=list(pending_joint_cases),
            )
        )
        factor_index += 1
        pending_joint_scope = None
        pending_joint_cases = []

    for item in section[1:]:
        if not isinstance(item, list) or not item:
            raise ParseError("Malformed item in `:init-belief`.")

        head = item[0]
        if not isinstance(head, str):
            raise ParseError("Malformed `:init-belief` item head.")

        if head == "joint":
            probability, true_predicates, scope = _parse_joint_item(item)
            if pending_joint_scope is None or set(pending_joint_scope) != set(scope):
                flush_joint_group()
                pending_joint_scope = list(scope)
            pending_joint_cases.append((probability, true_predicates))
            continue

        flush_joint_group()

        if head == "prob":
            predicate, probability = _parse_prob_item(item)
            factors.append(
                BeliefFactor(
                    name=f"prob_factor_{factor_index}",
                    scope=[predicate],
                    cases=[
                        (probability, [predicate]),
                        (1.0 - probability, []),
                    ],
                )
            )
            factor_index += 1
            continue

        if head == "oneof":
            factors.append(_parse_oneof_item(item, factor_index))
            factor_index += 1
            continue

        if head == "not":
            known_false.append(_parse_negated_predicate(item))
            continue

        known_true.append(_parse_ground_predicate(item))

    flush_joint_group()

    belief = FactorizedBelief(
        known_true=known_true,
        known_false=known_false,
        factors=factors,
    )
    belief.validate()
    return belief


def _parse_prob_item(item: list[SExpr]) -> tuple[Predicate, float]:
    if len(item) != 3:
        raise ParseError("`prob` belief item must have the form `(prob (<pred>) <p>)`.")
    pred_expr = item[1]
    if not isinstance(pred_expr, list) or not pred_expr:
        raise ParseError("`prob` belief item must target a predicate.")
    probability = _parse_probability(item[2])
    return _parse_ground_predicate(pred_expr), probability


def _parse_joint_item(item: list[SExpr]) -> tuple[float, list[Predicate], list[Predicate]]:
    if len(item) != 3:
        raise ParseError("`joint` belief item must have the form `(joint <p> <expr>)`.")
    probability = _parse_probability(item[1])
    true_predicates, false_predicates = _parse_assignment_expr(item[2])
    scope = list(dict.fromkeys(true_predicates + false_predicates))
    return probability, true_predicates, scope


def _parse_oneof_item(item: list[SExpr], factor_index: int) -> BeliefFactor:
    if len(item) < 2:
        raise ParseError("`oneof` belief item must contain at least one branch.")

    branch_count = len(item) - 1
    branch_probability = 1.0 / branch_count
    parsed_branches: list[tuple[list[Predicate], list[Predicate]]] = []
    scope: list[Predicate] = []

    for branch in item[1:]:
        true_predicates, false_predicates = _parse_assignment_expr(branch)
        parsed_branches.append((true_predicates, false_predicates))
        for predicate in true_predicates + false_predicates:
            if predicate not in scope:
                scope.append(predicate)

    cases: list[tuple[float, list[Predicate]]] = []
    for true_predicates, _false_predicates in parsed_branches:
        cases.append((branch_probability, true_predicates))

    return BeliefFactor(
        name=f"oneof_factor_{factor_index}",
        scope=scope,
        cases=cases,
    )


def _parse_assignment_expr(expr: SExpr) -> tuple[list[Predicate], list[Predicate]]:
    if not isinstance(expr, list) or not expr:
        raise ParseError("Malformed belief assignment expression.")

    head = expr[0]
    if not isinstance(head, str):
        raise ParseError("Malformed belief assignment expression head.")

    if head == "and":
        true_predicates: list[Predicate] = []
        false_predicates: list[Predicate] = []
        for subexpr in expr[1:]:
            sub_true, sub_false = _parse_assignment_expr(subexpr)
            true_predicates.extend(sub_true)
            false_predicates.extend(sub_false)
        return _dedupe_predicates(true_predicates), _dedupe_predicates(false_predicates)

    if head == "not":
        return [], [_parse_negated_predicate(expr)]

    return [_parse_ground_predicate(expr)], []


def _parse_negated_predicate(expr: list[SExpr]) -> Predicate:
    if len(expr) != 2 or not isinstance(expr[1], list) or not expr[1]:
        raise ParseError("Malformed negated belief predicate.")
    return _parse_ground_predicate(expr[1])


def _parse_ground_predicate(item: list[SExpr]) -> Predicate:
    head = item[0]
    if not isinstance(head, str):
        raise ParseError("Predicate head must be a symbol.")
    params: list[str] = []
    for token in item[1:]:
        if not isinstance(token, str):
            raise ParseError("Belief predicates must be grounded flat symbols.")
        params.append(token)
    return Predicate(head, params)


def _parse_probability(value: SExpr) -> float:
    if not isinstance(value, str):
        raise ParseError("Probability value must be a symbol or number literal.")
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            numerator = float(num)
            denominator = float(den)
        except ValueError as exc:
            raise ParseError(f"Invalid rational probability `{value}`.") from exc
        if denominator == 0:
            raise ParseError(f"Invalid probability `{value}` with zero denominator.")
        return numerator / denominator
    try:
        return float(value)
    except ValueError as exc:
        raise ParseError(f"Invalid probability literal `{value}`.") from exc


def _dedupe_predicates(predicates: list[Predicate]) -> list[Predicate]:
    ordered: list[Predicate] = []
    seen: set[Predicate] = set()
    for predicate in predicates:
        if predicate not in seen:
            seen.add(predicate)
            ordered.append(predicate)
    return ordered
