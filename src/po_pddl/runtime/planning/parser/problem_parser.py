"""Problem parser for the supported subset of POMDPDDL."""

from __future__ import annotations

from .belief_parser import parse_init_belief
from .domain_parser import (
    _find_section,
    _flatten_symbol_tokens,
    _format_lint_failure,
    _parse_typed_symbol_sequence,
)
from .schemas import ParsedProblem
from .sexpr import ParseError, SExpr, loads_sexpr
from ..data_structures.aliases import StateEntry
from ..data_structures.predicate import Predicate


def _parse_problem_raw(text: str) -> ParsedProblem:
    """Parse a problem text and extract supported problem sections."""

    roots = loads_sexpr(text)
    if len(roots) != 1 or not isinstance(roots[0], list):
        raise ParseError("Problem text must contain exactly one top-level `(define ...)` form.")

    root = roots[0]
    if not root or root[0] != "define":
        raise ParseError("Problem text must start with `(define ...)`.")

    problem_name = _extract_problem_name(root)
    domain_name = _extract_problem_domain_name(root)
    objects = _parse_objects(_find_section(root, ":objects"))
    init_state, other_init_items = _parse_init(_find_section(root, ":init"))
    goal = _parse_goal(_find_section(root, ":goal"))
    init_belief = parse_init_belief(_find_section(root, ":init-belief"))
    maximize_reward, metric_expr, metric_target_function = _parse_metric(_find_section(root, ":metric"))
    goal_reward = _parse_goal_reward(_find_section(root, ":goal-reward"))

    return ParsedProblem(
        problem_name=problem_name,
        domain_name=domain_name,
        objects=objects,
        init_state=init_state,
        other_init_items=other_init_items,
        goal=goal,
        init_belief=init_belief,
        maximize_reward=maximize_reward,
        metric_expr=metric_expr,
        metric_target_function=metric_target_function,
        goal_reward=goal_reward,
    )


def parse_problem(text: str) -> ParsedProblem:
    """Lint then parse a problem text."""

    from ..linter import lint_problem_text

    lint_result = lint_problem_text(text)
    if not lint_result.ok:
        raise ParseError(_format_lint_failure("Problem", lint_result))
    return _parse_problem_raw(text)


def _extract_problem_name(root: list[SExpr]) -> str:
    for item in root[1:]:
        if isinstance(item, list) and len(item) == 2 and item[0] == "problem":
            if not isinstance(item[1], str):
                raise ParseError("Problem name must be a symbol.")
            return item[1]
    raise ParseError("Problem header `(problem <name>)` not found.")


def _extract_problem_domain_name(root: list[SExpr]) -> str | None:
    for item in root[1:]:
        if isinstance(item, list) and len(item) == 2 and item[0] == ":domain":
            if not isinstance(item[1], str):
                raise ParseError("Problem domain reference must be a symbol.")
            return item[1]
    return None


def _parse_objects(section: list[SExpr] | None) -> dict[str, str]:
    if section is None:
        return {}

    declarations = _parse_typed_symbol_sequence(section[1:], default_type="object")
    return {name: type_name for name, type_name in declarations}


def _parse_init(section: list[SExpr] | None) -> tuple[StateEntry, list[SExpr]]:
    if section is None:
        return {}, []

    state_entries: StateEntry = {}
    other_items: list[SExpr] = []

    for item in section[1:]:
        parsed = _try_parse_init_predicate(item)
        if parsed is None:
            other_items.append(item)
        else:
            predicate, value = parsed
            state_entries[predicate] = value

    return state_entries, other_items


def _try_parse_init_predicate(item: SExpr) -> StateEntry | None:
    if not isinstance(item, list) or not item:
        return None

    if item[0] == "not":
        if len(item) != 2 or not isinstance(item[1], list) or not item[1]:
            raise ParseError("Malformed negated init predicate.")
        predicate = _parse_ground_predicate(item[1])
        return (predicate, False)

    if isinstance(item[0], str) and not str(item[0]).startswith(":") and item[0] != "=":
        predicate = _parse_ground_predicate(item)
        return (predicate, True)

    return None


def _parse_ground_predicate(item: list[SExpr]) -> Predicate:
    head = item[0]
    if not isinstance(head, str):
        raise ParseError("Predicate head must be a symbol.")
    params = _flatten_symbol_tokens(item[1:])
    return Predicate(head, params)


def _parse_goal(section: list[SExpr] | None) -> SExpr | None:
    if section is None:
        return None
    if len(section) != 2:
        raise ParseError("Goal section must have the form `(:goal <expr>)`.")
    return section[1]


def _parse_metric(section: list[SExpr] | None) -> tuple[bool, SExpr | None, str | None]:
    if section is None:
        return True, None, None
    if len(section) != 3:
        raise ParseError("Metric section must have the form `(:metric <maximize|minimize> <expr>)`.")
    direction = section[1]
    metric_expr = section[2]
    metric_target = _extract_metric_target(metric_expr)
    if direction == "maximize":
        return True, metric_expr, metric_target
    if direction == "minimize":
        return False, metric_expr, metric_target
    raise ParseError("Metric direction must be either `maximize` or `minimize`.")


def _extract_metric_target(expr: SExpr) -> str | None:
    if isinstance(expr, str):
        return expr
    if not expr:
        return None
    head = expr[0]
    if isinstance(head, str):
        return head
    return None


def _parse_goal_reward(section: list[SExpr] | None) -> float:
    if section is None:
        return 0.0
    if len(section) != 2:
        raise ParseError("Goal-reward section must have the form `(:goal-reward <number>)`.")
    value = section[1]
    if not isinstance(value, str):
        raise ParseError("Goal-reward value must be a numeric literal.")
    try:
        return float(value)
    except ValueError as exc:
        raise ParseError(f"Invalid goal-reward literal `{value}`.") from exc
