"""Default policy parser for the supported subset of POMDPDDL."""

from __future__ import annotations

from ..models.default_policy_rule import DefaultPolicyRule
from .domain_parser import _flatten_symbol_tokens, _format_lint_failure, _parse_typed_symbol_sequence
from .schemas import ParsedDefaultPolicy, ParsedDefaultPolicyRuleSchema
from .sexpr import ParseError, SExpr, loads_sexpr


def _parse_default_policy_raw(text: str) -> ParsedDefaultPolicy:
    """Parse a default policy file and extract policy/rule headers.

    Currently supported rule fields:
    - rule name
    - `:parameters`
    """

    roots = loads_sexpr(text)
    if len(roots) != 1 or not isinstance(roots[0], list):
        raise ParseError("Default policy text must contain exactly one top-level `(define ...)` form.")

    root = roots[0]
    if not root or root[0] != "define":
        raise ParseError("Default policy text must start with `(define ...)`.")

    policy_name = _extract_policy_name(root)
    rules = _parse_policy_rules(root)
    return ParsedDefaultPolicy(policy_name=policy_name, rules=rules)


def parse_default_policy(text: str) -> ParsedDefaultPolicy:
    """Lint then parse a default policy text."""

    from ..linter import lint_default_policy_text

    lint_result = lint_default_policy_text(text)
    if not lint_result.ok:
        raise ParseError(_format_lint_failure("Default policy", lint_result))
    return _parse_default_policy_raw(text)


def _extract_policy_name(root: list[SExpr]) -> str:
    for item in root[1:]:
        if isinstance(item, list) and len(item) == 2 and item[0] == "default_policy":
            if not isinstance(item[1], str):
                raise ParseError("Default policy name must be a symbol.")
            return item[1]
    raise ParseError("Default policy header `(default_policy <name>)` not found.")


def _parse_policy_rules(root: list[SExpr]) -> list[ParsedDefaultPolicyRuleSchema]:
    rules: list[ParsedDefaultPolicyRuleSchema] = []
    for item in root[1:]:
        if not isinstance(item, list) or not item or item[0] != ":policy":
            continue
        rules.append(_parse_policy_rule(item))
    return rules


def _parse_policy_rule(section: list[SExpr]) -> ParsedDefaultPolicyRuleSchema:
    if len(section) < 2 or not isinstance(section[1], str):
        raise ParseError("Policy rule block must have the form `(:policy <name> ...)`.")

    name = section[1]
    params: list[str] = []
    parameter_types: list[tuple[str, str]] = []
    precondition: SExpr | None = None
    action_expr: SExpr | None = None

    for index, item in enumerate(section[2:], start=2):
        if item == ":parameters":
            if index + 1 >= len(section) or not isinstance(section[index + 1], list):
                raise ParseError(f"Policy rule `{name}` has malformed `:parameters`.")
            parameter_types = _parse_typed_symbol_sequence(section[index + 1], default_type="object")
            params = [param_name for param_name, _type_name in parameter_types]
            continue
        if item == ":precondition":
            if index + 1 >= len(section):
                raise ParseError(f"Policy rule `{name}` has malformed `:precondition`.")
            precondition = section[index + 1]
            continue
        if item == ":action":
            if index + 1 >= len(section):
                raise ParseError(f"Policy rule `{name}` has malformed `:action`.")
            action_expr = section[index + 1]
            continue

    return ParsedDefaultPolicyRuleSchema(
        rule=DefaultPolicyRule(name=name, params=params),
        parameter_types=parameter_types,
        precondition=precondition,
        action_expr=action_expr,
    )


def _extract_param_names(tokens: list[SExpr]) -> list[str]:
    raw_tokens = _flatten_symbol_tokens(tokens)
    declarations = _parse_typed_symbol_sequence(raw_tokens, default_type="object")
    return [name for name, _type_name in declarations]
