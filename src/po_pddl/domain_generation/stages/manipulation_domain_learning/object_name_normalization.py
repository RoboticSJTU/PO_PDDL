from __future__ import annotations

import re

from po_pddl.domain_generation.infrastructure.fact_utils import (
    format_symbolic_literal,
    parse_symbolic_literal,
    try_parse_symbolic_literal,
)

_DIRECTIONAL_SUFFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?P<base>.+?)_(?:on_the_)?(?:left|right|front|back|near|far)$"),
    re.compile(r"^(?P<base>.+?)_(?:left|right|front|back)_side$"),
    re.compile(r"^(?P<base>.+?)_(?:near|far)_gripper$"),
)


def coarsen_object_identifier(name: str) -> str:
    stripped = str(name).strip()
    if not stripped or stripped.startswith("?"):
        return stripped
    current = stripped
    while True:
        next_value = current
        for pattern in _DIRECTIONAL_SUFFIX_PATTERNS:
            match = pattern.fullmatch(current)
            if match is not None:
                next_value = match.group("base")
                break
        if next_value == current:
            return current
        current = next_value


def coarsen_object_identifiers(values: list[str]) -> list[str]:
    return [coarsen_object_identifier(value) for value in values]


_BARE_NEGATED_LITERAL_PATTERN = re.compile(r"^\s*not\s+([a-z][a-z0-9_]*)\s*$")
_BARE_POSITIVE_LITERAL_PATTERN = re.compile(r"^\s*([a-z][a-z0-9_]*)\s*$")


def _normalize_bare_zero_arity_literal(text: str) -> str:
    stripped = str(text).strip()
    parsed = try_parse_symbolic_literal(stripped)
    if parsed is not None:
        return stripped

    negated_match = _BARE_NEGATED_LITERAL_PATTERN.fullmatch(stripped)
    if negated_match is not None:
        return format_symbolic_literal(negated_match.group(1), [], negated=True)

    positive_match = _BARE_POSITIVE_LITERAL_PATTERN.fullmatch(stripped)
    if positive_match is not None:
        return format_symbolic_literal(positive_match.group(1), [])

    return stripped


def coarsen_symbolic_literal_arguments(text: str) -> str:
    negated, predicate, arguments = parse_symbolic_literal(_normalize_bare_zero_arity_literal(text))
    return format_symbolic_literal(
        predicate,
        [coarsen_object_identifier(argument) for argument in arguments],
        negated=negated,
    )


def coarsen_symbolic_literal_list(values: list[str]) -> list[str]:
    return [coarsen_symbolic_literal_arguments(value) for value in values]
