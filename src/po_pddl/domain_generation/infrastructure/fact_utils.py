from __future__ import annotations

import re

_NEGATED_LITERAL_PATTERN = re.compile(r"^\s*not\s+([a-z][a-z0-9_]*)\(([^()]*)\)\s*$")
_POSITIVE_LITERAL_PATTERN = re.compile(r"^\s*([a-z][a-z0-9_]*)\(([^()]*)\)\s*$")


def _split_arguments(raw_args: str) -> list[str]:
    stripped = raw_args.strip()
    if not stripped:
        return []
    return [arg.strip() for arg in stripped.split(",") if arg.strip()]


def try_parse_symbolic_literal(text: str) -> tuple[bool, str, list[str]] | None:
    stripped = text.strip()
    negated_match = _NEGATED_LITERAL_PATTERN.fullmatch(stripped)
    if negated_match:
        return True, negated_match.group(1), _split_arguments(negated_match.group(2))

    positive_match = _POSITIVE_LITERAL_PATTERN.fullmatch(stripped)
    if positive_match:
        return False, positive_match.group(1), _split_arguments(positive_match.group(2))

    return None


def parse_symbolic_literal(text: str) -> tuple[bool, str, list[str]]:
    parsed = try_parse_symbolic_literal(text)
    if parsed is None:
        raise ValueError(f"Unsupported literal syntax: {text!r}")
    return parsed


def parse_positive_symbolic_fact(text: str) -> tuple[str, list[str]]:
    negated, predicate, arguments = parse_symbolic_literal(text)
    if negated:
        raise ValueError(f"Expected positive fact syntax, got negated literal: {text!r}")
    return predicate, arguments


def symbolic_literal_parts(text: str) -> tuple[str, list[str]]:
    _negated, predicate, arguments = parse_symbolic_literal(text)
    return predicate, arguments


def symbolic_literal_predicate(text: str) -> str | None:
    parsed = try_parse_symbolic_literal(text)
    return None if parsed is None else parsed[1]


def symbolic_literal_arguments(text: str) -> list[str]:
    parsed = try_parse_symbolic_literal(text)
    return [] if parsed is None else parsed[2]


def format_symbolic_literal(predicate: str, arguments: list[str], *, negated: bool = False) -> str:
    body = f"{predicate}({','.join(arg for arg in arguments if arg)})" if arguments else f"{predicate}()"
    return f"not {body}" if negated else body


def remap_symbolic_literal_arguments(text: str, argument_mapping: dict[str, str]) -> str:
    negated, predicate, arguments = parse_symbolic_literal(text)
    remapped_arguments = [argument_mapping.get(argument, argument) for argument in arguments if argument]
    return format_symbolic_literal(predicate, remapped_arguments, negated=negated)


def _resolve_canonical_name(name: str, canonical_object_map: dict[str, str]) -> str:
    seen: set[str] = set()
    current = name
    while current in canonical_object_map and current not in seen:
        seen.add(current)
        nxt = canonical_object_map[current]
        if nxt == current:
            break
        current = nxt
    return current


def canonicalize_symbolic_literal_arguments(text: str, canonical_object_map: dict[str, str]) -> str:
    negated, predicate, arguments = parse_symbolic_literal(text)
    canonical_arguments = [
        _resolve_canonical_name(argument, canonical_object_map) for argument in arguments if argument
    ]
    return format_symbolic_literal(predicate, canonical_arguments, negated=negated)


def render_symbolic_literal_to_pddl(text: str, argument_mapping: dict[str, str] | None = None) -> str:
    negated, predicate, arguments = parse_symbolic_literal(text)
    mapping = argument_mapping or {}
    rendered_arguments = [mapping.get(argument, argument) for argument in arguments if argument]
    body = f"({predicate} {' '.join(rendered_arguments)})" if rendered_arguments else f"({predicate})"
    if negated:
        return f"(not {body})"
    return body
