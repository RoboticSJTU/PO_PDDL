"""Domain parser for the supported subset of POMDPDDL."""

from __future__ import annotations

import re

from ..models.action import Action
from ..models.effect_bucket import ParsedEffectBucketAnnotation
from ..models.observable import Observable
from ..models.observation_rule import ObservationRule
from ..models.predicate import Predicate
from ..models.type_node import TypeNode
from .schemas import ParsedActionSchema, ParsedDomain, ParsedObservationRuleSchema
from .sexpr import ParseError, SExpr, loads_sexpr


def _parse_domain_raw(text: str) -> ParsedDomain:
    """Parse a domain text and extract the currently supported sections."""

    action_effect_bucket_annotations = _extract_action_effect_bucket_annotations(text)
    roots = loads_sexpr(text)
    if len(roots) != 1 or not isinstance(roots[0], list):
        raise ParseError("Domain text must contain exactly one top-level `(define ...)` form.")

    root = roots[0]
    if not root or root[0] != "define":
        raise ParseError("Domain text must start with `(define ...)`.")

    domain_name = _extract_domain_name(root)
    types = _parse_types(_find_section(root, ":types"))
    constants = _parse_constants(_find_section(root, ":constants"))
    functions = _parse_functions(_find_section(root, ":functions"))
    predicates, predicate_parameter_types = _parse_predicate_like_section(_find_section(root, ":predicates"), Predicate)
    observables, observable_parameter_types = _parse_predicate_like_section(
        _find_section(root, ":observables"), Observable
    )
    actions = _parse_actions(root, action_effect_bucket_annotations)
    observation_rules = _parse_observation_rules(root)

    return ParsedDomain(
        domain_name=domain_name,
        types=types,
        constants=constants,
        functions=functions,
        predicates=predicates,
        predicate_parameter_types=predicate_parameter_types,
        observables=observables,
        observable_parameter_types=observable_parameter_types,
        actions=actions,
        observation_rules=observation_rules,
    )


def parse_domain(text: str) -> ParsedDomain:
    """Lint then parse a domain text."""

    from ..linter import lint_domain_text

    lint_result = lint_domain_text(text)
    if not lint_result.ok:
        raise ParseError(_format_lint_failure("Domain", lint_result))
    return _parse_domain_raw(text)


def _extract_domain_name(root: list[SExpr]) -> str:
    for item in root[1:]:
        if isinstance(item, list) and len(item) == 2 and item[0] == "domain":
            if not isinstance(item[1], str):
                raise ParseError("Domain name must be a symbol.")
            return item[1]
    raise ParseError("Domain header `(domain <name>)` not found.")


def _find_section(root: list[SExpr], section_name: str) -> list[SExpr] | None:
    for item in root[1:]:
        if isinstance(item, list) and item and item[0] == section_name:
            return item
    return None


def _parse_types(section: list[SExpr] | None) -> dict[str, TypeNode]:
    nodes: dict[str, TypeNode] = {"object": TypeNode("object")}
    if section is None:
        return nodes

    declarations = _parse_typed_symbol_sequence(section[1:], default_type="object")
    for type_name, _parent_name in declarations:
        nodes.setdefault(type_name, TypeNode(type_name))
    for type_name, parent_name in declarations:
        parent = nodes.setdefault(parent_name, TypeNode(parent_name))
        child = nodes[type_name]
        if child.parent is None:
            child.parent = parent
        if child not in parent.children:
            parent.children.append(child)
    return nodes


def _parse_constants(section: list[SExpr] | None) -> dict[str, str]:
    if section is None:
        return {}
    declarations = _parse_typed_symbol_sequence(section[1:], default_type="object")
    return {name: type_name for name, type_name in declarations}


def _parse_functions(section: list[SExpr] | None) -> dict[str, list[tuple[str, str]]]:
    if section is None:
        return {}
    functions: dict[str, list[tuple[str, str]]] = {}
    for item in section[1:]:
        if not isinstance(item, list) or not item:
            raise ParseError("Malformed function declaration in `:functions` section.")
        head = item[0]
        if not isinstance(head, str):
            raise ParseError("Function name must be a symbol.")
        typed_params = _parse_typed_symbol_sequence(item[1:], default_type="object")
        functions[head] = typed_params
    return functions


def _parse_predicate_like_section(
    section: list[SExpr] | None,
    cls: type[Predicate] | type[Observable],
) -> tuple[
    list[Predicate] | list[Observable],
    dict[Predicate, list[tuple[str, str]]] | dict[Observable, list[tuple[str, str]]],
]:
    if section is None:
        return [], {}

    parsed: list[Predicate] | list[Observable] = []
    parameter_types: dict[Predicate, list[tuple[str, str]]] | dict[Observable, list[tuple[str, str]]] = {}
    for item in section[1:]:
        if not isinstance(item, list) or not item:
            raise ParseError(f"Malformed declaration in section `{section[0]}`.")
        head = item[0]
        if not isinstance(head, str):
            raise ParseError(f"Malformed symbol name in section `{section[0]}`.")
        typed_params = _parse_typed_symbol_sequence(item[1:], default_type="object")
        params = [name for name, _type_name in typed_params]
        declaration = cls(head, params)  # type: ignore[arg-type]
        parsed.append(declaration)
        parameter_types[declaration] = typed_params
    return parsed, parameter_types


def _parse_actions(
    root: list[SExpr],
    action_effect_bucket_annotations: dict[str, list[ParsedEffectBucketAnnotation]],
) -> list[ParsedActionSchema]:
    actions: list[ParsedActionSchema] = []
    for item in root[1:]:
        if not isinstance(item, list) or not item or item[0] != ":action":
            continue
        actions.append(
            _parse_action(
                item,
                action_effect_bucket_annotations.get(
                    item[1] if len(item) > 1 and isinstance(item[1], str) else "",
                    [],
                ),
            )
        )
    return actions


def _parse_action(
    section: list[SExpr],
    effect_bucket_annotations: list[ParsedEffectBucketAnnotation],
) -> ParsedActionSchema:
    if len(section) < 2 or not isinstance(section[1], str):
        raise ParseError("Action block must have the form `(:action <name> ...)`.")
    name = section[1]
    parameters: list[str] = []
    parameter_types: list[tuple[str, str]] = []
    precondition: SExpr | None = None
    effect: SExpr | None = None
    for index, item in enumerate(section[2:], start=2):
        if item == ":parameters":
            if index + 1 >= len(section) or not isinstance(section[index + 1], list):
                raise ParseError(f"Action `{name}` has malformed `:parameters`.")
            parameter_types = _parse_typed_symbol_sequence(section[index + 1], default_type="object")
            parameters = [param_name for param_name, _type_name in parameter_types]
            continue
        if item == ":precondition":
            if index + 1 >= len(section):
                raise ParseError(f"Action `{name}` has malformed `:precondition`.")
            precondition = section[index + 1]
            continue
        if item == ":effect":
            if index + 1 >= len(section):
                raise ParseError(f"Action `{name}` has malformed `:effect`.")
            effect = section[index + 1]
            continue
    return ParsedActionSchema(
        action=Action(name=name, params=parameters),
        parameter_types=parameter_types,
        precondition=precondition,
        effect=effect,
        effect_bucket_annotations=list(effect_bucket_annotations),
    )


_ACTION_START_RE = re.compile(r"\(\s*:action\s+([^\s()]+)")
_BUCKET_COMMENT_RE = re.compile(
    r";+\s*bucket:\s*([^,;]+?)(?:\s*,\s*(.+?))?\s*$",
    re.IGNORECASE,
)


def _extract_action_effect_bucket_annotations(
    text: str,
) -> dict[str, list[ParsedEffectBucketAnnotation]]:
    """Scan raw domain text for effect-bucket annotations embedded in comments."""

    annotations_by_action: dict[str, list[ParsedEffectBucketAnnotation]] = {}
    current_action_name: str | None = None
    current_action_depth = 0
    current_annotations: list[ParsedEffectBucketAnnotation] = []

    for raw_line in text.splitlines():
        code_part, *_comment_part = raw_line.split(";", 1)
        if current_action_name is None:
            action_match = _ACTION_START_RE.search(code_part)
            if action_match is not None:
                current_action_name = action_match.group(1)
                current_action_depth = code_part.count("(") - code_part.count(")")
                current_annotations = []
                if current_action_depth <= 0:
                    annotations_by_action[current_action_name] = list(current_annotations)
                    current_action_name = None
                continue
            continue

        comment_match = _BUCKET_COMMENT_RE.search(raw_line)
        if comment_match is not None:
            bucket_name = comment_match.group(1).strip()
            metadata_text = comment_match.group(2)
            parsed_metadata: dict[str, str] = {}
            if metadata_text:
                for chunk in metadata_text.split(","):
                    chunk = chunk.strip()
                    if not chunk or ":" not in chunk:
                        continue
                    key, value = chunk.split(":", 1)
                    parsed_metadata[key.strip().lower()] = value.strip()
            success_literal = parsed_metadata.get("success")
            variant_rank_literal = parsed_metadata.get("variant_rank")
            variant_rank: int | None = None
            if variant_rank_literal is not None:
                try:
                    variant_rank = int(variant_rank_literal)
                except ValueError:
                    variant_rank = None
            current_annotations.append(
                ParsedEffectBucketAnnotation(
                    bucket_name=bucket_name,
                    success=(None if success_literal is None else success_literal.strip().lower() == "true"),
                    variant_rank=variant_rank,
                )
            )

        current_action_depth += code_part.count("(") - code_part.count(")")
        if current_action_depth <= 0:
            annotations_by_action[current_action_name] = list(current_annotations)
            current_action_name = None
            current_action_depth = 0
            current_annotations = []

    return annotations_by_action


def _parse_observation_rules(root: list[SExpr]) -> list[ParsedObservationRuleSchema]:
    rules: list[ParsedObservationRuleSchema] = []
    for item in root[1:]:
        if not isinstance(item, list) or not item or item[0] != ":observation":
            continue
        rules.append(_parse_observation_rule(item))
    return rules


def _parse_observation_rule(section: list[SExpr]) -> ParsedObservationRuleSchema:
    if len(section) < 2 or not isinstance(section[1], str):
        raise ParseError("Observation block must have the form `(:observation <name> ...)`.")
    name = section[1]
    parameters: list[str] = []
    parameter_types: list[tuple[str, str]] = []
    condition: SExpr | None = None
    distribution_expr: SExpr | None = None
    for index, item in enumerate(section[2:], start=2):
        if item == ":parameters":
            if index + 1 >= len(section) or not isinstance(section[index + 1], list):
                raise ParseError(f"Observation `{name}` has malformed `:parameters`.")
            parameter_types = _parse_typed_symbol_sequence(section[index + 1], default_type="object")
            parameters = [param_name for param_name, _type_name in parameter_types]
            continue
        if item == ":condition":
            if index + 1 >= len(section):
                raise ParseError(f"Observation `{name}` has malformed `:condition`.")
            condition = section[index + 1]
            continue
        if item == ":distribution":
            if index + 1 >= len(section):
                raise ParseError(f"Observation `{name}` has malformed `:distribution`.")
            distribution_expr = section[index + 1]
            continue

    distribution = _extract_distribution_support(distribution_expr)
    return ParsedObservationRuleSchema(
        rule=ObservationRule(name=name, distribution=distribution),
        parameter_types=parameter_types,
        parameters=parameters,
        condition=condition,
        distribution_expr=distribution_expr,
    )


def _extract_distribution_support(expr: SExpr | None) -> list[Observable]:
    if expr is None:
        return []
    support: list[Observable] = []
    _collect_distribution_observables(expr, support)
    return _dedupe_observables(support)


def _collect_distribution_observables(expr: SExpr, out: list[Observable]) -> None:
    if isinstance(expr, str):
        return
    if not expr:
        return

    head = expr[0]
    if not isinstance(head, str):
        return

    if head == "probabilistic":
        index = 1
        while index < len(expr):
            if index + 1 >= len(expr):
                raise ParseError("Malformed `probabilistic` distribution.")
            _collect_distribution_observables(expr[index + 1], out)
            index += 2
        return

    if head == "and":
        for subexpr in expr[1:]:
            _collect_distribution_observables(subexpr, out)
        return

    if head == "not":
        return

    params = _flatten_symbol_tokens(expr[1:])
    out.append(Observable(head, params))


def _dedupe_observables(observables: list[Observable]) -> list[Observable]:
    ordered: list[Observable] = []
    seen: set[Observable] = set()
    for observable in observables:
        if observable not in seen:
            seen.add(observable)
            ordered.append(observable)
    return ordered


def _extract_param_names(tokens: list[SExpr]) -> list[str]:
    raw_tokens = _flatten_symbol_tokens(tokens)
    declarations = _parse_typed_symbol_sequence(raw_tokens, default_type="object")
    return [name for name, _type_name in declarations]


def _flatten_symbol_tokens(tokens: list[SExpr]) -> list[str]:
    flat: list[str] = []
    for token in tokens:
        if not isinstance(token, str):
            raise ParseError("Expected a flat typed symbol list, but found nested structure.")
        flat.append(token)
    return flat


def _parse_typed_symbol_sequence(tokens: list[SExpr] | list[str], *, default_type: str) -> list[tuple[str, str]]:
    raw_tokens = _flatten_symbol_tokens(list(tokens))  # type: ignore[arg-type]
    declarations: list[tuple[str, str]] = []
    pending_names: list[str] = []
    index = 0

    while index < len(raw_tokens):
        token = raw_tokens[index]
        if token == "-":
            if not pending_names:
                raise ParseError("Unexpected `-` in typed symbol sequence.")
            if index + 1 >= len(raw_tokens):
                raise ParseError("Missing type name after `-` in typed symbol sequence.")
            type_name = raw_tokens[index + 1]
            for name in pending_names:
                declarations.append((name, type_name))
            pending_names.clear()
            index += 2
            continue

        pending_names.append(token)
        index += 1

    for name in pending_names:
        declarations.append((name, default_type))

    return declarations


def _format_lint_failure(prefix: str, lint_result) -> str:
    messages = [f"{diagnostic.code}: {diagnostic.message}" for diagnostic in lint_result.errors]
    return f"{prefix} failed lint validation:\n- " + "\n- ".join(messages)
