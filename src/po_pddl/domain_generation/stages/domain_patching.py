from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass

from po_pddl.core.parser import parse_domain
from po_pddl.core.parser.schemas import ParsedActionSchema, ParsedDomain
from po_pddl.core.parser.sexpr import SExpr
from po_pddl.domain_generation.infrastructure.fact_utils import (
    parse_symbolic_literal,
    remap_symbolic_literal_arguments,
    render_symbolic_literal_to_pddl,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.models import ActionEffectBranch


@dataclass(frozen=True)
class DomainPredicatePatch:
    predicate_name: str
    comment: str | None = None
    parameter_types: list[str] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "predicate_name": self.predicate_name,
            "comment": self.comment,
            "parameter_types": list(self.parameter_types) if self.parameter_types is not None else None,
        }


@dataclass(frozen=True)
class DomainActionPatch:
    action_name: str
    action_comment: str | None = None
    parameter_types: list[str] | None = None
    precondition_literals: list[str] | None = None
    effect_branches: list[ActionEffectBranch] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "action_name": self.action_name,
            "action_comment": self.action_comment,
            "parameter_types": list(self.parameter_types) if self.parameter_types is not None else None,
            "precondition_literals": list(self.precondition_literals)
            if self.precondition_literals is not None
            else None,
            "effect_branches": [branch.to_dict() for branch in self.effect_branches or []]
            if self.effect_branches is not None
            else None,
        }


def _normalize_zero_arity_literal(text: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return cleaned
    if "(" in cleaned or ")" in cleaned:
        return cleaned
    if cleaned.startswith("not "):
        remainder = cleaned[4:].strip()
        if "(" in remainder or ")" in remainder or not remainder:
            return cleaned
        return f"not {remainder}()"
    return f"{cleaned}()"


def _render_comment_lines(text: str | None, *, indent: str = "  ") -> list[str]:
    if text is None:
        return []
    cleaned = " ".join(part.strip() for part in str(text).splitlines() if part.strip()).strip()
    if not cleaned:
        return []
    wrapped = textwrap.wrap(cleaned, width=max(20, 100 - len(indent) - 3))
    return [f"{indent};; {line}" for line in wrapped]


def _indent_multiline(text: str, indent: str) -> str:
    return "\n".join(f"{indent}{line}" if line else indent.rstrip() for line in text.splitlines())


def _render_quantified_precondition_block(
    quantifier: str,
    parameters: str,
    body_literals: list[str],
) -> str:
    lines = [f"({quantifier} ({parameters})", "  (and"]
    lines.extend(f"    {literal}" for literal in body_literals)
    lines.append("  )")
    lines.append(")")
    return "\n".join(lines)


def _render_sexpr(expr: SExpr | None) -> str:
    if expr is None:
        return "(and)"
    if isinstance(expr, str):
        return expr
    if not expr:
        return "()"
    rendered_items = [_render_sexpr(item) for item in expr]
    return f"({' '.join(rendered_items)})"


def _render_effect_branches(branches: list[ActionEffectBranch]) -> list[str]:
    lines = ["    :effect", "      (probabilistic"]
    for branch in branches:
        if branch.fixed_delta_add or branch.fixed_delta_del:
            lines.append(f"        ; fixed: add={list(branch.fixed_delta_add)}, del={list(branch.fixed_delta_del)}")
        lines.append(
            f"        ; bucket: {branch.effect_bucket}, success: {'true' if branch.success else 'false'}, "
            f"variant_rank: {branch.variant_rank if branch.variant_rank is not None else 0}"
        )
        if branch.residual_delta_add or branch.residual_delta_del:
            lines.append(
                f"        ; residual: add={list(branch.residual_delta_add)}, del={list(branch.residual_delta_del)}"
            )
        conjuncts = [render_symbolic_literal_to_pddl(_normalize_zero_arity_literal(fact)) for fact in branch.delta_add]
        conjuncts.extend(
            f"(not {render_symbolic_literal_to_pddl(_normalize_zero_arity_literal(fact))})" for fact in branch.delta_del
        )
        conjuncts.extend(str(item).strip() for item in branch.extra_pddl_effect_conjuncts if str(item).strip())
        effect_body = f"(and {' '.join(conjuncts)})" if conjuncts else "(and)"
        lines.append(f"        {branch.probability:.6f} {effect_body}")
    lines.append("      )")
    return lines


def _infer_existential_variable_types_from_literals(
    literals: list[str],
    *,
    predicate_parameter_types: dict[str, list[str]],
    action_parameter_symbols: set[str],
) -> dict[str, str]:
    def normalize_declared_type_name(type_entry: object) -> str:
        if isinstance(type_entry, (list, tuple)):
            if len(type_entry) >= 2 and str(type_entry[1]).strip():
                return str(type_entry[1]).strip()
            if len(type_entry) >= 1 and str(type_entry[0]).strip():
                return str(type_entry[0]).strip()
            return "object"
        text = str(type_entry).strip()
        if text.startswith("(") and text.endswith(")"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, (list, tuple)):
                return normalize_declared_type_name(parsed)
        return text or "object"

    inferred_types: dict[str, str] = {}
    for literal in literals:
        _negated, predicate, arguments = parse_symbolic_literal(literal)
        declared_types = predicate_parameter_types.get(predicate, [])
        for index, argument in enumerate(arguments):
            if not argument.startswith("?") or argument in action_parameter_symbols:
                continue
            candidate_type = (
                normalize_declared_type_name(declared_types[index]) if index < len(declared_types) else "object"
            )
            existing_type = inferred_types.get(argument)
            if existing_type is None or existing_type == candidate_type:
                inferred_types[argument] = candidate_type
            elif existing_type == "object":
                inferred_types[argument] = candidate_type
            elif candidate_type != "object":
                inferred_types[argument] = "object"
    return inferred_types


def _rename_quantified_literal_variables(
    literal: str,
    *,
    variable_mapping: dict[str, str],
) -> str:
    return remap_symbolic_literal_arguments(literal, variable_mapping) if variable_mapping else literal


def _render_precondition_text(
    literals: list[str],
    *,
    predicate_parameter_types: dict[str, list[str]],
    action_parameter_symbols: set[str],
) -> tuple[str, bool, bool]:
    if not literals:
        return "(and)", False, False
    existential_types = _infer_existential_variable_types_from_literals(
        literals,
        predicate_parameter_types=predicate_parameter_types,
        action_parameter_symbols=action_parameter_symbols,
    )
    if not existential_types:
        conjuncts = [render_symbolic_literal_to_pddl(item) for item in literals]
        lines = ["(and"]
        lines.extend(f"  {conjunct}" for conjunct in conjuncts)
        lines.append(")")
        return "\n".join(lines), False, False
    base_literals: list[str] = []
    existential_literals: list[str] = []
    for literal in literals:
        _negated, _predicate, arguments = parse_symbolic_literal(literal)
        extra_variables = [
            argument for argument in arguments if argument.startswith("?") and argument not in action_parameter_symbols
        ]
        if extra_variables:
            existential_literals.append(literal)
        else:
            base_literals.append(literal)
    rendered_base_conjuncts = [render_symbolic_literal_to_pddl(item) for item in base_literals]
    existential_negated = all(parse_symbolic_literal(literal)[0] for literal in existential_literals)
    if existential_literals and existential_negated:
        rendered_universal_conjuncts: list[str] = []
        for index, literal in enumerate(existential_literals):
            per_literal_types = _infer_existential_variable_types_from_literals(
                [literal],
                predicate_parameter_types=predicate_parameter_types,
                action_parameter_symbols=action_parameter_symbols,
            )
            variable_mapping = {
                variable: f"?dep{index}_{position}" for position, variable in enumerate(sorted(per_literal_types))
            }
            renamed_literal = _rename_quantified_literal_variables(
                literal,
                variable_mapping=variable_mapping,
            )
            negated, _predicate, _arguments = parse_symbolic_literal(renamed_literal)
            positive_literal = renamed_literal[4:].strip() if negated else renamed_literal
            rendered_positive_literal = render_symbolic_literal_to_pddl(positive_literal)
            universal_parameters = " ".join(
                f"{variable_mapping[variable]} - {per_literal_types[variable]}"
                for variable in sorted(per_literal_types)
            )
            rendered_universal_conjuncts.append(
                _render_quantified_precondition_block(
                    "forall",
                    universal_parameters,
                    [f"(not {rendered_positive_literal})"],
                )
            )
        rendered_all_conjuncts = rendered_base_conjuncts + rendered_universal_conjuncts
        lines = ["(and"]
        for conjunct in rendered_all_conjuncts:
            lines.append(_indent_multiline(conjunct, "  "))
        lines.append(")")
        return "\n".join(lines), False, True
    existential_variable_mapping = {
        variable: f"?dep_exists_{index}" for index, variable in enumerate(sorted(existential_types))
    }
    existential_parameters = " ".join(
        f"{existential_variable_mapping[variable]} - {existential_types[variable]}"
        for variable in sorted(existential_types)
    )
    rendered_existential_block = _render_quantified_precondition_block(
        "exists",
        existential_parameters,
        [
            render_symbolic_literal_to_pddl(
                _rename_quantified_literal_variables(
                    literal,
                    variable_mapping=existential_variable_mapping,
                )
            )
            for literal in existential_literals
        ],
    )
    rendered_all_conjuncts = rendered_base_conjuncts + [rendered_existential_block]
    lines = ["(and"]
    for conjunct in rendered_all_conjuncts:
        lines.append(_indent_multiline(conjunct, "  "))
    lines.append(")")
    return "\n".join(lines), True, False


def _render_requirements_text(
    domain_text: str,
    *,
    uses_negative_preconditions: bool,
    uses_existential_preconditions: bool,
    uses_universal_preconditions: bool,
) -> str:
    span = _find_block_span(domain_text, ":requirements")
    section = domain_text[span.start : span.end]
    requirement_tokens = [
        token
        for token in section.replace("(", " ").replace(")", " ").split()
        if token.startswith(":") and token != ":requirements"
    ]
    ordered_requirements: list[str] = []
    for token in requirement_tokens:
        if token not in ordered_requirements:
            ordered_requirements.append(token)
    if uses_negative_preconditions and ":negative-preconditions" not in ordered_requirements:
        ordered_requirements.append(":negative-preconditions")
    if uses_existential_preconditions and ":existential-preconditions" not in ordered_requirements:
        ordered_requirements.append(":existential-preconditions")
    if uses_universal_preconditions and ":universal-preconditions" not in ordered_requirements:
        ordered_requirements.append(":universal-preconditions")
    lines = ["  (:requirements"]
    lines.extend(f"    {requirement}" for requirement in ordered_requirements)
    lines.append("  )")
    return "\n".join(lines)


def _extract_raw_section_lines(action_text: str, section_name: str) -> list[str] | None:
    keyword_index = action_text.find(section_name)
    if keyword_index < 0:
        return None
    line_start = action_text.rfind("\n", 0, keyword_index) + 1
    pos = keyword_index + len(section_name)
    while pos < len(action_text) and action_text[pos].isspace():
        pos += 1
    if pos >= len(action_text) or action_text[pos] != "(":
        return None
    depth = 0
    end = pos
    while end < len(action_text):
        if action_text[end] == "(":
            depth += 1
        elif action_text[end] == ")":
            depth -= 1
            if depth == 0:
                end += 1
                break
        end += 1
    if depth != 0:
        return None
    return action_text[line_start:end].rstrip().splitlines()


def _render_action_block(
    *,
    parsed_action: ParsedActionSchema,
    patch: DomainActionPatch | None,
    existing_comment: str | None,
    existing_action_text: str | None,
    predicate_parameter_types: dict[str, list[str]],
) -> str:
    parameter_types = list(parsed_action.parameter_types)
    if patch is not None and patch.parameter_types is not None:
        if len(patch.parameter_types) != len(parameter_types):
            raise ValueError(
                f"Action patch for `{parsed_action.action.name}` provided {len(patch.parameter_types)} parameter types, "
                f"expected {len(parameter_types)}."
            )
        parameter_types = [
            (name, str(type_name).strip() or "object")
            for (name, _old_type), type_name in zip(parameter_types, patch.parameter_types)
        ]
    parameter_text = " ".join(f"{name} - {type_name}" for name, type_name in parameter_types)

    if patch is not None and patch.precondition_literals is not None:
        literals = [_normalize_zero_arity_literal(item) for item in patch.precondition_literals]
        precondition_text, _uses_existential, _uses_universal = _render_precondition_text(
            literals,
            predicate_parameter_types=predicate_parameter_types,
            action_parameter_symbols={name for name, _type_name in parameter_types},
        )
    else:
        precondition_text = _render_sexpr(parsed_action.precondition)

    if patch is not None and patch.effect_branches is not None:
        effect_lines = _render_effect_branches(patch.effect_branches)
    elif existing_action_text is not None:
        effect_lines = _extract_raw_section_lines(existing_action_text, ":effect")
        if effect_lines is None:
            effect_lines = [f"    :effect {_render_sexpr(parsed_action.effect)}"]
    else:
        effect_lines = [f"    :effect {_render_sexpr(parsed_action.effect)}"]

    comment_text = patch.action_comment if patch is not None and patch.action_comment is not None else existing_comment
    lines = []
    lines.extend(_render_comment_lines(comment_text))
    lines.append(f"  (:action {parsed_action.action.name}")
    lines.append(f"    :parameters ({parameter_text})")
    lines.append("    :precondition")
    lines.append(_indent_multiline(precondition_text, "      "))
    lines.extend(effect_lines)
    lines.append("  )")
    return "\n".join(lines)


@dataclass(frozen=True)
class _BlockSpan:
    start: int
    end: int


def _read_symbol(text: str, index: int) -> tuple[str, int]:
    start = index
    while index < len(text) and not text[index].isspace() and text[index] not in "()":
        index += 1
    return text[start:index], index


def _skip_ws_and_comments(text: str, index: int) -> int:
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        if text[index] == ";":
            while index < len(text) and text[index] != "\n":
                index += 1
            continue
        break
    return index


def _find_block_span(text: str, head: str, name: str | None = None) -> _BlockSpan:
    index = 0
    in_comment = False
    while index < len(text):
        char = text[index]
        if in_comment:
            if char == "\n":
                in_comment = False
            index += 1
            continue
        if char == ";":
            in_comment = True
            index += 1
            continue
        if char != "(":
            index += 1
            continue
        token_index = _skip_ws_and_comments(text, index + 1)
        token, token_index = _read_symbol(text, token_index)
        if token != head:
            index += 1
            continue
        if name is not None:
            token_index = _skip_ws_and_comments(text, token_index)
            next_token, _ = _read_symbol(text, token_index)
            if next_token != name:
                index += 1
                continue
        depth = 1
        cursor = index + 1
        nested_comment = False
        while cursor < len(text) and depth > 0:
            nested_char = text[cursor]
            if nested_comment:
                if nested_char == "\n":
                    nested_comment = False
                cursor += 1
                continue
            if nested_char == ";":
                nested_comment = True
                cursor += 1
                continue
            if nested_char == "(":
                depth += 1
            elif nested_char == ")":
                depth -= 1
            cursor += 1
        if depth != 0:
            raise ValueError(f"Unclosed block for `{head}` in domain text.")
        return _BlockSpan(start=index, end=cursor)
    target = f"{head} {name}" if name is not None else head
    raise ValueError(f"Could not find block `{target}` in domain text.")


def _extract_leading_comment(text: str, block_start: int) -> tuple[str | None, int]:
    line_start = text.rfind("\n", 0, block_start) + 1
    comments: list[str] = []
    cursor = line_start
    comment_start = block_start
    while cursor > 0:
        prev_line_end = cursor - 1
        prev_line_start = text.rfind("\n", 0, prev_line_end) + 1
        prev_line = text[prev_line_start:prev_line_end].rstrip()
        stripped = prev_line.strip()
        if not stripped:
            break
        if not stripped.startswith(";;"):
            break
        comments.insert(0, stripped[2:].strip())
        comment_start = prev_line_start
        cursor = prev_line_start
    combined = " ".join(part for part in comments if part)
    return combined or None, comment_start


def extract_predicate_comments(text: str) -> dict[str, str]:
    span = _find_block_span(text, ":predicates")
    section = text[span.start : span.end]
    comments: dict[str, str] = {}
    current_comment_lines: list[str] = []
    for raw_line in section.splitlines()[1:]:
        stripped = raw_line.strip()
        if stripped == ")":
            break
        if not stripped:
            current_comment_lines = []
            continue
        if stripped.startswith(";;"):
            current_comment_lines.append(stripped[2:].strip())
            continue
        if stripped.startswith("("):
            predicate_name = stripped[1:].split()[0].rstrip(")")
            comment = " ".join(part for part in current_comment_lines if part).strip()
            if comment:
                comments[predicate_name] = comment
            current_comment_lines = []
            continue
        current_comment_lines = []
    return comments


def extract_action_comments(text: str) -> dict[str, str]:
    parsed_domain = parse_domain(text)
    comments: dict[str, str] = {}
    for parsed_action in parsed_domain.actions:
        try:
            span = _find_block_span(text, ":action", parsed_action.action.name)
        except ValueError:
            continue
        comment, _replacement_start = _extract_leading_comment(text, span.start)
        if comment:
            comments[parsed_action.action.name] = comment
    return comments


def _render_predicates_section(
    *,
    parsed_domain: ParsedDomain,
    predicate_comments: dict[str, str],
    predicate_type_updates: dict[str, list[str]],
    predicate_additions: list[DomainPredicatePatch],
    predicate_removals: set[str],
) -> str:
    lines = ["  (:predicates"]
    rendered_existing: set[str] = set()
    for predicate in parsed_domain.predicates:
        name = predicate.name
        if name in predicate_removals:
            continue
        rendered_existing.add(name)
        lines.extend(_render_comment_lines(predicate_comments.get(name), indent="    "))
        parameter_types = list(parsed_domain.predicate_parameter_types.get(predicate, []))
        if name in predicate_type_updates:
            updated = [str(item).strip() or "object" for item in predicate_type_updates[name]]
            if len(updated) != len(parameter_types):
                raise ValueError(
                    f"Predicate type update for `{name}` provided {len(updated)} types, expected {len(parameter_types)}."
                )
            parameter_types = [
                (param_name, updated_type) for (param_name, _), updated_type in zip(parameter_types, updated)
            ]
        if not parameter_types:
            lines.append(f"    ({name})")
        else:
            params = " ".join(f"{param_name} - {type_name}" for param_name, type_name in parameter_types)
            lines.append(f"    ({name} {params})")
    for predicate_patch in predicate_additions:
        if predicate_patch.predicate_name in rendered_existing or predicate_patch.predicate_name in predicate_removals:
            continue
        lines.extend(_render_comment_lines(predicate_patch.comment, indent="    "))
        parameter_types = list(predicate_patch.parameter_types or [])
        if not parameter_types:
            lines.append(f"    ({predicate_patch.predicate_name})")
        else:
            params = " ".join(f"?x{i} - {type_name}" for i, type_name in enumerate(parameter_types))
            lines.append(f"    ({predicate_patch.predicate_name} {params})")
    lines.append("  )")
    return "\n".join(lines)


def apply_domain_repair_patch(
    *,
    domain_text: str,
    predicate_comment_updates: dict[str, str] | None,
    predicate_parameter_type_updates: dict[str, list[str]] | None,
    predicate_additions: list[DomainPredicatePatch] | None,
    predicate_removals: list[str] | None,
    action_patches: list[DomainActionPatch] | None,
) -> str:
    parsed_domain = parse_domain(domain_text)
    updated_text = domain_text
    predicate_parameter_types = {
        predicate.name: list(parsed_domain.predicate_parameter_types.get(predicate, []))
        for predicate in parsed_domain.predicates
    }

    merged_predicate_comments = extract_predicate_comments(domain_text)
    for name, comment in (predicate_comment_updates or {}).items():
        merged_predicate_comments[name] = str(comment).strip()
    predicates_span = _find_block_span(updated_text, ":predicates")
    rendered_predicates = _render_predicates_section(
        parsed_domain=parsed_domain,
        predicate_comments=merged_predicate_comments,
        predicate_type_updates=predicate_parameter_type_updates or {},
        predicate_additions=predicate_additions or [],
        predicate_removals={item.strip() for item in (predicate_removals or []) if item.strip()},
    )
    updated_text = updated_text[: predicates_span.start] + rendered_predicates + updated_text[predicates_span.end :]

    action_patch_map = {item.action_name: item for item in action_patches or []}
    uses_negative_preconditions = False
    uses_existential_preconditions = False
    uses_universal_preconditions = False
    for parsed_action in sorted(parsed_domain.actions, key=lambda item: len(item.action.name), reverse=True):
        if parsed_action.action.name not in action_patch_map:
            continue
        patch = action_patch_map[parsed_action.action.name]
        if patch.precondition_literals is not None:
            for literal in patch.precondition_literals:
                negated, _predicate, arguments = parse_symbolic_literal(_normalize_zero_arity_literal(literal))
                uses_negative_preconditions = uses_negative_preconditions or negated
                extra_variables = [argument for argument in arguments if argument.startswith("?dep")]
                if extra_variables and negated:
                    uses_universal_preconditions = True
                elif extra_variables:
                    uses_existential_preconditions = True
        span = _find_block_span(updated_text, ":action", parsed_action.action.name)
        existing_comment, replacement_start = _extract_leading_comment(updated_text, span.start)
        rendered_action = _render_action_block(
            parsed_action=parsed_action,
            patch=patch,
            existing_comment=existing_comment,
            existing_action_text=updated_text[span.start : span.end],
            predicate_parameter_types=predicate_parameter_types,
        )
        updated_text = updated_text[:replacement_start] + rendered_action + updated_text[span.end :]

    requirements_span = _find_block_span(updated_text, ":requirements")
    updated_requirements = _render_requirements_text(
        updated_text,
        uses_negative_preconditions=uses_negative_preconditions,
        uses_existential_preconditions=uses_existential_preconditions,
        uses_universal_preconditions=uses_universal_preconditions,
    )
    updated_text = (
        updated_text[: requirements_span.start] + updated_requirements + updated_text[requirements_span.end :]
    )

    return updated_text


__all__ = [
    "DomainActionPatch",
    "DomainPredicatePatch",
    "apply_domain_repair_patch",
    "extract_action_comments",
    "extract_predicate_comments",
]
