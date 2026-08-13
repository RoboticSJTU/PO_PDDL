from __future__ import annotations

import re
from dataclasses import replace

from po_pddl.domain_generation.stages.domain_patching import (
    DomainActionPatch,
    apply_domain_repair_patch,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.models import (
    ActionEffectBranch,
    ActionSchema,
)


def last_action_predicate_name(parameter_count: int) -> str:
    return f"last_action_{max(int(parameter_count), 0)}_param"


def last_action_variant_index(effect_bucket: str, variant_rank: int | None = None) -> int:
    if variant_rank is not None:
        return max(int(variant_rank) - 1, 0)
    suffix = str(effect_bucket).rsplit("_bucket_", 1)
    if len(suffix) == 2 and suffix[1].isdigit():
        return max(int(suffix[1]) - 1, 0)
    return 0


def last_action_constant_name(
    action_name: str,
    *,
    success: bool,
    effect_bucket: str,
    variant_rank: int | None = None,
) -> str:
    outcome = "success" if success else "fail"
    variant_index = last_action_variant_index(effect_bucket, variant_rank)
    return f"{action_name}_{outcome}_{variant_index}"


def augment_action_schemas_with_last_action_markers(
    action_schemas: list[ActionSchema],
) -> tuple[list[ActionSchema], list[str], list[int]]:
    arities = sorted({max(int(schema.parameter_count), 0) for schema in action_schemas})
    augmented: list[ActionSchema] = []
    constant_names: list[str] = []

    for schema in action_schemas:
        branches = list(schema.effect_branches) or [
            ActionEffectBranch(
                effect_bucket=f"{schema.canonical_action_name}_success",
                probability=1.0,
                success=True,
                delta_add=[],
                delta_del=[],
                variant_rank=0,
            )
        ]
        rendered_branches: list[ActionEffectBranch] = []
        for branch in branches:
            marker_constant = last_action_constant_name(
                schema.canonical_action_name,
                success=branch.success,
                effect_bucket=branch.effect_bucket,
                variant_rank=branch.variant_rank,
            )
            constant_names.append(marker_constant)
            reset_clauses = _all_last_action_reset_clauses(arities)
            set_true_clause = _current_last_action_true_clause(
                parameter_count=schema.parameter_count,
                marker_constant=marker_constant,
            )
            rendered_branches.append(
                replace(
                    branch,
                    extra_pddl_effect_conjuncts=[*reset_clauses, set_true_clause],
                )
            )
        augmented.append(
            ActionSchema(
                canonical_action_name=schema.canonical_action_name,
                action_category=schema.action_category,
                parameter_count=schema.parameter_count,
                parameter_roles=list(schema.parameter_roles),
                precondition_literals=list(schema.precondition_literals),
                schema_description=schema.schema_description,
                effect_branches=rendered_branches,
            )
        )

    return augmented, sorted(dict.fromkeys(constant_names)), arities


def inject_last_action_infrastructure_into_domain(
    *,
    domain_text: str,
    action_schemas: list[ActionSchema],
) -> str:
    augmented_schemas, constant_names, arities = augment_action_schemas_with_last_action_markers(action_schemas)
    updated_text = _ensure_last_action_marker_type(domain_text)
    updated_text = _ensure_predicates_block(updated_text)
    updated_text = _ensure_last_action_predicates(updated_text, arities)
    updated_text = _ensure_constants_block(updated_text, constant_names)
    try:
        updated_text = apply_domain_repair_patch(
            domain_text=updated_text,
            predicate_comment_updates=None,
            predicate_parameter_type_updates=None,
            predicate_additions=[],
            predicate_removals=None,
            action_patches=[
                DomainActionPatch(
                    action_name=schema.canonical_action_name,
                    effect_branches=list(schema.effect_branches),
                )
                for schema in augmented_schemas
            ],
        )
    except Exception:
        updated_text = _apply_last_action_effect_patches_without_parser(updated_text, augmented_schemas)
    return updated_text


def _all_last_action_reset_clauses(arities: list[int]) -> list[str]:
    clauses: list[str] = []
    for arity in arities:
        quantifiers = ["?m - last_action_marker", *[f"?x{index} - object" for index in range(arity)]]
        arguments = " ".join(["?m", *[f"?x{index}" for index in range(arity)]])
        clauses.append(f"(forall ({' '.join(quantifiers)}) (not ({last_action_predicate_name(arity)} {arguments})))")
    return clauses


def _current_last_action_true_clause(*, parameter_count: int, marker_constant: str) -> str:
    arguments = " ".join([marker_constant, *[f"?arg{index}" for index in range(max(int(parameter_count), 0))]])
    return f"({last_action_predicate_name(parameter_count)} {arguments})"


def _ensure_last_action_marker_type(domain_text: str) -> str:
    try:
        span = _find_block_span(domain_text, ":types")
    except ValueError:
        return _insert_top_level_block(
            domain_text,
            "  (:types\n    last_action_marker - object\n  )",
            after_heads=[":requirements"],
        )
    block_text = domain_text[span[0] : span[1]]
    if "last_action_marker" in block_text:
        return domain_text
    insertion = "    last_action_marker - object\n"
    return domain_text[: span[1] - 1] + insertion + domain_text[span[1] - 1 :]


def _ensure_predicates_block(domain_text: str) -> str:
    try:
        _find_block_span(domain_text, ":predicates")
    except ValueError:
        return _insert_top_level_block(
            domain_text,
            "  (:predicates\n  )",
            after_heads=[":constants", ":types", ":requirements"],
        )
    return domain_text


def _ensure_last_action_predicates(domain_text: str, arities: list[int]) -> str:
    if not arities:
        return domain_text
    span = _find_block_span(domain_text, ":predicates")
    block_text = domain_text[span[0] : span[1]]
    missing_lines: list[str] = []
    for arity in arities:
        params = " ".join(["?x0 - last_action_marker", *[f"?x{index + 1} - object" for index in range(arity)]])
        rendered = f"    ({last_action_predicate_name(arity)} {params})"
        if rendered not in block_text:
            missing_lines.append(rendered)
    if not missing_lines:
        return domain_text
    insertion = "\n".join(missing_lines) + "\n"
    return domain_text[: span[1] - 1] + insertion + domain_text[span[1] - 1 :]


def _ensure_constants_block(domain_text: str, constant_names: list[str]) -> str:
    if not constant_names:
        return domain_text
    rendered_lines = [f"    {name} - last_action_marker" for name in constant_names]
    try:
        span = _find_block_span(domain_text, ":constants")
    except ValueError:
        insertion = "  (:constants\n" + "\n".join(rendered_lines) + "\n  )"
        return _insert_top_level_block(
            domain_text,
            insertion,
            after_heads=[":predicates", ":types", ":requirements"],
        )
    block_text = domain_text[span[0] : span[1]]
    existing_names = {name for name in constant_names if f"{name} - last_action_marker" in block_text}
    missing = [line for name, line in zip(constant_names, rendered_lines) if name not in existing_names]
    if not missing:
        return domain_text
    insertion = "\n".join(missing) + "\n"
    return domain_text[: span[1] - 1] + insertion + domain_text[span[1] - 1 :]


def _find_block_span(text: str, head: str) -> tuple[int, int]:
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
        return index, cursor
    raise ValueError(f"Could not find block `{head}` in domain text.")


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


def _read_symbol(text: str, index: int) -> tuple[str, int]:
    start = index
    while index < len(text) and not text[index].isspace() and text[index] not in "()":
        index += 1
    return text[start:index], index


def _insert_top_level_block(domain_text: str, block_text: str, *, after_heads: list[str]) -> str:
    insertion_point: int | None = None
    for head in after_heads:
        try:
            _start, end = _find_block_span(domain_text, head)
        except ValueError:
            continue
        insertion_point = end if insertion_point is None else max(insertion_point, end)
    if insertion_point is None:
        match = re.search(r"\(define\s+\(domain[^\)]*\)\s*", domain_text)
        insertion_point = match.end() if match is not None else 0
    normalized = block_text if block_text.endswith("\n") else block_text + "\n"
    prefix = domain_text[:insertion_point]
    if prefix and not prefix.endswith("\n"):
        normalized = "\n" + normalized
    if prefix and not prefix.endswith("\n\n"):
        normalized = "\n" + normalized
    return prefix + normalized + domain_text[insertion_point:]


def _apply_last_action_effect_patches_without_parser(domain_text: str, action_schemas: list[ActionSchema]) -> str:
    updated_text = domain_text
    for schema in action_schemas:
        try:
            span = _find_named_block_span(updated_text, ":action", schema.canonical_action_name)
        except ValueError:
            continue
        action_block = updated_text[span[0] : span[1]]
        replacement = _replace_effect_block(action_block, schema.effect_branches)
        updated_text = updated_text[: span[0]] + replacement + updated_text[span[1] :]
    return updated_text


def _find_named_block_span(text: str, head: str, name: str) -> tuple[int, int]:
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
            raise ValueError(f"Unclosed block for `{head} {name}` in domain text.")
        return index, cursor
    raise ValueError(f"Could not find block `{head} {name}` in domain text.")


def _replace_effect_block(action_block: str, effect_branches: list[ActionEffectBranch]) -> str:
    effect_index = action_block.find(":effect")
    if effect_index < 0:
        return action_block
    line_start = action_block.rfind("\n", 0, effect_index) + 1
    tail_start = action_block.rfind("\n", 0, len(action_block) - 1)
    if tail_start <= line_start:
        return action_block
    effect_lines = _render_effect_branches(effect_branches)
    replacement = "\n".join(effect_lines)
    return action_block[:line_start] + replacement + action_block[tail_start:]


def _render_effect_branches(branches: list[ActionEffectBranch]) -> list[str]:
    lines = ["    :effect", "      (probabilistic"]
    for branch in branches:
        conjuncts = [f"({fact})" if fact.endswith(")") and not fact.startswith("(") else fact for fact in []]
        rendered_add = [f"({fact[:-2]})" if fact.endswith("()") else None for fact in []]
        del conjuncts, rendered_add
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
        add_parts = [_render_literal(item) for item in branch.delta_add]
        del_parts = [f"(not {_render_literal(item)})" for item in branch.delta_del]
        extra_parts = [str(item).strip() for item in branch.extra_pddl_effect_conjuncts if str(item).strip()]
        body_parts = [*add_parts, *del_parts, *extra_parts]
        effect_body = f"(and {' '.join(body_parts)})" if body_parts else "(and)"
        lines.append(f"        {branch.probability:.6f} {effect_body}")
    lines.append("      )")
    return lines


def _render_literal(text: str) -> str:
    stripped = str(text).strip()
    if not stripped:
        return stripped
    if stripped.startswith("("):
        return stripped
    if "(" in stripped:
        name, remainder = stripped.split("(", 1)
        args = remainder.rstrip(")")
        args = " ".join(part.strip() for part in args.split(",") if part.strip())
        return f"({name.strip()} {args})" if args else f"({name.strip()})"
    if stripped.endswith("()"):
        return f"({stripped[:-2]})"
    return f"({stripped})"


__all__ = [
    "augment_action_schemas_with_last_action_markers",
    "inject_last_action_infrastructure_into_domain",
    "last_action_constant_name",
    "last_action_predicate_name",
    "last_action_variant_index",
]
