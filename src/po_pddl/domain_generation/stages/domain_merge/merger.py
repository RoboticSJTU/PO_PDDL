from __future__ import annotations

import re
from dataclasses import dataclass

from po_pddl.core.parser.domain_parser import _parse_domain_raw
from po_pddl.core.parser.sexpr import SExpr

_LINE_COMMENT_RE = re.compile(r";;.*$")
_OBSERVATION_PLACEHOLDER_RE = re.compile(r"(?m)^[ \t]*;; Observation-action learning is not implemented yet\.[ \t]*\n?")
_PREDICATE_ENTRY_NAME_RE = re.compile(r"^\(\s*([a-zA-Z][a-zA-Z0-9_-]*)\b")
_BUILTIN_OPS = {
    "and",
    "or",
    "not",
    "imply",
    "forall",
    "all",
    "exists",
    "=",
    "when",
    "probabilistic",
    "increase",
    "decrease",
    "assign",
}


@dataclass(frozen=True)
class _TopLevelForm:
    keyword: str
    text: str
    start: int
    end: int


def _strip_line_comments_preserve_length(text: str) -> str:
    lines = []
    for line in text.splitlines(keepends=True):
        match = _LINE_COMMENT_RE.search(line)
        if match:
            start = match.start()
            comment = line[start:]
            lines.append(line[:start] + (" " * len(comment)))
        else:
            lines.append(line)
    return "".join(lines)


def _find_top_level_forms(text: str) -> list[_TopLevelForm]:
    cleaned = _strip_line_comments_preserve_length(text)
    forms: list[_TopLevelForm] = []
    depth = 0
    idx = 0
    while idx < len(cleaned):
        ch = cleaned[idx]
        if ch == "(":
            if depth == 1 and cleaned.startswith("(:", idx):
                start = idx
                form_depth = 0
                end = idx
                while end < len(cleaned):
                    if cleaned[end] == "(":
                        form_depth += 1
                    elif cleaned[end] == ")":
                        form_depth -= 1
                        if form_depth == 0:
                            end += 1
                            break
                    end += 1
                form_text = text[start:end]
                keyword_match = re.match(r"\(\s*:([a-zA-Z0-9_-]+)", form_text)
                if keyword_match:
                    forms.append(
                        _TopLevelForm(
                            keyword=keyword_match.group(1),
                            text=form_text,
                            start=start,
                            end=end,
                        )
                    )
                idx = end
                continue
            depth += 1
        elif ch == ")":
            depth -= 1
        idx += 1
    return forms


def _form_name(form_text: str, keyword: str) -> str | None:
    match = re.match(rf"\(\s*:{re.escape(keyword)}\s+([a-zA-Z0-9_-]+)", form_text)
    return match.group(1) if match else None


def _extract_block_entries(block_text: str) -> list[str]:
    lines = block_text.splitlines()
    if len(lines) < 2:
        return []
    entries: list[str] = []
    for line in lines[1:-1]:
        stripped = line.strip()
        if not stripped:
            continue
        entries.append(stripped)
    return entries


def _render_block(keyword: str, entries: list[str], *, indent: str = "  ") -> str:
    if not entries:
        return ""
    body = "\n".join(f"{indent}  {entry}" for entry in entries)
    return f"{indent}(:{keyword}\n{body}\n{indent})"


def _merge_entries(base_entries: list[str], extra_entries: list[str]) -> list[str]:
    merged = list(base_entries)
    seen = {entry.strip() for entry in base_entries}
    for entry in extra_entries:
        stripped = entry.strip()
        if stripped not in seen:
            merged.append(stripped)
            seen.add(stripped)
    return merged


def _typed_entry_arity(entry: str) -> int | None:
    stripped = entry.strip()
    if not stripped.startswith("("):
        return None
    return stripped.count(" - ")


def _dedupe_type_entries_keep_first(entries: list[str]) -> list[str]:
    merged_comments: list[str] = []
    seen_raw_comments: set[str] = set()
    type_to_parent: dict[str, str] = {}
    type_order: list[str] = []

    def _parse_type_entry(entry: str) -> tuple[list[str], str] | None:
        stripped = entry.strip()
        if not stripped or stripped.startswith(";;"):
            return None
        if " - " not in stripped:
            return None
        left, right = stripped.rsplit(" - ", 1)
        names = [item for item in left.split() if item]
        parent = right.strip()
        if not names or not parent:
            return None
        return names, parent

    def _choose_parent(existing: str | None, candidate: str) -> str:
        if existing is None:
            return candidate
        if existing == candidate:
            return existing
        if existing == "object" and candidate != "object":
            return candidate
        if candidate == "object" and existing != "object":
            return existing
        return candidate

    for entry in entries:
        stripped = entry.strip()
        if not stripped:
            continue
        if stripped.startswith(";;"):
            if stripped not in seen_raw_comments:
                merged_comments.append(stripped)
                seen_raw_comments.add(stripped)
            continue
        parsed = _parse_type_entry(stripped)
        if parsed is None:
            continue
        type_names, parent = parsed
        for type_name in type_names:
            if type_name not in type_to_parent:
                type_order.append(type_name)
            type_to_parent[type_name] = _choose_parent(type_to_parent.get(type_name), parent)

    parent_to_names: dict[str, list[str]] = {}
    for type_name in type_order:
        parent = type_to_parent.get(type_name)
        if not parent:
            continue
        parent_to_names.setdefault(parent, []).append(type_name)

    merged: list[str] = list(merged_comments)
    emitted_entries: set[str] = set()
    for type_name in type_order:
        parent = type_to_parent.get(type_name)
        if not parent or parent not in parent_to_names:
            continue
        names = parent_to_names.pop(parent)
        entry = f"{' '.join(names)} - {parent}"
        if entry not in emitted_entries:
            merged.append(entry)
            emitted_entries.add(entry)
    return merged


def _dedupe_predicate_like_entries_keep_first(entries: list[str]) -> list[str]:
    merged: list[str] = []
    seen_raw: set[str] = set()
    seen_keys: set[tuple[str, int | None]] = set()
    pending_comments: list[str] = []
    for entry in entries:
        stripped = entry.strip()
        if not stripped:
            continue
        if stripped.startswith(";;"):
            pending_comments.append(stripped)
            continue
        predicate_name = _predicate_name_from_entry(stripped)
        arity = _typed_entry_arity(stripped)
        key = (predicate_name, arity) if predicate_name is not None else None
        if key is not None and key in seen_keys:
            pending_comments = []
            continue
        for comment in pending_comments:
            if comment not in seen_raw:
                merged.append(comment)
                seen_raw.add(comment)
        pending_comments = []
        if stripped not in seen_raw:
            merged.append(stripped)
            seen_raw.add(stripped)
        if key is not None:
            seen_keys.add(key)
    return merged


def _typed_entry_identity(entry: str) -> tuple[str, tuple[str, ...]] | None:
    stripped = entry.strip()
    if not stripped.startswith("("):
        return None
    predicate_name = _predicate_name_from_entry(stripped)
    if predicate_name is None:
        return None
    types = tuple(match.group(1) for match in re.finditer(r"-\s*([a-zA-Z][a-zA-Z0-9_-]*)", stripped))
    return (predicate_name, types)


def _merge_predicate_like_entries(base_entries: list[str], extra_entries: list[str]) -> list[str]:
    merged = list(base_entries)
    seen_raw = {entry.strip() for entry in base_entries}
    seen_typed = {identity for entry in base_entries if (identity := _typed_entry_identity(entry)) is not None}
    for entry in extra_entries:
        stripped = entry.strip()
        identity = _typed_entry_identity(stripped)
        if identity is not None:
            if identity in seen_typed:
                continue
            seen_typed.add(identity)
            merged.append(stripped)
            seen_raw.add(stripped)
            continue
        if stripped not in seen_raw:
            merged.append(stripped)
            seen_raw.add(stripped)
    return merged


def _predicate_name_from_entry(entry: str) -> str | None:
    match = _PREDICATE_ENTRY_NAME_RE.match(entry.strip())
    return match.group(1) if match else None


def _collect_predicate_names_from_expr(expr: SExpr | None, out: set[str]) -> None:
    if expr is None:
        return
    if isinstance(expr, str):
        return
    if not expr:
        return
    head = expr[0]
    if isinstance(head, str) and head not in _BUILTIN_OPS:
        out.add(head)
    for item in expr[1:]:
        _collect_predicate_names_from_expr(item, out)


def _prune_unused_predicate_entries(domain_text: str, predicate_entries: list[str]) -> list[str]:
    if not predicate_entries:
        return []
    parsed_domain = _parse_domain_raw(domain_text)
    used_predicates: set[str] = set()
    for action_schema in parsed_domain.actions:
        _collect_predicate_names_from_expr(action_schema.precondition, used_predicates)
        _collect_predicate_names_from_expr(action_schema.effect, used_predicates)
    for observation_schema in parsed_domain.observation_rules:
        _collect_predicate_names_from_expr(observation_schema.condition, used_predicates)
    pruned_entries: list[str] = []
    pending_comments: list[str] = []
    for entry in predicate_entries:
        if entry.strip().startswith(";;"):
            pending_comments.append(entry)
            continue
        predicate_name = _predicate_name_from_entry(entry)
        if predicate_name is None:
            pruned_entries.extend(pending_comments)
            pending_comments = []
            pruned_entries.append(entry)
            continue
        if predicate_name in used_predicates:
            pruned_entries.extend(pending_comments)
            pending_comments = []
            pruned_entries.append(entry)
            continue
        pending_comments = []
    return pruned_entries


def _extract_reset_clauses(observation_text: str) -> list[str]:
    clauses = re.findall(
        r"\(\s*forall\s*\([^)]*\)\s*\(not\s*\(last_action\b.*?\)\)\s*\)",
        observation_text,
        flags=re.DOTALL,
    )
    unique: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        normalized = " ".join(clause.split())
        if normalized not in seen:
            seen.add(normalized)
            unique.append(clause.strip())
    return unique


def _find_effect_span(action_text: str) -> tuple[int, int] | None:
    effect_idx = action_text.find(":effect")
    if effect_idx < 0:
        return None
    pos = effect_idx + len(":effect")
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
    return pos, end


def _indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def _find_keyword_expr_span(form_text: str, keyword: str) -> tuple[int, int] | None:
    marker = f":{keyword}"
    field_idx = form_text.find(marker)
    if field_idx < 0:
        return None
    pos = field_idx + len(marker)
    while pos < len(form_text) and form_text[pos].isspace():
        pos += 1
    if pos >= len(form_text) or form_text[pos] != "(":
        return None
    depth = 0
    end = pos
    while end < len(form_text):
        if form_text[end] == "(":
            depth += 1
        elif form_text[end] == ")":
            depth -= 1
            if depth == 0:
                end += 1
                break
        end += 1
    return pos, end


def _ensure_literal_in_keyword_expr(form_text: str, keyword: str, literal: str) -> str:
    expr_span = _find_keyword_expr_span(form_text, keyword)
    if expr_span is None:
        return form_text
    expr_start, expr_end = expr_span
    expr_text = form_text[expr_start:expr_end]
    normalized_expr = " ".join(expr_text.split())
    normalized_literal = " ".join(literal.split())
    if normalized_literal in normalized_expr:
        return form_text
    new_expr = f"(and\n        {literal}\n        {_indent_block(expr_text, '').strip()}\n    )"
    return form_text[:expr_start] + new_expr + form_text[expr_end:]


def _inject_reset_clauses_into_action(action_text: str, reset_clauses: list[str]) -> str:
    if not reset_clauses:
        return action_text
    effect_span = _find_effect_span(action_text)
    if effect_span is None:
        return action_text
    effect_start, effect_end = effect_span
    original_effect = action_text[effect_start:effect_end].strip()
    missing = [clause for clause in reset_clauses if clause not in original_effect]
    if not missing:
        return action_text
    reset_block = "\n".join(f"        {clause}" for clause in missing)
    new_effect = "(\n".join([])  # placeholder to satisfy lint? no
    new_effect = f"(and\n{reset_block}\n        {_indent_block(original_effect, '').strip()}\n    )"
    return action_text[:effect_start] + new_effect + action_text[effect_end:]


def _collect_helper_types(constant_entries: list[str], predicate_entries: list[str]) -> list[str]:
    types: set[str] = set()
    for entry in constant_entries + predicate_entries:
        for match in re.finditer(r"-\s*([a-zA-Z][a-zA-Z0-9_-]*)", entry):
            types.add(match.group(1))
    return sorted(types)


def _extract_commented_observation_templates(observation_text: str) -> list[str]:
    matches = re.findall(
        r"(?ms)^;; Observation rule `.*?(?:\n;;.*?)*?(?=\n\s*\n|\Z)",
        observation_text,
    )
    unique: list[str] = []
    seen: set[str] = set()
    for block in matches:
        normalized = block.strip()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _is_observation_placeholder_comment(form_text: str) -> bool:
    normalized = " ".join(form_text.strip().split())
    return normalized == ";; Observation-action learning is not implemented yet."


def _strip_observation_placeholder_comment(text: str) -> str:
    return _OBSERVATION_PLACEHOLDER_RE.sub("", text)


def prune_unused_predicates_in_domain(domain_text: str) -> str:
    domain_text = _strip_observation_placeholder_comment(domain_text)
    forms = _find_top_level_forms(domain_text)
    base_singletons = {form.keyword: form for form in forms if form.keyword in {"requirements"}}
    base_types = next((form for form in forms if form.keyword == "types"), None)
    base_constants = next((form for form in forms if form.keyword == "constants"), None)
    base_predicates = next((form for form in forms if form.keyword == "predicates"), None)
    base_observables = next((form for form in forms if form.keyword == "observables"), None)
    base_actions = [form for form in forms if form.keyword == "action"]
    base_observations = [form for form in forms if form.keyword == "observation"]
    base_other_forms = [
        form
        for form in forms
        if form.keyword
        not in {"requirements", "types", "constants", "predicates", "observables", "action", "observation"}
    ]

    header_match = re.match(r"\s*\(define\s*\(domain\s+[^\)]+\)\s*", domain_text)
    footer = "\n)\n"
    if header_match:
        header = header_match.group(0)
    else:
        header = "(define "

    type_entries = _extract_block_entries(base_types.text) if base_types else []
    constant_entries = _extract_block_entries(base_constants.text) if base_constants else []
    predicate_entries = _extract_block_entries(base_predicates.text) if base_predicates else []
    observable_entries = _extract_block_entries(base_observables.text) if base_observables else []

    assembled_without_predicates = (
        header
        + "\n\n".join(
            [
                *([base_singletons["requirements"].text.strip()] if "requirements" in base_singletons else []),
                *([_render_block("types", type_entries)] if type_entries else []),
                *([_render_block("constants", constant_entries)] if constant_entries else []),
                *([_render_block("predicates", predicate_entries)] if predicate_entries else []),
                *([_render_block("observables", observable_entries)] if observable_entries else []),
                *[form.text.strip() for form in base_other_forms],
                *[form.text.strip() for form in base_actions],
                *[form.text.strip() for form in base_observations],
            ]
        )
        + footer
    )

    pruned_predicate_entries = _prune_unused_predicate_entries(
        assembled_without_predicates,
        predicate_entries,
    )

    ordered_forms: list[str] = []
    if "requirements" in base_singletons:
        ordered_forms.append(base_singletons["requirements"].text.strip())
    if type_entries:
        ordered_forms.append(_render_block("types", type_entries))
    if constant_entries:
        ordered_forms.append(_render_block("constants", constant_entries))
    if pruned_predicate_entries:
        ordered_forms.append(_render_block("predicates", pruned_predicate_entries))
    if observable_entries:
        ordered_forms.append(_render_block("observables", observable_entries))
    ordered_forms.extend(form.text.strip() for form in base_other_forms)
    ordered_forms.extend(form.text.strip() for form in base_actions)
    ordered_forms.extend(form.text.strip() for form in base_observations)

    return header + "\n\n".join(ordered_forms) + footer


def merge_domain_with_observation_modules(base_domain_text: str, observation_module_texts: list[str]) -> str:
    base_forms = _find_top_level_forms(base_domain_text)
    if not base_forms:
        raise ValueError("Base domain text does not contain top-level domain forms.")
    normalized_module_texts = [text for text in observation_module_texts if str(text).strip()]
    if not normalized_module_texts:
        return base_domain_text
    parsed_module_forms: list[list[_TopLevelForm]] = []
    retained_module_texts: list[str] = []
    for observation_module_text in normalized_module_texts:
        observation_forms = _find_top_level_forms(f"(define (domain tmp)\n{observation_module_text}\n)")
        if not observation_forms:
            # An observation learner can legitimately emit a comment-only empty module.
            continue
        parsed_module_forms.append(observation_forms)
        retained_module_texts.append(observation_module_text)
    if not parsed_module_forms:
        return base_domain_text

    header = base_domain_text[: base_forms[0].start]
    footer = _strip_observation_placeholder_comment(base_domain_text[base_forms[-1].end :])

    base_singletons: dict[str, _TopLevelForm] = {}
    base_actions: list[_TopLevelForm] = []
    base_observations: list[_TopLevelForm] = []
    base_other_forms: list[_TopLevelForm] = []
    for form in base_forms:
        if form.keyword in {"requirements", "types", "constants", "predicates", "observables"}:
            base_singletons[form.keyword] = form
        elif form.keyword == "action":
            base_actions.append(form)
        elif form.keyword == "observation":
            base_observations.append(form)
        else:
            base_other_forms.append(form)

    base_type_entries = _dedupe_type_entries_keep_first(
        _extract_block_entries(base_singletons["types"].text) if "types" in base_singletons else []
    )
    base_constant_entries = (
        _extract_block_entries(base_singletons["constants"].text) if "constants" in base_singletons else []
    )
    base_predicate_entries = _dedupe_predicate_like_entries_keep_first(
        _extract_block_entries(base_singletons["predicates"].text) if "predicates" in base_singletons else []
    )
    base_observable_entries = _dedupe_predicate_like_entries_keep_first(
        _extract_block_entries(base_singletons["observables"].text) if "observables" in base_singletons else []
    )
    merged_constant_entries = list(base_constant_entries)
    merged_predicate_entries = list(base_predicate_entries)
    merged_observable_entries = list(base_observable_entries)
    helper_types: list[str] = []
    observation_actions: list[_TopLevelForm] = []
    observation_observations: list[_TopLevelForm] = []
    commented_observation_templates: list[str] = []
    reset_clauses: list[str] = []
    for observation_module_text, observation_forms in zip(retained_module_texts, parsed_module_forms):
        observation_singletons: dict[str, _TopLevelForm] = {}
        for form in observation_forms:
            if form.keyword in {"constants", "predicates", "observables"}:
                observation_singletons[form.keyword] = form
            elif form.keyword == "action":
                observation_actions.append(form)
            elif form.keyword == "observation":
                observation_observations.append(form)
        observation_constant_entries = (
            _extract_block_entries(observation_singletons["constants"].text)
            if "constants" in observation_singletons
            else []
        )
        observation_predicate_entries = (
            _extract_block_entries(observation_singletons["predicates"].text)
            if "predicates" in observation_singletons
            else []
        )
        observation_observable_entries = (
            _extract_block_entries(observation_singletons["observables"].text)
            if "observables" in observation_singletons
            else []
        )
        merged_constant_entries = _merge_entries(merged_constant_entries, observation_constant_entries)
        merged_predicate_entries = _merge_predicate_like_entries(
            merged_predicate_entries, observation_predicate_entries
        )
        merged_observable_entries = _merge_predicate_like_entries(
            merged_observable_entries, observation_observable_entries
        )
        helper_types = _merge_entries(
            helper_types, _collect_helper_types(observation_constant_entries, observation_predicate_entries)
        )
        commented_observation_templates = _merge_entries(
            commented_observation_templates,
            _extract_commented_observation_templates(observation_module_text),
        )
        reset_clauses = _merge_entries(reset_clauses, _extract_reset_clauses(observation_module_text))

    merged_type_entries = list(base_type_entries)
    existing_type_names = {entry.split()[0] for entry in base_type_entries if entry.split()}
    for type_name in helper_types:
        if type_name in {"object", "container"}:
            continue
        if type_name not in existing_type_names:
            merged_type_entries.append(f"{type_name} - object")
            existing_type_names.add(type_name)
    merged_type_entries = _dedupe_type_entries_keep_first(merged_type_entries)
    merged_predicate_entries = _dedupe_predicate_like_entries_keep_first(merged_predicate_entries)
    merged_observable_entries = _dedupe_predicate_like_entries_keep_first(merged_observable_entries)

    observation_action_names = {
        _form_name(form.text, "action") for form in observation_actions if _form_name(form.text, "action") is not None
    }

    merged_action_texts: list[str] = []
    for action_form in base_actions:
        action_name = _form_name(action_form.text, "action")
        if action_name in observation_action_names:
            merged_action_texts.append(action_form.text.strip())
        else:
            merged_action_texts.append(_inject_reset_clauses_into_action(action_form.text, reset_clauses).strip())

    for form in observation_actions:
        action_name = _form_name(form.text, "action") or ""
        action_text = form.text.strip()
        if action_name.startswith("active_obs"):
            action_text = _ensure_literal_in_keyword_expr(
                action_text,
                "precondition",
                "(gripper_empty)",
            )
        merged_action_texts.append(action_text)

    merged_observation_texts = [form.text.strip() for form in base_observations]
    for form in observation_observations:
        observation_name = _form_name(form.text, "observation") or ""
        observation_text = form.text.strip()
        if observation_name.startswith("after_active_obs"):
            observation_text = _ensure_literal_in_keyword_expr(
                observation_text,
                "condition",
                "(gripper_empty)",
            )
        merged_observation_texts.append(observation_text)
    merged_observation_texts.extend(commented_observation_templates)
    other_form_texts = [
        form.text.strip() for form in base_other_forms if not _is_observation_placeholder_comment(form.text)
    ]

    ordered_forms: list[str] = []
    if "requirements" in base_singletons:
        ordered_forms.append(base_singletons["requirements"].text.strip())
    if merged_type_entries:
        ordered_forms.append(_render_block("types", merged_type_entries))
    if merged_constant_entries:
        ordered_forms.append(_render_block("constants", merged_constant_entries))
    if merged_predicate_entries:
        ordered_forms.append(_render_block("predicates", merged_predicate_entries))
    if merged_observable_entries:
        ordered_forms.append(_render_block("observables", merged_observable_entries))
    ordered_forms.extend(other_form_texts)
    ordered_forms.extend(merged_action_texts)
    ordered_forms.extend(merged_observation_texts)

    return header + "\n\n".join(ordered_forms) + footer


def merge_domain_with_observation_module(base_domain_text: str, observation_module_text: str) -> str:
    return merge_domain_with_observation_modules(base_domain_text, [observation_module_text])
