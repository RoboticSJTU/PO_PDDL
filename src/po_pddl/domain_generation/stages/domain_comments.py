from __future__ import annotations

from dataclasses import dataclass

from po_pddl.core.parser import parse_domain


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


def _extract_leading_comment(text: str, block_start: int) -> str | None:
    line_start = text.rfind("\n", 0, block_start) + 1
    comments: list[str] = []
    cursor = line_start
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
        cursor = prev_line_start
    combined = " ".join(part for part in comments if part)
    return combined or None


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
            predicate_name = stripped[1:].split()[0]
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
        comment = _extract_leading_comment(text, span.start)
        if comment:
            comments[parsed_action.action.name] = comment
    return comments


__all__ = [
    "extract_action_comments",
    "extract_predicate_comments",
]
