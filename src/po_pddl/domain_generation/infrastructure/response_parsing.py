from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_CODE_FENCE_PATTERN = re.compile(r"^```(?:[a-z0-9_+-]+)?\s*\n?", flags=re.IGNORECASE)


def strip_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _CODE_FENCE_PATTERN.sub("", stripped, count=1)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        stripped = stripped.strip()
    return stripped


def load_text(path: str | Path | None, default_path: Path, *, label: str) -> str:
    selected = Path(path) if path else default_path
    if not selected.exists():
        raise FileNotFoundError(f"{label} not found at {selected}")
    return selected.read_text(encoding="utf-8")


def _escape_control_characters_inside_json_strings(text: str) -> str:
    repaired: list[str] = []
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                repaired.append(char)
                escape = False
                continue
            if char == "\\":
                repaired.append(char)
                escape = True
                continue
            if char == '"':
                repaired.append(char)
                in_string = False
                continue
            if char == "\n":
                repaired.append("\\n")
                continue
            if char == "\r":
                repaired.append("\\r")
                continue
            if char == "\t":
                repaired.append("\\t")
                continue
            repaired.append(char)
            continue
        repaired.append(char)
        if char == '"':
            in_string = True
            escape = False
    return "".join(repaired)


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = strip_markdown_code_fence(text)

    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= 0 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        parsed = _try_parse_json_object(candidate)
        if parsed is not None:
            return parsed
        repaired_candidate = _escape_control_characters_inside_json_strings(candidate)
        parsed = _try_parse_json_object(repaired_candidate)
        if parsed is not None:
            return parsed

    raise ValueError(f"LLM response does not contain a valid JSON object.\nPreview:\n{text[:2000]}")


def extract_domain_text(text: str) -> str:
    stripped = strip_markdown_code_fence(text)

    define_index = stripped.find("(define")
    if define_index >= 0:
        stripped = stripped[define_index:].strip()

    if not stripped.startswith("(define"):
        raise ValueError(f"LLM response does not contain a domain `(define ...)` form.\nPreview:\n{text[:2000]}")
    return stripped
