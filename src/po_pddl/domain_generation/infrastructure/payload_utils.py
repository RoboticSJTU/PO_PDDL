from __future__ import annotations

import re

_SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text


def validate_snake_case(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not _SNAKE_CASE_PATTERN.fullmatch(cleaned):
        raise ValueError(f"Expected snake_case for {field_name}, got {value!r}")
    return cleaned


def lookup_first(data: dict[str, object], keys: list[str]) -> object | None:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None
