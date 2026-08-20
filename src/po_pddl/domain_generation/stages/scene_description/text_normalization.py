"""Conservative normalization for generated scene descriptions."""

from __future__ import annotations

import re


def remove_unseen_object_statements(text: str, object_names: list[str]) -> str:
    """Remove allowlist-induced claims that an object is absent from the view."""

    normalized = text.strip()
    for object_name in sorted(set(object_names), key=len, reverse=True):
        escaped_name = re.escape(str(object_name).strip())
        if not escaped_name:
            continue
        unseen = (
            rf"(?:the\s+)?{escaped_name}\s+"
            rf"(?:is|was)\s+(?:not\s+visible|absent|not\s+seen|out\s+of\s+view)"
        )
        normalized = re.sub(
            rf"(?:\s*[,;]\s*(?:and\s+|but\s+)?)?{unseen}",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            rf"(?:^|(?<=[.!?])\s+){unseen}[.!?]?",
            " ",
            normalized,
            flags=re.IGNORECASE,
        )
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,;.!?])", r"\1", normalized)
    normalized = re.sub(r"[,;]\s*([.!?])", r"\1", normalized)
    normalized = re.sub(r"([.!?])\1+", r"\1", normalized)
    return normalized.strip()


__all__ = ["remove_unseen_object_statements"]
