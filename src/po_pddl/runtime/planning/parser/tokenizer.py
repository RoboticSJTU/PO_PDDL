"""Tokenizer for PDDL/POMDPDDL-style S-expressions."""

from __future__ import annotations


def tokenize(text: str) -> list[str]:
    """Tokenize a PDDL/POMDPDDL text buffer into a flat token list.

    Rules:
    - ``;`` starts a comment that runs to the end of the line
    - ``(`` and ``)`` are emitted as standalone tokens
    - all other non-whitespace runs are emitted as symbol tokens
    """

    tokens: list[str] = []
    current: list[str] = []
    in_comment = False

    def flush_current() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for char in text:
        if in_comment:
            if char == "\n":
                in_comment = False
            continue

        if char == ";":
            flush_current()
            in_comment = True
            continue

        if char.isspace():
            flush_current()
            continue

        if char in ("(", ")"):
            flush_current()
            tokens.append(char)
            continue

        current.append(char)

    flush_current()
    return tokens
