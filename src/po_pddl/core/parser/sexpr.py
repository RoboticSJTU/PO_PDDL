"""S-expression parser for PDDL/POMDPDDL-style syntax."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .tokenizer import tokenize

SExpr: TypeAlias = str | list["SExpr"]


@dataclass
class ParseError(Exception):
    """Raised when an S-expression text cannot be parsed."""

    message: str

    def __str__(self) -> str:
        return self.message


def loads_sexpr(text: str) -> list[SExpr]:
    """Parse a text buffer into a list of top-level S-expressions."""

    tokens = tokenize(text)
    stack: list[list[SExpr]] = []
    roots: list[SExpr] = []

    for token in tokens:
        if token == "(":
            new_list: list[SExpr] = []
            if stack:
                stack[-1].append(new_list)
            else:
                roots.append(new_list)
            stack.append(new_list)
            continue

        if token == ")":
            if not stack:
                raise ParseError("Unexpected `)` while parsing S-expression.")
            stack.pop()
            continue

        if stack:
            stack[-1].append(token)
        else:
            roots.append(token)

    if stack:
        raise ParseError("Unclosed `(` while parsing S-expression.")

    return roots
