"""Predicate data class."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Predicate:
    """Representation of a predicate with a name and string parameters."""

    name: str
    params: list[str]

    def to_pddl_str(self) -> str:
        if not self.params:
            return f"({self.name})"
        return f"({self.name} {' '.join(self.params)})"

    def __str__(self) -> str:
        return self.to_pddl_str()

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.params)))

    def same_signature(self, other: "Predicate") -> bool:
        """Return True when both predicates share the same name and arity."""
        return self.name == other.name and len(self.params) == len(other.params)
