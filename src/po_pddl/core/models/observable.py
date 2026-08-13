"""Observable data class."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Observable:
    """Representation of an observable with a name and string parameters."""

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
