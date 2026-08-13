"""Observation rule data structure."""

from __future__ import annotations

from dataclasses import dataclass, field

from .observable import Observable


@dataclass(frozen=True)
class ObservationRule:
    """Representation of an observation rule."""

    name: str
    distribution: list[Observable] = field(default_factory=list)

    def to_pddl_str(self) -> str:
        """Render this observation rule in PDDL-like syntax."""
        return f"({self.name})"

    def __str__(self) -> str:
        return self.to_pddl_str()

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.distribution)))
