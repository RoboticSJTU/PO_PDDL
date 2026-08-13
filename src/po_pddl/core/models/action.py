"""Action data structure."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Action:
    """Representation of an action with a name and string parameters."""

    name: str
    params: list[str] = field(default_factory=list)

    def to_pddl_str(self) -> str:
        """Render this action in PDDL-like syntax."""
        if not self.params:
            return f"({self.name})"
        return f"({self.name} {' '.join(self.params)})"

    def __str__(self) -> str:
        return self.to_pddl_str()

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.params)))
