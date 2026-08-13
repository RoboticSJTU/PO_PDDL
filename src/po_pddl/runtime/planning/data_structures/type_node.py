"""Type hierarchy node."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TypeNode:
    """Tree node used to represent the type hierarchy."""

    name: str
    parent: Optional["TypeNode"] = None
    children: list["TypeNode"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.parent is not None and self not in self.parent.children:
            self.parent.children.append(self)

    def add_child(self, child: "TypeNode") -> None:
        """Attach ``child`` under the current type node."""
        if child not in self.children:
            self.children.append(child)
        child.parent = self

    def ancestors(self) -> list["TypeNode"]:
        """Return the ancestor chain from parent to root."""
        chain: list[TypeNode] = []
        node = self.parent
        while node is not None:
            chain.append(node)
            node = node.parent
        return chain

    def descendants(self) -> list["TypeNode"]:
        """Return all descendants in depth-first order."""
        nodes: list[TypeNode] = []
        for child in self.children:
            nodes.append(child)
            nodes.extend(child.descendants())
        return nodes

    def is_subtype_of(self, other: "TypeNode | str") -> bool:
        """Check whether the current node is a subtype of ``other``."""
        other_name = other.name if isinstance(other, TypeNode) else other
        if self.name == other_name:
            return True
        node = self.parent
        while node is not None:
            if node.name == other_name:
                return True
            node = node.parent
        return False
