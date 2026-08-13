"""Index-based factorized belief structure for bitwise models."""

from __future__ import annotations

from dataclasses import dataclass, field

from .indexed_belief_factor import IndexedBeliefFactor


@dataclass
class IndexedFactorizedBelief:
    """A factorized belief represented only with grounded-predicate indices."""

    grounded_predicates_count: int = 0
    known_true: list[int] = field(default_factory=list)
    known_false: list[int] = field(default_factory=list)
    factors: list[IndexedBeliefFactor] = field(default_factory=list)

    def validate(self, *, tolerance: float = 1e-9) -> None:
        """Validate the indexed belief structure."""
        true_set = set(self.known_true)
        false_set = set(self.known_false)

        if len(true_set) != len(self.known_true):
            raise ValueError("IndexedFactorizedBelief contains duplicate indices in known_true.")
        if len(false_set) != len(self.known_false):
            raise ValueError("IndexedFactorizedBelief contains duplicate indices in known_false.")
        if true_set & false_set:
            raise ValueError(
                "IndexedFactorizedBelief contains indices that appear in both known_true and known_false."
            )
        for index in [*self.known_true, *self.known_false]:
            if index < 0 or index >= self.grounded_predicates_count:
                raise ValueError(
                    f"IndexedFactorizedBelief index `{index}` is outside the grounded predicate range."
                )

        covered: set[int] = set()
        for factor in self.factors:
            factor.validate(
                grounded_predicates_count=self.grounded_predicates_count,
                tolerance=tolerance,
            )

            factor_scope = set(factor.scope)
            if factor_scope & true_set:
                raise ValueError("Indexed belief factor scope overlaps with known_true.")
            if factor_scope & false_set:
                raise ValueError("Indexed belief factor scope overlaps with known_false.")
            if factor_scope & covered:
                raise ValueError("Indexed belief factor scope overlaps with another factor scope.")
            covered |= factor_scope

    def all_uncertain_indices(self) -> list[int]:
        """Return the concatenated uncertain predicate indices."""
        indices: list[int] = []
        for factor in self.factors:
            indices.extend(factor.scope)
        return indices
