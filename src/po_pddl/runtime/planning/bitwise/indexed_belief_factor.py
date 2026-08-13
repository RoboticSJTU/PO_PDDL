"""Index-based local factor for bitwise factorized beliefs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IndexedBeliefFactor:
    """A local factor over grounded-predicate indices.

    Each case is represented as:
    - probability: float
    - true_indices: list[int]

    Indices in ``scope`` that are not listed in a sampled case are treated as
    false for that case.
    """

    scope: list[int] = field(default_factory=list)
    cases: list[tuple[float, list[int]]] = field(default_factory=list)

    def validate(self, *, grounded_predicates_count: int, tolerance: float = 1e-9) -> None:
        """Validate internal consistency of the indexed factor."""
        scope_set = set(self.scope)
        if len(scope_set) != len(self.scope):
            raise ValueError("IndexedBeliefFactor has duplicate indices in scope.")
        for index in self.scope:
            if index < 0 or index >= grounded_predicates_count:
                raise ValueError(
                    f"IndexedBeliefFactor scope index `{index}` is outside the grounded predicate range."
                )
        if not self.cases:
            raise ValueError("IndexedBeliefFactor must contain at least one case.")

        prob_sum = 0.0
        for probability, true_indices in self.cases:
            if probability < 0.0:
                raise ValueError("IndexedBeliefFactor contains a negative probability.")
            true_set = set(true_indices)
            if len(true_set) != len(true_indices):
                raise ValueError("IndexedBeliefFactor contains duplicate indices in a case.")
            if not true_set.issubset(scope_set):
                raise ValueError("IndexedBeliefFactor contains a case index outside its scope.")
            for index in true_indices:
                if index < 0 or index >= grounded_predicates_count:
                    raise ValueError(
                        f"IndexedBeliefFactor case index `{index}` is outside the grounded predicate range."
                    )
            prob_sum += probability

        if abs(prob_sum - 1.0) > tolerance:
            raise ValueError(
                f"IndexedBeliefFactor probabilities sum to {prob_sum}, not 1.0."
            )

    def case_count(self) -> int:
        """Return the number of mutually exclusive cases in the factor."""
        return len(self.cases)
