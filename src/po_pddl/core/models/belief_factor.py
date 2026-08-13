"""Factor definition for a predicate-centered factorized belief."""

from __future__ import annotations

from dataclasses import dataclass, field

from .predicate import Predicate


@dataclass
class BeliefFactor:
    """A local factor over a scope of predicates.

    Each case is represented as:
    - probability: float
    - true_predicates: list[Predicate]

    Predicates in ``scope`` that are not listed in a sampled case are treated
    as false for that case.
    """

    name: str
    scope: list[Predicate] = field(default_factory=list)
    cases: list[tuple[float, list[Predicate]]] = field(default_factory=list)

    def validate(self, *, tolerance: float = 1e-3) -> None:
        """Validate internal consistency of the factor."""
        scope_set = set(self.scope)
        if len(scope_set) != len(self.scope):
            raise ValueError(f"Belief factor `{self.name}` has duplicate predicates in scope.")

        if not self.cases:
            raise ValueError(f"Belief factor `{self.name}` must contain at least one case.")

        prob_sum = 0.0
        for probability, true_predicates in self.cases:
            if probability < 0.0:
                raise ValueError(f"Belief factor `{self.name}` contains a negative probability.")
            true_set = set(true_predicates)
            if len(true_set) != len(true_predicates):
                raise ValueError(f"Belief factor `{self.name}` contains duplicate predicates in a case.")
            if not true_set.issubset(scope_set):
                raise ValueError(f"Belief factor `{self.name}` contains a case predicate outside its scope.")
            prob_sum += probability

        if abs(prob_sum - 1.0) > tolerance:
            raise ValueError(f"Belief factor `{self.name}` probabilities sum to {prob_sum}, not 1.0.")

    def case_count(self) -> int:
        """Return the number of mutually exclusive cases in the factor."""
        return len(self.cases)
