"""Factorized belief data structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .belief_factor import BeliefFactor
from .predicate import Predicate


@dataclass
class FactorizedBelief:
    """A predicate-centered factorized belief.

    The belief is composed of:
    - known_true: predicates that are certainly true
    - known_false: predicates that are certainly false
    - factors: independent local factors over uncertain predicates
    """

    known_true: list[Predicate] = field(default_factory=list)
    known_false: list[Predicate] = field(default_factory=list)
    factors: list[BeliefFactor] = field(default_factory=list)

    def validate(self, *, tolerance: float = 1e-3) -> None:
        """Validate the overall belief structure."""
        true_set = set(self.known_true)
        false_set = set(self.known_false)

        if len(true_set) != len(self.known_true):
            raise ValueError("FactorizedBelief contains duplicate predicates in known_true.")
        if len(false_set) != len(self.known_false):
            raise ValueError("FactorizedBelief contains duplicate predicates in known_false.")
        if true_set & false_set:
            raise ValueError("FactorizedBelief contains predicates that appear in both known_true and known_false.")

        covered: set[Predicate] = set()
        for factor in self.factors:
            factor.validate(tolerance=tolerance)

            factor_scope = set(factor.scope)
            if factor_scope & true_set:
                raise ValueError(f"Belief factor `{factor.name}` overlaps with known_true predicates.")
            if factor_scope & false_set:
                raise ValueError(f"Belief factor `{factor.name}` overlaps with known_false predicates.")
            if factor_scope & covered:
                raise ValueError(f"Belief factor `{factor.name}` overlaps with another factor scope.")
            covered |= factor_scope

    def all_uncertain_predicates(self) -> list[Predicate]:
        """Return the concatenated uncertain predicate scope."""
        predicates: list[Predicate] = []
        for factor in self.factors:
            predicates.extend(factor.scope)
        return predicates

    def to_json(
        self,
        *,
        grounded_predicates: list[Predicate] | None = None,
        predicate_index: dict[Predicate, int] | None = None,
        grounded_predicates_count: int | None = None,
    ) -> dict[str, Any]:
        """Convert this belief into the JSON structure used by bitwise ``init_belief.json``.

        Provide either:
        - ``grounded_predicates``: the ordered grounded predicate list whose index order
          matches the target bitwise model, or
        - ``predicate_index``: an explicit ``Predicate -> index`` mapping.
        """

        if grounded_predicates is None and predicate_index is None:
            raise ValueError("FactorizedBelief.to_json requires either grounded_predicates or predicate_index.")
        if grounded_predicates is not None and predicate_index is not None:
            raise ValueError("FactorizedBelief.to_json accepts grounded_predicates or predicate_index, not both.")

        self.validate()

        if grounded_predicates is not None:
            predicate_index = {predicate: index for index, predicate in enumerate(grounded_predicates)}
            if grounded_predicates_count is None:
                grounded_predicates_count = len(grounded_predicates)
        else:
            assert predicate_index is not None
            if grounded_predicates_count is None:
                grounded_predicates_count = len(predicate_index)

        def _index_of(predicate: Predicate) -> int:
            try:
                return predicate_index[predicate]  # type: ignore[index]
            except KeyError as exc:
                raise ValueError(
                    f"Predicate `{predicate}` does not exist in the provided grounded predicate index."
                ) from exc

        return {
            "grounded_predicates_count": int(grounded_predicates_count),
            "known_true": [_index_of(predicate) for predicate in self.known_true],
            "known_false": [_index_of(predicate) for predicate in self.known_false],
            "factors": [
                {
                    "scope": [_index_of(predicate) for predicate in factor.scope],
                    "cases": [
                        {
                            "probability": probability,
                            "true_indices": [_index_of(predicate) for predicate in true_predicates],
                        }
                        for probability, true_predicates in factor.cases
                    ],
                }
                for factor in self.factors
            ],
        }
