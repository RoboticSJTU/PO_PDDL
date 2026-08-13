"""Index-based particle belief structure for bitwise models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IndexedParticleBelief:
    """A weighted particle belief represented with grounded-predicate bitvectors."""

    grounded_predicates_count: int = 0
    particles: list[tuple[int, float]] = field(default_factory=list)

    def validate(self, *, tolerance: float = 1e-9) -> None:
        """Validate particle weights and bit ranges."""
        if self.grounded_predicates_count < 0:
            raise ValueError("IndexedParticleBelief grounded_predicates_count must be non-negative.")
        if not self.particles:
            raise ValueError("IndexedParticleBelief must contain at least one particle.")

        total_probability = 0.0
        seen_bits: set[int] = set()
        max_valid_bits = (
            (1 << self.grounded_predicates_count) - 1
            if self.grounded_predicates_count > 0
            else 0
        )
        for state_bits, probability in self.particles:
            if state_bits < 0:
                raise ValueError("IndexedParticleBelief contains a negative bitvector.")
            if self.grounded_predicates_count >= 0 and state_bits & ~max_valid_bits:
                raise ValueError(
                    "IndexedParticleBelief contains state bits outside the grounded predicate range."
                )
            if state_bits in seen_bits:
                raise ValueError("IndexedParticleBelief contains duplicate particle states.")
            if probability < 0.0:
                raise ValueError("IndexedParticleBelief contains a negative probability.")
            seen_bits.add(state_bits)
            total_probability += probability

        if abs(total_probability - 1.0) > tolerance:
            raise ValueError(
                f"IndexedParticleBelief probabilities sum to {total_probability}, not 1.0."
            )

    def particle_count(self) -> int:
        """Return the number of weighted support states."""
        return len(self.particles)
