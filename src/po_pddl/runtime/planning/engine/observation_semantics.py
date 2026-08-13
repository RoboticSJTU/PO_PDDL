"""Shared observation normalization helpers for special POMDPDDL semantics."""

from __future__ import annotations

from ..data_structures.aliases import ObservationEntry
from ..data_structures.observable import Observable


EMPTY_OBSERVABLE_NAME = "obs-nothing"


def normalize_semantic_observation_entry(
    observation: ObservationEntry,
) -> ObservationEntry:
    """Apply special observation semantics to one semantic observation map.

    `(obs-nothing)` is a global fallback signal. When any other positive
    observation is present, `(obs-nothing)` must disappear from the merged
    observation.
    """

    if not observation:
        return {}

    has_non_empty_positive = any(
        value and observable.name != EMPTY_OBSERVABLE_NAME
        for observable, value in observation.items()
    )
    if not has_non_empty_positive:
        return dict(observation)

    return {
        observable: value
        for observable, value in observation.items()
        if observable.name != EMPTY_OBSERVABLE_NAME
    }


def normalize_bitwise_observation_bits(
    observation_bits: int,
    observation_mask: int,
    observables: list[Observable],
) -> tuple[int, int]:
    """Apply the `(obs-nothing)` fallback semantics to bitwise observations."""

    empty_index = next(
        (index for index, observable in enumerate(observables) if observable.name == EMPTY_OBSERVABLE_NAME),
        None,
    )
    if empty_index is None:
        return observation_bits, observation_mask

    empty_bit = 1 << empty_index
    if not (observation_mask & empty_bit and observation_bits & empty_bit):
        return observation_bits, observation_mask

    other_positive_bits = observation_bits & ~empty_bit
    if other_positive_bits == 0:
        return observation_bits, observation_mask

    return observation_bits & ~empty_bit, observation_mask & ~empty_bit
