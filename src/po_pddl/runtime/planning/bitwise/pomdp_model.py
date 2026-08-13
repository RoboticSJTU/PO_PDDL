"""Bitwise-only abstract POMDP model interface."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..data_structures.effect_bucket import EffectBucket
from .aliases import BitVector, ObservationEntry, StateEntry


@dataclass
class POMDPModelBase(ABC):
    """Abstract base class for bitwise-only POMDP models.

    This interface intentionally carries no semantic-domain objects such as
    predicates, observables, actions, or observation rules. Implementations
    are expected to encode everything using bit vectors, masks, and integer
    identifiers.
    """

    grounded_predicates_count: int = 0
    grounded_observables_count: int = 0
    total_actions: int = 0
    maximize_reward: bool = True
    goal_reward: float = 0.0
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)
    _last_transition_reward_cache: dict = field(default_factory=dict, init=False, repr=False)
    _last_effect_bucket: EffectBucket | None = field(default=None, init=False, repr=False)
    _pending_effect_bucket_filter: EffectBucket | None = field(default=None, init=False, repr=False)

    def clear_last_effect_bucket(self) -> None:
        self._last_effect_bucket = None

    def get_last_effect_bucket(self) -> EffectBucket | None:
        return self._last_effect_bucket

    def set_effect_bucket_filter(
        self,
        *,
        bucket_name: str | None = None,
        success: bool | None = None,
        branch_index: int | None = None,
        variant_rank: int | None = None,
    ) -> None:
        self._pending_effect_bucket_filter = EffectBucket(
            bucket_name=bucket_name,
            success=success,
            branch_index=branch_index,
            variant_rank=variant_rank,
        )

    def clear_effect_bucket_filter(self) -> None:
        self._pending_effect_bucket_filter = None

    def _select_effect_branch_index(
        self,
        *,
        action: int,
        weights: list[float],
        bucket_names: list[str | None] | None = None,
        successes: list[bool | None] | None = None,
        branch_indices: list[int | None] | None = None,
        variant_ranks: list[int | None] | None = None,
        record_sample: bool = True,
    ) -> int | None:
        cleaned = [max(weight, 0.0) for weight in weights]
        candidate_indices = [index for index, weight in enumerate(cleaned) if weight > 0.0]
        pending_filter = self._pending_effect_bucket_filter
        if (
            pending_filter is not None
            and bucket_names is not None
            and successes is not None
        ):
            matching_indices = [
                index
                for index in candidate_indices
                if (
                    pending_filter.bucket_name is None
                    or bucket_names[index] == pending_filter.bucket_name
                )
                and (
                    pending_filter.success is None
                    or successes[index] == pending_filter.success
                )
                and (
                    pending_filter.branch_index is None
                    or (
                        branch_indices[index] if branch_indices is not None else index
                    ) == pending_filter.branch_index
                )
                and (
                    pending_filter.variant_rank is None
                    or (
                        variant_ranks[index] if variant_ranks is not None else None
                    ) == pending_filter.variant_rank
                )
            ]
            if matching_indices:
                candidate_indices = matching_indices
                self._pending_effect_bucket_filter = None
        total = sum(cleaned[index] for index in candidate_indices)
        if total <= 0.0:
            return None
        draw = self._rng.random() * total
        cumulative = 0.0
        selected_index = candidate_indices[-1]
        for index in candidate_indices:
            cumulative += cleaned[index]
            if draw <= cumulative:
                selected_index = index
                break
        if record_sample:
            bucket_name = bucket_names[selected_index] if bucket_names is not None else None
            success = successes[selected_index] if successes is not None else None
            branch_index = (
                branch_indices[selected_index]
                if branch_indices is not None
                else selected_index
            )
            self._last_effect_bucket = EffectBucket(
                bucket_name=bucket_name,
                success=success,
                branch_index=branch_index,
                variant_rank=(
                    variant_ranks[selected_index]
                    if variant_ranks is not None
                    else None
                ),
            )
        return selected_index

    @abstractmethod
    def is_goal(self, state: StateEntry) -> bool:
        """Return whether the given bitwise state satisfies the goal."""

    @abstractmethod
    def check_action_precondition(self, action: int, state: StateEntry) -> bool:
        """Return whether the grounded action identifier is legal in the given state."""

    @abstractmethod
    def forward_action(
        self,
        action: int,
        state: StateEntry,
    ) -> tuple[StateEntry, float]:
        """Apply one action step and return `(next_state, step_reward)`."""

    @abstractmethod
    def get_action_reward(
        self,
        action: int,
        state: StateEntry,
        next_state: StateEntry,
    ) -> float:
        """Return the one-step reward for a transition induced by an action."""

    def is_active_perception_action(self, action: int | None) -> bool:
        """Return whether the grounded action id should be treated as active perception."""

        return False

    def should_skip_observation_rule_before_check(
        self,
        observation_rule: int,
        current_action: int | None = None,
    ) -> bool:
        """Return whether a before-observation rule should be skipped for this step."""

        return False

    @abstractmethod
    def check_observation_rule_condition(
        self,
        observation_rule: int,
        state: StateEntry,
        current_action: int | None = None,
    ) -> bool:
        """Return whether an observation rule identifier is active in the given state."""

    @abstractmethod
    def observe_with_rule(
        self,
        observation_rule: int,
        state: StateEntry,
        current_action: int | None = None,
    ) -> tuple[ObservationEntry, BitVector]:
        """Return `(observation, mask)` for one observation rule.

        `observation` is a bit vector and `mask` indicates which observation bits
        are explicitly produced by the selected rule.
        """

    @abstractmethod
    def check_default_policy_rule_condition(
        self,
        default_policy_rule: int,
        state: StateEntry,
    ) -> bool:
        """Return whether a default-policy rule identifier is active in the given state."""

    @abstractmethod
    def get_default_policy_rule_action(
        self,
        default_policy_rule: int,
        state: StateEntry,
    ) -> int:
        """Return the grounded action identifier selected by one default-policy rule."""
