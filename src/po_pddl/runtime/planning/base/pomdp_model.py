"""POMDP model base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random

from ..data_structures.action import Action
from ..data_structures.aliases import ObservationEntry, StateEntry
from ..data_structures.default_policy_rule import DefaultPolicyRule
from ..data_structures.effect_bucket import EffectBucket
from ..data_structures.observable import Observable
from ..data_structures.observation_rule import ObservationRule
from ..data_structures.predicate import Predicate


@dataclass
class POMDPModelBase(ABC):
    """Minimal base container plus execution interfaces for a POMDP model."""

    predicates: list[Predicate] = field(default_factory=list)
    observables: list[Observable] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    observation_rules: list[ObservationRule] = field(default_factory=list)
    default_policy_rules: list[DefaultPolicyRule] = field(default_factory=list)
    maximize_reward: bool = True
    goal_reward: float = 0.0
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)
    _last_transition_reward_cache: dict = field(default_factory=dict, init=False, repr=False)
    _last_effect_bucket: EffectBucket | None = field(default=None, init=False, repr=False)
    _pending_effect_bucket_filter: EffectBucket | None = field(default=None, init=False, repr=False)

    def clear_last_effect_bucket(self) -> None:
        """Clear the last sampled top-level effect bucket metadata."""

        self._last_effect_bucket = None

    def get_last_effect_bucket(self) -> EffectBucket | None:
        """Return the last sampled top-level effect bucket metadata."""

        return self._last_effect_bucket

    def set_effect_bucket_filter(
        self,
        *,
        bucket_name: str | None = None,
        success: bool | None = None,
        branch_index: int | None = None,
        variant_rank: int | None = None,
    ) -> None:
        """Restrict the next annotated top-level effect sample to one bucket family."""

        self._pending_effect_bucket_filter = EffectBucket(
            bucket_name=bucket_name,
            success=success,
            branch_index=branch_index,
            variant_rank=variant_rank,
        )

    def clear_effect_bucket_filter(self) -> None:
        """Clear any pending effect-bucket filter."""

        self._pending_effect_bucket_filter = None

    def _select_effect_branch_index(
        self,
        *,
        action: Action,
        weights: list[float],
        bucket_names: list[str | None] | None = None,
        successes: list[bool | None] | None = None,
        branch_indices: list[int | None] | None = None,
        variant_ranks: list[int | None] | None = None,
        record_sample: bool = True,
    ) -> int | None:
        """Sample one effect branch, optionally conditioning on bucket metadata."""

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
        """Return True if ``state`` satisfies the goal condition."""

    @abstractmethod
    def check_action_precondition(self, action: Action, state: StateEntry) -> bool:
        """Return True if ``action`` is applicable in ``state``."""

    @abstractmethod
    def forward_action(
        self, action: Action, state: StateEntry
    ) -> tuple[StateEntry, float]:
        """Apply ``action`` to ``state`` and return the next state and reward."""

    @abstractmethod
    def get_action_reward(
        self,
        action: Action,
        state: StateEntry,
        next_state: StateEntry,
    ) -> float:
        """Return the single-step reward for ``action`` from ``state`` to ``next_state``."""

    def is_active_perception_action(self, action: Action | None) -> bool:
        """Return whether the grounded action should be treated as active perception."""

        return bool(action is not None and getattr(action, "name", "").startswith("active_obs_"))

    def should_skip_observation_rule_before_check(
        self,
        observation_rule: ObservationRule,
        current_action: Action | None = None,
    ) -> bool:
        """Return whether a before-observation rule should be skipped for this step."""

        return bool(
            self.is_active_perception_action(current_action)
            and getattr(observation_rule, "name", "").startswith("before_")
        )

    @abstractmethod
    def check_observation_rule_condition(
        self,
        observation_rule: ObservationRule,
        state: StateEntry,
        current_action: Action | None = None,
    ) -> bool:
        """Return True if ``observation_rule`` is active in ``state``."""

    @abstractmethod
    def observe_with_rule(
        self,
        observation_rule: ObservationRule,
        state: StateEntry,
        current_action: Action | None = None,
    ) -> ObservationEntry:
        """Return the observation entries produced by ``observation_rule`` in ``state``."""

    @abstractmethod
    def check_default_policy_rule_condition(
        self,
        default_policy_rule: DefaultPolicyRule,
        state: StateEntry,
    ) -> bool:
        """Return True if ``default_policy_rule`` should fire in ``state``."""

    @abstractmethod
    def get_default_policy_rule_action(
        self,
        default_policy_rule: DefaultPolicyRule,
        state: StateEntry,
    ) -> Action:
        """Return the action selected by ``default_policy_rule`` in ``state``."""
