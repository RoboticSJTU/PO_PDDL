"""Bitwise-backed simulator world with a semantic-compatible interface."""

from __future__ import annotations

from dataclasses import dataclass

from ..bitwise import (
    BitwiseIndexLayout,
    POMDPModelBase as BitwisePOMDPModelBase,
    state_to_bitvector,
)
from ..data_structures.action import Action
from ..data_structures.aliases import ObservationEntry, StateEntry
from ..data_structures.effect_bucket import EffectBucket
from .observation_semantics import normalize_bitwise_observation_bits, normalize_semantic_observation_entry


@dataclass
class InitialBitwiseWorldHistoryEntry:
    """History entry for the initial bitwise-backed world state."""

    state_bits: int
    state: StateEntry


@dataclass
class TransitionBitwiseWorldHistoryEntry:
    """History entry for one executed action in a bitwise-backed world."""

    action: Action | int
    action_id: int
    state_bits: int
    state: StateEntry
    observation_bits: int
    observation_mask: int
    observation: ObservationEntry
    reward: float
    effect_bucket: EffectBucket | None = None


class BitwisePOMDPWorld:
    """Simulator wrapper that executes state transitions in bitwise form."""

    def __init__(
        self,
        init_state: StateEntry | int,
        bitwise_pomdp_model: BitwisePOMDPModelBase,
        layout: BitwiseIndexLayout,
        *,
        gamma: float = 0.95,
        max_step: int = 30,
        random_seed: int | None = None,
    ) -> None:
        self.layout = layout
        self.bitwise_pomdp_model = bitwise_pomdp_model
        self.gamma = gamma
        self.max_step = max_step
        self.random_seed = random_seed

        self.init_state_bits = self._normalize_init_state_bits(init_state)
        self.init_state = self._state_bits_to_semantic_state(self.init_state_bits)
        self.current_state_bits = self.init_state_bits
        self.current_state = dict(self.init_state)

        self.current_observation_bits = 0
        self.current_observation_mask = 0
        self.current_observation: ObservationEntry = {}
        self.last_effect_bucket: EffectBucket | None = None
        self.last_reward = 0.0
        self.total_undiscounted_reward = 0.0
        self.total_discounted_reward = 0.0
        self._goal_reported_successfully = False
        self.current_step = 0
        self.history: list[InitialBitwiseWorldHistoryEntry | TransitionBitwiseWorldHistoryEntry] = [
            InitialBitwiseWorldHistoryEntry(
                state_bits=self.current_state_bits,
                state=dict(self.current_state),
            )
        ]

    def _normalize_init_state_bits(self, init_state: StateEntry | int) -> int:
        if isinstance(init_state, int):
            return int(init_state)
        return state_to_bitvector(init_state, self.layout)

    def _state_bits_to_semantic_state(self, state_bits: int) -> StateEntry:
        return {
            predicate: True
            for index, predicate in enumerate(self.layout.predicates)
            if (state_bits >> index) & 1
        }

    def _observation_bits_to_semantic_observation(
        self,
        observation_bits: int,
        observation_mask: int,
    ) -> ObservationEntry:
        observation: ObservationEntry = {}
        for index, observable in enumerate(self.layout.observables):
            bit = 1 << index
            if observation_mask & bit:
                observation[observable] = bool(observation_bits & bit)
        return observation

    def _step_seed(self) -> int | None:
        if self.random_seed is None:
            return None
        return int(self.random_seed) + int(self.current_step)

    def _action_to_id(self, action: Action | int) -> int:
        if isinstance(action, int):
            return int(action)
        if action not in self.layout.action_index:
            raise ValueError(f"Unknown grounded action: {action.to_pddl_str()}")
        return self.layout.action_index[action]

    def _no_feasible_action_id(self) -> int:
        return len(self.layout.action_index)

    def _is_no_feasible_action(self, action_id: int) -> bool:
        return int(action_id) == self._no_feasible_action_id()

    def _format_state(self, state: StateEntry) -> str:
        true_predicates = sorted(predicate.to_pddl_str() for predicate, value in state.items() if value)
        return f"true={true_predicates}"

    def _format_observation(self, observation: ObservationEntry) -> str:
        if not observation:
            return "{}"
        true_observables = sorted(
            observable.to_pddl_str()
            for observable, value in observation.items()
            if value
        )
        false_observables = sorted(
            observable.to_pddl_str()
            for observable, value in observation.items()
            if not value
        )
        if false_observables:
            return f"true={true_observables}, false={false_observables}"
        return f"true={true_observables}"

    def format_history_entry(
        self,
        entry: InitialBitwiseWorldHistoryEntry | TransitionBitwiseWorldHistoryEntry,
        *,
        index: int | None = None,
    ) -> str:
        prefix = f"step={index}" if index is not None else "step=?"
        if isinstance(entry, InitialBitwiseWorldHistoryEntry):
            return f"{prefix} | initial_state | {self._format_state(entry.state)}"
        action_label = (
            entry.action.to_pddl_str()
            if isinstance(entry.action, Action)
            else (
                f"[no-feasible-action id={entry.action_id}]"
                if self._is_no_feasible_action(entry.action_id)
                else f"action_id={entry.action_id}"
            )
        )
        return (
            f"{prefix} | action={action_label} | "
            f"state={self._format_state(entry.state)} | "
            f"observation={self._format_observation(entry.observation)} | "
            f"reward={entry.reward}"
            + (
                f" | effect_bucket={entry.effect_bucket}"
                if entry.effect_bucket is not None
                else ""
            )
        )

    def format_history(self) -> str:
        return "\n".join(
            self.format_history_entry(entry, index=index)
            for index, entry in enumerate(self.history)
        )

    def history_summary(self) -> str:
        return "\n".join(
            [
                self.format_history(),
                f"current_step={self.current_step}",
                f"discounted_total_reward={self.total_discounted_reward}",
                f"undiscounted_total_reward={self.total_undiscounted_reward}",
            ]
        )

    def get_observation_bits(self, current_action_id: int | None = None) -> tuple[int, int]:
        observation_bits = 0
        observation_mask = 0
        total_rules = len(self.bitwise_pomdp_model.observation_rule_condition_checks)
        for rule_id in range(total_rules):
            if not self.bitwise_pomdp_model.check_observation_rule_condition(
                rule_id,
                self.current_state_bits,
                current_action=current_action_id,
            ):
                continue
            rule_bits, rule_mask = self.bitwise_pomdp_model.observe_with_rule(
                rule_id,
                self.current_state_bits,
                current_action=current_action_id,
            )
            observation_bits = (observation_bits & ~rule_mask) | (rule_bits & rule_mask)
            observation_mask |= rule_mask
        observation_bits, observation_mask = normalize_bitwise_observation_bits(
            observation_bits,
            observation_mask,
            self.layout.observables,
        )
        self.current_observation_bits = observation_bits
        self.current_observation_mask = observation_mask
        return observation_bits, observation_mask

    def get_observation(self, current_action_id: int | None = None) -> ObservationEntry:
        observation_bits, observation_mask = self.get_observation_bits(current_action_id)
        observation = self._observation_bits_to_semantic_observation(observation_bits, observation_mask)
        observation = normalize_semantic_observation_entry(observation)
        self.current_observation = dict(observation)
        return dict(observation)

    def execute_action(
        self,
        action: Action | int,
        *,
        return_effect_bucket: bool = False,
    ) -> tuple[StateEntry, ObservationEntry, float] | tuple[StateEntry, ObservationEntry, float, EffectBucket | None]:
        if self.current_step >= self.max_step:
            raise ValueError("Maximum number of execution steps has been reached.")

        step_seed = self._step_seed()
        if step_seed is not None:
            self.bitwise_pomdp_model._rng.seed(step_seed)

        action_id = self._action_to_id(action)
        if self._is_no_feasible_action(action_id):
            self.bitwise_pomdp_model.clear_last_effect_bucket()
            self.last_effect_bucket = None
            reward = 0.0
        else:
            self.bitwise_pomdp_model.clear_last_effect_bucket()
            next_state_bits, reward = self.bitwise_pomdp_model.forward_action(
                action_id,
                self.current_state_bits,
            )
            self.last_effect_bucket = self.bitwise_pomdp_model.get_last_effect_bucket()
            self.current_state_bits = int(next_state_bits)
            self.current_state = self._state_bits_to_semantic_state(self.current_state_bits)
            if bool(getattr(self.bitwise_pomdp_model, "enable_report_goal_action", False)) and bool(
                getattr(self.bitwise_pomdp_model, "is_report_goal_action", lambda _action: False)(action_id)
            ):
                self._goal_reported_successfully = bool(
                    self.last_effect_bucket is not None and self.last_effect_bucket.success
                )
        observation_action_id = None if self._is_no_feasible_action(action_id) else action_id
        observation = self.get_observation(observation_action_id)

        self.current_step += 1
        self.last_reward = reward
        self.total_undiscounted_reward += reward
        self.total_discounted_reward += (self.gamma ** (self.current_step - 1)) * reward

        self.history.append(
            TransitionBitwiseWorldHistoryEntry(
                action=action,
                action_id=action_id,
                state_bits=self.current_state_bits,
                state=dict(self.current_state),
                observation_bits=self.current_observation_bits,
                observation_mask=self.current_observation_mask,
                observation=dict(observation),
                reward=reward,
                effect_bucket=self.last_effect_bucket,
            )
        )

        result = (dict(self.current_state), dict(observation), reward)
        if return_effect_bucket:
            return (*result, self.last_effect_bucket)
        return result

    def is_goal(self) -> bool:
        if bool(getattr(self.bitwise_pomdp_model, "enable_report_goal_action", False)):
            return bool(self._goal_reported_successfully)
        return self.bitwise_pomdp_model.is_goal(self.current_state_bits)
