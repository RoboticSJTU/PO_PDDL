"""Template simulator world for Python POMDP models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..base.pomdp_model import POMDPModelBase
from ..data_structures.action import Action
from ..data_structures.aliases import ObservationEntry, StateEntry
from ..data_structures.effect_bucket import EffectBucket
from ..parser import parse_problem
from .observation_semantics import normalize_semantic_observation_entry


@dataclass
class InitialWorldHistoryEntry:
    """History entry for step 0, which only contains the initial state."""

    state: StateEntry


@dataclass
class TransitionWorldHistoryEntry:
    """History entry for one executed action."""

    action: Action | int
    state: StateEntry
    observation: ObservationEntry
    reward: float
    effect_bucket: EffectBucket | None = None


class POMDPWorld:
    """Generic simulator wrapper around a Python ``POMDPModelBase``."""

    def __init__(
        self,
        init_state: StateEntry,
        pomdp_model: POMDPModelBase,
        gamma: float = 0.95,
        max_step: int = 30,
        random_seed: int | None = None,
    ) -> None:
        self.init_state = dict(init_state)
        self.pomdp_model = pomdp_model
        self.gamma = gamma
        self.max_step = max_step
        self.random_seed = random_seed
        self.current_state = self._copy_state(self.init_state)
        self.history: list[InitialWorldHistoryEntry | TransitionWorldHistoryEntry] = []
        self.current_step = 0
        self.current_observation: ObservationEntry = {}
        self.last_effect_bucket: EffectBucket | None = None
        self.last_reward = 0.0
        self.total_undiscounted_reward = 0.0
        self.total_discounted_reward = 0.0
        self._goal_reported_successfully = False
        self.history = [InitialWorldHistoryEntry(state=self._copy_state(self.current_state))]

    @classmethod
    def from_problem_text(
        cls,
        problem_text: str,
        pomdp_model: POMDPModelBase,
        *,
        gamma: float = 0.95,
        max_step: int = 90,
    ) -> "POMDPWorld":
        """Build a simulator world from the ``:init`` section of one problem text."""
        parsed_problem = parse_problem(problem_text)
        return cls(
            init_state=dict(parsed_problem.init_state),
            pomdp_model=pomdp_model,
            gamma=gamma,
            max_step=max_step,
        )

    @classmethod
    def from_problem_file(
        cls,
        problem_file: str | Path,
        pomdp_model: POMDPModelBase,
        *,
        gamma: float = 0.95,
        max_step: int = 90,
    ) -> "POMDPWorld":
        """Build a simulator world from the ``:init`` section of one problem file."""
        problem_path = Path(problem_file)
        return cls.from_problem_text(
            problem_path.read_text(encoding="utf-8"),
            pomdp_model,
            gamma=gamma,
            max_step=max_step,
        )

    def _copy_state(self, state: StateEntry) -> StateEntry:
        return dict(state)

    def _copy_observation(self, observation: ObservationEntry) -> ObservationEntry:
        return dict(observation)

    def _merge_observations(self, observations: list[ObservationEntry]) -> ObservationEntry:
        merged: ObservationEntry = {}
        for observation in observations:
            merged.update(observation)
        return normalize_semantic_observation_entry(merged)

    def _format_state(self, state: StateEntry) -> str:
        true_predicates = sorted(
            str(predicate) for predicate, value in state.items() if value
        )
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
        entry: InitialWorldHistoryEntry | TransitionWorldHistoryEntry,
        *,
        index: int | None = None,
    ) -> str:
        """Render one history entry in a readable single-step format."""
        prefix = f"step={index}" if index is not None else "step=?"
        if isinstance(entry, InitialWorldHistoryEntry):
            return f"{prefix} | initial_state | {self._format_state(entry.state)}"
        return (
            f"{prefix} | action={self._format_action(entry.action)} | "
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
        """Render the whole execution history as readable multi-line text."""
        return "\n".join(
            self.format_history_entry(entry, index=index)
            for index, entry in enumerate(self.history)
        )

    def history_summary(self) -> str:
        """Render history plus current reward totals in one compact summary."""
        summary_lines = [self.format_history()]
        summary_lines.append(f"current_step={self.current_step}")
        summary_lines.append(f"discounted_total_reward={self.total_discounted_reward}")
        summary_lines.append(f"undiscounted_total_reward={self.total_undiscounted_reward}")
        return "\n".join(summary_lines)

    def get_observation(self, current_action: Action | int | None = None) -> ObservationEntry:
        """Collect and merge all active observation rules in the current state."""
        observation_entries: list[ObservationEntry] = []
        for observation_rule in self.pomdp_model.observation_rules:
            if self.pomdp_model.check_observation_rule_condition(
                observation_rule,
                self.current_state,
                current_action=current_action,
            ):
                observation_entries.append(
                    self.pomdp_model.observe_with_rule(
                        observation_rule,
                        self.current_state,
                        current_action=current_action,
                    )
                )
        merged = self._merge_observations(observation_entries)
        self.current_observation = self._copy_observation(merged)
        return self._copy_observation(merged)

    def _step_seed(self) -> int | None:
        if self.random_seed is None:
            return None
        return int(self.random_seed) + int(self.current_step)

    def _no_feasible_action_id(self) -> int:
        return len(self.pomdp_model.actions)

    def _is_no_feasible_action(self, action: Action | int) -> bool:
        return isinstance(action, int) and action == self._no_feasible_action_id()

    def _format_action(self, action: Action | int) -> str:
        if isinstance(action, Action):
            return action.to_pddl_str()
        if self._is_no_feasible_action(action):
            return f"[no-feasible-action id={action}]"
        return f"[action_id={action}]"

    def execute_action(
        self,
        action: Action | int,
        *,
        return_effect_bucket: bool = False,
    ) -> tuple[StateEntry, ObservationEntry, float] | tuple[StateEntry, ObservationEntry, float, EffectBucket | None]:
        """Execute one action, update the world state, and append a history entry."""
        if self.current_step >= self.max_step:
            raise ValueError("Maximum number of execution steps has been reached.")

        step_seed = self._step_seed()
        if step_seed is not None:
            self.pomdp_model._rng.seed(step_seed)
        if self._is_no_feasible_action(action):
            self.pomdp_model.clear_last_effect_bucket()
            self.last_effect_bucket = None
            reward = 0.0
        else:
            self.pomdp_model.clear_last_effect_bucket()
            next_state, reward = self.pomdp_model.forward_action(action, self.current_state)
            self.last_effect_bucket = self.pomdp_model.get_last_effect_bucket()
            self.current_state = self._copy_state(next_state)
            if bool(getattr(self.pomdp_model, "enable_report_goal_action", False)) and bool(
                getattr(self.pomdp_model, "is_report_goal_action", lambda _action: False)(action)
            ):
                self._goal_reported_successfully = bool(
                    self.last_effect_bucket is not None and self.last_effect_bucket.success
                )
        observation_action = None if self._is_no_feasible_action(action) else action
        observation = self.get_observation(observation_action)

        self.current_step += 1
        self.last_reward = reward
        self.total_undiscounted_reward += reward
        self.total_discounted_reward += (self.gamma ** (self.current_step - 1)) * reward

        self.history.append(
            TransitionWorldHistoryEntry(
                action=action,
                state=self._copy_state(self.current_state),
                observation=self._copy_observation(observation),
                reward=reward,
                effect_bucket=self.last_effect_bucket,
            )
        )

        result = (
            self._copy_state(self.current_state),
            self._copy_observation(observation),
            reward,
        )
        if return_effect_bucket:
            return (*result, self.last_effect_bucket)
        return result

    def is_goal(self) -> bool:
        """Return True when the current state satisfies the goal condition."""
        if bool(getattr(self.pomdp_model, "enable_report_goal_action", False)):
            return bool(self._goal_reported_successfully)
        return self.pomdp_model.is_goal(self.current_state)
