"""Opt-in runtime augmentation for a synthetic report-goal action."""

from __future__ import annotations

from dataclasses import dataclass

from ..base.pomdp_model import POMDPModelBase as SemanticPOMDPModelBase
from ..bitwise import POMDPModelBase as BitwisePOMDPModelBase
from ..data_structures import Action, EffectBucket
from .semantic_metadata_model import RuntimeSemanticMetadataModel


REPORT_GOAL_ACTION_NAME = "report-goal"
REPORT_GOAL_BUCKET_NAME = "report_goal"
REPORT_GOAL_ACTION = Action(REPORT_GOAL_ACTION_NAME, [])


def _report_branch_index(success: bool) -> int:
    return 0 if success else 1


@dataclass(frozen=True)
class ReportGoalRuntimeConfig:
    """Resolved runtime config for the synthetic report-goal action."""

    enable_report_goal_action: bool = False
    report_goal_failure_penalty: float = -10.0


class ReportGoalActionSemanticModel(SemanticPOMDPModelBase):
    """Semantic execution wrapper that prepends an opt-in report-goal action."""

    def __init__(
        self,
        base_model: SemanticPOMDPModelBase,
        *,
        report_goal_failure_penalty: float,
    ) -> None:
        self.base_model = base_model
        self.report_goal_action = REPORT_GOAL_ACTION
        self.enable_report_goal_action = True
        self.report_goal_failure_penalty = float(report_goal_failure_penalty)
        self.action_id_offset = 1
        super().__init__(
            predicates=list(base_model.predicates),
            observables=list(base_model.observables),
            actions=[self.report_goal_action, *list(base_model.actions)],
            observation_rules=list(base_model.observation_rules),
            default_policy_rules=list(base_model.default_policy_rules),
            maximize_reward=base_model.maximize_reward,
            goal_reward=base_model.goal_reward,
        )

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in {"_sample_branch_index", "_select_effect_branch_index", "_rng"} and "base_model" in self.__dict__:
            setattr(self.base_model, name, value)

    def __getattr__(self, name):
        return getattr(self.base_model, name)

    def clear_last_effect_bucket(self) -> None:
        super().clear_last_effect_bucket()
        self.base_model.clear_last_effect_bucket()

    def set_effect_bucket_filter(
        self,
        *,
        bucket_name: str | None = None,
        success: bool | None = None,
        branch_index: int | None = None,
    ) -> None:
        super().set_effect_bucket_filter(
            bucket_name=bucket_name,
            success=success,
            branch_index=branch_index,
        )
        self.base_model.set_effect_bucket_filter(
            bucket_name=bucket_name,
            success=success,
            branch_index=branch_index,
        )

    def clear_effect_bucket_filter(self) -> None:
        super().clear_effect_bucket_filter()
        self.base_model.clear_effect_bucket_filter()

    def raw_is_goal(self, state) -> bool:
        return bool(self.base_model.is_goal(state))

    def is_goal(self, state) -> bool:
        return False

    def is_report_goal_action(self, action: Action) -> bool:
        return action == self.report_goal_action

    def effect_bucket_matches_state(
        self,
        action: Action,
        state,
        effect_bucket: EffectBucket | None,
    ) -> bool:
        if effect_bucket is None or not self.is_report_goal_action(action):
            return True
        if effect_bucket.bucket_name not in (None, REPORT_GOAL_BUCKET_NAME):
            return False
        success = self.raw_is_goal(state)
        if effect_bucket.success is not None and effect_bucket.success != success:
            return False
        if effect_bucket.branch_index is not None and effect_bucket.branch_index != _report_branch_index(success):
            return False
        return True

    def _cache_transition_reward(self, action: Action, state, next_state, reward: float) -> None:
        cache_key = (action, frozenset(state.items()), frozenset(next_state.items()))
        self._last_transition_reward_cache[cache_key] = reward

    def _lookup_cached_reward(self, action: Action, state, next_state) -> float | None:
        cache_key = (action, frozenset(state.items()), frozenset(next_state.items()))
        return self._last_transition_reward_cache.get(cache_key)

    def check_action_precondition(self, action: Action, state) -> bool:
        if self.is_report_goal_action(action):
            return self.raw_is_goal(state)
        return bool(self.base_model.check_action_precondition(action, state))

    def forward_action(self, action: Action, state) -> tuple[dict, float]:
        if self.is_report_goal_action(action):
            next_state = dict(state)
            success = self.raw_is_goal(state)
            self._last_effect_bucket = EffectBucket(
                bucket_name=REPORT_GOAL_BUCKET_NAME,
                success=success,
                branch_index=_report_branch_index(success),
            )
            reward = self.goal_reward if success else self.report_goal_failure_penalty
            self._cache_transition_reward(action, state, next_state, reward)
            return next_state, reward

        self.base_model.clear_last_effect_bucket()
        next_state, reward = self.base_model.forward_action(action, state)
        if self.base_model.is_goal(next_state):
            reward -= self.base_model.goal_reward
        self._last_effect_bucket = self.base_model.get_last_effect_bucket()
        self._cache_transition_reward(action, state, next_state, reward)
        return dict(next_state), reward

    def get_action_reward(self, action: Action, state, next_state) -> float:
        cached = self._lookup_cached_reward(action, state, next_state)
        if cached is not None:
            return cached
        if self.is_report_goal_action(action):
            return self.goal_reward if self.raw_is_goal(state) else self.report_goal_failure_penalty
        reward = self.base_model.get_action_reward(action, state, next_state)
        if self.base_model.is_goal(next_state):
            reward -= self.base_model.goal_reward
        return reward

    def is_active_perception_action(self, action: Action | None) -> bool:
        if action is None or self.is_report_goal_action(action):
            return False
        return bool(self.base_model.is_active_perception_action(action))

    def check_observation_rule_condition(self, observation_rule, state, current_action=None) -> bool:
        return bool(
            self.base_model.check_observation_rule_condition(
                observation_rule,
                state,
                current_action=current_action,
            )
        )

    def observe_with_rule(self, observation_rule, state, current_action=None):
        return self.base_model.observe_with_rule(
            observation_rule,
            state,
            current_action=current_action,
        )

    def check_default_policy_rule_condition(self, default_policy_rule, state) -> bool:
        return bool(self.base_model.check_default_policy_rule_condition(default_policy_rule, state))

    def get_default_policy_rule_action(self, default_policy_rule, state):
        return self.base_model.get_default_policy_rule_action(default_policy_rule, state)


class ReportGoalActionBitwiseModel(BitwisePOMDPModelBase):
    """Bitwise execution wrapper that prepends an opt-in report-goal action."""

    def __init__(
        self,
        base_model: BitwisePOMDPModelBase,
        *,
        report_goal_failure_penalty: float,
    ) -> None:
        self.base_model = base_model
        self.enable_report_goal_action = True
        self.report_goal_failure_penalty = float(report_goal_failure_penalty)
        self.action_id_offset = 1
        self.raw_goal_check = getattr(base_model, "goal_check", None)
        super().__init__(
            grounded_predicates_count=base_model.grounded_predicates_count,
            grounded_observables_count=base_model.grounded_observables_count,
            total_actions=int(base_model.total_actions) + 1,
            maximize_reward=base_model.maximize_reward,
            goal_reward=base_model.goal_reward,
        )

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in {"_sample_branch_index", "_select_effect_branch_index", "_rng"} and "base_model" in self.__dict__:
            setattr(self.base_model, name, value)

    def __getattr__(self, name):
        return getattr(self.base_model, name)

    def clear_last_effect_bucket(self) -> None:
        super().clear_last_effect_bucket()
        self.base_model.clear_last_effect_bucket()

    def set_effect_bucket_filter(
        self,
        *,
        bucket_name: str | None = None,
        success: bool | None = None,
        branch_index: int | None = None,
    ) -> None:
        super().set_effect_bucket_filter(
            bucket_name=bucket_name,
            success=success,
            branch_index=branch_index,
        )
        self.base_model.set_effect_bucket_filter(
            bucket_name=bucket_name,
            success=success,
            branch_index=branch_index,
        )

    def clear_effect_bucket_filter(self) -> None:
        super().clear_effect_bucket_filter()
        self.base_model.clear_effect_bucket_filter()

    def raw_is_goal(self, state: int) -> bool:
        return bool(self.base_model.is_goal(state))

    def is_goal(self, state: int) -> bool:
        return False

    def is_report_goal_action(self, action: int) -> bool:
        return int(action) == 0

    def effect_bucket_matches_state(
        self,
        action: int,
        state: int,
        effect_bucket: EffectBucket | None,
    ) -> bool:
        if effect_bucket is None or not self.is_report_goal_action(action):
            return True
        if effect_bucket.bucket_name not in (None, REPORT_GOAL_BUCKET_NAME):
            return False
        success = self.raw_is_goal(state)
        if effect_bucket.success is not None and effect_bucket.success != success:
            return False
        if effect_bucket.branch_index is not None and effect_bucket.branch_index != _report_branch_index(success):
            return False
        return True

    def _cache_transition_reward(self, action: int, state: int, next_state: int, reward: float) -> None:
        self._last_transition_reward_cache[(action, state, next_state)] = reward

    def _lookup_cached_reward(self, action: int, state: int, next_state: int) -> float | None:
        return self._last_transition_reward_cache.get((action, state, next_state))

    def check_action_precondition(self, action: int, state: int) -> bool:
        if self.is_report_goal_action(action):
            return self.raw_is_goal(state)
        return bool(self.base_model.check_action_precondition(int(action) - self.action_id_offset, state))

    def forward_action(self, action: int, state: int) -> tuple[int, float]:
        if self.is_report_goal_action(action):
            next_state = int(state)
            success = self.raw_is_goal(state)
            self._last_effect_bucket = EffectBucket(
                bucket_name=REPORT_GOAL_BUCKET_NAME,
                success=success,
                branch_index=_report_branch_index(success),
            )
            reward = self.goal_reward if success else self.report_goal_failure_penalty
            self._cache_transition_reward(int(action), int(state), next_state, reward)
            return next_state, reward

        self.base_model.clear_last_effect_bucket()
        next_state, reward = self.base_model.forward_action(int(action) - self.action_id_offset, state)
        if self.base_model.is_goal(next_state):
            reward -= self.base_model.goal_reward
        self._last_effect_bucket = self.base_model.get_last_effect_bucket()
        self._cache_transition_reward(int(action), int(state), int(next_state), reward)
        return int(next_state), reward

    def get_action_reward(self, action: int, state: int, next_state: int) -> float:
        cached = self._lookup_cached_reward(int(action), int(state), int(next_state))
        if cached is not None:
            return cached
        if self.is_report_goal_action(action):
            return self.goal_reward if self.raw_is_goal(state) else self.report_goal_failure_penalty
        reward = self.base_model.get_action_reward(int(action) - self.action_id_offset, state, next_state)
        if self.base_model.is_goal(next_state):
            reward -= self.base_model.goal_reward
        return reward

    def is_active_perception_action(self, action: int | None) -> bool:
        if action is None or self.is_report_goal_action(action):
            return False
        return bool(
            self.base_model.is_active_perception_action(int(action) - self.action_id_offset)
        )

    def check_observation_rule_condition(
        self,
        observation_rule: int,
        state: int,
        current_action: int | None = None,
    ) -> bool:
        base_action = None
        if current_action is not None and not self.is_report_goal_action(int(current_action)):
            base_action = int(current_action) - self.action_id_offset
        return bool(
            self.base_model.check_observation_rule_condition(
                observation_rule,
                state,
                current_action=base_action,
            )
        )

    def observe_with_rule(
        self,
        observation_rule: int,
        state: int,
        current_action: int | None = None,
    ):
        base_action = None
        if current_action is not None and not self.is_report_goal_action(int(current_action)):
            base_action = int(current_action) - self.action_id_offset
        return self.base_model.observe_with_rule(
            observation_rule,
            state,
            current_action=base_action,
        )

    def check_default_policy_rule_condition(self, default_policy_rule: int, state: int) -> bool:
        return bool(self.base_model.check_default_policy_rule_condition(default_policy_rule, state))

    def get_default_policy_rule_action(self, default_policy_rule: int, state: int) -> int:
        base_action_id = int(self.base_model.get_default_policy_rule_action(default_policy_rule, state))
        if base_action_id < 0:
            return 0
        return base_action_id + self.action_id_offset


def build_report_goal_planner_metadata_model(
    base_model: RuntimeSemanticMetadataModel,
    *,
    report_goal_failure_penalty: float,
) -> RuntimeSemanticMetadataModel:
    """Return a planner metadata model with the report-goal action prepended."""

    wrapped = RuntimeSemanticMetadataModel(
        predicates=list(base_model.predicates),
        observables=list(base_model.observables),
        actions=[REPORT_GOAL_ACTION, *list(base_model.actions)],
        observation_rules=list(base_model.observation_rules),
        default_policy_rules=list(base_model.default_policy_rules),
        maximize_reward=base_model.maximize_reward,
        goal_reward=base_model.goal_reward,
    )
    wrapped.enable_report_goal_action = True
    wrapped.report_goal_failure_penalty = float(report_goal_failure_penalty)
    wrapped.report_goal_action = REPORT_GOAL_ACTION
    wrapped.action_id_offset = 1
    return wrapped
