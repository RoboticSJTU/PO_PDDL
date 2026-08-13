"""Lightweight semantic-model metadata container for runtime fast paths."""

from __future__ import annotations

from ..base.pomdp_model import POMDPModelBase
from ..data_structures import Action, DefaultPolicyRule, Observable, ObservationEntry, ObservationRule, Predicate


class RuntimeSemanticMetadataModel(POMDPModelBase):
    """Concrete semantic-model shell that only carries grounded member lists.

    This class is intentionally minimal: it exists so runtime code can keep using
    semantic identifiers such as grounded actions / predicates / observables
    without forcing the parser to emit and import a generated explicit package.
    Execution methods are intentionally unsupported in this fast path.
    """

    def __init__(
        self,
        *,
        predicates: list[Predicate],
        observables: list[Observable],
        actions: list[Action],
        observation_rules: list[ObservationRule],
        default_policy_rules: list[DefaultPolicyRule],
        maximize_reward: bool,
        goal_reward: float,
    ) -> None:
        super().__init__(
            predicates=list(predicates),
            observables=list(observables),
            actions=list(actions),
            observation_rules=list(observation_rules),
            default_policy_rules=list(default_policy_rules),
            maximize_reward=maximize_reward,
            goal_reward=goal_reward,
        )

    def is_report_goal_action(self, action) -> bool:
        report_goal_action = getattr(self, "report_goal_action", None)
        return bool(report_goal_action is not None and action == report_goal_action)

    def is_goal(self, state):
        raise NotImplementedError("RuntimeSemanticMetadataModel does not implement semantic execution.")

    def check_action_precondition(self, action, state):
        raise NotImplementedError("RuntimeSemanticMetadataModel does not implement semantic execution.")

    def forward_action(self, action, state):
        raise NotImplementedError("RuntimeSemanticMetadataModel does not implement semantic execution.")

    def get_action_reward(self, action, state, next_state):
        raise NotImplementedError("RuntimeSemanticMetadataModel does not implement semantic execution.")

    def check_observation_rule_condition(self, observation_rule, state, current_action=None):
        raise NotImplementedError("RuntimeSemanticMetadataModel does not implement semantic execution.")

    def observe_with_rule(self, observation_rule, state, current_action=None) -> ObservationEntry:
        raise NotImplementedError("RuntimeSemanticMetadataModel does not implement semantic execution.")

    def check_default_policy_rule_condition(self, default_policy_rule, state):
        raise NotImplementedError("RuntimeSemanticMetadataModel does not implement semantic execution.")

    def get_default_policy_rule_action(self, default_policy_rule, state):
        raise NotImplementedError("RuntimeSemanticMetadataModel does not implement semantic execution.")
