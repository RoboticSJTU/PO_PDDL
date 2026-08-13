from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from po_pddl.runtime.planning.data_structures import Action
from po_pddl.runtime.planning.data_structures.effect_bucket import ParsedEffectBucketAnnotation
from po_pddl.runtime.terminal.feedback import ScriptedFeedback
from po_pddl.runtime.terminal.session import TerminalSession


@dataclass
class FakePlanner:
    actions: list[Action]

    def __post_init__(self) -> None:
        self.current_step = 0
        self.pomdp_model = SimpleNamespace(
            actions=self.actions,
            observables=[],
            enable_report_goal_action=True,
            is_report_goal_action=lambda action: action.name == "report-goal",
        )

    def search(self) -> Action:
        return self.actions[self.current_step]

    def estimate_expected_reward(self, action, *, effect_bucket):
        return 100.0 if action.name == "report-goal" else -2.0

    def applicable_observables(self, action, *, effect_bucket):
        return []

    def belief_update(self, action, observation, **kwargs):
        self.current_step += 1


def test_session_runs_until_successful_goal_report(tmp_path) -> None:
    planner = FakePlanner([Action("move"), Action("report-goal")])
    schemas = {
        "move": SimpleNamespace(
            effect_bucket_annotations=[ParsedEffectBucketAnnotation("move_success", True, 0)]
        )
    }
    feedback = ScriptedFeedback(
        [
            {"action": "move", "success": True},
            {"action": "report-goal", "success": True},
        ]
    )

    result = TerminalSession(planner, schemas, feedback, output_dir=tmp_path).run()

    assert result.termination_reason == "report_goal_success"
    assert [step.action for step in result.steps] == ["(move)", "(report-goal)"]
    assert result.cumulative_reward == 98.0
    assert (tmp_path / "steps.jsonl").is_file()
    assert (tmp_path / "session_summary.json").is_file()
