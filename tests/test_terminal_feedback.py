from __future__ import annotations

import pytest

from po_pddl.runtime.planning.data_structures import Action, Observable
from po_pddl.runtime.planning.data_structures.effect_bucket import ParsedEffectBucketAnnotation
from po_pddl.runtime.terminal.feedback import ConsoleFeedback, ScriptedFeedback


def test_scripted_feedback_checks_expected_action_and_selects_branch() -> None:
    feedback = ScriptedFeedback([{"action": "move", "success": False, "variant_rank": 0}])
    annotations = [
        ParsedEffectBucketAnnotation("move_success", True, 0),
        ParsedEffectBucketAnnotation("move_failure", False, 0),
    ]

    result = feedback.select_effect(Action("move", ["a"]), annotations, report_goal_enabled=False)

    assert result is not None
    assert result.bucket_name == "move_failure"
    assert result.success is False


def test_scripted_feedback_rejects_non_applicable_observation() -> None:
    feedback = ScriptedFeedback([{"success": True, "observations": ["(obs-visible b)"]}])
    feedback.select_effect(Action("move"), [], report_goal_enabled=False)

    with pytest.raises(ValueError, match="not applicable"):
        feedback.select_observation([Observable("obs-visible", ["a"])], [])


def test_console_feedback_reprompts_on_conflicting_literals() -> None:
    replies = iter(["0,1", "0"])
    messages: list[str] = []
    feedback = ConsoleFeedback(read=lambda _prompt: next(replies), write=messages.append)
    observable = Observable("obs-visible", ["a"])

    observation, skipped = feedback.select_observation([observable], [observable])

    assert observation == {observable: True}
    assert skipped is False
    assert any("both true and false" in message for message in messages)
