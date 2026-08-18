from po_pddl.domain_generation.stages.manipulation_domain_learning.problem_grounding_runner import (
    _overlay_predicate_facts,
    reverse_effects_to_reconstruct_states,
)
from po_pddl.domain_generation.stages.problem_grounding.models import (
    GroundedTrajectoryStep,
    ObjectDeclaration,
    ProblemGroundingResult,
    ProblemSpec,
    ValidationStepReport,
)


def test_overlay_replaces_opposite_literal_polarity() -> None:
    result = _overlay_predicate_facts(
        base_facts={"open(drawer_a)"},
        replacement_facts={"not open(drawer_a)"},
        predicate_names={"open"},
    )

    assert result == {"not open(drawer_a)"}


def test_reverse_deleted_fact_restores_positive_initial_value() -> None:
    spec = ProblemSpec(
        problem_name="close_test",
        domain_name="test",
        objects=[ObjectDeclaration("drawer_a", "drawer")],
        init_facts=["open(drawer_a)"],
        goal_facts=["not open(drawer_a)"],
    )
    step = GroundedTrajectoryStep(
        episode_name="episode0",
        step_index=1,
        raw_action_text="close drawer",
        action_category="manipulation",
        canonical_action_name="close_drawer",
        ground_arguments=["drawer_a"],
        ground_action_pddl="(close_drawer drawer_a)",
        effect_bucket="close_drawer_success",
        delta_add=[],
        delta_del=["open(drawer_a)"],
        success=True,
        observation_text="The drawer is closed.",
        extra_info=None,
    )
    result = ProblemGroundingResult(
        problem_spec=spec,
        problem_pddl="",
        grounded_steps=[step],
        validation_steps=[
            ValidationStepReport(
                step_index=1,
                action_name="close_drawer",
                effect_bucket="close_drawer_success",
                status="applied",
                state_before=["open(drawer_a)"],
                state_after=[],
            )
        ],
        validation_issues=[],
        goal_satisfied=True,
    )

    rewritten, _summary = reverse_effects_to_reconstruct_states(result)

    assert rewritten.problem_spec.init_facts == ["open(drawer_a)"]
    assert rewritten.validation_steps[0].state_before == ["open(drawer_a)"]
    assert rewritten.validation_steps[0].state_after == ["not open(drawer_a)"]
