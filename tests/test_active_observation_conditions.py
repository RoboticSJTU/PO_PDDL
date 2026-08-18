from po_pddl.domain_generation.stages.active_observation_learning.learner import ActiveObservationLearner
from po_pddl.domain_generation.stages.active_observation_learning.models import ActiveObservationExample
from po_pddl.domain_generation.stages.manipulation_domain_learning.models import PredicateSchema


def test_condition_mining_preserves_negative_literal_polarity() -> None:
    example = ActiveObservationExample(
        episode_name="episode0",
        step_index=0,
        canonical_action_name="look_in_container",
        effect_bucket="look_in_container_success",
        success=True,
        variant_rank=0,
        predicate_name="in",
        grounded_literal="in(item,container)",
        action_arguments=["container"],
        action_argument_types=["container"],
        extra_arguments=["item"],
        extra_argument_types=["item"],
        ground_truth_value=True,
        observed_value=True,
        current_state=["in(item,container)", "not occluded(item)"],
        source_kind="test",
    )

    learner = object.__new__(ActiveObservationLearner)
    condition = learner._mine_condition(
        examples=[example],
        predicate_schema=PredicateSchema("in", ["item", "container"], "containment"),
    )

    assert condition.target_literal_template == "in(?obs0,?arg0)"
    assert condition.condition_literals == ["not occluded(?obs0)"]


def test_condition_mining_excludes_alternative_relation_on_target_tuple() -> None:
    example = ActiveObservationExample(
        episode_name="episode0",
        step_index=0,
        canonical_action_name="look_in_container",
        effect_bucket="look_in_container_success",
        success=True,
        variant_rank=0,
        predicate_name="in",
        grounded_literal="in(item,container)",
        action_arguments=["container"],
        action_argument_types=["container"],
        extra_arguments=["item"],
        extra_argument_types=["item"],
        ground_truth_value=True,
        observed_value=True,
        current_state=["in(item,container)", "not on_top_of(item,container)"],
        source_kind="test",
    )

    learner = object.__new__(ActiveObservationLearner)
    condition = learner._mine_condition(
        examples=[example],
        predicate_schema=PredicateSchema("in", ["item", "container"], "containment"),
    )

    assert condition.condition_literals == []
