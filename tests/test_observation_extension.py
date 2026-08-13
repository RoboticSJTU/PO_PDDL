from dataclasses import replace

from po_pddl.domain_generation.extension.observation_learning import merge_passive_results
from po_pddl.domain_generation.stages.passive_observation_learning.models import (
    PassiveObservationLearningResult,
    PassiveObservationRuleSchema,
)


def _schema(**counts: int) -> PassiveObservationRuleSchema:
    true_total = counts["total_ground_truth_true"]
    false_total = counts["total_ground_truth_false"]
    return PassiveObservationRuleSchema(
        canonical_action_name="open_container",
        effect_bucket="open_container_success",
        success=True,
        variant_rank=0,
        predicate_name="contains_item",
        observable_name="obs_contains_item",
        last_action_constant="open_container_success_0",
        last_action_predicate_name="last_action_1_param",
        action_argument_types=["container"],
        extra_argument_types=["movable_item"],
        last_action_predicate_parameter_types=["last_action_marker", "container"],
        target_predicate_parameter_types=["movable_item", "container"],
        target_literal_template="contains_item(?obs0,?arg0)",
        condition_literals=[],
        true_rule_name="observe_contains_item_true",
        false_rule_name="observe_contains_item_false",
        prob_observable_true_given_ground_truth_true=(
            counts["observed_true_when_ground_truth_true"] / true_total if true_total else 0.0
        ),
        prob_observable_false_given_ground_truth_true=(
            counts["observed_false_when_ground_truth_true"] / true_total if true_total else 1.0
        ),
        prob_observable_true_given_ground_truth_false=(
            counts["observed_true_when_ground_truth_false"] / false_total if false_total else 0.0
        ),
        prob_observable_false_given_ground_truth_false=(
            counts["observed_false_when_ground_truth_false"] / false_total if false_total else 1.0
        ),
        **counts,
    )


def _result(schema: PassiveObservationRuleSchema) -> PassiveObservationLearningResult:
    return PassiveObservationLearningResult(
        source_records=[],
        review_results_by_variant={},
        seed_examples=[],
        conditions=[],
        expanded_examples=[],
        schemas=[schema],
        predicate_inventory=[],
        predicate_comments={},
        rendered_module_text="",
    )


def test_incremental_passive_observation_counts_are_added_and_probabilities_recomputed() -> None:
    existing = _schema(
        total_ground_truth_true=2,
        observed_true_when_ground_truth_true=1,
        observed_false_when_ground_truth_true=1,
        total_ground_truth_false=1,
        observed_true_when_ground_truth_false=0,
        observed_false_when_ground_truth_false=1,
    )
    new = replace(
        existing,
        total_ground_truth_true=1,
        observed_true_when_ground_truth_true=1,
        observed_false_when_ground_truth_true=0,
        prob_observable_true_given_ground_truth_true=1.0,
        prob_observable_false_given_ground_truth_true=0.0,
        total_ground_truth_false=1,
        observed_true_when_ground_truth_false=1,
        observed_false_when_ground_truth_false=0,
        prob_observable_true_given_ground_truth_false=1.0,
        prob_observable_false_given_ground_truth_false=0.0,
    )

    merged = merge_passive_results(_result(existing), _result(new))

    assert len(merged.schemas) == 1
    schema = merged.schemas[0]
    assert schema.total_ground_truth_true == 3
    assert schema.prob_observable_true_given_ground_truth_true == 2 / 3
    assert schema.total_ground_truth_false == 2
    assert schema.prob_observable_true_given_ground_truth_false == 0.5
    assert "0.666667" in merged.rendered_module_text
