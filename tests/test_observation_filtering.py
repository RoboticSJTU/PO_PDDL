from po_pddl.domain_generation.stages.passive_observation_learning.learner import (
    _exclude_direct_effect_assignments,
)


def test_direct_effect_predicates_are_not_treated_as_observation_disagreement() -> None:
    assignments = [
        "in_front_of(item_a,surface_a)",
        "not in_front_of(item_a,surface_b)",
        "feature(item_a)",
    ]

    assert _exclude_direct_effect_assignments(assignments, ["in_front_of"]) == ["feature(item_a)"]
