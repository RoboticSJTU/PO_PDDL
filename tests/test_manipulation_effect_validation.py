import pytest

from po_pddl.domain_generation.stages.manipulation_domain_learning.models import (
    ManipulationEffectRecord,
    PredicateSchema,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.modules import (
    validate_manipulation_records_against_predicate_inventory,
)


def _record(*, delta_add: list[str], delta_del: list[str]) -> ManipulationEffectRecord:
    return ManipulationEffectRecord(
        episode_name="episode0",
        step_index=1,
        raw_action_text="place the cup",
        canonical_action_name="place_object",
        action_arguments=["cup"],
        pre_observation_text=None,
        post_observation_text=None,
        extra_info=None,
        delta_add=delta_add,
        delta_del=delta_del,
        effect_bucket="place_object_success",
        success=True,
    )


def test_action_effect_cannot_modify_static_feature_predicate() -> None:
    predicates = [
        PredicateSchema(
            predicate_name="intrinsic_feature",
            parameter_types=["object"],
            comment="An immutable object property.",
            predicate_kind="feature",
        )
    ]

    with pytest.raises(ValueError, match="static feature"):
        validate_manipulation_records_against_predicate_inventory(
            records=[_record(delta_add=[], delta_del=["intrinsic_feature(cup)"])],
            predicate_inventory=predicates,
        )


def test_action_effect_can_modify_state_predicate() -> None:
    predicates = [
        PredicateSchema(
            predicate_name="held",
            parameter_types=["object"],
            comment="The object is held.",
            predicate_kind="state",
        )
    ]

    validate_manipulation_records_against_predicate_inventory(
        records=[_record(delta_add=["held(cup)"], delta_del=[])],
        predicate_inventory=predicates,
    )
