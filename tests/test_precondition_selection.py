from po_pddl.domain_generation.stages.precondition_learning.modules import (
    retain_identity_anchored_preconditions,
)


def test_unanchored_dependency_literal_is_not_forced() -> None:
    selected = retain_identity_anchored_preconditions(
        action_name="pick_up_item",
        universally_supported_literals=[
            "gripper_empty()",
            "not supported_by(?dep0,?dep1)",
        ],
        selected_literals=[],
    )

    assert selected == ["gripper_empty()"]


def test_anchored_causal_dependency_literal_is_retained() -> None:
    selected = retain_identity_anchored_preconditions(
        action_name="place_item",
        universally_supported_literals=[
            "gripper_empty()",
            "not occupied_by(?dep0,?arg1)",
        ],
        selected_literals=[],
    )

    assert selected == ["gripper_empty()", "not occupied_by(?dep0,?arg1)"]


def test_selector_can_keep_direct_dependent_clearance_constraint() -> None:
    selected = retain_identity_anchored_preconditions(
        action_name="place_item",
        universally_supported_literals=[
            "gripper_empty()",
            "not occupied_by(?dep0,?arg1)",
        ],
        selected_literals=["not occupied_by(?dep0,?arg1)"],
    )

    assert selected == ["gripper_empty()", "not occupied_by(?dep0,?arg1)"]


def test_holding_item_does_not_force_redundant_old_location_absence() -> None:
    selected = retain_identity_anchored_preconditions(
        action_name="place_item_in_container",
        universally_supported_literals=[
            "gripper_holding(?arg0)",
            "not on(?arg0,?dep0)",
            "open(?arg1)",
        ],
        selected_literals=["gripper_holding(?arg0)", "open(?arg1)"],
    )

    assert selected == ["gripper_holding(?arg0)", "open(?arg1)"]


def test_holding_item_keeps_destination_clearance_for_another_object() -> None:
    selected = retain_identity_anchored_preconditions(
        action_name="place_item_on_surface",
        universally_supported_literals=[
            "gripper_holding(?arg0)",
            "not occupied_by(?dep0,?arg1)",
        ],
        selected_literals=["gripper_holding(?arg0)"],
    )

    assert selected == ["gripper_holding(?arg0)", "not occupied_by(?dep0,?arg1)"]


def test_positive_zero_arity_resource_is_still_enforced() -> None:
    selected = retain_identity_anchored_preconditions(
        action_name="inspect_fixture",
        universally_supported_literals=["sensor_available()"],
        selected_literals=[],
    )

    assert selected == ["sensor_available()"]
