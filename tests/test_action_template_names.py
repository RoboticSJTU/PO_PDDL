from po_pddl.domain_generation.stages.manipulation_domain_learning.structured_action_templates import (
    canonical_action_name_from_template_text,
    induced_template_from_dict,
    normalize_argument_value,
)


def test_action_name_preserves_relations_directions_and_all_entity_slots() -> None:
    assert (
        canonical_action_name_from_template_text(
            "push {param_1} on {param_2} on the right to {param_3} on the left"
        )
        == "push_object_on_object_on_right_to_object_on_left"
    )


def test_induced_template_uses_deterministic_name_instead_of_llm_alias() -> None:
    template = induced_template_from_dict(
        {
            "template_id": "short_alias",
            "canonical_action_name": "another_alias",
            "template_text": "place {param_1} on top of {param_2} on the right",
        },
        require_action_category=False,
    )

    assert template.template_id == "place_object_on_top_of_object_on_right"
    assert template.canonical_action_name == "place_object_on_top_of_object_on_right"


def test_argument_normalization_does_not_make_articles_part_of_object_identity() -> None:
    assert normalize_argument_value("the blue item") == "blue_item"
    assert normalize_argument_value("an orange") == "orange"
    assert normalize_argument_value("a container") == "container"
