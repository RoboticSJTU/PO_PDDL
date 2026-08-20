from po_pddl.domain_generation.stages.scene_description.text_normalization import (
    remove_unseen_object_statements,
)


def test_unseen_object_sentence_is_removed_without_touching_visible_content() -> None:
    text = "The yellow_drawer is closed. The blue_block is not visible. No dark liquid is visible."

    assert remove_unseen_object_statements(text, ["yellow_drawer", "blue_block"]) == (
        "The yellow_drawer is closed. No dark liquid is visible."
    )


def test_unseen_object_clause_is_removed_from_a_mixed_sentence() -> None:
    text = "The yellow_drawer is closed; the red_block is not visible."

    assert remove_unseen_object_statements(text, ["yellow_drawer", "red_block"]) == (
        "The yellow_drawer is closed."
    )


def test_non_allowlisted_visibility_statement_is_not_rewritten() -> None:
    text = "No liquid is visible inside the green_cup."

    assert remove_unseen_object_statements(text, ["green_cup"]) == text
