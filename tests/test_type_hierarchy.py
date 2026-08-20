from po_pddl.domain_generation.infrastructure.type_hierarchy import build_type_parent_map


def test_special_supertype_matches_rendered_pddl_hierarchy() -> None:
    rows = [
        {
            "type_name": "drawer",
            "parent_type": "fixed_item",
            "special_supertypes": ["containable_item", "fixed_item"],
        },
        {
            "type_name": "cup",
            "parent_type": "movable_item",
            "special_supertypes": ["movable_item"],
        },
    ]

    assert build_type_parent_map(rows) == {
        "drawer": "containable_item",
        "containable_item": "fixed_item",
        "cup": "movable_item",
    }


def test_ambiguous_special_supertype_does_not_rewrite_hierarchy() -> None:
    rows = [
        {
            "type_name": "cabinet",
            "parent_type": "fixture",
            "special_supertypes": ["storage"],
        },
        {
            "type_name": "bag",
            "parent_type": "movable_item",
            "special_supertypes": ["storage"],
        },
    ]

    assert build_type_parent_map(rows) == {
        "cabinet": "fixture",
        "bag": "movable_item",
    }
