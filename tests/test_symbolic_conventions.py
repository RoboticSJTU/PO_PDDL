from po_pddl.core.conventions import (
    CONTAINMENT_CONTAINER_INDEX,
    CONTAINMENT_MOVABLE_INDEX,
    CONTAINMENT_PARAMETER_TYPES,
    containment_arguments,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.models import (
    ObjectTypeDefinition,
    PredicateSchema,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.problem_grounding_runner import (
    _canonicalize_fact,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.renderer import (
    _render_types_block,
)


def test_containment_uses_movable_subject_then_container_reference() -> None:
    assert CONTAINMENT_PARAMETER_TYPES == ("movable_item", "containable_item")
    assert CONTAINMENT_MOVABLE_INDEX == 0
    assert CONTAINMENT_CONTAINER_INDEX == 1
    assert containment_arguments("item_a", "container_b") == ["item_a", "container_b"]


def test_grounding_canonicalizes_containment_to_subject_reference_order() -> None:
    object_types = {"item_a": "block", "container_b": "drawer"}

    assert (
        _canonicalize_fact("in(container_b,item_a)", object_type_by_name=object_types)
        == "in(item_a,container_b)"
    )
    assert (
        _canonicalize_fact("in(item_a,container_b)", object_type_by_name=object_types)
        == "in(item_a,container_b)"
    )


def test_fixed_containers_compile_to_a_real_containable_subtype() -> None:
    lines = _render_types_block(
        [],
        predicate_inventory=[
            PredicateSchema(
                predicate_name="in",
                parameter_types=["movable_item", "containable_item"],
                comment=None,
            )
        ],
        object_types=[
            ObjectTypeDefinition(
                type_name="drawer",
                member_object_names=["drawer"],
                parent_type="fixed_item",
                special_supertypes=["containable_item", "fixed_item"],
            ),
            ObjectTypeDefinition(
                type_name="refrigerator",
                member_object_names=["refrigerator"],
                parent_type="fixed_item",
                special_supertypes=["containable_item", "fixed_item"],
            ),
        ],
    )

    assert "    movable_item fixed_item - object" in lines
    assert "    containable_item - fixed_item" in lines
    assert "    drawer refrigerator - containable_item" in lines
