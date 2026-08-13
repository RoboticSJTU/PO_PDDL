"""Shared symbolic conventions used across generation pipelines."""

CONTAINMENT_PREDICATE = "in"
CONTAINMENT_PARAMETER_TYPES = ("movable_item", "containable_item")
CONTAINMENT_MOVABLE_INDEX = 0
CONTAINMENT_CONTAINER_INDEX = 1


def containment_arguments(movable_item: str, containable_item: str) -> list[str]:
    """Return containment arguments in canonical subject-reference order."""
    return [movable_item, containable_item]
