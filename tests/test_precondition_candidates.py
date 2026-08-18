from po_pddl.core.parser import parse_domain
from po_pddl.domain_generation.stages.precondition_learning.learner import (
    _abstract_state_before_candidates,
    _missing_ground_facts_for_objects,
)

DOMAIN = """
(define (domain static-qualifier-test)
  (:requirements :strips :typing)
  (:types drawer - object)
  (:predicates (is_left ?drawer - drawer))
)
"""


def _candidates(state_before: list[str]) -> set[str]:
    parsed_domain = parse_domain(DOMAIN)
    known_objects = [("green_drawer", "drawer"), ("yellow_drawer", "drawer")]
    state_set = set(state_before)
    missing = _missing_ground_facts_for_objects(
        parsed_domain=parsed_domain,
        known_objects=known_objects,
        state_set=state_set,
        zero_arity_predicates=set(),
    )
    candidates, _eligible = _abstract_state_before_candidates(
        parsed_domain=parsed_domain,
        known_objects=known_objects,
        state_before=state_before,
        ground_arguments=["green_drawer"],
        object_type_by_name=dict(known_objects),
        missing_ground_facts=missing,
    )
    return candidates


def test_static_parameter_qualifier_can_be_positive_precondition_candidate() -> None:
    assert "is_left(?arg0)" in _candidates(["is_left(green_drawer)"])


def test_static_parameter_qualifier_can_be_negative_precondition_candidate() -> None:
    assert "not is_left(?arg0)" in _candidates(["is_left(yellow_drawer)"])
