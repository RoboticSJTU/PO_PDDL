from po_pddl.runtime.planning.data_structures import Action, ObservationRule
from po_pddl.runtime.planning.parser import parse_domain

DOMAIN = """
(define (domain observation-context)
  (:types item - object)
  (:predicates (ready ?x - item))
  (:observables (obs-ready ?x - item))
  (:action inspect
    :parameters (?x - item)
    :precondition (ready ?x)
    :effect (and))
  (:observation initial-ready
    :last-action init
    :parameters (?x - item)
    :condition (ready ?x)
    :distribution (obs-ready ?x))
  (:observation after-inspect
    :last-action inspect
    :parameters (?x - item)
    :condition (ready ?x)
    :distribution (obs-ready ?x))
  (:observation generic-ready
    :parameters (?x - item)
    :condition (ready ?x)
    :distribution (obs-ready ?x)))
"""


def test_parser_records_optional_last_action_context():
    parsed = parse_domain(DOMAIN)
    assert [rule.last_action for rule in parsed.observation_rules] == ["init", "inspect", None]
    assert [rule.rule.last_action for rule in parsed.observation_rules] == ["init", "inspect", None]


def test_semantic_context_matching_without_state_predicates():
    from po_pddl.runtime.planning.base.pomdp_model import POMDPModelBase

    matcher = POMDPModelBase.observation_rule_matches_last_action
    init = ObservationRule("init", last_action="init")
    after = ObservationRule("after", last_action="inspect")
    generic = ObservationRule("generic")
    assert matcher(None, init, None)
    assert not matcher(None, init, Action("inspect", ["cup"]))
    assert matcher(None, after, Action("inspect", ["cup"]))
    assert not matcher(None, after, Action("open", ["drawer"]))
    assert matcher(None, generic, None)
    assert matcher(None, generic, Action("inspect", ["cup"]))
