"""Exercise cup clearance using the shipped domains and real transition compiler."""

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from po_pddl.runtime.planning.bitwise import (
    build_bitwise_index_layout,
    build_bitwise_model_from_parsed,
    state_to_bitvector,
)
from po_pddl.runtime.planning.codegen import build_codegen_plan_from_parsed
from po_pddl.runtime.planning.data_structures import Action, Predicate
from po_pddl.runtime.planning.engine.runtime_parser import _specialize_last_action_domain_and_problem
from po_pddl.runtime.planning.linter import lint_domain_text
from po_pddl.runtime.planning.parser import parse_domain, parse_problem

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = [ROOT / "example_problem/domain.pddl"]
DOMAINS += sorted((ROOT / "examples/final_bundle").glob("**/*domain*.pddl"))
SCHEMA_FRAGMENT = ROOT / "examples/final_bundle/manipulation_domain/precondition_action_schemas.pddl"
if SCHEMA_FRAGMENT.exists():
    DOMAINS.append(SCHEMA_FRAGMENT)


@lru_cache(maxsize=None)
def _compile_domain(domain_path):
    domain_text = domain_path.read_text(encoding="utf-8")
    assert lint_domain_text(domain_text).ok
    domain = parse_domain(domain_text)
    problem = parse_problem((ROOT / "example_problem/problem_online.pddl").read_text(encoding="utf-8"))
    # Some shipped intermediate domains use a different fragment name.
    problem.domain_name = domain.domain_name
    problem.goal = ["and", ["open", "green_drawer"], ["open", "yellow_drawer"]]
    domain, problem, _ = _specialize_last_action_domain_and_problem(domain, problem, None)
    plan = build_codegen_plan_from_parsed(domain, problem)
    metadata = SimpleNamespace(
        predicates=plan.predicates,
        observables=plan.observables,
        actions=[case.action for case in plan.grounded_actions],
        observation_rules=[case.rule for case in plan.grounded_observation_rules],
        default_policy_rules=[],
        maximize_reward=plan.maximize_reward,
        goal_reward=plan.goal_reward,
    )
    layout = build_bitwise_index_layout(metadata)
    model = build_bitwise_model_from_parsed(metadata, parsed_domain=domain, parsed_problem=problem)
    return problem, layout, model


@pytest.mark.parametrize("domain_path", DOMAINS, ids=lambda path: str(path.relative_to(ROOT)))
@pytest.mark.parametrize("right_first", [True, False])
@pytest.mark.parametrize("box_on_right", [True, False])
def test_two_cups_can_clear_drawers_with_an_occupied_top(domain_path, right_first, box_on_right):
    problem, layout, model = _compile_domain(domain_path)
    initial = dict(problem.init_state)
    if not box_on_right:
        initial[Predicate("on_top_of", ["black_box", "green_drawer"])] = False
        initial[Predicate("on_top_of", ["black_box", "yellow_drawer"])] = True
    state = state_to_bitvector(initial, layout)

    def action_id(name, *params):
        return layout.action_index[Action(name, list(params))]

    # The actual access-path constraint must still prevent opening either drawer.
    for side, drawer in [("right", "green_drawer"), ("left", "yellow_drawer")]:
        assert not model.check_action_precondition(action_id(f"open_object_on_{side}", drawer), state)

    cups = [("right", "green_cup", "green_drawer"), ("left", "pink_cup", "yellow_drawer")]
    if not right_first:
        cups.reverse()
    for side, cup, source in cups:
        pickup = action_id(f"pick_up_object_in_front_of_object_on_{side}", cup, source)
        place = action_id("place_object_on_top_of_object_on_right", cup, "green_drawer")
        assert not model.check_action_precondition(place, state)
        assert model.check_action_precondition(pickup, state)
        state, _ = model.forward_action(pickup, state)
        assert not model.check_action_precondition(pickup, state)
        assert model.check_action_precondition(place, state)
        state, _ = model.forward_action(place, state)

    for side, drawer in [("right", "green_drawer"), ("left", "yellow_drawer")]:
        opening = action_id(f"open_object_on_{side}", drawer)
        assert model.check_action_precondition(opening, state)
        state, _ = model.forward_action(opening, state)
    assert model.is_goal(state)
    for cup in ("green_cup", "pink_cup"):
        assert state & (1 << layout.predicate_index[Predicate("on_top_of", [cup, "green_drawer"])])
    box_support = "green_drawer" if box_on_right else "yellow_drawer"
    assert state & (1 << layout.predicate_index[Predicate("on_top_of", ["black_box", box_support])])
