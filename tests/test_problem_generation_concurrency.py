from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from po_pddl.core.models.factorized_belief import FactorizedBelief
from po_pddl.core.models.predicate import Predicate
from po_pddl.problem_generation.generator import ProblemGenerator
from po_pddl.problem_generation.models import PredicateTruthJudgment, VisibleObject


class _ObjectAgent:
    def infer_visible_objects(self, **_kwargs) -> list[VisibleObject]:
        return [VisibleObject(name="item", type_name="movable_item")]

    def infer_named_object_types(self, **_kwargs) -> list[VisibleObject]:
        return []

    @staticmethod
    def to_object_declarations(objects: list[VisibleObject]):
        return [item.to_object_declaration() for item in objects]


class _InitialBeliefAgent:
    def __init__(self, barrier: threading.Barrier) -> None:
        self._barrier = barrier
        self.inference_strategy = "batch"
        self.last_inference_diagnostics = {
            "mode": "test",
            "deterministic_predicate_count": 1,
            "uncertain_predicate_count": 0,
        }

    def set_initial_state_hint(self, _hint: str | None) -> None:
        pass

    def set_current_visible_object_names(self, _names: set[str] | None) -> None:
        pass

    def infer_init_and_belief(self, **_kwargs):
        self._barrier.wait(timeout=2)
        predicate = Predicate("ready", ["item"])
        judgment = PredicateTruthJudgment(predicate=predicate, truth_value=True)
        return {predicate: True}, FactorizedBelief(known_true=[predicate]), [judgment]


class _GoalAgent:
    def __init__(self, barrier: threading.Barrier) -> None:
        self._barrier = barrier

    def infer_goal_expr(self, **_kwargs):
        self._barrier.wait(timeout=2)
        return ["ready", "item"]


class _SequentialInitialBeliefAgent:
    def __init__(self) -> None:
        self.current_visible_names: set[str] | None = {"unexpected"}
        self.inference_strategy = "batch"
        self.last_inference_diagnostics = {
            "mode": "test",
            "deterministic_predicate_count": 1,
            "uncertain_predicate_count": 0,
        }

    def set_initial_state_hint(self, _hint: str | None) -> None:
        pass

    def set_current_visible_object_names(self, names: set[str] | None) -> None:
        self.current_visible_names = names

    def infer_init_and_belief(self, **_kwargs):
        predicate = Predicate("ready", ["item"])
        judgment = PredicateTruthJudgment(predicate=predicate, truth_value=True)
        return {predicate: True}, FactorizedBelief(known_true=[predicate]), [judgment]


class _CloseDomainObjectAgent(_ObjectAgent):
    def __init__(self) -> None:
        self.visible_extraction_calls = 0

    def infer_visible_objects(self, **_kwargs) -> list[VisibleObject]:
        self.visible_extraction_calls += 1
        raise AssertionError("close-domain mode must not run visible-object extraction")


def test_initial_belief_and_goal_inference_run_concurrently(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "scene.jpg"
    image_path.write_bytes(b"unused")

    @contextmanager
    def fake_prepared_image(_path):
        yield SimpleNamespace(
            image_path=image_path,
            image_input_note="single view",
            is_stitched_multiview=False,
            source_image_paths=(image_path,),
        )

    monkeypatch.setattr("po_pddl.problem_generation.generator.prepare_scene_image", fake_prepared_image)
    barrier = threading.Barrier(2)
    generator = ProblemGenerator(
        object_agent=_ObjectAgent(),
        init_belief_agent=_InitialBeliefAgent(barrier),
        goal_agent=_GoalAgent(barrier),
    )

    result = generator.build_problem_from_text(
        domain_text="""
        (define (domain test)
          (:requirements :strips :typing)
          (:types movable_item - object)
          (:predicates (ready ?item - movable_item)))
        """,
        image_path=image_path,
        instruction="Make the item ready.",
        concurrent_inference_branches=True,
    )

    assert result.spec.goal_expr == ["ready", "item"]
    assert Predicate("ready", ["item"]) in result.spec.init_belief.known_true


def test_close_domain_skips_visible_object_extraction(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "scene.jpg"
    image_path.write_bytes(b"unused")
    domain_file = tmp_path / "domain.pddl"
    domain_file.write_text("unused", encoding="utf-8")
    objects_file = tmp_path / "objects.txt"
    objects_file.write_text("item\n", encoding="utf-8")
    grounding_root = tmp_path / "4_problem_grounding_all"
    grounding_root.mkdir()

    @contextmanager
    def fake_prepared_image(_path):
        yield SimpleNamespace(
            image_path=image_path,
            image_input_note="single view",
            is_stitched_multiview=False,
            source_image_paths=(image_path,),
        )

    object_agent = _CloseDomainObjectAgent()
    belief_agent = _SequentialInitialBeliefAgent()
    generator = ProblemGenerator(
        object_agent=object_agent,
        init_belief_agent=belief_agent,
        goal_agent=type("Goal", (), {"infer_goal_expr": lambda self, **_kwargs: ["ready", "item"]})(),
    )
    monkeypatch.setattr("po_pddl.problem_generation.generator.prepare_scene_image", fake_prepared_image)
    monkeypatch.setattr(
        generator,
        "_resolve_historical_grounding_root",
        lambda **_kwargs: grounding_root,
    )
    monkeypatch.setattr(
        generator,
        "_infer_closed_domain_objects_from_grounding",
        lambda **_kwargs: [VisibleObject(name="item", type_name="movable_item")],
    )

    result = generator.build_problem_from_text(
        domain_text="""
        (define (domain test)
          (:requirements :strips :typing)
          (:types movable_item - object)
          (:predicates (ready ?item - movable_item)))
        """,
        domain_file=domain_file,
        image_path=image_path,
        instruction="Make the item ready.",
        objects_file=objects_file,
        close_domain=True,
    )

    assert object_agent.visible_extraction_calls == 0
    assert belief_agent.current_visible_names is None
    assert [item.name for item in result.visible_objects] == ["item"]
