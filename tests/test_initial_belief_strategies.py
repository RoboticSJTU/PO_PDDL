import json
from pathlib import Path

from po_pddl.core.models.predicate import Predicate
from po_pddl.domain_generation.stages.problem_grounding.models import ObjectDeclaration
from po_pddl.problem_generation.domain_analysis import analyze_domain
from po_pddl.problem_generation.initial_belief import InitialBeliefGenerator


def test_deterministic_predicates_are_judged_in_one_batch(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "scene.jpg"
    image_path.write_bytes(b"test-image")
    analysis = analyze_domain(
        """
        (define (domain test)
          (:requirements :strips :typing)
          (:types movable_item surface - object)
          (:predicates
            (in_front_of ?item - movable_item ?surface - surface)
            (on_top_of ?item - movable_item ?surface - surface)))
        """
    )
    predicates = [
        Predicate("in_front_of", ["cup", "drawer"]),
        Predicate("on_top_of", ["cup", "drawer"]),
    ]
    calls: list[object] = []

    monkeypatch.setattr(
        "po_pddl.problem_generation.initial_belief.make_client",
        lambda **_kwargs: object(),
    )

    def fake_safe_chat(*args, **kwargs):
        calls.append(kwargs.get("user_content", args[2] if len(args) > 2 else None))
        return json.dumps(
            {
                "scene_reasoning": {
                    "object_identity_and_views": "one cup and one drawer",
                    "spatial_relations": "the cup is in front of the drawer",
                    "consistency_check": "only one location is true",
                },
                "predicate_judgments": [
                    {
                        "predicate": "(in_front_of cup drawer)",
                        "truth_value": "true",
                        "justification": "visible in front",
                    },
                    {
                        "predicate": "(on_top_of cup drawer)",
                        "truth_value": "false",
                        "justification": "not supported by drawer",
                    },
                ],
            }
        )

    monkeypatch.setattr("po_pddl.problem_generation.initial_belief.safe_chat", fake_safe_chat)
    agent = InitialBeliefGenerator(model="test-model")
    agent.set_initial_state_hint("The drawer is closed.")

    judgments = agent._judge_deterministic_predicates_batch(
        domain_analysis=analysis,
        image_path=image_path,
        image_input_note="single view",
        instruction="Place the cup on the drawer.",
        objects=[
            ObjectDeclaration(name="cup", type_name="movable_item"),
            ObjectDeclaration(name="drawer", type_name="surface"),
        ],
        predicates=predicates,
    )

    assert len(calls) == 1
    assert '"initial_state_hint": "The drawer is closed."' in str(calls[0])
    assert {item.predicate: item.truth_value for item in judgments} == {
        predicates[0]: True,
        predicates[1]: False,
    }


def test_deterministic_predicate_chunks_can_be_judged_in_parallel(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "scene.jpg"
    image_path.write_bytes(b"test-image")
    analysis = analyze_domain(
        """
        (define (domain test)
          (:requirements :strips :typing)
          (:types movable_item surface - object)
          (:predicates
            (in_front_of ?item - movable_item ?surface - surface)
            (on_top_of ?item - movable_item ?surface - surface)))
        """
    )
    predicates = [
        Predicate("in_front_of", ["cup", "drawer"]),
        Predicate("on_top_of", ["cup", "drawer"]),
    ]
    calls: list[object] = []
    monkeypatch.setattr(
        "po_pddl.problem_generation.initial_belief.make_client",
        lambda **_kwargs: object(),
    )

    def fake_safe_chat(*args, **kwargs):
        content = kwargs.get("user_content", args[2] if len(args) > 2 else None)
        calls.append(content)
        text_block = next(block for block in content if block["type"] == "text")
        payload = json.loads(text_block["text"])
        predicates = payload["candidate_predicates"]
        return json.dumps(
            {
                "scene_reasoning": {
                    "object_identity_and_views": "one cup and drawer",
                    "spatial_relations": "jointly evaluated",
                    "consistency_check": "one predicate in this chunk",
                },
                "predicate_judgments": [
                    {
                        "predicate": predicate,
                        "truth_value": "true" if predicate.startswith("(in_front_of") else "false",
                        "justification": "chunked image judgment",
                    }
                    for predicate in predicates
                ],
            }
        )

    monkeypatch.setattr("po_pddl.problem_generation.initial_belief.safe_chat", fake_safe_chat)
    agent = InitialBeliefGenerator(
        model="test-model",
        inference_strategy="parallel",
        inference_batch_size=1,
    )
    judgments = agent._judge_deterministic_predicates(
        domain_analysis=analysis,
        image_path=image_path,
        image_input_note="single view",
        instruction="Place the cup on the drawer.",
        objects=[
            ObjectDeclaration(name="cup", type_name="movable_item"),
            ObjectDeclaration(name="drawer", type_name="surface"),
        ],
        predicates=predicates,
        fixed_judgments=[],
        max_workers=2,
    )

    assert len(calls) == 2
    assert {item.predicate: item.truth_value for item in judgments} == {
        predicates[0]: True,
        predicates[1]: False,
    }


def test_location_visibility_batches_never_split_an_object_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "scene.jpg"
    image_path.write_bytes(b"test-image")
    analysis = analyze_domain(
        """
        (define (domain test)
          (:requirements :strips :typing)
          (:types movable_item surface - object)
          (:predicates (on_top_of ?item - movable_item ?surface - surface)))
        """
    )
    objects = [
        ObjectDeclaration(name="black_box", type_name="movable_item"),
        ObjectDeclaration(name="blue_block", type_name="movable_item"),
        ObjectDeclaration(name="drawer", type_name="surface"),
    ]
    calls: list[dict[str, object]] = []
    agent = InitialBeliefGenerator(
        model="test-model",
        inference_strategy="parallel",
        location_visibility_batch_size=3,
    )
    agent.set_current_visible_object_names(None)

    monkeypatch.setattr(
        "po_pddl.problem_generation.initial_belief.make_client",
        lambda **_kwargs: object(),
    )

    def fake_safe_chat(*args, **kwargs):
        content = kwargs.get("user_content", args[2] if len(args) > 2 else None)
        text_block = next(block for block in content if block["type"] == "text")
        payload = json.loads(text_block["text"])
        calls.append(payload)
        return json.dumps(
            {
                "object_visibility": [
                    {
                        "target_object": item["target_object"],
                        "visually_resolved": item["target_object"] == "black_box",
                        "reasoning": "global scene judgment",
                    }
                    for item in payload["target_location_groups"]
                ]
            }
        )

    monkeypatch.setattr("po_pddl.problem_generation.initial_belief.safe_chat", fake_safe_chat)
    results = agent._classify_object_location_visibility(
        domain_analysis=analysis,
        image_path=image_path,
        image_input_note="single view",
        instruction="Move the objects.",
        objects=objects,
        grouped_location_predicates={
            "black_box": [
                Predicate("on_top_of", ["black_box", "drawer"]),
                Predicate("in_front_of", ["black_box", "drawer"]),
                Predicate("left_of", ["black_box", "drawer"]),
                Predicate("right_of", ["black_box", "drawer"]),
            ],
            "blue_block": [
                Predicate("on_top_of", ["blue_block", "drawer"]),
                Predicate("in_front_of", ["blue_block", "drawer"]),
            ],
        },
        max_workers=2,
    )

    assert len(calls) == 2
    object_occurrences = [
        group["target_object"]
        for call in calls
        for group in call["target_location_groups"]
    ]
    assert object_occurrences.count("black_box") == 1
    assert object_occurrences.count("blue_block") == 1
    black_box_group = next(
        group
        for call in calls
        for group in call["target_location_groups"]
        if group["target_object"] == "black_box"
    )
    assert len(black_box_group["candidate_location_predicates"]) == 4
    assert results == [("black_box", True), ("blue_block", False)]
