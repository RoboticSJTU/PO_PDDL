import json

from po_pddl.core.models.predicate import Predicate
from po_pddl.problem_generation.goal import GoalInferenceAgent


def test_goal_assignments_are_evaluated_in_one_batch_and_returned_in_order(monkeypatch) -> None:
    agent = object.__new__(GoalInferenceAgent)
    agent.verbose = False
    agent.model = "test-model"
    agent.base_url = "test://local"
    agent.api_key = None
    agent.temperature = 0.0
    agent.max_tokens = 1024
    agent.inference_strategy = "batch"
    agent.inference_batch_size = 20
    agent._goal_state_prompt = "test prompt"
    calls: list[object] = []

    monkeypatch.setattr(
        "po_pddl.problem_generation.goal.make_client",
        lambda **_kwargs: object(),
    )

    def fake_safe_chat(*args, **kwargs):
        calls.append(kwargs.get("user_content", args[2] if len(args) > 2 else None))
        return json.dumps(
            {
                "assignment_evaluations": [
                    {
                        "assignment_id": f"assignment_{index:04d}",
                        "reasoning": f"evaluation {index}",
                        "satisfies_instruction": index % 2 == 0,
                    }
                    for index in range(1, 5)
                ]
            }
        )

    monkeypatch.setattr("po_pddl.problem_generation.goal.safe_chat", fake_safe_chat)
    predicate = Predicate("state", ["item"])
    assignments = [{predicate: value} for value in [False, True, False, True]]

    results = agent._evaluate_goal_assignments(
        domain_analysis=type(
            "Analysis",
            (),
            {
                "parsed_domain": type(
                    "Domain",
                    (),
                    {
                        "domain_name": "test",
                        "constants": {},
                        "predicates": [],
                        "predicate_parameter_types": {},
                        "types": {},
                        "actions": [],
                    },
                )(),
                "has_observation_module": False,
            },
        )(),
        instruction="test instruction",
        objects=[],
        assignments=assignments,
        max_workers=4,
    )

    assert len(calls) == 1
    assert [item.assignment_id for item in results] == [f"assignment_{index:04d}" for index in range(1, 5)]
    assert [item.satisfies_instruction for item in results] == [False, True, False, True]


def test_goal_assignment_chunks_can_be_evaluated_in_parallel(monkeypatch) -> None:
    agent = object.__new__(GoalInferenceAgent)
    agent.verbose = False
    agent.model = "test-model"
    agent.base_url = "test://local"
    agent.api_key = None
    agent.temperature = 0.0
    agent.max_tokens = 1024
    agent.inference_strategy = "parallel"
    agent.inference_batch_size = 2
    agent._goal_state_prompt = "test prompt"
    calls: list[object] = []

    monkeypatch.setattr("po_pddl.problem_generation.goal.make_client", lambda **_kwargs: object())

    def fake_safe_chat(*args, **kwargs):
        content = kwargs.get("user_content", args[2] if len(args) > 2 else None)
        calls.append(content)
        payload = json.loads(content)
        return json.dumps(
            {
                "assignment_evaluations": [
                    {
                        "assignment_id": item["assignment_id"],
                        "reasoning": "batched evaluation",
                        "satisfies_instruction": item["assignment_id"]
                        in {"assignment_0002", "assignment_0004"},
                    }
                    for item in payload["assignments"]
                ]
            }
        )

    monkeypatch.setattr("po_pddl.problem_generation.goal.safe_chat", fake_safe_chat)
    predicate = Predicate("state", ["item"])
    assignments = [{predicate: value} for value in [False, True, False, True]]

    results = agent._evaluate_goal_assignments(
        domain_analysis=type(
            "Analysis",
            (),
            {
                "parsed_domain": type(
                    "Domain",
                    (),
                    {
                        "domain_name": "test",
                        "constants": {},
                        "predicates": [],
                        "predicate_parameter_types": {},
                        "types": {},
                        "actions": [],
                    },
                )(),
                "has_observation_module": False,
            },
        )(),
        instruction="test instruction",
        objects=[],
        assignments=assignments,
        max_workers=2,
    )

    assert len(calls) == 2
    assert [item.assignment_id for item in results] == [f"assignment_{index:04d}" for index in range(1, 5)]
    assert [item.satisfies_instruction for item in results] == [False, True, False, True]
