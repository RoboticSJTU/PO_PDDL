import json

from po_pddl.domain_generation.stages.manipulation_domain_learning import structured_template_modules as modules
from po_pddl.domain_generation.stages.manipulation_domain_learning.models import RawTrajectoryStep


def _step(episode: str, index: int, action_text: str, *, frame_path: str = "") -> RawTrajectoryStep:
    return RawTrajectoryStep(
        episode_name=episode,
        instruction="Complete the task.",
        step_index=index,
        start_time_sec=0.0,
        end_time_sec=1.0,
        action_text=action_text,
        observation_text=None,
        extra_info=None,
        previous_observation_text=None,
        previous_known_observation_text=None,
        frame_paths=[frame_path] if frame_path else [],
    )


def _template_response(*, include_open: bool = True) -> str:
    templates = [
        {
            "template_id": "ignored_by_validation",
            "template_text": "pick up the {param_1}",
            "canonical_action_name": "ignored_by_validation",
            "action_category": "manipulation",
        }
    ]
    if include_open:
        templates.append(
            {
                "template_id": "ignored_by_validation",
                "template_text": "open the {param_1}",
                "canonical_action_name": "ignored_by_validation",
                "action_category": "manipulation",
            }
        )
    return json.dumps({"action_templates": templates})


def test_action_templates_are_induced_once_from_the_full_dataset(monkeypatch) -> None:
    calls: list[dict] = []

    monkeypatch.setattr(modules, "make_client", lambda **_kwargs: object())

    def fake_chat(_client, _prompt, payload, **_kwargs):
        calls.append(json.loads(payload))
        return _template_response()

    monkeypatch.setattr(modules, "safe_chat", fake_chat)
    registry = modules.InducedTemplateRegistry()
    preprocessor = modules.LLMSemanticActionTextPreprocessingModule(registry=registry)
    steps = [
        _step("episode0", 0, "pick up the red block", frame_path="frame.jpg"),
        _step("episode1", 0, "pick up the blue block"),
        _step("episode2", 0, "open the green drawer"),
    ]

    result = preprocessor.preprocess_steps(steps)

    assert len(calls) == 1
    assert {row["action_text"] for row in calls[0]["action_texts"]} == {
        "pick up the red block",
        "pick up the blue block",
        "open the green drawer",
    }
    assert [row.canonical_action_name for row in registry.templates] == ["pick_up_object", "open_object"]
    assert [row.resolution_kind for row in result.normalization_records] == ["global_induction"] * 3
    assert result.normalized_steps[0].frame_paths == ["frame.jpg"]

    category_module = modules.LLMTemplateActionCategoryModule(registry=registry)
    classified = category_module.classify_template_categories(steps=result.normalized_steps)
    assert len(calls) == 1
    assert [template.action_category for template in classified] == ["manipulation", "manipulation"]


def test_invalid_global_inventory_is_retried_as_one_batch(monkeypatch) -> None:
    calls: list[dict] = []

    monkeypatch.setattr(modules, "make_client", lambda **_kwargs: object())

    def fake_chat(_client, _prompt, payload, **_kwargs):
        calls.append(json.loads(payload))
        return _template_response(include_open=len(calls) > 1)

    monkeypatch.setattr(modules, "safe_chat", fake_chat)
    preprocessor = modules.LLMSemanticActionTextPreprocessingModule(
        registry=modules.InducedTemplateRegistry()
    )

    result = preprocessor.preprocess_steps(
        [
            _step("episode0", 0, "pick up the red block"),
            _step("episode1", 0, "open the green drawer"),
        ]
    )

    assert len(result.normalized_steps) == 2
    assert len(calls) == 2
    assert "validation_feedback" not in calls[0]
    assert "open the green drawer" in calls[1]["validation_feedback"]
    assert len(calls[1]["action_texts"]) == 2
