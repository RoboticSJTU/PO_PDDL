from po_pddl.domain_generation.infrastructure.llm_shared.llm_client import (
    _build_request_kwargs,
)


def test_gpt_56_sol_omits_unsupported_temperature() -> None:
    request = _build_request_kwargs(
        model="gpt-5.6-sol",
        system_prompt="system",
        user_content="user",
        temperature=0.0,
        max_tokens=100,
        use_max_completion_tokens=True,
    )
    assert "temperature" not in request
    assert request["max_completion_tokens"] == 100


def test_other_models_keep_temperature() -> None:
    request = _build_request_kwargs(
        model="example-model",
        system_prompt="system",
        user_content="user",
        temperature=0.2,
        max_tokens=100,
        use_max_completion_tokens=False,
    )
    assert request["temperature"] == 0.2
