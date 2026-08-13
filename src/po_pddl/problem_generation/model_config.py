from __future__ import annotations

from dataclasses import dataclass

from ..domain_generation.infrastructure.llm_shared import load_llm_config


@dataclass(frozen=True)
class ResolvedLLMConfig:
    model: str
    api_key: str | None
    base_url: str | None
    temperature: float


def resolve_online_llm_config(
    *,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    temperature: float | None,
    config_path: str | None = None,
    config_name: str | None = None,
    default_model: str,
    default_temperature: float,
) -> ResolvedLLMConfig:
    config = load_llm_config(config_path=config_path, config_name=config_name)
    resolved_temperature = temperature
    if resolved_temperature is None:
        config_temperature = config.get("temperature")
        if config_temperature is None:
            resolved_temperature = default_temperature
        else:
            resolved_temperature = float(config_temperature)

    return ResolvedLLMConfig(
        model=model or config.get("model") or default_model,
        api_key=api_key or config.get("api_key"),
        base_url=base_url or config.get("base_url"),
        temperature=resolved_temperature,
    )
