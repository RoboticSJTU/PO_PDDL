from __future__ import annotations

import logging
from argparse import Namespace
from dataclasses import dataclass

from po_pddl.config import DEFAULT_MODEL

from .inferer import ProblemInferenceLearner
from .modules import (
    LLMInitCompletionModule,
    LLMInitialStateInferenceModule,
    LLMLatentObjectDiscoveryModule,
    LLMVisibleObjectExtractionModule,
)
from .shared import load_llm_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModuleModeSummary:
    visible_object_module: str
    latent_object_module: str
    initial_state_module: str
    init_completion_module: str


def build_learner_from_args(args: Namespace) -> tuple[ProblemInferenceLearner, ModuleModeSummary]:
    logger.info("Loading LLM configuration for problem inference")
    config = load_llm_config(
        getattr(args, "config", None),
        config_name=getattr(args, "config_name", "openai_config"),
    )
    model = getattr(args, "model", None) or config.get("model") or DEFAULT_MODEL
    api_key = getattr(args, "api_key", None) or config.get("api_key")
    base_url = getattr(args, "base_url", None) or config.get("base_url")
    temperature = (
        getattr(args, "temperature", None)
        if getattr(args, "temperature", None) is not None
        else (config.get("temperature") if config.get("temperature") is not None else 0.0)
    )
    max_tokens = int(getattr(args, "max_tokens", 3000))
    verbose = bool(getattr(args, "verbose", False))
    learner = ProblemInferenceLearner(
        visible_object_module=LLMVisibleObjectExtractionModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        ),
        latent_object_module=LLMLatentObjectDiscoveryModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        ),
        initial_state_module=LLMInitialStateInferenceModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        ),
        init_completion_module=LLMInitCompletionModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        ),
    )
    return learner, ModuleModeSummary(
        visible_object_module="llm",
        latent_object_module="llm",
        initial_state_module="llm",
        init_completion_module="llm_optional_grounded_completion",
    )
