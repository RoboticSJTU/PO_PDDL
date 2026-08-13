from __future__ import annotations

import logging
from argparse import Namespace
from dataclasses import dataclass

from po_pddl.config import DEFAULT_MODEL

from .learner import PassiveObservationLearner
from .modules import (
    LLMContainableRevealActionSelectionModule,
    LLMDescriptionContradictionReviewModule,
    LLMVisibleContainedObjectsModule,
    VLMContradictionConfirmationModuleLLM,
)
from .shared import load_llm_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModuleModeSummary:
    description_review_module: str
    vlm_confirmation_module: str


def build_learner_from_args(args: Namespace) -> tuple[PassiveObservationLearner, ModuleModeSummary]:
    logger.debug("Loading LLM configuration")
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
    max_tokens = int(getattr(args, "max_tokens", 2500))
    verbose = bool(getattr(args, "verbose", False))
    learner = PassiveObservationLearner(
        description_review_module=LLMDescriptionContradictionReviewModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        ),
        vlm_confirmation_module=VLMContradictionConfirmationModuleLLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        ),
        containable_reveal_action_selection_module=LLMContainableRevealActionSelectionModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        ),
        visible_contained_objects_module=LLMVisibleContainedObjectsModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        ),
        manipulation_artifact_dir=getattr(args, "manipulation_artifact_dir", None),
        scene_description_dir=getattr(args, "scene_description_dir", None),
        max_workers=max(1, int(getattr(args, "max_workers", 1))),
    )
    return learner, ModuleModeSummary(
        description_review_module="llm",
        vlm_confirmation_module="vlm",
    )


__all__ = ["ModuleModeSummary", "build_learner_from_args"]
