from __future__ import annotations

from argparse import Namespace

from po_pddl.config import DEFAULT_MODEL

from .learner import PreconditionLearningLearner
from .modules import LLMPreconditionSelectionModule
from .shared import load_llm_config


def build_learner_from_args(args: Namespace) -> PreconditionLearningLearner:
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
    max_workers = int(getattr(args, "max_workers", 1))
    keep_all = bool(getattr(args, "precondition_learning_keep_all_intersection_preconditions", False))
    return PreconditionLearningLearner(
        selection_module=(
            None
            if keep_all
            else LLMPreconditionSelectionModule(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=int(getattr(args, "max_tokens", 4000)),
                verbose=bool(getattr(args, "verbose", False)),
            )
        ),
        max_workers=max_workers,
        keep_all_intersection_preconditions=keep_all,
    )
