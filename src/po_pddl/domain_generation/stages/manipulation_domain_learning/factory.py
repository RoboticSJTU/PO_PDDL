from __future__ import annotations

import logging
from argparse import Namespace
from dataclasses import dataclass

from po_pddl.config import DEFAULT_MODEL

from .effect_completeness_review import VLMEffectCompletenessReviewModule
from .effect_variant_review import LLMEffectVariantReviewModule
from .episode_grounded_effect_learning import LLMEpisodeGroundedEffectLearningModule
from .learner import ManipulationDomainLearningLearner
from .object_typing import LLMObjectTypingModule
from .predicate_comments import LLMPredicateCommentModule
from .predicate_inventory import LLMPredicateInventoryModule
from .shared import load_llm_config
from .structured_template_modules import (
    InducedTemplateActionSchemaConsolidationModule,
    InducedTemplateRegistry,
    LLMSemanticActionTextPreprocessingModule,
    LLMTemplateActionCategoryModule,
    SemanticTemplateActionTaxonomyModule,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModuleModeSummary:
    action_taxonomy_module: str
    action_schema_consolidation_module: str
    manipulation_effect_module: str

    def to_dict(self) -> dict[str, str]:
        return {
            "action_taxonomy_module": self.action_taxonomy_module,
            "action_schema_consolidation_module": self.action_schema_consolidation_module,
            "manipulation_effect_module": self.manipulation_effect_module,
        }


def build_manipulation_domain_learner_from_args(
    args: Namespace,
) -> tuple[ManipulationDomainLearningLearner, ModuleModeSummary]:
    logger.info("Loading LLM configuration for manipulation-domain learning")
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
    max_tokens = int(getattr(args, "max_tokens", 4000))
    max_workers = int(getattr(args, "max_workers", 1))
    max_iterations = int(getattr(args, "max_iterations", 3))
    verbose = bool(getattr(args, "verbose", False))
    logger.info(
        "Manipulation-domain learning LLM configuration ready: model=%s, base_url=%s, temperature=%s, max_tokens=%d, max_workers=%d, max_iterations=%d",
        model,
        base_url or "<default>",
        temperature,
        max_tokens,
        max_workers,
        max_iterations,
    )

    registry = InducedTemplateRegistry()
    preprocessing_module = LLMSemanticActionTextPreprocessingModule(
        registry=registry,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=verbose,
    )
    category_module = LLMTemplateActionCategoryModule(
        registry=registry,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=verbose,
    )

    learner = ManipulationDomainLearningLearner(
        output_dir=getattr(args, "output_dir", None),
        episode_object_inventory_builder=lambda steps: [],
        action_text_preprocessing_module=preprocessing_module,
        object_typing_module=LLMObjectTypingModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        ),
        action_taxonomy_module=SemanticTemplateActionTaxonomyModule(
            registry=registry,
            category_module=category_module,
            max_workers=max_workers,
        ),
        action_schema_consolidation_module=InducedTemplateActionSchemaConsolidationModule(
            registry=registry,
        ),
        manipulation_effect_module=LLMEpisodeGroundedEffectLearningModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_workers=max_workers,
            max_iterations=max_iterations,
            verbose=verbose,
        ),
        effect_merge_module=None,
        effect_variant_review_module=LLMEffectVariantReviewModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_workers=max_workers,
            verbose=verbose,
        ),
        effect_completeness_review_module=VLMEffectCompletenessReviewModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        ),
        predicate_inventory_module=LLMPredicateInventoryModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        ),
        predicate_comment_module=LLMPredicateCommentModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        ),
        action_template_builder=lambda taxonomy_records: registry.export_artifacts(),
    )
    return learner, ModuleModeSummary(
        action_taxonomy_module="semantic_template_category:generic",
        action_schema_consolidation_module="induced_template:generic",
        manipulation_effect_module="episode_grounded_llm",
    )

