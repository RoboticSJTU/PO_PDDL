"""Stable service layer for scene-conditioned problem generation."""

from __future__ import annotations

from collections.abc import Callable

from po_pddl.config import DEFAULT_MODEL, ProblemGenerationConfig

from .generator import ProblemGenerator
from .goal import GoalInferenceAgent
from .initial_belief import InitialBeliefGenerator
from .model_config import resolve_online_llm_config
from .models import OnlinePlanningProblemResult
from .objects import VisibleObjectExtractionAgent


def generate_problem(
    config: ProblemGenerationConfig,
    *,
    logger: Callable[[str], None] | None = None,
) -> OnlinePlanningProblemResult:
    llm = resolve_online_llm_config(
        model=config.llm.model,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        temperature=config.llm.temperature,
        config_path=str(config.llm.config_path) if config.llm.config_path else None,
        config_name=config.llm.config_name,
        default_model=DEFAULT_MODEL,
        default_temperature=0.1,
    )
    common_agent_kwargs = {
        "model": llm.model,
        "base_url": llm.base_url,
        "api_key": llm.api_key,
        "temperature": llm.temperature,
        "max_tokens": config.llm.max_tokens,
        "verbose": config.verbose,
        "config_path": str(config.llm.config_path) if config.llm.config_path else None,
        "config_name": config.llm.config_name,
    }
    agent = ProblemGenerator(
        object_agent=VisibleObjectExtractionAgent(**common_agent_kwargs),
        init_belief_agent=InitialBeliefGenerator(
            **common_agent_kwargs,
            inference_strategy=config.inference_strategy,
            inference_batch_size=config.inference_batch_size,
            location_visibility_batch_size=config.location_visibility_batch_size,
        ),
        goal_agent=GoalInferenceAgent(
            **common_agent_kwargs,
            inference_strategy=config.inference_strategy,
            inference_batch_size=config.inference_batch_size,
        ),
        logger=logger,
    )
    return agent.build_problem_from_files(
        domain_file=config.domain_file,
        image_path=config.image_path,
        instruction=config.instruction,
        initial_state_hint=config.initial_state_hint,
        final_bundle_dir=config.final_bundle_dir,
        objects_file=config.objects_file,
        reuse_problem_file=config.reuse_problem_file,
        problem_name=config.problem_name,
        output_path=config.resolved_output_file,
        max_workers=config.max_workers,
        prior_data_confidence=config.prior_data_confidence,
        close_domain=config.close_domain,
        skip_init_observation=config.skip_init_observation,
        concurrent_inference_branches=config.concurrent_inference_branches,
    )
