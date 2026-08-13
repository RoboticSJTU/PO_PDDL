from __future__ import annotations

from argparse import Namespace

from po_pddl.config import DEFAULT_MODEL
from po_pddl.domain_generation.stages.manipulation_domain_learning.problem_grounding_runner import (
    ModuleModeSummary,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.shared import load_llm_config
from po_pddl.domain_generation.stages.problem_grounding.grounder import ProblemGroundingLearner
from po_pddl.domain_generation.stages.problem_grounding.modules import (
    LLMGoalInferenceModule,
    ProblemInferenceObjectInitModule,
    RuleBasedActionArgumentObjectsInitModule,
    RuleBasedProblemAssemblyModule,
    RuleBasedTemplateTrajectoryGroundingModule,
)


def build_learner_from_args(args: Namespace) -> tuple[ProblemGroundingLearner, ModuleModeSummary]:
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

    base_object_init = ProblemInferenceObjectInitModule.from_args(args)
    learner = ProblemGroundingLearner(
        object_init_module=RuleBasedActionArgumentObjectsInitModule(base=base_object_init),
        goal_inference_module=LLMGoalInferenceModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        ),
        assembly_module=RuleBasedProblemAssemblyModule(),
        trajectory_grounding_module=RuleBasedTemplateTrajectoryGroundingModule(),
    )
    return (
        learner,
        ModuleModeSummary(
            object_init_module="problem_inference_init_plus_rule_based_action_objects",
            goal_inference_module="llm",
            assembly_module="rule_based",
            trajectory_grounding_module="rule_based_template:generic",
            validator_module="rule_based",
        ),
    )


__all__ = ["ModuleModeSummary", "build_learner_from_args", "load_llm_config"]
