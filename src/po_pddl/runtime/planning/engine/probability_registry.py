"""Probability registry for hot-reload of POMDPDDL domain probabilities.

Captures probability weight arrays from a BitwiseModelSkeleton and provides
functions to extract updated probabilities from re-parsed domain text, update
the C++ planner at runtime, and update the Python-side bitwise model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dc_replace

from ..bitwise.parser import (
    BitwiseModelSkeleton,
    build_bitwise_model_from_parsed,
)
from ..codegen import build_codegen_plan_from_parsed
from ..parser import (
    parse_default_policy,
    parse_domain,
    parse_problem,
)


@dataclass
class ProbabilityRegistry:
    """Maps action/observation-rule indices to their probability weight vectors."""

    action_weights: dict[int, list[float]] = field(default_factory=dict)
    observation_weights: dict[int, list[float]] = field(default_factory=dict)


def build_probability_registry(model: BitwiseModelSkeleton) -> ProbabilityRegistry:
    """Build a ProbabilityRegistry from the current bitwise model state."""
    reg = ProbabilityRegistry()
    action_id_offset = int(getattr(model, "action_id_offset", 0))
    for i, branches in enumerate(model.action_effect_distributions):
        if len(branches) > 1:
            reg.action_weights[i + action_id_offset] = [b.probability for b in branches]
    for i, branches in enumerate(model.observation_rule_distributions):
        if len(branches) > 1:
            reg.observation_weights[i] = [b.probability for b in branches]
    return reg


def _build_semantic_model_from_parsed(parsed_domain, parsed_problem, parsed_default_policy):
    """Replicate the runtime_parser model-build pipeline for re-parsing."""
    from .semantic_metadata_model import RuntimeSemanticMetadataModel

    plan = build_codegen_plan_from_parsed(
        parsed_domain,
        parsed_problem,
        parsed_default_policy,
    )
    return RuntimeSemanticMetadataModel(
        predicates=list(plan.predicates),
        observables=list(plan.observables),
        actions=[case.action for case in plan.grounded_actions],
        observation_rules=[case.rule for case in plan.grounded_observation_rules],
        default_policy_rules=[case.rule for case in plan.grounded_default_policy_rules],
        maximize_reward=plan.maximize_reward,
        goal_reward=plan.goal_reward,
    )


def extract_updated_probabilities(
    domain_text: str,
    problem_text: str,
    default_policy_text: str | None,
    reference_model: BitwiseModelSkeleton,
) -> ProbabilityRegistry:
    """Re-parse domain/problem and extract updated probability weights."""
    parsed_domain = parse_domain(domain_text)
    parsed_problem = parse_problem(problem_text)
    parsed_dp = (
        parse_default_policy(default_policy_text) if default_policy_text else None
    )
    semantic_model = _build_semantic_model_from_parsed(
        parsed_domain, parsed_problem, parsed_dp,
    )
    new_bitwise = build_bitwise_model_from_parsed(
        semantic_model,
        parsed_domain=parsed_domain,
        parsed_problem=parsed_problem,
        parsed_default_policy=parsed_dp,
    )
    registry = build_probability_registry(new_bitwise)
    action_id_offset = int(getattr(reference_model, "action_id_offset", 0))
    if action_id_offset <= 0:
        return registry
    registry.action_weights = {
        action_id + action_id_offset: list(weights)
        for action_id, weights in registry.action_weights.items()
    }
    return registry


def update_bitwise_model_probabilities(
    skeleton: BitwiseModelSkeleton,
    registry: ProbabilityRegistry,
) -> None:
    """Mutate *skeleton* in-place to use the weights from *registry*.

    BitwiseEffectBranch and BitwiseObservationBranch are frozen dataclasses,
    so we replace the branch lists wholesale.
    """
    action_id_offset = int(getattr(skeleton, "action_id_offset", 0))
    for action_id, weights in registry.action_weights.items():
        internal_action_id = int(action_id) - action_id_offset
        if internal_action_id < 0 or internal_action_id >= len(skeleton.action_effect_distributions):
            continue
        old_branches = skeleton.action_effect_distributions[internal_action_id]
        if len(weights) != len(old_branches):
            continue
        new_branches = [
            dc_replace(b, probability=w)
            for b, w in zip(old_branches, weights)
        ]
        skeleton.action_effect_distributions[internal_action_id] = new_branches

    for rule_id, weights in registry.observation_weights.items():
        if rule_id >= len(skeleton.observation_rule_distributions):
            continue
        old_branches = skeleton.observation_rule_distributions[rule_id]
        if len(weights) != len(old_branches):
            continue
        new_branches = [
            dc_replace(b, probability=w)
            for b, w in zip(old_branches, weights)
        ]
        skeleton.observation_rule_distributions[rule_id] = new_branches
