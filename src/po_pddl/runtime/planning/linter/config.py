"""Configuration for the POMDPDDL linter."""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_ALLOWED_OPS_BY_CONTEXT: dict[str, set[str]] = {
    "goal": {"and", "or", "not", "imply", "forall", "all", "exists", "="},
    "action_precondition": {"and", "or", "not", "imply", "forall", "all", "exists", "="},
    "action_effect": {"and", "not", "when", "probabilistic", "forall", "increase", "decrease", "assign"},
    "observation_condition": {"and", "or", "not", "imply", "forall", "all", "exists", "="},
    "observation_distribution": {"and", "not", "probabilistic"},
    "default_policy_precondition": {"and", "or", "not", "imply", "forall", "all", "exists", "="},
}


@dataclass
class LinterConfig:
    """Configurable linter settings."""

    allowed_ops_by_context: dict[str, set[str]] = field(
        default_factory=lambda: {key: set(value) for key, value in DEFAULT_ALLOWED_OPS_BY_CONTEXT.items()}
    )
    enable_type_checks: bool = True
    enable_duplicate_name_checks: bool = True
    enable_belief_checks: bool = True
    treat_warnings_as_errors: bool = False
