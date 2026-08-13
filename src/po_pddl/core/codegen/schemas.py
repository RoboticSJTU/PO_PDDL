"""Schemas for explicit grounded Python model generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import (
    Action,
    DefaultPolicyRule,
    FactorizedBelief,
    Observable,
    ObservationRule,
    Predicate,
    TypeNode,
)
from ..parser.schemas import (
    ParsedActionSchema,
    ParsedDefaultPolicyRuleSchema,
    ParsedObservationRuleSchema,
)
from ..parser.sexpr import SExpr


@dataclass(frozen=True)
class GroundedActionCase:
    """One grounded action together with its source schema and fixed bindings."""

    action: Action
    schema: ParsedActionSchema
    bindings: dict[str, str]


@dataclass(frozen=True)
class GroundedObservationRuleCase:
    """One grounded observation rule together with its source schema and bindings."""

    rule: ObservationRule
    schema: ParsedObservationRuleSchema
    bindings: dict[str, str]


@dataclass(frozen=True)
class GroundedDefaultPolicyRuleCase:
    """One grounded default-policy rule together with its source schema and bindings."""

    rule: DefaultPolicyRule
    schema: ParsedDefaultPolicyRuleSchema
    bindings: dict[str, str]


@dataclass
class PythonModelCodegenPlan:
    """Grounded plan for emitting an explicit Python model class.

    This object is intentionally codegen-oriented instead of runtime-oriented.
    It collects everything needed to write clear, explicit, grounded Python code:
    - all grounded predicates / observables
    - all grounded actions / observation rules / default policy rules
    - fixed bindings for every grounded case
    - goal / reward / belief metadata
    """

    domain_name: str
    problem_name: str | None = None
    types: dict[str, TypeNode] = field(default_factory=dict)
    constants: dict[str, str] = field(default_factory=dict)
    objects: dict[str, str] = field(default_factory=dict)
    predicates: list[Predicate] = field(default_factory=list)
    observables: list[Observable] = field(default_factory=list)
    grounded_actions: list[GroundedActionCase] = field(default_factory=list)
    grounded_observation_rules: list[GroundedObservationRuleCase] = field(default_factory=list)
    grounded_default_policy_rules: list[GroundedDefaultPolicyRuleCase] = field(default_factory=list)
    init_belief: FactorizedBelief = field(default_factory=FactorizedBelief)
    goal_expr: SExpr | None = None
    maximize_reward: bool = True
    goal_reward: float = 0.0

    def summary(self) -> dict[str, int]:
        """Return a compact grounded-size summary."""
        return {
            "predicates": len(self.predicates),
            "observables": len(self.observables),
            "actions": len(self.grounded_actions),
            "observation_rules": len(self.grounded_observation_rules),
            "default_policy_rules": len(self.grounded_default_policy_rules),
        }
