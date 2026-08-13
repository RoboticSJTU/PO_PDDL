"""Lightweight parser result schemas."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models.action import Action
from ..models.aliases import StateEntry
from ..models.default_policy_rule import DefaultPolicyRule
from ..models.effect_bucket import ParsedEffectBucketAnnotation
from ..models.factorized_belief import FactorizedBelief
from ..models.observable import Observable
from ..models.observation_rule import ObservationRule
from ..models.predicate import Predicate
from ..models.type_node import TypeNode
from .sexpr import SExpr


@dataclass
class ParsedActionSchema:
    """Parser-side schema for an action block."""

    action: Action
    parameter_types: list[tuple[str, str]] = field(default_factory=list)
    precondition: SExpr | None = None
    effect: SExpr | None = None
    effect_bucket_annotations: list[ParsedEffectBucketAnnotation] = field(default_factory=list)


@dataclass
class ParsedObservationRuleSchema:
    """Parser-side schema for an observation block."""

    rule: ObservationRule
    parameter_types: list[tuple[str, str]] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    condition: SExpr | None = None
    distribution_expr: SExpr | None = None


@dataclass
class ParsedDefaultPolicyRuleSchema:
    """Parser-side schema for a default policy rule block."""

    rule: DefaultPolicyRule
    parameter_types: list[tuple[str, str]] = field(default_factory=list)
    precondition: SExpr | None = None
    action_expr: SExpr | None = None


@dataclass
class ParsedDomain:
    """Structured result of parsing the supported domain sections."""

    domain_name: str
    types: dict[str, TypeNode] = field(default_factory=dict)
    constants: dict[str, str] = field(default_factory=dict)
    functions: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    predicates: list[Predicate] = field(default_factory=list)
    predicate_parameter_types: dict[Predicate, list[tuple[str, str]]] = field(default_factory=dict)
    observables: list[Observable] = field(default_factory=list)
    observable_parameter_types: dict[Observable, list[tuple[str, str]]] = field(default_factory=dict)
    actions: list[ParsedActionSchema] = field(default_factory=list)
    observation_rules: list[ParsedObservationRuleSchema] = field(default_factory=list)


@dataclass
class ParsedProblem:
    """Structured result of parsing the supported problem sections."""

    problem_name: str
    domain_name: str | None = None
    objects: dict[str, str] = field(default_factory=dict)
    init_state: StateEntry = field(default_factory=dict)
    other_init_items: list[SExpr] = field(default_factory=list)
    goal: SExpr | None = None
    init_belief: FactorizedBelief = field(default_factory=FactorizedBelief)
    maximize_reward: bool = True
    metric_expr: SExpr | None = None
    metric_target_function: str | None = None
    goal_reward: float = 0.0


@dataclass
class ParsedDefaultPolicy:
    """Structured result of parsing a default policy file."""

    policy_name: str
    rules: list[ParsedDefaultPolicyRuleSchema] = field(default_factory=list)
