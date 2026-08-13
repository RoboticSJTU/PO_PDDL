from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.models.factorized_belief import FactorizedBelief
from ..core.models.predicate import Predicate
from ..core.parser.schemas import ParsedDomain
from ..core.parser.sexpr import SExpr
from ..domain_generation.stages.problem_grounding.models import ObjectDeclaration


@dataclass(frozen=True)
class VisibleObject:
    name: str
    type_name: str = "object"
    justification: str | None = None

    def to_object_declaration(self) -> ObjectDeclaration:
        return ObjectDeclaration(name=self.name, type_name=self.type_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type_name": self.type_name,
            "justification": self.justification,
        }


@dataclass(frozen=True)
class PredicateTruthJudgment:
    predicate: Predicate
    truth_value: bool
    justification: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate.to_pddl_str(),
            "truth_value": self.truth_value,
            "justification": self.justification,
        }


@dataclass(frozen=True)
class OnlinePlanningProblemSpec:
    problem_name: str
    domain_name: str
    objects: list[ObjectDeclaration]
    init_state: dict[Predicate, bool]
    init_belief: FactorizedBelief
    goal_expr: SExpr | None
    has_observation_module: bool


@dataclass(frozen=True)
class OnlinePlanningProblemResult:
    spec: OnlinePlanningProblemSpec
    problem_pddl: str
    parsed_domain: ParsedDomain
    visible_objects: list[VisibleObject]
    predicate_judgments: list[PredicateTruthJudgment] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    reused_problem_file: str | None = None
