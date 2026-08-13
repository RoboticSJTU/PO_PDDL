from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from po_pddl.domain_generation.stages.problem_grounding.models import ProblemSpec


@dataclass(frozen=True)
class VisibleObjectCandidate:
    name: str
    type_name: str
    supporting_observation: str | None = None
    visible_facts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatentObjectCandidate:
    name: str
    type_name: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferredInitFact:
    fact: str
    confidence: str = "medium"
    justification: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InitCompletionFact:
    fact: str
    confidence: str = "medium"
    justification: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InitStateRepairPlan:
    should_repair: bool
    init_facts_add: list[str] = field(default_factory=list)
    init_facts_remove: list[str] = field(default_factory=list)
    repair_summary: str | None = None
    raw_llm_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProblemInferenceResult:
    problem_spec: ProblemSpec
    problem_pddl: str
    visible_objects: list[VisibleObjectCandidate]
    latent_objects: list[LatentObjectCandidate]
    inferred_init_facts: list[InferredInitFact]
    completed_init_facts: list[InitCompletionFact]
    diagnostics: dict[str, Any]
    raw_llm_outputs: dict[str, str] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "problem_spec": self.problem_spec.to_dict(),
            "visible_objects": [item.to_dict() for item in self.visible_objects],
            "latent_objects": [item.to_dict() for item in self.latent_objects],
            "inferred_init_facts": [item.to_dict() for item in self.inferred_init_facts],
            "completed_init_facts": [item.to_dict() for item in self.completed_init_facts],
            "diagnostics": self.diagnostics,
            "raw_llm_outputs": dict(self.raw_llm_outputs),
        }
