from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GroundedPreconditionExample:
    episode_name: str
    step_index: int
    action_name: str
    ground_arguments: list[str]
    state_before: list[str]
    candidate_literals: list[str]
    eligible_literals: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreconditionCandidateStat:
    literal: str
    occurrence_count: int
    eligible_example_count: int
    support: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionPreconditionCandidateBundle:
    action_name: str
    example_count: int
    candidate_stats: list[PreconditionCandidateStat]
    examples: list[GroundedPreconditionExample]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "example_count": self.example_count,
            "candidate_stats": [item.to_dict() for item in self.candidate_stats],
            "examples": [item.to_dict() for item in self.examples],
        }


@dataclass(frozen=True)
class LearnedPreconditionSummary:
    zero_arity_predicates: list[str]
    example_counts_by_action: dict[str, int]
    selected_preconditions_by_action: dict[str, list[str]]
    candidate_stats_by_action: dict[str, list[dict[str, Any]]]
    selection_summaries_by_action: dict[str, str] = field(default_factory=dict)
    updated_action_schemas: list[dict[str, Any]] = field(default_factory=list)
    rendered_action_schema_pddl: str = ""
    rendered_domain_pddl: str = ""
    predicate_comments: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "zero_arity_predicates": list(self.zero_arity_predicates),
            "example_counts_by_action": dict(self.example_counts_by_action),
            "selected_preconditions_by_action": {
                key: list(value) for key, value in self.selected_preconditions_by_action.items()
            },
            "candidate_stats_by_action": {key: list(value) for key, value in self.candidate_stats_by_action.items()},
            "selection_summaries_by_action": dict(self.selection_summaries_by_action),
            "updated_action_schemas": list(self.updated_action_schemas),
            "rendered_action_schema_pddl": self.rendered_action_schema_pddl,
            "rendered_domain_pddl": self.rendered_domain_pddl,
            "predicate_comments": dict(self.predicate_comments),
        }
