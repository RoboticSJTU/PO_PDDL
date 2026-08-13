from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from po_pddl.domain_generation.stages.manipulation_domain_learning.models import PredicateSchema
from po_pddl.domain_generation.stages.passive_observation_learning.models import (
    ConfirmedContradiction,
    DescriptionReviewResult,
    SuspectContradiction,
    VLMConfirmationResult,
)


@dataclass(frozen=True)
class InitObservationSourceRecord:
    episode_name: str
    instruction: str
    init_facts: list[str]
    goal_facts: list[str]
    scene_description: str | None
    frame_paths: list[str]
    objects: dict[str, str]
    camera_order_top_to_bottom: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InitObservationExample:
    episode_name: str
    predicate_name: str
    grounded_literal: str
    argument_values: list[str]
    argument_types: list[str]
    ground_truth_value: bool
    observed_value: bool
    init_facts: list[str]
    source_kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InitObservationCondition:
    predicate_name: str
    argument_types: list[str]
    target_literal_template: str
    condition_literals: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InitObservationRuleSchema:
    predicate_name: str
    observable_name: str
    parameter_types: list[str]
    target_literal_template: str
    condition_literals: list[str]
    last_action_false_conditions: list[str]
    true_rule_name: str
    false_rule_name: str
    total_ground_truth_true: int
    observed_true_when_ground_truth_true: int
    observed_false_when_ground_truth_true: int
    prob_observable_true_given_ground_truth_true: float
    prob_observable_false_given_ground_truth_true: float
    total_ground_truth_false: int
    observed_true_when_ground_truth_false: int
    observed_false_when_ground_truth_false: int
    prob_observable_true_given_ground_truth_false: float
    prob_observable_false_given_ground_truth_false: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InitObservationUncertainPredicateDiscovery:
    predicate_names: list[str]
    rationale_by_predicate: dict[str, str] = field(default_factory=dict)
    summary: str | None = None
    raw_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InitObservationLearningResult:
    source_records: list[InitObservationSourceRecord]
    review_results: list[dict[str, Any]]
    seed_examples: list[InitObservationExample]
    conditions: list[InitObservationCondition]
    expanded_examples: list[InitObservationExample]
    schemas: list[InitObservationRuleSchema]
    predicate_inventory: list[PredicateSchema]
    predicate_comments: dict[str, str]
    rendered_module_text: str
    uncertain_predicate_discovery: InitObservationUncertainPredicateDiscovery | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_records": [item.to_dict() for item in self.source_records],
            "review_results": list(self.review_results),
            "seed_examples": [item.to_dict() for item in self.seed_examples],
            "conditions": [item.to_dict() for item in self.conditions],
            "expanded_examples": [item.to_dict() for item in self.expanded_examples],
            "schemas": [item.to_dict() for item in self.schemas],
            "predicate_inventory": [item.to_dict() for item in self.predicate_inventory],
            "predicate_comments": dict(self.predicate_comments),
            "rendered_module_text": self.rendered_module_text,
            "uncertain_predicate_discovery": (
                self.uncertain_predicate_discovery.to_dict() if self.uncertain_predicate_discovery is not None else None
            ),
        }


__all__ = [
    "ConfirmedContradiction",
    "DescriptionReviewResult",
    "InitObservationCondition",
    "InitObservationExample",
    "InitObservationLearningResult",
    "InitObservationRuleSchema",
    "InitObservationSourceRecord",
    "InitObservationUncertainPredicateDiscovery",
    "SuspectContradiction",
    "VLMConfirmationResult",
]
