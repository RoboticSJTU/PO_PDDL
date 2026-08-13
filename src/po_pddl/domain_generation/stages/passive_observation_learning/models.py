from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from po_pddl.domain_generation.stages.manipulation_domain_learning.models import PredicateSchema


@dataclass(frozen=True)
class PassiveObservationSourceRecord:
    episode_name: str
    step_index: int
    instruction: str
    raw_action_text: str
    canonical_action_name: str
    effect_bucket: str
    success: bool
    variant_rank: int
    action_arguments: list[str]
    action_argument_types: list[str]
    state_before: list[str]
    ground_truth_facts: list[str]
    scene_description: str | None
    frame_paths: list[str]
    objects: dict[str, str]
    effect_predicate_names: list[str] = field(default_factory=list)
    camera_order_top_to_bottom: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SuspectContradiction:
    predicate_name: str
    grounded_literal: str
    observed_value: bool
    ground_truth_value: bool
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DescriptionReviewResult:
    should_review_with_vlm: bool
    suspect_contradictions: list[SuspectContradiction]
    summary: str | None = None
    raw_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["suspect_contradictions"] = [item.to_dict() for item in self.suspect_contradictions]
        return payload


@dataclass(frozen=True)
class ConfirmedContradiction:
    predicate_name: str
    grounded_literal: str
    observed_value: bool
    ground_truth_value: bool
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VLMConfirmationResult:
    confirmed_contradictions: list[ConfirmedContradiction]
    summary: str | None = None
    raw_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confirmed_contradictions"] = [item.to_dict() for item in self.confirmed_contradictions]
        return payload


@dataclass(frozen=True)
class ContainableRevealActionSelectionResult:
    action_names: list[str]
    summary: str | None = None
    raw_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisibleContainedObjectsResult:
    visible_object_names: list[str]
    summary: str | None = None
    raw_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PassiveObservationExample:
    episode_name: str
    step_index: int
    canonical_action_name: str
    effect_bucket: str
    success: bool
    variant_rank: int
    predicate_name: str
    grounded_literal: str
    action_arguments: list[str]
    action_argument_types: list[str]
    extra_arguments: list[str]
    extra_argument_types: list[str]
    ground_truth_value: bool
    observed_value: bool
    state_before: list[str]
    source_kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PassiveObservationCondition:
    canonical_action_name: str
    effect_bucket: str
    success: bool
    variant_rank: int
    predicate_name: str
    action_argument_types: list[str]
    extra_argument_types: list[str]
    target_literal_template: str
    condition_literals: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PassiveObservationRuleSchema:
    canonical_action_name: str
    effect_bucket: str
    success: bool
    variant_rank: int
    predicate_name: str
    observable_name: str
    last_action_constant: str
    last_action_predicate_name: str
    action_argument_types: list[str]
    extra_argument_types: list[str]
    last_action_predicate_parameter_types: list[str]
    target_predicate_parameter_types: list[str]
    target_literal_template: str
    condition_literals: list[str]
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
class PassiveObservationLearningResult:
    source_records: list[PassiveObservationSourceRecord]
    review_results_by_variant: dict[str, dict[str, Any]]
    seed_examples: list[PassiveObservationExample]
    conditions: list[PassiveObservationCondition]
    expanded_examples: list[PassiveObservationExample]
    schemas: list[PassiveObservationRuleSchema]
    predicate_inventory: list[PredicateSchema]
    predicate_comments: dict[str, str]
    rendered_module_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_records": [item.to_dict() for item in self.source_records],
            "review_results_by_variant": self.review_results_by_variant,
            "seed_examples": [item.to_dict() for item in self.seed_examples],
            "conditions": [item.to_dict() for item in self.conditions],
            "expanded_examples": [item.to_dict() for item in self.expanded_examples],
            "schemas": [item.to_dict() for item in self.schemas],
            "predicate_inventory": [item.to_dict() for item in self.predicate_inventory],
            "predicate_comments": dict(self.predicate_comments),
            "rendered_module_text": self.rendered_module_text,
        }
