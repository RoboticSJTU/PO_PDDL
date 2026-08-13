from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from po_pddl.domain_generation.stages.manipulation_domain_learning.models import PredicateSchema


@dataclass(frozen=True)
class ActiveObservationSourceRecord:
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
    previous_scene_description: str | None
    current_scene_description: str | None
    current_state: list[str]
    ground_truth_facts: list[str]
    frame_paths: list[str]
    objects: dict[str, str]
    effect_predicate_names: list[str]
    camera_order_top_to_bottom: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscoveredObservation:
    predicate_name: str
    grounded_literal: str
    observed_value: bool
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActiveObservationDiscoveryResult:
    discovered_observations: list[DiscoveredObservation]
    summary: str | None = None
    raw_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["discovered_observations"] = [item.to_dict() for item in self.discovered_observations]
        return payload


@dataclass(frozen=True)
class ConfirmedObservation:
    predicate_name: str
    grounded_literal: str
    ground_truth_value: bool
    observed_value: bool
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VLMObservationConfirmationResult:
    confirmed_observations: list[ConfirmedObservation]
    summary: str | None = None
    raw_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confirmed_observations"] = [item.to_dict() for item in self.confirmed_observations]
        return payload


@dataclass(frozen=True)
class ActiveObservationExample:
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
    current_state: list[str]
    source_kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActiveObservationCondition:
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
class ActiveObservationRuleSchema:
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
class ActiveObservationLearningResult:
    source_records: list[ActiveObservationSourceRecord]
    discovery_results: list[dict[str, Any]]
    seed_examples: list[ActiveObservationExample]
    conditions: list[ActiveObservationCondition]
    confirmed_examples: list[ActiveObservationExample]
    schemas: list[ActiveObservationRuleSchema]
    predicate_inventory: list[PredicateSchema]
    predicate_comments: dict[str, str]
    rendered_module_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_records": [item.to_dict() for item in self.source_records],
            "discovery_results": list(self.discovery_results),
            "seed_examples": [item.to_dict() for item in self.seed_examples],
            "conditions": [item.to_dict() for item in self.conditions],
            "confirmed_examples": [item.to_dict() for item in self.confirmed_examples],
            "schemas": [item.to_dict() for item in self.schemas],
            "predicate_inventory": [item.to_dict() for item in self.predicate_inventory],
            "predicate_comments": dict(self.predicate_comments),
            "rendered_module_text": self.rendered_module_text,
        }
