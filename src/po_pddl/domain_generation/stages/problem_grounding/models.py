from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EpisodeStep:
    step_index: int
    start_time_sec: float | None
    end_time_sec: float | None
    action_text: str | None
    observation_text: str | None
    extra_info: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeContext:
    episode_name: str
    instruction: str
    step0_observation_text: str | None
    steps: list[EpisodeStep]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_name": self.episode_name,
            "instruction": self.instruction,
            "step0_observation_text": self.step0_observation_text,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class ObjectDeclaration:
    name: str
    type_name: str = "object"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoalFact:
    fact: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProblemSpec:
    problem_name: str
    domain_name: str
    objects: list[ObjectDeclaration]
    init_facts: list[str]
    goal_facts: list[str]
    canonical_object_map: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_name": self.problem_name,
            "domain_name": self.domain_name,
            "objects": [item.to_dict() for item in self.objects],
            "init_facts": list(self.init_facts),
            "goal_facts": list(self.goal_facts),
            "canonical_object_map": dict(self.canonical_object_map),
        }


@dataclass(frozen=True)
class ActionTaxonomyArtifact:
    episode_name: str
    step_index: int
    canonical_action_name: str
    action_category: str
    action_arguments: list[str]
    raw_action_text: str
    observation_text: str | None
    extra_info: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManipulationRecordArtifact:
    episode_name: str
    step_index: int
    canonical_action_name: str
    action_arguments: list[str]
    effect_bucket: str
    delta_add: list[str]
    delta_del: list[str]
    success: bool
    raw_action_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionSchemaArtifact:
    canonical_action_name: str
    action_category: str
    parameter_count: int
    parameter_roles: list[str]
    precondition_literals: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DomainLearningArtifacts:
    action_schemas: list[ActionSchemaArtifact]
    taxonomy_records: list[ActionTaxonomyArtifact]
    manipulation_records: list[ManipulationRecordArtifact]
    episode_object_names: list[str] = field(default_factory=list)
    action_templates: list[dict[str, Any]] = field(default_factory=list)
    action_name_map: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_schemas": [item.to_dict() for item in self.action_schemas],
            "taxonomy_records": [item.to_dict() for item in self.taxonomy_records],
            "manipulation_records": [item.to_dict() for item in self.manipulation_records],
            "episode_object_names": list(self.episode_object_names),
            "action_templates": list(self.action_templates),
            "action_name_map": dict(self.action_name_map),
        }


@dataclass(frozen=True)
class GroundedTrajectoryStep:
    episode_name: str
    step_index: int
    raw_action_text: str | None
    action_category: str | None
    canonical_action_name: str | None
    ground_arguments: list[str]
    ground_action_pddl: str | None
    effect_bucket: str | None
    delta_add: list[str]
    delta_del: list[str]
    success: bool | None
    observation_text: str | None
    extra_info: str | None
    requested_effect_bucket: str | None = None
    branch_expectation: str | None = None
    missing_effect_branch: bool = False
    available_effect_buckets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationIssue:
    step_index: int
    error_code: str
    message: str
    action_name: str | None = None
    requested_effect_bucket: str | None = None
    expected_branch: str | None = None
    available_effect_buckets: list[str] = field(default_factory=list)
    failed_preconditions: list[str] = field(default_factory=list)
    state_before: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationStepReport:
    step_index: int
    action_name: str | None
    effect_bucket: str | None
    status: str
    state_before: list[str]
    state_after: list[str]
    failed_preconditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProblemGroundingResult:
    problem_spec: ProblemSpec
    problem_pddl: str
    grounded_steps: list[GroundedTrajectoryStep]
    validation_steps: list[ValidationStepReport]
    validation_issues: list[ValidationIssue]
    goal_satisfied: bool
    object_init_raw_llm_outputs: dict[str, str] = field(default_factory=dict)
    goal_inference_raw_output: str | None = None

    def validation_summary(self) -> dict[str, Any]:
        return {
            "goal_satisfied": self.goal_satisfied,
            "issue_count": len(self.validation_issues),
            "issues": [issue.to_dict() for issue in self.validation_issues],
            "steps": [step.to_dict() for step in self.validation_steps],
        }
