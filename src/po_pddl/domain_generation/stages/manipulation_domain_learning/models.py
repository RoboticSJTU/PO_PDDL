from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RawTrajectoryStep:
    episode_name: str
    instruction: str
    step_index: int
    start_time_sec: float | None
    end_time_sec: float | None
    action_text: str | None
    observation_text: str | None
    extra_info: str | None
    previous_observation_text: str | None
    previous_known_observation_text: str | None
    frame_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionTaxonomyRecord:
    episode_name: str
    step_index: int
    raw_action_text: str
    proposed_action_name: str
    canonical_action_name: str
    action_category: str
    action_arguments: list[str]
    object_mentions: list[str]
    observation_text: str | None
    extra_info: str | None
    template_text: str | None = None
    parameter_placeholders: list[str] = field(default_factory=list)
    action_argument_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not self.action_argument_types:
            payload.pop("action_argument_types", None)
        return payload


@dataclass(frozen=True)
class EpisodeObjectInventory:
    episode_name: str
    object_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredicateSchema:
    predicate_name: str
    parameter_types: list[str]
    comment: str | None
    predicate_kind: str | None = None
    is_static_feature: bool = False

    def __post_init__(self) -> None:
        kind = str(self.predicate_kind or "").strip().lower()
        if not kind:
            kind = "feature" if self.is_static_feature else "state"
        if kind not in {"state", "feature"}:
            raise ValueError(f"Unsupported predicate_kind: {self.predicate_kind!r}")
        object.__setattr__(self, "predicate_kind", kind)
        object.__setattr__(self, "is_static_feature", kind == "feature")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["predicate_kind"] = str(self.predicate_kind)
        if not self.is_static_feature:
            payload.pop("is_static_feature", None)
        return payload


@dataclass(frozen=True)
class ObjectTypeDefinition:
    type_name: str
    member_object_names: list[str]
    parent_type: str | None = None
    special_supertypes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not self.parent_type:
            payload.pop("parent_type", None)
        if not self.special_supertypes:
            payload.pop("special_supertypes", None)
        return payload


@dataclass(frozen=True)
class PredicateInventoryResult:
    predicate_inventory: list[PredicateSchema]
    uses_movable_item_type: bool = False
    movable_item_member_types: list[str] = field(default_factory=list)
    uses_fixed_item_type: bool = False
    fixed_item_member_types: list[str] = field(default_factory=list)
    uses_containable_item_type: bool = False
    containable_item_member_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManipulationEffectRecord:
    episode_name: str
    step_index: int
    raw_action_text: str
    canonical_action_name: str
    action_arguments: list[str]
    pre_observation_text: str | None
    post_observation_text: str | None
    extra_info: str | None
    delta_add: list[str]
    delta_del: list[str]
    effect_bucket: str
    success: bool
    raw_llm_output: str | None = None
    execution_time_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not self.raw_llm_output:
            payload.pop("raw_llm_output", None)
        if self.execution_time_sec is None:
            payload.pop("execution_time_sec", None)
        return payload


@dataclass(frozen=True)
class ActionSchema:
    canonical_action_name: str
    action_category: str
    parameter_count: int
    parameter_roles: list[str]
    precondition_literals: list[str]
    schema_description: str | None
    effect_branches: list["ActionEffectBranch"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionEffectBranch:
    effect_bucket: str
    probability: float
    success: bool
    delta_add: list[str]
    delta_del: list[str]
    variant_rank: int | None = None
    fixed_delta_add: list[str] = field(default_factory=list)
    fixed_delta_del: list[str] = field(default_factory=list)
    residual_delta_add: list[str] = field(default_factory=list)
    residual_delta_del: list[str] = field(default_factory=list)
    extra_pddl_effect_conjuncts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionEffectStatistic:
    canonical_action_name: str
    effect_bucket: str
    count: int
    probability: float
    delta_add: list[str]
    delta_del: list[str]
    success: bool = False
    variant_rank: int | None = None
    fixed_delta_add: list[str] = field(default_factory=list)
    fixed_delta_del: list[str] = field(default_factory=list)
    residual_delta_add: list[str] = field(default_factory=list)
    residual_delta_del: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
