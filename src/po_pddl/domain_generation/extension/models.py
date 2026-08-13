from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExtensionReviewResult:
    should_extend: bool
    review_summary: str
    reusable_aliases: dict[str, str]
    approved_new_schema_names: list[str]
    supporting_episode_names_by_new_schema: dict[str, list[str]]
    raw_llm_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "should_extend": self.should_extend,
            "review_summary": self.review_summary,
            "reusable_aliases": dict(self.reusable_aliases),
            "approved_new_schema_names": list(self.approved_new_schema_names),
            "supporting_episode_names_by_new_schema": {
                key: list(value) for key, value in self.supporting_episode_names_by_new_schema.items()
            },
        }


@dataclass(frozen=True)
class BundleUpdateResult:
    scene_description_dir: str
    manipulation_learning_dir: str
    action_review_dir: str
    manipulation_merge_dir: str
    problem_grounding_root: str
    passive_observation_learning_dir: str | None
    init_observation_learning_dir: str | None
    active_observation_learning_dir: str | None
    merged_domain_file: str
    final_bundle_dir: str
    updated_episode_count: int
    action_schema_extended: bool
    observation_updated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
