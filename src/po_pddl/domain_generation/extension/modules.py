from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol

from po_pddl.config import DEFAULT_MODEL
from po_pddl.domain_generation.stages.manipulation_domain_learning.models import (
    ActionSchema,
    ActionTaxonomyRecord,
    ManipulationEffectRecord,
)

from .models import ExtensionReviewResult
from .shared import extract_json_object, load_prompt, make_client, safe_chat

logger = logging.getLogger(__name__)


class ActionExtensionReviewModule(Protocol):
    def review(
        self,
        *,
        existing_action_schemas: list[ActionSchema],
        new_action_schemas: list[ActionSchema],
        new_taxonomy_records: list[ActionTaxonomyRecord],
        new_effect_records: list[ManipulationEffectRecord],
    ) -> ExtensionReviewResult: ...


def _validate_action_aliases(
    reusable_aliases: dict[str, str],
    *,
    existing_schemas: list[ActionSchema],
    new_schemas: list[ActionSchema],
) -> dict[str, str]:
    existing_by_name = {schema.canonical_action_name: schema for schema in existing_schemas}
    new_by_name = {schema.canonical_action_name: schema for schema in new_schemas}
    broadly_compatible_roles = {"object", "movable_item", "fixed_item", "containable_item"}
    validated: dict[str, str] = {}
    for new_name, existing_name in reusable_aliases.items():
        new_schema = new_by_name.get(new_name)
        existing_schema = existing_by_name.get(existing_name)
        if new_schema is None or existing_schema is None:
            continue
        if new_schema.parameter_count != existing_schema.parameter_count:
            continue
        if any(
            existing_role != new_role and existing_role not in broadly_compatible_roles
            for new_role, existing_role in zip(new_schema.parameter_roles, existing_schema.parameter_roles)
        ):
            logger.warning(
                "Rejected non-groundable action alias %s -> %s due to incompatible parameter roles %s -> %s.",
                new_name,
                existing_name,
                new_schema.parameter_roles,
                existing_schema.parameter_roles,
            )
            continue
        validated[new_name] = existing_name
    return validated


def _validate_supporting_episodes(
    supporting_episode_names_by_new_schema: dict[str, list[str]],
    *,
    approved_new_names: set[str],
    candidate_episode_names: set[str],
) -> dict[str, list[str]]:
    validated: dict[str, list[str]] = {}
    for schema_name, episode_names in supporting_episode_names_by_new_schema.items():
        if schema_name not in approved_new_names:
            continue
        filtered = sorted({name for name in episode_names if name in candidate_episode_names})
        if filtered:
            validated[schema_name] = filtered
    return validated


@dataclass
class LLMActionExtensionReviewModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 3000
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("action_extension_review_prompt.md")

    def review(
        self,
        *,
        existing_action_schemas: list[ActionSchema],
        new_action_schemas: list[ActionSchema],
        new_taxonomy_records: list[ActionTaxonomyRecord],
        new_effect_records: list[ManipulationEffectRecord],
    ) -> ExtensionReviewResult:
        payload = {
            "existing_action_schemas": [schema.to_dict() for schema in existing_action_schemas],
            "new_action_schemas": [schema.to_dict() for schema in new_action_schemas],
            "new_taxonomy_records": [record.to_dict() for record in new_taxonomy_records],
            "new_effect_records": [record.to_dict() for record in new_effect_records],
        }
        reply = safe_chat(
            self._client,
            self._prompt,
            json.dumps(payload, ensure_ascii=False, indent=2),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        existing_names = {schema.canonical_action_name for schema in existing_action_schemas}
        new_names = {schema.canonical_action_name for schema in new_action_schemas}
        approved_new_names = [
            str(name).strip()
            for name in data.get("approved_new_schema_names", [])
            if str(name).strip() in new_names and str(name).strip() not in existing_names
        ]
        reusable_aliases = _validate_action_aliases(
            {str(key).strip(): str(value).strip() for key, value in dict(data.get("reusable_aliases", {})).items()},
            existing_schemas=existing_action_schemas,
            new_schemas=new_action_schemas,
        )
        candidate_episode_names = {record.episode_name for record in new_taxonomy_records}
        supporting_episode_names_by_new_schema = _validate_supporting_episodes(
            {
                str(key).strip(): [str(item).strip() for item in value if str(item).strip()]
                for key, value in dict(data.get("supporting_episode_names_by_new_schema", {})).items()
                if isinstance(value, list)
            },
            approved_new_names=set(approved_new_names),
            candidate_episode_names=candidate_episode_names,
        )
        return ExtensionReviewResult(
            should_extend=bool(data.get("should_extend")) and bool(approved_new_names),
            review_summary=str(data.get("review_summary", "")).strip(),
            reusable_aliases=reusable_aliases,
            approved_new_schema_names=approved_new_names,
            supporting_episode_names_by_new_schema=supporting_episode_names_by_new_schema,
            raw_llm_output=reply,
        )


__all__ = [
    "ActionExtensionReviewModule",
    "LLMActionExtensionReviewModule",
]
