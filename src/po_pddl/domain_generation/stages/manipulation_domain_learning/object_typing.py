from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from po_pddl.config import DEFAULT_MODEL
from po_pddl.domain_generation.infrastructure.payload_utils import validate_snake_case

from .models import (
    ActionSchema,
    ActionTaxonomyRecord,
    EpisodeObjectInventory,
    ObjectTypeDefinition,
)
from .shared import extract_json_object, load_prompt, make_client, safe_chat


def _most_common_text(values: list[str | None], *, fallback: str) -> str:
    normalized = [str(value).strip() for value in values if str(value or "").strip()]
    if not normalized:
        return fallback
    counts = Counter(normalized)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _default_placeholder_names(argument_count: int) -> list[str]:
    return [f"param_{index + 1}" for index in range(argument_count)]


def _coerce_type_signature(values: list[str]) -> tuple[str, ...]:
    return tuple(
        validate_snake_case(str(value).strip() or "object", field_name="action_argument_type") for value in values
    )


def _type_signature_key(values: list[str]) -> str:
    return "|".join(_coerce_type_signature(values))


def _collect_object_names(taxonomy_records: list[ActionTaxonomyRecord]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for record in taxonomy_records:
        for argument in record.action_arguments:
            name = validate_snake_case(str(argument).strip(), field_name="object_name")
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
    return ordered


def _episode_object_inventories_from_records(
    taxonomy_records: list[ActionTaxonomyRecord],
) -> list[EpisodeObjectInventory]:
    objects_by_episode: dict[str, set[str]] = defaultdict(set)
    for record in taxonomy_records:
        objects_by_episode[record.episode_name].update(record.action_arguments)
    return [
        EpisodeObjectInventory(
            episode_name=episode_name,
            object_names=sorted(object_names),
        )
        for episode_name, object_names in sorted(objects_by_episode.items())
    ]


def _split_action_name(base_name: str, signature: tuple[str, ...]) -> str:
    if not signature:
        return base_name
    suffix = "_".join(signature)
    return validate_snake_case(f"{base_name}_typed_{suffix}", field_name="canonical_action_name")


def _build_object_usage_examples(
    taxonomy_records: list[ActionTaxonomyRecord],
    *,
    max_examples_per_object: int = 3,
) -> list[dict[str, Any]]:
    examples_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in taxonomy_records:
        for index, argument in enumerate(record.action_arguments):
            bucket = examples_by_object[argument]
            if len(bucket) >= max_examples_per_object:
                continue
            bucket.append(
                {
                    "canonical_action_name": record.canonical_action_name,
                    "raw_action_text": record.raw_action_text,
                    "argument_index": index,
                    "co_arguments": [
                        item for position, item in enumerate(record.action_arguments) if position != index
                    ],
                }
            )
    return [
        {
            "object_name": object_name,
            "examples": examples_by_object.get(object_name, []),
        }
        for object_name in sorted(examples_by_object)
    ]


@dataclass(frozen=True)
class ObjectTypingResult:
    object_types: list[ObjectTypeDefinition]
    object_type_map: dict[str, str]
    episode_object_inventories: list[EpisodeObjectInventory]
    taxonomy_records: list[ActionTaxonomyRecord]
    action_templates: list[dict[str, Any]]
    action_name_map: dict[str, Any]
    action_schemas: list[ActionSchema]


@dataclass
class LLMObjectTypingModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1800
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("object_typing_prompt.md")

    def build_typed_artifacts(
        self,
        taxonomy_records: list[ActionTaxonomyRecord],
        *,
        base_action_templates: list[dict[str, Any]] | None = None,
    ) -> ObjectTypingResult:
        object_names = _collect_object_names(taxonomy_records)
        object_types, object_type_map = self._induce_object_types(
            object_names=object_names,
            taxonomy_records=taxonomy_records,
        )
        typed_records = self._apply_types_to_taxonomy_records(
            taxonomy_records=taxonomy_records,
            object_type_map=object_type_map,
        )
        typed_records = self._split_actions_by_type_signature(typed_records)
        episode_object_inventories = _episode_object_inventories_from_records(typed_records)
        action_templates, action_schemas = self._build_typed_action_templates_and_schemas(
            typed_records,
            base_action_templates=base_action_templates,
        )
        action_name_map = _build_action_name_map_from_schemas(
            taxonomy_records=typed_records,
            action_schemas=action_schemas,
            action_templates=action_templates,
            object_type_map=object_type_map,
            object_types=object_types,
        )
        return ObjectTypingResult(
            object_types=object_types,
            object_type_map=object_type_map,
            episode_object_inventories=episode_object_inventories,
            taxonomy_records=typed_records,
            action_templates=action_templates,
            action_name_map=action_name_map,
            action_schemas=action_schemas,
        )

    def _induce_object_types(
        self,
        *,
        object_names: list[str],
        taxonomy_records: list[ActionTaxonomyRecord],
    ) -> tuple[list[ObjectTypeDefinition], dict[str, str]]:
        if not object_names:
            return [], {}
        payload = {
            "object_names": object_names,
            "object_usage_examples": _build_object_usage_examples(taxonomy_records),
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
        rows = data.get("object_types", [])
        if not isinstance(rows, list) or not rows:
            raise ValueError("Object typing response must contain a non-empty object_types list")
        allowed_names = set(object_names)
        assignments: dict[str, str] = {}
        object_types: list[ObjectTypeDefinition] = []
        seen_type_names: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid object type payload: {row!r}")
            type_name = validate_snake_case(str(row.get("type_name") or "").strip(), field_name="type_name")
            if type_name in seen_type_names:
                raise ValueError(f"Duplicate type_name in object typing response: {type_name}")
            member_names_raw = row.get("member_object_names", [])
            if not isinstance(member_names_raw, list) or not member_names_raw:
                raise ValueError(f"member_object_names must be a non-empty list for type {type_name}")
            member_names = [
                validate_snake_case(str(item).strip(), field_name="member_object_name")
                for item in member_names_raw
                if str(item).strip()
            ]
            invalid_names = [item for item in member_names if item not in allowed_names]
            if invalid_names:
                raise ValueError(
                    f"Object typing response assigned unknown object names to type {type_name}: {invalid_names}"
                )
            for member_name in member_names:
                existing = assignments.get(member_name)
                if existing is not None and existing != type_name:
                    raise ValueError(
                        f"Object name {member_name!r} was assigned to multiple types: {existing!r} and {type_name!r}"
                    )
                assignments[member_name] = type_name
            object_types.append(
                ObjectTypeDefinition(
                    type_name=type_name,
                    member_object_names=sorted(set(member_names)),
                    parent_type=None,
                )
            )
            seen_type_names.add(type_name)
        missing = [name for name in object_names if name not in assignments]
        if missing:
            raise ValueError(f"Object typing response did not classify all object names. Missing: {missing}")
        object_types.sort(key=lambda item: item.type_name)
        return object_types, assignments

    @staticmethod
    def _apply_types_to_taxonomy_records(
        *,
        taxonomy_records: list[ActionTaxonomyRecord],
        object_type_map: dict[str, str],
    ) -> list[ActionTaxonomyRecord]:
        typed_records: list[ActionTaxonomyRecord] = []
        for record in taxonomy_records:
            action_argument_types = [object_type_map.get(argument, "object") for argument in record.action_arguments]
            typed_records.append(
                ActionTaxonomyRecord(
                    episode_name=record.episode_name,
                    step_index=record.step_index,
                    raw_action_text=record.raw_action_text,
                    proposed_action_name=record.proposed_action_name,
                    canonical_action_name=record.canonical_action_name,
                    action_category=record.action_category,
                    action_arguments=list(record.action_arguments),
                    object_mentions=list(record.object_mentions),
                    observation_text=record.observation_text,
                    extra_info=record.extra_info,
                    template_text=record.template_text,
                    parameter_placeholders=list(record.parameter_placeholders),
                    action_argument_types=action_argument_types,
                )
            )
        return typed_records

    @staticmethod
    def _split_actions_by_type_signature(
        taxonomy_records: list[ActionTaxonomyRecord],
    ) -> list[ActionTaxonomyRecord]:
        signatures_by_action: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for record in taxonomy_records:
            signatures_by_action[record.canonical_action_name].add(_coerce_type_signature(record.action_argument_types))

        typed_records: list[ActionTaxonomyRecord] = []
        for record in taxonomy_records:
            signature = _coerce_type_signature(record.action_argument_types)
            canonical_action_name = record.canonical_action_name
            if len(signatures_by_action[record.canonical_action_name]) > 1:
                canonical_action_name = _split_action_name(record.canonical_action_name, signature)
            typed_records.append(
                ActionTaxonomyRecord(
                    episode_name=record.episode_name,
                    step_index=record.step_index,
                    raw_action_text=record.raw_action_text,
                    proposed_action_name=record.canonical_action_name,
                    canonical_action_name=canonical_action_name,
                    action_category=record.action_category,
                    action_arguments=list(record.action_arguments),
                    object_mentions=list(record.object_mentions),
                    observation_text=record.observation_text,
                    extra_info=record.extra_info,
                    template_text=record.template_text,
                    parameter_placeholders=list(record.parameter_placeholders),
                    action_argument_types=list(signature),
                )
            )
        return typed_records

    @staticmethod
    def _build_typed_action_templates_and_schemas(
        taxonomy_records: list[ActionTaxonomyRecord],
        *,
        base_action_templates: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[ActionSchema]]:
        records_by_action: dict[str, list[ActionTaxonomyRecord]] = defaultdict(list)
        for record in taxonomy_records:
            records_by_action[record.canonical_action_name].append(record)
        base_templates_by_action = {
            validate_snake_case(
                str(item.get("canonical_action_name") or "").strip(), field_name="canonical_action_name"
            ): item
            for item in (base_action_templates or [])
            if isinstance(item, dict) and str(item.get("canonical_action_name") or "").strip()
        }

        action_templates: list[dict[str, Any]] = []
        action_schemas: list[ActionSchema] = []
        for action_name in sorted(records_by_action):
            records = records_by_action[action_name]
            exemplar = records[0]
            base_template = base_templates_by_action.get(exemplar.proposed_action_name) or base_templates_by_action.get(
                exemplar.canonical_action_name
            )
            parameter_roles = list(exemplar.action_argument_types) or ["object"] * len(exemplar.action_arguments)
            placeholder_candidates = [
                tuple(record.parameter_placeholders) for record in records if record.parameter_placeholders
            ]
            if placeholder_candidates:
                chosen_placeholders = list(
                    sorted(
                        Counter(placeholder_candidates).items(),
                        key=lambda item: (-item[1], item[0]),
                    )[0][0]
                )
            elif base_template and isinstance(base_template.get("parameter_placeholders"), list):
                chosen_placeholders = [
                    str(item).strip() for item in base_template.get("parameter_placeholders", []) if str(item).strip()
                ]
            else:
                chosen_placeholders = _default_placeholder_names(len(parameter_roles))
            template_text = _most_common_text(
                [
                    record.template_text or _templated_action_text(record.raw_action_text, record.action_arguments)
                    for record in records
                ],
                fallback=(str(base_template.get("template_text") or "").strip() or action_name.replace("_", " "))
                if base_template
                else action_name.replace("_", " "),
            )
            action_category = exemplar.action_category
            if action_name == exemplar.proposed_action_name and base_template:
                template_id = str(base_template.get("template_id") or action_name).strip() or action_name
            else:
                template_id = action_name
            action_templates.append(
                {
                    "template_id": template_id,
                    "template_text": template_text,
                    "canonical_action_name": action_name,
                    "action_category": action_category,
                    "parameter_roles": list(parameter_roles),
                    "parameter_placeholders": list(chosen_placeholders),
                    "success_effect_bucket": f"{action_name}_success",
                    "failure_effect_bucket": f"{action_name}_failure",
                }
            )
            action_schemas.append(
                ActionSchema(
                    canonical_action_name=action_name,
                    action_category=action_category,
                    parameter_count=len(parameter_roles),
                    parameter_roles=list(parameter_roles),
                    precondition_literals=[],
                    schema_description=template_text,
                )
            )
        return action_templates, action_schemas


_WHITESPACE_RE = re.compile(r"\s+")


def _default_placeholder_name(index: int) -> str:
    return "object" if index == 0 else f"object_{index + 1}"


def _surface_from_argument(argument: str) -> str:
    return argument.replace("_", " ")


def _templated_action_text(
    raw_action_text: str | None,
    action_arguments: list[str],
) -> str | None:
    text = str(raw_action_text).strip() if raw_action_text is not None else None
    if text == "":
        text = None
    if text is None:
        return None
    templated = text
    for index, argument in sorted(
        enumerate(action_arguments),
        key=lambda item: len(_surface_from_argument(item[1])),
        reverse=True,
    ):
        surface = _surface_from_argument(argument).strip()
        if not surface:
            continue
        placeholder = "{" + _default_placeholder_name(index) + "}"
        pattern = re.compile(rf"(?i)\b{re.escape(surface)}\b")
        templated, _count = pattern.subn(placeholder, templated, count=1)
    templated = _WHITESPACE_RE.sub(" ", templated).strip()
    return templated or None


def _build_action_name_map_from_schemas(
    *,
    taxonomy_records: list[ActionTaxonomyRecord],
    action_schemas: list[ActionSchema],
    action_templates: list[dict[str, Any]],
    object_type_map: dict[str, str],
    object_types: list[ObjectTypeDefinition],
) -> dict[str, Any]:
    records_by_action: dict[str, list[ActionTaxonomyRecord]] = defaultdict(list)
    for record in taxonomy_records:
        if record.canonical_action_name:
            records_by_action[record.canonical_action_name].append(record)

    schema_by_action = {schema.canonical_action_name: schema for schema in action_schemas}
    template_by_action = {
        validate_snake_case(
            str(item.get("canonical_action_name") or "").strip(), field_name="canonical_action_name"
        ): item
        for item in action_templates
        if isinstance(item, dict) and str(item.get("canonical_action_name") or "").strip()
    }
    actions_by_name: dict[str, dict[str, Any]] = {}
    template_id_to_action: dict[str, str] = {}
    template_text_to_action: dict[str, str] = {}
    template_text_to_type_signature_to_action: dict[str, dict[str, str]] = {}

    for action_name, records in sorted(records_by_action.items()):
        schema = schema_by_action[action_name]
        template_info = template_by_action.get(action_name, {})
        placeholder_candidates = [
            tuple(record.parameter_placeholders) for record in records if record.parameter_placeholders
        ]
        if placeholder_candidates:
            parameter_placeholders = list(
                sorted(
                    Counter(placeholder_candidates).items(),
                    key=lambda item: (-item[1], item[0]),
                )[0][0]
            )
        else:
            parameter_placeholders = [_default_placeholder_name(index) for index in range(schema.parameter_count)]
        template_text = _most_common_text(
            [
                record.template_text or _templated_action_text(record.raw_action_text, record.action_arguments)
                for record in records
            ],
            fallback=str(template_info.get("template_text") or "").strip() or action_name,
        )
        template_id = str(template_info.get("template_id") or action_name).strip() or action_name
        actions_by_name[action_name] = {
            "action_name": action_name,
            "action_category": schema.action_category,
            "template_id": template_id,
            "template_text": template_text,
            "parameter_roles": list(schema.parameter_roles),
            "parameter_placeholders": parameter_placeholders,
            "effect_buckets": {
                "success": f"{action_name}_success",
                "failure": f"{action_name}_failure",
            },
            "source_action_name": None,
            "ground_truth_positive_literal": None,
        }
        template_id_to_action[template_id] = action_name
        signature_key = _type_signature_key(list(schema.parameter_roles))
        template_text_to_type_signature_to_action.setdefault(template_text, {})
        existing_signature_action = template_text_to_type_signature_to_action[template_text].get(signature_key)
        if existing_signature_action is not None and existing_signature_action != action_name:
            raise ValueError(
                f"Conflicting typed action-map entry for template_text={template_text!r} "
                f"and signature={signature_key!r}: {existing_signature_action!r} vs {action_name!r}"
            )
        template_text_to_type_signature_to_action[template_text][signature_key] = action_name
        existing_action = template_text_to_action.get(template_text)
        if existing_action is None:
            template_text_to_action[template_text] = action_name
        elif existing_action != action_name:
            template_text_to_action.pop(template_text, None)

    return {
        "schema_version": 2,
        "actions_by_name": actions_by_name,
        "typing": {
            "object_name_to_type": dict(sorted(object_type_map.items())),
            "type_to_object_names": {
                item.type_name: list(item.member_object_names)
                for item in sorted(object_types, key=lambda row: row.type_name)
            },
            "type_to_parent_type": {
                item.type_name: item.parent_type
                for item in sorted(object_types, key=lambda row: row.type_name)
                if item.parent_type
            },
        },
        "lookup": {
            "template_id_to_action": template_id_to_action,
            "template_text_to_action": template_text_to_action,
            "template_text_to_type_signature_to_action": {
                template_text: dict(sorted(signature_map.items()))
                for template_text, signature_map in sorted(template_text_to_type_signature_to_action.items())
            },
        },
    }


__all__ = [
    "LLMObjectTypingModule",
    "ObjectTypingResult",
]
