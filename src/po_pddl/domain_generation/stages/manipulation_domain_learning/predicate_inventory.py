from __future__ import annotations

import json
from dataclasses import dataclass

from po_pddl.config import DEFAULT_MODEL
from po_pddl.core.conventions import CONTAINMENT_PARAMETER_TYPES, CONTAINMENT_PREDICATE
from po_pddl.domain_generation.infrastructure.payload_utils import validate_snake_case

from .models import (
    ActionTaxonomyRecord,
    PredicateInventoryResult,
    PredicateSchema,
    RawTrajectoryStep,
)
from .shared import extract_json_object, load_prompt, make_client, safe_chat

_CONTAINABLE_ACTION_HINTS = (
    "open",
    "close",
    "look_into",
    "look_into_object",
    "look in",
    "look into",
    "inspect",
    "drawer",
    "cabinet",
    "lid",
    "door",
)


@dataclass
class LLMPredicateInventoryModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1800
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("predicate_inventory_prompt.md")

    def generate_predicate_inventory(
        self,
        *,
        steps: list[RawTrajectoryStep],
        taxonomy_records: list[ActionTaxonomyRecord],
        action_templates: list[dict[str, object]] | None = None,
        available_types: list[str] | None = None,
    ) -> PredicateInventoryResult:
        instructions = list(
            dict.fromkeys(str(step.instruction).strip() for step in steps if str(step.instruction).strip())
        )
        payload = {
            "instruction": instructions[0] if instructions else "",
            "instructions": instructions,
            "taxonomy_records": [record.to_dict() for record in taxonomy_records],
            "action_texts": [
                {
                    "episode_name": step.episode_name,
                    "step_index": step.step_index,
                    "action_text": step.action_text,
                    "extra_info": step.extra_info,
                    "observation_text": step.observation_text,
                }
                for step in steps
                if step.action_text
            ],
        }
        if action_templates:
            payload["action_templates"] = action_templates
        if available_types:
            payload["available_types"] = [str(type_name) for type_name in available_types if str(type_name).strip()]
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
        rows = data.get("predicate_inventory", [])
        if not isinstance(rows, list) or not rows:
            raise ValueError("Predicate inventory response must contain a non-empty predicate_inventory list")
        inventory: list[PredicateSchema] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid predicate inventory row: {row!r}")
            name = validate_snake_case(str(row.get("predicate_name") or "").strip(), field_name="predicate_name")
            if name in seen:
                raise ValueError(f"Duplicate predicate_name in predicate inventory: {name}")
            seen.add(name)
            parameter_types_raw = row.get("parameter_types", [])
            if not isinstance(parameter_types_raw, list):
                raise ValueError(f"predicate inventory parameter_types must be a list for {name}")
            parameter_types = [
                validate_snake_case(str(item).strip(), field_name="predicate_parameter_type")
                for item in parameter_types_raw
                if str(item).strip()
            ]
            comment = str(row.get("comment") or "").strip() or None
            raw_predicate_kind = row.get("predicate_kind")
            predicate_kind = str(raw_predicate_kind or "").strip().lower()
            is_static_feature = row.get("is_static_feature", False)
            if raw_predicate_kind is not None and predicate_kind not in {"state", "feature"}:
                raise ValueError(f"predicate inventory predicate_kind must be either 'state' or 'feature' for {name}")
            if not isinstance(is_static_feature, bool):
                raise ValueError(f"predicate inventory is_static_feature must be a boolean for {name}")
            if not predicate_kind:
                predicate_kind = "feature" if is_static_feature else "state"
            if predicate_kind == "state" and is_static_feature:
                raise ValueError(
                    f"predicate inventory row for {name} marked is_static_feature=true but predicate_kind=state"
                )
            inventory.append(
                PredicateSchema(
                    predicate_name=name,
                    parameter_types=parameter_types,
                    comment=comment,
                    predicate_kind=predicate_kind,
                    is_static_feature=(predicate_kind == "feature"),
                )
            )

        available_type_names = {
            validate_snake_case(str(item).strip(), field_name="available_type")
            for item in (available_types or [])
            if str(item).strip()
        }

        uses_movable_item_type = _parse_optional_boolean(data, "uses_movable_item_type")
        movable_item_member_types = _parse_special_type_members(
            data,
            key="movable_item_member_types",
            available_type_names=available_type_names,
        )
        uses_fixed_item_type = _parse_optional_boolean(data, "uses_fixed_item_type")
        fixed_item_member_types = _parse_special_type_members(
            data,
            key="fixed_item_member_types",
            available_type_names=available_type_names,
        )
        uses_containable_item_type = _parse_optional_boolean(data, "uses_containable_item_type")
        containable_item_member_types = _parse_special_type_members(
            data,
            key="containable_item_member_types",
            available_type_names=available_type_names,
        )
        if uses_containable_item_type and not containable_item_member_types:
            containable_item_member_types = _infer_containable_item_member_types(
                taxonomy_records=taxonomy_records,
                action_templates=action_templates or [],
                available_type_names=available_type_names,
            )

        if uses_movable_item_type and not movable_item_member_types:
            raise ValueError(
                "Predicate inventory response enabled movable_item but provided no movable_item_member_types"
            )
        if not uses_movable_item_type and movable_item_member_types:
            raise ValueError(
                "Predicate inventory response provided movable_item_member_types while uses_movable_item_type is false"
            )
        if uses_fixed_item_type and not fixed_item_member_types:
            raise ValueError("Predicate inventory response enabled fixed_item but provided no fixed_item_member_types")
        if not uses_fixed_item_type and fixed_item_member_types:
            raise ValueError(
                "Predicate inventory response provided fixed_item_member_types while uses_fixed_item_type is false"
            )
        if uses_containable_item_type and not containable_item_member_types:
            raise ValueError(
                "Predicate inventory response enabled containable_item but provided no containable_item_member_types"
            )
        if not uses_containable_item_type and containable_item_member_types:
            raise ValueError(
                "Predicate inventory response provided containable_item_member_types while uses_containable_item_type is false"
            )

        overlap = sorted(set(movable_item_member_types) & set(fixed_item_member_types))
        if overlap:
            raise ValueError(
                "Predicate inventory response assigned the same concrete type to both movable_item and fixed_item: "
                f"{overlap}"
            )
        if uses_containable_item_type:
            in_predicate = next(
                (item for item in inventory if item.predicate_name == CONTAINMENT_PREDICATE),
                None,
            )
            if in_predicate is None:
                raise ValueError(
                    "Predicate inventory response enabled containable_item but did not include the required `in` predicate."
                )
            if tuple(in_predicate.parameter_types) != CONTAINMENT_PARAMETER_TYPES:
                raise ValueError(
                    "When containable_item is enabled, predicate `in` must have parameter_types "
                    "['movable_item', 'containable_item']."
                )
            _validate_no_type_specific_containment_aliases(
                inventory=inventory,
                movable_item_member_types=movable_item_member_types,
                containable_item_member_types=containable_item_member_types,
            )
        gripper_empty_predicate = next((item for item in inventory if item.predicate_name == "gripper_empty"), None)
        if gripper_empty_predicate is None:
            raise ValueError("Predicate inventory response must include the required predicate `gripper_empty`.")
        if list(gripper_empty_predicate.parameter_types) != []:
            raise ValueError("Predicate `gripper_empty` must have zero parameters.")
        gripper_holding_predicate = next((item for item in inventory if item.predicate_name == "gripper_holding"), None)
        if gripper_holding_predicate is None:
            raise ValueError("Predicate inventory response must include the required predicate `gripper_holding`.")
        if list(gripper_holding_predicate.parameter_types) != ["movable_item"]:
            raise ValueError("Predicate `gripper_holding` must have parameter_types ['movable_item'].")
        if not uses_movable_item_type:
            raise ValueError(
                "Predicate inventory response must enable movable_item when using the required predicate "
                "`gripper_holding(movable_item)`."
            )

        return PredicateInventoryResult(
            predicate_inventory=inventory,
            uses_movable_item_type=uses_movable_item_type,
            movable_item_member_types=movable_item_member_types,
            uses_fixed_item_type=uses_fixed_item_type,
            fixed_item_member_types=fixed_item_member_types,
            uses_containable_item_type=uses_containable_item_type,
            containable_item_member_types=containable_item_member_types,
        )


def _parse_optional_boolean(data: dict[str, object], key: str) -> bool:
    value = data.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"Predicate inventory field {key} must be a boolean.")
    return value


def _parse_special_type_members(
    data: dict[str, object],
    *,
    key: str,
    available_type_names: set[str],
) -> list[str]:
    raw = data.get(key, [])
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError(f"Predicate inventory field {key} must be a list.")
    members = sorted(
        {
            validate_snake_case(str(item).strip(), field_name=key[:-1] if key.endswith("s") else key)
            for item in raw
            if str(item).strip()
        }
    )
    if available_type_names:
        unknown = [item for item in members if item not in available_type_names]
        if unknown:
            raise ValueError(
                f"Predicate inventory field {key} referenced types not present in available_types: {unknown}"
            )
    return members


def _infer_containable_item_member_types(
    *,
    taxonomy_records: list[ActionTaxonomyRecord],
    action_templates: list[dict[str, object]],
    available_type_names: set[str],
) -> list[str]:
    if not available_type_names:
        return []

    inferred: set[str] = set()

    for record in taxonomy_records:
        if not _contains_any_hint(
            (
                record.canonical_action_name,
                record.proposed_action_name,
                record.raw_action_text,
                record.template_text,
            ),
            _CONTAINABLE_ACTION_HINTS,
        ):
            continue
        for type_name in record.action_argument_types:
            normalized = validate_snake_case(str(type_name).strip(), field_name="containable_item_member_type")
            if normalized in available_type_names:
                inferred.add(normalized)

    if inferred:
        return sorted(inferred)

    for template in action_templates:
        if not isinstance(template, dict):
            continue
        if not _contains_any_hint(
            (
                template.get("canonical_action_name"),
                template.get("template_text"),
                template.get("template_id"),
            ),
            _CONTAINABLE_ACTION_HINTS,
        ):
            continue
        parameter_roles = template.get("parameter_roles", [])
        if isinstance(parameter_roles, list):
            for role in parameter_roles:
                normalized = str(role or "").strip()
                if not normalized:
                    continue
                normalized = validate_snake_case(normalized, field_name="containable_item_member_type")
                if normalized in available_type_names:
                    inferred.add(normalized)

    return sorted(inferred)


def _validate_no_type_specific_containment_aliases(
    *,
    inventory: list[PredicateSchema],
    movable_item_member_types: list[str],
    containable_item_member_types: list[str],
) -> None:
    def _looks_like_containment_alias(predicate_name: str) -> bool:
        lowered = predicate_name.strip().lower()
        if lowered == "in":
            return False
        containment_markers = (
            "inside",
            "contained",
            "contains",
            "within",
        )
        if any(marker in lowered for marker in containment_markers):
            return True
        return lowered.startswith("in_") and "front" not in lowered

    movable_types = set(movable_item_member_types)
    containable_types = set(containable_item_member_types)
    invalid_aliases: list[str] = []
    for predicate in inventory:
        if predicate.predicate_name == "in":
            continue
        if not _looks_like_containment_alias(predicate.predicate_name):
            continue
        parameter_types = list(predicate.parameter_types)
        if len(parameter_types) != 2:
            continue
        first_type, second_type = parameter_types
        if first_type in containable_types and second_type in movable_types:
            invalid_aliases.append(f"{predicate.predicate_name}({first_type},{second_type})")
            continue
        if first_type in movable_types and second_type in containable_types:
            invalid_aliases.append(f"{predicate.predicate_name}({first_type},{second_type})")
    if invalid_aliases:
        raise ValueError(
            "Containment relations must use the single predicate `in(movable_item, containable_item)`; "
            "found type-specific containment aliases: "
            f"{sorted(invalid_aliases)}"
        )


def _contains_any_hint(values: tuple[object, ...], hints: tuple[str, ...]) -> bool:
    haystack = " ".join(str(value or "").strip().lower() for value in values)
    return any(hint in haystack for hint in hints)


__all__ = ["LLMPredicateInventoryModule"]
