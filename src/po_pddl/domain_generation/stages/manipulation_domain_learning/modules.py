from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from po_pddl.config import DEFAULT_MODEL
from po_pddl.domain_generation.infrastructure.artifact_io import discover_episode_files, load_episode_payload
from po_pddl.domain_generation.infrastructure.fact_utils import parse_positive_symbolic_fact
from po_pddl.domain_generation.infrastructure.payload_utils import (
    lookup_first as _lookup_first,
)
from po_pddl.domain_generation.infrastructure.payload_utils import (
    normalize_optional_text as _normalize_optional_text,
)
from po_pddl.domain_generation.infrastructure.payload_utils import (
    validate_snake_case as _validate_snake_case,
)

from .models import (
    ActionSchema,
    ActionTaxonomyRecord,
    EpisodeObjectInventory,
    ManipulationEffectRecord,
    PredicateSchema,
    RawTrajectoryStep,
)
from .object_name_normalization import (
    coarsen_object_identifiers,
    coarsen_symbolic_literal_list,
)
from .shared import extract_json_object, load_prompt, make_client, safe_chat

logger = logging.getLogger(__name__)
_FAILURE_MARKERS = ("failure", "failed", "失败")


class ActionTaxonomyModule(Protocol):
    def classify_actions(self, steps: list[RawTrajectoryStep]) -> list[ActionTaxonomyRecord]: ...


class EpisodeObjectInventoryModule(Protocol):
    def build_episode_object_inventories(self, steps: list[RawTrajectoryStep]) -> list[EpisodeObjectInventory]: ...


class ManipulationEffectLearningModule(Protocol):
    def learn_effects(
        self,
        steps: list[RawTrajectoryStep],
        action_schemas: list[ActionSchema],
        taxonomy_records: list[ActionTaxonomyRecord],
        predicate_inventory: list[PredicateSchema] | None = None,
    ) -> list[ManipulationEffectRecord]: ...


class ActionSchemaConsolidationModule(Protocol):
    def consolidate(
        self,
        steps: list[RawTrajectoryStep],
        taxonomy_records: list[ActionTaxonomyRecord],
        predicate_inventory: list[PredicateSchema] | None = None,
    ) -> tuple[list[ActionSchema], list[ActionTaxonomyRecord]]: ...


class ObservationActionLearningModule(Protocol):
    def learn_observation_actions(
        self,
        steps: list[RawTrajectoryStep],
        taxonomy_records: list[ActionTaxonomyRecord],
    ) -> dict[str, object]: ...


def _normalize_snake_case_candidate(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


def _coerce_snake_case(value: object, *, field_name: str) -> str:
    normalized = _normalize_snake_case_candidate(value)
    return _validate_snake_case(normalized, field_name=field_name)


def _coerce_string_list(value: object, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        items = [str(value).strip()] if str(value).strip() else []
    return [_validate_snake_case(item, field_name=field_name) for item in items]


def _coerce_literal_list(value: object, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        items = [str(value).strip()] if str(value).strip() else []
    return items


def _coerce_fact_list(value: object, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _predicate_name_from_literal(text: str) -> str:
    literal = str(text).strip()
    negative_prefix = "not "
    if literal.startswith(negative_prefix):
        literal = literal[len(negative_prefix) :].strip()
    if "(" not in literal:
        return _validate_snake_case(literal, field_name="predicate_name")
    return _validate_snake_case(literal.split("(", 1)[0].strip(), field_name="predicate_name")


def _predicate_arity_map(predicate_inventory: list[PredicateSchema] | None) -> dict[str, int]:
    return {item.predicate_name: len(item.parameter_types) for item in predicate_inventory or []}


def _validate_effect_literals_against_inventory(
    *,
    delta_add: list[str],
    delta_del: list[str],
    allowed_predicates: set[str],
    predicate_arities: dict[str, int],
    immutable_predicates: set[str] | None = None,
    context: str,
) -> None:
    immutable_predicates = immutable_predicates or set()
    for literal in [*delta_add, *delta_del]:
        predicate_name, arguments = parse_positive_symbolic_fact(literal)
        if allowed_predicates and predicate_name not in allowed_predicates:
            raise ValueError(
                f"{context}: predicate `{predicate_name}` is not present in the allowed predicate inventory."
            )
        expected_arity = predicate_arities.get(predicate_name)
        if expected_arity is not None and len(arguments) != expected_arity:
            raise ValueError(
                f"{context}: predicate `{predicate_name}` expects {expected_arity} arguments, "
                f"got {len(arguments)} in literal `{literal}`."
            )
        if predicate_name in immutable_predicates:
            raise ValueError(
                f"{context}: predicate `{predicate_name}` is a static feature and cannot appear in an action effect."
            )


def validate_manipulation_records_against_predicate_inventory(
    *,
    records: list[ManipulationEffectRecord],
    predicate_inventory: list[PredicateSchema] | None,
) -> None:
    allowed_predicates = {item.predicate_name for item in predicate_inventory or []}
    predicate_arities = _predicate_arity_map(predicate_inventory)
    immutable_predicates = {
        item.predicate_name for item in predicate_inventory or [] if item.is_static_feature
    }
    if not allowed_predicates and not predicate_arities:
        return
    for record in records:
        _validate_effect_literals_against_inventory(
            delta_add=record.delta_add,
            delta_del=record.delta_del,
            allowed_predicates=allowed_predicates,
            predicate_arities=predicate_arities,
            immutable_predicates=immutable_predicates,
            context=(
                f"Manipulation effect record [{record.episode_name} step {record.step_index}] "
                f"for action `{record.canonical_action_name}`"
            ),
        )


def _coerce_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "success"}:
            return True
        if normalized in {"false", "no", "failure", "failed"}:
            return False
    raise ValueError(f"Expected boolean-like value for {field_name}, got {value!r}")


def _extract_payload(data: dict[str, object], nested_key: str) -> dict[str, object]:
    nested = data.get(nested_key)
    if isinstance(nested, dict):
        merged = dict(nested)
        for key, value in data.items():
            merged.setdefault(key, value)
        return merged
    return data


def _is_failure(extra_info: str | None) -> bool:
    normalized = (_normalize_optional_text(extra_info) or "").lower()
    return any(marker in normalized for marker in _FAILURE_MARKERS)


def _step_prompt_dict(step: RawTrajectoryStep, *, include_current_observation: bool) -> dict[str, object]:
    data = step.to_dict()
    if not include_current_observation:
        data["observation_text"] = None
    data["previous_observation_text"] = None
    data["previous_known_observation_text"] = None
    return data


def _taxonomy_record_prompt_dict(
    record: ActionTaxonomyRecord,
    *,
    include_observation: bool,
) -> dict[str, object]:
    data = record.to_dict()
    if not include_observation:
        data["observation_text"] = None
    return data


def _normalize_allowed_object_name(name: str, allowed_object_names: set[str], *, field_name: str) -> str:
    normalized = _validate_snake_case(str(name).strip(), field_name=field_name)
    if normalized in allowed_object_names:
        return normalized
    coarse_matches = [
        candidate
        for candidate in allowed_object_names
        if normalized == candidate or normalized.startswith(f"{candidate}_")
    ]
    if len(coarse_matches) == 1:
        return coarse_matches[0]
    allowed_preview = sorted(allowed_object_names)
    raise ValueError(
        f"{field_name} returned object name {normalized!r} outside the parsed episode object inventory. "
        f"Allowed names: {allowed_preview}"
    )


def _normalize_allowed_object_names(
    values: list[str],
    allowed_object_names: set[str],
    *,
    field_name: str,
) -> list[str]:
    return [_normalize_allowed_object_name(item, allowed_object_names, field_name=field_name) for item in values]


def _coerce_placeholder_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        text = str(value).strip()
        items = [text] if text else []
    return [_validate_snake_case(item, field_name="parameter_placeholders") for item in items]


def load_raw_trajectory_steps(
    input_dir: str | Path,
) -> list[RawTrajectoryStep]:
    episode_paths = discover_episode_files(input_dir)
    logger.debug("Discovered %d full-chain episode files under %s", len(episode_paths), input_dir)

    steps: list[RawTrajectoryStep] = []
    for path in episode_paths:
        payload = load_episode_payload(path)
        frame_paths_by_step = _load_frame_paths_by_step(path)
        episode_name = str(payload.get("episode_name", path.parent.name))
        instruction = str(payload.get("instruction", "")).strip()
        if not instruction:
            raise ValueError(f"Episode {episode_name!r} is missing required field `instruction` in {path}")
        previous_observation_text: str | None = None
        previous_known_observation_text: str | None = None
        for step in payload.get("steps", []):
            observation_text = _normalize_optional_text(step.get("observation_text"))
            action_text = _normalize_optional_text(step.get("action_text"))
            extra_info = _normalize_optional_text(step.get("extra_info"))
            raw_step = RawTrajectoryStep(
                episode_name=episode_name,
                instruction=instruction,
                step_index=int(step["step_index"]),
                start_time_sec=step.get("start_time_sec"),
                end_time_sec=step.get("end_time_sec"),
                action_text=action_text,
                observation_text=observation_text,
                extra_info=extra_info,
                previous_observation_text=previous_observation_text,
                previous_known_observation_text=previous_known_observation_text,
                frame_paths=list(frame_paths_by_step.get(int(step["step_index"]), [])),
            )
            steps.append(raw_step)
            previous_observation_text = observation_text
            if observation_text is not None:
                previous_known_observation_text = observation_text
    logger.debug("Loaded %d trajectory steps", len(steps))
    return steps


def _load_frame_paths_by_step(episode_file: Path) -> dict[int, list[str]]:
    manifest_candidates = [
        episode_file.parent / "frames" / "frame_manifest.json",
        episode_file.parent / "frame_manifest.json",
    ]
    manifest_path = next((item for item in manifest_candidates if item.exists()), None)
    if manifest_path is None:
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read frame manifest for %s at %s", episode_file, manifest_candidates[0])
        return {}
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return {}
    frame_paths_by_step: dict[int, list[str]] = {}
    for item in steps:
        if not isinstance(item, dict):
            continue
        try:
            step_index = int(item.get("step_index"))
        except (TypeError, ValueError):
            continue
        frame_paths = item.get("frame_paths")
        if not isinstance(frame_paths, list):
            continue
        frame_paths_by_step[step_index] = [str(path).strip() for path in frame_paths if str(path).strip()]
    return frame_paths_by_step


@dataclass
class LLMActionTaxonomyModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 900
    max_workers: int = 1
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {self.max_workers}")
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("action_taxonomy_prompt.md")

    def classify_actions(self, steps: list[RawTrajectoryStep]) -> list[ActionTaxonomyRecord]:
        return self._classify_actions_impl(steps, allowed_action_schemas=None, episode_object_inventories=None)

    def classify_actions_with_episode_objects(
        self,
        steps: list[RawTrajectoryStep],
        episode_object_inventories: list[EpisodeObjectInventory],
    ) -> list[ActionTaxonomyRecord]:
        return self._classify_actions_impl(
            steps,
            allowed_action_schemas=None,
            episode_object_inventories=episode_object_inventories,
        )

    def classify_actions_with_allowed_schemas(
        self,
        steps: list[RawTrajectoryStep],
        allowed_action_schemas: list[ActionSchema],
        episode_object_inventories: list[EpisodeObjectInventory] | None = None,
    ) -> list[ActionTaxonomyRecord]:
        return self._classify_actions_impl(
            steps,
            allowed_action_schemas=allowed_action_schemas,
            episode_object_inventories=episode_object_inventories,
        )

    def _classify_actions_impl(
        self,
        steps: list[RawTrajectoryStep],
        *,
        allowed_action_schemas: list[ActionSchema] | None,
        episode_object_inventories: list[EpisodeObjectInventory] | None,
    ) -> list[ActionTaxonomyRecord]:
        actionable_steps = [step for step in steps if step.action_text]
        inventory_by_episode = {
            item.episode_name: set(item.object_names) for item in (episode_object_inventories or [])
        }
        if allowed_action_schemas:
            logger.info(
                "Action taxonomy (LLM): constraining classification to %d allowed action schemas",
                len(allowed_action_schemas),
            )
        logger.info(
            "Action taxonomy (LLM): classifying %d action steps (max_workers=%d)",
            len(actionable_steps),
            self.max_workers,
        )
        if self.max_workers == 1:
            records = self._classify_actions_sequentially(
                actionable_steps,
                allowed_action_schemas,
                inventory_by_episode,
            )
        else:
            records = self._classify_actions_in_parallel(
                actionable_steps,
                allowed_action_schemas,
                inventory_by_episode,
            )
        logger.info("Action taxonomy (LLM) complete: produced %d records", len(records))
        return records

    def _classify_actions_sequentially(
        self,
        actionable_steps: list[RawTrajectoryStep],
        allowed_action_schemas: list[ActionSchema] | None,
        inventory_by_episode: dict[str, set[str]],
    ) -> list[ActionTaxonomyRecord]:
        records: list[ActionTaxonomyRecord] = []
        total = len(actionable_steps)
        for index, step in enumerate(actionable_steps, start=1):
            logger.info(
                "Action taxonomy (LLM): step %d/%d [%s step %d]",
                index,
                total,
                step.episode_name,
                step.step_index,
            )
            records.append(
                self._classify_single_action(step, allowed_action_schemas, inventory_by_episode.get(step.episode_name))
            )
        return records

    def _classify_actions_in_parallel(
        self,
        actionable_steps: list[RawTrajectoryStep],
        allowed_action_schemas: list[ActionSchema] | None,
        inventory_by_episode: dict[str, set[str]],
    ) -> list[ActionTaxonomyRecord]:
        total = len(actionable_steps)
        if total == 0:
            return []
        logger.info("Action taxonomy (LLM): submitting %d tasks to thread pool", total)
        ordered_records: list[ActionTaxonomyRecord | None] = [None] * total
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="action-taxonomy") as executor:
            future_to_index = {
                executor.submit(
                    self._classify_single_action,
                    step,
                    allowed_action_schemas,
                    inventory_by_episode.get(step.episode_name),
                ): index
                for index, step in enumerate(actionable_steps)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                step = actionable_steps[index]
                try:
                    ordered_records[index] = future.result()
                except Exception as exc:
                    logger.error(
                        "Action taxonomy (LLM) failed at [%s step %d]: %s",
                        step.episode_name,
                        step.step_index,
                        exc,
                    )
                    raise
                completed += 1
                logger.debug(
                    "Action taxonomy (LLM): completed %d/%d [%s step %d]",
                    completed,
                    total,
                    step.episode_name,
                    step.step_index,
                )
        return [record for record in ordered_records if record is not None]

    def _classify_single_action(
        self,
        step: RawTrajectoryStep,
        allowed_action_schemas: list[ActionSchema] | None,
        allowed_object_names: set[str] | None,
    ) -> ActionTaxonomyRecord:
        allowed_schema_map = (
            {schema.canonical_action_name: schema for schema in allowed_action_schemas}
            if allowed_action_schemas
            else {}
        )
        payload = {
            "instruction": step.instruction,
            "step": step.to_dict(),
        }
        if allowed_object_names:
            payload["allowed_object_names"] = sorted(allowed_object_names)
        if allowed_action_schemas:
            payload["allowed_action_schemas"] = [schema.to_dict() for schema in allowed_action_schemas]
        reply = safe_chat(
            self._client,
            self._prompt,
            json.dumps(payload, ensure_ascii=False, indent=2),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = _extract_payload(extract_json_object(reply), "action_record")
        canonical_action_name = _coerce_snake_case(
            _lookup_first(data, ["canonical_action_name", "action_name"]) or "",
            field_name="canonical_action_name",
        )
        if allowed_schema_map and canonical_action_name not in allowed_schema_map:
            raise ValueError(
                f"Action taxonomy response returned unknown canonical_action_name {canonical_action_name!r}. "
                f"Allowed: {sorted(allowed_schema_map)}"
            )
        action_category = str(_lookup_first(data, ["action_category", "category"]) or "").strip().lower()
        if allowed_schema_map:
            expected_category = allowed_schema_map[canonical_action_name].action_category
            if action_category and action_category != expected_category:
                logger.warning(
                    "Action taxonomy category mismatch for %s: got %s, expected %s. Using expected category.",
                    canonical_action_name,
                    action_category,
                    expected_category,
                )
            action_category = expected_category
        if action_category not in {"manipulation", "active_observation"}:
            raise ValueError(f"Unsupported action_category from LLM: {action_category!r}")
        action_arguments = coarsen_object_identifiers(
            _coerce_string_list(
                _lookup_first(data, ["action_arguments", "arguments"]),
                field_name="action_arguments",
            )
        )
        object_mentions = coarsen_object_identifiers(
            _coerce_string_list(
                _lookup_first(data, ["object_mentions", "objects"]),
                field_name="object_mentions",
            )
        )
        if allowed_object_names:
            action_arguments = _normalize_allowed_object_names(
                action_arguments,
                allowed_object_names,
                field_name="action_arguments",
            )
            object_mentions = _normalize_allowed_object_names(
                object_mentions,
                allowed_object_names,
                field_name="object_mentions",
            )
        parameter_placeholders = _coerce_placeholder_list(
            _lookup_first(data, ["parameter_placeholders", "template_parameter_placeholders"])
        )
        if not parameter_placeholders and action_arguments:
            parameter_placeholders = [
                "object" if index == 0 else f"object_{index + 1}" for index in range(len(action_arguments))
            ]
        template_text_raw = _normalize_optional_text(_lookup_first(data, ["template_text", "action_template_text"]))
        return ActionTaxonomyRecord(
            episode_name=step.episode_name,
            step_index=step.step_index,
            raw_action_text=step.action_text or "",
            proposed_action_name=canonical_action_name,
            canonical_action_name=canonical_action_name,
            action_category=action_category,
            action_arguments=action_arguments,
            object_mentions=object_mentions,
            observation_text=step.observation_text,
            extra_info=step.extra_info,
            template_text=template_text_raw,
            parameter_placeholders=parameter_placeholders,
        )


@dataclass
class LLMActionSchemaConsolidationModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1800
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("action_schema_consolidation_prompt.md")

    def consolidate(
        self,
        steps: list[RawTrajectoryStep],
        taxonomy_records: list[ActionTaxonomyRecord],
        predicate_inventory: list[PredicateSchema] | None = None,
    ) -> tuple[list[ActionSchema], list[ActionTaxonomyRecord]]:
        return self._consolidate_impl(
            steps,
            taxonomy_records,
            allowed_action_schemas=None,
            predicate_inventory=predicate_inventory,
        )

    def consolidate_with_allowed_schemas(
        self,
        steps: list[RawTrajectoryStep],
        taxonomy_records: list[ActionTaxonomyRecord],
        allowed_action_schemas: list[ActionSchema],
        predicate_inventory: list[PredicateSchema] | None = None,
    ) -> tuple[list[ActionSchema], list[ActionTaxonomyRecord]]:
        return self._consolidate_impl(
            steps,
            taxonomy_records,
            allowed_action_schemas=allowed_action_schemas,
            predicate_inventory=predicate_inventory,
        )

    def _consolidate_impl(
        self,
        steps: list[RawTrajectoryStep],
        taxonomy_records: list[ActionTaxonomyRecord],
        *,
        allowed_action_schemas: list[ActionSchema] | None,
        predicate_inventory: list[PredicateSchema] | None,
    ) -> tuple[list[ActionSchema], list[ActionTaxonomyRecord]]:
        logger.info(
            "Action schema consolidation (LLM): consolidating %d taxonomy records",
            len(taxonomy_records),
        )
        payload = {
            "steps": [step.to_dict() for step in steps if step.action_text],
            "taxonomy_records": [
                _taxonomy_record_prompt_dict(record, include_observation=False) for record in taxonomy_records
            ],
            "instruction": steps[0].instruction if steps else "",
        }
        payload["steps"] = [
            _step_prompt_dict(step, include_current_observation=_is_failure(step.extra_info))
            for step in steps
            if step.action_text
        ]
        if allowed_action_schemas:
            payload["allowed_action_schemas"] = [schema.to_dict() for schema in allowed_action_schemas]
        if predicate_inventory:
            payload["allowed_predicates"] = [item.to_dict() for item in predicate_inventory]
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
        schema_payloads = data.get("action_schemas", [])
        assignment_payloads = data.get("record_assignments", [])
        if not isinstance(schema_payloads, list) or not isinstance(assignment_payloads, list):
            raise ValueError("LLM consolidation response must contain lists: action_schemas and record_assignments")

        allowed_schema_map = (
            {schema.canonical_action_name: schema for schema in allowed_action_schemas}
            if allowed_action_schemas
            else {}
        )
        schemas: list[ActionSchema] = []
        if not allowed_schema_map:
            for item in schema_payloads:
                if not isinstance(item, dict):
                    raise ValueError(f"Invalid action schema payload: {item!r}")
                canonical_action_name = _coerce_snake_case(
                    _lookup_first(item, ["canonical_action_name", "action_name"]) or "",
                    field_name="canonical_action_name",
                )
                action_category = str(_lookup_first(item, ["action_category", "category"]) or "").strip().lower()
                if action_category not in {"manipulation", "active_observation"}:
                    raise ValueError(f"Unsupported action schema category from LLM: {action_category!r}")
                parameter_roles_raw = item.get("parameter_roles", [])
                if not isinstance(parameter_roles_raw, list):
                    parameter_roles_raw = []
                parameter_roles = [str(role).strip() for role in parameter_roles_raw if str(role).strip()]
                parameter_count = int(_lookup_first(item, ["parameter_count"]) or len(parameter_roles))
                # Preconditions are learned in a separate post-grounding stage.
                precondition_literals: list[str] = []
                schemas.append(
                    ActionSchema(
                        canonical_action_name=canonical_action_name,
                        action_category=action_category,
                        parameter_count=parameter_count,
                        parameter_roles=parameter_roles,
                        precondition_literals=precondition_literals,
                        schema_description=_normalize_optional_text(
                            _lookup_first(item, ["schema_description", "description"])
                        ),
                    )
                )
        schema_name_set = {schema.canonical_action_name for schema in schemas}
        normalized_schema_map = {schema.canonical_action_name: schema for schema in schemas}

        assignment_map: dict[tuple[str, int], str] = {}
        for item in assignment_payloads:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid record assignment payload: {item!r}")
            episode_name = str(_lookup_first(item, ["episode_name"]) or "").strip()
            step_index = int(_lookup_first(item, ["step_index"]) or -1)
            canonical_action_name = _coerce_snake_case(
                _lookup_first(item, ["canonical_action_name", "action_name"]) or "",
                field_name="canonical_action_name",
            )
            if allowed_schema_map and canonical_action_name not in allowed_schema_map:
                raise ValueError(
                    f"Consolidation assignment references disallowed action schema: {canonical_action_name!r}. "
                    f"Allowed: {sorted(allowed_schema_map)}"
                )
            if not allowed_schema_map and canonical_action_name not in schema_name_set:
                raise ValueError(
                    f"Consolidation assignment references unknown action schema: {canonical_action_name!r}"
                )
            assignment_map[(episode_name, step_index)] = canonical_action_name

        normalized_records: list[ActionTaxonomyRecord] = []
        for record in taxonomy_records:
            canonical_action_name = assignment_map.get(
                (record.episode_name, record.step_index), record.canonical_action_name
            )
            if allowed_schema_map and canonical_action_name not in allowed_schema_map:
                raise ValueError(
                    f"Normalized taxonomy record references disallowed action schema: {canonical_action_name!r}. "
                    f"Allowed: {sorted(allowed_schema_map)}"
                )
            resolved_schema = (
                allowed_schema_map.get(canonical_action_name)
                if allowed_schema_map
                else normalized_schema_map.get(canonical_action_name)
            )
            normalized_action_category = (
                resolved_schema.action_category if resolved_schema is not None else record.action_category
            )
            normalized_records.append(
                ActionTaxonomyRecord(
                    episode_name=record.episode_name,
                    step_index=record.step_index,
                    raw_action_text=record.raw_action_text,
                    proposed_action_name=record.proposed_action_name,
                    canonical_action_name=canonical_action_name,
                    action_category=normalized_action_category,
                    action_arguments=record.action_arguments,
                    object_mentions=record.object_mentions,
                    observation_text=record.observation_text,
                    extra_info=record.extra_info,
                )
            )
        if allowed_schema_map:
            used_action_names = {record.canonical_action_name for record in normalized_records}
            schemas = [
                schema for schema in allowed_action_schemas or [] if schema.canonical_action_name in used_action_names
            ]
        logger.info(
            "Action schema consolidation (LLM) complete: %d schemas, %d normalized records",
            len(schemas),
            len(normalized_records),
        )
        return schemas, normalized_records


@dataclass
class LLMManipulationEffectLearningModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1400
    max_workers: int = 1
    max_validation_attempts: int = 2
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {self.max_workers}")
        if self.max_validation_attempts < 1:
            raise ValueError(f"max_validation_attempts must be at least 1, got {self.max_validation_attempts}")
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("manipulation_delta_prompt.md")

    def learn_effects(
        self,
        steps: list[RawTrajectoryStep],
        action_schemas: list[ActionSchema],
        taxonomy_records: list[ActionTaxonomyRecord],
        predicate_inventory: list[PredicateSchema] | None = None,
    ) -> list[ManipulationEffectRecord]:
        step_map = {(step.episode_name, step.step_index): step for step in steps}
        manipulation_records = [record for record in taxonomy_records if record.action_category == "manipulation"]
        manipulation_schemas = [
            schema.to_dict() for schema in action_schemas if schema.action_category == "manipulation"
        ]
        allowed_action_names = {schema["canonical_action_name"] for schema in manipulation_schemas}
        allowed_predicates = {item.predicate_name for item in predicate_inventory or []}
        allowed_predicate_inventory = [item.to_dict() for item in predicate_inventory or []]
        predicate_arities = _predicate_arity_map(predicate_inventory)
        immutable_predicates = {
            item.predicate_name for item in predicate_inventory or [] if item.is_static_feature
        }
        logger.info(
            "Manipulation effect learning (LLM): learning from %d manipulation records (max_workers=%d)",
            len(manipulation_records),
            self.max_workers,
        )
        if self.max_workers == 1:
            outputs = self._learn_effects_sequentially(
                step_map,
                manipulation_records,
                manipulation_schemas,
                allowed_action_names,
                allowed_predicates,
                allowed_predicate_inventory,
                predicate_arities,
                immutable_predicates,
            )
        else:
            outputs = self._learn_effects_in_parallel(
                step_map,
                manipulation_records,
                manipulation_schemas,
                allowed_action_names,
                allowed_predicates,
                allowed_predicate_inventory,
                predicate_arities,
                immutable_predicates,
            )
        logger.info(
            "Manipulation effect learning (LLM) complete: produced %d effect records",
            len(outputs),
        )
        return outputs

    def _learn_effects_sequentially(
        self,
        step_map: dict[tuple[str, int], RawTrajectoryStep],
        manipulation_records: list[ActionTaxonomyRecord],
        manipulation_schemas: list[dict[str, object]],
        allowed_action_names: set[str],
        allowed_predicates: set[str],
        allowed_predicate_inventory: list[dict[str, object]],
        predicate_arities: dict[str, int],
        immutable_predicates: set[str],
    ) -> list[ManipulationEffectRecord]:
        outputs: list[ManipulationEffectRecord] = []
        total = len(manipulation_records)
        for index, taxonomy_record in enumerate(manipulation_records, start=1):
            step = step_map[(taxonomy_record.episode_name, taxonomy_record.step_index)]
            logger.info(
                "Manipulation effect learning (LLM): record %d/%d [%s step %d]",
                index,
                total,
                step.episode_name,
                step.step_index,
            )
            outputs.append(
                self._learn_single_effect(
                    step,
                    taxonomy_record,
                    manipulation_schemas,
                    allowed_action_names,
                    allowed_predicates,
                    allowed_predicate_inventory,
                    predicate_arities,
                    immutable_predicates,
                )
            )
        return outputs

    def _learn_effects_in_parallel(
        self,
        step_map: dict[tuple[str, int], RawTrajectoryStep],
        manipulation_records: list[ActionTaxonomyRecord],
        manipulation_schemas: list[dict[str, object]],
        allowed_action_names: set[str],
        allowed_predicates: set[str],
        allowed_predicate_inventory: list[dict[str, object]],
        predicate_arities: dict[str, int],
        immutable_predicates: set[str],
    ) -> list[ManipulationEffectRecord]:
        total = len(manipulation_records)
        if total == 0:
            return []
        logger.info("Manipulation effect learning (LLM): submitting %d tasks to thread pool", total)
        ordered_records: list[ManipulationEffectRecord | None] = [None] * total
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="manip-effects") as executor:
            future_to_index = {
                executor.submit(
                    self._learn_single_effect,
                    step_map[(record.episode_name, record.step_index)],
                    record,
                    manipulation_schemas,
                    allowed_action_names,
                    allowed_predicates,
                    allowed_predicate_inventory,
                    predicate_arities,
                    immutable_predicates,
                ): index
                for index, record in enumerate(manipulation_records)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                taxonomy_record = manipulation_records[index]
                step = step_map[(taxonomy_record.episode_name, taxonomy_record.step_index)]
                try:
                    ordered_records[index] = future.result()
                except Exception as exc:
                    logger.error(
                        "Manipulation effect learning (LLM) failed at [%s step %d]: %s",
                        step.episode_name,
                        step.step_index,
                        exc,
                    )
                    raise
                completed += 1
                logger.debug(
                    "Manipulation effect learning (LLM): completed %d/%d [%s step %d]",
                    completed,
                    total,
                    step.episode_name,
                    step.step_index,
                )
        return [record for record in ordered_records if record is not None]

    def _learn_single_effect(
        self,
        step: RawTrajectoryStep,
        taxonomy_record: ActionTaxonomyRecord,
        manipulation_schemas: list[dict[str, object]],
        allowed_action_names: set[str],
        allowed_predicates: set[str],
        allowed_predicate_inventory: list[dict[str, object]],
        predicate_arities: dict[str, int],
        immutable_predicates: set[str],
    ) -> ManipulationEffectRecord:
        validation_feedback: str | None = None
        for attempt in range(1, self.max_validation_attempts + 1):
            payload = {
                "instruction": step.instruction,
                "manipulation_action_schemas": manipulation_schemas,
                "taxonomy_record": _taxonomy_record_prompt_dict(taxonomy_record, include_observation=False),
                "step": _step_prompt_dict(step, include_current_observation=_is_failure(step.extra_info)),
            }
            if allowed_predicate_inventory:
                payload["allowed_predicates"] = allowed_predicate_inventory
            if validation_feedback:
                payload["previous_validation_error"] = validation_feedback
            reply = safe_chat(
                self._client,
                self._prompt,
                json.dumps(payload, ensure_ascii=False, indent=2),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                verbose=self.verbose,
            )
            data = _extract_payload(extract_json_object(reply), "manipulation_record")
            effect_bucket = _coerce_snake_case(
                _lookup_first(data, ["effect_bucket", "bucket"]) or "",
                field_name="effect_bucket",
            )
            action_name = _coerce_snake_case(
                _lookup_first(data, ["canonical_action_name", "action_name"]) or taxonomy_record.canonical_action_name,
                field_name="canonical_action_name",
            )
            if action_name not in allowed_action_names:
                raise ValueError(
                    f"Manipulation effect response returned unknown canonical_action_name {action_name!r}. "
                    f"Allowed: {sorted(allowed_action_names)}"
                )
            delta_add = coarsen_symbolic_literal_list(
                _coerce_fact_list(_lookup_first(data, ["delta_add", "add_facts"]), field_name="delta_add")
            )
            delta_del = coarsen_symbolic_literal_list(
                _coerce_fact_list(_lookup_first(data, ["delta_del", "del_facts"]), field_name="delta_del")
            )
            try:
                _validate_effect_literals_against_inventory(
                    delta_add=delta_add,
                    delta_del=delta_del,
                    allowed_predicates=allowed_predicates,
                    predicate_arities=predicate_arities,
                    immutable_predicates=immutable_predicates,
                    context=(
                        f"Manipulation effect learning response [{step.episode_name} step {step.step_index}] "
                        f"for action `{action_name}`"
                    ),
                )
            except ValueError as exc:
                if attempt >= self.max_validation_attempts:
                    raise
                validation_feedback = str(exc)
                logger.warning(
                    "Manipulation effect learning validation failed at [%s step %d] attempt %d/%d: %s. Retrying this sample.",
                    step.episode_name,
                    step.step_index,
                    attempt,
                    self.max_validation_attempts,
                    validation_feedback,
                )
                continue
            return ManipulationEffectRecord(
                episode_name=step.episode_name,
                step_index=step.step_index,
                raw_action_text=taxonomy_record.raw_action_text,
                canonical_action_name=action_name,
                action_arguments=coarsen_object_identifiers(
                    _coerce_string_list(
                        _lookup_first(data, ["action_arguments", "arguments"]) or taxonomy_record.action_arguments,
                        field_name="action_arguments",
                    )
                ),
                pre_observation_text=None,
                post_observation_text=None,
                extra_info=step.extra_info,
                delta_add=delta_add,
                delta_del=delta_del,
                effect_bucket=effect_bucket,
                success=_coerce_bool(_lookup_first(data, ["success", "is_success"]), field_name="success"),
            )
        raise AssertionError(
            "Unreachable: manipulation effect learning validation retry loop exhausted without returning."
        )


__all__ = [
    "ActionTaxonomyModule",
    "ActionSchemaConsolidationModule",
    "LLMActionTaxonomyModule",
    "LLMActionSchemaConsolidationModule",
    "LLMManipulationEffectLearningModule",
    "ManipulationEffectLearningModule",
    "ObservationActionLearningModule",
    "load_raw_trajectory_steps",
]
