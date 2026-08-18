from __future__ import annotations

import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from po_pddl.domain_generation.infrastructure.artifact_io import load_json, load_json_object, load_jsonl
from po_pddl.domain_generation.infrastructure.fact_utils import parse_symbolic_literal
from po_pddl.domain_generation.infrastructure.type_hierarchy import build_type_parent_map
from po_pddl.domain_generation.stages.last_action_markers import (
    last_action_constant_name,
    last_action_predicate_name,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.learner import load_raw_trajectory_steps
from po_pddl.domain_generation.stages.manipulation_domain_learning.models import (
    ActionTaxonomyRecord,
    PredicateSchema,
)

from .models import (
    ActiveObservationCondition,
    ActiveObservationDiscoveryResult,
    ActiveObservationExample,
    ActiveObservationLearningResult,
    ActiveObservationRuleSchema,
    ActiveObservationSourceRecord,
    ConfirmedObservation,
    DiscoveredObservation,
)
from .modules import ActiveObservationDiscoveryModule, ActiveObservationValueConfirmationModule
from .renderer import render_active_observation_module

logger = logging.getLogger(__name__)


EXCLUDED_ACTIVE_OBSERVATION_PREDICATES = {"gripper_empty", "gripper_holding"}


def _normalize_discovered_observation_predicate(
    discovered: DiscoveredObservation,
    *,
    predicate_by_name: dict[str, PredicateSchema],
    record: ActiveObservationSourceRecord,
    parent_by_type: dict[str, str],
) -> DiscoveredObservation | None:
    """Resolve state-predicate names from the grounded literal, not observable aliases."""
    negated, literal_predicate, arguments = parse_symbolic_literal(discovered.grounded_literal)
    if literal_predicate not in predicate_by_name:
        return None
    predicate_schema = predicate_by_name[literal_predicate]
    if len(arguments) != len(predicate_schema.parameter_types):
        return None
    if len(arguments) > 1 and len(set(arguments)) != len(arguments):
        return None
    for argument, required_type in zip(arguments, predicate_schema.parameter_types):
        object_type = record.objects.get(argument)
        if object_type is None or not _object_matches_type(object_type, required_type, parent_by_type):
            return None
    grounded_literal = _instantiate_template(literal_predicate, arguments)
    return DiscoveredObservation(
        predicate_name=literal_predicate,
        grounded_literal=grounded_literal,
        observed_value=False if negated else discovered.observed_value,
        rationale=discovered.rationale,
    )


def _literal_arguments(literal: str) -> list[str]:
    return parse_symbolic_literal(literal)[2]


def _instantiate_template(predicate_name: str, parameter_names: list[str]) -> str:
    return f"{predicate_name}({','.join(parameter_names)})" if parameter_names else f"{predicate_name}()"


def _bucket_variant_rank(effect_bucket: str) -> int:
    suffix = str(effect_bucket).rsplit("_bucket_", 1)
    if len(suffix) == 2 and suffix[1].isdigit():
        return int(suffix[1])
    return 0


def _is_failure_extra_info(extra_info: str | None) -> bool:
    text = str(extra_info or "").strip().lower()
    if not text:
        return False
    return any(token in text for token in ("fail", "failed", "failure", "unsuccess", "error"))


def _filter_current_state_for_action_scope(
    current_state: list[str],
    action_arguments: list[str],
    effect_predicate_names: list[str] | None = None,
) -> list[str]:
    action_argument_set = set(action_arguments)
    effect_predicate_set = {str(name).strip() for name in (effect_predicate_names or []) if str(name).strip()}
    filtered: list[str] = []
    for fact in current_state:
        _negated, predicate, arguments = parse_symbolic_literal(fact)
        if predicate in EXCLUDED_ACTIVE_OBSERVATION_PREDICATES:
            continue
        if predicate in effect_predicate_set:
            continue
        if not arguments:
            filtered.append(fact)
            continue
        if any(argument in action_argument_set for argument in arguments):
            filtered.append(fact)
    return sorted(dict.fromkeys(filtered))


def _filter_current_state_for_condition_mining(
    current_state: list[str],
    effect_predicate_names: list[str] | None = None,
) -> list[str]:
    effect_predicate_set = {str(name).strip() for name in (effect_predicate_names or []) if str(name).strip()}
    filtered: list[str] = []
    for fact in current_state:
        _negated, predicate, _arguments = parse_symbolic_literal(fact)
        if predicate in EXCLUDED_ACTIVE_OBSERVATION_PREDICATES:
            continue
        if predicate in effect_predicate_set:
            continue
        filtered.append(fact)
    return sorted(dict.fromkeys(filtered))


def _filter_ground_truth_facts(facts: list[str]) -> list[str]:
    filtered: list[str] = []
    for fact in facts:
        _negated, predicate, _arguments = parse_symbolic_literal(fact)
        if predicate in EXCLUDED_ACTIVE_OBSERVATION_PREDICATES:
            continue
        filtered.append(fact)
    return sorted(dict.fromkeys(filtered))


def _object_matches_type(object_type: str, required_type: str, parent_by_type: dict[str, str]) -> bool:
    if required_type == "object":
        return True
    current = object_type
    seen: set[str] = set()
    while current and current not in seen:
        if current == required_type:
            return True
        seen.add(current)
        current = parent_by_type.get(current, "")
    return False


def _build_complete_ground_truth_assignments(
    *,
    objects: dict[str, str],
    predicate_inventory: list[PredicateSchema],
    true_facts: list[str],
    parent_by_type: dict[str, str],
) -> list[str]:
    filtered_true_facts = set(_filter_ground_truth_facts(true_facts))
    assignments: list[str] = []
    for predicate in predicate_inventory:
        if predicate.predicate_name in EXCLUDED_ACTIVE_OBSERVATION_PREDICATES:
            continue
        if not predicate.parameter_types:
            positive_literal = f"{predicate.predicate_name}()"
            assignments.append(
                positive_literal if positive_literal in filtered_true_facts else f"not {positive_literal}"
            )
            continue
        candidate_object_lists: list[list[str]] = []
        for required_type in predicate.parameter_types:
            compatible_names = sorted(
                object_name
                for object_name, object_type in objects.items()
                if _object_matches_type(object_type, required_type, parent_by_type)
            )
            if not compatible_names:
                candidate_object_lists = []
                break
            candidate_object_lists.append(compatible_names)
        if not candidate_object_lists:
            continue
        tuples: list[list[str]] = [[]]
        for names in candidate_object_lists:
            next_rows: list[list[str]] = []
            for prefix in tuples:
                for name in names:
                    next_rows.append([*prefix, name])
            tuples = next_rows
        for argument_values in tuples:
            positive_literal = _instantiate_template(predicate.predicate_name, argument_values)
            assignments.append(
                positive_literal if positive_literal in filtered_true_facts else f"not {positive_literal}"
            )
    return sorted(dict.fromkeys(assignments))


def _load_camera_order_by_episode(scene_root: Path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for summary_file in sorted(scene_root.glob("*/scene_description_summary.json")):
        try:
            payload = load_json_object(summary_file)
        except Exception:
            continue
        episode_name = str(payload.get("episode_name") or summary_file.parent.name).strip()
        if not episode_name:
            continue
        raw_order = payload.get("camera_order_top_to_bottom")
        if isinstance(raw_order, list):
            mapping[episode_name] = [str(item) for item in raw_order if str(item).strip()]
    return mapping


@dataclass
class ActiveObservationLearner:
    discovery_module: ActiveObservationDiscoveryModule
    vlm_confirmation_module: ActiveObservationValueConfirmationModule
    manipulation_artifact_dir: str | Path | None = None
    scene_description_dir: str | Path | None = None
    passive_observation_learning_dir: str | Path | None = None
    init_observation_learning_dir: str | Path | None = None
    max_workers: int = 1

    def learn_from_source_records(
        self,
        *,
        source_records: list[ActiveObservationSourceRecord],
        predicate_inventory: list[PredicateSchema],
        predicate_comments: dict[str, str],
        type_parents: dict[str, str] | None = None,
    ) -> ActiveObservationLearningResult:
        parent_by_type = dict(type_parents or {})
        observation_predicate_inventory = [
            item for item in predicate_inventory if item.predicate_name not in EXCLUDED_ACTIVE_OBSERVATION_PREDICATES
        ]
        filtered_predicate_comments = {
            name: comment
            for name, comment in predicate_comments.items()
            if name not in EXCLUDED_ACTIVE_OBSERVATION_PREDICATES
        }
        predicate_by_name = {item.predicate_name: item for item in observation_predicate_inventory}
        feature_predicate_names = [
            item.predicate_name for item in observation_predicate_inventory if item.predicate_kind == "feature"
        ]
        available_observables = self._load_available_observables(observation_predicate_inventory)

        ordered_records = sorted(
            source_records, key=lambda item: (item.canonical_action_name, item.episode_name, item.step_index)
        )
        discovery_payloads: list[dict[str, Any]] = [None] * len(ordered_records)  # type: ignore[list-item]
        if ordered_records:
            worker_count = max(1, min(self.max_workers, len(ordered_records)))
            if worker_count == 1:
                for index, record in enumerate(ordered_records):
                    discovery_payloads[index] = self._discover_single_record(
                        record=record,
                        predicate_inventory=observation_predicate_inventory,
                        parent_by_type=parent_by_type,
                        predicate_comments=filtered_predicate_comments,
                        feature_predicate_names=feature_predicate_names,
                        available_observables=available_observables,
                    )
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_map = {
                        executor.submit(
                            self._discover_single_record,
                            record=record,
                            predicate_inventory=observation_predicate_inventory,
                            parent_by_type=parent_by_type,
                            predicate_comments=filtered_predicate_comments,
                            feature_predicate_names=feature_predicate_names,
                            available_observables=available_observables,
                        ): index
                        for index, record in enumerate(ordered_records)
                    }
                    for future in as_completed(future_map):
                        discovery_payloads[future_map[future]] = future.result()

        discovery_results: list[dict[str, Any]] = []
        seed_examples: list[ActiveObservationExample] = []
        grouped_seed_examples: dict[tuple[str, str, str], list[ActiveObservationExample]] = defaultdict(list)
        for payload in discovery_payloads:
            if payload is None:
                continue
            record = payload["record"]
            discovery_results.append(payload["summary"])
            for discovered in payload["discovery_result"].discovered_observations:
                discovered = _normalize_discovered_observation_predicate(
                    discovered,
                    predicate_by_name=predicate_by_name,
                    record=record,
                    parent_by_type=parent_by_type,
                )
                if discovered is None:
                    continue
                predicate_schema = predicate_by_name.get(discovered.predicate_name)
                if predicate_schema is None:
                    continue
                example = self._build_seed_example(
                    record=record,
                    condition_mining_current_state=payload["condition_mining_current_state"],
                    discovered=discovered,
                    predicate_schema=predicate_schema,
                )
                seed_examples.append(example)
                grouped_seed_examples[
                    (example.canonical_action_name, example.effect_bucket, example.predicate_name)
                ].append(example)

        conditions: list[ActiveObservationCondition] = []
        for group_key, examples in sorted(grouped_seed_examples.items()):
            predicate_schema = predicate_by_name.get(group_key[2])
            if predicate_schema is None:
                continue
            conditions.append(self._mine_condition(examples=examples, predicate_schema=predicate_schema))

        conditions_by_action_bucket: dict[tuple[str, str], list[ActiveObservationCondition]] = defaultdict(list)
        for condition in conditions:
            conditions_by_action_bucket[(condition.canonical_action_name, condition.effect_bucket)].append(condition)

        confirmation_jobs: list[dict[str, Any]] = []
        for payload in discovery_payloads:
            if payload is None:
                continue
            record = payload["record"]
            targets = self._build_confirmation_targets(
                record=record,
                action_conditions=conditions_by_action_bucket.get(
                    (record.canonical_action_name, record.effect_bucket), []
                ),
                predicate_by_name=predicate_by_name,
                parent_by_type=parent_by_type,
            )
            if not targets:
                continue
            confirmation_jobs.append(
                {
                    "record": record,
                    "filtered_current_state": payload["filtered_current_state"],
                    "candidate_ground_truth_facts": payload["candidate_ground_truth_facts"],
                    "targets": targets,
                }
            )

        confirmation_payloads: list[dict[str, Any]] = [None] * len(confirmation_jobs)  # type: ignore[list-item]
        if confirmation_jobs:
            worker_count = max(1, min(self.max_workers, len(confirmation_jobs)))
            if worker_count == 1:
                for index, job in enumerate(confirmation_jobs):
                    confirmation_payloads[index] = self._confirm_single_record(
                        job=job,
                        predicate_comments=filtered_predicate_comments,
                    )
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_map = {
                        executor.submit(
                            self._confirm_single_record, job=job, predicate_comments=filtered_predicate_comments
                        ): index
                        for index, job in enumerate(confirmation_jobs)
                    }
                    for future in as_completed(future_map):
                        confirmation_payloads[future_map[future]] = future.result()

        confirmed_examples: list[ActiveObservationExample] = []
        for payload in confirmation_payloads:
            if payload is None:
                continue
            record = payload["record"]
            for observation in payload["confirmation_result"].confirmed_observations:
                predicate_schema = predicate_by_name.get(observation.predicate_name)
                if predicate_schema is None:
                    continue
                confirmed_examples.append(
                    self._build_confirmed_example(
                        record=record,
                        condition_mining_current_state=_filter_current_state_for_condition_mining(
                            record.current_state,
                            record.effect_predicate_names,
                        ),
                        observation=observation,
                        predicate_schema=predicate_schema,
                    )
                )

        schemas: list[ActiveObservationRuleSchema] = []
        for condition in conditions:
            predicate_schema = predicate_by_name.get(condition.predicate_name)
            if predicate_schema is None:
                continue
            related_examples = [
                item
                for item in confirmed_examples
                if (
                    item.canonical_action_name == condition.canonical_action_name
                    and item.effect_bucket == condition.effect_bucket
                    and item.predicate_name == condition.predicate_name
                )
            ]
            if not related_examples:
                continue
            schemas.append(
                self._synthesize_schema(
                    condition=condition,
                    predicate_schema=predicate_schema,
                    examples=related_examples,
                )
            )

        rendered_module_text = render_active_observation_module(schemas)
        return ActiveObservationLearningResult(
            source_records=source_records,
            discovery_results=discovery_results,
            seed_examples=seed_examples,
            conditions=conditions,
            confirmed_examples=confirmed_examples,
            schemas=schemas,
            predicate_inventory=observation_predicate_inventory,
            predicate_comments=filtered_predicate_comments,
            rendered_module_text=rendered_module_text,
        )

    def learn_from_pairs(
        self,
        *,
        domain_file: str | Path,
        episode_grounding_pairs: list[tuple[str | Path, str | Path]],
    ) -> ActiveObservationLearningResult:
        del domain_file
        if self.manipulation_artifact_dir is None or self.scene_description_dir is None:
            raise ValueError(
                "ActiveObservationLearner requires manipulation_artifact_dir and scene_description_dir "
                "to learn from pipeline episode-grounding pairs."
            )
        source_records, predicate_inventory, predicate_comments, parent_by_type = load_active_observation_inputs(
            manipulation_artifact_dir=self.manipulation_artifact_dir,
            scene_description_dir=self.scene_description_dir,
            episode_grounding_pairs=episode_grounding_pairs,
        )
        return self.learn_from_source_records(
            source_records=source_records,
            predicate_inventory=predicate_inventory,
            predicate_comments=predicate_comments,
            type_parents=parent_by_type,
        )

    def _discover_single_record(
        self,
        *,
        record: ActiveObservationSourceRecord,
        predicate_inventory: list[PredicateSchema],
        parent_by_type: dict[str, str],
        predicate_comments: dict[str, str],
        feature_predicate_names: list[str],
        available_observables: list[str],
    ) -> dict[str, Any]:
        filtered_current_state = _filter_current_state_for_action_scope(
            record.current_state,
            record.action_arguments,
            record.effect_predicate_names,
        )
        if not filtered_current_state:
            discovery_result = ActiveObservationDiscoveryResult(
                discovered_observations=[],
                summary="Skipped because all current-state predicates were filtered out.",
                raw_output=None,
            )
            candidate_ground_truth_facts: list[str] = []
        else:
            candidate_ground_truth_facts = _build_complete_ground_truth_assignments(
                objects=record.objects,
                predicate_inventory=predicate_inventory,
                true_facts=list(record.ground_truth_facts),
                parent_by_type=parent_by_type,
            )
            discovery_result = self.discovery_module.discover(
                record=record,
                filtered_current_state=filtered_current_state,
                candidate_ground_truth_facts=candidate_ground_truth_facts,
                predicate_comments=predicate_comments,
                feature_predicate_names=feature_predicate_names,
                available_observables=available_observables,
            )
        return {
            "record": record,
            "filtered_current_state": list(filtered_current_state),
            "condition_mining_current_state": _filter_current_state_for_condition_mining(
                record.current_state,
                record.effect_predicate_names,
            ),
            "candidate_ground_truth_facts": list(candidate_ground_truth_facts),
            "discovery_result": discovery_result,
            "summary": {
                "episode_name": record.episode_name,
                "step_index": record.step_index,
                "discovery_result": discovery_result.to_dict(),
            },
        }

    def _load_available_observables(self, predicate_inventory: list[PredicateSchema]) -> list[str]:
        observable_names: set[str] = set()
        for directory, filename in [
            (self.passive_observation_learning_dir, "passive_observation_schemas.json"),
            (self.init_observation_learning_dir, "init_observation_schemas.json"),
        ]:
            if directory is None:
                continue
            schema_file = Path(directory) / filename
            if not schema_file.exists():
                continue
            try:
                payload = load_json(schema_file)
            except Exception:
                continue
            if not isinstance(payload, list):
                continue
            for row in payload:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("observable_name") or row.get("positive_observable") or "").strip()
                if name:
                    observable_names.add(name)
        if not observable_names:
            observable_names = {
                f"obs_{item.predicate_name}"
                for item in predicate_inventory
                if item.predicate_name not in EXCLUDED_ACTIVE_OBSERVATION_PREDICATES
            }
        return sorted(observable_names)

    def _confirm_single_record(
        self,
        *,
        job: dict[str, Any],
        predicate_comments: dict[str, str],
    ) -> dict[str, Any]:
        record = job["record"]
        logger.debug(
            "Active observation confirmation started: episode=%s step=%d action=%s bucket=%s frames=%d",
            record.episode_name,
            record.step_index,
            record.canonical_action_name,
            record.effect_bucket,
            len(record.frame_paths),
        )
        result = self.vlm_confirmation_module.confirm(
            record=record,
            filtered_current_state=list(job["filtered_current_state"]),
            candidate_ground_truth_facts=list(job["candidate_ground_truth_facts"]),
            targets=list(job["targets"]),
            predicate_comments=predicate_comments,
        )
        payload = {
            "record": record,
            "confirmation_result": result,
        }
        logger.debug(
            "Active observation confirmation completed: episode=%s step=%d action=%s bucket=%s",
            record.episode_name,
            record.step_index,
            record.canonical_action_name,
            record.effect_bucket,
        )
        return payload

    def _build_seed_example(
        self,
        *,
        record: ActiveObservationSourceRecord,
        condition_mining_current_state: list[str],
        discovered: DiscoveredObservation,
        predicate_schema: PredicateSchema,
    ) -> ActiveObservationExample:
        literal_arguments = _literal_arguments(discovered.grounded_literal)
        action_argument_set = set(record.action_arguments)
        extra_arguments = [argument for argument in literal_arguments if argument not in action_argument_set]
        extra_argument_types = [
            record.objects.get(
                argument,
                predicate_schema.parameter_types[index] if index < len(predicate_schema.parameter_types) else "object",
            )
            for index, argument in enumerate(literal_arguments)
            if argument not in action_argument_set
        ]
        ground_truth_value = discovered.grounded_literal in set(record.ground_truth_facts)
        return ActiveObservationExample(
            episode_name=record.episode_name,
            step_index=record.step_index,
            canonical_action_name=record.canonical_action_name,
            effect_bucket=record.effect_bucket,
            success=record.success,
            variant_rank=record.variant_rank,
            predicate_name=discovered.predicate_name,
            grounded_literal=discovered.grounded_literal,
            action_arguments=list(record.action_arguments),
            action_argument_types=list(record.action_argument_types),
            extra_arguments=extra_arguments,
            extra_argument_types=extra_argument_types,
            ground_truth_value=ground_truth_value,
            observed_value=discovered.observed_value,
            current_state=list(condition_mining_current_state),
            source_kind="discovery_seed",
        )

    def _mine_condition(
        self,
        *,
        examples: list[ActiveObservationExample],
        predicate_schema: PredicateSchema,
    ) -> ActiveObservationCondition:
        first = examples[0]
        intersections: set[str] | None = None
        target_argument_templates = self._target_argument_templates(first)
        target_literal_template = _instantiate_template(first.predicate_name, target_argument_templates)
        for example in examples:
            candidate_literals: set[str] = set()
            extra_binding = {argument: f"?obs{index}" for index, argument in enumerate(example.extra_arguments)}
            action_binding = {argument: f"?arg{index}" for index, argument in enumerate(example.action_arguments)}
            binding = {**action_binding, **extra_binding}
            for fact in example.current_state:
                negated, predicate_name, literal_arguments = parse_symbolic_literal(fact)
                if predicate_name == example.predicate_name:
                    continue
                if not example.extra_arguments:
                    continue
                if not any(argument in example.extra_arguments for argument in literal_arguments):
                    continue
                if not all(argument in binding for argument in literal_arguments):
                    continue
                templated_arguments = [binding[argument] for argument in literal_arguments]
                if len(templated_arguments) == len(target_argument_templates) and set(
                    templated_arguments
                ) == set(target_argument_templates):
                    # Relations over exactly the target entities are commonly
                    # mutually exclusive state alternatives, not visibility conditions.
                    continue
                templated = _instantiate_template(
                    predicate_name,
                    templated_arguments,
                )
                if negated:
                    templated = f"not {templated}"
                candidate_literals.add(templated)
            intersections = (
                candidate_literals if intersections is None else intersections.intersection(candidate_literals)
            )
        return ActiveObservationCondition(
            canonical_action_name=first.canonical_action_name,
            effect_bucket=first.effect_bucket,
            success=first.success,
            variant_rank=first.variant_rank,
            predicate_name=first.predicate_name,
            action_argument_types=list(first.action_argument_types),
            extra_argument_types=list(first.extra_argument_types),
            target_literal_template=target_literal_template,
            condition_literals=sorted(intersections or set()),
        )

    def _build_confirmation_targets(
        self,
        *,
        record: ActiveObservationSourceRecord,
        action_conditions: list[ActiveObservationCondition],
        predicate_by_name: dict[str, PredicateSchema],
        parent_by_type: dict[str, str],
    ) -> list[dict[str, object]]:
        targets: list[dict[str, object]] = []
        for condition in action_conditions:
            predicate_schema = predicate_by_name.get(condition.predicate_name)
            if predicate_schema is None:
                continue
            position_mapping = self._target_position_mapping(
                record,
                condition.predicate_name,
                predicate_schema,
                parent_by_type,
            )
            action_positions = position_mapping["action_positions"]
            extra_positions = position_mapping["extra_positions"]
            extra_parameter_types = [predicate_schema.parameter_types[index] for index in extra_positions]
            candidate_object_lists: list[list[str]] = []
            for required_type in extra_parameter_types:
                candidate_object_lists.append(
                    sorted(
                        object_name
                        for object_name, object_type in record.objects.items()
                        if _object_matches_type(object_type, required_type, parent_by_type)
                    )
                )
            candidate_extra_bindings = [[]]
            if candidate_object_lists:
                candidate_extra_bindings = [[]]
                for object_names in candidate_object_lists:
                    next_bindings: list[list[str]] = []
                    for prefix in candidate_extra_bindings:
                        for object_name in object_names:
                            next_bindings.append([*prefix, object_name])
                    candidate_extra_bindings = next_bindings
            for extra_arguments in candidate_extra_bindings:
                if not self._condition_holds(
                    current_state=_filter_current_state_for_condition_mining(
                        record.current_state,
                        record.effect_predicate_names,
                    ),
                    condition_literals=condition.condition_literals,
                    action_arguments=record.action_arguments,
                    extra_arguments=extra_arguments,
                ):
                    continue
                target_arguments: list[str] = []
                action_index_by_position = {position: index for index, position in enumerate(action_positions)}
                extra_index_by_position = {position: index for index, position in enumerate(extra_positions)}
                for position in range(len(predicate_schema.parameter_types)):
                    if position in action_index_by_position:
                        target_arguments.append(record.action_arguments[action_index_by_position[position]])
                    else:
                        target_arguments.append(extra_arguments[extra_index_by_position[position]])
                grounded_literal = _instantiate_template(condition.predicate_name, target_arguments)
                targets.append(
                    {
                        "predicate_name": condition.predicate_name,
                        "grounded_literal": grounded_literal,
                        "ground_truth_value": grounded_literal in set(record.ground_truth_facts),
                    }
                )
        deduped: dict[tuple[str, str], dict[str, object]] = {}
        for item in targets:
            deduped[(str(item["predicate_name"]), str(item["grounded_literal"]))] = item
        return list(deduped.values())

    def _build_confirmed_example(
        self,
        *,
        record: ActiveObservationSourceRecord,
        condition_mining_current_state: list[str],
        observation: ConfirmedObservation,
        predicate_schema: PredicateSchema,
    ) -> ActiveObservationExample:
        literal_arguments = _literal_arguments(observation.grounded_literal)
        action_argument_set = set(record.action_arguments)
        extra_arguments = [argument for argument in literal_arguments if argument not in action_argument_set]
        extra_argument_types = [
            record.objects.get(
                argument,
                predicate_schema.parameter_types[index] if index < len(predicate_schema.parameter_types) else "object",
            )
            for index, argument in enumerate(literal_arguments)
            if argument not in action_argument_set
        ]
        return ActiveObservationExample(
            episode_name=record.episode_name,
            step_index=record.step_index,
            canonical_action_name=record.canonical_action_name,
            effect_bucket=record.effect_bucket,
            success=record.success,
            variant_rank=record.variant_rank,
            predicate_name=observation.predicate_name,
            grounded_literal=observation.grounded_literal,
            action_arguments=list(record.action_arguments),
            action_argument_types=list(record.action_argument_types),
            extra_arguments=extra_arguments,
            extra_argument_types=extra_argument_types,
            ground_truth_value=observation.ground_truth_value,
            observed_value=observation.observed_value,
            current_state=list(condition_mining_current_state),
            source_kind="vlm_confirmed",
        )

    def _synthesize_schema(
        self,
        *,
        condition: ActiveObservationCondition,
        predicate_schema: PredicateSchema,
        examples: list[ActiveObservationExample],
    ) -> ActiveObservationRuleSchema:
        observable_name = f"obs_{condition.predicate_name}"
        last_action_constant = last_action_constant_name(
            condition.canonical_action_name,
            success=condition.success,
            effect_bucket=condition.effect_bucket,
            variant_rank=condition.variant_rank,
        )
        last_action_predicate = last_action_predicate_name(len(condition.action_argument_types))
        true_examples = [item for item in examples if item.ground_truth_value]
        false_examples = [item for item in examples if not item.ground_truth_value]
        observed_true_when_true = sum(1 for item in true_examples if item.observed_value)
        observed_false_when_true = len(true_examples) - observed_true_when_true
        observed_true_when_false = sum(1 for item in false_examples if item.observed_value)
        observed_false_when_false = len(false_examples) - observed_true_when_false
        prob_true_given_true = observed_true_when_true / len(true_examples) if true_examples else 0.0
        prob_false_given_true = observed_false_when_true / len(true_examples) if true_examples else 1.0
        prob_true_given_false = observed_true_when_false / len(false_examples) if false_examples else 0.0
        prob_false_given_false = observed_false_when_false / len(false_examples) if false_examples else 1.0
        rule_prefix = (
            f"active_{condition.canonical_action_name}_{'success' if condition.success else 'fail'}_"
            f"{condition.variant_rank}_{condition.predicate_name}"
        )
        return ActiveObservationRuleSchema(
            canonical_action_name=condition.canonical_action_name,
            effect_bucket=condition.effect_bucket,
            success=condition.success,
            variant_rank=condition.variant_rank,
            predicate_name=condition.predicate_name,
            observable_name=observable_name,
            last_action_constant=last_action_constant,
            last_action_predicate_name=last_action_predicate,
            action_argument_types=list(condition.action_argument_types),
            extra_argument_types=list(condition.extra_argument_types),
            last_action_predicate_parameter_types=[
                "last_action_marker",
                *(["object"] * len(condition.action_argument_types)),
            ],
            target_predicate_parameter_types=list(predicate_schema.parameter_types),
            target_literal_template=condition.target_literal_template,
            condition_literals=list(condition.condition_literals),
            true_rule_name=f"{rule_prefix}_true",
            false_rule_name=f"{rule_prefix}_false",
            total_ground_truth_true=len(true_examples),
            observed_true_when_ground_truth_true=observed_true_when_true,
            observed_false_when_ground_truth_true=observed_false_when_true,
            prob_observable_true_given_ground_truth_true=prob_true_given_true,
            prob_observable_false_given_ground_truth_true=prob_false_given_true,
            total_ground_truth_false=len(false_examples),
            observed_true_when_ground_truth_false=observed_true_when_false,
            observed_false_when_ground_truth_false=observed_false_when_false,
            prob_observable_true_given_ground_truth_false=prob_true_given_false,
            prob_observable_false_given_ground_truth_false=prob_false_given_false,
        )

    def write_outputs(self, result: ActiveObservationLearningResult, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "active_observation_learning_summary.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_path / "active_observation_module.pddl").write_text(
            result.rendered_module_text,
            encoding="utf-8",
        )
        (output_path / "active_observation_schemas.json").write_text(
            json.dumps([item.to_dict() for item in result.schemas], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_path / "active_observation_rule_statistics.json").write_text(
            json.dumps(
                {
                    key: value
                    for item in result.schemas
                    for key, value in {
                        item.true_rule_name: {
                            "total_ground_truth_true": item.total_ground_truth_true,
                            "observed_true_when_ground_truth_true": item.observed_true_when_ground_truth_true,
                            "observed_false_when_ground_truth_true": item.observed_false_when_ground_truth_true,
                        },
                        item.false_rule_name: {
                            "total_ground_truth_false": item.total_ground_truth_false,
                            "observed_true_when_ground_truth_false": item.observed_true_when_ground_truth_false,
                            "observed_false_when_ground_truth_false": item.observed_false_when_ground_truth_false,
                        },
                    }.items()
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_path / "active_observation_evidence.jsonl").write_text(
            "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in result.confirmed_examples),
            encoding="utf-8",
        )

    @staticmethod
    def _target_argument_templates(example: ActiveObservationExample) -> list[str]:
        arguments = _literal_arguments(example.grounded_literal)
        action_binding = {argument: f"?arg{index}" for index, argument in enumerate(example.action_arguments)}
        extra_binding = {argument: f"?obs{index}" for index, argument in enumerate(example.extra_arguments)}
        binding = {**action_binding, **extra_binding}
        return [binding.get(argument, argument) for argument in arguments]

    @staticmethod
    def _condition_holds(
        *,
        current_state: list[str],
        condition_literals: list[str],
        action_arguments: list[str],
        extra_arguments: list[str],
    ) -> bool:
        action_binding = {f"?arg{index}": argument for index, argument in enumerate(action_arguments)}
        extra_binding = {f"?obs{index}": argument for index, argument in enumerate(extra_arguments)}
        binding = {**action_binding, **extra_binding}
        grounded_state = set(current_state)
        for literal in condition_literals:
            grounded_literal = literal
            for placeholder, value in binding.items():
                grounded_literal = grounded_literal.replace(placeholder, value)
            if grounded_literal not in grounded_state:
                return False
        return True

    @staticmethod
    def _target_position_mapping(
        record: ActiveObservationSourceRecord,
        predicate_name: str,
        predicate_schema: PredicateSchema,
        parent_by_type: dict[str, str],
    ) -> dict[str, list[int]]:
        del predicate_name
        action_positions: list[int] = []
        used_action_indices: set[int] = set()
        for position, required_type in enumerate(predicate_schema.parameter_types):
            for action_index, action_type in enumerate(record.action_argument_types):
                if action_index in used_action_indices:
                    continue
                if _object_matches_type(action_type, required_type, parent_by_type):
                    action_positions.append(position)
                    used_action_indices.add(action_index)
                    break
        extra_positions = [
            index for index in range(len(predicate_schema.parameter_types)) if index not in action_positions
        ]
        return {"action_positions": action_positions, "extra_positions": extra_positions}


def load_active_observation_inputs(
    *,
    manipulation_artifact_dir: str | Path,
    scene_description_dir: str | Path,
    episode_grounding_pairs: list[tuple[str | Path, str | Path]],
) -> tuple[list[ActiveObservationSourceRecord], list[PredicateSchema], dict[str, str], dict[str, str]]:
    artifact_path = Path(manipulation_artifact_dir)
    scene_root = Path(scene_description_dir)
    predicate_inventory = [PredicateSchema(**row) for row in load_json(artifact_path / "predicate_inventory.json")]
    predicate_comments = dict(load_json_object(artifact_path / "predicate_comments.json"))
    camera_order_by_episode = _load_camera_order_by_episode(scene_root)
    object_type_rows = load_json(artifact_path / "object_types.json")
    parent_by_type = build_type_parent_map(object_type_rows)
    raw_steps = {
        (item.episode_name, item.step_index): item
        for item in load_raw_trajectory_steps(scene_root)
        if item.action_text is not None
    }
    taxonomy_records = {
        (item["episode_name"], int(item["step_index"])): ActionTaxonomyRecord(**item)
        for item in load_jsonl(artifact_path / "action_taxonomy.jsonl")
        if isinstance(item, dict)
    }
    source_records: list[ActiveObservationSourceRecord] = []
    for episode_file, grounding_dir in episode_grounding_pairs:
        episode_name = str(Path(episode_file).parent.name).strip()
        if not episode_name:
            continue
        grounding_path = Path(grounding_dir)
        problem_spec = load_json_object(grounding_path / "problem_summary.json")
        validation_payload = load_json_object(grounding_path / "validation_report.json")
        grounded_steps = [
            item for item in load_jsonl(grounding_path / "grounded_trajectory.jsonl") if isinstance(item, dict)
        ]
        objects = {
            str(item["name"]): str(item.get("type_name") or "object")
            for item in problem_spec.get("objects", [])
            if isinstance(item, dict) and item.get("name")
        }
        validation_by_step = {
            int(item["step_index"]): item
            for item in validation_payload.get("steps", [])
            if isinstance(item, dict) and "step_index" in item
        }
        for grounded_step in grounded_steps:
            step_index = int(grounded_step.get("step_index", -1))
            raw_step = raw_steps.get((episode_name, step_index))
            taxonomy_record = taxonomy_records.get((episode_name, step_index))
            if raw_step is None or taxonomy_record is None:
                continue
            action_category = str(grounded_step.get("action_category") or taxonomy_record.action_category or "").strip()
            if action_category != "active_observation":
                continue
            validation_row = validation_by_step.get(step_index, {})
            success = grounded_step.get("success")
            if success is None:
                success = not _is_failure_extra_info(raw_step.extra_info)
            effect_bucket = str(
                grounded_step.get("effect_bucket")
                or f"{taxonomy_record.canonical_action_name}_{'success' if success else 'fail'}"
            )
            current_state = [str(item) for item in validation_row.get("state_before", [])]
            if not current_state:
                current_state = [str(item) for item in validation_row.get("state_after", [])]
            source_records.append(
                ActiveObservationSourceRecord(
                    episode_name=episode_name,
                    step_index=step_index,
                    instruction=raw_step.instruction,
                    raw_action_text=str(
                        grounded_step.get("raw_action_text")
                        or raw_step.action_text
                        or taxonomy_record.raw_action_text
                        or ""
                    ),
                    canonical_action_name=str(
                        grounded_step.get("canonical_action_name") or taxonomy_record.canonical_action_name or ""
                    ),
                    effect_bucket=effect_bucket,
                    success=bool(success),
                    variant_rank=_bucket_variant_rank(effect_bucket),
                    action_arguments=[
                        str(item) for item in grounded_step.get("ground_arguments", taxonomy_record.action_arguments)
                    ],
                    action_argument_types=list(taxonomy_record.action_argument_types),
                    previous_scene_description=raw_step.previous_observation_text,
                    current_scene_description=raw_step.observation_text,
                    current_state=list(current_state),
                    ground_truth_facts=[str(item) for item in validation_row.get("state_after", current_state)],
                    frame_paths=list(raw_step.frame_paths),
                    objects=dict(objects),
                    effect_predicate_names=[],
                    camera_order_top_to_bottom=list(camera_order_by_episode.get(str(episode_name), [])),
                )
            )
    return source_records, predicate_inventory, predicate_comments, parent_by_type
