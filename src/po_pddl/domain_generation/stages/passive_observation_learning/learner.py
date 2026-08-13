from __future__ import annotations

import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from po_pddl.core.conventions import (
    CONTAINMENT_CONTAINER_INDEX,
    CONTAINMENT_MOVABLE_INDEX,
    containment_arguments,
)
from po_pddl.domain_generation.infrastructure.artifact_io import load_json, load_json_object, load_jsonl
from po_pddl.domain_generation.infrastructure.fact_utils import parse_symbolic_literal
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
    ConfirmedContradiction,
    DescriptionReviewResult,
    PassiveObservationCondition,
    PassiveObservationExample,
    PassiveObservationLearningResult,
    PassiveObservationRuleSchema,
    PassiveObservationSourceRecord,
    VisibleContainedObjectsResult,
    VLMConfirmationResult,
)
from .modules import (
    ContainableRevealActionSelectionModule,
    DescriptionContradictionReviewModule,
    VisibleContainedObjectsModule,
    VLMContradictionConfirmationModule,
)
from .renderer import render_passive_observation_module

EXCLUDED_OBSERVATION_PREDICATES = {"gripper_empty", "gripper_holding"}
logger = logging.getLogger(__name__)


def _surface_name(name: str) -> str:
    return str(name).replace("_", " ").strip().lower()


def _bucket_variant_rank(effect_bucket: str) -> int:
    suffix = str(effect_bucket).rsplit("_bucket_", 1)
    if len(suffix) == 2 and suffix[1].isdigit():
        return int(suffix[1])
    return 0


def _filter_state_before_for_action_scope(
    state_before: list[str],
    action_arguments: list[str],
    effect_predicate_names: list[str] | None = None,
) -> list[str]:
    action_argument_set = set(action_arguments)
    effect_predicate_set = {str(name).strip() for name in (effect_predicate_names or []) if str(name).strip()}
    filtered: list[str] = []
    for fact in state_before:
        negated, predicate, arguments = parse_symbolic_literal(fact)
        del negated
        if predicate in EXCLUDED_OBSERVATION_PREDICATES:
            continue
        if predicate in effect_predicate_set:
            continue
        if not arguments:
            filtered.append(fact)
            continue
        if any(argument in action_argument_set for argument in arguments):
            filtered.append(fact)
    return sorted(dict.fromkeys(filtered))


def _filter_observation_relevant_facts(facts: list[str]) -> list[str]:
    filtered: list[str] = []
    for fact in facts:
        _negated, predicate, _arguments = parse_symbolic_literal(fact)
        if predicate in EXCLUDED_OBSERVATION_PREDICATES:
            continue
        filtered.append(fact)
    return sorted(dict.fromkeys(filtered))


def _exclude_direct_effect_assignments(
    assignments: list[str],
    effect_predicate_names: list[str],
) -> list[str]:
    changed_predicates = {str(name).strip() for name in effect_predicate_names if str(name).strip()}
    return [fact for fact in assignments if _predicate_name(fact) not in changed_predicates]


def _filter_state_before_for_condition_mining(
    state_before: list[str],
    effect_predicate_names: list[str] | None = None,
) -> list[str]:
    effect_predicate_set = {str(name).strip() for name in (effect_predicate_names or []) if str(name).strip()}
    filtered: list[str] = []
    for fact in state_before:
        _negated, predicate, _arguments = parse_symbolic_literal(fact)
        if predicate in EXCLUDED_OBSERVATION_PREDICATES:
            continue
        if predicate in effect_predicate_set:
            continue
        filtered.append(fact)
    return sorted(dict.fromkeys(filtered))


def _build_complete_ground_truth_assignments(
    *,
    objects: dict[str, str],
    predicate_inventory: list[PredicateSchema],
    true_facts: list[str],
    parent_by_type: dict[str, str],
) -> list[str]:
    filtered_true_facts = set(_filter_observation_relevant_facts(true_facts))
    assignments: list[str] = []
    for predicate in predicate_inventory:
        if predicate.predicate_name in EXCLUDED_OBSERVATION_PREDICATES:
            continue
        parameter_types = list(predicate.parameter_types)
        if not parameter_types:
            positive_literal = f"{predicate.predicate_name}()"
            assignments.append(
                positive_literal if positive_literal in filtered_true_facts else f"not {positive_literal}"
            )
            continue
        candidate_object_lists: list[list[str]] = []
        for required_type in parameter_types:
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


def _predicate_name(literal: str) -> str:
    return parse_symbolic_literal(literal)[1]


def _literal_arguments(literal: str) -> list[str]:
    return parse_symbolic_literal(literal)[2]


def _object_matches_type(
    object_type: str,
    required_type: str,
    parent_by_type: dict[str, str],
) -> bool:
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


def _instantiate_template(
    predicate_name: str,
    parameter_names: list[str],
) -> str:
    return f"{predicate_name}({','.join(parameter_names)})" if parameter_names else f"{predicate_name}()"


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
class PassiveObservationLearner:
    description_review_module: DescriptionContradictionReviewModule
    vlm_confirmation_module: VLMContradictionConfirmationModule
    containable_reveal_action_selection_module: ContainableRevealActionSelectionModule | None = None
    visible_contained_objects_module: VisibleContainedObjectsModule | None = None
    manipulation_artifact_dir: str | Path | None = None
    scene_description_dir: str | Path | None = None
    max_workers: int = 1

    def learn_from_source_records(
        self,
        *,
        source_records: list[PassiveObservationSourceRecord],
        predicate_inventory: list[PredicateSchema],
        predicate_comments: dict[str, str],
        type_parents: dict[str, str] | None = None,
    ) -> PassiveObservationLearningResult:
        parent_by_type = dict(type_parents or {})
        observation_predicate_inventory = [
            item for item in predicate_inventory if item.predicate_name not in EXCLUDED_OBSERVATION_PREDICATES
        ]
        filtered_predicate_comments = {
            name: comment for name, comment in predicate_comments.items() if name not in EXCLUDED_OBSERVATION_PREDICATES
        }
        predicate_by_name = {item.predicate_name: item for item in observation_predicate_inventory}
        feature_predicate_names = [
            item.predicate_name for item in observation_predicate_inventory if item.predicate_kind == "feature"
        ]
        review_results_by_variant: dict[str, dict[str, Any]] = {}
        confirmed_contradiction_literals_by_record: dict[tuple[str, int, str], set[str]] = defaultdict(set)

        records_by_variant: dict[tuple[str, str], list[PassiveObservationSourceRecord]] = defaultdict(list)
        for record in source_records:
            records_by_variant[(record.canonical_action_name, record.effect_bucket)].append(record)

        seed_examples: list[PassiveObservationExample] = []
        grouped_seed_examples: dict[tuple[str, str, str], list[PassiveObservationExample]] = defaultdict(list)
        ordered_records = sorted(
            source_records,
            key=lambda item: (item.canonical_action_name, item.effect_bucket, item.episode_name, item.step_index),
        )
        review_payloads: list[dict[str, Any]] = [None] * len(ordered_records)  # type: ignore[list-item]
        if ordered_records:
            worker_count = max(1, min(self.max_workers, len(ordered_records)))
            if worker_count == 1:
                for index, record in enumerate(ordered_records):
                    review_payloads[index] = self._review_single_record(
                        record=record,
                        predicate_inventory=observation_predicate_inventory,
                        parent_by_type=parent_by_type,
                        predicate_comments=filtered_predicate_comments,
                        feature_predicate_names=feature_predicate_names,
                    )
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_map = {
                        executor.submit(
                            self._review_single_record,
                            record=record,
                            predicate_inventory=observation_predicate_inventory,
                            parent_by_type=parent_by_type,
                            predicate_comments=filtered_predicate_comments,
                            feature_predicate_names=feature_predicate_names,
                        ): index
                        for index, record in enumerate(ordered_records)
                    }
                    for future in as_completed(future_map):
                        index = future_map[future]
                        review_payloads[index] = future.result()

        for payload in review_payloads:
            if payload is None:
                continue
            record = payload["record"]
            variant_review_key = f"{record.canonical_action_name}::{record.effect_bucket}"
            review_results_by_variant.setdefault(
                variant_review_key,
                {
                    "record_reviews": [],
                },
            )["record_reviews"].append(payload["review_summary"])
            for contradiction in payload["confirmed_review"].confirmed_contradictions:
                confirmed_contradiction_literals_by_record[
                    (record.episode_name, record.step_index, contradiction.predicate_name)
                ].add(contradiction.grounded_literal)
                predicate_schema = predicate_by_name.get(contradiction.predicate_name)
                if predicate_schema is None:
                    continue
                if contradiction.predicate_name == "in":
                    # `in(...)` is learned only by the dedicated reveal-action branch.
                    continue
                example = self._build_seed_example(
                    record=record,
                    condition_mining_state_before=payload["condition_mining_state_before"],
                    contradiction=contradiction,
                    predicate_schema=predicate_schema,
                )
                seed_examples.append(example)
                grouped_seed_examples[
                    (
                        example.canonical_action_name,
                        example.effect_bucket,
                        example.predicate_name,
                    )
                ].append(example)

        for variant_key, variant_records in records_by_variant.items():
            summary = review_results_by_variant.setdefault(
                variant_key[0] + "::" + variant_key[1], {"record_reviews": []}
            )
            summary["total_records"] = len(variant_records)
            summary["seed_example_count"] = sum(
                1
                for item in seed_examples
                if item.canonical_action_name == variant_key[0] and item.effect_bucket == variant_key[1]
            )

        conditions: list[PassiveObservationCondition] = []
        for group_key, group_examples in sorted(grouped_seed_examples.items()):
            predicate_schema = predicate_by_name[group_key[2]]
            conditions.append(
                self._mine_condition(
                    examples=group_examples,
                    predicate_schema=predicate_schema,
                )
            )

        expanded_examples: list[PassiveObservationExample] = []
        specialized_examples_by_group: dict[tuple[str, str, str], list[PassiveObservationExample]] = {}
        specialized_conditions_by_group: dict[tuple[str, str, str], PassiveObservationCondition] = {}
        reveal_action_names: set[str] = set()
        if (
            "in" in predicate_by_name
            and self.containable_reveal_action_selection_module is not None
            and self.visible_contained_objects_module is not None
        ):
            (
                reveal_action_names,
                specialized_conditions_by_group,
                specialized_examples_by_group,
                specialized_review_summary,
            ) = self._learn_specialized_in_examples(
                records_by_variant=records_by_variant,
                predicate_schema=predicate_by_name["in"],
                parent_by_type=parent_by_type,
                predicate_comments=filtered_predicate_comments,
            )
            for variant_key, summary in specialized_review_summary.items():
                review_results_by_variant.setdefault(variant_key, {}).update(summary)
            if specialized_conditions_by_group:
                # For actions that reveal container interiors, we only keep the dedicated
                # `in(...)` passive observation learning results and suppress generic
                # passive rules such as `open(...)`.
                grouped_seed_examples = defaultdict(
                    list,
                    {
                        key: value
                        for key, value in grouped_seed_examples.items()
                        if not (key[0] in reveal_action_names and key[2] != "in")
                    },
                )
                conditions = [
                    item
                    for item in conditions
                    if (
                        (item.canonical_action_name, item.effect_bucket, item.predicate_name)
                        not in specialized_conditions_by_group
                        and not (item.canonical_action_name in reveal_action_names and item.predicate_name != "in")
                    )
                ]
                conditions.extend(specialized_conditions_by_group.values())

        for condition in conditions:
            group_key = (condition.canonical_action_name, condition.effect_bucket, condition.predicate_name)
            if group_key in specialized_examples_by_group:
                expanded_examples.extend(specialized_examples_by_group[group_key])
                continue
            predicate_schema = predicate_by_name[condition.predicate_name]
            for record in records_by_variant[(condition.canonical_action_name, condition.effect_bucket)]:
                expanded_examples.extend(
                    self._expand_examples_for_record(
                        record=record,
                        condition=condition,
                        predicate_schema=predicate_schema,
                        parent_by_type=parent_by_type,
                        confirmed_contradiction_literals=confirmed_contradiction_literals_by_record.get(
                            (record.episode_name, record.step_index, condition.predicate_name),
                            set(),
                        ),
                    )
                )

        schemas: list[PassiveObservationRuleSchema] = []
        for condition in conditions:
            predicate_schema = predicate_by_name[condition.predicate_name]
            group_key = (condition.canonical_action_name, condition.effect_bucket, condition.predicate_name)
            if group_key in specialized_examples_by_group:
                related_examples = list(specialized_examples_by_group[group_key])
            else:
                related_examples = [
                    item
                    for item in [*seed_examples, *expanded_examples]
                    if (
                        item.canonical_action_name == condition.canonical_action_name
                        and item.effect_bucket == condition.effect_bucket
                        and item.predicate_name == condition.predicate_name
                    )
                ]
            schemas.append(
                self._synthesize_schema(
                    condition=condition,
                    predicate_schema=predicate_schema,
                    examples=related_examples,
                )
            )

        rendered_module_text = render_passive_observation_module(schemas)
        return PassiveObservationLearningResult(
            source_records=source_records,
            review_results_by_variant=review_results_by_variant,
            seed_examples=seed_examples,
            conditions=conditions,
            expanded_examples=expanded_examples,
            schemas=schemas,
            predicate_inventory=observation_predicate_inventory,
            predicate_comments=filtered_predicate_comments,
            rendered_module_text=rendered_module_text,
        )

    def _review_single_record(
        self,
        *,
        record: PassiveObservationSourceRecord,
        predicate_inventory: list[PredicateSchema],
        parent_by_type: dict[str, str],
        predicate_comments: dict[str, str],
        feature_predicate_names: list[str],
    ) -> dict[str, Any]:
        logger.debug(
            "Passive observation review started: episode=%s step=%d action=%s bucket=%s",
            record.episode_name,
            record.step_index,
            record.canonical_action_name,
            record.effect_bucket,
        )
        filtered_state_before = _filter_state_before_for_action_scope(
            record.state_before,
            record.action_arguments,
            record.effect_predicate_names,
        )
        if not filtered_state_before:
            description_review = DescriptionReviewResult(
                should_review_with_vlm=False,
                suspect_contradictions=[],
                summary="Skipped because all state-before predicates were filtered out.",
                raw_output=None,
            )
            confirmed_review = VLMConfirmationResult(
                confirmed_contradictions=[],
                summary="Skipped because all state-before predicates were filtered out.",
                raw_output=None,
            )
        else:
            candidate_ground_truth_facts = _build_complete_ground_truth_assignments(
                objects=record.objects,
                predicate_inventory=predicate_inventory,
                true_facts=list(record.ground_truth_facts),
                parent_by_type=parent_by_type,
            )
            candidate_ground_truth_facts = _exclude_direct_effect_assignments(
                candidate_ground_truth_facts,
                record.effect_predicate_names,
            )
            description_review = self.description_review_module.review(
                record=record,
                filtered_state_before=filtered_state_before,
                candidate_ground_truth_facts=candidate_ground_truth_facts,
                predicate_comments=predicate_comments,
                feature_predicate_names=feature_predicate_names,
            )
            confirmed_review = VLMConfirmationResult(confirmed_contradictions=[], summary="No suspect contradictions.")
            if description_review.should_review_with_vlm and description_review.suspect_contradictions:
                confirmed_review = self.vlm_confirmation_module.confirm(
                    record=record,
                    filtered_state_before=filtered_state_before,
                    candidate_ground_truth_facts=candidate_ground_truth_facts,
                    suspects=description_review.suspect_contradictions,
                    predicate_comments=predicate_comments,
                )
        payload = {
            "record": record,
            "filtered_state_before": list(filtered_state_before),
            "condition_mining_state_before": _filter_state_before_for_condition_mining(
                record.state_before,
                record.effect_predicate_names,
            ),
            "description_review": description_review,
            "confirmed_review": confirmed_review,
            "review_summary": {
                "episode_name": record.episode_name,
                "step_index": record.step_index,
                "description_review": description_review.to_dict(),
                "vlm_confirmation": confirmed_review.to_dict(),
            },
        }
        logger.debug(
            "Passive observation review completed: episode=%s step=%d action=%s bucket=%s",
            record.episode_name,
            record.step_index,
            record.canonical_action_name,
            record.effect_bucket,
        )
        return payload

    def _learn_specialized_in_examples(
        self,
        *,
        records_by_variant: dict[tuple[str, str], list[PassiveObservationSourceRecord]],
        predicate_schema: PredicateSchema,
        parent_by_type: dict[str, str],
        predicate_comments: dict[str, str],
    ) -> tuple[
        set[str],
        dict[tuple[str, str, str], PassiveObservationCondition],
        dict[tuple[str, str, str], list[PassiveObservationExample]],
        dict[str, dict[str, Any]],
    ]:
        assert self.containable_reveal_action_selection_module is not None
        assert self.visible_contained_objects_module is not None

        action_summaries = self._build_containable_action_summaries(records_by_variant)
        containable_type_names = self._containable_type_names(records_by_variant, parent_by_type, predicate_schema)
        selection = self.containable_reveal_action_selection_module.select_actions(
            action_summaries=action_summaries,
            predicate_comments=predicate_comments,
            containable_type_names=containable_type_names,
        )
        selected_actions = {name for name in selection.action_names if str(name).strip()}
        if not selected_actions:
            return set(), {}, {}, {}

        review_summary: dict[str, dict[str, Any]] = {}
        jobs: list[tuple[PassiveObservationSourceRecord, str, list[str]]] = []
        for (action_name, _effect_bucket), variant_records in records_by_variant.items():
            if action_name not in selected_actions:
                continue
            for record in variant_records:
                if not record.success:
                    continue
                container_names = self._matching_container_arguments(
                    record=record,
                    containable_type=predicate_schema.parameter_types[CONTAINMENT_CONTAINER_INDEX],
                    parent_by_type=parent_by_type,
                )
                candidate_object_names = sorted(
                    object_name
                    for object_name, object_type in record.objects.items()
                    if _object_matches_type(
                        object_type,
                        predicate_schema.parameter_types[CONTAINMENT_MOVABLE_INDEX],
                        parent_by_type,
                    )
                )
                for container_name in container_names:
                    jobs.append((record, container_name, candidate_object_names))

        visible_results: list[tuple[PassiveObservationSourceRecord, str, list[str], VisibleContainedObjectsResult]] = []
        if jobs:
            worker_count = max(1, min(self.max_workers, len(jobs)))
            if worker_count == 1:
                for record, container_name, candidate_object_names in jobs:
                    visible_results.append(
                        (
                            record,
                            container_name,
                            candidate_object_names,
                            self.visible_contained_objects_module.identify_visible_objects(
                                record=record,
                                container_object_name=container_name,
                                candidate_object_names=candidate_object_names,
                                predicate_comments=predicate_comments,
                            ),
                        )
                    )
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_map = {
                        executor.submit(
                            self.visible_contained_objects_module.identify_visible_objects,
                            record=record,
                            container_object_name=container_name,
                            candidate_object_names=candidate_object_names,
                            predicate_comments=predicate_comments,
                        ): (record, container_name, candidate_object_names)
                        for record, container_name, candidate_object_names in jobs
                    }
                    for future in as_completed(future_map):
                        record, container_name, candidate_object_names = future_map[future]
                        visible_results.append((record, container_name, candidate_object_names, future.result()))

        examples_by_group: dict[tuple[str, str, str], list[PassiveObservationExample]] = defaultdict(list)
        for record, container_name, candidate_object_names, visible_result in visible_results:
            variant_key = f"{record.canonical_action_name}::{record.effect_bucket}"
            review_summary.setdefault(
                variant_key,
                {
                    "containable_reveal_action_selection": selection.to_dict(),
                    "specialized_in_visible_object_reviews": [],
                },
            )["specialized_in_visible_object_reviews"].append(
                {
                    "episode_name": record.episode_name,
                    "step_index": record.step_index,
                    "container_object_name": container_name,
                    "candidate_object_names": list(candidate_object_names),
                    "visible_object_names": list(visible_result.visible_object_names),
                    "summary": visible_result.summary,
                }
            )
            visible_names = set(visible_result.visible_object_names)
            condition_state_before = _filter_state_before_for_condition_mining(
                record.state_before,
                record.effect_predicate_names,
            )
            ground_truth_set = set(record.ground_truth_facts)
            for object_name in candidate_object_names:
                if object_name == container_name:
                    continue
                grounded_literal = _instantiate_template(
                    predicate_schema.predicate_name,
                    containment_arguments(object_name, container_name),
                )
                ground_truth_value = grounded_literal in ground_truth_set
                observed_value = object_name in visible_names
                examples_by_group[
                    (record.canonical_action_name, record.effect_bucket, predicate_schema.predicate_name)
                ].append(
                    PassiveObservationExample(
                        episode_name=record.episode_name,
                        step_index=record.step_index,
                        canonical_action_name=record.canonical_action_name,
                        effect_bucket=record.effect_bucket,
                        success=record.success,
                        variant_rank=record.variant_rank,
                        predicate_name=predicate_schema.predicate_name,
                        grounded_literal=grounded_literal,
                        action_arguments=list(record.action_arguments),
                        action_argument_types=list(record.action_argument_types),
                        extra_arguments=[object_name],
                        extra_argument_types=[
                            record.objects.get(
                                object_name,
                                predicate_schema.parameter_types[CONTAINMENT_MOVABLE_INDEX],
                            )
                        ],
                        ground_truth_value=ground_truth_value,
                        observed_value=observed_value,
                        state_before=list(condition_state_before),
                        source_kind="specialized_in_observation",
                    )
                )

        conditions_by_group: dict[tuple[str, str, str], PassiveObservationCondition] = {}
        filtered_examples_by_group: dict[tuple[str, str, str], list[PassiveObservationExample]] = {}
        for group_key, examples in examples_by_group.items():
            positive_observation_examples = [item for item in examples if item.observed_value]
            condition_seed_examples = positive_observation_examples or [
                item for item in examples if item.ground_truth_value
            ]
            if not condition_seed_examples:
                continue
            condition = self._mine_condition(
                examples=condition_seed_examples,
                predicate_schema=predicate_schema,
            )
            conditions_by_group[group_key] = condition
            filtered_examples_by_group[group_key] = [
                item
                for item in examples
                if self._condition_holds(
                    state_before=item.state_before,
                    condition_literals=condition.condition_literals,
                    action_arguments=item.action_arguments,
                    extra_arguments=item.extra_arguments,
                )
            ]
        return selected_actions, conditions_by_group, filtered_examples_by_group, review_summary

    def learn_from_pairs(
        self,
        *,
        domain_file: str | Path,
        episode_grounding_pairs: list[tuple[str | Path, str | Path]],
    ) -> PassiveObservationLearningResult:
        del domain_file
        if self.manipulation_artifact_dir is None or self.scene_description_dir is None:
            raise ValueError(
                "PassiveObservationLearner requires manipulation_artifact_dir and scene_description_dir "
                "to learn from pipeline episode-grounding pairs."
            )
        source_records, predicate_inventory, predicate_comments, parent_by_type = load_passive_observation_inputs(
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

    def _build_seed_example(
        self,
        *,
        record: PassiveObservationSourceRecord,
        condition_mining_state_before: list[str],
        contradiction: ConfirmedContradiction,
        predicate_schema: PredicateSchema,
    ) -> PassiveObservationExample:
        literal_arguments = _literal_arguments(contradiction.grounded_literal)
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
        return PassiveObservationExample(
            episode_name=record.episode_name,
            step_index=record.step_index,
            canonical_action_name=record.canonical_action_name,
            effect_bucket=record.effect_bucket,
            success=record.success,
            variant_rank=record.variant_rank,
            predicate_name=contradiction.predicate_name,
            grounded_literal=contradiction.grounded_literal,
            action_arguments=list(record.action_arguments),
            action_argument_types=list(record.action_argument_types),
            extra_arguments=extra_arguments,
            extra_argument_types=extra_argument_types,
            ground_truth_value=contradiction.ground_truth_value,
            observed_value=contradiction.observed_value,
            state_before=list(condition_mining_state_before),
            source_kind="contradiction_seed",
        )

    def _mine_condition(
        self,
        *,
        examples: list[PassiveObservationExample],
        predicate_schema: PredicateSchema,
    ) -> PassiveObservationCondition:
        first = examples[0]
        intersections: set[str] | None = None
        target_argument_templates = self._target_argument_templates(first)
        target_literal_template = _instantiate_template(first.predicate_name, target_argument_templates)

        for example in examples:
            candidate_literals: set[str] = set()
            extra_binding = {argument: f"?obs{index}" for index, argument in enumerate(example.extra_arguments)}
            action_binding = {argument: f"?arg{index}" for index, argument in enumerate(example.action_arguments)}
            binding = {**action_binding, **extra_binding}
            for fact in example.state_before:
                predicate_name = _predicate_name(fact)
                if predicate_name == example.predicate_name:
                    continue
                literal_arguments = _literal_arguments(fact)
                if not example.extra_arguments:
                    continue
                if not any(argument in example.extra_arguments for argument in literal_arguments):
                    continue
                if not all(argument in binding for argument in literal_arguments):
                    continue
                templated = _instantiate_template(
                    predicate_name,
                    [binding[argument] for argument in literal_arguments],
                )
                candidate_literals.add(templated)
            intersections = (
                candidate_literals if intersections is None else intersections.intersection(candidate_literals)
            )

        return PassiveObservationCondition(
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

    @staticmethod
    def _build_containable_action_summaries(
        records_by_variant: dict[tuple[str, str], list[PassiveObservationSourceRecord]],
    ) -> list[dict[str, object]]:
        summaries: list[dict[str, object]] = []
        seen_actions: set[str] = set()
        for (action_name, _effect_bucket), records in sorted(records_by_variant.items()):
            if action_name in seen_actions or not records:
                continue
            seen_actions.add(action_name)
            representative = sorted(records, key=lambda item: (item.episode_name, item.step_index))[0]
            summaries.append(
                {
                    "canonical_action_name": action_name,
                    "sample_action_text": representative.raw_action_text,
                    "action_argument_types": list(representative.action_argument_types),
                    "effect_buckets": sorted({item.effect_bucket for item in records}),
                }
            )
        return summaries

    @staticmethod
    def _containable_type_names(
        records_by_variant: dict[tuple[str, str], list[PassiveObservationSourceRecord]],
        parent_by_type: dict[str, str],
        predicate_schema: PredicateSchema,
    ) -> list[str]:
        containable_type = predicate_schema.parameter_types[CONTAINMENT_CONTAINER_INDEX]
        type_names: set[str] = {containable_type}
        for records in records_by_variant.values():
            for record in records:
                type_names.update(record.action_argument_types)
                type_names.update(record.objects.values())
        return sorted(
            type_name
            for type_name in type_names
            if _object_matches_type(type_name, containable_type, parent_by_type)
        )

    @staticmethod
    def _matching_container_arguments(
        *,
        record: PassiveObservationSourceRecord,
        containable_type: str,
        parent_by_type: dict[str, str],
    ) -> list[str]:
        matches: list[str] = []
        for argument_name, argument_type in zip(record.action_arguments, record.action_argument_types, strict=False):
            if _object_matches_type(argument_type, containable_type, parent_by_type):
                matches.append(argument_name)
        return matches

    def _expand_examples_for_record(
        self,
        *,
        record: PassiveObservationSourceRecord,
        condition: PassiveObservationCondition,
        predicate_schema: PredicateSchema,
        parent_by_type: dict[str, str],
        confirmed_contradiction_literals: set[str],
    ) -> list[PassiveObservationExample]:
        position_mapping = self._target_position_mapping(record, condition.predicate_name, predicate_schema)
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

        expanded: list[PassiveObservationExample] = []
        for extra_arguments in candidate_extra_bindings:
            target_arguments: list[str] = []
            action_index_by_position = {position: index for index, position in enumerate(action_positions)}
            extra_index_by_position = {position: index for index, position in enumerate(extra_positions)}
            for position in range(len(predicate_schema.parameter_types)):
                if position in action_index_by_position:
                    target_arguments.append(record.action_arguments[action_index_by_position[position]])
                else:
                    target_arguments.append(extra_arguments[extra_index_by_position[position]])
            grounded_literal = _instantiate_template(condition.predicate_name, target_arguments)
            if grounded_literal in confirmed_contradiction_literals:
                continue
            condition_state_before = _filter_state_before_for_condition_mining(
                record.state_before,
                record.effect_predicate_names,
            )
            if not self._condition_holds(
                state_before=condition_state_before,
                condition_literals=condition.condition_literals,
                action_arguments=record.action_arguments,
                extra_arguments=extra_arguments,
            ):
                continue
            ground_truth_value = grounded_literal in set(record.ground_truth_facts)
            expanded.append(
                PassiveObservationExample(
                    episode_name=record.episode_name,
                    step_index=record.step_index,
                    canonical_action_name=record.canonical_action_name,
                    effect_bucket=record.effect_bucket,
                    success=record.success,
                    variant_rank=record.variant_rank,
                    predicate_name=condition.predicate_name,
                    grounded_literal=grounded_literal,
                    action_arguments=list(record.action_arguments),
                    action_argument_types=list(record.action_argument_types),
                    extra_arguments=list(extra_arguments),
                    extra_argument_types=list(extra_parameter_types),
                    ground_truth_value=ground_truth_value,
                    observed_value=ground_truth_value,
                    state_before=list(condition_state_before),
                    source_kind="expanded_correct_observation",
                )
            )
        return expanded

    def _synthesize_schema(
        self,
        *,
        condition: PassiveObservationCondition,
        predicate_schema: PredicateSchema,
        examples: list[PassiveObservationExample],
    ) -> PassiveObservationRuleSchema:
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
        if condition.predicate_name == "in":
            # For containment, when ground truth is false we deterministically emit
            # not-obs_in instead of estimating this side from sparse data.
            observed_true_when_false = 0
            observed_false_when_false = len(false_examples)
            prob_true_given_false = 0.0
            prob_false_given_false = 1.0
        rule_prefix = (
            f"passive_{condition.canonical_action_name}_{'success' if condition.success else 'fail'}_"
            f"{condition.variant_rank}_{condition.predicate_name}"
        )
        return PassiveObservationRuleSchema(
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

    def write_outputs(self, result: PassiveObservationLearningResult, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "passive_observation_learning_summary.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_path / "passive_observation_module.pddl").write_text(
            result.rendered_module_text,
            encoding="utf-8",
        )
        (output_path / "passive_observation_schemas.json").write_text(
            json.dumps([item.to_dict() for item in result.schemas], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_path / "passive_observation_rule_statistics.json").write_text(
            json.dumps(
                {item.true_rule_name: item.to_dict() for item in result.schemas},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_path / "passive_observation_evidence.jsonl").write_text(
            "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in result.seed_examples),
            encoding="utf-8",
        )

    @staticmethod
    def _target_argument_templates(example: PassiveObservationExample) -> list[str]:
        arguments = _literal_arguments(example.grounded_literal)
        action_binding = {argument: f"?arg{index}" for index, argument in enumerate(example.action_arguments)}
        extra_binding = {argument: f"?obs{index}" for index, argument in enumerate(example.extra_arguments)}
        binding = {**action_binding, **extra_binding}
        return [binding.get(argument, argument) for argument in arguments]

    @staticmethod
    def _condition_holds(
        *,
        state_before: list[str],
        condition_literals: list[str],
        action_arguments: list[str],
        extra_arguments: list[str],
    ) -> bool:
        action_binding = {f"?arg{index}": argument for index, argument in enumerate(action_arguments)}
        extra_binding = {f"?obs{index}": argument for index, argument in enumerate(extra_arguments)}
        binding = {**action_binding, **extra_binding}
        grounded_state = set(state_before)
        for literal in condition_literals:
            grounded_literal = literal
            for placeholder, value in binding.items():
                grounded_literal = grounded_literal.replace(placeholder, value)
            if grounded_literal not in grounded_state:
                return False
        return True

    @staticmethod
    def _target_position_mapping(
        record: PassiveObservationSourceRecord,
        predicate_name: str,
        predicate_schema: PredicateSchema,
    ) -> dict[str, list[int]]:
        # Prefer action arguments already appearing in the representative action order.
        action_positions: list[int] = []
        used_action_indices: set[int] = set()
        for position, _type_name in enumerate(predicate_schema.parameter_types):
            argument_value = (
                record.action_arguments[0]
                if len(record.action_arguments) == 1 and position == len(predicate_schema.parameter_types) - 1
                else None
            )
            if argument_value is not None and argument_value == record.action_arguments[0]:
                action_positions.append(position)
                used_action_indices.add(0)
        if not action_positions:
            for position, required_type in enumerate(predicate_schema.parameter_types):
                for action_index, action_type in enumerate(record.action_argument_types):
                    if action_index in used_action_indices:
                        continue
                    if action_type == required_type:
                        action_positions.append(position)
                        used_action_indices.add(action_index)
                        break
        extra_positions = [
            index for index in range(len(predicate_schema.parameter_types)) if index not in action_positions
        ]
        return {
            "action_positions": action_positions,
            "extra_positions": extra_positions,
        }

def load_passive_observation_inputs(
    *,
    manipulation_artifact_dir: str | Path,
    scene_description_dir: str | Path,
    episode_grounding_pairs: list[tuple[str | Path, str | Path]],
) -> tuple[list[PassiveObservationSourceRecord], list[PredicateSchema], dict[str, str], dict[str, str]]:
    artifact_path = Path(manipulation_artifact_dir)
    scene_root = Path(scene_description_dir)
    predicate_inventory = [PredicateSchema(**row) for row in load_json(artifact_path / "predicate_inventory.json")]
    predicate_comments = dict(load_json_object(artifact_path / "predicate_comments.json"))
    camera_order_by_episode = _load_camera_order_by_episode(scene_root)
    object_type_rows = load_json(artifact_path / "object_types.json")
    parent_by_type = {
        str(row["type_name"]): str(row.get("parent_type") or "")
        for row in object_type_rows
        if isinstance(row, dict) and row.get("parent_type")
    }
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
    manipulation_records = {
        (item["episode_name"], int(item["step_index"])): item
        for item in load_jsonl(artifact_path / "manipulation_records.jsonl")
        if isinstance(item, dict) and "episode_name" in item and "step_index" in item
    }
    source_records: list[PassiveObservationSourceRecord] = []
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
            if str(grounded_step.get("action_category") or "").strip() != "manipulation":
                continue
            step_index = int(grounded_step["step_index"])
            raw_step = raw_steps.get((episode_name, step_index))
            taxonomy_record = taxonomy_records.get((episode_name, step_index))
            validation_row = validation_by_step.get(step_index, {})
            manipulation_record = manipulation_records.get((episode_name, step_index), {})
            if raw_step is None or taxonomy_record is None:
                continue
            effect_predicate_names = sorted(
                {
                    _predicate_name(str(literal))
                    for literal in [
                        *(manipulation_record.get("delta_add", []) or []),
                        *(manipulation_record.get("delta_del", []) or []),
                    ]
                    if str(literal).strip()
                }
            )
            source_records.append(
                PassiveObservationSourceRecord(
                    episode_name=episode_name,
                    step_index=step_index,
                    instruction=raw_step.instruction,
                    raw_action_text=str(grounded_step.get("raw_action_text") or raw_step.action_text or ""),
                    canonical_action_name=str(grounded_step.get("canonical_action_name") or ""),
                    effect_bucket=str(grounded_step.get("effect_bucket") or ""),
                    success=bool(grounded_step.get("success", False)),
                    variant_rank=_bucket_variant_rank(str(grounded_step.get("effect_bucket") or "")),
                    action_arguments=[str(item) for item in grounded_step.get("ground_arguments", [])],
                    action_argument_types=list(taxonomy_record.action_argument_types),
                    state_before=[str(item) for item in validation_row.get("state_before", [])],
                    ground_truth_facts=[str(item) for item in validation_row.get("state_after", [])],
                    effect_predicate_names=effect_predicate_names,
                    scene_description=raw_step.observation_text,
                    frame_paths=list(raw_step.frame_paths),
                    objects=dict(objects),
                    camera_order_top_to_bottom=list(camera_order_by_episode.get(str(episode_name), [])),
                )
            )
    return source_records, predicate_inventory, predicate_comments, parent_by_type
