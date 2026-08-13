"""Incremental merging for the three v2 observation models."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

from po_pddl.domain_generation.stages.active_observation_learning.models import (
    ActiveObservationCondition,
    ActiveObservationExample,
    ActiveObservationLearningResult,
    ActiveObservationRuleSchema,
    ActiveObservationSourceRecord,
)
from po_pddl.domain_generation.stages.active_observation_learning.renderer import (
    render_active_observation_module,
)
from po_pddl.domain_generation.stages.init_observation_learning.models import (
    InitObservationCondition,
    InitObservationExample,
    InitObservationLearningResult,
    InitObservationRuleSchema,
    InitObservationSourceRecord,
    InitObservationUncertainPredicateDiscovery,
)
from po_pddl.domain_generation.stages.init_observation_learning.renderer import (
    render_init_observation_module,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.models import PredicateSchema
from po_pddl.domain_generation.stages.passive_observation_learning.models import (
    PassiveObservationCondition,
    PassiveObservationExample,
    PassiveObservationLearningResult,
    PassiveObservationRuleSchema,
    PassiveObservationSourceRecord,
)
from po_pddl.domain_generation.stages.passive_observation_learning.renderer import (
    render_passive_observation_module,
)

SchemaT = TypeVar("SchemaT", PassiveObservationRuleSchema, InitObservationRuleSchema, ActiveObservationRuleSchema)

_COUNT_FIELDS = (
    "total_ground_truth_true",
    "observed_true_when_ground_truth_true",
    "observed_false_when_ground_truth_true",
    "total_ground_truth_false",
    "observed_true_when_ground_truth_false",
    "observed_false_when_ground_truth_false",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _merge_sequence(existing: list[Any], new: list[Any], *, key) -> list[Any]:
    merged: dict[Any, Any] = {key(item): item for item in existing}
    for item in new:
        merged[key(item)] = item
    return list(merged.values())


def _merge_inventory(existing: list[PredicateSchema], new: list[PredicateSchema]) -> list[PredicateSchema]:
    return _merge_sequence(existing, new, key=lambda item: item.predicate_name)


def _merge_counted_schema(existing: SchemaT, new: SchemaT) -> SchemaT:
    counts = {field: int(getattr(existing, field)) + int(getattr(new, field)) for field in _COUNT_FIELDS}
    true_total = counts["total_ground_truth_true"]
    false_total = counts["total_ground_truth_false"]
    probabilities = {
        "prob_observable_true_given_ground_truth_true": (
            counts["observed_true_when_ground_truth_true"] / true_total if true_total else 0.0
        ),
        "prob_observable_false_given_ground_truth_true": (
            counts["observed_false_when_ground_truth_true"] / true_total if true_total else 1.0
        ),
        "prob_observable_true_given_ground_truth_false": (
            counts["observed_true_when_ground_truth_false"] / false_total if false_total else 0.0
        ),
        "prob_observable_false_given_ground_truth_false": (
            counts["observed_false_when_ground_truth_false"] / false_total if false_total else 1.0
        ),
    }
    return replace(existing, **counts, **probabilities)


def _merge_schemas(existing: list[SchemaT], new: list[SchemaT], *, key) -> list[SchemaT]:
    merged: dict[Any, SchemaT] = {key(item): item for item in existing}
    order = [key(item) for item in existing]
    for item in new:
        item_key = key(item)
        if item_key in merged:
            merged[item_key] = _merge_counted_schema(merged[item_key], item)
        else:
            merged[item_key] = item
            order.append(item_key)
    return [merged[item_key] for item_key in order]


def _passive_schema_key(schema: PassiveObservationRuleSchema) -> tuple[Any, ...]:
    return (
        schema.canonical_action_name,
        schema.effect_bucket,
        schema.success,
        schema.variant_rank,
        schema.predicate_name,
        tuple(schema.action_argument_types),
        tuple(schema.extra_argument_types),
        schema.target_literal_template,
    )


def _init_schema_key(schema: InitObservationRuleSchema) -> tuple[Any, ...]:
    return (schema.predicate_name, tuple(schema.parameter_types), schema.target_literal_template)


def _active_schema_key(schema: ActiveObservationRuleSchema) -> tuple[Any, ...]:
    return (
        schema.canonical_action_name,
        schema.effect_bucket,
        schema.success,
        schema.variant_rank,
        schema.predicate_name,
        tuple(schema.action_argument_types),
        tuple(schema.extra_argument_types),
        schema.target_literal_template,
    )


def merge_passive_results(
    existing: PassiveObservationLearningResult | None,
    new: PassiveObservationLearningResult,
) -> PassiveObservationLearningResult:
    if existing is None:
        return new
    schemas = _merge_schemas(existing.schemas, new.schemas, key=_passive_schema_key)
    return PassiveObservationLearningResult(
        source_records=_merge_sequence(
            existing.source_records,
            new.source_records,
            key=lambda item: (item.episode_name, item.step_index, item.canonical_action_name, item.effect_bucket),
        ),
        review_results_by_variant={**existing.review_results_by_variant, **new.review_results_by_variant},
        seed_examples=_merge_sequence(
            existing.seed_examples,
            new.seed_examples,
            key=lambda item: (
                item.episode_name,
                item.step_index,
                item.canonical_action_name,
                item.effect_bucket,
                item.grounded_literal,
                item.source_kind,
            ),
        ),
        conditions=_merge_sequence(
            existing.conditions,
            new.conditions,
            key=lambda item: (
                item.canonical_action_name,
                item.effect_bucket,
                item.predicate_name,
                tuple(item.action_argument_types),
                tuple(item.extra_argument_types),
            ),
        ),
        expanded_examples=_merge_sequence(
            existing.expanded_examples,
            new.expanded_examples,
            key=lambda item: (
                item.episode_name,
                item.step_index,
                item.canonical_action_name,
                item.effect_bucket,
                item.grounded_literal,
                item.source_kind,
            ),
        ),
        schemas=schemas,
        predicate_inventory=_merge_inventory(existing.predicate_inventory, new.predicate_inventory),
        predicate_comments={**existing.predicate_comments, **new.predicate_comments},
        rendered_module_text=render_passive_observation_module(schemas),
    )


def merge_init_results(
    existing: InitObservationLearningResult | None,
    new: InitObservationLearningResult,
) -> InitObservationLearningResult:
    if existing is None:
        return new
    schemas = _merge_schemas(existing.schemas, new.schemas, key=_init_schema_key)
    discovery = new.uncertain_predicate_discovery or existing.uncertain_predicate_discovery
    if existing.uncertain_predicate_discovery and new.uncertain_predicate_discovery:
        predicate_names = sorted(
            set(existing.uncertain_predicate_discovery.predicate_names)
            | set(new.uncertain_predicate_discovery.predicate_names)
        )
        discovery = replace(
            existing.uncertain_predicate_discovery,
            predicate_names=predicate_names,
            rationale_by_predicate={
                **existing.uncertain_predicate_discovery.rationale_by_predicate,
                **new.uncertain_predicate_discovery.rationale_by_predicate,
            },
            summary="Merged uncertainty discovery from the base bundle and extension data.",
            raw_output=None,
        )
    return InitObservationLearningResult(
        source_records=_merge_sequence(
            existing.source_records, new.source_records, key=lambda item: item.episode_name
        ),
        review_results=[*existing.review_results, *new.review_results],
        seed_examples=_merge_sequence(
            existing.seed_examples,
            new.seed_examples,
            key=lambda item: (item.episode_name, item.grounded_literal, item.source_kind),
        ),
        conditions=_merge_sequence(
            existing.conditions,
            new.conditions,
            key=lambda item: (item.predicate_name, tuple(item.argument_types)),
        ),
        expanded_examples=_merge_sequence(
            existing.expanded_examples,
            new.expanded_examples,
            key=lambda item: (item.episode_name, item.grounded_literal, item.source_kind),
        ),
        schemas=schemas,
        predicate_inventory=_merge_inventory(existing.predicate_inventory, new.predicate_inventory),
        predicate_comments={**existing.predicate_comments, **new.predicate_comments},
        rendered_module_text=render_init_observation_module(schemas),
        uncertain_predicate_discovery=discovery,
    )


def merge_active_results(
    existing: ActiveObservationLearningResult | None,
    new: ActiveObservationLearningResult,
) -> ActiveObservationLearningResult:
    if existing is None:
        return new
    schemas = _merge_schemas(existing.schemas, new.schemas, key=_active_schema_key)
    return ActiveObservationLearningResult(
        source_records=_merge_sequence(
            existing.source_records,
            new.source_records,
            key=lambda item: (item.episode_name, item.step_index, item.canonical_action_name, item.effect_bucket),
        ),
        discovery_results=[*existing.discovery_results, *new.discovery_results],
        seed_examples=_merge_sequence(
            existing.seed_examples,
            new.seed_examples,
            key=lambda item: (
                item.episode_name,
                item.step_index,
                item.canonical_action_name,
                item.effect_bucket,
                item.grounded_literal,
                item.source_kind,
            ),
        ),
        conditions=_merge_sequence(
            existing.conditions,
            new.conditions,
            key=lambda item: (
                item.canonical_action_name,
                item.effect_bucket,
                item.predicate_name,
                tuple(item.action_argument_types),
                tuple(item.extra_argument_types),
            ),
        ),
        confirmed_examples=_merge_sequence(
            existing.confirmed_examples,
            new.confirmed_examples,
            key=lambda item: (
                item.episode_name,
                item.step_index,
                item.canonical_action_name,
                item.effect_bucket,
                item.grounded_literal,
                item.source_kind,
            ),
        ),
        schemas=schemas,
        predicate_inventory=_merge_inventory(existing.predicate_inventory, new.predicate_inventory),
        predicate_comments={**existing.predicate_comments, **new.predicate_comments},
        rendered_module_text=render_active_observation_module(schemas),
    )


def load_passive_result(summary_file: Path | None) -> PassiveObservationLearningResult | None:
    if summary_file is None or not summary_file.exists():
        return None
    payload = _read_json(summary_file)
    return PassiveObservationLearningResult(
        source_records=[PassiveObservationSourceRecord(**item) for item in payload.get("source_records", [])],
        review_results_by_variant=dict(payload.get("review_results_by_variant", {})),
        seed_examples=[PassiveObservationExample(**item) for item in payload.get("seed_examples", [])],
        conditions=[PassiveObservationCondition(**item) for item in payload.get("conditions", [])],
        expanded_examples=[PassiveObservationExample(**item) for item in payload.get("expanded_examples", [])],
        schemas=[PassiveObservationRuleSchema(**item) for item in payload.get("schemas", [])],
        predicate_inventory=[PredicateSchema(**item) for item in payload.get("predicate_inventory", [])],
        predicate_comments=dict(payload.get("predicate_comments", {})),
        rendered_module_text=str(payload.get("rendered_module_text") or ""),
    )


def load_init_result(summary_file: Path | None) -> InitObservationLearningResult | None:
    if summary_file is None or not summary_file.exists():
        return None
    payload = _read_json(summary_file)
    discovery_payload = payload.get("uncertain_predicate_discovery")
    return InitObservationLearningResult(
        source_records=[InitObservationSourceRecord(**item) for item in payload.get("source_records", [])],
        review_results=list(payload.get("review_results", [])),
        seed_examples=[InitObservationExample(**item) for item in payload.get("seed_examples", [])],
        conditions=[InitObservationCondition(**item) for item in payload.get("conditions", [])],
        expanded_examples=[InitObservationExample(**item) for item in payload.get("expanded_examples", [])],
        schemas=[InitObservationRuleSchema(**item) for item in payload.get("schemas", [])],
        predicate_inventory=[PredicateSchema(**item) for item in payload.get("predicate_inventory", [])],
        predicate_comments=dict(payload.get("predicate_comments", {})),
        rendered_module_text=str(payload.get("rendered_module_text") or ""),
        uncertain_predicate_discovery=(
            InitObservationUncertainPredicateDiscovery(**discovery_payload)
            if isinstance(discovery_payload, dict)
            else None
        ),
    )


def load_active_result(summary_file: Path | None) -> ActiveObservationLearningResult | None:
    if summary_file is None or not summary_file.exists():
        return None
    payload = _read_json(summary_file)
    return ActiveObservationLearningResult(
        source_records=[ActiveObservationSourceRecord(**item) for item in payload.get("source_records", [])],
        discovery_results=list(payload.get("discovery_results", [])),
        seed_examples=[ActiveObservationExample(**item) for item in payload.get("seed_examples", [])],
        conditions=[ActiveObservationCondition(**item) for item in payload.get("conditions", [])],
        confirmed_examples=[ActiveObservationExample(**item) for item in payload.get("confirmed_examples", [])],
        schemas=[ActiveObservationRuleSchema(**item) for item in payload.get("schemas", [])],
        predicate_inventory=[PredicateSchema(**item) for item in payload.get("predicate_inventory", [])],
        predicate_comments=dict(payload.get("predicate_comments", {})),
        rendered_module_text=str(payload.get("rendered_module_text") or ""),
    )


__all__ = [
    "load_active_result",
    "load_init_result",
    "load_passive_result",
    "merge_active_results",
    "merge_init_results",
    "merge_passive_results",
]
