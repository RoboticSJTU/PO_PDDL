from __future__ import annotations

from pathlib import Path
from typing import Any

from po_pddl.domain_generation.infrastructure.artifact_io import load_json_object
from po_pddl.domain_generation.infrastructure.fact_utils import parse_symbolic_literal
from po_pddl.domain_generation.stages.active_observation_learning.learner import (
    ActiveObservationLearner,
)
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
from po_pddl.domain_generation.stages.manipulation_domain_learning.grounding_update import (
    load_action_schemas,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.models import PredicateSchema
from po_pddl.domain_generation.stages.passive_observation_learning.learner import (
    PassiveObservationLearner,
)
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


def _predicate_names_from_literals(literals: list[str]) -> set[str]:
    names: set[str] = set()
    for literal in literals:
        try:
            _negated, predicate_name, _arguments = parse_symbolic_literal(str(literal))
        except Exception:
            continue
        names.add(predicate_name)
    return names


def _precondition_predicates_by_action(precondition_learning_dir: str | Path) -> dict[str, set[str]]:
    schema_file = Path(precondition_learning_dir) / "action_schemas.json"
    if not schema_file.exists():
        return {}
    return {
        schema.canonical_action_name: _predicate_names_from_literals(list(schema.precondition_literals))
        for schema in load_action_schemas(schema_file)
    }


def _load_passive_result(output_dir: str | Path) -> PassiveObservationLearningResult | None:
    summary_file = Path(output_dir) / "passive_observation_learning_summary.json"
    if not summary_file.exists():
        return None
    payload = load_json_object(summary_file)
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


def _load_active_result(output_dir: str | Path) -> ActiveObservationLearningResult | None:
    summary_file = Path(output_dir) / "active_observation_learning_summary.json"
    if not summary_file.exists():
        return None
    payload = load_json_object(summary_file)
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


def _passive_schema_key(schema: PassiveObservationRuleSchema) -> tuple[str, str, bool, int, str]:
    return (
        schema.canonical_action_name,
        schema.effect_bucket,
        schema.success,
        schema.variant_rank,
        schema.predicate_name,
    )


def _active_schema_key(schema: ActiveObservationRuleSchema) -> tuple[str, str, bool, int, str]:
    return (
        schema.canonical_action_name,
        schema.effect_bucket,
        schema.success,
        schema.variant_rank,
        schema.predicate_name,
    )


def _passive_example_key(example: PassiveObservationExample) -> tuple[str, str, bool, int, str]:
    return (
        example.canonical_action_name,
        example.effect_bucket,
        example.success,
        example.variant_rank,
        example.predicate_name,
    )


def _active_example_key(example: ActiveObservationExample) -> tuple[str, str, bool, int, str]:
    return (
        example.canonical_action_name,
        example.effect_bucket,
        example.success,
        example.variant_rank,
        example.predicate_name,
    )


def prune_passive_and_active_observation_outputs_by_action_preconditions(
    *,
    precondition_learning_dir: str | Path,
    passive_observation_learning_dir: str | Path | None = None,
    active_observation_learning_dir: str | Path | None = None,
) -> dict[str, Any]:
    precondition_predicates = _precondition_predicates_by_action(precondition_learning_dir)
    summary: dict[str, Any] = {
        "precondition_actions": sorted(precondition_predicates.keys()),
        "passive_removed_schema_count": 0,
        "active_removed_schema_count": 0,
    }

    if passive_observation_learning_dir is not None:
        passive_result = _load_passive_result(passive_observation_learning_dir)
        if passive_result is not None:
            kept_passive_schemas = [
                schema
                for schema in passive_result.schemas
                if schema.predicate_name not in precondition_predicates.get(schema.canonical_action_name, set())
            ]
            kept_passive_keys = {_passive_schema_key(item) for item in kept_passive_schemas}
            summary["passive_removed_schema_count"] = len(passive_result.schemas) - len(kept_passive_schemas)
            if len(kept_passive_schemas) != len(passive_result.schemas):
                updated_passive_result = PassiveObservationLearningResult(
                    source_records=list(passive_result.source_records),
                    review_results_by_variant=dict(passive_result.review_results_by_variant),
                    seed_examples=[
                        item for item in passive_result.seed_examples if _passive_example_key(item) in kept_passive_keys
                    ],
                    conditions=[
                        item for item in passive_result.conditions if _passive_schema_key(item) in kept_passive_keys
                    ],
                    expanded_examples=[
                        item
                        for item in passive_result.expanded_examples
                        if _passive_example_key(item) in kept_passive_keys
                    ],
                    schemas=kept_passive_schemas,
                    predicate_inventory=list(passive_result.predicate_inventory),
                    predicate_comments=dict(passive_result.predicate_comments),
                    rendered_module_text=render_passive_observation_module(kept_passive_schemas),
                )
                PassiveObservationLearner(
                    description_review_module=None,  # type: ignore[arg-type]
                    vlm_confirmation_module=None,  # type: ignore[arg-type]
                ).write_outputs(updated_passive_result, passive_observation_learning_dir)

    if active_observation_learning_dir is not None:
        active_result = _load_active_result(active_observation_learning_dir)
        if active_result is not None:
            kept_active_schemas = [
                schema
                for schema in active_result.schemas
                if schema.predicate_name not in precondition_predicates.get(schema.canonical_action_name, set())
            ]
            kept_active_keys = {_active_schema_key(item) for item in kept_active_schemas}
            summary["active_removed_schema_count"] = len(active_result.schemas) - len(kept_active_schemas)
            if len(kept_active_schemas) != len(active_result.schemas):
                updated_active_result = ActiveObservationLearningResult(
                    source_records=list(active_result.source_records),
                    discovery_results=list(active_result.discovery_results),
                    seed_examples=[
                        item for item in active_result.seed_examples if _active_example_key(item) in kept_active_keys
                    ],
                    conditions=[
                        item for item in active_result.conditions if _active_schema_key(item) in kept_active_keys
                    ],
                    confirmed_examples=[
                        item
                        for item in active_result.confirmed_examples
                        if _active_example_key(item) in kept_active_keys
                    ],
                    schemas=kept_active_schemas,
                    predicate_inventory=list(active_result.predicate_inventory),
                    predicate_comments=dict(active_result.predicate_comments),
                    rendered_module_text=render_active_observation_module(kept_active_schemas),
                )
                ActiveObservationLearner(
                    discovery_module=None,  # type: ignore[arg-type]
                    vlm_confirmation_module=None,  # type: ignore[arg-type]
                ).write_outputs(updated_active_result, active_observation_learning_dir)

    return summary


__all__ = ["prune_passive_and_active_observation_outputs_by_action_preconditions"]
