from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from po_pddl.config import DEFAULT_MODEL
from po_pddl.domain_generation.infrastructure.fact_utils import parse_positive_symbolic_fact
from po_pddl.domain_generation.infrastructure.llm_shared import build_user_content

from .models import (
    ActionEffectStatistic,
    ActionTaxonomyRecord,
    ManipulationEffectRecord,
    ObjectTypeDefinition,
    PredicateSchema,
    RawTrajectoryStep,
)
from .modules import _coerce_bool, _coerce_snake_case, _coerce_string_list
from .shared import extract_json_object, load_prompt, make_client, safe_chat

logger = logging.getLogger(__name__)

_SPECIAL_REVIEW_TYPES = frozenset({"movable_item", "fixed_item", "containable_item"})


@dataclass(frozen=True)
class PredicateAdditionProposal:
    predicate_name: str
    parameter_types: list[str]
    comment: str
    grounded_fact: str
    value_after_action: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "predicate_name": self.predicate_name,
            "parameter_types": list(self.parameter_types),
            "comment": self.comment,
            "grounded_fact": self.grounded_fact,
            "value_after_action": self.value_after_action,
        }


@dataclass(frozen=True)
class EffectCompletenessRepairPlan:
    should_apply_repair: bool
    repair_summary: str
    episode_name: str | None = None
    step_index: int | None = None
    canonical_action_name: str | None = None
    effect_bucket: str | None = None
    variant_rank: int | None = None
    predicate_additions: list[PredicateAdditionProposal] = field(default_factory=list)
    raw_llm_output: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "should_apply_repair": self.should_apply_repair,
            "repair_summary": self.repair_summary,
            "episode_name": self.episode_name,
            "step_index": self.step_index,
            "canonical_action_name": self.canonical_action_name,
            "effect_bucket": self.effect_bucket,
            "variant_rank": self.variant_rank,
            "predicate_additions": [item.to_dict() for item in self.predicate_additions],
        }
        if self.raw_llm_output:
            payload["raw_llm_output"] = self.raw_llm_output
        return payload


@dataclass
class VLMEffectCompletenessReviewModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2500
    verbose: bool = False

    def __post_init__(self) -> None:
        try:
            self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            self._client = None
        self._prompt = load_prompt("effect_completeness_review_prompt.md")
        self.last_review_summary: dict[str, object] = {}

    def review_and_suggest_repairs(
        self,
        *,
        steps: list[RawTrajectoryStep],
        taxonomy_records: list[ActionTaxonomyRecord],
        records: list[ManipulationEffectRecord],
        action_statistics: dict[str, list[ActionEffectStatistic]],
        predicate_inventory: list[PredicateSchema],
        predicate_comments: dict[str, str] | None,
        object_types: list[ObjectTypeDefinition] | None,
        episode_results: dict[str, object],
        rendered_domain_pddl: str,
        target_bucket_variants: set[tuple[str, str, int | None]] | None = None,
        start_after_bucket_variant: tuple[str, str, int | None] | None = None,
    ) -> EffectCompletenessRepairPlan:
        jobs = _collect_review_jobs(
            steps=steps,
            taxonomy_records=taxonomy_records,
            records=records,
            action_statistics=action_statistics,
            predicate_inventory=predicate_inventory,
            predicate_comments=predicate_comments or {},
            object_types=object_types or [],
            episode_results=episode_results,
            rendered_domain_pddl=rendered_domain_pddl,
            target_bucket_variants=target_bucket_variants,
            start_after_bucket_variant=start_after_bucket_variant,
        )
        reviewed_jobs: list[dict[str, object]] = []
        if not jobs:
            self.last_review_summary = {
                "reviewed_job_count": 0,
                "repair_plan": None,
            }
            return EffectCompletenessRepairPlan(
                should_apply_repair=False,
                repair_summary="No effect variants were available for completeness review.",
            )
        for index, job in enumerate(jobs, start=1):
            logger.debug(
                "Effect completeness review: checking sample %d/%d for action=%s bucket=%s [%s step %s].",
                index,
                len(jobs),
                job["canonical_action_name"],
                job["effect_bucket"],
                job["episode_name"],
                job["step_index"],
            )
            plan = self._review_single_job(job)
            reviewed_jobs.append(
                {
                    "job": {
                        key: value for key, value in job.items() if key not in {"frame_paths", "rendered_domain_pddl"}
                    },
                    "repair_plan": plan.to_dict(),
                }
            )
            if plan.should_apply_repair:
                self.last_review_summary = {
                    "reviewed_job_count": len(reviewed_jobs),
                    "repair_plan": plan.to_dict(),
                    "reviewed_jobs": reviewed_jobs,
                }
                return plan
        no_repair_plan = EffectCompletenessRepairPlan(
            should_apply_repair=False,
            repair_summary="All reviewed effect variants already describe the action-relevant target state completely enough.",
        )
        self.last_review_summary = {
            "reviewed_job_count": len(reviewed_jobs),
            "repair_plan": no_repair_plan.to_dict(),
            "reviewed_jobs": reviewed_jobs,
        }
        return no_repair_plan

    def _review_single_job(self, job: dict[str, object]) -> EffectCompletenessRepairPlan:
        user_prompt = (
            "Input JSON:\n"
            f"{json.dumps({key: value for key, value in job.items() if key != 'frame_paths'}, ensure_ascii=False, indent=2)}"
        )
        if self._client is None:
            self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        user_content = build_user_content(
            user_prompt,
            image_paths=[path for path in job.get("frame_paths", []) if isinstance(path, str) and path.strip()],
        )
        reply = safe_chat(
            self._client,
            self._prompt,
            user_content,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        should_apply_repair = bool(data.get("should_apply_repair"))
        repair_summary = str(data.get("repair_summary") or "").strip()
        predicate_additions = self._parse_predicate_additions(
            raw_items=data.get("predicate_additions", []),
            job=job,
        )
        if should_apply_repair and not predicate_additions:
            logger.warning(
                "Effect completeness review requested repair for [%s step %s] but returned no predicate_additions; "
                "downgrading to no-op review result.",
                job["episode_name"],
                job["step_index"],
            )
            should_apply_repair = False
            repair_summary = (
                repair_summary or "Review requested repair but did not provide any valid predicate additions."
            )
        return EffectCompletenessRepairPlan(
            should_apply_repair=should_apply_repair,
            repair_summary=repair_summary,
            episode_name=str(job["episode_name"]),
            step_index=int(job["step_index"]),
            canonical_action_name=str(job["canonical_action_name"]),
            effect_bucket=str(job["effect_bucket"]),
            variant_rank=(int(job["variant_rank"]) if job.get("variant_rank") is not None else None),
            predicate_additions=predicate_additions,
            raw_llm_output=reply,
        )

    def _parse_predicate_additions(
        self,
        *,
        raw_items: object,
        job: dict[str, object],
    ) -> list[PredicateAdditionProposal]:
        if raw_items is None:
            return []
        if not isinstance(raw_items, list):
            raise ValueError("predicate_additions must be a list.")
        available_types = _collect_available_review_types(job)
        object_name_to_type = {
            str(item["name"]).strip(): str(item["type_name"]).strip()
            for item in job.get("grounded_objects", [])
            if isinstance(item, dict)
            and str(item.get("name") or "").strip()
            and str(item.get("type_name") or "").strip()
        }
        parent_by_type = {
            str(item["type_name"]).strip(): str(item["parent_type"]).strip()
            for item in job.get("object_types", [])
            if isinstance(item, dict)
            and str(item.get("type_name") or "").strip()
            and str(item.get("parent_type") or "").strip()
        }
        special_supertypes_by_type = {
            str(item["type_name"]).strip(): {
                str(value).strip() for value in item.get("special_supertypes", []) if str(value).strip()
            }
            for item in job.get("object_types", [])
            if isinstance(item, dict) and str(item.get("type_name") or "").strip()
        }
        existing_predicates = {
            str(item.get("predicate_name") or "").strip(): list(item.get("parameter_types") or [])
            for item in job.get("predicates", [])
            if isinstance(item, dict) and str(item.get("predicate_name") or "").strip()
        }
        outputs: list[PredicateAdditionProposal] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise ValueError(f"predicate_additions[{index}] must be an object.")
            predicate_name = _coerce_snake_case(item.get("predicate_name"), field_name="predicate_name")
            comment = str(item.get("comment") or "").strip()
            if not comment:
                raise ValueError(f"predicate_additions[{index}] must include a non-empty comment.")
            grounded_fact = str(item.get("grounded_fact") or "").strip()
            if not grounded_fact:
                raise ValueError(f"predicate_additions[{index}] must include grounded_fact.")
            fact_predicate, arguments = parse_positive_symbolic_fact(grounded_fact)
            if fact_predicate != predicate_name:
                raise ValueError(
                    f"predicate_additions[{index}] grounded_fact predicate {fact_predicate!r} does not match "
                    f"predicate_name {predicate_name!r}."
                )
            parameter_types = _coerce_string_list(item.get("parameter_types", []), field_name="parameter_types")
            if not parameter_types:
                parameter_types = [object_name_to_type.get(argument, "").strip() for argument in arguments]
                if not all(parameter_types):
                    raise ValueError(f"predicate_additions[{index}] must define parameter_types.")
            if len(arguments) != len(parameter_types):
                raise ValueError(
                    f"predicate_additions[{index}] grounded_fact arity {len(arguments)} does not match "
                    f"parameter_types arity {len(parameter_types)}."
                )
            if any(type_name == "object" for type_name in parameter_types):
                raise ValueError("predicate_additions must not use the default PDDL type `object`.")
            unknown_types = [type_name for type_name in parameter_types if type_name not in available_types]
            if unknown_types:
                raise ValueError(
                    f"predicate_additions[{index}] references unknown parameter types {unknown_types}; "
                    f"available={sorted(available_types)}."
                )
            existing_parameter_types = existing_predicates.get(predicate_name)
            if existing_parameter_types is not None and list(existing_parameter_types) != parameter_types:
                raise ValueError(
                    f"predicate_additions[{index}] conflicts with existing predicate `{predicate_name}` "
                    f"signature {existing_parameter_types}."
                )
            for argument, expected_type in zip(arguments, parameter_types, strict=True):
                object_type = object_name_to_type.get(argument)
                if object_type is None:
                    raise ValueError(
                        f"predicate_additions[{index}] grounded_fact references unknown object {argument!r}."
                    )
                if not _object_type_matches_expected(
                    object_type=object_type,
                    expected_type=expected_type,
                    parent_by_type=parent_by_type,
                    special_supertypes_by_type=special_supertypes_by_type,
                ):
                    raise ValueError(
                        f"predicate_additions[{index}] grounded_fact argument {argument!r} has type "
                        f"{object_type!r}, incompatible with expected type {expected_type!r}."
                    )
            value_after_action = _coerce_bool(item.get("value_after_action"), field_name="value_after_action")
            outputs.append(
                PredicateAdditionProposal(
                    predicate_name=predicate_name,
                    parameter_types=parameter_types,
                    comment=comment,
                    grounded_fact=grounded_fact,
                    value_after_action=value_after_action,
                )
            )
        return outputs


def _object_type_matches_expected(
    *,
    object_type: str,
    expected_type: str,
    parent_by_type: dict[str, str],
    special_supertypes_by_type: dict[str, set[str]] | None = None,
) -> bool:
    if object_type == expected_type:
        return True
    if expected_type in (special_supertypes_by_type or {}).get(object_type, set()):
        return True
    current = parent_by_type.get(object_type)
    while current:
        if current == expected_type:
            return True
        current = parent_by_type.get(current)
    return False


def _collect_available_review_types(job: dict[str, object]) -> set[str]:
    available_types: set[str] = set()
    parent_by_type: dict[str, str] = {}

    for item in job.get("object_types", []):
        if not isinstance(item, dict):
            continue
        type_name = str(item.get("type_name") or "").strip()
        parent_type = str(item.get("parent_type") or "").strip()
        if type_name:
            available_types.add(type_name)
        if type_name and parent_type:
            parent_by_type[type_name] = parent_type

    for item in job.get("grounded_objects", []):
        if not isinstance(item, dict):
            continue
        type_name = str(item.get("type_name") or "").strip()
        if type_name:
            available_types.add(type_name)

    for item in job.get("predicates", []):
        if not isinstance(item, dict):
            continue
        for type_name in item.get("parameter_types", []):
            normalized_type = str(type_name or "").strip()
            if normalized_type:
                available_types.add(normalized_type)

    for type_name in list(available_types):
        current = parent_by_type.get(type_name)
        while current:
            available_types.add(current)
            current = parent_by_type.get(current)

    rendered_domain_pddl = str(job.get("rendered_domain_pddl") or "")
    for special_type in _SPECIAL_REVIEW_TYPES:
        if special_type in available_types or special_type in rendered_domain_pddl:
            available_types.add(special_type)

    available_types.discard("")
    return available_types


def _collect_review_jobs(
    *,
    steps: list[RawTrajectoryStep],
    taxonomy_records: list[ActionTaxonomyRecord],
    records: list[ManipulationEffectRecord],
    action_statistics: dict[str, list[ActionEffectStatistic]],
    predicate_inventory: list[PredicateSchema],
    predicate_comments: dict[str, str],
    object_types: list[ObjectTypeDefinition],
    episode_results: dict[str, object],
    rendered_domain_pddl: str,
    target_bucket_variants: set[tuple[str, str, int | None]] | None = None,
    start_after_bucket_variant: tuple[str, str, int | None] | None = None,
) -> list[dict[str, object]]:
    def _last_frame_only(frame_paths: list[str]) -> list[str]:
        normalized = [str(path).strip() for path in frame_paths if str(path).strip()]
        if not normalized:
            return []
        return [normalized[-1]]

    step_map = {(step.episode_name, step.step_index): step for step in steps}
    taxonomy_map = {(record.episode_name, record.step_index): record for record in taxonomy_records}
    records_by_bucket: dict[tuple[str, str], list[ManipulationEffectRecord]] = {}
    for record in records:
        records_by_bucket.setdefault((record.canonical_action_name, record.effect_bucket), []).append(record)
    predicate_rows = [
        {
            "predicate_name": item.predicate_name,
            "parameter_types": list(item.parameter_types),
            "comment": str(predicate_comments.get(item.predicate_name) or item.comment or "").strip(),
        }
        for item in predicate_inventory
    ]
    object_type_rows = [item.to_dict() for item in object_types]
    jobs: list[dict[str, object]] = []
    seen_start_after = start_after_bucket_variant is None
    for canonical_action_name, stats in sorted(action_statistics.items()):
        for stat in sorted(stats, key=lambda item: (str(item.effect_bucket), int(item.variant_rank or 0))):
            signature = (
                str(canonical_action_name),
                str(stat.effect_bucket),
                int(stat.variant_rank) if stat.variant_rank is not None else None,
            )
            if not seen_start_after:
                if signature == start_after_bucket_variant:
                    seen_start_after = True
                continue
            if target_bucket_variants is not None and signature not in target_bucket_variants:
                continue
            bucket_records = sorted(
                records_by_bucket.get((canonical_action_name, stat.effect_bucket), []),
                key=lambda item: (item.episode_name, item.step_index),
            )
            if not bucket_records:
                continue
            sample_record = bucket_records[0]
            step = step_map.get((sample_record.episode_name, sample_record.step_index))
            taxonomy_record = taxonomy_map.get((sample_record.episode_name, sample_record.step_index))
            episode_result = episode_results.get(sample_record.episode_name)
            if step is None or taxonomy_record is None or episode_result is None:
                continue
            grounded_objects = [
                {
                    "name": item.name,
                    "type_name": item.type_name,
                }
                for item in episode_result.problem_spec.objects
            ]
            jobs.append(
                {
                    "episode_name": sample_record.episode_name,
                    "step_index": sample_record.step_index,
                    "instruction": step.instruction,
                    "action_text": sample_record.raw_action_text,
                    "scene_description": step.observation_text,
                    "canonical_action_name": canonical_action_name,
                    "effect_bucket": stat.effect_bucket,
                    "success": stat.success,
                    "variant_rank": stat.variant_rank,
                    "current_effect_record": sample_record.to_dict(),
                    "taxonomy_record": taxonomy_record.to_dict(),
                    "grounded_objects": grounded_objects,
                    "object_types": object_type_rows,
                    "predicates": predicate_rows,
                    "rendered_domain_pddl": rendered_domain_pddl,
                    "frame_paths": _last_frame_only(list(step.frame_paths)),
                }
            )
    return jobs
