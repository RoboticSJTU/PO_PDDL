from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from po_pddl.config import DEFAULT_MODEL
from po_pddl.domain_generation.infrastructure.fact_utils import (
    remap_symbolic_literal_arguments,
    try_parse_symbolic_literal,
)

from .models import ActionEffectStatistic, ActionTaxonomyRecord, ManipulationEffectRecord, RawTrajectoryStep
from .renderer import classify_records_by_effect_statistics, collect_action_effect_statistics
from .shared import extract_json_object, load_prompt, make_client, safe_chat

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EffectVariantMergePlan:
    canonical_action_name: str
    success: bool
    source_variant_ranks: list[int]
    merged_delta_add: list[str]
    merged_delta_del: list[str]
    rationale: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "canonical_action_name": self.canonical_action_name,
            "success": self.success,
            "source_variant_ranks": list(self.source_variant_ranks),
            "merged_delta_add": list(self.merged_delta_add),
            "merged_delta_del": list(self.merged_delta_del),
        }
        if self.rationale:
            payload["rationale"] = self.rationale
        return payload


@dataclass(frozen=True)
class EffectVariantReviewDecision:
    canonical_action_name: str
    success: bool
    merge_plans: list[EffectVariantMergePlan]
    raw_llm_output: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "canonical_action_name": self.canonical_action_name,
            "success": self.success,
            "merge_plans": [item.to_dict() for item in self.merge_plans],
        }
        if self.raw_llm_output:
            payload["raw_llm_output"] = self.raw_llm_output
        return payload


@dataclass
class LLMEffectVariantReviewModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 3000
    max_workers: int = 1
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {self.max_workers}")
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("effect_variant_review_prompt.md")
        self.last_review_summary: dict[str, object] = {}

    def review_and_merge_variants(
        self,
        *,
        steps: list[RawTrajectoryStep],
        taxonomy_records: list[ActionTaxonomyRecord],
        records: list[ManipulationEffectRecord],
        target_action_outcomes: set[tuple[str, bool]] | None = None,
    ) -> tuple[list[ManipulationEffectRecord], dict[str, list[ActionEffectStatistic]]]:
        bucketized_records = classify_records_by_effect_statistics(records)
        statistics = collect_action_effect_statistics(bucketized_records)
        review_jobs = _collect_review_jobs(
            steps=steps,
            taxonomy_records=taxonomy_records,
            records=bucketized_records,
            statistics=statistics,
            target_action_outcomes=target_action_outcomes,
        )
        if not review_jobs:
            self.last_review_summary = {
                "reviewed_group_count": 0,
                "decisions": [],
            }
            return bucketized_records, statistics

        logger.info(
            "Effect-variant review: reviewing %d action/outcome group(s) with multiple effect variants.",
            len(review_jobs),
        )
        if self.max_workers == 1 or len(review_jobs) == 1:
            decisions = [self._review_single_job(job) for job in review_jobs]
        else:
            decisions = [None] * len(review_jobs)
            with ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="effect-variant-review"
            ) as executor:
                future_to_index = {
                    executor.submit(self._review_single_job, job): index for index, job in enumerate(review_jobs)
                }
                for future in as_completed(future_to_index):
                    decisions[future_to_index[future]] = future.result()
            decisions = [item for item in decisions if item is not None]

        merged_records = _apply_review_decisions(bucketized_records, decisions)
        merged_records = classify_records_by_effect_statistics(merged_records)
        merged_statistics = collect_action_effect_statistics(merged_records)
        self.last_review_summary = {
            "reviewed_group_count": len(review_jobs),
            "decisions": [item.to_dict() for item in decisions],
        }
        return merged_records, merged_statistics

    def _review_single_job(self, job: dict[str, object]) -> EffectVariantReviewDecision:
        canonical_action_name = str(job["canonical_action_name"])
        success = bool(job["success"])
        logger.info(
            "Effect-variant review: reviewing action=%s success=%s with %d variant(s).",
            canonical_action_name,
            success,
            len(job["variants"]),
        )
        reply = safe_chat(
            self._client,
            self._prompt,
            json.dumps(job, ensure_ascii=False, indent=2),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        raw_merge_plans = data.get("merge_plans", [])
        if raw_merge_plans is None:
            raw_merge_plans = []
        if not isinstance(raw_merge_plans, list):
            raise ValueError("effect variant review response field `merge_plans` must be a list.")

        available_variant_ranks = {
            int(item["variant_rank"]) for item in job["variants"] if int(item["variant_rank"]) > 0
        }
        merge_plans: list[EffectVariantMergePlan] = []
        seen_variant_ranks: set[int] = set()
        for index, row in enumerate(raw_merge_plans):
            if not isinstance(row, dict):
                raise ValueError(f"merge_plans[{index}] must be an object.")
            source_variant_ranks = sorted({int(item) for item in row.get("source_variant_ranks", [])})
            if len(source_variant_ranks) < 2:
                raise ValueError(f"merge_plans[{index}] must contain at least two source_variant_ranks.")
            unknown_variant_ranks = [item for item in source_variant_ranks if item not in available_variant_ranks]
            if unknown_variant_ranks:
                raise ValueError(
                    f"merge_plans[{index}] references unknown variant ranks {unknown_variant_ranks}; "
                    f"available={sorted(available_variant_ranks)}."
                )
            overlap = seen_variant_ranks.intersection(source_variant_ranks)
            if overlap:
                raise ValueError(f"merge_plans[{index}] overlaps previous plans on variant ranks {sorted(overlap)}.")
            seen_variant_ranks.update(source_variant_ranks)
            merged_delta_add = _validate_abstract_effect_literals(
                row.get("merged_delta_add", []),
                field_name=f"merge_plans[{index}].merged_delta_add",
            )
            merged_delta_del = _validate_abstract_effect_literals(
                row.get("merged_delta_del", []),
                field_name=f"merge_plans[{index}].merged_delta_del",
            )
            merge_plans.append(
                EffectVariantMergePlan(
                    canonical_action_name=canonical_action_name,
                    success=success,
                    source_variant_ranks=source_variant_ranks,
                    merged_delta_add=merged_delta_add,
                    merged_delta_del=merged_delta_del,
                    rationale=str(row.get("rationale") or "").strip() or None,
                )
            )

        if len(merge_plans) != 1:
            raise ValueError(
                "effect variant review response must provide exactly one merge plan "
                f"for action={canonical_action_name} success={success}; got {len(merge_plans)}."
            )
        if seen_variant_ranks != available_variant_ranks:
            missing_variant_ranks = sorted(available_variant_ranks.difference(seen_variant_ranks))
            raise ValueError(
                "effect variant review response must merge every available variant rank into the single final bucket; "
                f"missing={missing_variant_ranks}, available={sorted(available_variant_ranks)}."
            )

        return EffectVariantReviewDecision(
            canonical_action_name=canonical_action_name,
            success=success,
            merge_plans=merge_plans,
            raw_llm_output=reply,
        )


def _bucket_variant_rank(effect_bucket: str) -> int | None:
    suffix = str(effect_bucket).rsplit("_bucket_", 1)
    if len(suffix) == 2 and suffix[1].isdigit():
        return int(suffix[1])
    return None


def _validate_abstract_effect_literals(value: object, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")
    outputs: list[str] = []
    for index, item in enumerate(value):
        text = str(item).strip()
        if not text:
            continue
        if try_parse_symbolic_literal(text) is None:
            raise ValueError(f"{field_name}[{index}] is not a valid symbolic literal: {text!r}")
        outputs.append(text)
    return outputs


def _instantiate_abstract_literals(literals: list[str], action_arguments: list[str]) -> list[str]:
    mapping = {f"?param_{index + 1}": argument for index, argument in enumerate(action_arguments)}
    mapping.update({f"?arg{index}": argument for index, argument in enumerate(action_arguments)})
    return [remap_symbolic_literal_arguments(literal, mapping) if mapping else literal for literal in literals]


def _collect_review_jobs(
    *,
    steps: list[RawTrajectoryStep],
    taxonomy_records: list[ActionTaxonomyRecord],
    records: list[ManipulationEffectRecord],
    statistics: dict[str, list[ActionEffectStatistic]],
    target_action_outcomes: set[tuple[str, bool]] | None = None,
) -> list[dict[str, object]]:
    step_map = {(step.episode_name, step.step_index): step for step in steps}
    taxonomy_map = {(record.episode_name, record.step_index): record for record in taxonomy_records}
    records_by_action_outcome: dict[tuple[str, bool], list[ManipulationEffectRecord]] = {}
    for record in records:
        records_by_action_outcome.setdefault((record.canonical_action_name, record.success), []).append(record)

    jobs: list[dict[str, object]] = []
    for canonical_action_name, action_stats in sorted(statistics.items()):
        grouped_stats: dict[bool, list[ActionEffectStatistic]] = {}
        for stat in action_stats:
            grouped_stats.setdefault(bool(stat.success), []).append(stat)
        for success_value, outcome_stats in sorted(grouped_stats.items(), key=lambda item: (item[0] is False,)):
            if (
                target_action_outcomes is not None
                and (
                    canonical_action_name,
                    success_value,
                )
                not in target_action_outcomes
            ):
                continue
            ranked_stats = [item for item in outcome_stats if item.variant_rank is not None]
            if len(ranked_stats) <= 1:
                continue
            outcome_records = records_by_action_outcome.get((canonical_action_name, success_value), [])
            records_by_variant_rank: dict[int, list[ManipulationEffectRecord]] = {}
            for record in outcome_records:
                variant_rank = _bucket_variant_rank(record.effect_bucket)
                if variant_rank is None:
                    continue
                records_by_variant_rank.setdefault(variant_rank, []).append(record)
            variants_payload: list[dict[str, object]] = []
            for stat in sorted(ranked_stats, key=lambda item: int(item.variant_rank or 0)):
                variant_rank = int(stat.variant_rank or 0)
                variant_records = records_by_variant_rank.get(variant_rank, [])
                variants_payload.append(
                    {
                        "variant_rank": variant_rank,
                        "effect_bucket": stat.effect_bucket,
                        "count": stat.count,
                        "probability": stat.probability,
                        "fixed_delta_add": list(stat.fixed_delta_add),
                        "fixed_delta_del": list(stat.fixed_delta_del),
                        "residual_delta_add": list(stat.residual_delta_add),
                        "residual_delta_del": list(stat.residual_delta_del),
                        "full_delta_add": list(stat.delta_add),
                        "full_delta_del": list(stat.delta_del),
                        "records": [
                            {
                                "episode_name": record.episode_name,
                                "step_index": record.step_index,
                                "step": step_map[(record.episode_name, record.step_index)].to_dict()
                                if (record.episode_name, record.step_index) in step_map
                                else None,
                                "taxonomy_record": taxonomy_map[(record.episode_name, record.step_index)].to_dict()
                                if (record.episode_name, record.step_index) in taxonomy_map
                                else None,
                                "manipulation_record": record.to_dict(),
                            }
                            for record in variant_records
                        ],
                    }
                )
            jobs.append(
                {
                    "canonical_action_name": canonical_action_name,
                    "success": success_value,
                    "variants": variants_payload,
                }
            )
    return jobs


def _apply_review_decisions(
    records: list[ManipulationEffectRecord],
    decisions: list[EffectVariantReviewDecision],
) -> list[ManipulationEffectRecord]:
    if not decisions:
        return list(records)
    plans_by_action_outcome: dict[tuple[str, bool], list[EffectVariantMergePlan]] = {}
    for decision in decisions:
        if decision.merge_plans:
            plans_by_action_outcome[(decision.canonical_action_name, decision.success)] = list(decision.merge_plans)
    if not plans_by_action_outcome:
        return list(records)

    updated_records: list[ManipulationEffectRecord] = []
    for record in records:
        plans = plans_by_action_outcome.get((record.canonical_action_name, record.success))
        if not plans:
            updated_records.append(record)
            continue
        variant_rank = _bucket_variant_rank(record.effect_bucket)
        if variant_rank is None:
            updated_records.append(record)
            continue
        matched_plan = next(
            (plan for plan in plans if variant_rank in set(plan.source_variant_ranks)),
            None,
        )
        if matched_plan is None:
            updated_records.append(record)
            continue
        updated_records.append(
            ManipulationEffectRecord(
                episode_name=record.episode_name,
                step_index=record.step_index,
                raw_action_text=record.raw_action_text,
                canonical_action_name=record.canonical_action_name,
                action_arguments=list(record.action_arguments),
                pre_observation_text=record.pre_observation_text,
                post_observation_text=record.post_observation_text,
                extra_info=record.extra_info,
                delta_add=_instantiate_abstract_literals(matched_plan.merged_delta_add, record.action_arguments),
                delta_del=_instantiate_abstract_literals(matched_plan.merged_delta_del, record.action_arguments),
                effect_bucket=record.effect_bucket,
                success=record.success,
            )
        )
    return updated_records


__all__ = [
    "EffectVariantMergePlan",
    "EffectVariantReviewDecision",
    "LLMEffectVariantReviewModule",
]
