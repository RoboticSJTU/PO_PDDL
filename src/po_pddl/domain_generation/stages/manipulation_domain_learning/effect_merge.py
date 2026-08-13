from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Iterable

from po_pddl.config import DEFAULT_MODEL
from po_pddl.domain_generation.infrastructure.fact_utils import remap_symbolic_literal_arguments

from .models import ActionSchema, ManipulationEffectRecord
from .object_name_normalization import (
    coarsen_object_identifiers,
    coarsen_symbolic_literal_list,
)
from .shared import extract_json_object, load_prompt, make_client, safe_chat

logger = logging.getLogger(__name__)

_BARE_ZERO_ARITY_PATTERN = re.compile(r"^\s*([a-z][a-z0-9_]*)\s*$")
_BARE_NEGATED_ZERO_ARITY_PATTERN = re.compile(r"^\s*not\s+([a-z][a-z0-9_]*)\s*$")


def standard_effect_bucket_name(action_name: str, success: bool) -> str:
    return f"{action_name}_{'success' if success else 'failure'}"


def _effect_signature(record: ManipulationEffectRecord) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(sorted(str(item) for item in record.delta_add)),
        tuple(sorted(str(item) for item in record.delta_del)),
    )


def _effect_literal_count(signature: tuple[tuple[str, ...], tuple[str, ...]]) -> int:
    return len(signature[0]) + len(signature[1])


def _record_with_effect(
    record: ManipulationEffectRecord,
    *,
    effect_bucket: str,
    delta_add: Iterable[str],
    delta_del: Iterable[str],
) -> ManipulationEffectRecord:
    return ManipulationEffectRecord(
        episode_name=record.episode_name,
        step_index=record.step_index,
        raw_action_text=record.raw_action_text,
        canonical_action_name=record.canonical_action_name,
        action_arguments=coarsen_object_identifiers(list(record.action_arguments)),
        pre_observation_text=record.pre_observation_text,
        post_observation_text=record.post_observation_text,
        extra_info=record.extra_info,
        delta_add=coarsen_symbolic_literal_list([str(item) for item in delta_add]),
        delta_del=coarsen_symbolic_literal_list([str(item) for item in delta_del]),
        effect_bucket=effect_bucket,
        success=record.success,
        execution_time_sec=record.execution_time_sec,
    )


def _instantiate_literals_for_record(
    literals: Iterable[str],
    action_arguments: list[str],
) -> list[str]:
    argument_mapping = {f"?arg{index}": argument for index, argument in enumerate(action_arguments)}
    argument_mapping.update({f"?param_{index + 1}": argument for index, argument in enumerate(action_arguments)})
    normalized_literals = [_normalize_zero_arity_literal(str(item)) for item in literals]
    return coarsen_symbolic_literal_list(
        [
            remap_symbolic_literal_arguments(item, argument_mapping) if argument_mapping else item
            for item in normalized_literals
        ]
    )


def _normalize_zero_arity_literal(text: str) -> str:
    stripped = text.strip()
    negated_match = _BARE_NEGATED_ZERO_ARITY_PATTERN.fullmatch(stripped)
    if negated_match:
        return f"not {negated_match.group(1)}()"
    positive_match = _BARE_ZERO_ARITY_PATTERN.fullmatch(stripped)
    if positive_match:
        return f"{positive_match.group(1)}()"
    return stripped


def prune_replay_noop_effects(
    records: list[ManipulationEffectRecord],
    *,
    state_before_by_step: dict[tuple[str, int], list[str]],
) -> tuple[list[ManipulationEffectRecord], int]:
    """Remove effects that never change replay state for their action outcome."""
    transition_counts: dict[tuple[str, bool, str, str], int] = {}
    observed_keys: set[tuple[str, bool, str, str]] = set()

    def abstract_literal(record: ManipulationEffectRecord, literal: str) -> str:
        mapping = {argument: f"?arg{index}" for index, argument in enumerate(record.action_arguments)}
        return remap_symbolic_literal_arguments(literal, mapping) if mapping else literal

    for record in records:
        state_before = state_before_by_step.get((record.episode_name, record.step_index))
        if state_before is None:
            continue
        state_keys = {str(item).strip() for item in state_before if str(item).strip()}
        for effect_kind, literals in (("add", record.delta_add), ("del", record.delta_del)):
            for literal in literals:
                key = (
                    record.canonical_action_name,
                    record.success,
                    effect_kind,
                    abstract_literal(record, literal),
                )
                observed_keys.add(key)
                changes_state = literal not in state_keys if effect_kind == "add" else literal in state_keys
                if changes_state:
                    transition_counts[key] = transition_counts.get(key, 0) + 1

    no_op_keys = {key for key in observed_keys if transition_counts.get(key, 0) == 0}
    if not no_op_keys:
        return list(records), 0

    changed_count = 0
    pruned_records: list[ManipulationEffectRecord] = []
    for record in records:
        delta_add = [
            literal
            for literal in record.delta_add
            if (
                record.canonical_action_name,
                record.success,
                "add",
                abstract_literal(record, literal),
            )
            not in no_op_keys
        ]
        delta_del = [
            literal
            for literal in record.delta_del
            if (
                record.canonical_action_name,
                record.success,
                "del",
                abstract_literal(record, literal),
            )
            not in no_op_keys
        ]
        if delta_add != record.delta_add or delta_del != record.delta_del:
            changed_count += 1
            record = replace(record, delta_add=delta_add, delta_del=delta_del)
        pruned_records.append(record)
    return pruned_records, changed_count


def rewrite_records_using_action_schemas(
    records: list[ManipulationEffectRecord],
    action_schemas: list[ActionSchema],
) -> list[ManipulationEffectRecord]:
    branch_by_action_outcome: dict[tuple[str, bool], tuple[str, list[str], list[str]]] = {}
    for schema in action_schemas:
        grouped: dict[bool, list[tuple[str, list[str], list[str]]]] = {True: [], False: []}
        for branch in schema.effect_branches:
            grouped[branch.success].append((branch.effect_bucket, list(branch.delta_add), list(branch.delta_del)))
        for success_value, candidates in grouped.items():
            if len(candidates) == 1:
                branch_by_action_outcome[(schema.canonical_action_name, success_value)] = candidates[0]

    rewritten: list[ManipulationEffectRecord] = []
    for record in records:
        selected = branch_by_action_outcome.get((record.canonical_action_name, record.success))
        if selected is None:
            rewritten.append(
                _record_with_effect(
                    record,
                    effect_bucket=standard_effect_bucket_name(record.canonical_action_name, record.success),
                    delta_add=record.delta_add,
                    delta_del=record.delta_del,
                )
            )
            continue
        effect_bucket, delta_add, delta_del = selected
        rewritten.append(
            _record_with_effect(
                record,
                effect_bucket=effect_bucket,
                delta_add=_instantiate_literals_for_record(delta_add, record.action_arguments),
                delta_del=_instantiate_literals_for_record(delta_del, record.action_arguments),
            )
        )
    return rewritten


@dataclass
class LLMManipulationEffectMergeModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1400
    max_workers: int = 1
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {self.max_workers}")
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("effect_merge_prompt.md")

    def merge_records(self, records: list[ManipulationEffectRecord]) -> list[ManipulationEffectRecord]:
        grouped: dict[tuple[str, bool], list[ManipulationEffectRecord]] = {}
        order: list[tuple[str, bool]] = []
        for record in records:
            key = (record.canonical_action_name, record.success)
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(record)

        if self.max_workers == 1 or len(order) <= 1:
            selections = {key: self._select_group_effect(grouped[key]) for key in order}
        else:
            selections = self._select_group_effects_in_parallel(grouped, order)

        rewritten: list[ManipulationEffectRecord] = []
        for record in records:
            effect_bucket, delta_add, delta_del = selections[(record.canonical_action_name, record.success)]
            rewritten.append(
                _record_with_effect(
                    record,
                    effect_bucket=effect_bucket,
                    delta_add=_instantiate_literals_for_record(delta_add, record.action_arguments),
                    delta_del=_instantiate_literals_for_record(delta_del, record.action_arguments),
                )
            )
        return rewritten

    def _select_group_effects_in_parallel(
        self,
        grouped: dict[tuple[str, bool], list[ManipulationEffectRecord]],
        order: list[tuple[str, bool]],
    ) -> dict[tuple[str, bool], tuple[str, list[str], list[str]]]:
        selections: dict[tuple[str, bool], tuple[str, list[str], list[str]]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="effect-merge") as executor:
            future_to_key = {executor.submit(self._select_group_effect, grouped[key]): key for key in order}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                selections[key] = future.result()
        return selections

    def _select_group_effect(
        self,
        records: list[ManipulationEffectRecord],
    ) -> tuple[str, list[str], list[str]]:
        if not records:
            raise ValueError("Expected non-empty manipulation effect record group.")
        action_name = records[0].canonical_action_name
        success_value = records[0].success
        canonical_bucket = standard_effect_bucket_name(action_name, success_value)

        by_signature: dict[tuple[tuple[str, ...], tuple[str, ...]], list[ManipulationEffectRecord]] = {}
        signature_order: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        for record in records:
            signature = _effect_signature(record)
            if signature not in by_signature:
                by_signature[signature] = []
                signature_order.append(signature)
            by_signature[signature].append(record)

        if len(signature_order) == 1:
            only_signature = signature_order[0]
            delta_add, delta_del = only_signature
            return canonical_bucket, list(delta_add), list(delta_del)

        signature_positions = {signature: index for index, signature in enumerate(signature_order)}
        ordered_signatures = sorted(
            signature_order,
            key=lambda item: (
                -_effect_literal_count(item),
                -len(by_signature[item]),
                signature_positions[item],
            ),
        )

        candidates_payload: list[dict[str, object]] = []
        for index, signature in enumerate(ordered_signatures, start=1):
            candidate_records = by_signature[signature]
            sample_rows = [
                {
                    "episode_name": item.episode_name,
                    "step_index": item.step_index,
                    "action_text": item.raw_action_text,
                    "extra_info": item.extra_info,
                    "observation_text": item.post_observation_text,
                }
                for item in candidate_records[:8]
            ]
            candidates_payload.append(
                {
                    "candidate_id": f"candidate_{index}",
                    "count": len(candidate_records),
                    "delta_add": list(signature[0]),
                    "delta_del": list(signature[1]),
                    "delta_add_count": len(signature[0]),
                    "delta_del_count": len(signature[1]),
                    "total_literal_count": _effect_literal_count(signature),
                    "is_most_complete_reference": index == 1,
                    "sample_steps": sample_rows,
                }
            )

        payload = {
            "canonical_action_name": action_name,
            "success": success_value,
            "raw_step_count": len(records),
            "candidates": candidates_payload,
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
        selected_candidate_id = str(data.get("selected_candidate_id") or "").strip()
        if not selected_candidate_id:
            raise ValueError("Effect merge response is missing selected_candidate_id.")
        selected_candidate = next(
            (item for item in candidates_payload if item["candidate_id"] == selected_candidate_id),
            None,
        )
        if selected_candidate is None:
            raise ValueError(f"Effect merge selected unknown candidate_id {selected_candidate_id!r} for {action_name}")
        repaired_delta_add = data.get("repaired_delta_add")
        repaired_delta_del = data.get("repaired_delta_del")
        if repaired_delta_add is not None or repaired_delta_del is not None:
            if repaired_delta_add is None or repaired_delta_del is None:
                raise ValueError(
                    "Effect merge repair response must provide both repaired_delta_add and repaired_delta_del."
                )
            if not isinstance(repaired_delta_add, list) or not isinstance(repaired_delta_del, list):
                raise ValueError(
                    "Effect merge repair response must provide repaired_delta_add and repaired_delta_del as lists."
                )
            return (
                canonical_bucket,
                coarsen_symbolic_literal_list([str(item).strip() for item in repaired_delta_add if str(item).strip()]),
                coarsen_symbolic_literal_list([str(item).strip() for item in repaired_delta_del if str(item).strip()]),
            )
        return (
            canonical_bucket,
            [str(item) for item in selected_candidate["delta_add"]],
            [str(item) for item in selected_candidate["delta_del"]],
        )
