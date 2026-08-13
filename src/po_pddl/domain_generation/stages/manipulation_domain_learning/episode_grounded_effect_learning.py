from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from typing import Protocol

from po_pddl.config import DEFAULT_MODEL
from po_pddl.core.parser import parse_domain, parse_problem
from po_pddl.domain_generation.infrastructure.fact_utils import format_symbolic_literal, parse_positive_symbolic_fact
from po_pddl.domain_generation.stages.problem_grounding.models import (
    GroundedTrajectoryStep,
    ProblemSpec,
    ValidationIssue,
    ValidationStepReport,
)
from po_pddl.domain_generation.stages.problem_grounding.renderer import render_problem_pddl
from po_pddl.domain_generation.stages.problem_grounding.validator import (
    execute_grounded_step,
    normalize_fact_key,
)

from .effect_merge import prune_replay_noop_effects
from .episode_problem_context import EpisodeProblemContextModule, EpisodeProblemContextResult
from .models import (
    ActionSchema,
    ActionTaxonomyRecord,
    ManipulationEffectRecord,
    ObjectTypeDefinition,
    PredicateSchema,
    RawTrajectoryStep,
)
from .modules import (
    _coerce_bool,
    _coerce_fact_list,
    _coerce_snake_case,
    _coerce_string_list,
    _extract_payload,
    _is_failure,
    _lookup_first,
    _predicate_arity_map,
    _taxonomy_record_prompt_dict,
    _validate_effect_literals_against_inventory,
)
from .renderer import render_action_schema_fragment
from .shared import extract_json_object, load_prompt, make_client, safe_chat

logger = logging.getLogger(__name__)


def _object_matches_expected_type(
    *,
    object_name: str,
    expected_type_name: str,
    object_type_by_name: dict[str, str],
) -> bool:
    object_type_name = str(object_type_by_name.get(object_name) or "").strip()
    if not object_type_name:
        return False
    if expected_type_name in {"object", object_type_name}:
        return True
    if expected_type_name == "fixed_item":
        return object_type_name in {"drawer", "cabinet", "bin", "station", "shelf", "container", "fixed_item"}
    if expected_type_name == "movable_item":
        return object_type_name in {"block", "box", "cup", "bottle", "tool", "movable_item"}
    if expected_type_name == "containable_item":
        return object_type_name in {"drawer", "cabinet", "bin", "container", "containable_item"}
    return False


def _normalize_repair_fact_list_against_inventory(
    fact_list: list[str],
    *,
    predicate_inventory: list[PredicateSchema] | None,
    object_type_by_name: dict[str, str],
) -> list[str]:
    if not predicate_inventory:
        return list(fact_list)
    predicate_by_name = {item.predicate_name: item for item in predicate_inventory}
    normalized_facts: list[str] = []
    for fact in fact_list:
        predicate_name, arguments = parse_positive_symbolic_fact(fact)
        predicate_schema = predicate_by_name.get(predicate_name)
        if predicate_schema is None:
            normalized_facts.append(fact)
            continue
        expected_types = list(predicate_schema.parameter_types)
        if len(expected_types) != 1 or len(arguments) != 2:
            normalized_facts.append(fact)
            continue
        matching_arguments = [
            argument
            for argument in arguments
            if _object_matches_expected_type(
                object_name=argument,
                expected_type_name=expected_types[0],
                object_type_by_name=object_type_by_name,
            )
        ]
        if len(matching_arguments) != 1:
            normalized_facts.append(fact)
            continue
        rewritten_fact = format_symbolic_literal(predicate_name, [matching_arguments[0]])
        logger.info(
            "Episode effect repair: rewrote invalid unary predicate literal %s -> %s based on predicate inventory.",
            fact,
            rewritten_fact,
        )
        normalized_facts.append(rewritten_fact)
    return normalized_facts


@dataclass(frozen=True)
class EpisodeEffectRepairStepPatch:
    step_index: int
    delta_add_add: list[str] = field(default_factory=list)
    delta_add_remove: list[str] = field(default_factory=list)
    delta_del_add: list[str] = field(default_factory=list)
    delta_del_remove: list[str] = field(default_factory=list)
    success: bool | None = None
    effect_bucket: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "delta_add_add": list(self.delta_add_add),
            "delta_add_remove": list(self.delta_add_remove),
            "delta_del_add": list(self.delta_del_add),
            "delta_del_remove": list(self.delta_del_remove),
            "success": self.success,
            "effect_bucket": self.effect_bucket,
        }


@dataclass(frozen=True)
class EpisodeEffectRepairPlan:
    should_apply_repair: bool
    repair_summary: str
    suspected_issue_sources: list[str]
    init_facts_add: list[str]
    init_facts_remove: list[str]
    goal_facts_add: list[str]
    goal_facts_remove: list[str]
    step_effect_repairs: list[EpisodeEffectRepairStepPatch]
    raw_llm_output: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "should_apply_repair": self.should_apply_repair,
            "repair_summary": self.repair_summary,
            "suspected_issue_sources": list(self.suspected_issue_sources),
            "init_facts_add": list(self.init_facts_add),
            "init_facts_remove": list(self.init_facts_remove),
            "goal_facts_add": list(self.goal_facts_add),
            "goal_facts_remove": list(self.goal_facts_remove),
            "step_effect_repairs": [item.to_dict() for item in self.step_effect_repairs],
        }
        if self.raw_llm_output:
            payload["raw_llm_output"] = self.raw_llm_output
        return payload


@dataclass(frozen=True)
class EpisodeGroundedEffectIteration:
    iteration_index: int
    problem_spec: ProblemSpec
    problem_pddl: str
    manipulation_records: list[ManipulationEffectRecord]
    grounded_steps: list[GroundedTrajectoryStep]
    validation_steps: list[ValidationStepReport]
    validation_issues: list[ValidationIssue]
    goal_satisfied: bool
    repair_plan: EpisodeEffectRepairPlan | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration_index": self.iteration_index,
            "problem_spec": self.problem_spec.to_dict(),
            "problem_pddl": self.problem_pddl,
            "manipulation_records": [item.to_dict() for item in self.manipulation_records],
            "grounded_steps": [item.to_dict() for item in self.grounded_steps],
            "validation_steps": [item.to_dict() for item in self.validation_steps],
            "validation_issues": [item.to_dict() for item in self.validation_issues],
            "goal_satisfied": self.goal_satisfied,
            "repair_plan": self.repair_plan.to_dict() if self.repair_plan is not None else None,
        }


@dataclass(frozen=True)
class EpisodeGroundedEffectResult:
    episode_name: str
    problem_spec: ProblemSpec
    problem_pddl: str
    manipulation_records: list[ManipulationEffectRecord]
    grounded_steps: list[GroundedTrajectoryStep]
    validation_steps: list[ValidationStepReport]
    validation_issues: list[ValidationIssue]
    goal_satisfied: bool
    iterations: list[EpisodeGroundedEffectIteration]
    problem_context: EpisodeProblemContextResult

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_name": self.episode_name,
            "problem_spec": self.problem_spec.to_dict(),
            "problem_pddl": self.problem_pddl,
            "manipulation_records": [item.to_dict() for item in self.manipulation_records],
            "grounded_steps": [item.to_dict() for item in self.grounded_steps],
            "validation_steps": [item.to_dict() for item in self.validation_steps],
            "validation_issues": [item.to_dict() for item in self.validation_issues],
            "goal_satisfied": self.goal_satisfied,
            "iterations": [item.to_dict() for item in self.iterations],
            "problem_context": {
                "problem_spec_without_goal": self.problem_context.problem_spec_without_goal.to_dict(),
                "problem_spec": self.problem_context.problem_spec.to_dict(),
                "object_init_raw_llm_outputs": dict(self.problem_context.object_init_raw_llm_outputs),
                "goal_inference_raw_output": self.problem_context.goal_inference_raw_output,
            },
        }


class EpisodeStepEffectModule(Protocol):
    def infer_step_effect(
        self,
        *,
        episode_context: EpisodeProblemContextResult,
        step: RawTrajectoryStep,
        taxonomy_record: ActionTaxonomyRecord,
        action_schema: ActionSchema,
        filtered_effect_candidates: list[str],
        allowed_predicates: list[PredicateSchema] | None = None,
        review_guidance: dict[str, object] | None = None,
    ) -> ManipulationEffectRecord: ...


class EpisodeEffectRepairModule(Protocol):
    def review_and_repair(
        self,
        *,
        episode_context: EpisodeProblemContextResult,
        action_schemas: list[ActionSchema],
        predicate_inventory: list[PredicateSchema] | None,
        iteration_history: list[EpisodeGroundedEffectIteration],
        current_problem_spec: ProblemSpec,
        current_records: list[ManipulationEffectRecord],
        current_grounded_steps: list[GroundedTrajectoryStep],
        current_validation_steps: list[ValidationStepReport],
        current_validation_issues: list[ValidationIssue],
        current_goal_satisfied: bool,
        final_state: set[str],
    ) -> EpisodeEffectRepairPlan: ...


@dataclass
class LLMEpisodeStepEffectModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1600
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("episode_effect_delta_prompt.md")

    def infer_step_effect(
        self,
        *,
        episode_context: EpisodeProblemContextResult,
        step: RawTrajectoryStep,
        taxonomy_record: ActionTaxonomyRecord,
        action_schema: ActionSchema,
        filtered_effect_candidates: list[str],
        allowed_predicates: list[PredicateSchema] | None = None,
        review_guidance: dict[str, object] | None = None,
    ) -> ManipulationEffectRecord:
        allowed_action_names = {action_schema.canonical_action_name}
        allowed_predicate_inventory = [item.to_dict() for item in allowed_predicates or []]
        allowed_predicates_set = {item.predicate_name for item in allowed_predicates or []}
        predicate_arities = _predicate_arity_map(allowed_predicates)
        payload = {
            "instruction": step.instruction,
            "current_action_schema": action_schema.to_dict(),
            "taxonomy_record": _taxonomy_record_prompt_dict(taxonomy_record, include_observation=False),
            "step": _step_prompt_dict_with_scene(step),
            "problem_spec_without_goal": episode_context.problem_spec_without_goal.to_dict(),
            "goal_facts": list(episode_context.problem_spec.goal_facts),
            "filtered_effect_candidates": list(filtered_effect_candidates),
        }
        if allowed_predicate_inventory:
            payload["allowed_predicates"] = allowed_predicate_inventory
        if review_guidance:
            payload["review_guidance"] = dict(review_guidance)
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
        action_name = _coerce_snake_case(
            _lookup_first(data, ["canonical_action_name", "action_name"]) or taxonomy_record.canonical_action_name,
            field_name="canonical_action_name",
        )
        if action_name not in allowed_action_names:
            raise ValueError(
                f"Episode step effect response returned unknown canonical_action_name {action_name!r}. "
                f"Allowed: {sorted(allowed_action_names)}"
            )
        action_arguments = _coerce_string_list(
            _lookup_first(data, ["action_arguments", "arguments"]) or taxonomy_record.action_arguments,
            field_name="action_arguments",
        )
        if action_arguments != list(taxonomy_record.action_arguments):
            raise ValueError(
                "Episode step effect response must preserve the existing action_arguments exactly. "
                f"Expected {taxonomy_record.action_arguments}, got {action_arguments}."
            )
        delta_add = _coerce_fact_list(_lookup_first(data, ["delta_add", "add_facts"]), field_name="delta_add")
        delta_del = _coerce_fact_list(_lookup_first(data, ["delta_del", "del_facts"]), field_name="delta_del")
        _validate_effect_literals_against_inventory(
            delta_add=delta_add,
            delta_del=delta_del,
            allowed_predicates=allowed_predicates_set,
            predicate_arities=predicate_arities,
            context=f"Episode step effect response [{step.episode_name} step {step.step_index}]",
        )
        success = _coerce_bool(
            _lookup_first(data, ["success"]) if "success" in data else (not _is_failure(step.extra_info)),
            field_name="success",
        )
        effect_bucket = _coerce_snake_case(
            _lookup_first(data, ["effect_bucket", "bucket"]) or f"{action_name}_{'success' if success else 'failure'}",
            field_name="effect_bucket",
        )
        return ManipulationEffectRecord(
            episode_name=step.episode_name,
            step_index=step.step_index,
            raw_action_text=taxonomy_record.raw_action_text,
            canonical_action_name=action_name,
            action_arguments=action_arguments,
            pre_observation_text=step.previous_observation_text,
            post_observation_text=step.observation_text,
            extra_info=step.extra_info,
            delta_add=delta_add,
            delta_del=delta_del,
            effect_bucket=effect_bucket,
            success=success,
            raw_llm_output=reply,
        )


@dataclass
class LLMEpisodeEffectRepairModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 3000
    max_validation_attempts: int = 3
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("episode_effect_repair_prompt.md")

    def review_and_repair(
        self,
        *,
        episode_context: EpisodeProblemContextResult,
        action_schemas: list[ActionSchema],
        predicate_inventory: list[PredicateSchema] | None,
        iteration_history: list[EpisodeGroundedEffectIteration],
        current_problem_spec: ProblemSpec,
        current_records: list[ManipulationEffectRecord],
        current_grounded_steps: list[GroundedTrajectoryStep],
        current_validation_steps: list[ValidationStepReport],
        current_validation_issues: list[ValidationIssue],
        current_goal_satisfied: bool,
        final_state: set[str],
    ) -> EpisodeEffectRepairPlan:
        payload = {
            "episode_context": episode_context.episode.to_dict(),
            "problem_spec": current_problem_spec.to_dict(),
            "final_state": sorted(final_state),
            "goal_satisfied": current_goal_satisfied,
            "action_schemas": [item.to_dict() for item in action_schemas],
            "allowed_predicates": [item.to_dict() for item in predicate_inventory or []],
            "manipulation_records": [item.to_dict() for item in current_records],
            "grounded_steps": [item.to_dict() for item in current_grounded_steps],
            "validation_steps": [item.to_dict() for item in current_validation_steps],
            "validation_issues": [item.to_dict() for item in current_validation_issues],
            "iteration_history": [item.to_dict() for item in iteration_history],
        }
        allowed_predicates_set = {item.predicate_name for item in predicate_inventory or []}
        predicate_arities = _predicate_arity_map(predicate_inventory)
        object_names = {item.name for item in current_problem_spec.objects}
        object_type_by_name = {item.name: item.type_name for item in current_problem_spec.objects}
        validation_feedback: str | None = None
        for attempt in range(1, self.max_validation_attempts + 1):
            request_payload = dict(payload)
            if validation_feedback:
                request_payload["previous_validation_error"] = validation_feedback
            reply = safe_chat(
                self._client,
                self._prompt,
                json.dumps(request_payload, ensure_ascii=False, indent=2),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                verbose=self.verbose,
            )
            data = extract_json_object(reply)
            try:
                should_apply_repair = bool(data.get("should_apply_repair"))
                repair_summary = str(data.get("repair_summary", "")).strip()
                suspected_issue_sources = (
                    [str(item).strip() for item in data.get("suspected_issue_sources", []) if str(item).strip()]
                    if isinstance(data.get("suspected_issue_sources", []), list)
                    else []
                )

                init_facts_add = _normalize_repair_fact_list_against_inventory(
                    _coerce_fact_list(data.get("init_facts_add", []), field_name="init_facts_add"),
                    predicate_inventory=predicate_inventory,
                    object_type_by_name=object_type_by_name,
                )
                init_facts_remove = _normalize_repair_fact_list_against_inventory(
                    _coerce_fact_list(data.get("init_facts_remove", []), field_name="init_facts_remove"),
                    predicate_inventory=predicate_inventory,
                    object_type_by_name=object_type_by_name,
                )
                goal_facts_add = _normalize_repair_fact_list_against_inventory(
                    _coerce_fact_list(data.get("goal_facts_add", []), field_name="goal_facts_add"),
                    predicate_inventory=predicate_inventory,
                    object_type_by_name=object_type_by_name,
                )
                goal_facts_remove = _normalize_repair_fact_list_against_inventory(
                    _coerce_fact_list(data.get("goal_facts_remove", []), field_name="goal_facts_remove"),
                    predicate_inventory=predicate_inventory,
                    object_type_by_name=object_type_by_name,
                )
                _validate_grounded_fact_lists(
                    [init_facts_add, init_facts_remove, goal_facts_add, goal_facts_remove],
                    allowed_predicates=allowed_predicates_set,
                    predicate_arities=predicate_arities,
                    object_names=object_names,
                    context="episode effect repair",
                )

                raw_step_effect_repairs = data.get("step_effect_repairs", [])
                if raw_step_effect_repairs is None:
                    raw_step_effect_repairs = []
                if not isinstance(raw_step_effect_repairs, list):
                    raise ValueError("step_effect_repairs must be a list.")
                step_effect_repairs: list[EpisodeEffectRepairStepPatch] = []
                for index, item in enumerate(raw_step_effect_repairs):
                    if not isinstance(item, dict):
                        raise ValueError(f"step_effect_repairs[{index}] must be an object.")
                    step_index = int(item.get("step_index", -1))
                    delta_add_add = _normalize_repair_fact_list_against_inventory(
                        _coerce_fact_list(item.get("delta_add_add", []), field_name="delta_add_add"),
                        predicate_inventory=predicate_inventory,
                        object_type_by_name=object_type_by_name,
                    )
                    delta_add_remove = _normalize_repair_fact_list_against_inventory(
                        _coerce_fact_list(item.get("delta_add_remove", []), field_name="delta_add_remove"),
                        predicate_inventory=predicate_inventory,
                        object_type_by_name=object_type_by_name,
                    )
                    delta_del_add = _normalize_repair_fact_list_against_inventory(
                        _coerce_fact_list(item.get("delta_del_add", []), field_name="delta_del_add"),
                        predicate_inventory=predicate_inventory,
                        object_type_by_name=object_type_by_name,
                    )
                    delta_del_remove = _normalize_repair_fact_list_against_inventory(
                        _coerce_fact_list(item.get("delta_del_remove", []), field_name="delta_del_remove"),
                        predicate_inventory=predicate_inventory,
                        object_type_by_name=object_type_by_name,
                    )
                    _validate_grounded_fact_lists(
                        [delta_add_add, delta_add_remove, delta_del_add, delta_del_remove],
                        allowed_predicates=allowed_predicates_set,
                        predicate_arities=predicate_arities,
                        object_names=object_names,
                        context=f"episode effect repair step {step_index}",
                    )
                    success_raw = item.get("success")
                    success = None if success_raw is None else _coerce_bool(success_raw, field_name="success")
                    effect_bucket_raw = item.get("effect_bucket")
                    effect_bucket = None
                    if effect_bucket_raw is not None and str(effect_bucket_raw).strip():
                        effect_bucket = _coerce_snake_case(effect_bucket_raw, field_name="effect_bucket")
                    step_effect_repairs.append(
                        EpisodeEffectRepairStepPatch(
                            step_index=step_index,
                            delta_add_add=delta_add_add,
                            delta_add_remove=delta_add_remove,
                            delta_del_add=delta_del_add,
                            delta_del_remove=delta_del_remove,
                            success=success,
                            effect_bucket=effect_bucket,
                        )
                    )

                return EpisodeEffectRepairPlan(
                    should_apply_repair=should_apply_repair,
                    repair_summary=repair_summary,
                    suspected_issue_sources=suspected_issue_sources,
                    init_facts_add=init_facts_add,
                    init_facts_remove=init_facts_remove,
                    goal_facts_add=goal_facts_add,
                    goal_facts_remove=goal_facts_remove,
                    step_effect_repairs=step_effect_repairs,
                    raw_llm_output=reply,
                )
            except ValueError as exc:
                if attempt >= self.max_validation_attempts:
                    raise
                validation_feedback = str(exc)
                logger.warning(
                    "Episode effect repair validation failed attempt %d/%d: %s. Retrying repair prompt.",
                    attempt,
                    self.max_validation_attempts,
                    validation_feedback,
                )
                continue
        raise AssertionError("Unreachable: episode effect repair validation retry loop exhausted without returning.")


@dataclass
class LLMEpisodeGroundedEffectLearningModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 3000
    max_workers: int = 1
    max_iterations: int = 3
    verbose: bool = False
    problem_context_module: EpisodeProblemContextModule | None = None
    step_effect_module: EpisodeStepEffectModule | None = None
    repair_module: EpisodeEffectRepairModule | None = None

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {self.max_workers}")
        if self.max_iterations < 0:
            raise ValueError(f"max_iterations must be at least 0, got {self.max_iterations}")
        if self.problem_context_module is None:
            self.problem_context_module = EpisodeProblemContextModule(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_workers=self.max_workers,
                verbose=self.verbose,
            )
        if self.step_effect_module is None:
            self.step_effect_module = LLMEpisodeStepEffectModule(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                verbose=self.verbose,
            )
        if self.repair_module is None:
            self.repair_module = LLMEpisodeEffectRepairModule(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                verbose=self.verbose,
            )
        self.last_episode_results: dict[str, EpisodeGroundedEffectResult] = {}
        self.last_post_statistics_repair_results: dict[str, EpisodeGroundedEffectResult] = {}
        self.last_post_statistics_repair_summary: dict[str, object] = {}

    def learn_effects(
        self,
        steps: list[RawTrajectoryStep],
        action_schemas: list[ActionSchema],
        taxonomy_records: list[ActionTaxonomyRecord],
        predicate_inventory: list[PredicateSchema] | None = None,
        object_types: list[ObjectTypeDefinition] | None = None,
        review_guidance_by_episode_step: dict[tuple[str, int], dict[str, object]] | None = None,
    ) -> list[ManipulationEffectRecord]:
        logger.info(
            "Episode-grounded effect learning: building per-episode problem contexts for %d step(s) across %d episode(s).",
            len(steps),
            len({step.episode_name for step in steps}),
        )
        episode_contexts = self.problem_context_module.build_episode_problem_contexts(
            steps=steps,
            action_schemas=action_schemas,
            taxonomy_records=taxonomy_records,
            predicate_inventory=predicate_inventory,
            object_types=object_types,
        )
        grouped_steps = _group_steps_by_episode(steps)
        grouped_taxonomy = _group_taxonomy_by_episode(taxonomy_records)
        action_schema_map = {item.canonical_action_name: item for item in action_schemas}
        episode_names = sorted(set(grouped_steps) & set(episode_contexts))

        if self.max_workers == 1:
            episode_results = [
                self._process_episode(
                    episode_name=episode_name,
                    problem_context=episode_contexts[episode_name],
                    steps=grouped_steps.get(episode_name, []),
                    taxonomy_records=grouped_taxonomy.get(episode_name, []),
                    action_schema_map=action_schema_map,
                    predicate_inventory=predicate_inventory,
                    review_guidance_by_episode_step=review_guidance_by_episode_step or {},
                )
                for episode_name in episode_names
            ]
        else:
            episode_results = [None] * len(episode_names)
            with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="episode-effects") as executor:
                future_to_index = {
                    executor.submit(
                        self._process_episode,
                        episode_name=episode_name,
                        problem_context=episode_contexts[episode_name],
                        steps=grouped_steps.get(episode_name, []),
                        taxonomy_records=grouped_taxonomy.get(episode_name, []),
                        action_schema_map=action_schema_map,
                        predicate_inventory=predicate_inventory,
                        review_guidance_by_episode_step=review_guidance_by_episode_step or {},
                    ): index
                    for index, episode_name in enumerate(episode_names)
                }
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    episode_results[index] = future.result()

        merged_records = [
            record for result in episode_results if result is not None for record in result.manipulation_records
        ]
        self.last_episode_results = {result.episode_name: result for result in episode_results if result is not None}
        return merged_records

    def _process_episode(
        self,
        *,
        episode_name: str,
        problem_context: EpisodeProblemContextResult,
        steps: list[RawTrajectoryStep],
        taxonomy_records: list[ActionTaxonomyRecord],
        action_schema_map: dict[str, ActionSchema],
        predicate_inventory: list[PredicateSchema] | None,
        review_guidance_by_episode_step: dict[tuple[str, int], dict[str, object]],
    ) -> EpisodeGroundedEffectResult:
        logger.debug(
            "Episode-grounded effect learning: start episode %s with %d step(s) and %d taxonomy record(s).",
            episode_name,
            len(steps),
            len(taxonomy_records),
        )
        manipulation_records = self._infer_episode_step_effects(
            problem_context=problem_context,
            steps=steps,
            taxonomy_records=taxonomy_records,
            action_schema_map=action_schema_map,
            predicate_inventory=predicate_inventory,
            review_guidance_by_episode_step=review_guidance_by_episode_step,
        )
        logger.debug(
            "Episode-grounded effect learning: episode %s problem context has %d object(s), %d init fact(s), %d goal fact(s).",
            episode_name,
            len(problem_context.problem_spec.objects),
            len(problem_context.problem_spec.init_facts),
            len(problem_context.problem_spec.goal_facts),
        )
        result = self._run_episode_repair_loop(
            episode_name=episode_name,
            problem_context=problem_context,
            current_problem_spec=problem_context.problem_spec,
            current_records=manipulation_records,
            action_schema_map=action_schema_map,
            predicate_inventory=predicate_inventory,
        )
        logger.debug(
            "Episode-grounded effect learning: finish episode %s with goal_satisfied=%s after %d iteration(s).",
            episode_name,
            result.goal_satisfied,
            len(result.iterations),
        )
        return result

    def revalidate_and_repair_records(
        self,
        *,
        steps: list[RawTrajectoryStep],
        action_schemas: list[ActionSchema],
        taxonomy_records: list[ActionTaxonomyRecord],
        records: list[ManipulationEffectRecord],
        predicate_inventory: list[PredicateSchema] | None = None,
    ) -> tuple[list[ManipulationEffectRecord], bool]:
        base_episode_results = dict(self.last_episode_results)
        if not base_episode_results:
            logger.info(
                "Episode-grounded effect learning post-merge validation: no previous episode contexts available; skipping.",
            )
            self.last_post_statistics_repair_results = {}
            self.last_post_statistics_repair_summary = {
                "episode_count": 0,
                "changed_record_episode_names": [],
                "goal_failed_episode_names": [],
            }
            return list(records), False

        grouped_steps = _group_steps_by_episode(steps)
        grouped_taxonomy = _group_taxonomy_by_episode(taxonomy_records)
        grouped_records = _group_records_by_episode(records)
        action_schema_map = {item.canonical_action_name: item for item in action_schemas}
        episode_names = sorted(base_episode_results)
        logger.info(
            "Episode-grounded effect learning post-merge validation: replaying %d episode(s) after record/statistics merge.",
            len(episode_names),
        )

        if self.max_workers == 1 or len(episode_names) <= 1:
            episode_results = [
                self._revalidate_episode_records(
                    episode_name=episode_name,
                    base_result=base_episode_results[episode_name],
                    steps=grouped_steps.get(episode_name, []),
                    taxonomy_records=grouped_taxonomy.get(episode_name, []),
                    current_records=grouped_records.get(episode_name, []),
                    action_schema_map=action_schema_map,
                    predicate_inventory=predicate_inventory,
                )
                for episode_name in episode_names
            ]
        else:
            episode_results = [None] * len(episode_names)
            with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="episode-post-merge") as executor:
                future_to_index = {
                    executor.submit(
                        self._revalidate_episode_records,
                        episode_name=episode_name,
                        base_result=base_episode_results[episode_name],
                        steps=grouped_steps.get(episode_name, []),
                        taxonomy_records=grouped_taxonomy.get(episode_name, []),
                        current_records=grouped_records.get(episode_name, []),
                        action_schema_map=action_schema_map,
                        predicate_inventory=predicate_inventory,
                    ): index
                    for index, episode_name in enumerate(episode_names)
                }
                for future in as_completed(future_to_index):
                    episode_results[future_to_index[future]] = future.result()

        changed_record_episode_names: list[str] = []
        merged_records: list[ManipulationEffectRecord] = []
        results_by_episode: dict[str, EpisodeGroundedEffectResult] = {}
        for episode_name, result in zip(episode_names, episode_results):
            if result is None:
                continue
            results_by_episode[episode_name] = result
            merged_records.extend(result.manipulation_records)
            original_records = grouped_records.get(episode_name, [])
            if result.manipulation_records != original_records:
                changed_record_episode_names.append(episode_name)

        self.last_post_statistics_repair_results = results_by_episode
        self.last_post_statistics_repair_summary = {
            "episode_count": len(results_by_episode),
            "changed_record_episode_names": sorted(changed_record_episode_names),
            "goal_failed_episode_names": sorted(
                episode_name for episode_name, result in results_by_episode.items() if not result.goal_satisfied
            ),
            "episode_results": {
                episode_name: result.to_dict() for episode_name, result in sorted(results_by_episode.items())
            },
        }
        logger.info(
            "Episode-grounded effect learning post-merge validation: %d/%d episode(s) changed records.",
            len(changed_record_episode_names),
            len(results_by_episode),
        )
        return merged_records, bool(changed_record_episode_names)

    def prune_replay_noop_effects(
        self,
        records: list[ManipulationEffectRecord],
    ) -> tuple[list[ManipulationEffectRecord], bool]:
        episode_results = self.last_post_statistics_repair_results or self.last_episode_results
        state_before_by_step = {
            (episode_name, report.step_index): list(report.state_before)
            for episode_name, result in episode_results.items()
            for report in result.validation_steps
        }
        pruned_records, changed_count = prune_replay_noop_effects(
            records,
            state_before_by_step=state_before_by_step,
        )
        if changed_count:
            logger.info(
                "Episode-grounded effect learning: pruned replay-confirmed no-op effects from %d record(s).",
                changed_count,
            )
        return pruned_records, changed_count > 0

    def rerun_records_from_effect_guidance(
        self,
        *,
        steps: list[RawTrajectoryStep],
        action_schemas: list[ActionSchema],
        taxonomy_records: list[ActionTaxonomyRecord],
        records: list[ManipulationEffectRecord],
        predicate_inventory: list[PredicateSchema] | None = None,
        review_guidance_by_signature: dict[tuple[str, str, int | None], dict[str, object]] | None = None,
    ) -> tuple[list[ManipulationEffectRecord], bool]:
        guidance_by_signature = {
            key: dict(value) for key, value in (review_guidance_by_signature or {}).items() if value
        }
        if not guidance_by_signature:
            self.last_post_statistics_repair_summary = {
                "episode_count": 0,
                "changed_record_episode_names": [],
                "rerun_from_step_by_episode": {},
                "mode": "local_effect_guidance_rerun",
            }
            return list(records), False

        base_episode_results = dict(self.last_post_statistics_repair_results or self.last_episode_results)
        if not base_episode_results:
            logger.info(
                "Episode-grounded effect learning local rerun: no prior episode results available; skipping.",
            )
            self.last_post_statistics_repair_summary = {
                "episode_count": 0,
                "changed_record_episode_names": [],
                "rerun_from_step_by_episode": {},
                "mode": "local_effect_guidance_rerun",
            }
            return list(records), False

        grouped_steps = _group_steps_by_episode(steps)
        grouped_taxonomy = _group_taxonomy_by_episode(taxonomy_records)
        grouped_records = _group_records_by_episode(records)
        action_schema_map = {item.canonical_action_name: item for item in action_schemas}
        rerun_from_step_by_episode = _collect_guided_rerun_targets(
            records=records,
            guidance_by_signature=guidance_by_signature,
        )
        if not rerun_from_step_by_episode:
            logger.info(
                "Episode-grounded effect learning local rerun: no records matched the requested bucket/variant guidance.",
            )
            self.last_post_statistics_repair_summary = {
                "episode_count": 0,
                "changed_record_episode_names": [],
                "rerun_from_step_by_episode": {},
                "mode": "local_effect_guidance_rerun",
            }
            return list(records), False

        episode_names = sorted(grouped_records)
        episode_results: list[EpisodeGroundedEffectResult | None] = [None] * len(episode_names)

        def _run_one(episode_name: str) -> EpisodeGroundedEffectResult:
            base_result = base_episode_results[episode_name]
            rerun_from_step = rerun_from_step_by_episode.get(episode_name)
            if rerun_from_step is None:
                return base_result
            logger.info(
                "Episode-grounded effect learning local rerun: episode %s will recompute effects from step %d onward.",
                episode_name,
                rerun_from_step,
            )
            return self._rerun_episode_from_step(
                episode_name=episode_name,
                base_result=base_result,
                steps=grouped_steps.get(episode_name, []),
                taxonomy_records=grouped_taxonomy.get(episode_name, []),
                current_records=grouped_records.get(episode_name, []),
                action_schema_map=action_schema_map,
                predicate_inventory=predicate_inventory,
                review_guidance_by_signature=guidance_by_signature,
                rerun_from_step=rerun_from_step,
            )

        if self.max_workers == 1 or len(episode_names) <= 1:
            for index, episode_name in enumerate(episode_names):
                episode_results[index] = _run_one(episode_name)
        else:
            with ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="episode-guided-rerun"
            ) as executor:
                future_to_index = {
                    executor.submit(_run_one, episode_name): index for index, episode_name in enumerate(episode_names)
                }
                for future in as_completed(future_to_index):
                    episode_results[future_to_index[future]] = future.result()

        merged_records: list[ManipulationEffectRecord] = []
        results_by_episode: dict[str, EpisodeGroundedEffectResult] = {}
        changed_record_episode_names: list[str] = []
        for episode_name, result in zip(episode_names, episode_results):
            if result is None:
                continue
            results_by_episode[episode_name] = result
            merged_records.extend(result.manipulation_records)
            if result.manipulation_records != grouped_records.get(episode_name, []):
                changed_record_episode_names.append(episode_name)

        self.last_post_statistics_repair_results = results_by_episode
        self.last_post_statistics_repair_summary = {
            "episode_count": len(results_by_episode),
            "changed_record_episode_names": sorted(changed_record_episode_names),
            "rerun_from_step_by_episode": {
                episode_name: rerun_from_step_by_episode[episode_name]
                for episode_name in sorted(rerun_from_step_by_episode)
            },
            "mode": "local_effect_guidance_rerun",
            "episode_results": {
                episode_name: result.to_dict() for episode_name, result in sorted(results_by_episode.items())
            },
        }
        logger.info(
            "Episode-grounded effect learning local rerun: %d/%d episode(s) changed records.",
            len(changed_record_episode_names),
            len(results_by_episode),
        )
        return merged_records, bool(changed_record_episode_names)

    def _infer_episode_step_effects(
        self,
        *,
        problem_context: EpisodeProblemContextResult,
        steps: list[RawTrajectoryStep],
        taxonomy_records: list[ActionTaxonomyRecord],
        action_schema_map: dict[str, ActionSchema],
        predicate_inventory: list[PredicateSchema] | None,
        review_guidance_by_episode_step: dict[tuple[str, int], dict[str, object]],
    ) -> list[ManipulationEffectRecord]:
        step_map = {step.step_index: step for step in steps}
        manipulation_taxonomy_records = [
            item
            for item in sorted(taxonomy_records, key=lambda row: row.step_index)
            if item.action_category == "manipulation"
        ]
        outputs: list[ManipulationEffectRecord] = []
        for taxonomy_record in manipulation_taxonomy_records:
            step = step_map[taxonomy_record.step_index]
            schema = action_schema_map.get(taxonomy_record.canonical_action_name)
            if schema is None:
                raise ValueError(
                    f"Episode grounded effect learning cannot find action schema for "
                    f"{taxonomy_record.canonical_action_name!r}."
                )
            filtered_effect_candidates = _filtered_effect_candidates(
                predicate_inventory=predicate_inventory,
                action_arguments=taxonomy_record.action_arguments,
                action_argument_types=taxonomy_record.action_argument_types or schema.parameter_roles,
            )
            logger.debug(
                "Episode-grounded effect learning: episode %s step %d filtered_effect_candidates=%s",
                problem_context.episode_name,
                taxonomy_record.step_index,
                filtered_effect_candidates,
            )
            review_guidance = dict(
                review_guidance_by_episode_step.get((problem_context.episode_name, taxonomy_record.step_index), {})
            )
            try:
                record = self.step_effect_module.infer_step_effect(
                    episode_context=problem_context,
                    step=step,
                    taxonomy_record=taxonomy_record,
                    action_schema=schema,
                    filtered_effect_candidates=filtered_effect_candidates,
                    allowed_predicates=predicate_inventory,
                    review_guidance=review_guidance or None,
                )
            except TypeError as exc:
                if "review_guidance" not in str(exc):
                    raise
                record = self.step_effect_module.infer_step_effect(
                    episode_context=problem_context,
                    step=step,
                    taxonomy_record=taxonomy_record,
                    action_schema=schema,
                    filtered_effect_candidates=filtered_effect_candidates,
                    allowed_predicates=predicate_inventory,
                )
            outputs.append(record)
            logger.debug(
                "Episode-grounded effect learning: episode %s step %d inferred bucket=%s success=%s delta_add=%s delta_del=%s",
                problem_context.episode_name,
                taxonomy_record.step_index,
                record.effect_bucket,
                record.success,
                record.delta_add,
                record.delta_del,
            )
        return outputs

    def _rerun_episode_from_step(
        self,
        *,
        episode_name: str,
        base_result: EpisodeGroundedEffectResult,
        steps: list[RawTrajectoryStep],
        taxonomy_records: list[ActionTaxonomyRecord],
        current_records: list[ManipulationEffectRecord],
        action_schema_map: dict[str, ActionSchema],
        predicate_inventory: list[PredicateSchema] | None,
        review_guidance_by_signature: dict[tuple[str, str, int | None], dict[str, object]],
        rerun_from_step: int,
    ) -> EpisodeGroundedEffectResult:
        prefix_records = [item for item in current_records if item.step_index < rerun_from_step]
        suffix_steps = [item for item in steps if item.step_index >= rerun_from_step]
        suffix_taxonomy = [item for item in taxonomy_records if item.step_index >= rerun_from_step]
        suffix_guidance = _expand_guidance_for_records(
            records=[item for item in current_records if item.step_index >= rerun_from_step],
            guidance_by_signature=review_guidance_by_signature,
        )
        state_before_rerun = _state_before_step(
            base_result=base_result,
            rerun_from_step=rerun_from_step,
        )
        suffix_problem_spec = ProblemSpec(
            problem_name=base_result.problem_spec.problem_name,
            domain_name=base_result.problem_spec.domain_name,
            objects=list(base_result.problem_spec.objects),
            init_facts=sorted(state_before_rerun),
            goal_facts=list(base_result.problem_spec.goal_facts),
            canonical_object_map=dict(base_result.problem_spec.canonical_object_map),
        )
        suffix_records = self._infer_episode_step_effects(
            problem_context=base_result.problem_context,
            steps=suffix_steps,
            taxonomy_records=suffix_taxonomy,
            action_schema_map=action_schema_map,
            predicate_inventory=predicate_inventory,
            review_guidance_by_episode_step=suffix_guidance,
        )
        suffix_result = self._run_episode_repair_loop(
            episode_name=episode_name,
            problem_context=base_result.problem_context,
            current_problem_spec=suffix_problem_spec,
            current_records=suffix_records,
            action_schema_map=action_schema_map,
            predicate_inventory=predicate_inventory,
        )
        prefix_grounded_steps = [item for item in base_result.grounded_steps if item.step_index < rerun_from_step]
        prefix_validation_steps = [item for item in base_result.validation_steps if item.step_index < rerun_from_step]
        prefix_validation_issues = [item for item in base_result.validation_issues if item.step_index < rerun_from_step]
        return EpisodeGroundedEffectResult(
            episode_name=episode_name,
            problem_spec=base_result.problem_spec,
            problem_pddl=render_problem_pddl(base_result.problem_spec),
            manipulation_records=prefix_records + suffix_result.manipulation_records,
            grounded_steps=prefix_grounded_steps + suffix_result.grounded_steps,
            validation_steps=prefix_validation_steps + suffix_result.validation_steps,
            validation_issues=prefix_validation_issues + suffix_result.validation_issues,
            goal_satisfied=suffix_result.goal_satisfied,
            iterations=suffix_result.iterations,
            problem_context=base_result.problem_context,
        )

    def _revalidate_episode_records(
        self,
        *,
        episode_name: str,
        base_result: EpisodeGroundedEffectResult,
        steps: list[RawTrajectoryStep],
        taxonomy_records: list[ActionTaxonomyRecord],
        current_records: list[ManipulationEffectRecord],
        action_schema_map: dict[str, ActionSchema],
        predicate_inventory: list[PredicateSchema] | None,
    ) -> EpisodeGroundedEffectResult:
        logger.debug(
            "Episode-grounded effect learning post-merge validation: start episode %s with %d record(s).",
            episode_name,
            len(current_records),
        )
        if steps:
            logger.debug(
                "Episode-grounded effect learning post-merge validation: episode %s has %d step(s) and %d taxonomy record(s).",
                episode_name,
                len(steps),
                len(taxonomy_records),
            )
        result = self._run_episode_repair_loop(
            episode_name=episode_name,
            problem_context=base_result.problem_context,
            current_problem_spec=base_result.problem_spec,
            current_records=list(current_records),
            action_schema_map=action_schema_map,
            predicate_inventory=predicate_inventory,
        )
        logger.debug(
            "Episode-grounded effect learning post-merge validation: finish episode %s with goal_satisfied=%s after %d iteration(s).",
            episode_name,
            result.goal_satisfied,
            len(result.iterations),
        )
        return result

    def _run_episode_repair_loop(
        self,
        *,
        episode_name: str,
        problem_context: EpisodeProblemContextResult,
        current_problem_spec: ProblemSpec,
        current_records: list[ManipulationEffectRecord],
        action_schema_map: dict[str, ActionSchema],
        predicate_inventory: list[PredicateSchema] | None,
    ) -> EpisodeGroundedEffectResult:
        iterations: list[EpisodeGroundedEffectIteration] = []
        final_grounded_steps: list[GroundedTrajectoryStep] = []
        final_validation_steps: list[ValidationStepReport] = []
        final_validation_issues: list[ValidationIssue] = []
        final_goal_satisfied = False
        final_state: set[str] = set()

        for iteration_index in range(self.max_iterations + 1):
            logger.debug(
                "Episode-grounded effect learning: episode %s iteration %d/%d replay start.",
                episode_name,
                iteration_index,
                self.max_iterations,
            )
            grounded_steps, validation_steps, validation_issues, goal_satisfied, final_state = _replay_episode_records(
                problem_spec=current_problem_spec,
                manipulation_records=current_records,
                action_schemas=action_schema_map,
                predicate_inventory=predicate_inventory,
            )
            logger.debug(
                "Episode-grounded effect learning: episode %s iteration %d replay result: goal_satisfied=%s, validation_issues=%d.",
                episode_name,
                iteration_index,
                goal_satisfied,
                len(validation_issues),
            )
            logger.debug(
                "Episode-grounded effect learning: episode %s iteration %d final_state=%s",
                episode_name,
                iteration_index,
                sorted(final_state),
            )
            repair_plan = None
            if goal_satisfied:
                iterations.append(
                    EpisodeGroundedEffectIteration(
                        iteration_index=iteration_index,
                        problem_spec=current_problem_spec,
                        problem_pddl=render_problem_pddl(current_problem_spec),
                        manipulation_records=current_records,
                        grounded_steps=grounded_steps,
                        validation_steps=validation_steps,
                        validation_issues=validation_issues,
                        goal_satisfied=goal_satisfied,
                    )
                )
                final_grounded_steps = grounded_steps
                final_validation_steps = validation_steps
                final_validation_issues = validation_issues
                final_goal_satisfied = goal_satisfied
                logger.debug(
                    "Episode-grounded effect learning: episode %s converged at iteration %d.",
                    episode_name,
                    iteration_index,
                )
                break
            if iteration_index >= self.max_iterations:
                iterations.append(
                    EpisodeGroundedEffectIteration(
                        iteration_index=iteration_index,
                        problem_spec=current_problem_spec,
                        problem_pddl=render_problem_pddl(current_problem_spec),
                        manipulation_records=current_records,
                        grounded_steps=grounded_steps,
                        validation_steps=validation_steps,
                        validation_issues=validation_issues,
                        goal_satisfied=goal_satisfied,
                    )
                )
                final_grounded_steps = grounded_steps
                final_validation_steps = validation_steps
                final_validation_issues = validation_issues
                final_goal_satisfied = goal_satisfied
                logger.warning(
                    "Episode-grounded effect learning: episode %s reached max_iterations=%d without satisfying goal.",
                    episode_name,
                    self.max_iterations,
                )
                break
            logger.info(
                "Episode-grounded effect learning: episode %s iteration %d requesting review/repair.",
                episode_name,
                iteration_index,
            )
            repair_plan = self.repair_module.review_and_repair(
                episode_context=problem_context,
                action_schemas=list(action_schema_map.values()),
                predicate_inventory=predicate_inventory,
                iteration_history=iterations,
                current_problem_spec=current_problem_spec,
                current_records=current_records,
                current_grounded_steps=grounded_steps,
                current_validation_steps=validation_steps,
                current_validation_issues=validation_issues,
                current_goal_satisfied=goal_satisfied,
                final_state=final_state,
            )
            iterations.append(
                EpisodeGroundedEffectIteration(
                    iteration_index=iteration_index,
                    problem_spec=current_problem_spec,
                    problem_pddl=render_problem_pddl(current_problem_spec),
                    manipulation_records=current_records,
                    grounded_steps=grounded_steps,
                    validation_steps=validation_steps,
                    validation_issues=validation_issues,
                    goal_satisfied=goal_satisfied,
                    repair_plan=repair_plan,
                )
            )
            logger.info(
                "Episode-grounded effect learning: episode %s iteration %d repair decision: should_apply_repair=%s, suspected_sources=%s, summary=%s",
                episode_name,
                iteration_index,
                repair_plan.should_apply_repair,
                ",".join(repair_plan.suspected_issue_sources) if repair_plan.suspected_issue_sources else "<none>",
                repair_plan.repair_summary or "<empty>",
            )
            if not repair_plan.should_apply_repair:
                final_grounded_steps = grounded_steps
                final_validation_steps = validation_steps
                final_validation_issues = validation_issues
                final_goal_satisfied = goal_satisfied
                logger.info(
                    "Episode-grounded effect learning: episode %s iteration %d review declined repair; stopping.",
                    episode_name,
                    iteration_index,
                )
                break
            current_problem_spec = _apply_problem_spec_repair(current_problem_spec, repair_plan)
            current_records = _apply_step_effect_repairs(current_records, repair_plan)
            logger.info(
                "Episode-grounded effect learning: episode %s iteration %d applied repair: init(+%d/-%d) goal(+%d/-%d) step_repairs=%d.",
                episode_name,
                iteration_index,
                len(repair_plan.init_facts_add),
                len(repair_plan.init_facts_remove),
                len(repair_plan.goal_facts_add),
                len(repair_plan.goal_facts_remove),
                len(repair_plan.step_effect_repairs),
            )
            final_grounded_steps = grounded_steps
            final_validation_steps = validation_steps
            final_validation_issues = validation_issues
            final_goal_satisfied = goal_satisfied

        return EpisodeGroundedEffectResult(
            episode_name=episode_name,
            problem_spec=current_problem_spec,
            problem_pddl=render_problem_pddl(current_problem_spec),
            manipulation_records=current_records,
            grounded_steps=final_grounded_steps,
            validation_steps=final_validation_steps,
            validation_issues=final_validation_issues,
            goal_satisfied=final_goal_satisfied,
            iterations=iterations,
            problem_context=problem_context,
        )


def _step_prompt_dict_with_scene(step: RawTrajectoryStep) -> dict[str, object]:
    data = step.to_dict()
    data["previous_observation_text"] = None
    data["previous_known_observation_text"] = None
    return data


def _group_steps_by_episode(steps: list[RawTrajectoryStep]) -> dict[str, list[RawTrajectoryStep]]:
    grouped: dict[str, list[RawTrajectoryStep]] = {}
    for step in steps:
        grouped.setdefault(step.episode_name, []).append(step)
    for episode_name in list(grouped):
        grouped[episode_name] = sorted(grouped[episode_name], key=lambda item: item.step_index)
    return grouped


def _group_taxonomy_by_episode(
    taxonomy_records: list[ActionTaxonomyRecord],
) -> dict[str, list[ActionTaxonomyRecord]]:
    grouped: dict[str, list[ActionTaxonomyRecord]] = {}
    for record in taxonomy_records:
        grouped.setdefault(record.episode_name, []).append(record)
    for episode_name in list(grouped):
        grouped[episode_name] = sorted(grouped[episode_name], key=lambda item: item.step_index)
    return grouped


def _group_records_by_episode(
    records: list[ManipulationEffectRecord],
) -> dict[str, list[ManipulationEffectRecord]]:
    grouped: dict[str, list[ManipulationEffectRecord]] = {}
    for record in records:
        grouped.setdefault(record.episode_name, []).append(record)
    for episode_name in list(grouped):
        grouped[episode_name] = sorted(grouped[episode_name], key=lambda item: item.step_index)
    return grouped


def _bucket_variant_rank(effect_bucket: str | None) -> int | None:
    bucket_text = str(effect_bucket or "").strip()
    if not bucket_text:
        return None
    prefix, separator, suffix = bucket_text.rpartition("_bucket_")
    if not separator:
        return None
    return int(suffix) if suffix.isdigit() else None


def _effect_guidance_signature_key(
    canonical_action_name: str,
    effect_bucket: str,
    variant_rank: int | None,
) -> tuple[str, str, int | None]:
    return (str(canonical_action_name), str(effect_bucket), int(variant_rank) if variant_rank is not None else None)


def _record_guidance_signature_keys(record: ManipulationEffectRecord) -> list[tuple[str, str, int | None]]:
    record_variant_rank = _bucket_variant_rank(record.effect_bucket)
    keys = [
        _effect_guidance_signature_key(
            record.canonical_action_name,
            record.effect_bucket,
            record_variant_rank,
        )
    ]
    if record_variant_rank is None:
        keys.append(
            _effect_guidance_signature_key(
                record.canonical_action_name,
                record.effect_bucket,
                None,
            )
        )
    return keys


def _expand_guidance_for_records(
    *,
    records: list[ManipulationEffectRecord],
    guidance_by_signature: dict[tuple[str, str, int | None], dict[str, object]],
) -> dict[tuple[str, int], dict[str, object]]:
    expanded: dict[tuple[str, int], dict[str, object]] = {}
    for record in records:
        for key in _record_guidance_signature_keys(record):
            guidance = guidance_by_signature.get(key)
            if not guidance:
                continue
            expanded[(record.episode_name, record.step_index)] = dict(guidance)
            break
    return expanded


def _collect_guided_rerun_targets(
    *,
    records: list[ManipulationEffectRecord],
    guidance_by_signature: dict[tuple[str, str, int | None], dict[str, object]],
) -> dict[str, int]:
    rerun_from_step_by_episode: dict[str, int] = {}
    for record in records:
        matched = any(guidance_by_signature.get(key) for key in _record_guidance_signature_keys(record))
        if not matched:
            continue
        existing = rerun_from_step_by_episode.get(record.episode_name)
        if existing is None or record.step_index < existing:
            rerun_from_step_by_episode[record.episode_name] = record.step_index
    return rerun_from_step_by_episode


def _state_before_step(
    *,
    base_result: EpisodeGroundedEffectResult,
    rerun_from_step: int,
) -> set[str]:
    for report in base_result.validation_steps:
        if report.step_index == rerun_from_step:
            return set(report.state_before)
    if rerun_from_step <= 0:
        return set(base_result.problem_spec.init_facts)
    latest_state = set(base_result.problem_spec.init_facts)
    for report in base_result.validation_steps:
        if report.step_index < rerun_from_step:
            latest_state = set(report.state_after)
    return latest_state


def _type_matches_argument(expected_type: str | None, argument_type: str | None) -> bool:
    normalized_expected = str(expected_type or "").strip()
    normalized_argument = str(argument_type or "").strip()
    if not normalized_expected:
        return True
    if not normalized_argument:
        return True
    if normalized_expected == normalized_argument:
        return True
    if normalized_expected in {"movable_item", "fixed_item"}:
        return True
    return False


def _filtered_effect_candidates(
    *,
    predicate_inventory: list[PredicateSchema] | None,
    action_arguments: list[str],
    action_argument_types: list[str] | None,
) -> list[str]:
    if not predicate_inventory:
        return []
    normalized_argument_types = list(action_argument_types or [])
    if len(normalized_argument_types) < len(action_arguments):
        normalized_argument_types.extend([""] * (len(action_arguments) - len(normalized_argument_types)))
    argument_rows = list(zip(action_arguments, normalized_argument_types))
    candidates: set[str] = set()
    for predicate in predicate_inventory:
        parameter_types = list(predicate.parameter_types)
        if not parameter_types:
            candidates.add(f"{predicate.predicate_name}()")
            continue
        compatible_argument_options: list[list[str]] = []
        for expected_type in parameter_types:
            compatible_arguments = [
                argument_name
                for argument_name, argument_type in argument_rows
                if _type_matches_argument(expected_type, argument_type)
            ]
            if not compatible_arguments:
                compatible_argument_options = []
                break
            compatible_argument_options.append(compatible_arguments)
        if not compatible_argument_options:
            continue
        for argument_tuple in product(*compatible_argument_options):
            candidates.add(f"{predicate.predicate_name}({','.join(argument_tuple)})")
    return sorted(candidates)


def _render_ground_action_pddl(action_name: str, arguments: list[str]) -> str:
    if arguments:
        return f"({action_name} {' '.join(arguments)})"
    return f"({action_name})"


def _apply_effects_to_state(state: set[str], delta_add: list[str], delta_del: list[str]) -> set[str]:
    next_state = {normalize_fact_key(fact) for fact in state}
    for fact in delta_del:
        next_state.discard(normalize_fact_key(fact))
    for fact in delta_add:
        next_state.add(normalize_fact_key(fact))
    return next_state


def _replay_episode_records(
    *,
    problem_spec: ProblemSpec,
    manipulation_records: list[ManipulationEffectRecord],
    action_schemas: dict[str, ActionSchema],
    predicate_inventory: list[PredicateSchema] | None,
) -> tuple[list[GroundedTrajectoryStep], list[ValidationStepReport], list[ValidationIssue], bool, set[str]]:
    domain_text = render_action_schema_fragment(
        list(action_schemas.values()),
        predicate_inventory=predicate_inventory,
    )
    parsed_domain = parse_domain(domain_text)
    parse_problem(render_problem_pddl(problem_spec))
    object_names = {item.name for item in problem_spec.objects}
    action_map = {schema.action.name: schema for schema in parsed_domain.actions}
    state = set(problem_spec.init_facts)
    grounded_steps: list[GroundedTrajectoryStep] = []
    reports: list[ValidationStepReport] = []
    issues: list[ValidationIssue] = []

    for record in manipulation_records:
        grounded_step = GroundedTrajectoryStep(
            episode_name=record.episode_name,
            step_index=record.step_index,
            raw_action_text=record.raw_action_text,
            action_category="manipulation",
            canonical_action_name=record.canonical_action_name,
            ground_arguments=list(record.action_arguments),
            ground_action_pddl=_render_ground_action_pddl(record.canonical_action_name, record.action_arguments),
            effect_bucket=record.effect_bucket,
            delta_add=list(record.delta_add),
            delta_del=list(record.delta_del),
            success=record.success,
            observation_text=record.post_observation_text,
            extra_info=record.extra_info,
        )
        grounded_steps.append(grounded_step)
        report, issue, _unused_next_state = execute_grounded_step(
            parsed_domain=parsed_domain,
            object_names=object_names,
            state=state,
            step=grounded_step,
            action_map=action_map,
            check_preconditions=True,
        )
        continued_state = _apply_effects_to_state(state, record.delta_add, record.delta_del)
        reports.append(
            ValidationStepReport(
                step_index=record.step_index,
                action_name=record.canonical_action_name,
                effect_bucket=record.effect_bucket,
                status=report.status,
                state_before=list(report.state_before),
                state_after=sorted(continued_state),
                failed_preconditions=list(report.failed_preconditions),
            )
        )
        if issue is not None:
            issues.append(issue)
        state = continued_state

    goal_satisfied = all(goal_fact in state for goal_fact in problem_spec.goal_facts)
    return grounded_steps, reports, issues, goal_satisfied, state


def _validate_grounded_fact_lists(
    fact_lists: list[list[str]],
    *,
    allowed_predicates: set[str],
    predicate_arities: dict[str, int],
    object_names: set[str],
    context: str,
) -> None:
    for fact_list in fact_lists:
        _validate_effect_literals_against_inventory(
            delta_add=fact_list,
            delta_del=[],
            allowed_predicates=allowed_predicates,
            predicate_arities=predicate_arities,
            context=context,
        )
        for fact in fact_list:
            _predicate_name, arguments = parse_positive_symbolic_fact(fact)
            unknown = [item for item in arguments if item not in object_names]
            if unknown:
                raise ValueError(f"{context}: fact {fact!r} references unknown objects {unknown}.")


def _apply_problem_spec_repair(problem_spec: ProblemSpec, repair_plan: EpisodeEffectRepairPlan) -> ProblemSpec:
    init_facts = [fact for fact in problem_spec.init_facts if fact not in set(repair_plan.init_facts_remove)]
    for fact in repair_plan.init_facts_add:
        if fact not in init_facts:
            init_facts.append(fact)
    goal_facts = [fact for fact in problem_spec.goal_facts if fact not in set(repair_plan.goal_facts_remove)]
    for fact in repair_plan.goal_facts_add:
        if fact not in goal_facts:
            goal_facts.append(fact)
    return ProblemSpec(
        problem_name=problem_spec.problem_name,
        domain_name=problem_spec.domain_name,
        objects=list(problem_spec.objects),
        init_facts=init_facts,
        goal_facts=goal_facts,
        canonical_object_map=dict(problem_spec.canonical_object_map),
    )


def _apply_step_effect_repairs(
    records: list[ManipulationEffectRecord],
    repair_plan: EpisodeEffectRepairPlan,
) -> list[ManipulationEffectRecord]:
    patches_by_step = {item.step_index: item for item in repair_plan.step_effect_repairs}
    updated: list[ManipulationEffectRecord] = []
    for record in records:
        patch = patches_by_step.get(record.step_index)
        if patch is None:
            updated.append(record)
            continue
        delta_add = [fact for fact in record.delta_add if fact not in set(patch.delta_add_remove)]
        for fact in patch.delta_add_add:
            if fact not in delta_add:
                delta_add.append(fact)
        delta_del = [fact for fact in record.delta_del if fact not in set(patch.delta_del_remove)]
        for fact in patch.delta_del_add:
            if fact not in delta_del:
                delta_del.append(fact)
        updated.append(
            ManipulationEffectRecord(
                episode_name=record.episode_name,
                step_index=record.step_index,
                raw_action_text=record.raw_action_text,
                canonical_action_name=record.canonical_action_name,
                action_arguments=list(record.action_arguments),
                pre_observation_text=record.pre_observation_text,
                post_observation_text=record.post_observation_text,
                extra_info=record.extra_info,
                delta_add=delta_add,
                delta_del=delta_del,
                effect_bucket=patch.effect_bucket or record.effect_bucket,
                success=record.success if patch.success is None else patch.success,
            )
        )
    return updated


__all__ = [
    "EpisodeEffectRepairPlan",
    "EpisodeEffectRepairStepPatch",
    "EpisodeGroundedEffectIteration",
    "EpisodeGroundedEffectResult",
    "LLMEpisodeEffectRepairModule",
    "LLMEpisodeGroundedEffectLearningModule",
    "LLMEpisodeStepEffectModule",
]
