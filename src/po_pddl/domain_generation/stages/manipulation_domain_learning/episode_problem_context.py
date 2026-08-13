from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from po_pddl.config import DEFAULT_MODEL
from po_pddl.core.parser import parse_domain
from po_pddl.domain_generation.infrastructure.artifact_io import write_jsonl
from po_pddl.domain_generation.stages.problem_grounding.models import EpisodeContext, ProblemSpec
from po_pddl.domain_generation.stages.problem_grounding.modules import (
    GoalInferenceModule,
    LLMGoalInferenceModule,
    ObjectInitInferenceModule,
    ProblemAssemblyModule,
    RuleBasedProblemAssemblyModule,
    _filter_facts_to_known_objects,
    _typed_objects_for_episode_from_actions,
    load_domain_learning_artifacts,
    load_episode_context,
)
from po_pddl.domain_generation.stages.problem_inference.modules import (
    InitialStateInferenceModule,
    InitStateRepairModule,
    LLMInitialStateInferenceModule,
    LLMInitStateRepairModule,
    VisibleObjectCandidate,
    summarize_parsed_domain,
)

from .models import (
    ActionSchema,
    ActionTaxonomyRecord,
    ObjectTypeDefinition,
    PredicateSchema,
    RawTrajectoryStep,
)
from .renderer import render_action_schema_fragment


def _object_matches_type(
    parsed_domain,
    *,
    object_type_name: str,
    expected_type_name: str,
    special_type_memberships: dict[str, set[str]] | None = None,
) -> bool:
    if object_type_name == expected_type_name or expected_type_name == "object":
        return True
    memberships = special_type_memberships or {}
    if object_type_name in memberships.get(expected_type_name, set()):
        return True
    object_type = parsed_domain.types.get(object_type_name)
    if object_type is None:
        return False
    return object_type.is_subtype_of(expected_type_name)


def _special_type_memberships_from_artifacts(artifacts) -> dict[str, set[str]]:
    action_name_map = artifacts.action_name_map if isinstance(artifacts.action_name_map, dict) else {}
    typing = action_name_map.get("typing") if isinstance(action_name_map, dict) else {}
    if not isinstance(typing, dict):
        return {}
    raw_memberships = typing.get("type_to_special_supertypes")
    if isinstance(raw_memberships, dict):
        special_type_memberships: dict[str, set[str]] = {}
        for concrete_type_name, raw_special_types in raw_memberships.items():
            normalized_type_name = str(concrete_type_name).strip()
            if not normalized_type_name or not isinstance(raw_special_types, list):
                continue
            for raw_special_type in raw_special_types:
                normalized_special_type = str(raw_special_type).strip()
                if not normalized_special_type:
                    continue
                special_type_memberships.setdefault(normalized_special_type, set()).add(normalized_type_name)
        if special_type_memberships:
            return special_type_memberships
    parent_map = typing.get("type_to_parent_type")
    if not isinstance(parent_map, dict):
        return {}
    fallback_memberships: dict[str, set[str]] = {}
    for concrete_type_name, special_type_name in parent_map.items():
        normalized_type_name = str(concrete_type_name).strip()
        normalized_special_type = str(special_type_name).strip()
        if normalized_type_name and normalized_special_type:
            fallback_memberships.setdefault(normalized_special_type, set()).add(normalized_type_name)
    return fallback_memberships


def _build_grounded_predicates_for_episode(
    parsed_domain,
    *,
    selected_objects: list,
    special_type_memberships: dict[str, set[str]] | None = None,
    excluded_predicate_names: set[str] | None = None,
) -> list[str]:
    excluded_predicate_names = excluded_predicate_names or set()
    grounded_predicates: list[str] = []
    for predicate in parsed_domain.predicates:
        predicate_name = predicate.name
        if predicate_name in excluded_predicate_names:
            continue
        parameter_types = parsed_domain.predicate_parameter_types.get(predicate, [])
        if not parameter_types:
            grounded_predicates.append(f"{predicate_name}()")
            continue
        candidate_object_names_per_param: list[list[str]] = []
        for _param_name, expected_type_name in parameter_types:
            compatible_names = [
                item.name
                for item in selected_objects
                if _object_matches_type(
                    parsed_domain,
                    object_type_name=item.type_name,
                    expected_type_name=expected_type_name,
                    special_type_memberships=special_type_memberships,
                )
            ]
            if not compatible_names:
                candidate_object_names_per_param = []
                break
            candidate_object_names_per_param.append(compatible_names)
        if not candidate_object_names_per_param:
            continue
        for argument_tuple in product(*candidate_object_names_per_param):
            grounded_predicates.append(f"{predicate_name}({','.join(argument_tuple)})")
    return sorted(dict.fromkeys(grounded_predicates))


def _default_gripper_init_facts(
    parsed_domain,
    *,
    selected_objects: list,
    special_type_memberships: dict[str, set[str]] | None = None,
) -> tuple[list[str], list[str]]:
    default_true_facts: list[str] = []
    default_false_facts: list[str] = []
    gripper_empty_predicate = next((item for item in parsed_domain.predicates if item.name == "gripper_empty"), None)
    if gripper_empty_predicate is not None and not parsed_domain.predicate_parameter_types.get(
        gripper_empty_predicate, []
    ):
        default_true_facts.append("gripper_empty()")
    gripper_holding_predicate = next(
        (item for item in parsed_domain.predicates if item.name == "gripper_holding"), None
    )
    if gripper_holding_predicate is not None:
        parameter_types = parsed_domain.predicate_parameter_types.get(gripper_holding_predicate, [])
        expected_type_name = parameter_types[0][1] if parameter_types else "object"
        for item in selected_objects:
            if _object_matches_type(
                parsed_domain,
                object_type_name=item.type_name,
                expected_type_name=expected_type_name,
                special_type_memberships=special_type_memberships,
            ):
                default_false_facts.append(f"gripper_holding({item.name})")
    return default_true_facts, default_false_facts


@dataclass(frozen=True)
class EpisodeProblemContextResult:
    episode_name: str
    episode: EpisodeContext
    problem_spec_without_goal: ProblemSpec
    problem_spec: ProblemSpec
    object_init_raw_llm_outputs: dict[str, str] = field(default_factory=dict)
    goal_inference_raw_output: str | None = None


@dataclass
class RuleBasedActionArgumentInitialStateModule:
    initial_state_module: InitialStateInferenceModule
    init_state_repair_module: InitStateRepairModule | None = None

    def __post_init__(self) -> None:
        self.last_result = None

    def induce_problem_object_init(
        self,
        *,
        domain_file: str | Path,
        episode_file: str | Path,
        domain_learning_dir: str | Path,
        review_guidance: dict[str, object] | None = None,
    ) -> ProblemSpec:
        parsed_domain = parse_domain(Path(domain_file).read_text(encoding="utf-8"))
        domain_summary = summarize_parsed_domain(parsed_domain)
        episode = load_episode_context(episode_file)
        artifacts = load_domain_learning_artifacts(domain_learning_dir, episode_name=episode.episode_name)
        special_type_memberships = _special_type_memberships_from_artifacts(artifacts)
        selected_objects = _typed_objects_for_episode_from_actions(episode, artifacts)
        visible_objects = [
            VisibleObjectCandidate(
                name=item.name,
                type_name=item.type_name,
            )
            for item in selected_objects
        ]
        default_true_init_facts, default_false_init_facts = _default_gripper_init_facts(
            parsed_domain,
            selected_objects=selected_objects,
            special_type_memberships=special_type_memberships,
        )
        grounded_predicates = _build_grounded_predicates_for_episode(
            parsed_domain,
            selected_objects=selected_objects,
            special_type_memberships=special_type_memberships,
            excluded_predicate_names={"gripper_empty", "gripper_holding"},
        )
        try:
            inferred_init_facts = self.initial_state_module.infer_initial_state(
                episode=episode,
                domain_summary=domain_summary,
                visible_objects=visible_objects,
                visible_facts=[],
                latent_objects=[],
                grounded_predicates=grounded_predicates,
                review_guidance=review_guidance,
            )
        except TypeError as exc:
            if "grounded_predicates" not in str(exc):
                raise
            inferred_init_facts = self.initial_state_module.infer_initial_state(
                episode=episode,
                domain_summary=domain_summary,
                visible_objects=visible_objects,
                visible_facts=[],
                latent_objects=[],
                review_guidance=review_guidance,
            )
        selected_object_names = {item.name for item in selected_objects}
        inferred_true_init_facts = _filter_facts_to_known_objects(
            [item.fact for item in inferred_init_facts],
            selected_object_names,
        )
        blocked_default_false_facts = set(default_false_init_facts)
        init_facts = list(default_true_init_facts)
        for fact in inferred_true_init_facts:
            if fact in blocked_default_false_facts:
                continue
            if fact not in init_facts:
                init_facts.append(fact)
        repair_plan = None
        if self.init_state_repair_module is not None:
            repair_plan = self.init_state_repair_module.review_initial_state(
                episode=episode,
                domain_summary=domain_summary,
                selected_objects=visible_objects,
                current_true_init_facts=list(init_facts),
                grounded_predicates=grounded_predicates,
                review_guidance=review_guidance,
            )
            if repair_plan.should_repair:
                remove_set = set(repair_plan.init_facts_remove)
                init_facts = [fact for fact in init_facts if fact not in remove_set]
                for fact in repair_plan.init_facts_add:
                    if fact not in init_facts and fact not in blocked_default_false_facts:
                        init_facts.append(fact)
        self.last_result = SimpleNamespace(
            raw_llm_outputs={
                key: value
                for key, value in {
                    "initial_state_inference": getattr(self.initial_state_module, "last_raw_output", None),
                    "initial_state_location_repair": (
                        getattr(repair_plan, "raw_llm_output", None) if repair_plan is not None else None
                    ),
                }.items()
                if isinstance(value, str) and value.strip()
            },
            visible_objects=visible_objects,
            latent_objects=[],
            inferred_init_facts=inferred_init_facts,
            grounded_predicates=grounded_predicates,
            default_true_init_facts=default_true_init_facts,
            default_false_init_facts=default_false_init_facts,
            init_state_repair_plan=repair_plan,
        )
        return ProblemSpec(
            problem_name=f"{episode.episode_name}_inferred_problem",
            domain_name=parsed_domain.domain_name,
            objects=selected_objects,
            init_facts=init_facts,
            goal_facts=[],
            canonical_object_map={},
        )


def build_default_problem_context_object_init_module(
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 3000,
    verbose: bool = False,
) -> ObjectInitInferenceModule:
    return RuleBasedActionArgumentInitialStateModule(
        initial_state_module=LLMInitialStateInferenceModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        ),
        init_state_repair_module=LLMInitStateRepairModule(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        ),
    )


@dataclass
class EpisodeProblemContextModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 3000
    max_workers: int = 1
    verbose: bool = False
    object_init_module: ObjectInitInferenceModule | None = None
    goal_inference_module: GoalInferenceModule | None = None
    assembly_module: ProblemAssemblyModule | None = None

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {self.max_workers}")
        self._uses_default_object_init_module = self.object_init_module is None
        if self.object_init_module is None:
            self.object_init_module = build_default_problem_context_object_init_module(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                verbose=self.verbose,
            )
        self._uses_default_goal_inference_module = self.goal_inference_module is None
        if self.goal_inference_module is None:
            self.goal_inference_module = LLMGoalInferenceModule(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                verbose=self.verbose,
            )
        self._uses_default_assembly_module = self.assembly_module is None
        if self.assembly_module is None:
            self.assembly_module = RuleBasedProblemAssemblyModule()

    def build_episode_problem_contexts(
        self,
        *,
        steps: list[RawTrajectoryStep],
        action_schemas: list[ActionSchema],
        taxonomy_records: list[ActionTaxonomyRecord],
        predicate_inventory: list[PredicateSchema] | None = None,
        object_types: list[ObjectTypeDefinition] | None = None,
        review_guidance_by_episode: dict[str, dict[str, object]] | None = None,
    ) -> dict[str, EpisodeProblemContextResult]:
        if not steps:
            return {}
        review_guidance_by_episode = review_guidance_by_episode or {}
        grouped_steps = _group_steps_by_episode(steps)
        with TemporaryDirectory(prefix="effect_problem_context_") as tmp_dir:
            root = Path(tmp_dir)
            domain_file = root / "action_schemas.pddl"
            domain_file.write_text(
                render_action_schema_fragment(
                    action_schemas,
                    predicate_inventory=predicate_inventory,
                    object_types=object_types,
                ),
                encoding="utf-8",
            )
            artifact_dir = root / "domain_learning"
            self._write_domain_learning_artifacts(
                artifact_dir=artifact_dir,
                action_schemas=action_schemas,
                taxonomy_records=taxonomy_records,
                object_types=list(object_types or []),
            )

            episode_jobs: list[tuple[str, Path, dict[str, object] | None]] = []
            for episode_name, episode_steps in sorted(grouped_steps.items()):
                episode_file = root / "episodes" / episode_name / "episode.json"
                episode_file.parent.mkdir(parents=True, exist_ok=True)
                episode_file.write_text(
                    json.dumps(_episode_payload_from_steps(episode_name, episode_steps), ensure_ascii=False, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
                episode_jobs.append((episode_name, episode_file, review_guidance_by_episode.get(episode_name)))

            results: dict[str, EpisodeProblemContextResult] = {}
            if self.max_workers == 1 or len(episode_jobs) <= 1:
                ordered_results = [
                    self._build_episode_problem_context(
                        domain_file=domain_file,
                        artifact_dir=artifact_dir,
                        episode_name=episode_name,
                        episode_file=episode_file,
                        predicate_inventory=predicate_inventory,
                        object_types=list(object_types or []),
                        review_guidance=review_guidance,
                        use_worker_modules=False,
                    )
                    for episode_name, episode_file, review_guidance in episode_jobs
                ]
            else:
                ordered_results: list[tuple[str, EpisodeProblemContextResult] | None] = [None] * len(episode_jobs)
                with ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="episode-problem-context",
                ) as executor:
                    future_to_index = {
                        executor.submit(
                            self._build_episode_problem_context,
                            domain_file=domain_file,
                            artifact_dir=artifact_dir,
                            episode_name=episode_name,
                            episode_file=episode_file,
                            predicate_inventory=predicate_inventory,
                            object_types=list(object_types or []),
                            review_guidance=review_guidance,
                            use_worker_modules=True,
                        ): index
                        for index, (episode_name, episode_file, review_guidance) in enumerate(episode_jobs)
                    }
                    for future in as_completed(future_to_index):
                        ordered_results[future_to_index[future]] = future.result()

            for item in ordered_results:
                if item is None:
                    continue
                episode_name, result = item
                results[episode_name] = result
            return results

    def _build_episode_problem_context(
        self,
        *,
        domain_file: Path,
        artifact_dir: Path,
        episode_name: str,
        episode_file: Path,
        predicate_inventory: list[PredicateSchema] | None,
        object_types: list[ObjectTypeDefinition],
        review_guidance: dict[str, object] | None,
        use_worker_modules: bool,
    ) -> tuple[str, EpisodeProblemContextResult]:
        if use_worker_modules:
            object_init_module = self._clone_for_parallel(
                self.object_init_module,
                factory=self._build_object_init_module_worker if self._uses_default_object_init_module else None,
                label="object_init_module",
            )
            goal_inference_module = self._clone_for_parallel(
                self.goal_inference_module,
                factory=self._build_goal_inference_module_worker if self._uses_default_goal_inference_module else None,
                label="goal_inference_module",
            )
            assembly_module = self._clone_for_parallel(
                self.assembly_module,
                factory=self._build_assembly_module_worker if self._uses_default_assembly_module else None,
                label="assembly_module",
            )
        else:
            object_init_module = self.object_init_module
            goal_inference_module = self.goal_inference_module
            assembly_module = self.assembly_module

        episode = load_episode_context(episode_file)
        problem_spec_without_goal = object_init_module.induce_problem_object_init(
            domain_file=domain_file,
            episode_file=episode_file,
            domain_learning_dir=artifact_dir,
            review_guidance=review_guidance,
        )
        predicate_names = [item.predicate_name for item in predicate_inventory or []]
        goal_facts = (
            goal_inference_module.induce_goal_facts(
                episode=episode,
                domain_name=problem_spec_without_goal.domain_name,
                predicate_names=predicate_names,
                predicate_schemas=[item.to_dict() for item in predicate_inventory or []],
                type_memberships={
                    item.type_name: sorted(
                        {
                            *(item.special_supertypes or []),
                            *([item.parent_type] if item.parent_type else []),
                        }
                    )
                    for item in object_types
                },
                problem_spec_without_goal=problem_spec_without_goal,
                review_guidance=review_guidance,
            )
            if predicate_names
            else []
        )
        problem_spec = assembly_module.assemble_problem_spec(
            base_problem_spec=problem_spec_without_goal,
            goal_facts=goal_facts,
        )
        goal_inference_raw_output = getattr(goal_inference_module, "last_raw_output", None)
        object_init_raw_outputs = _extract_object_init_raw_outputs(object_init_module)
        return (
            episode_name,
            EpisodeProblemContextResult(
                episode_name=episode_name,
                episode=episode,
                problem_spec_without_goal=problem_spec_without_goal,
                problem_spec=problem_spec,
                object_init_raw_llm_outputs=object_init_raw_outputs,
                goal_inference_raw_output=goal_inference_raw_output
                if isinstance(goal_inference_raw_output, str)
                else None,
            ),
        )

    def _build_object_init_module_worker(self) -> ObjectInitInferenceModule:
        return build_default_problem_context_object_init_module(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )

    def _build_goal_inference_module_worker(self) -> GoalInferenceModule:
        return LLMGoalInferenceModule(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )

    @staticmethod
    def _build_assembly_module_worker() -> ProblemAssemblyModule:
        return RuleBasedProblemAssemblyModule()

    @staticmethod
    def _clone_for_parallel(module, *, factory, label: str):
        clone_for_parallel = getattr(module, "clone_for_parallel", None)
        if callable(clone_for_parallel):
            return clone_for_parallel()
        if factory is not None:
            return factory()
        try:
            return deepcopy(module)
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise ValueError(f"{label} must support parallel cloning via clone_for_parallel() or deepcopy()") from exc

    @staticmethod
    def _write_domain_learning_artifacts(
        *,
        artifact_dir: Path,
        action_schemas: list[ActionSchema],
        taxonomy_records: list[ActionTaxonomyRecord],
        object_types: list[ObjectTypeDefinition],
    ) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "action_schemas.json").write_text(
            json.dumps([schema.to_dict() for schema in action_schemas], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_jsonl(
            artifact_dir / "action_taxonomy.jsonl",
            [record.to_dict() for record in taxonomy_records],
        )
        write_jsonl(artifact_dir / "manipulation_records.jsonl", [])
        write_jsonl(
            artifact_dir / "episode_object_inventory.jsonl",
            _episode_object_inventory_rows(taxonomy_records),
        )
        object_name_to_type = {
            object_name: item.type_name for item in object_types for object_name in item.member_object_names
        }
        (artifact_dir / "action_name_map.json").write_text(
            json.dumps(
                {
                    "typing": {
                        "object_name_to_type": object_name_to_type,
                        "type_to_parent_type": {
                            item.type_name: item.parent_type for item in object_types if item.parent_type
                        },
                        "type_to_special_supertypes": {
                            item.type_name: list(item.special_supertypes)
                            for item in object_types
                            if item.special_supertypes
                        },
                    }
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _group_steps_by_episode(steps: list[RawTrajectoryStep]) -> dict[str, list[RawTrajectoryStep]]:
    grouped: dict[str, list[RawTrajectoryStep]] = defaultdict(list)
    for step in steps:
        grouped[step.episode_name].append(step)
    for episode_name in list(grouped):
        grouped[episode_name] = sorted(grouped[episode_name], key=lambda item: item.step_index)
    return grouped


def _episode_payload_from_steps(episode_name: str, steps: list[RawTrajectoryStep]) -> dict[str, object]:
    instruction = next((str(step.instruction).strip() for step in steps if str(step.instruction).strip()), "")
    return {
        "episode_name": episode_name,
        "instruction": instruction,
        "steps": [
            {
                "step_index": step.step_index,
                "start_time_sec": step.start_time_sec,
                "end_time_sec": step.end_time_sec,
                "action_text": step.action_text,
                "observation_text": step.observation_text,
                "extra_info": step.extra_info,
            }
            for step in sorted(steps, key=lambda item: item.step_index)
        ],
    }


def _episode_object_inventory_rows(
    taxonomy_records: list[ActionTaxonomyRecord],
) -> list[dict[str, object]]:
    ordered_names_by_episode: dict[str, list[str]] = defaultdict(list)
    seen_by_episode: dict[str, set[str]] = defaultdict(set)
    for record in taxonomy_records:
        for object_name in record.action_arguments:
            if object_name in seen_by_episode[record.episode_name]:
                continue
            seen_by_episode[record.episode_name].add(object_name)
            ordered_names_by_episode[record.episode_name].append(object_name)
    return [
        {
            "episode_name": episode_name,
            "object_names": object_names,
        }
        for episode_name, object_names in sorted(ordered_names_by_episode.items())
    ]


def _extract_object_init_raw_outputs(object_init_module: ObjectInitInferenceModule) -> dict[str, str]:
    last_result = getattr(object_init_module, "last_result", None)
    if last_result is None and hasattr(object_init_module, "base"):
        last_result = getattr(getattr(object_init_module, "base"), "last_result", None)
    raw_outputs = getattr(last_result, "raw_llm_outputs", None)
    if isinstance(raw_outputs, dict):
        return {str(key): str(value) for key, value in raw_outputs.items() if str(key).strip() and str(value).strip()}
    return {}


__all__ = [
    "EpisodeProblemContextModule",
    "EpisodeProblemContextResult",
    "build_default_problem_context_object_init_module",
]
