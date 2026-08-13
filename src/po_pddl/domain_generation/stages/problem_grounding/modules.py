from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from po_pddl.config import DEFAULT_MODEL
from po_pddl.core.parser.schemas import ParsedActionSchema, ParsedDomain
from po_pddl.domain_generation.infrastructure.artifact_io import (
    load_domain_learning_artifact_rows,
    load_episode_payload,
    load_json,
    load_json_object,
    load_jsonl,
)
from po_pddl.domain_generation.infrastructure.fact_utils import (
    canonicalize_symbolic_literal_arguments as _canonicalize_fact,
)
from po_pddl.domain_generation.infrastructure.fact_utils import (
    remap_symbolic_literal_arguments as _remap_fact_arguments,
)
from po_pddl.domain_generation.infrastructure.fact_utils import (
    symbolic_literal_arguments as _extract_fact_arguments,
)
from po_pddl.domain_generation.infrastructure.fact_utils import (
    symbolic_literal_predicate as _extract_fact_predicate,
)
from po_pddl.domain_generation.infrastructure.fact_utils import (
    try_parse_symbolic_literal,
)
from po_pddl.domain_generation.infrastructure.payload_utils import (
    normalize_optional_text as _normalize_optional_text,
)
from po_pddl.domain_generation.infrastructure.payload_utils import (
    validate_snake_case as _validate_snake_case,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.structured_action_templates import (
    InducedActionTemplateArtifact,
    induced_template_from_dict,
    match_action_text_to_template,
)

from .models import (
    ActionSchemaArtifact,
    ActionTaxonomyArtifact,
    DomainLearningArtifacts,
    EpisodeContext,
    EpisodeStep,
    GroundedTrajectoryStep,
    ManipulationRecordArtifact,
    ObjectDeclaration,
    ProblemSpec,
)
from .shared import extract_json_object, load_prompt, make_client, safe_chat

if TYPE_CHECKING:
    from po_pddl.domain_generation.stages.problem_inference.inferer import ProblemInferenceLearner


logger = logging.getLogger(__name__)
_FAILURE_MARKERS = ("失败", "failed", "failure", "did not succeed", "unsuccessful")


class ProblemSpecInductionModule(Protocol):
    def induce_problem_spec(
        self,
        episode: EpisodeContext,
        domain_name: str,
        predicate_names: list[str],
        artifacts: DomainLearningArtifacts,
    ) -> ProblemSpec: ...


class ObjectInitInferenceModule(Protocol):
    def induce_problem_object_init(
        self,
        *,
        domain_file: str | Path,
        episode_file: str | Path,
        domain_learning_dir: str | Path,
        review_guidance: dict[str, object] | None = None,
    ) -> ProblemSpec: ...


class GoalInferenceModule(Protocol):
    def induce_goal_facts(
        self,
        *,
        episode: EpisodeContext,
        domain_name: str,
        predicate_names: list[str],
        predicate_schemas: list[dict[str, object]] | None = None,
        type_memberships: dict[str, list[str]] | None = None,
        problem_spec_without_goal: ProblemSpec,
        review_guidance: dict[str, object] | None = None,
    ) -> list[str]: ...


class ProblemAssemblyModule(Protocol):
    def assemble_problem_spec(
        self,
        *,
        base_problem_spec: ProblemSpec,
        goal_facts: list[str],
    ) -> ProblemSpec: ...


class TrajectoryGroundingModule(Protocol):
    def ground_step(
        self,
        *,
        episode: EpisodeContext,
        step: EpisodeStep,
        state_before: list[str],
        executed_actions: list[GroundedTrajectoryStep],
        problem_spec: ProblemSpec,
        artifacts: DomainLearningArtifacts,
        parsed_domain: ParsedDomain,
        review_guidance: dict[str, object] | None = None,
    ) -> GroundedTrajectoryStep: ...


@dataclass
class ProblemInferenceObjectInitModule:
    learner: "ProblemInferenceLearner"

    def __post_init__(self) -> None:
        self.last_result = None

    @classmethod
    def from_args(cls, args: object) -> "ProblemInferenceObjectInitModule":
        from po_pddl.domain_generation.stages.problem_inference.factory import (
            build_learner_from_args as build_problem_inference_learner_from_args,
        )

        learner, _ = build_problem_inference_learner_from_args(args)
        return cls(learner=learner)

    def induce_problem_object_init(
        self,
        *,
        domain_file: str | Path,
        episode_file: str | Path,
        domain_learning_dir: str | Path,
        review_guidance: dict[str, object] | None = None,
    ) -> ProblemSpec:
        self.last_result = self.learner.infer_from_files(
            domain_file=domain_file,
            episode_file=episode_file,
            domain_learning_dir=domain_learning_dir,
            review_guidance=review_guidance,
        )
        return self.last_result.problem_spec


@dataclass
class MappedObjectConstrainedInitModule:
    base: ObjectInitInferenceModule

    def induce_problem_object_init(
        self,
        *,
        domain_file: str | Path,
        episode_file: str | Path,
        domain_learning_dir: str | Path,
        review_guidance: dict[str, object] | None = None,
    ) -> ProblemSpec:
        base_spec = self.base.induce_problem_object_init(
            domain_file=domain_file,
            episode_file=episode_file,
            domain_learning_dir=domain_learning_dir,
            review_guidance=review_guidance,
        )
        episode = load_episode_context(episode_file)
        artifacts = load_domain_learning_artifacts(domain_learning_dir, episode_name=episode.episode_name)
        mapped_mentions = _mapped_object_names_for_episode(episode, artifacts)
        selected_objects = _select_problem_objects_from_map(
            base_objects=base_spec.objects,
            mapped_mentions=mapped_mentions,
        )
        selected_object_names = {item.name for item in selected_objects}
        filtered_init_facts = _filter_facts_to_known_objects(base_spec.init_facts, selected_object_names)
        filtered_canonical_object_map = {
            key: value
            for key, value in base_spec.canonical_object_map.items()
            if key in selected_object_names and value in selected_object_names
        }
        return ProblemSpec(
            problem_name=base_spec.problem_name,
            domain_name=base_spec.domain_name,
            objects=selected_objects,
            init_facts=filtered_init_facts,
            goal_facts=[],
            canonical_object_map=filtered_canonical_object_map,
        )


@dataclass
class RuleBasedActionArgumentObjectsInitModule:
    base: ObjectInitInferenceModule

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
        base_spec = self.base.induce_problem_object_init(
            domain_file=domain_file,
            episode_file=episode_file,
            domain_learning_dir=domain_learning_dir,
            review_guidance=review_guidance,
        )
        self.last_result = getattr(self.base, "last_result", None)
        episode = load_episode_context(episode_file)
        artifacts = load_domain_learning_artifacts(domain_learning_dir, episode_name=episode.episode_name)
        selected_objects = _typed_objects_for_episode_from_actions(episode, artifacts)
        if not selected_objects:
            selected_objects = list(base_spec.objects)
        selected_object_names = {item.name for item in selected_objects}
        filtered_init_facts = _filter_facts_to_known_objects(base_spec.init_facts, selected_object_names)
        filtered_canonical_object_map = {
            key: value
            for key, value in base_spec.canonical_object_map.items()
            if key in selected_object_names and value in selected_object_names
        }
        return ProblemSpec(
            problem_name=base_spec.problem_name,
            domain_name=base_spec.domain_name,
            objects=selected_objects,
            init_facts=filtered_init_facts,
            goal_facts=[],
            canonical_object_map=filtered_canonical_object_map,
        )


def _coerce_fact_list(value: object, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        items = [str(value).strip()] if str(value).strip() else []
    for item in items:
        if try_parse_symbolic_literal(item) is None:
            raise ValueError(f"Expected symbolic fact literal for {field_name}, got {item!r}")
    return items


def _canonicalize_name(name: str, canonical_object_map: dict[str, str]) -> str:
    seen: set[str] = set()
    current = name
    while current in canonical_object_map and current not in seen:
        seen.add(current)
        nxt = canonical_object_map[current]
        if nxt == current:
            break
        current = nxt
    return current


_DIRECTION_MARKERS: tuple[str, ...] = ("left", "right", "front", "back", "near", "far")


def _load_induced_action_templates(artifacts: DomainLearningArtifacts) -> list[InducedActionTemplateArtifact]:
    templates: list[InducedActionTemplateArtifact] = []
    for row in artifacts.action_templates:
        if isinstance(row, dict):
            templates.append(induced_template_from_dict(row))
    return templates


def _action_map_object_type_map(artifacts: DomainLearningArtifacts) -> dict[str, str]:
    action_name_map = artifacts.action_name_map if isinstance(artifacts.action_name_map, dict) else {}
    typing = action_name_map.get("typing") if isinstance(action_name_map, dict) else {}
    if not isinstance(typing, dict):
        return {}
    return {
        str(object_name): str(type_name)
        for object_name, type_name in (typing.get("object_name_to_type") or {}).items()
        if str(object_name).strip() and str(type_name).strip()
    }


def _extract_direction_markers(text: str | None) -> set[str]:
    lowered = (text or "").strip().lower()
    markers: set[str] = set()
    for marker in _DIRECTION_MARKERS:
        if marker in lowered:
            markers.add(marker)
    return markers


def _fact_contains_direction_for_object(fact: str, *, object_name: str, markers: set[str]) -> bool:
    predicate = _extract_fact_predicate(fact)
    if predicate is None or not any(marker in predicate for marker in markers):
        return False
    return object_name in _extract_fact_arguments(fact)


def _select_declared_object_for_argument(
    *,
    raw_argument: str,
    action_name: str,
    state_before: list[str],
    problem_spec: ProblemSpec,
    argument_index: int,
    total_arguments: int,
) -> str:
    del action_name, state_before, argument_index, total_arguments
    declared_names = [item.name for item in problem_spec.objects]
    canonical_name = _canonicalize_name(raw_argument, problem_spec.canonical_object_map)
    if canonical_name in declared_names:
        return canonical_name
    if raw_argument in declared_names:
        return raw_argument
    return raw_argument


def _mapped_action_for_step(
    *,
    step: EpisodeStep,
    artifacts: DomainLearningArtifacts,
    state_before: list[str],
    problem_spec: ProblemSpec,
) -> tuple[str | None, str | None, list[str]]:
    taxonomy = _taxonomy_for_step(artifacts, step.step_index)
    if taxonomy is not None:
        resolved_arguments = [
            _select_declared_object_for_argument(
                raw_argument=raw_argument,
                action_name=taxonomy.canonical_action_name,
                state_before=state_before,
                problem_spec=problem_spec,
                argument_index=index,
                total_arguments=len(taxonomy.action_arguments),
            )
            for index, raw_argument in enumerate(taxonomy.action_arguments)
        ]
        return taxonomy.canonical_action_name, taxonomy.action_category, resolved_arguments

    templates = _load_induced_action_templates(artifacts)
    object_type_map = _action_map_object_type_map(artifacts)
    if templates and step.action_text:
        try:
            parsed = match_action_text_to_template(
                step.action_text,
                templates,
                object_type_map=object_type_map,
            )
            resolved_arguments = [
                _select_declared_object_for_argument(
                    raw_argument=raw_argument,
                    action_name=parsed.canonical_action_name,
                    state_before=state_before,
                    problem_spec=problem_spec,
                    argument_index=index,
                    total_arguments=len(parsed.action_arguments),
                )
                for index, raw_argument in enumerate(parsed.action_arguments)
            ]
            return parsed.canonical_action_name, parsed.action_category, resolved_arguments
        except Exception:
            pass
    return None, None, []


def _mapped_object_names_for_episode(
    episode: EpisodeContext,
    artifacts: DomainLearningArtifacts,
) -> list[str]:
    if artifacts.episode_object_names:
        return list(artifacts.episode_object_names)
    templates = _load_induced_action_templates(artifacts)
    object_type_map = _action_map_object_type_map(artifacts)
    ordered: list[str] = []
    seen: set[str] = set()

    def add_name(name: str) -> None:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)

    for step in sorted((item for item in episode.steps if item.step_index > 0), key=lambda item: item.step_index):
        parsed_arguments: list[str] = []
        if templates and step.action_text:
            try:
                parsed_arguments = match_action_text_to_template(
                    step.action_text,
                    templates,
                    object_type_map=object_type_map,
                ).action_arguments
            except Exception:
                parsed_arguments = []
        if not parsed_arguments:
            taxonomy = _taxonomy_for_step(artifacts, step.step_index)
            parsed_arguments = list(taxonomy.action_arguments) if taxonomy is not None else []
        for argument in parsed_arguments:
            add_name(argument)
    return ordered


def _parameter_roles_for_action(
    artifacts: DomainLearningArtifacts,
    canonical_action_name: str,
) -> list[str]:
    schema = _schema_for_action(artifacts, canonical_action_name)
    if schema is not None and schema.parameter_roles:
        return list(schema.parameter_roles)
    action_name_map = artifacts.action_name_map if isinstance(artifacts.action_name_map, dict) else {}
    actions_by_name = action_name_map.get("actions_by_name") if isinstance(action_name_map, dict) else {}
    if isinstance(actions_by_name, dict):
        action_payload = actions_by_name.get(canonical_action_name)
        if isinstance(action_payload, dict):
            return [str(role).strip() for role in action_payload.get("parameter_roles", []) if str(role).strip()]
    return []


def _merge_object_type_name(*, name: str, existing: str | None, candidate: str | None) -> str:
    normalized_existing = str(existing or "").strip() or None
    normalized_candidate = str(candidate or "").strip() or "object"
    if normalized_existing in {None, "", "object"}:
        return normalized_candidate
    if normalized_candidate in {"", "object", normalized_existing}:
        return normalized_existing
    raise ValueError(
        f"Conflicting object types inferred for action argument {name!r}: "
        f"{normalized_existing!r} vs {normalized_candidate!r}"
    )


def _typed_objects_for_episode_from_actions(
    episode: EpisodeContext,
    artifacts: DomainLearningArtifacts,
) -> list[ObjectDeclaration]:
    templates = _load_induced_action_templates(artifacts)
    object_type_map = _action_map_object_type_map(artifacts)
    ordered_names: list[str] = []
    type_by_name: dict[str, str] = {}

    def add_object(name: str, type_name: str) -> None:
        normalized_name = str(name).strip()
        if not normalized_name:
            return
        if normalized_name not in type_by_name:
            ordered_names.append(normalized_name)
        type_by_name[normalized_name] = _merge_object_type_name(
            name=normalized_name,
            existing=type_by_name.get(normalized_name),
            candidate=type_name,
        )

    for step in sorted((item for item in episode.steps if item.step_index > 0), key=lambda item: item.step_index):
        canonical_action_name: str | None = None
        action_arguments: list[str] = []
        taxonomy = _taxonomy_for_step(artifacts, step.step_index)
        if taxonomy is not None:
            canonical_action_name = taxonomy.canonical_action_name
            action_arguments = list(taxonomy.action_arguments)
        elif templates and step.action_text:
            try:
                parsed = match_action_text_to_template(
                    step.action_text,
                    templates,
                    object_type_map=object_type_map,
                )
                canonical_action_name = parsed.canonical_action_name
                action_arguments = list(parsed.action_arguments)
            except Exception:
                canonical_action_name = None
                action_arguments = []
        if not canonical_action_name or not action_arguments:
            continue
        parameter_roles = _parameter_roles_for_action(artifacts, canonical_action_name)
        for index, argument in enumerate(action_arguments):
            type_name = (
                parameter_roles[index] if index < len(parameter_roles) and parameter_roles[index].strip() else "object"
            )
            add_object(argument, type_name)

    return [ObjectDeclaration(name=name, type_name=type_by_name.get(name, "object")) for name in ordered_names]


def _filter_facts_to_known_objects(facts: list[str], allowed_objects: set[str]) -> list[str]:
    filtered: list[str] = []
    for fact in facts:
        predicate = _extract_fact_predicate(fact)
        arguments = _extract_fact_arguments(fact)
        if predicate is None:
            continue
        remapped_arguments: list[str] = []
        keep_fact = True
        for argument in arguments:
            mapped_argument = argument
            if mapped_argument not in allowed_objects:
                keep_fact = False
                break
            remapped_arguments.append(mapped_argument)
        if not keep_fact:
            continue
        if remapped_arguments:
            filtered.append(f"{predicate}({','.join(remapped_arguments)})")
        else:
            filtered.append(f"{predicate}()")
    return filtered


def _select_problem_objects_from_map(
    *,
    base_objects: list[ObjectDeclaration],
    mapped_mentions: list[str],
) -> list[ObjectDeclaration]:
    if not mapped_mentions:
        return list(base_objects)
    ordered: list[ObjectDeclaration] = []
    seen: set[str] = set()
    base_by_name = {item.name: item for item in base_objects}
    for mention in mapped_mentions:
        matches = [
            item
            for item in base_objects
            if item.name == mention or item.name.startswith(f"{mention}_") or mention.startswith(f"{item.name}_")
        ]
        if mention in base_by_name:
            matches = [base_by_name[mention]]
        for item in matches:
            if item.name in seen:
                continue
            seen.add(item.name)
            ordered.append(item)
    return ordered


def _instantiate_effect_from_exemplar(
    exemplar: ManipulationRecordArtifact,
    ground_arguments: list[str],
) -> tuple[list[str], list[str]]:
    argument_mapping = {
        source_argument: target_argument
        for source_argument, target_argument in zip(exemplar.action_arguments, ground_arguments)
    }
    delta_add = [_remap_fact_arguments(fact, argument_mapping) for fact in exemplar.delta_add]
    delta_del = [_remap_fact_arguments(fact, argument_mapping) for fact in exemplar.delta_del]
    return delta_add, delta_del


def _ground_fact(predicate: str, arguments: list[str], binding: dict[str, str]) -> str:
    grounded_arguments = [binding.get(argument, argument) for argument in arguments]
    if grounded_arguments:
        return f"{predicate}({','.join(grounded_arguments)})"
    return f"{predicate}()"


def _merge_branch_lists(
    left: list[tuple[list[str], list[str]]],
    right: list[tuple[list[str], list[str]]],
) -> list[tuple[list[str], list[str]]]:
    merged: list[tuple[list[str], list[str]]] = []
    for left_add, left_del in left:
        for right_add, right_del in right:
            merged.append((left_add + right_add, left_del + right_del))
    return merged


def _effect_expr_to_branches(
    expr: object,
    binding: dict[str, str],
) -> list[tuple[list[str], list[str]]]:
    if expr is None:
        return [([], [])]
    if isinstance(expr, str):
        return [([_ground_fact(expr, [], binding)], [])]
    if not isinstance(expr, list) or not expr:
        return [([], [])]

    head = expr[0]
    if not isinstance(head, str):
        return [([], [])]

    if head == "and":
        branches: list[tuple[list[str], list[str]]] = [([], [])]
        for item in expr[1:]:
            item_branches = _effect_expr_to_branches(item, binding)
            branches = _merge_branch_lists(branches, item_branches)
        return branches

    if head == "not":
        if len(expr) == 2 and isinstance(expr[1], list) and expr[1]:
            inner_head = expr[1][0]
            if isinstance(inner_head, str):
                return [([], [_ground_fact(inner_head, [str(arg) for arg in expr[1][1:]], binding)])]
        return [([], [])]

    if head == "probabilistic":
        branches: list[tuple[list[str], list[str]]] = []
        index = 1
        while index + 1 < len(expr):
            branch_expr = expr[index + 1]
            branches.extend(_effect_expr_to_branches(branch_expr, binding))
            index += 2
        return branches or [([], [])]

    if head in {"when", "forall", "increase", "decrease", "assign"}:
        return [([], [])]

    return [([_ground_fact(head, [str(arg) for arg in expr[1:]], binding)], [])]


def _choose_domain_effect_branch(
    action_schema: ParsedActionSchema,
    ground_arguments: list[str],
    exemplar_add: list[str],
    exemplar_del: list[str],
) -> tuple[list[str], list[str]] | None:
    binding = {parameter: argument for parameter, argument in zip(action_schema.action.params, ground_arguments)}
    branches = _effect_expr_to_branches(action_schema.effect, binding)
    if not branches:
        return None
    if len(branches) == 1:
        return branches[0]

    exemplar_add_set = set(exemplar_add)
    exemplar_del_set = set(exemplar_del)
    best_branch: tuple[list[str], list[str]] | None = None
    best_score: tuple[int, int, int] | None = None
    for add_facts, del_facts in branches:
        add_set = set(add_facts)
        del_set = set(del_facts)
        overlap = len(add_set & exemplar_add_set) + len(del_set & exemplar_del_set)
        symmetric_penalty = len((add_set ^ exemplar_add_set)) + len((del_set ^ exemplar_del_set))
        branch_size = len(add_set) + len(del_set)
        score = (overlap, -symmetric_penalty, branch_size)
        if best_score is None or score > best_score:
            best_score = score
            best_branch = (add_facts, del_facts)
    return best_branch


def canonicalize_domain_learning_artifacts(
    artifacts: DomainLearningArtifacts,
    canonical_object_map: dict[str, str],
) -> DomainLearningArtifacts:
    normalized_taxonomy = [
        ActionTaxonomyArtifact(
            episode_name=record.episode_name,
            step_index=record.step_index,
            canonical_action_name=record.canonical_action_name,
            action_category=record.action_category,
            action_arguments=[
                _canonicalize_name(argument, canonical_object_map) for argument in record.action_arguments
            ],
            raw_action_text=record.raw_action_text,
            observation_text=record.observation_text,
            extra_info=record.extra_info,
        )
        for record in artifacts.taxonomy_records
    ]
    normalized_manipulation = [
        ManipulationRecordArtifact(
            episode_name=record.episode_name,
            step_index=record.step_index,
            canonical_action_name=record.canonical_action_name,
            action_arguments=[
                _canonicalize_name(argument, canonical_object_map) for argument in record.action_arguments
            ],
            effect_bucket=record.effect_bucket,
            delta_add=[_canonicalize_fact(fact, canonical_object_map) for fact in record.delta_add],
            delta_del=[_canonicalize_fact(fact, canonical_object_map) for fact in record.delta_del],
            success=record.success,
            raw_action_text=record.raw_action_text,
        )
        for record in artifacts.manipulation_records
    ]
    return DomainLearningArtifacts(
        action_schemas=artifacts.action_schemas,
        taxonomy_records=normalized_taxonomy,
        manipulation_records=normalized_manipulation,
        action_templates=list(artifacts.action_templates),
        action_name_map=dict(artifacts.action_name_map),
    )


def _taxonomy_for_step(artifacts: DomainLearningArtifacts, step_index: int) -> ActionTaxonomyArtifact | None:
    return next((record for record in artifacts.taxonomy_records if record.step_index == step_index), None)


def _manipulation_for_step(artifacts: DomainLearningArtifacts, step_index: int) -> ManipulationRecordArtifact | None:
    return next((record for record in artifacts.manipulation_records if record.step_index == step_index), None)


def _schema_for_action(
    artifacts: DomainLearningArtifacts,
    canonical_action_name: str,
) -> ActionSchemaArtifact | None:
    return next(
        (schema for schema in artifacts.action_schemas if schema.canonical_action_name == canonical_action_name),
        None,
    )


def _candidate_manipulation_schemas_for_taxonomy(
    artifacts: DomainLearningArtifacts,
    taxonomy: ActionTaxonomyArtifact,
) -> list[ActionSchemaArtifact]:
    candidates = [
        schema
        for schema in artifacts.action_schemas
        if schema.action_category == "manipulation" and schema.parameter_count == len(taxonomy.action_arguments)
    ]
    deduped: dict[str, ActionSchemaArtifact] = {}
    for schema in candidates:
        deduped[schema.canonical_action_name] = schema
    ordered = sorted(deduped.values(), key=lambda item: item.canonical_action_name)
    preferred = [item for item in ordered if item.canonical_action_name == taxonomy.canonical_action_name]
    others = [item for item in ordered if item.canonical_action_name != taxonomy.canonical_action_name]
    return preferred + others


def _infer_expected_branch(
    step: EpisodeStep,
    requested_effect_bucket: str | None,
) -> str | None:
    extra_info = (step.extra_info or "").strip().lower()
    if any(marker in extra_info for marker in _FAILURE_MARKERS):
        return "failure"
    bucket = (requested_effect_bucket or "").strip()
    if bucket.endswith("_failure"):
        return "failure"
    if bucket.endswith("_success"):
        return "success"
    if step.action_text:
        return "success"
    return None


def _select_effect_for_expected_branch(
    *,
    action_name: str,
    expected_branch: str | None,
    selected_effect: ManipulationRecordArtifact | None,
    manipulation: ManipulationRecordArtifact | None,
    effect_candidates: list[ManipulationRecordArtifact],
) -> ManipulationRecordArtifact | None:
    if expected_branch not in {"success", "failure"}:
        return selected_effect
    expected_success = expected_branch == "success"
    if (
        manipulation is not None
        and manipulation.canonical_action_name == action_name
        and manipulation.success == expected_success
    ):
        return manipulation
    if selected_effect is not None and selected_effect.success == expected_success:
        return selected_effect
    matching_candidates = [record for record in effect_candidates if record.success == expected_success]
    if not matching_candidates:
        return None
    preferred = [record for record in matching_candidates if record.effect_bucket.endswith(f"_{expected_branch}")]
    return (preferred or matching_candidates)[0]


def _placeholder_effect_bucket_name(action_name: str, expected_branch: str | None) -> str | None:
    if expected_branch not in {"success", "failure"}:
        return None
    return f"{action_name}_{expected_branch}"


@dataclass
class RuleBasedTemplateTrajectoryGroundingModule:
    def ground_step(
        self,
        *,
        episode: EpisodeContext,
        step: EpisodeStep,
        state_before: list[str],
        executed_actions: list[GroundedTrajectoryStep],
        problem_spec: ProblemSpec,
        artifacts: DomainLearningArtifacts,
        parsed_domain: ParsedDomain,
        review_guidance: dict[str, object] | None = None,
    ) -> GroundedTrajectoryStep:
        del executed_actions, review_guidance
        manipulation = _manipulation_for_step(artifacts, step.step_index)
        action_name, action_category, ground_arguments = _mapped_action_for_step(
            step=step,
            artifacts=artifacts,
            state_before=state_before,
            problem_spec=problem_spec,
        )
        action_schema = _schema_for_action(artifacts, action_name) if action_name is not None else None
        effective_action_category = (
            action_schema.action_category
            if action_schema is not None and action_schema.action_category
            else action_category
        )

        if effective_action_category != "manipulation":
            ground_action_pddl = None
            if action_name is not None:
                ground_action_pddl = f"({action_name}{(' ' + ' '.join(ground_arguments)) if ground_arguments else ''})"
            return GroundedTrajectoryStep(
                episode_name=episode.episode_name,
                step_index=step.step_index,
                raw_action_text=step.action_text,
                action_category=effective_action_category,
                canonical_action_name=action_name,
                ground_arguments=ground_arguments,
                ground_action_pddl=ground_action_pddl,
                effect_bucket=None,
                delta_add=[],
                delta_del=[],
                success=None,
                observation_text=step.observation_text,
                extra_info=step.extra_info,
            )

        if action_name is None:
            return GroundedTrajectoryStep(
                episode_name=episode.episode_name,
                step_index=step.step_index,
                raw_action_text=step.action_text,
                action_category=action_category,
                canonical_action_name=None,
                ground_arguments=[],
                ground_action_pddl=None,
                effect_bucket=None,
                delta_add=[],
                delta_del=[],
                success=None,
                observation_text=step.observation_text,
                extra_info=step.extra_info,
            )

        expected_branch = (
            "failure" if any(marker in (step.extra_info or "").lower() for marker in _FAILURE_MARKERS) else "success"
        )
        expected_success = expected_branch == "success"
        effect_candidates = [
            record for record in artifacts.manipulation_records if record.canonical_action_name == action_name
        ]
        selected_effect = next(
            (record for record in effect_candidates if record.success == expected_success),
            None,
        )
        if selected_effect is None and manipulation is not None and manipulation.success == expected_success:
            selected_effect = manipulation

        available_effect_buckets = sorted({record.effect_bucket for record in effect_candidates})
        ground_action_pddl = f"({action_name}{(' ' + ' '.join(ground_arguments)) if ground_arguments else ''})"
        if selected_effect is None:
            placeholder_bucket = _placeholder_effect_bucket_name(action_name, expected_branch)
            return GroundedTrajectoryStep(
                episode_name=episode.episode_name,
                step_index=step.step_index,
                raw_action_text=step.action_text,
                action_category=effective_action_category,
                canonical_action_name=action_name,
                ground_arguments=ground_arguments,
                ground_action_pddl=ground_action_pddl,
                effect_bucket=placeholder_bucket,
                delta_add=[],
                delta_del=[],
                success=expected_success,
                observation_text=step.observation_text,
                extra_info=step.extra_info,
                requested_effect_bucket=placeholder_bucket,
                branch_expectation=expected_branch,
                missing_effect_branch=True,
                available_effect_buckets=available_effect_buckets,
            )

        exemplar_add, exemplar_del = _instantiate_effect_from_exemplar(selected_effect, ground_arguments)
        parsed_action_schema = next((item for item in parsed_domain.actions if item.action.name == action_name), None)
        domain_branch = (
            _choose_domain_effect_branch(parsed_action_schema, ground_arguments, exemplar_add, exemplar_del)
            if parsed_action_schema is not None
            else None
        )
        if domain_branch is not None:
            delta_add, delta_del = domain_branch
        else:
            delta_add, delta_del = exemplar_add, exemplar_del

        return GroundedTrajectoryStep(
            episode_name=episode.episode_name,
            step_index=step.step_index,
            raw_action_text=step.action_text,
            action_category=effective_action_category,
            canonical_action_name=action_name,
            ground_arguments=ground_arguments,
            ground_action_pddl=ground_action_pddl,
            effect_bucket=selected_effect.effect_bucket,
            delta_add=delta_add,
            delta_del=delta_del,
            success=selected_effect.success,
            observation_text=step.observation_text,
            extra_info=step.extra_info,
            requested_effect_bucket=selected_effect.effect_bucket,
            branch_expectation=expected_branch,
            missing_effect_branch=False,
            available_effect_buckets=available_effect_buckets,
        )


def load_episode_context(path: str | Path) -> EpisodeContext:
    episode_path = Path(path)
    payload = load_episode_payload(episode_path)
    steps = [
        EpisodeStep(
            step_index=int(step["step_index"]),
            start_time_sec=step.get("start_time_sec"),
            end_time_sec=step.get("end_time_sec"),
            action_text=_normalize_optional_text(step.get("action_text")),
            observation_text=_normalize_optional_text(step.get("observation_text")),
            extra_info=_normalize_optional_text(step.get("extra_info")),
        )
        for step in payload.get("steps", [])
    ]
    step0 = next((step for step in steps if step.step_index == 0), None)
    return EpisodeContext(
        episode_name=str(payload.get("episode_name", episode_path.parent.name)),
        instruction=str(payload.get("instruction", "")).strip(),
        step0_observation_text=step0.observation_text if step0 else None,
        steps=steps,
    )


def load_domain_learning_artifacts(path: str | Path, *, episode_name: str) -> DomainLearningArtifacts:
    rows = load_domain_learning_artifact_rows(path)
    root = Path(path)
    action_templates_payload: Any = []
    if (root / "action_templates.json").exists():
        action_templates_payload = load_json(root / "action_templates.json")
    if isinstance(action_templates_payload, dict):
        action_templates_rows = action_templates_payload.get("action_templates", [])
    elif isinstance(action_templates_payload, list):
        action_templates_rows = action_templates_payload
    else:
        action_templates_rows = []
    episode_object_names: list[str] = []
    episode_inventory_path = root / "episode_object_inventory.jsonl"
    if episode_inventory_path.exists():
        for item in load_jsonl(episode_inventory_path):
            if item.get("episode_name") != episode_name:
                continue
            episode_object_names = [
                _validate_snake_case(str(name), field_name="episode_object_name")
                for name in item.get("object_names", [])
                if str(name).strip()
            ]
            break

    parsed_schemas = [
        ActionSchemaArtifact(
            canonical_action_name=_validate_snake_case(
                str(item["canonical_action_name"]), field_name="canonical_action_name"
            ),
            action_category=str(item["action_category"]).strip(),
            parameter_count=int(item["parameter_count"]),
            parameter_roles=[str(role).strip() for role in item.get("parameter_roles", []) if str(role).strip()],
            precondition_literals=_coerce_fact_list(
                item.get("precondition_literals", []), field_name="precondition_literals"
            ),
        )
        for item in rows.action_schemas
    ]
    taxonomy_records = []
    for item in rows.taxonomy_records:
        if item.get("episode_name") != episode_name:
            continue
        taxonomy_records.append(
            ActionTaxonomyArtifact(
                episode_name=str(item["episode_name"]),
                step_index=int(item["step_index"]),
                canonical_action_name=_validate_snake_case(
                    str(item["canonical_action_name"]), field_name="canonical_action_name"
                ),
                action_category=str(item["action_category"]).strip(),
                action_arguments=[
                    _validate_snake_case(str(arg), field_name="action_arguments")
                    for arg in item.get("action_arguments", [])
                ],
                raw_action_text=str(item.get("raw_action_text", "")),
                observation_text=_normalize_optional_text(item.get("observation_text")),
                extra_info=_normalize_optional_text(item.get("extra_info")),
            )
        )
    manipulation_records = []
    for item in rows.manipulation_records:
        if item.get("episode_name") != episode_name:
            continue
        manipulation_records.append(
            ManipulationRecordArtifact(
                episode_name=str(item["episode_name"]),
                step_index=int(item["step_index"]),
                canonical_action_name=_validate_snake_case(
                    str(item["canonical_action_name"]), field_name="canonical_action_name"
                ),
                action_arguments=[
                    _validate_snake_case(str(arg), field_name="action_arguments")
                    for arg in item.get("action_arguments", [])
                ],
                effect_bucket=_validate_snake_case(str(item["effect_bucket"]), field_name="effect_bucket"),
                delta_add=_coerce_fact_list(item.get("delta_add", []), field_name="delta_add"),
                delta_del=_coerce_fact_list(item.get("delta_del", []), field_name="delta_del"),
                success=bool(item.get("success")),
                raw_action_text=str(item.get("raw_action_text", "")),
            )
        )
    return DomainLearningArtifacts(
        action_schemas=parsed_schemas,
        taxonomy_records=sorted(taxonomy_records, key=lambda item: item.step_index),
        manipulation_records=sorted(manipulation_records, key=lambda item: item.step_index),
        episode_object_names=episode_object_names,
        action_templates=[item for item in action_templates_rows if isinstance(item, dict)],
        action_name_map=load_json_object(root / "action_name_map.json")
        if (root / "action_name_map.json").exists()
        else {},
    )


@dataclass
class LLMGoalInferenceModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1800
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("goal_inference_prompt.md")
        self.last_raw_output: str | None = None

    def induce_goal_facts(
        self,
        *,
        episode: EpisodeContext,
        domain_name: str,
        predicate_names: list[str],
        problem_spec_without_goal: ProblemSpec,
        predicate_schemas: list[dict[str, object]] | None = None,
        type_memberships: dict[str, list[str]] | None = None,
        review_guidance: dict[str, object] | None = None,
    ) -> list[str]:
        logger.debug("Goal inference (LLM): inducing goal for %s", episode.episode_name)
        base_payload = {
            "domain_name": domain_name,
            "allowed_predicates": sorted(predicate_names),
            "predicate_signatures": list(predicate_schemas or []),
            "type_memberships": dict(type_memberships or {}),
            "instruction": episode.instruction,
            "objects": [item.to_dict() for item in problem_spec_without_goal.objects],
            "init_facts": list(problem_spec_without_goal.init_facts),
            "review_guidance": review_guidance or {},
        }
        validation_error: str | None = None
        for _attempt in range(3):
            payload = dict(base_payload)
            if validation_error:
                payload["validation_feedback"] = validation_error
            reply = safe_chat(
                self._client,
                self._prompt,
                json.dumps(payload, ensure_ascii=False, indent=2),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                verbose=self.verbose,
            )
            self.last_raw_output = reply
            try:
                data = extract_json_object(reply)
                goal_facts = _coerce_fact_list(data.get("goal_facts", []), field_name="goal_facts")
                self._validate_goal_facts(
                    goal_facts=goal_facts,
                    predicate_names=predicate_names,
                    predicate_schemas=list(predicate_schemas or []),
                    type_memberships=dict(type_memberships or {}),
                    problem_spec_without_goal=problem_spec_without_goal,
                )
                return list(dict.fromkeys(goal_facts))
            except ValueError as exc:
                validation_error = str(exc)
                logger.warning(
                    "Rejected goal inference response for %s: %s",
                    episode.episode_name,
                    validation_error,
                )
        raise ValueError(f"Goal inference failed signature validation: {validation_error}")

    @staticmethod
    def _validate_goal_facts(
        *,
        goal_facts: list[str],
        predicate_names: list[str],
        predicate_schemas: list[dict[str, object]],
        type_memberships: dict[str, list[str]],
        problem_spec_without_goal: ProblemSpec,
    ) -> None:
        allowed_predicates = set(predicate_names)
        object_types = {item.name: item.type_name for item in problem_spec_without_goal.objects}
        signature_by_name = {
            str(row.get("predicate_name") or "").strip(): [
                str(item).strip() for item in row.get("parameter_types", []) if str(item).strip()
            ]
            for row in predicate_schemas
            if isinstance(row, dict) and str(row.get("predicate_name") or "").strip()
        }
        for fact in goal_facts:
            predicate = _extract_fact_predicate(fact)
            if predicate is None or predicate not in allowed_predicates:
                raise ValueError(f"Goal inference returned unsupported predicate in fact {fact!r}")
            arguments = _extract_fact_arguments(fact)
            expected_types = signature_by_name.get(predicate)
            if expected_types is not None and len(arguments) != len(expected_types):
                raise ValueError(
                    f"Goal fact {fact!r} has arity {len(arguments)} but {predicate!r} "
                    f"expects {len(expected_types)} arguments"
                )
            for index, argument in enumerate(arguments):
                if argument.startswith("?"):
                    raise ValueError(f"Goal inference must return grounded object names, got variable {argument!r}")
                object_type = object_types.get(argument)
                if object_type is None:
                    raise ValueError(f"Goal inference returned unknown object {argument!r} in fact {fact!r}")
                if expected_types is None:
                    continue
                expected_type = expected_types[index]
                memberships = set(type_memberships.get(object_type, []))
                if expected_type not in {"object", object_type} and expected_type not in memberships:
                    raise ValueError(
                        f"Goal fact {fact!r} uses {argument!r} of type {object_type!r} "
                        f"at argument {index + 1}, which expects {expected_type!r}"
                    )


@dataclass(frozen=True)
class RuleBasedProblemAssemblyModule:
    def assemble_problem_spec(
        self,
        *,
        base_problem_spec: ProblemSpec,
        goal_facts: list[str],
    ) -> ProblemSpec:
        deduped_goals: list[str] = []
        seen: set[str] = set()
        for fact in goal_facts:
            if fact not in seen:
                seen.add(fact)
                deduped_goals.append(fact)
        return ProblemSpec(
            problem_name=base_problem_spec.problem_name,
            domain_name=base_problem_spec.domain_name,
            objects=base_problem_spec.objects,
            init_facts=base_problem_spec.init_facts,
            goal_facts=deduped_goals,
            canonical_object_map=base_problem_spec.canonical_object_map,
        )


def ground_trajectory_steps(
    episode: EpisodeContext,
    artifacts: DomainLearningArtifacts,
) -> list[GroundedTrajectoryStep]:
    taxonomy_by_step = {record.step_index: record for record in artifacts.taxonomy_records}
    manipulation_by_step = {record.step_index: record for record in artifacts.manipulation_records}

    grounded_steps: list[GroundedTrajectoryStep] = []
    for step in sorted((item for item in episode.steps if item.step_index > 0), key=lambda item: item.step_index):
        taxonomy = taxonomy_by_step.get(step.step_index)
        manipulation = manipulation_by_step.get(step.step_index)
        action_name = taxonomy.canonical_action_name if taxonomy else None
        action_category = taxonomy.action_category if taxonomy else None
        arguments = list(taxonomy.action_arguments) if taxonomy else []
        ground_action_pddl = None
        if action_name is not None:
            if arguments:
                ground_action_pddl = f"({action_name} {' '.join(arguments)})"
            else:
                ground_action_pddl = f"({action_name})"
        grounded_steps.append(
            GroundedTrajectoryStep(
                episode_name=episode.episode_name,
                step_index=step.step_index,
                raw_action_text=step.action_text,
                action_category=action_category,
                canonical_action_name=action_name,
                ground_arguments=arguments,
                ground_action_pddl=ground_action_pddl,
                effect_bucket=manipulation.effect_bucket if manipulation else None,
                delta_add=list(manipulation.delta_add) if manipulation else [],
                delta_del=list(manipulation.delta_del) if manipulation else [],
                success=manipulation.success if manipulation else None,
                observation_text=step.observation_text,
                extra_info=step.extra_info,
            )
        )
    return grounded_steps
