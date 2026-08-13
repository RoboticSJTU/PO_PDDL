from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from po_pddl.core.parser import parse_domain, parse_problem
from po_pddl.domain_generation.stages.problem_grounding import (
    canonicalize_domain_learning_artifacts,
    ground_trajectory_steps,
    load_domain_learning_artifacts,
)
from po_pddl.domain_generation.stages.problem_grounding.models import ObjectDeclaration, ProblemSpec
from po_pddl.domain_generation.stages.problem_grounding.modules import load_episode_context
from po_pddl.domain_generation.stages.problem_grounding.validator import build_initial_state, execute_grounded_step

from .models import (
    InferredInitFact,
    InitCompletionFact,
    LatentObjectCandidate,
    ProblemInferenceResult,
    VisibleObjectCandidate,
)
from .modules import (
    InitCompletionModule,
    InitialStateInferenceModule,
    LatentObjectDiscoveryModule,
    VisibleObjectExtractionModule,
    summarize_parsed_domain,
)
from .renderer import render_problem_pddl

logger = logging.getLogger(__name__)
_BARE_PREDICATE_PATTERN = re.compile(r"^(?:not\s+)?[a-z][a-z0-9_]*$")
_FACT_PATTERN = re.compile(r"^(?:not\s+)?([a-z][a-z0-9_]*)\(([^()]*)\)$")


def _normalize_allowed_object_name(name: str, allowed_object_names: set[str], *, field_name: str) -> str:
    normalized = str(name).strip()
    if normalized in allowed_object_names:
        return normalized
    coarse_matches = [
        candidate
        for candidate in allowed_object_names
        if normalized == candidate or normalized.startswith(f"{candidate}_")
    ]
    if len(coarse_matches) == 1:
        return coarse_matches[0]
    raise ValueError(
        f"{field_name} returned object name {normalized!r} outside the parsed episode object inventory. "
        f"Allowed names: {sorted(allowed_object_names)}"
    )


def _canonicalize_fact_to_allowed(fact: str, allowed_object_names: set[str]) -> str:
    normalized = _normalize_fact_literal(fact)
    match = _FACT_PATTERN.fullmatch(normalized.strip())
    if match is None:
        return normalized
    predicate_name = match.group(1)
    raw_args = match.group(2).strip()
    arguments = [arg.strip() for arg in raw_args.split(",")] if raw_args else []
    canonical_arguments = [
        _normalize_allowed_object_name(argument, allowed_object_names, field_name="fact_argument")
        for argument in arguments
    ]
    if canonical_arguments:
        return f"{predicate_name}({','.join(canonical_arguments)})"
    return f"{predicate_name}()"


def _normalize_visible_objects_to_allowed(
    visible_objects: list[VisibleObjectCandidate],
    *,
    allowed_object_names: set[str],
) -> tuple[list[VisibleObjectCandidate], dict[str, str]]:
    normalized_candidates: list[VisibleObjectCandidate] = []
    alias_map: dict[str, str] = {}
    for candidate in visible_objects:
        canonical_name = _normalize_allowed_object_name(
            candidate.name,
            allowed_object_names,
            field_name="visible_object_name",
        )
        alias_map[candidate.name] = canonical_name
        normalized_candidates.append(
            VisibleObjectCandidate(
                name=canonical_name,
                type_name=candidate.type_name,
                supporting_observation=candidate.supporting_observation,
                visible_facts=[
                    _canonicalize_fact_to_allowed(fact, allowed_object_names) for fact in candidate.visible_facts
                ],
            )
        )
    return normalized_candidates, alias_map


def _normalize_latent_objects_to_allowed(
    latent_objects: list[LatentObjectCandidate],
    *,
    allowed_object_names: set[str],
    alias_map: dict[str, str],
) -> tuple[list[LatentObjectCandidate], dict[str, str]]:
    normalized_candidates: list[LatentObjectCandidate] = []
    for candidate in latent_objects:
        canonical_name = _normalize_allowed_object_name(
            candidate.name,
            allowed_object_names,
            field_name="latent_object_name",
        )
        alias_map[candidate.name] = canonical_name
        normalized_candidates.append(
            LatentObjectCandidate(
                name=canonical_name,
                type_name=candidate.type_name,
                evidence=list(candidate.evidence),
            )
        )
    return normalized_candidates, alias_map


@dataclass(frozen=True)
class ProblemInferenceSummary:
    visible_object_count: int
    latent_object_count: int
    init_fact_count: int
    problem_name: str
    domain_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "visible_object_count": self.visible_object_count,
            "latent_object_count": self.latent_object_count,
            "init_fact_count": self.init_fact_count,
            "problem_name": self.problem_name,
            "domain_name": self.domain_name,
        }


def _dedupe_object_declarations(
    visible_objects: list[VisibleObjectCandidate],
    latent_objects: list[LatentObjectCandidate],
) -> list[ObjectDeclaration]:
    ordered: list[ObjectDeclaration] = []
    seen: dict[str, str] = {}
    for item in visible_objects:
        if item.name not in seen:
            seen[item.name] = item.type_name
            ordered.append(ObjectDeclaration(name=item.name, type_name=item.type_name))
    for item in latent_objects:
        if item.name not in seen:
            seen[item.name] = item.type_name
            ordered.append(ObjectDeclaration(name=item.name, type_name=item.type_name))
    return ordered


def _merge_init_facts(
    visible_objects: list[VisibleObjectCandidate],
    visible_facts: list[str],
    inferred_init_facts: list[InferredInitFact],
) -> list[str]:
    ordered_facts: list[str] = []
    seen: set[str] = set()
    for candidate in visible_objects:
        for fact in candidate.visible_facts:
            normalized = _normalize_fact_literal(fact)
            if normalized not in seen:
                seen.add(normalized)
                ordered_facts.append(normalized)
    for fact in visible_facts:
        normalized = _normalize_fact_literal(fact)
        if normalized not in seen:
            seen.add(normalized)
            ordered_facts.append(normalized)
    for item in inferred_init_facts:
        normalized = _normalize_fact_literal(item.fact)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered_facts.append(normalized)
    return ordered_facts


def _canonicalize_init_facts_to_declared_objects(
    init_facts: list[str],
    objects: list[ObjectDeclaration],
) -> list[str]:
    allowed_object_names = {item.name for item in objects}
    canonicalized: list[str] = []
    for fact in init_facts:
        canonicalized.append(_canonicalize_fact_to_allowed(fact, allowed_object_names))
    return canonicalized


def _normalize_fact_literal(text: str) -> str:
    stripped = text.strip()
    if _BARE_PREDICATE_PATTERN.fullmatch(stripped):
        if stripped.startswith("not "):
            return f"not {stripped[4:].strip()}()"
        return f"{stripped}()"
    return stripped


def _augment_objects_from_init_facts(
    objects: list[ObjectDeclaration],
    init_facts: list[str],
    *,
    parsed_domain,
) -> list[ObjectDeclaration]:
    ordered = list(objects)
    seen = {item.name for item in ordered}
    predicate_types: dict[str, list[str]] = {}
    for predicate, parameter_types in parsed_domain.predicate_parameter_types.items():
        predicate_types[predicate.name] = [type_name for _, type_name in parameter_types]

    for fact in init_facts:
        match = _FACT_PATTERN.fullmatch(fact.strip())
        if match is None:
            continue
        predicate_name = match.group(1)
        raw_args = match.group(2).strip()
        arguments = [arg.strip() for arg in raw_args.split(",")] if raw_args else []
        argument_types = predicate_types.get(predicate_name, [])
        for index, argument in enumerate(arguments):
            if not argument or argument in seen or argument.startswith("?"):
                continue
            type_name = argument_types[index] if index < len(argument_types) else "object"
            ordered.append(ObjectDeclaration(name=argument, type_name=type_name))
            seen.add(argument)
    return ordered


class ProblemInferenceLearner:
    def __init__(
        self,
        *,
        visible_object_module: VisibleObjectExtractionModule,
        latent_object_module: LatentObjectDiscoveryModule,
        initial_state_module: InitialStateInferenceModule,
        init_completion_module: InitCompletionModule | None = None,
    ) -> None:
        self._visible_object_module = visible_object_module
        self._latent_object_module = latent_object_module
        self._initial_state_module = initial_state_module
        self._init_completion_module = init_completion_module
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _collect_raw_llm_outputs(*, learner: "ProblemInferenceLearner") -> dict[str, str]:
        outputs: dict[str, str] = {}
        module_mapping = {
            "visible_object_extraction": learner._visible_object_module,
            "latent_object_discovery": learner._latent_object_module,
            "initial_state_inference": learner._initial_state_module,
            "init_completion": learner._init_completion_module,
        }
        for key, module in module_mapping.items():
            if module is None:
                continue
            raw_output = getattr(module, "last_raw_output", None)
            if isinstance(raw_output, str) and raw_output.strip():
                outputs[key] = raw_output
        return outputs

    def infer_from_files(
        self,
        *,
        domain_file: str | Path,
        episode_file: str | Path,
        domain_learning_dir: str | Path | None = None,
        review_guidance: dict[str, object] | None = None,
    ) -> ProblemInferenceResult:
        domain_path = Path(domain_file)
        episode_path = Path(episode_file)
        self._logger.info("Stage 0/4: loading domain and episode context")
        parsed_domain = parse_domain(domain_path.read_text(encoding="utf-8"))
        domain_summary = summarize_parsed_domain(parsed_domain)
        episode = load_episode_context(episode_path)
        allowed_object_names: list[str] | None = None
        if domain_learning_dir is not None:
            artifacts_for_inventory = load_domain_learning_artifacts(
                domain_learning_dir, episode_name=episode.episode_name
            )
            if artifacts_for_inventory.episode_object_names:
                allowed_object_names = list(artifacts_for_inventory.episode_object_names)
        allowed_object_name_set = set(allowed_object_names or [])

        self._logger.info("Stage 1/4: visible object extraction")
        visible_objects, visible_facts = self._visible_object_module.extract_visible_objects(
            episode=episode,
            domain_summary=domain_summary,
            allowed_object_names=allowed_object_names,
            review_guidance=review_guidance,
        )
        canonical_object_map: dict[str, str] = {}
        if allowed_object_name_set:
            visible_objects, canonical_object_map = _normalize_visible_objects_to_allowed(
                visible_objects,
                allowed_object_names=allowed_object_name_set,
            )
            visible_facts = [_canonicalize_fact_to_allowed(fact, allowed_object_name_set) for fact in visible_facts]

        self._logger.info("Stage 2/4: latent object discovery")
        latent_objects = self._latent_object_module.discover_latent_objects(
            episode=episode,
            domain_summary=domain_summary,
            visible_objects=visible_objects,
            visible_facts=visible_facts,
            allowed_object_names=allowed_object_names,
            review_guidance=review_guidance,
        )
        if allowed_object_name_set:
            latent_objects, canonical_object_map = _normalize_latent_objects_to_allowed(
                latent_objects,
                allowed_object_names=allowed_object_name_set,
                alias_map=canonical_object_map,
            )

        self._logger.info("Stage 3/4: initial state inference")
        inferred_init_facts = self._initial_state_module.infer_initial_state(
            episode=episode,
            domain_summary=domain_summary,
            visible_objects=visible_objects,
            visible_facts=visible_facts,
            latent_objects=latent_objects,
            review_guidance=review_guidance,
        )
        if allowed_object_name_set:
            inferred_init_facts = [
                InferredInitFact(
                    fact=_canonicalize_fact_to_allowed(item.fact, allowed_object_name_set),
                    confidence=item.confidence,
                    justification=item.justification,
                )
                for item in inferred_init_facts
            ]

        self._logger.info("Stage 4/5: problem assembly")
        problem_spec = ProblemSpec(
            problem_name=f"{episode.episode_name}_inferred_problem",
            domain_name=parsed_domain.domain_name,
            objects=_dedupe_object_declarations(visible_objects, latent_objects),
            init_facts=[],
            goal_facts=[],
            canonical_object_map=canonical_object_map,
        )
        problem_spec = ProblemSpec(
            problem_name=problem_spec.problem_name,
            domain_name=problem_spec.domain_name,
            objects=problem_spec.objects,
            init_facts=_canonicalize_init_facts_to_declared_objects(
                _merge_init_facts(visible_objects, visible_facts, inferred_init_facts),
                problem_spec.objects,
            ),
            goal_facts=problem_spec.goal_facts,
            canonical_object_map=problem_spec.canonical_object_map,
        )
        completed_init_facts: list[InitCompletionFact] = []
        completion_diagnostics: dict[str, Any] = {"enabled": False}

        if self._init_completion_module is not None and domain_learning_dir is not None:
            self._logger.info("Stage 5/5: init completion from grounded replay hints")
            initial_problem_pddl = render_problem_pddl(problem_spec)
            parsed_problem = parse_problem(initial_problem_pddl)
            artifacts = load_domain_learning_artifacts(domain_learning_dir, episode_name=episode.episode_name)
            artifacts = canonicalize_domain_learning_artifacts(artifacts, problem_spec.canonical_object_map)
            grounded_steps = ground_trajectory_steps(episode, artifacts)
            state = build_initial_state(parsed_problem)
            object_names = set(parsed_problem.objects)
            action_map = {schema.action.name: schema for schema in parsed_domain.actions}
            grounded_state_trace: list[dict[str, object]] = []
            for step in grounded_steps:
                report, _, next_state = execute_grounded_step(
                    parsed_domain=parsed_domain,
                    object_names=object_names,
                    state=state,
                    step=step,
                    action_map=action_map,
                )
                grounded_state_trace.append(report.to_dict())
                state = next_state
            completed_init_facts = self._init_completion_module.complete_initial_state(
                episode=episode,
                domain_summary=domain_summary,
                existing_true_init_facts=list(problem_spec.init_facts),
                grounded_steps=[item.to_dict() for item in grounded_steps],
                grounded_state_trace=grounded_state_trace,
                review_guidance=review_guidance,
            )
            if allowed_object_name_set:
                completed_init_facts = [
                    InitCompletionFact(
                        fact=_canonicalize_fact_to_allowed(item.fact, allowed_object_name_set),
                        confidence=item.confidence,
                        justification=item.justification,
                    )
                    for item in completed_init_facts
                ]
            if completed_init_facts:
                problem_spec = ProblemSpec(
                    problem_name=problem_spec.problem_name,
                    domain_name=problem_spec.domain_name,
                    objects=problem_spec.objects,
                    init_facts=_canonicalize_init_facts_to_declared_objects(
                        _merge_init_facts(
                            visible_objects=[],
                            visible_facts=list(problem_spec.init_facts),
                            inferred_init_facts=[
                                InferredInitFact(
                                    fact=item.fact,
                                    confidence=item.confidence,
                                    justification=item.justification,
                                )
                                for item in completed_init_facts
                            ],
                        ),
                        problem_spec.objects,
                    ),
                    goal_facts=problem_spec.goal_facts,
                    canonical_object_map=problem_spec.canonical_object_map,
                )
            completion_diagnostics = {
                "enabled": True,
                "grounded_step_count": len(grounded_steps),
                "completed_init_fact_count": len(completed_init_facts),
            }

        problem_pddl = render_problem_pddl(problem_spec)
        diagnostics = {
            "domain_summary": domain_summary,
            "episode_name": episode.episode_name,
            "step_count": len(episode.steps),
            "init_completion": completion_diagnostics,
            "review_guidance": review_guidance or {},
        }
        return ProblemInferenceResult(
            problem_spec=problem_spec,
            problem_pddl=problem_pddl,
            visible_objects=visible_objects,
            latent_objects=latent_objects,
            inferred_init_facts=inferred_init_facts,
            completed_init_facts=completed_init_facts,
            diagnostics=diagnostics,
            raw_llm_outputs=self._collect_raw_llm_outputs(learner=self),
        )

    def write_outputs(self, result: ProblemInferenceResult, output_dir: str | Path) -> ProblemInferenceSummary:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "visible_object_extraction.json").write_text(
            json.dumps([item.to_dict() for item in result.visible_objects], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_path / "latent_object_discovery.json").write_text(
            json.dumps([item.to_dict() for item in result.latent_objects], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_path / "initial_state_inference.json").write_text(
            json.dumps([item.to_dict() for item in result.inferred_init_facts], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_path / "init_completion.json").write_text(
            json.dumps([item.to_dict() for item in result.completed_init_facts], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_path / "problem_inference_diagnostics.json").write_text(
            json.dumps(result.diagnostics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raw_output_files: dict[str, str] = {}
        for name, text in result.raw_llm_outputs.items():
            path = output_path / f"{name}_raw_output.txt"
            path.write_text(text, encoding="utf-8")
            raw_output_files[name] = str(path)
        (output_path / "problem_summary.json").write_text(
            json.dumps(result.summary_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_path / "problem.pddl").write_text(result.problem_pddl, encoding="utf-8")
        summary = ProblemInferenceSummary(
            visible_object_count=len(result.visible_objects),
            latent_object_count=len(result.latent_objects),
            init_fact_count=len(result.problem_spec.init_facts),
            problem_name=result.problem_spec.problem_name,
            domain_name=result.problem_spec.domain_name,
        )
        (output_path / "summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if raw_output_files:
            (output_path / "raw_llm_output_manifest.json").write_text(
                json.dumps(raw_output_files, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return summary
