from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..core.models.belief_factor import BeliefFactor
from ..core.models.factorized_belief import FactorizedBelief
from ..core.models.predicate import Predicate
from ..core.parser import parse_problem
from ..domain_generation.stages.problem_grounding.models import ObjectDeclaration
from .domain_analysis import analyze_domain, build_grounded_predicates_for_objects
from .goal import GoalInferenceAgent
from .images import prepare_scene_image
from .initial_belief import InitialBeliefGenerator
from .models import OnlinePlanningProblemResult, OnlinePlanningProblemSpec, VisibleObject
from .objects import VisibleObjectExtractionAgent
from .rendering import render_online_problem_pddl


def _write_problem_output(output_path: str | Path, problem_pddl: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(problem_pddl, encoding="utf-8")
    return path


class ProblemGenerator:
    def __init__(
        self,
        *,
        object_agent: VisibleObjectExtractionAgent | None = None,
        init_belief_agent: InitialBeliefGenerator | None = None,
        goal_agent: GoalInferenceAgent | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.object_agent = object_agent or VisibleObjectExtractionAgent()
        self.init_belief_agent = init_belief_agent or InitialBeliefGenerator()
        self.goal_agent = goal_agent or GoalInferenceAgent()
        self._logger = logger

    def _log(self, message: str) -> None:
        if self._logger is not None:
            self._logger(message)

    @staticmethod
    def _is_last_action_helper(predicate: Predicate) -> bool:
        return predicate.name.startswith("last_action_")

    @staticmethod
    def _is_gripper_empty_predicate(predicate: Predicate) -> bool:
        return predicate.name == "gripper_empty"

    @staticmethod
    def _is_gripper_holding_predicate(predicate: Predicate) -> bool:
        return predicate.name == "gripper_holding"

    @staticmethod
    def _load_close_domain_object_allowlist(
        domain_file: str | Path | None,
    ) -> set[str] | None:
        if domain_file is None:
            return None
        allowlist_path = Path(domain_file).resolve().parent / "objects.txt"
        if not allowlist_path.exists() or not allowlist_path.is_file():
            return None
        tokens = allowlist_path.read_text(encoding="utf-8").split()
        allowlist = {token.strip() for token in tokens if token.strip()}
        return allowlist or set()

    @classmethod
    def _force_predicates_false_in_belief(
        cls,
        belief: FactorizedBelief,
        predicates_to_force_false: list[Predicate],
    ) -> FactorizedBelief:
        forced_false = set(predicates_to_force_false)
        if not forced_false:
            return belief

        known_true = [predicate for predicate in belief.known_true if predicate not in forced_false]
        known_false_set = set(belief.known_false) | forced_false
        updated_factors: list[BeliefFactor] = []

        for factor in belief.factors:
            helper_scope = [predicate for predicate in factor.scope if predicate in forced_false]
            if not helper_scope:
                updated_factors.append(factor)
                continue

            remaining_scope = [predicate for predicate in factor.scope if predicate not in forced_false]
            if not remaining_scope:
                continue

            aggregated_probabilities: dict[tuple[Predicate, ...], float] = {}
            helper_scope_set = set(helper_scope)
            for probability, true_predicates in factor.cases:
                true_set = set(true_predicates)
                if true_set & helper_scope_set:
                    continue
                remaining_true = tuple(
                    sorted(
                        (predicate for predicate in true_predicates if predicate in remaining_scope),
                        key=lambda item: item.to_pddl_str(),
                    )
                )
                aggregated_probabilities[remaining_true] = aggregated_probabilities.get(remaining_true, 0.0) + float(
                    probability
                )

            total_probability = sum(aggregated_probabilities.values())
            if total_probability <= 0.0:
                normalized_cases = [(1.0, [])]
            else:
                normalized_cases = [
                    (probability / total_probability, list(true_predicates))
                    for true_predicates, probability in sorted(
                        aggregated_probabilities.items(),
                        key=lambda item: [predicate.to_pddl_str() for predicate in item[0]],
                    )
                ]
            updated_factors.append(
                BeliefFactor(
                    name=f"{factor.name}:forced_observation_helpers_false",
                    scope=remaining_scope,
                    cases=normalized_cases,
                )
            )

        forced_belief = FactorizedBelief(
            known_true=sorted(known_true, key=lambda item: item.to_pddl_str()),
            known_false=sorted(known_false_set, key=lambda item: item.to_pddl_str()),
            factors=updated_factors,
        )
        forced_belief.validate()
        return forced_belief

    @classmethod
    def _force_predicates_true_in_belief(
        cls,
        belief: FactorizedBelief,
        predicates_to_force_true: list[Predicate],
    ) -> FactorizedBelief:
        forced_true = set(predicates_to_force_true)
        if not forced_true:
            return belief

        known_true_set = set(belief.known_true) | forced_true
        known_false = [predicate for predicate in belief.known_false if predicate not in forced_true]
        updated_factors: list[BeliefFactor] = []

        for factor in belief.factors:
            helper_scope = [predicate for predicate in factor.scope if predicate in forced_true]
            if not helper_scope:
                updated_factors.append(factor)
                continue

            remaining_scope = [predicate for predicate in factor.scope if predicate not in forced_true]
            if not remaining_scope:
                continue

            aggregated_probabilities: dict[tuple[Predicate, ...], float] = {}
            helper_scope_set = set(helper_scope)
            for probability, true_predicates in factor.cases:
                true_set = set(true_predicates)
                if not helper_scope_set.issubset(true_set):
                    continue
                remaining_true = tuple(
                    sorted(
                        (predicate for predicate in true_predicates if predicate in remaining_scope),
                        key=lambda item: item.to_pddl_str(),
                    )
                )
                aggregated_probabilities[remaining_true] = aggregated_probabilities.get(remaining_true, 0.0) + float(
                    probability
                )

            total_probability = sum(aggregated_probabilities.values())
            if total_probability <= 0.0:
                normalized_cases = [(1.0, [])]
            else:
                normalized_cases = [
                    (probability / total_probability, list(true_predicates))
                    for true_predicates, probability in sorted(
                        aggregated_probabilities.items(),
                        key=lambda item: [predicate.to_pddl_str() for predicate in item[0]],
                    )
                ]
            updated_factors.append(
                BeliefFactor(
                    name=f"{factor.name}:forced_helpers_true",
                    scope=remaining_scope,
                    cases=normalized_cases,
                )
            )

        forced_belief = FactorizedBelief(
            known_true=sorted(known_true_set, key=lambda item: item.to_pddl_str()),
            known_false=sorted(known_false, key=lambda item: item.to_pddl_str()),
            factors=updated_factors,
        )
        forced_belief.validate()
        return forced_belief

    @staticmethod
    def _resolve_manipulation_records_path(
        *,
        final_bundle_dir: str | Path | None,
    ) -> Path | None:
        if final_bundle_dir is None:
            return None
        bundle_root = Path(final_bundle_dir).resolve()
        candidates = [
            bundle_root / "manipulation_domain" / "manipulation_records.jsonl",
            bundle_root / "manipulation_records.jsonl",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _resolve_historical_grounding_root(
        *,
        image_path: str | Path,
        final_bundle_dir: str | Path | None,
        output_path: str | Path | None,
    ) -> Path | None:
        candidates: list[Path] = []
        image_root = Path(image_path)
        if image_root.exists() and image_root.is_dir():
            candidates.append(image_root / "4_problem_grounding_all")
        if final_bundle_dir is not None:
            candidates.append(Path(final_bundle_dir).resolve() / "problem_grounding_all")
        if output_path is not None:
            candidates.append(Path(output_path).resolve().parent / "4_problem_grounding_all")
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    @staticmethod
    def _select_merged_object_type(
        *,
        current_type: str | None,
        candidate_type: str,
        domain_analysis: object,
    ) -> str:
        if current_type is None or current_type == candidate_type:
            return candidate_type
        parsed_domain = domain_analysis.parsed_domain
        current_node = parsed_domain.types.get(current_type)
        candidate_node = parsed_domain.types.get(candidate_type)
        if current_node is not None and current_node.is_subtype_of(candidate_type):
            return current_type
        if candidate_node is not None and candidate_node.is_subtype_of(current_type):
            return candidate_type
        if current_type == "object":
            return candidate_type
        if candidate_type == "object":
            return current_type
        raise ValueError(
            f"Conflicting object types while merging close-domain inventory: {current_type!r} vs {candidate_type!r}."
        )

    def _infer_closed_domain_objects_from_grounding(
        self,
        *,
        domain_analysis,
        historical_grounding_root: str | Path,
        allowed_object_names: set[str] | None = None,
    ) -> list[VisibleObject]:
        grounding_root = Path(historical_grounding_root).resolve()
        if not grounding_root.exists() or not grounding_root.is_dir():
            raise ValueError(
                "Close-domain object generation requires an existing historical grounding directory, "
                f"but got: {grounding_root}"
            )
        episode_problem_files = sorted(
            path for path in grounding_root.iterdir() if path.is_dir() and (path / "problem.pddl").exists()
        )
        if not episode_problem_files:
            raise ValueError(
                "Close-domain object generation requires episode problem files under historical grounding root, "
                f"but none were found in: {grounding_root}"
            )

        merged_types_by_name: dict[str, str] = {}
        seen_in_episodes: dict[str, set[str]] = {}
        for episode_dir in episode_problem_files:
            parsed_problem = parse_problem((episode_dir / "problem.pddl").read_text(encoding="utf-8"))
            for object_name, type_name in sorted(parsed_problem.objects.items()):
                merged_types_by_name[object_name] = self._select_merged_object_type(
                    current_type=merged_types_by_name.get(object_name),
                    candidate_type=type_name,
                    domain_analysis=domain_analysis,
                )
                seen_in_episodes.setdefault(object_name, set()).add(episode_dir.name)

        visible_objects: list[VisibleObject] = []
        for object_name in sorted(merged_types_by_name):
            if allowed_object_names is not None and object_name not in allowed_object_names:
                continue
            type_name = merged_types_by_name[object_name]
            episode_names = sorted(seen_in_episodes.get(object_name, set()))
            justification = "close-domain inventory merged from historical grounding episodes: " + ", ".join(
                episode_names
            )
            visible_objects.append(
                VisibleObject(
                    name=object_name,
                    type_name=type_name,
                    justification=justification,
                )
            )
        return visible_objects

    def _merge_closed_domain_objects(
        self,
        *,
        domain_analysis,
        object_groups: list[list[VisibleObject]],
    ) -> list[VisibleObject]:
        merged: dict[str, VisibleObject] = {}
        for objects in object_groups:
            for item in objects:
                previous = merged.get(item.name)
                type_name = self._select_merged_object_type(
                    current_type=previous.type_name if previous is not None else None,
                    candidate_type=item.type_name,
                    domain_analysis=domain_analysis,
                )
                justifications = [
                    text
                    for text in (
                        previous.justification if previous is not None else None,
                        item.justification,
                    )
                    if text
                ]
                merged[item.name] = VisibleObject(
                    name=item.name,
                    type_name=type_name,
                    justification="; ".join(dict.fromkeys(justifications)) or None,
                )
        return [merged[name] for name in sorted(merged)]

    @staticmethod
    def _restrict_objects_to_allowlist(
        objects: list[VisibleObject],
        allowed_object_names: set[str] | None,
    ) -> tuple[list[VisibleObject], list[str]]:
        if allowed_object_names is None:
            return list(objects), []
        accepted = [item for item in objects if item.name in allowed_object_names]
        rejected_names = sorted({item.name for item in objects if item.name not in allowed_object_names})
        return accepted, rejected_names

    def build_problem_from_text(
        self,
        *,
        domain_text: str,
        domain_file: str | Path | None = None,
        image_path: str | Path,
        instruction: str,
        initial_state_hint: str | None = None,
        final_bundle_dir: str | Path | None = None,
        reuse_problem_file: str | Path | None = None,
        problem_name: str | None = None,
        output_path: str | Path | None = None,
        max_workers: int = 8,
        prior_data_confidence: float = 0.0,
        close_domain: bool = False,
        skip_init_observation: bool = False,
    ) -> OnlinePlanningProblemResult:
        self._log("Loading and analyzing domain")
        domain_analysis = analyze_domain(domain_text)
        resolved_problem_name = problem_name or f"{domain_analysis.parsed_domain.domain_name}_online_problem"
        manipulation_records_path = self._resolve_manipulation_records_path(
            final_bundle_dir=final_bundle_dir,
        )
        historical_grounding_root = self._resolve_historical_grounding_root(
            image_path=image_path,
            final_bundle_dir=final_bundle_dir,
            output_path=output_path,
        )
        self._log(
            "Domain analysis ready: "
            f"domain={domain_analysis.parsed_domain.domain_name}, "
            f"observation_module={'yes' if domain_analysis.has_observation_module else 'no'}"
        )
        if historical_grounding_root is not None and prior_data_confidence > 0.0:
            self._log(f"Historical grounding prior source: {historical_grounding_root}")
        if final_bundle_dir is not None:
            self._log(f"Final bundle source: {Path(final_bundle_dir).resolve()}")
        reused_problem_path = Path(reuse_problem_file).resolve() if reuse_problem_file is not None else None
        parsed_reused_problem = None
        if reused_problem_path is not None:
            self._log(f"Reusing objects and goal from problem file: {reused_problem_path}")
        if prior_data_confidence <= 0.0:
            self._log(
                "Uniform uncertainty mode: LLM/VLM is used for grouping only; "
                "historical priors and probability calibration are disabled"
            )
        elif prior_data_confidence >= 1.0:
            self._log("Prior data confidence is 1.0; uncertain groups will fully trust historical data")
        else:
            self._log(f"Prior data confidence blends historical and uniform priors: {prior_data_confidence:.3f}")
        if close_domain:
            self._log("Close-domain object generation enabled; objects will be merged from historical grounding")

        self._log("Preparing scene image input")
        self.init_belief_agent.set_initial_state_hint(initial_state_hint)
        with prepare_scene_image(image_path) as prepared_image:
            if prepared_image.is_stitched_multiview:
                self._log(
                    "Scene image prepared: stitched multi-view image from "
                    + ", ".join(path.name for path in prepared_image.source_image_paths)
                )
            else:
                self._log(f"Scene image prepared: {prepared_image.image_path.name}")

            if reused_problem_path is not None:
                self._log("Loading reused objects and goal from existing online problem file")
                parsed_reused_problem = parse_problem(reused_problem_path.read_text(encoding="utf-8"))
                if (
                    parsed_reused_problem.domain_name is not None
                    and parsed_reused_problem.domain_name != domain_analysis.parsed_domain.domain_name
                ):
                    raise ValueError(
                        "Reused problem file domain mismatch: "
                        f"expected {domain_analysis.parsed_domain.domain_name}, "
                        f"got {parsed_reused_problem.domain_name}."
                    )
                object_declarations = [
                    ObjectDeclaration(name=name, type_name=type_name)
                    for name, type_name in sorted(parsed_reused_problem.objects.items())
                ]
                visible_objects = [
                    VisibleObject(
                        name=item.name,
                        type_name=item.type_name,
                        justification="reused from existing problem file",
                    )
                    for item in object_declarations
                ]
                reused_goal_expr = parsed_reused_problem.goal
                self._log(f"Reused {len(object_declarations)} objects from existing problem file")
                object_source = "reused_problem_file"
                current_visible_object_names = None
            elif close_domain:
                if historical_grounding_root is None:
                    raise ValueError(
                        "Close-domain object generation requires a resolved `4_problem_grounding_all` directory. "
                        "Pass `--final-bundle-dir`, place `4_problem_grounding_all` next to the image directory, "
                        "or choose an output path under the learning pipeline root."
                    )
                allowed_object_names = self._load_close_domain_object_allowlist(domain_file)
                if allowed_object_names is None:
                    self._log("No objects.txt next to domain file; using full close-domain object inventory")
                else:
                    self._log(
                        "Applying close-domain object allowlist from objects.txt: "
                        f"{len(allowed_object_names)} objects allowed"
                    )
                self._log("Collecting close-domain object inventory from historical grounding")
                historical_objects = self._infer_closed_domain_objects_from_grounding(
                    domain_analysis=domain_analysis,
                    historical_grounding_root=historical_grounding_root,
                    allowed_object_names=allowed_object_names,
                )
                self._log("Checking the current image for additional task-relevant objects")
                current_visible_objects = self.object_agent.infer_visible_objects(
                    domain_analysis=domain_analysis,
                    image_path=prepared_image.image_path,
                    image_input_note=prepared_image.image_input_note,
                    instruction=instruction,
                    manipulation_records_path=manipulation_records_path,
                    known_object_names=allowed_object_names,
                )
                current_visible_objects, rejected_visible_names = self._restrict_objects_to_allowlist(
                    current_visible_objects,
                    allowed_object_names,
                )
                if rejected_visible_names:
                    self._log(
                        "Discarding current-image objects outside the strict "
                        "objects.txt allowlist: " + ", ".join(rejected_visible_names)
                    )
                covered_names = {item.name for item in historical_objects + current_visible_objects}
                missing_allowed_names = (
                    set(allowed_object_names) - covered_names if allowed_object_names is not None else set()
                )
                if missing_allowed_names:
                    self._log(
                        "Inferring domain types for allowlist objects absent from historical grounding "
                        f"and current visible extraction: {len(missing_allowed_names)} objects"
                    )
                typed_missing_objects = self.object_agent.infer_named_object_types(
                    domain_analysis=domain_analysis,
                    image_path=prepared_image.image_path,
                    image_input_note=prepared_image.image_input_note,
                    instruction=instruction,
                    object_names=missing_allowed_names,
                    manipulation_records_path=manipulation_records_path,
                )
                visible_objects = self._merge_closed_domain_objects(
                    domain_analysis=domain_analysis,
                    object_groups=[
                        historical_objects,
                        current_visible_objects,
                        typed_missing_objects,
                    ],
                )
                object_declarations = self.object_agent.to_object_declarations(visible_objects)
                current_visible_object_names = {item.name for item in current_visible_objects}
                reused_goal_expr = None
                self._log(
                    "Close-domain object inventory ready: "
                    f"{len(object_declarations)} merged objects "
                    f"(historical={len(historical_objects)}, "
                    f"visible={len(current_visible_objects)}, "
                    f"typed_missing={len(typed_missing_objects)})"
                )
                object_source = "close_domain_historical_grounding"
            else:
                self._log("Inferring task-relevant objects")
                allowed_object_names = self._load_close_domain_object_allowlist(domain_file)
                if allowed_object_names is not None:
                    self._log(
                        "Applying strict object allowlist from objects.txt: "
                        f"{len(allowed_object_names)} objects allowed"
                    )
                inferred_visible_objects = self.object_agent.infer_visible_objects(
                    domain_analysis=domain_analysis,
                    image_path=prepared_image.image_path,
                    image_input_note=prepared_image.image_input_note,
                    instruction=instruction,
                    manipulation_records_path=manipulation_records_path,
                    known_object_names=allowed_object_names,
                )
                inferred_visible_objects, rejected_visible_names = self._restrict_objects_to_allowlist(
                    inferred_visible_objects,
                    allowed_object_names,
                )
                if rejected_visible_names:
                    self._log(
                        "Discarding current-image objects outside the strict "
                        "objects.txt allowlist: " + ", ".join(rejected_visible_names)
                    )
                inferred_names = {item.name for item in inferred_visible_objects}
                missing_allowed_names = (
                    set(allowed_object_names) - inferred_names if allowed_object_names is not None else set()
                )
                typed_missing_objects = self.object_agent.infer_named_object_types(
                    domain_analysis=domain_analysis,
                    image_path=prepared_image.image_path,
                    image_input_note=prepared_image.image_input_note,
                    instruction=instruction,
                    object_names=missing_allowed_names,
                    manipulation_records_path=manipulation_records_path,
                )
                visible_objects = self._merge_closed_domain_objects(
                    domain_analysis=domain_analysis,
                    object_groups=[inferred_visible_objects, typed_missing_objects],
                )
                object_declarations = self.object_agent.to_object_declarations(visible_objects)
                current_visible_object_names = {item.name for item in inferred_visible_objects}
                reused_goal_expr = None
                self._log(f"Object inference complete: {len(object_declarations)} objects")
                object_source = "llm_visible_object_inference"

            self.init_belief_agent.set_current_visible_object_names(current_visible_object_names)
            self._log("Inferring initial state and initial belief")
            if skip_init_observation:
                self._log(
                    "Skipping observation-aware init observation and belief update; "
                    "using deterministic init-belief inference directly"
                )
                init_state, init_belief, predicate_judgments = (
                    self.init_belief_agent.infer_deterministic_init_and_belief(
                        domain_analysis=domain_analysis,
                        image_path=prepared_image.image_path,
                        image_input_note=prepared_image.image_input_note,
                        instruction=instruction,
                        objects=object_declarations,
                        manipulation_records_path=manipulation_records_path,
                        historical_grounding_root=historical_grounding_root,
                        problem_name=resolved_problem_name,
                        max_workers=max_workers,
                    )
                )
            else:
                init_state, init_belief, predicate_judgments = self.init_belief_agent.infer_init_and_belief(
                    domain_analysis=domain_analysis,
                    image_path=prepared_image.image_path,
                    image_input_note=prepared_image.image_input_note,
                    instruction=instruction,
                    objects=object_declarations,
                    manipulation_records_path=manipulation_records_path,
                    historical_grounding_root=historical_grounding_root,
                    problem_name=resolved_problem_name,
                    max_workers=max_workers,
                    prior_data_confidence=prior_data_confidence,
                )
            grounded_predicates = build_grounded_predicates_for_objects(
                domain_analysis.parsed_domain,
                object_declarations,
                problem_name=resolved_problem_name,
            )
            force_false_predicates = [
                predicate
                for predicate in grounded_predicates
                if self._is_last_action_helper(predicate) or self._is_gripper_holding_predicate(predicate)
            ]
            force_true_predicates = [
                predicate for predicate in grounded_predicates if self._is_gripper_empty_predicate(predicate)
            ]
            if force_false_predicates or force_true_predicates:
                self._log(
                    "Applying rule-based initial predicate defaults: "
                    f"{len(force_false_predicates)} forced false, {len(force_true_predicates)} forced true"
                )
                force_false_set = set(force_false_predicates)
                force_true_set = set(force_true_predicates)
                init_state = {
                    predicate: value
                    for predicate, value in init_state.items()
                    if predicate not in force_false_set and predicate not in force_true_set
                }
                init_state.update({predicate: False for predicate in force_false_predicates})
                init_state.update({predicate: True for predicate in force_true_predicates})
                init_belief = self._force_predicates_false_in_belief(init_belief, force_false_predicates)
                init_belief = self._force_predicates_true_in_belief(init_belief, force_true_predicates)
            belief_diagnostics = self.init_belief_agent.last_inference_diagnostics
            mode = belief_diagnostics.get("mode", "unknown")
            grounded_count = (
                len(init_belief.known_true) + len(init_belief.known_false) + len(init_belief.all_uncertain_predicates())
            )
            self._log(
                "Initial belief inference complete: "
                f"mode={mode}, grounded_predicates={grounded_count}, "
                f"deterministic={belief_diagnostics.get('deterministic_predicate_count', len(predicate_judgments))}, "
                f"uncertain={belief_diagnostics.get('uncertain_predicate_count', 0)}"
            )

            if reused_problem_path is not None:
                self._log("Reusing goal expression from existing problem file")
                goal_expr = reused_goal_expr
            else:
                self._log("Inferring symbolic goal from initial belief")
                goal_expr = self.goal_agent.infer_goal_expr(
                    domain_analysis=domain_analysis,
                    instruction=instruction,
                    objects=object_declarations,
                    max_workers=max_workers,
                )
                self._log("Goal inference complete")
            spec = OnlinePlanningProblemSpec(
                problem_name=(
                    problem_name
                    or (parsed_reused_problem.problem_name if parsed_reused_problem is not None else None)
                    or resolved_problem_name
                ),
                domain_name=domain_analysis.parsed_domain.domain_name,
                objects=object_declarations,
                init_state=init_state,
                init_belief=init_belief,
                goal_expr=goal_expr,
                has_observation_module=domain_analysis.has_observation_module,
            )
            self._log("Rendering online problem PDDL")
            problem_pddl = render_online_problem_pddl(spec)
            if output_path is not None:
                self._log(f"Writing problem file to {output_path}")
                _write_problem_output(output_path, problem_pddl)

            return OnlinePlanningProblemResult(
                spec=spec,
                problem_pddl=problem_pddl,
                parsed_domain=domain_analysis.parsed_domain,
                visible_objects=visible_objects,
                predicate_judgments=predicate_judgments,
                diagnostics={
                    "has_observation_module": domain_analysis.has_observation_module,
                    "belief_inference": belief_diagnostics,
                    "object_count": len(object_declarations),
                    "grounded_predicate_count": grounded_count,
                    "stitched_multiview": prepared_image.is_stitched_multiview,
                    "source_image_paths": [str(path) for path in prepared_image.source_image_paths],
                    "image_input_note": prepared_image.image_input_note,
                    "final_bundle_dir": str(Path(final_bundle_dir).resolve()) if final_bundle_dir is not None else None,
                    "reused_problem_file": str(reused_problem_path) if reused_problem_path is not None else None,
                    "manipulation_records_path": str(manipulation_records_path)
                    if manipulation_records_path is not None
                    else None,
                    "historical_grounding_root": str(historical_grounding_root)
                    if historical_grounding_root is not None
                    else None,
                    "prior_data_confidence": prior_data_confidence,
                    "close_domain": close_domain,
                    "skip_init_observation": skip_init_observation,
                    "inference_strategy": self.init_belief_agent.inference_strategy,
                    "object_source": object_source,
                },
                reused_problem_file=str(reused_problem_path) if reused_problem_path is not None else None,
            )

    def build_problem_from_files(
        self,
        *,
        domain_file: str | Path,
        image_path: str | Path,
        instruction: str,
        initial_state_hint: str | None = None,
        final_bundle_dir: str | Path | None = None,
        reuse_problem_file: str | Path | None = None,
        problem_name: str | None = None,
        output_path: str | Path | None = None,
        max_workers: int = 8,
        prior_data_confidence: float = 0.0,
        close_domain: bool = False,
        skip_init_observation: bool = False,
    ) -> OnlinePlanningProblemResult:
        return self.build_problem_from_text(
            domain_text=Path(domain_file).read_text(encoding="utf-8"),
            domain_file=domain_file,
            image_path=image_path,
            instruction=instruction,
            initial_state_hint=initial_state_hint,
            final_bundle_dir=final_bundle_dir,
            reuse_problem_file=reuse_problem_file,
            problem_name=problem_name,
            output_path=output_path,
            max_workers=max_workers,
            prior_data_confidence=prior_data_confidence,
            close_domain=close_domain,
            skip_init_observation=skip_init_observation,
        )
