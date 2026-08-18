from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from po_pddl.config import DEFAULT_MODEL

from ..core.models.belief_factor import BeliefFactor
from ..core.models.factorized_belief import FactorizedBelief
from ..core.models.predicate import Predicate
from ..core.parser import parse_problem
from ..core.parser.sexpr import loads_sexpr
from ..domain_generation.infrastructure.llm_shared import build_user_content, make_client, safe_chat
from ..domain_generation.infrastructure.response_parsing import extract_json_object
from ..domain_generation.stages.manipulation_domain_learning.grounding_update import load_manipulation_records
from ..domain_generation.stages.problem_grounding.models import ObjectDeclaration
from ..prompts import load_prompt
from .domain_analysis import DomainAnalysisResult, build_grounded_predicates_for_objects
from .model_config import resolve_online_llm_config
from .models import PredicateTruthJudgment


def _load_prompt(name: str) -> str:
    return load_prompt(name)


def _parse_ground_predicate(text: str) -> Predicate:
    roots = loads_sexpr(text)
    if len(roots) != 1 or not isinstance(roots[0], list) or not roots[0]:
        raise ValueError(f"Expected exactly one grounded predicate, got {text!r}")
    root = roots[0]
    head = root[0]
    if not isinstance(head, str):
        raise ValueError(f"Predicate head must be a symbol, got {text!r}")
    arguments: list[str] = []
    for item in root[1:]:
        if not isinstance(item, str):
            raise ValueError(f"Predicate arguments must be symbols, got {text!r}")
        arguments.append(item)
    return Predicate(head, arguments)


def _normalize_probability(value: object, *, default: float) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        probability = default
    if probability < 0.0:
        return 0.0
    if probability > 1.0:
        return 1.0
    return probability


def _validate_prior_data_confidence(value: float) -> float:
    confidence = float(value)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"prior_data_confidence must be within [0.0, 1.0], got {confidence!r}.")
    return confidence


def _parse_typed_symbol_sequence(tokens: list[object], *, default_type: str) -> list[tuple[str, str]]:
    raw_tokens: list[str] = []
    for token in tokens:
        if not isinstance(token, str):
            raise ValueError(f"Expected flat typed symbol tokens, got {tokens!r}")
        raw_tokens.append(token)
    declarations: list[tuple[str, str]] = []
    pending_names: list[str] = []
    index = 0
    while index < len(raw_tokens):
        token = raw_tokens[index]
        if token == "-":
            if not pending_names or index + 1 >= len(raw_tokens):
                raise ValueError(f"Malformed typed symbol sequence: {tokens!r}")
            type_name = raw_tokens[index + 1]
            for name in pending_names:
                declarations.append((name, type_name))
            pending_names.clear()
            index += 2
            continue
        pending_names.append(token)
        index += 1
    for name in pending_names:
        declarations.append((name, default_type))
    return declarations


class InitialBeliefGenerator:
    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        verbose: bool = False,
        deterministic_collapse_threshold: float = 0.95,
        inference_strategy: Literal["batch", "parallel"] = "batch",
        inference_batch_size: int = 20,
        config_path: str | None = None,
        config_name: str | None = None,
    ) -> None:
        resolved = resolve_online_llm_config(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            config_path=config_path,
            config_name=config_name,
            default_model=DEFAULT_MODEL,
            default_temperature=0.0,
        )
        self.model = resolved.model
        self.base_url = resolved.base_url
        self.api_key = resolved.api_key
        self.temperature = resolved.temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.deterministic_collapse_threshold = deterministic_collapse_threshold
        if inference_strategy not in {"batch", "parallel"}:
            raise ValueError("inference_strategy must be 'batch' or 'parallel'")
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        self.inference_strategy = inference_strategy
        self.inference_batch_size = inference_batch_size
        self._deterministic_batch_prompt = _load_prompt("deterministic_predicates.md")
        self._location_predicate_prompt = _load_prompt("location_predicates.md")
        self._object_location_visibility_prompt = _load_prompt("object_location_visibility.md")
        self._object_location_visibility_batch_prompt = _load_prompt("object_location_visibilities.md")
        self._uncertain_grouping_prompt = _load_prompt("uncertain_groups.md")
        self._observable_judgment_prompt = _load_prompt("observable.md")
        self.last_inference_diagnostics: dict[str, Any] = {}
        self._historical_init_cache: dict[str, list[tuple[set[str], dict[Predicate, bool]]]] = {}
        self._location_predicate_cache: dict[str, set[str]] = {}
        self._visually_resolved_location_objects: set[str] = set()
        self._current_visible_object_names: set[str] | None = None
        self._initial_state_hint: str | None = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[online_init_belief] {message}", flush=True)

    def set_current_visible_object_names(
        self,
        object_names: set[str] | None,
    ) -> None:
        self._current_visible_object_names = set(object_names) if object_names is not None else None

    def set_initial_state_hint(self, hint: str | None) -> None:
        cleaned = str(hint or "").strip()
        self._initial_state_hint = cleaned or None

    def infer_init_and_belief(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None = None,
        instruction: str,
        objects: list[ObjectDeclaration],
        manipulation_records_path: str | Path | None = None,
        historical_grounding_root: str | Path | None = None,
        problem_name: str | None = None,
        max_workers: int = 8,
        prior_data_confidence: float = 0.0,
    ) -> tuple[dict[Predicate, bool], FactorizedBelief, list[PredicateTruthJudgment]]:
        prior_data_confidence = _validate_prior_data_confidence(prior_data_confidence)
        self._visually_resolved_location_objects = set()
        if domain_analysis.has_observation_module:
            self._log("Observation module detected; using observation-aware initial belief flow")
            return self._infer_observation_aware_init_and_belief(
                domain_analysis=domain_analysis,
                image_path=image_path,
                image_input_note=image_input_note,
                instruction=instruction,
                objects=objects,
                manipulation_records_path=manipulation_records_path,
                historical_grounding_root=historical_grounding_root,
                problem_name=problem_name,
                max_workers=max_workers,
                prior_data_confidence=prior_data_confidence,
            )
        self._log("No observation module detected; using deterministic initial belief flow")
        return self.infer_deterministic_init_and_belief(
            domain_analysis=domain_analysis,
            image_path=image_path,
            image_input_note=image_input_note,
            instruction=instruction,
            objects=objects,
            manipulation_records_path=manipulation_records_path,
            problem_name=problem_name,
            max_workers=max_workers,
        )

    def infer_deterministic_init_and_belief(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None = None,
        instruction: str,
        objects: list[ObjectDeclaration],
        manipulation_records_path: str | Path | None = None,
        historical_grounding_root: str | Path | None = None,
        problem_name: str | None = None,
        max_workers: int = 8,
    ) -> tuple[dict[Predicate, bool], FactorizedBelief, list[PredicateTruthJudgment]]:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive.")

        grounded_predicates = sorted(
            build_grounded_predicates_for_objects(
                domain_analysis.parsed_domain,
                objects,
                problem_name=problem_name,
            ),
            key=lambda predicate: predicate.to_pddl_str(),
        )
        self._log(
            f"Deterministic belief path: judging {len(grounded_predicates)} grounded predicates "
            f"using {self.inference_strategy} inference"
        )
        rule_based_default_predicates = [
            predicate for predicate in grounded_predicates if self._predicate_has_rule_based_initial_default(predicate)
        ]
        llm_grounded_predicates = [
            predicate
            for predicate in grounded_predicates
            if not self._predicate_has_rule_based_initial_default(predicate)
        ]
        prior_fixed_judgments, llm_grounded_predicates = self._build_historical_prior_judgments(
            domain_analysis=domain_analysis,
            predicates=llm_grounded_predicates,
            historical_grounding_root=historical_grounding_root,
        )
        judgments: list[PredicateTruthJudgment] = list(prior_fixed_judgments)
        judgments.extend(
            self._judge_deterministic_predicates(
                domain_analysis=domain_analysis,
                image_path=image_path,
                image_input_note=image_input_note,
                instruction=instruction,
                objects=objects,
                predicates=llm_grounded_predicates,
                fixed_judgments=prior_fixed_judgments,
                max_workers=max_workers,
            )
        )
        self._log(
            f"Deterministic truth judgments complete via {self.inference_strategy} inference "
            f"({len(llm_grounded_predicates)} model-classified predicates)"
        )

        judgments.extend(self._build_rule_based_default_judgments(rule_based_default_predicates))
        judgments = self._repair_directional_reference_judgments(judgments)
        judgments = sorted(judgments, key=lambda item: item.predicate.to_pddl_str())
        init_state = {item.predicate: item.truth_value for item in judgments}
        belief = FactorizedBelief(
            known_true=[item.predicate for item in judgments if item.truth_value],
            known_false=[item.predicate for item in judgments if not item.truth_value],
            factors=[],
        )
        belief.validate()
        self.last_inference_diagnostics = {
            "mode": "fully_observable_deterministic",
            "inference_strategy": self.inference_strategy,
            "grounded_predicate_count": len(grounded_predicates),
            "deterministic_predicate_count": len(grounded_predicates),
            "llm_deterministic_predicate_count": len(llm_grounded_predicates),
            "rule_based_default_predicate_count": len(rule_based_default_predicates),
            "uncertain_predicate_count": 0,
            "group_count": 0,
        }
        return init_state, belief, judgments

    def _judge_deterministic_predicates(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None,
        instruction: str,
        objects: list[ObjectDeclaration],
        predicates: list[Predicate],
        fixed_judgments: list[PredicateTruthJudgment] | None,
        max_workers: int,
    ) -> list[PredicateTruthJudgment]:
        return self._judge_deterministic_predicates_in_chunks(
            domain_analysis=domain_analysis,
            image_path=image_path,
            image_input_note=image_input_note,
            instruction=instruction,
            objects=objects,
            predicates=predicates,
            fixed_judgments=fixed_judgments,
            max_workers=max_workers,
        )

    def _judge_deterministic_predicates_batch(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None,
        instruction: str,
        objects: list[ObjectDeclaration],
        predicates: list[Predicate],
        fixed_judgments: list[PredicateTruthJudgment] | None = None,
    ) -> list[PredicateTruthJudgment]:
        if not predicates:
            return []
        ordered_predicates = sorted(predicates, key=lambda item: item.to_pddl_str())
        available = {predicate.to_pddl_str(): predicate for predicate in ordered_predicates}
        payload: dict[str, Any] = {
            "domain_summary": domain_analysis.render_summary(),
            "instruction": instruction.strip(),
            "initial_state_hint": self._initial_state_hint,
            "image_input_note": image_input_note
            or "The provided image is a single-view snapshot of the initial scene.",
            "objects": [{"name": item.name, "type_name": item.type_name} for item in objects],
            "fixed_predicate_judgments": [
                {
                    "predicate": item.predicate.to_pddl_str(),
                    "truth_value": "true" if item.truth_value else "false",
                    "justification": item.justification,
                }
                for item in (fixed_judgments or [])
            ],
            "candidate_predicates": list(available),
        }
        client = make_client(api_key=self.api_key, base_url=self.base_url)
        last_error: str | None = None
        for attempt in range(2):
            if attempt:
                payload["correction"] = (
                    "The previous response was structurally invalid. Return every candidate "
                    "predicate exactly once, without additions or omissions."
                )
            reply = safe_chat(
                client,
                self._deterministic_batch_prompt,
                build_user_content(
                    text=json.dumps(payload, ensure_ascii=False, indent=2),
                    image_path=str(image_path),
                ),
                model=self.model,
                temperature=self.temperature,
                max_tokens=max(self.max_tokens, 4096),
                verbose=self.verbose,
            )
            data = extract_json_object(reply)
            raw_judgments = data.get("predicate_judgments", [])
            if not isinstance(raw_judgments, list):
                last_error = "response did not contain a predicate_judgments list"
                continue
            parsed: dict[str, PredicateTruthJudgment] = {}
            invalid_predicates: list[str] = []
            for row in raw_judgments:
                if not isinstance(row, dict):
                    invalid_predicates.append("<non-object>")
                    continue
                predicate_text = str(row.get("predicate", "")).strip()
                if predicate_text not in available or predicate_text in parsed:
                    invalid_predicates.append(predicate_text)
                    continue
                truth_value = str(row.get("truth_value", "")).strip().lower()
                if truth_value not in {"true", "false"}:
                    invalid_predicates.append(predicate_text)
                    continue
                parsed[predicate_text] = PredicateTruthJudgment(
                    predicate=available[predicate_text],
                    truth_value=(truth_value == "true"),
                    justification=str(row.get("justification", "")).strip() or None,
                )
            missing = [text for text in available if text not in parsed]
            if invalid_predicates or missing or len(raw_judgments) != len(available):
                last_error = (
                    f"invalid={invalid_predicates or 'none'}, missing={missing or 'none'}, "
                    f"expected_count={len(available)}, actual_count={len(raw_judgments)}"
                )
                continue
            return [parsed[predicate.to_pddl_str()] for predicate in ordered_predicates]
        raise ValueError(f"Invalid deterministic batch judgment after retry: {last_error}.")

    def _judge_deterministic_predicates_in_chunks(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None,
        instruction: str,
        objects: list[ObjectDeclaration],
        predicates: list[Predicate],
        fixed_judgments: list[PredicateTruthJudgment] | None,
        max_workers: int,
    ) -> list[PredicateTruthJudgment]:
        if not predicates:
            return []
        ordered_predicates = sorted(predicates, key=lambda item: item.to_pddl_str())

        chunks = [
            ordered_predicates[index : index + self.inference_batch_size]
            for index in range(0, len(ordered_predicates), self.inference_batch_size)
        ]

        def _judge(chunk: list[Predicate]) -> list[PredicateTruthJudgment]:
            return self._judge_deterministic_predicates_batch(
                domain_analysis=domain_analysis,
                image_path=image_path,
                image_input_note=image_input_note,
                instruction=instruction,
                objects=objects,
                predicates=chunk,
                fixed_judgments=fixed_judgments,
            )

        self._log(
            f"Judging {len(ordered_predicates)} deterministic predicates in {len(chunks)} "
            f"chunk(s) of at most {self.inference_batch_size} using {self.inference_strategy} scheduling"
        )
        if self.inference_strategy == "batch" or len(chunks) == 1:
            chunk_results = [_judge(chunk) for chunk in chunks]
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as executor:
                chunk_results = list(executor.map(_judge, chunks))
        return [judgment for chunk in chunk_results for judgment in chunk]

    def _infer_observation_aware_init_and_belief(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None,
        instruction: str,
        objects: list[ObjectDeclaration],
        manipulation_records_path: str | Path | None,
        historical_grounding_root: str | Path | None,
        problem_name: str | None,
        max_workers: int,
        prior_data_confidence: float,
    ) -> tuple[dict[Predicate, bool], FactorizedBelief, list[PredicateTruthJudgment]]:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive.")

        grounded_predicates = sorted(
            build_grounded_predicates_for_objects(
                domain_analysis.parsed_domain,
                objects,
                problem_name=problem_name,
            ),
            key=lambda predicate: predicate.to_pddl_str(),
        )
        self._log(
            f"Observation-aware belief path: classifying {len(grounded_predicates)} grounded predicates "
            f"using {self.inference_strategy} inference"
        )
        manipulation_summary = self._load_manipulation_summary(manipulation_records_path)
        action_constant_names = {name for name in domain_analysis.parsed_domain.constants if name.startswith("action_")}
        observation_reliability = self._extract_init_observation_rule_summary(domain_analysis)
        screening_results = self._classify_predicates_from_observation_module(
            domain_analysis=domain_analysis,
            grounded_predicates=grounded_predicates,
            action_constant_names=action_constant_names,
        )
        self._log("Rule-based predicate classification from observation module complete")
        location_predicate_names = self._select_location_predicate_names(domain_analysis=domain_analysis)
        screening_results = self._promote_visually_resolved_location_predicates(
            domain_analysis=domain_analysis,
            image_path=image_path,
            image_input_note=image_input_note,
            instruction=instruction,
            objects=objects,
            manipulation_summary=manipulation_summary,
            screening_results=screening_results,
            location_predicate_names=location_predicate_names,
            max_workers=max_workers,
        )
        forced_false_location_predicates = {
            item["predicate"] for item in screening_results if item.get("forced_truth_value") is False
        }

        deterministic_predicates = [item["predicate"] for item in screening_results if bool(item["is_deterministic"])]
        uncertain_predicates = [item["predicate"] for item in screening_results if not bool(item["is_deterministic"])]
        invalid_uncertain_predicates = [
            item["predicate"]
            for item in screening_results
            if not bool(item["is_deterministic"]) and not bool(item.get("observation_uncertainty_eligible"))
        ]
        if invalid_uncertain_predicates:
            rendered = ", ".join(predicate.to_pddl_str() for predicate in invalid_uncertain_predicates)
            raise ValueError(
                f"Only predicates covered by the observation module may enter the uncertain branch; found: {rendered}"
            )
        rule_based_default_predicates = [
            predicate
            for predicate in deterministic_predicates
            if self._predicate_has_rule_based_initial_default(predicate)
        ]
        location_deterministic_predicates = [
            predicate
            for predicate in deterministic_predicates
            if predicate.name in location_predicate_names
            and predicate not in forced_false_location_predicates
            and not self._predicate_has_rule_based_initial_default(predicate)
        ]
        non_location_deterministic_predicates = [
            predicate
            for predicate in deterministic_predicates
            if predicate.name not in location_predicate_names
            and not self._predicate_has_rule_based_initial_default(predicate)
        ]
        if prior_data_confidence <= 1e-12:
            prior_fixed_judgments = []
        else:
            (
                prior_fixed_judgments,
                non_location_deterministic_predicates,
            ) = self._build_historical_prior_judgments(
                domain_analysis=domain_analysis,
                predicates=non_location_deterministic_predicates,
                historical_grounding_root=historical_grounding_root,
            )
        prior_fixed_judgments.extend(
            PredicateTruthJudgment(
                predicate=predicate,
                truth_value=False,
                justification=(
                    "The target object is not visible, so an exposed spatial relation cannot hold in the current scene."
                ),
            )
            for predicate in sorted(
                forced_false_location_predicates,
                key=lambda item: item.to_pddl_str(),
            )
        )
        llm_deterministic_predicates = location_deterministic_predicates + non_location_deterministic_predicates
        self._log(
            f"Screening summary: deterministic={len(deterministic_predicates)}, uncertain={len(uncertain_predicates)}"
        )

        deterministic_judgments: list[PredicateTruthJudgment] = list(prior_fixed_judgments)
        deterministic_judgments.extend(
            self._judge_deterministic_predicates(
                domain_analysis=domain_analysis,
                image_path=image_path,
                image_input_note=image_input_note,
                instruction=instruction,
                objects=objects,
                predicates=llm_deterministic_predicates,
                fixed_judgments=prior_fixed_judgments,
                max_workers=max_workers,
            )
        )
        deterministic_judgments.extend(self._build_rule_based_default_judgments(rule_based_default_predicates))
        if deterministic_predicates:
            self._log(
                f"Deterministic truth judgments for screened predicates complete via {self.inference_strategy} "
                f"({len(llm_deterministic_predicates)} model-classified predicates)"
            )

        deterministic_judgments = sorted(
            deterministic_judgments,
            key=lambda item: item.predicate.to_pddl_str(),
        )
        deterministic_judgments = self._repair_directional_reference_judgments(deterministic_judgments)
        init_state = {item.predicate: item.truth_value for item in deterministic_judgments}
        deterministic_truth_by_predicate = dict(init_state)
        self._apply_rule_based_initial_defaults_to_truth_map(
            truth_map=deterministic_truth_by_predicate,
            grounded_predicates=grounded_predicates,
        )

        known_true: list[Predicate] = []
        known_false: list[Predicate] = []
        factors: list[BeliefFactor] = []
        group_summaries: list[dict[str, Any]] = []
        confidence_calibration_summaries: list[dict[str, Any]] = []
        directionally_impossible_uncertain = [
            predicate
            for predicate in uncertain_predicates
            if self._is_directionally_inconsistent(
                predicate,
                deterministic_truth_by_predicate,
            )
        ]
        if directionally_impossible_uncertain:
            impossible_set = set(directionally_impossible_uncertain)
            uncertain_predicates = [predicate for predicate in uncertain_predicates if predicate not in impossible_set]
            known_false.extend(directionally_impossible_uncertain)
            self._log(
                "Forced directionally inconsistent uncertain location predicates false: "
                f"{len(directionally_impossible_uncertain)}"
            )

        rule_based_default_judgments = [
            item for item in deterministic_judgments if self._predicate_has_rule_based_initial_default(item.predicate)
        ]
        llm_deterministic_judgments = [
            item
            for item in deterministic_judgments
            if not self._predicate_has_rule_based_initial_default(item.predicate)
        ]

        deterministic_base_factors = [
            (
                item.predicate,
                BeliefFactor(
                    name=f"{item.predicate.name}_deterministic",
                    scope=[item.predicate],
                    cases=[(1.0, [item.predicate]) if item.truth_value else (1.0, [])],
                ),
            )
            for item in llm_deterministic_judgments
            if item.predicate not in forced_false_location_predicates
        ]
        if deterministic_base_factors:

            def _calibrate_deterministic(
                item: tuple[Predicate, BeliefFactor],
            ) -> tuple[BeliefFactor, dict[str, Any] | None]:
                predicate, base_factor = item
                return self._bayesian_update_singleton_distribution_from_observation_rules(
                    domain_analysis=domain_analysis,
                    image_path=image_path,
                    image_input_note=image_input_note,
                    instruction=instruction,
                    objects=objects,
                    manipulation_summary=manipulation_summary,
                    predicate=predicate,
                    base_factor=base_factor,
                    observation_reliability=observation_reliability,
                    deterministic_truth_by_predicate=deterministic_truth_by_predicate,
                )

            with ThreadPoolExecutor(max_workers=min(max_workers, len(deterministic_base_factors))) as executor:
                calibrated_deterministic = list(executor.map(_calibrate_deterministic, deterministic_base_factors))
            for calibrated_factor, calibration_summary in calibrated_deterministic:
                if calibration_summary is not None:
                    confidence_calibration_summaries.append(calibration_summary)
                self._store_singleton_factor_or_known(
                    calibrated_factor,
                    known_true=known_true,
                    known_false=known_false,
                    factors=factors,
                )
        known_false.extend(
            sorted(
                forced_false_location_predicates,
                key=lambda item: item.to_pddl_str(),
            )
        )

        if uncertain_predicates:
            self._log("Grouping uncertain predicates")
            validated_groups = self._group_uncertain_predicates(
                domain_analysis=domain_analysis,
                image_path=image_path,
                image_input_note=image_input_note,
                instruction=instruction,
                objects=objects,
                manipulation_summary=manipulation_summary,
                uncertain_predicates=uncertain_predicates,
            )
            self._log(f"Uncertain grouping complete: {len(validated_groups)} groups")

            def _estimate(group: dict[str, Any]) -> tuple[BeliefFactor, dict[str, Any]]:
                factor = self._estimate_uncertain_group_distribution(
                    domain_analysis=domain_analysis,
                    group=group,
                    historical_grounding_root=historical_grounding_root,
                    prior_data_confidence=prior_data_confidence,
                )
                return factor, {
                    "name": group["name"],
                    "group_kind": group["group_kind"],
                    "predicate_count": len(group["predicates"]),
                    "case_count": len(factor.cases),
                    "prior_source": factor.name.rsplit(":", 1)[-1] if ":" in factor.name else "unknown",
                }

            with ThreadPoolExecutor(max_workers=min(max_workers, len(validated_groups))) as executor:
                estimated = list(executor.map(_estimate, validated_groups))
            singleton_estimated: list[tuple[Predicate, BeliefFactor, dict[str, Any]]] = []
            grouped_estimated: list[tuple[dict[str, Any], BeliefFactor, dict[str, Any]]] = []
            for factor, summary in estimated:
                group_summaries.append(summary)
                if len(factor.scope) == 1:
                    singleton_estimated.append((factor.scope[0], factor, summary))
                else:
                    matching_group = next(
                        (
                            group
                            for group in validated_groups
                            if str(group.get("name", "")).strip() == str(summary.get("name", "")).strip()
                        ),
                        None,
                    )
                    if matching_group is None:
                        matching_group = {
                            "name": summary.get("name"),
                            "group_kind": "correlated",
                            "predicates": list(factor.scope),
                        }
                    grouped_estimated.append((matching_group, factor, summary))

            calibrated_factors: list[BeliefFactor] = []
            if prior_data_confidence <= 1e-12:
                calibrated_factors = [factor for factor, _summary in estimated]
                self._log(
                    "Uniform uncertainty mode: skipped historical priors and observation-based probability reweighting"
                )
            elif singleton_estimated:

                def _calibrate_estimated(
                    item: tuple[Predicate, BeliefFactor, dict[str, Any]],
                ) -> tuple[BeliefFactor, dict[str, Any] | None]:
                    predicate, factor, _summary = item
                    return self._bayesian_update_singleton_distribution_from_observation_rules(
                        domain_analysis=domain_analysis,
                        image_path=image_path,
                        image_input_note=image_input_note,
                        instruction=instruction,
                        objects=objects,
                        manipulation_summary=manipulation_summary,
                        predicate=predicate,
                        base_factor=factor,
                        observation_reliability=observation_reliability,
                        deterministic_truth_by_predicate=deterministic_truth_by_predicate,
                    )

                with ThreadPoolExecutor(max_workers=min(max_workers, len(singleton_estimated))) as executor:
                    calibrated_singletons = list(executor.map(_calibrate_estimated, singleton_estimated))
                for calibrated_factor, calibration_summary in calibrated_singletons:
                    if calibration_summary is not None:
                        confidence_calibration_summaries.append(calibration_summary)
                    calibrated_factors.append(calibrated_factor)

            if grouped_estimated and prior_data_confidence > 1e-12:

                def _calibrate_group(
                    item: tuple[dict[str, Any], BeliefFactor, dict[str, Any]],
                ) -> tuple[BeliefFactor, dict[str, Any] | None]:
                    group, factor, _summary = item
                    return self._bayesian_update_group_distribution_from_observation_rules(
                        domain_analysis=domain_analysis,
                        image_path=image_path,
                        image_input_note=image_input_note,
                        instruction=instruction,
                        objects=objects,
                        manipulation_summary=manipulation_summary,
                        group=group,
                        base_factor=factor,
                        observation_reliability=observation_reliability,
                        deterministic_truth_by_predicate=deterministic_truth_by_predicate,
                    )

                with ThreadPoolExecutor(max_workers=min(max_workers, len(grouped_estimated))) as executor:
                    calibrated_groups = list(executor.map(_calibrate_group, grouped_estimated))
                for calibrated_factor, calibration_summary in calibrated_groups:
                    if calibration_summary is not None:
                        confidence_calibration_summaries.append(calibration_summary)
                    calibrated_factors.append(calibrated_factor)

            for factor in calibrated_factors:
                self._store_singleton_factor_or_known(
                    factor,
                    known_true=known_true,
                    known_false=known_false,
                    factors=factors,
                )
            self._log("Uncertain group probability estimation complete")

        for item in rule_based_default_judgments:
            if item.truth_value:
                known_true.append(item.predicate)
            else:
                known_false.append(item.predicate)

        belief = FactorizedBelief(
            known_true=known_true,
            known_false=known_false,
            factors=factors,
        )
        belief.validate()
        self.last_inference_diagnostics = {
            "mode": "observation_aware_factorized_belief",
            "inference_strategy": self.inference_strategy,
            "grounded_predicate_count": len(grounded_predicates),
            "deterministic_predicate_count": len(deterministic_predicates),
            "llm_deterministic_predicate_count": len(llm_deterministic_predicates),
            "rule_based_default_predicate_count": len(rule_based_default_predicates),
            "uncertain_predicate_count": len(uncertain_predicates),
            "group_count": len(factors),
            "group_summaries": group_summaries,
            "deterministic_collapse_threshold": self.deterministic_collapse_threshold,
            "observation_reliability": observation_reliability,
            "confidence_calibrations": confidence_calibration_summaries,
            "prior_data_confidence": prior_data_confidence,
            "screening_results": [
                {
                    "predicate": item["predicate"].to_pddl_str(),
                    "is_deterministic": bool(item["is_deterministic"]),
                    "justification": item.get("justification"),
                }
                for item in sorted(screening_results, key=lambda row: row["predicate"].to_pddl_str())
            ],
        }
        self._log("Observation-aware belief assembly complete")
        return init_state, belief, deterministic_judgments

    @staticmethod
    def _predicate_direction(predicate_name: str) -> str | None:
        tokens = predicate_name.split("_")
        if "left" in tokens and "right" not in tokens:
            return "left"
        if "right" in tokens and "left" not in tokens:
            return "right"
        return None

    @classmethod
    def _is_directionally_inconsistent(
        cls,
        predicate: Predicate,
        truth_by_predicate: dict[Predicate, bool],
    ) -> bool:
        side = cls._predicate_direction(predicate.name)
        if side is None or len(predicate.params) < 2:
            return False
        reference_name = predicate.params[-1]
        side_predicate = Predicate(f"on_{side}", [reference_name])
        opposite = "right" if side == "left" else "left"
        opposite_predicate = Predicate(f"on_{opposite}", [reference_name])
        return truth_by_predicate.get(side_predicate) is False and truth_by_predicate.get(opposite_predicate) is True

    @classmethod
    def _repair_directional_reference_judgments(
        cls,
        judgments: list[PredicateTruthJudgment],
    ) -> list[PredicateTruthJudgment]:
        if not judgments:
            return judgments
        truth_by_predicate = {item.predicate: item.truth_value for item in judgments}
        judgment_by_predicate = {item.predicate: item for item in judgments}
        repairs: dict[Predicate, bool] = {}
        for item in judgments:
            predicate = item.predicate
            if not item.truth_value or not cls._is_directionally_inconsistent(
                predicate,
                truth_by_predicate,
            ):
                continue
            side = cls._predicate_direction(predicate.name)
            if side is None:
                continue
            opposite = "right" if side == "left" else "left"
            swapped_name = "_".join(opposite if token == side else token for token in predicate.name.split("_"))
            same_reference_replacement = Predicate(
                swapped_name,
                list(predicate.params),
            )
            if same_reference_replacement in judgment_by_predicate:
                repairs[predicate] = False
                repairs[same_reference_replacement] = True
                continue
            replacements = [
                candidate
                for candidate in judgment_by_predicate
                if candidate.name == predicate.name
                and candidate.params[:-1] == predicate.params[:-1]
                and truth_by_predicate.get(Predicate(f"on_{side}", [candidate.params[-1]])) is True
            ]
            if len(replacements) != 1:
                continue
            repairs[predicate] = False
            repairs[replacements[0]] = True
        if not repairs:
            return judgments
        repaired: list[PredicateTruthJudgment] = []
        for item in judgments:
            if item.predicate not in repairs:
                repaired.append(item)
                continue
            repaired.append(
                PredicateTruthJudgment(
                    predicate=item.predicate,
                    truth_value=repairs[item.predicate],
                    justification=(
                        (item.justification + " " if item.justification else "")
                        + "[directional reference consistency repair applied]"
                    ).strip(),
                )
            )
        return repaired

    @staticmethod
    def _predicate_uses_action_constant(
        predicate: Predicate,
        action_constant_names: set[str],
    ) -> bool:
        return any(argument in action_constant_names for argument in predicate.params)

    @staticmethod
    def _is_last_action_predicate(predicate_name: str) -> bool:
        return predicate_name.startswith("last_action_")

    @staticmethod
    def _is_gripper_empty_predicate(predicate_name: str) -> bool:
        return predicate_name == "gripper_empty"

    @staticmethod
    def _is_gripper_holding_predicate(predicate_name: str) -> bool:
        return predicate_name == "gripper_holding"

    def _predicate_has_rule_based_initial_default(self, predicate: Predicate) -> bool:
        return (
            self._is_last_action_predicate(predicate.name)
            or self._is_gripper_empty_predicate(predicate.name)
            or self._is_gripper_holding_predicate(predicate.name)
        )

    def _build_rule_based_default_judgments(
        self,
        predicates: list[Predicate],
    ) -> list[PredicateTruthJudgment]:
        judgments: list[PredicateTruthJudgment] = []
        for predicate in predicates:
            truth_value = self._is_gripper_empty_predicate(predicate.name)
            justification = (
                "Rule-based default: gripper_empty is initially true."
                if truth_value
                else "Rule-based default: last_action and gripper_holding predicates are initially false."
            )
            judgments.append(
                PredicateTruthJudgment(
                    predicate=predicate,
                    truth_value=truth_value,
                    justification=justification,
                )
            )
        return judgments

    def _apply_rule_based_initial_defaults_to_truth_map(
        self,
        *,
        truth_map: dict[Predicate, bool],
        grounded_predicates: list[Predicate],
    ) -> None:
        for predicate in grounded_predicates:
            if self._is_last_action_predicate(predicate.name) or self._is_gripper_holding_predicate(predicate.name):
                truth_map[predicate] = False
            elif self._is_gripper_empty_predicate(predicate.name):
                truth_map[predicate] = True

    @staticmethod
    def _store_singleton_factor_or_known(
        factor: BeliefFactor,
        *,
        known_true: list[Predicate],
        known_false: list[Predicate],
        factors: list[BeliefFactor],
    ) -> None:
        if len(factor.scope) != 1:
            factors.append(factor)
            return
        predicate = factor.scope[0]
        positive_probability = sum(
            probability for probability, true_predicates in factor.cases if predicate in true_predicates
        )
        if positive_probability >= 1.0 - 1e-9:
            known_true.append(predicate)
            return
        if positive_probability <= 1e-9:
            known_false.append(predicate)
            return
        factors.append(factor)

    def _load_manipulation_summary(
        self,
        manipulation_records_path: str | Path | None,
    ) -> list[dict[str, object]]:
        manipulation_records = (
            load_manipulation_records(manipulation_records_path)
            if manipulation_records_path is not None and Path(manipulation_records_path).exists()
            else []
        )
        return [
            {
                "action": record.canonical_action_name,
                "arguments": list(record.action_arguments),
                "success": record.success,
                "effect_bucket": record.effect_bucket,
            }
            for record in manipulation_records
        ]

    def _classify_predicates_from_observation_module(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        grounded_predicates: list[Predicate],
        action_constant_names: set[str],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for predicate in grounded_predicates:
            if self._predicate_uses_action_constant(predicate, action_constant_names):
                results.append(
                    {
                        "predicate": predicate,
                        "is_deterministic": True,
                        "observation_uncertainty_eligible": False,
                        "justification": "Predicate references an action constant and is forced into the deterministic branch.",
                    }
                )
                continue
            if (
                self._resolve_observable_name_for_predicate(
                    domain_analysis=domain_analysis,
                    predicate_name=predicate.name,
                )
                is not None
            ):
                results.append(
                    {
                        "predicate": predicate,
                        "is_deterministic": False,
                        "observation_uncertainty_eligible": True,
                        "justification": "Predicate has a matching observable in the observation module and is treated as uncertain before image-based calibration.",
                    }
                )
            else:
                results.append(
                    {
                        "predicate": predicate,
                        "is_deterministic": True,
                        "observation_uncertainty_eligible": False,
                        "justification": "Predicate is not covered by the observation module and is forced into the deterministic branch.",
                    }
                )
        return results

    def _select_location_predicate_names(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
    ) -> set[str]:
        cache_key = self._location_predicate_cache_key(domain_analysis)
        cached = self._location_predicate_cache.get(cache_key)
        if cached is not None:
            return set(cached)

        candidate_predicates: list[dict[str, Any]] = []
        for predicate in domain_analysis.parsed_domain.predicates:
            if self._is_last_action_predicate(predicate.name):
                continue
            parameter_types = domain_analysis.parsed_domain.predicate_parameter_types.get(predicate, [])
            candidate_predicates.append(
                {
                    "predicate_name": predicate.name,
                    "parameter_types": [type_name for _param_name, type_name in parameter_types],
                }
            )
        payload = {
            "domain_summary": domain_analysis.render_summary(),
            "candidate_predicates": candidate_predicates,
        }
        client = make_client(api_key=self.api_key, base_url=self.base_url)
        reply = safe_chat(
            client,
            self._location_predicate_prompt,
            json.dumps(payload, ensure_ascii=False, indent=2),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        selected = data.get("location_predicates", [])
        if not isinstance(selected, list):
            raise ValueError("Location predicate selector must return a location_predicates list.")
        available_names = {item["predicate_name"] for item in candidate_predicates}
        validated = {str(item).strip() for item in selected if str(item).strip() in available_names}
        self._location_predicate_cache[cache_key] = set(validated)
        return validated

    def _partition_location_predicates_by_object(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        objects: list[ObjectDeclaration],
        predicates: list[Predicate],
        location_predicate_names: set[str],
    ) -> tuple[dict[str, list[Predicate]], list[Predicate]]:
        object_types = {item.name: item.type_name for item in objects}
        grouped: dict[str, list[Predicate]] = defaultdict(list)
        standalone: list[Predicate] = []
        parameter_types_by_name = {
            predicate.name: [
                type_name
                for _param_name, type_name in domain_analysis.parsed_domain.predicate_parameter_types.get(predicate, [])
            ]
            for predicate in domain_analysis.parsed_domain.predicates
        }

        for predicate in predicates:
            if predicate.name not in location_predicate_names:
                standalone.append(predicate)
                continue
            param_types = parameter_types_by_name.get(predicate.name, [])
            movable_argument_indices: list[int] = []
            for index, argument in enumerate(predicate.params):
                object_type = object_types.get(argument)
                if object_type is None:
                    continue
                declared_type = param_types[index] if index < len(param_types) else object_type
                if self._is_type_subtype(domain_analysis, declared_type, "movable_item") or self._is_type_subtype(
                    domain_analysis, object_type, "movable_item"
                ):
                    movable_argument_indices.append(index)
            if len(movable_argument_indices) != 1:
                if len(predicate.params) == 1 and predicate.params[0] in object_types:
                    grouped[predicate.params[0]].append(predicate)
                    continue
                standalone.append(predicate)
                continue
            target_index = movable_argument_indices[0]
            target_object_name = predicate.params[target_index]
            grouped[target_object_name].append(predicate)
        for predicate_list in grouped.values():
            predicate_list.sort(key=lambda item: item.to_pddl_str())
        return dict(grouped), standalone

    def _promote_visually_resolved_location_predicates(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None,
        instruction: str,
        objects: list[ObjectDeclaration],
        manipulation_summary: list[dict[str, object]],
        screening_results: list[dict[str, Any]],
        location_predicate_names: set[str],
        max_workers: int,
    ) -> list[dict[str, Any]]:
        candidate_location_predicates = [
            item["predicate"]
            for item in screening_results
            if item["predicate"].name in location_predicate_names
            and not self._predicate_has_rule_based_initial_default(item["predicate"])
        ]
        grouped, _standalone = self._partition_location_predicates_by_object(
            domain_analysis=domain_analysis,
            objects=objects,
            predicates=candidate_location_predicates,
            location_predicate_names=location_predicate_names,
        )
        if not grouped:
            return screening_results

        del manipulation_summary
        visibility_results = self._classify_object_location_visibility(
            domain_analysis=domain_analysis,
            image_path=image_path,
            image_input_note=image_input_note,
            instruction=instruction,
            objects=objects,
            grouped_location_predicates=grouped,
            max_workers=max_workers,
        )
        resolved_objects = {object_name for object_name, is_resolved in visibility_results if is_resolved}
        self._visually_resolved_location_objects.update(resolved_objects)
        resolution_by_predicate = {
            predicate: object_name in resolved_objects
            for object_name, predicates in grouped.items()
            for predicate in predicates
        }
        updated: list[dict[str, Any]] = []
        promoted_count = 0
        demoted_count = 0
        forced_false_count = 0
        for item in screening_results:
            if item["predicate"] not in resolution_by_predicate:
                updated.append(item)
                continue
            is_visually_resolved = resolution_by_predicate[item["predicate"]]
            can_hold_while_hidden = self._location_relation_can_hold_while_hidden(item["predicate"].name)
            is_forced_false = not is_visually_resolved and not can_hold_while_hidden
            is_observation_eligible = bool(item.get("observation_uncertainty_eligible"))
            is_deterministic = True if is_visually_resolved or is_forced_false else not is_observation_eligible
            if is_deterministic and not bool(item["is_deterministic"]):
                promoted_count += 1
            elif not is_deterministic and bool(item["is_deterministic"]):
                demoted_count += 1
            if is_forced_false:
                forced_false_count += 1
            updated.append(
                {
                    **item,
                    "is_deterministic": is_deterministic,
                    "forced_truth_value": False if is_forced_false else None,
                    "justification": (
                        "The target object's location alternatives are clearly visible "
                        "in the current image and can be judged jointly."
                        if is_visually_resolved
                        else "The target object is not visible, so this exposed spatial "
                        "relation is deterministically false."
                        if is_forced_false
                        else "The target object is not visible, but this containment relation "
                        "can hold while occluded and remains uncertain because it is "
                        "covered by the observation module."
                        if is_observation_eligible
                        else "The target object is not visible, but this predicate is not "
                        "covered by the observation module and therefore remains in the "
                        "deterministic branch."
                    ),
                }
            )
        self._log(
            "Reconciled location predicates with current-image visibility: "
            f"resolved_objects={len(resolved_objects)}, "
            f"promoted={promoted_count}, demoted={demoted_count}, "
            f"forced_false_exposed={forced_false_count}"
        )
        return updated

    def _classify_object_location_visibility(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None,
        instruction: str,
        objects: list[ObjectDeclaration],
        grouped_location_predicates: dict[str, list[Predicate]],
        max_workers: int,
    ) -> list[tuple[str, bool]]:
        ordered_groups = sorted(grouped_location_predicates.items())
        if not ordered_groups:
            return []

        # Object extraction is the global identity pass. Inventory-only objects
        # may still be hidden in containers, but independent workers must not
        # relabel a visible instance that the global pass assigned elsewhere.
        visible_names = self._current_visible_object_names
        unresolved_inventory_names = {
            object_name
            for object_name, _predicates in ordered_groups
            if visible_names is not None and object_name not in visible_names
        }
        candidate_groups = {
            object_name: predicates
            for object_name, predicates in ordered_groups
            if object_name not in unresolved_inventory_names
        }
        resolved_by_name = {object_name: False for object_name in unresolved_inventory_names}

        if self.inference_strategy == "batch":
            if candidate_groups:
                resolved_by_name.update(
                    self._classify_object_location_visibility_batch(
                        domain_analysis=domain_analysis,
                        image_path=image_path,
                        image_input_note=image_input_note,
                        instruction=instruction,
                        objects=objects,
                        grouped_location_predicates=candidate_groups,
                    )
                )
            return [(object_name, resolved_by_name[object_name]) for object_name, _predicates in ordered_groups]

        def _classify(item: tuple[str, list[Predicate]]) -> tuple[str, bool]:
            object_name, predicates = item
            return (
                object_name,
                self._is_object_location_visually_resolved(
                    domain_analysis=domain_analysis,
                    image_path=image_path,
                    image_input_note=image_input_note,
                    instruction=instruction,
                    objects=objects,
                    target_object_name=object_name,
                    candidate_predicates=predicates,
                ),
            )

        candidate_items = sorted(candidate_groups.items())
        if candidate_items:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(candidate_items))) as executor:
                resolved_by_name.update(executor.map(_classify, candidate_items))
        return [(object_name, resolved_by_name[object_name]) for object_name, _predicates in ordered_groups]

    def _classify_object_location_visibility_batch(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None,
        instruction: str,
        objects: list[ObjectDeclaration],
        grouped_location_predicates: dict[str, list[Predicate]],
    ) -> list[tuple[str, bool]]:
        if not grouped_location_predicates:
            return []
        object_types = {item.name: item.type_name for item in objects}
        ordered_names = sorted(grouped_location_predicates)
        payload: dict[str, Any] = {
            "domain_summary": domain_analysis.render_summary(),
            "instruction": instruction.strip(),
            "initial_state_hint": self._initial_state_hint,
            "image_input_note": image_input_note
            or "The provided image is a single-view snapshot of the initial scene.",
            "objects": [{"name": item.name, "type_name": item.type_name} for item in objects],
            "target_location_groups": [
                {
                    "target_object": object_name,
                    "target_type": object_types.get(object_name, "object"),
                    "candidate_location_predicates": [
                        predicate.to_pddl_str() for predicate in grouped_location_predicates[object_name]
                    ],
                }
                for object_name in ordered_names
            ],
        }
        client = make_client(api_key=self.api_key, base_url=self.base_url)
        last_error: str | None = None
        for attempt in range(2):
            if attempt:
                payload["correction"] = (
                    "The previous response was structurally invalid. Return every target_object "
                    "exactly once without additions or omissions."
                )
            reply = safe_chat(
                client,
                self._object_location_visibility_batch_prompt,
                build_user_content(
                    text=json.dumps(payload, ensure_ascii=False, indent=2),
                    image_path=str(image_path),
                ),
                model=self.model,
                temperature=self.temperature,
                max_tokens=max(self.max_tokens, 4096),
                verbose=self.verbose,
            )
            data = extract_json_object(reply)
            rows = data.get("object_visibility", [])
            if not isinstance(rows, list):
                last_error = "response did not contain an object_visibility list"
                continue
            parsed: dict[str, bool] = {}
            invalid_names: list[str] = []
            for row in rows:
                if not isinstance(row, dict):
                    invalid_names.append("<non-object>")
                    continue
                object_name = str(row.get("target_object", "")).strip()
                value = row.get("visually_resolved")
                if object_name not in grouped_location_predicates or object_name in parsed or not isinstance(value, bool):
                    invalid_names.append(object_name)
                    continue
                parsed[object_name] = value
            missing_names = [name for name in ordered_names if name not in parsed]
            if invalid_names or missing_names or len(rows) != len(ordered_names):
                last_error = (
                    f"invalid={invalid_names or 'none'}, missing={missing_names or 'none'}, "
                    f"expected_count={len(ordered_names)}, actual_count={len(rows)}"
                )
                continue
            return [(name, parsed[name]) for name in ordered_names]
        raise ValueError(f"Invalid batched object visibility judgment after retry: {last_error}.")

    @staticmethod
    def _location_relation_can_hold_while_hidden(predicate_name: str) -> bool:
        normalized = predicate_name.strip().lower()
        return normalized in {
            "in",
            "inside",
            "within",
            "contains",
            "contained_in",
            "enclosed_in",
        }

    def _is_object_location_visually_resolved(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None,
        instruction: str,
        objects: list[ObjectDeclaration],
        target_object_name: str,
        candidate_predicates: list[Predicate],
    ) -> bool:
        object_types = {item.name: item.type_name for item in objects}
        payload = {
            "domain_summary": domain_analysis.render_summary(),
            "instruction": instruction.strip(),
            "initial_state_hint": self._initial_state_hint,
            "image_input_note": image_input_note
            or "The provided image is a single-view snapshot of the initial scene.",
            "objects": [{"name": item.name, "type_name": item.type_name} for item in objects],
            "target_object": {
                "name": target_object_name,
                "type_name": object_types.get(target_object_name, "object"),
            },
            "candidate_location_predicates": [predicate.to_pddl_str() for predicate in candidate_predicates],
        }
        user_content = build_user_content(
            text=json.dumps(payload, ensure_ascii=False, indent=2),
            image_path=str(image_path),
        )
        client = make_client(api_key=self.api_key, base_url=self.base_url)
        reply = safe_chat(
            client,
            self._object_location_visibility_prompt,
            user_content,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        value = data.get("visually_resolved")
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized not in {"true", "false"}:
            raise ValueError("Object location visibility judgment must return visually_resolved=true/false.")
        return normalized == "true"

    def _build_historical_prior_judgments(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        predicates: list[Predicate],
        historical_grounding_root: str | Path | None,
    ) -> tuple[list[PredicateTruthJudgment], list[Predicate]]:
        fixed_judgments: list[PredicateTruthJudgment] = []
        remaining_predicates: list[Predicate] = []
        for predicate in predicates:
            prior_truth = self._estimate_single_predicate_truth_from_historical_groundings(
                domain_analysis=domain_analysis,
                predicate=predicate,
                historical_grounding_root=historical_grounding_root,
            )
            if prior_truth is None:
                remaining_predicates.append(predicate)
                continue
            fixed_judgments.append(
                PredicateTruthJudgment(
                    predicate=predicate,
                    truth_value=prior_truth,
                    justification=(
                        "Historical init-state prior: all compatible historical samples "
                        f"set this predicate to {'true' if prior_truth else 'false'}."
                    ),
                )
            )
        return fixed_judgments, remaining_predicates

    def _estimate_single_predicate_truth_from_historical_groundings(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        predicate: Predicate,
        historical_grounding_root: str | Path | None,
    ) -> bool | None:
        if historical_grounding_root is None:
            return None
        histories = self._load_historical_problem_init_states(
            domain_analysis=domain_analysis,
            historical_grounding_root=historical_grounding_root,
        )
        if not histories:
            return None
        seen_values: set[bool] = set()
        for object_names, predicate_truths in histories:
            if not all(argument in object_names for argument in predicate.params):
                continue
            if predicate not in predicate_truths:
                continue
            seen_values.add(predicate_truths[predicate])
            if len(seen_values) > 1:
                return None
        if len(seen_values) != 1:
            return None
        return next(iter(seen_values))

    @staticmethod
    def _location_predicate_cache_key(domain_analysis: DomainAnalysisResult) -> str:
        entries: list[str] = []
        for predicate in sorted(domain_analysis.parsed_domain.predicates, key=lambda item: item.name):
            parameter_types = domain_analysis.parsed_domain.predicate_parameter_types.get(predicate, [])
            rendered_types = ",".join(type_name for _param_name, type_name in parameter_types)
            entries.append(f"{predicate.name}:{rendered_types}")
        return f"{domain_analysis.parsed_domain.domain_name}|{'|'.join(entries)}"

    @staticmethod
    def _is_type_subtype(
        domain_analysis: DomainAnalysisResult,
        type_name: str,
        ancestor_type_name: str,
    ) -> bool:
        node = domain_analysis.parsed_domain.types.get(type_name)
        if node is None:
            return type_name == ancestor_type_name
        return node.is_subtype_of(ancestor_type_name)

    def _group_uncertain_predicates(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None,
        instruction: str,
        objects: list[ObjectDeclaration],
        manipulation_summary: list[dict[str, object]],
        uncertain_predicates: list[Predicate],
    ) -> list[dict[str, Any]]:
        location_predicate_names = self._select_location_predicate_names(domain_analysis=domain_analysis)
        location_groups, standalone_predicates = self._partition_location_predicates_by_object(
            domain_analysis=domain_analysis,
            objects=objects,
            predicates=uncertain_predicates,
            location_predicate_names=location_predicate_names,
        )
        predefined_groups = [
            {
                "name": f"{object_name}_location",
                "group_kind": "mutex" if len(predicates) > 1 else "binary",
                "predicates": list(predicates),
                "description": (
                    "Alternative learned-domain location states for one object; "
                    "the object occupies exactly one represented location."
                ),
            }
            for object_name, predicates in sorted(location_groups.items())
        ]
        if not standalone_predicates:
            return predefined_groups

        payload = {
            "domain_summary": domain_analysis.render_summary(),
            "instruction": instruction.strip(),
            "initial_state_hint": self._initial_state_hint,
            "image_input_note": image_input_note
            or "The provided image is a single-view snapshot of the initial scene.",
            "objects": [{"name": item.name, "type_name": item.type_name} for item in objects],
            "manipulation_records": manipulation_summary,
            "uncertain_ground_predicates": [predicate.to_pddl_str() for predicate in standalone_predicates],
        }
        user_content = build_user_content(
            text=json.dumps(payload, ensure_ascii=False, indent=2),
            image_path=str(image_path),
        )
        client = make_client(api_key=self.api_key, base_url=self.base_url)
        reply = safe_chat(
            client,
            self._uncertain_grouping_prompt,
            user_content,
            model=self.model,
            temperature=self.temperature,
            max_tokens=max(self.max_tokens, 4096),
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        raw_groups = data.get("groups", [])
        if not isinstance(raw_groups, list):
            raise ValueError("Uncertain grouping response must contain a `groups` list.")
        available = {predicate.to_pddl_str(): predicate for predicate in standalone_predicates}
        validated: list[dict[str, Any]] = []
        covered: set[str] = set()
        for index, raw_group in enumerate(raw_groups, start=1):
            if not isinstance(raw_group, dict):
                raise ValueError(f"Invalid group entry at index {index}: {raw_group!r}")
            name = str(raw_group.get("name", f"group_{index}")).strip() or f"group_{index}"
            group_kind = str(raw_group.get("group_kind", "binary")).strip().lower()
            if group_kind not in {"binary", "mutex", "correlated"}:
                raise ValueError(f"Unsupported group_kind {group_kind!r}; expected binary, mutex, or correlated.")
            predicate_strings = raw_group.get("predicates", [])
            if not isinstance(predicate_strings, list) or not predicate_strings:
                raise ValueError(f"Group {name!r} must contain at least one predicate.")
            scope: list[Predicate] = []
            seen_local: set[str] = set()
            for predicate_text in predicate_strings:
                key = str(predicate_text).strip()
                if key not in available:
                    raise ValueError(f"Group {name!r} references unknown uncertain predicate {key!r}.")
                if key in seen_local:
                    continue
                seen_local.add(key)
                scope.append(available[key])
            if group_kind == "binary" and len(scope) != 1:
                raise ValueError(f"Binary group {name!r} must contain exactly one predicate.")
            overlap = seen_local & covered
            if overlap:
                raise ValueError(f"Uncertain groups overlap on predicates: {sorted(overlap)!r}")
            covered |= seen_local
            if group_kind == "correlated" and len(scope) > 8:
                validated.extend(
                    {
                        "name": f"{name}_{predicate.name}_{item_index}",
                        "group_kind": "binary",
                        "predicates": [predicate],
                        "description": (
                            "Large unstructured correlated group split into bounded "
                            "singleton factors to avoid exponential case enumeration."
                        ),
                    }
                    for item_index, predicate in enumerate(scope, start=1)
                )
                continue
            validated.append(
                {
                    "name": name,
                    "group_kind": group_kind,
                    "predicates": scope,
                    "description": str(raw_group.get("description", "")).strip() or None,
                }
            )
        missing = set(available) - covered
        if missing:
            raise ValueError(f"Uncertain grouping did not cover all uncertain predicates. Missing={sorted(missing)!r}")
        return predefined_groups + validated

    def _estimate_uncertain_group_distribution(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        group: dict[str, Any],
        historical_grounding_root: str | Path | None,
        prior_data_confidence: float = 0.0,
    ) -> BeliefFactor:
        prior_data_confidence = _validate_prior_data_confidence(prior_data_confidence)
        uniform_cases = self._uniform_group_cases(
            group,
        )
        if prior_data_confidence <= 1e-12:
            cases = uniform_cases
            factor_name = f"{group['name']}:uniform"
        else:
            historical_cases = self._estimate_group_distribution_from_historical_groundings(
                domain_analysis=domain_analysis,
                group=group,
                historical_grounding_root=historical_grounding_root,
            )
            if historical_cases is None:
                cases = uniform_cases
                factor_name = f"{group['name']}:uniform"
            elif prior_data_confidence >= 1.0 - 1e-12:
                cases = historical_cases
                factor_name = f"{group['name']}:historical"
            else:
                cases = self._mix_group_cases(
                    scope=list(group["predicates"]),
                    data_cases=historical_cases,
                    uniform_cases=uniform_cases,
                    prior_data_confidence=prior_data_confidence,
                )
                factor_name = f"{group['name']}:blended"
        if cases:
            max_probability, max_true_predicates = max(cases, key=lambda item: item[0])
            if max_probability > self.deterministic_collapse_threshold:
                cases = [(1.0, list(max_true_predicates))]
        factor = BeliefFactor(
            name=factor_name,
            scope=list(group["predicates"]),
            cases=cases,
        )
        factor.validate()
        return factor

    @staticmethod
    def _mix_group_cases(
        *,
        scope: list[Predicate],
        data_cases: list[tuple[float, list[Predicate]]],
        uniform_cases: list[tuple[float, list[Predicate]]],
        prior_data_confidence: float,
    ) -> list[tuple[float, list[Predicate]]]:
        case_probabilities: dict[tuple[bool, ...], float] = {}

        def _assignment_key(true_predicates: list[Predicate]) -> tuple[bool, ...]:
            true_set = set(true_predicates)
            return tuple(predicate in true_set for predicate in scope)

        for probability, true_predicates in data_cases:
            case_probabilities[_assignment_key(true_predicates)] = case_probabilities.get(
                _assignment_key(true_predicates), 0.0
            ) + prior_data_confidence * float(probability)
        for probability, true_predicates in uniform_cases:
            case_probabilities[_assignment_key(true_predicates)] = case_probabilities.get(
                _assignment_key(true_predicates), 0.0
            ) + (1.0 - prior_data_confidence) * float(probability)

        mixed_cases: list[tuple[float, list[Predicate]]] = []
        for assignment, probability in sorted(case_probabilities.items()):
            if probability <= 0.0:
                continue
            mixed_cases.append(
                (
                    probability,
                    [predicate for predicate, truth in zip(scope, assignment) if truth],
                )
            )
        total_probability = sum(probability for probability, _true_predicates in mixed_cases)
        if total_probability <= 0.0:
            raise ValueError("Blended group prior produced zero total probability.")
        return [
            (probability / total_probability, list(true_predicates)) for probability, true_predicates in mixed_cases
        ]

    def _estimate_group_distribution_from_historical_groundings(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        group: dict[str, Any],
        historical_grounding_root: str | Path | None,
    ) -> list[tuple[float, list[Predicate]]] | None:
        if historical_grounding_root is None:
            return None
        histories = self._load_historical_problem_init_states(
            domain_analysis=domain_analysis,
            historical_grounding_root=historical_grounding_root,
        )
        if not histories:
            return None

        scope = list(group["predicates"])
        counts: dict[tuple[bool, ...], int] = {}
        usable_sample_count = 0
        for object_names, init_state in histories:
            if not self._group_scope_is_usable_in_sample(scope=scope, object_names=object_names):
                continue
            assignment = tuple(bool(init_state.get(predicate, False)) for predicate in scope)
            if not self._assignment_is_valid_for_group(assignment, group_kind=group["group_kind"]):
                continue
            counts[assignment] = counts.get(assignment, 0) + 1
            usable_sample_count += 1

        if usable_sample_count == 0:
            return None

        cases: list[tuple[float, list[Predicate]]] = []
        for assignment, count in sorted(counts.items()):
            probability = count / usable_sample_count
            true_predicates = [predicate for predicate, truth in zip(scope, assignment) if truth]
            cases.append((probability, true_predicates))
        return cases or None

    def _load_historical_problem_init_states(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        historical_grounding_root: str | Path,
    ) -> list[tuple[set[str], dict[Predicate, bool]]]:
        root = str(Path(historical_grounding_root).resolve())
        cached = self._historical_init_cache.get(root)
        if cached is not None:
            return cached

        grounding_root = Path(historical_grounding_root)
        if not grounding_root.exists():
            self._historical_init_cache[root] = []
            return []

        constant_names = set(domain_analysis.parsed_domain.constants)
        loaded: list[tuple[set[str], dict[Predicate, bool]]] = []
        problem_files = sorted(grounding_root.glob("episode*/problem.pddl"))
        if not problem_files:
            problem_files = sorted(grounding_root.glob("episode_*/problem.pddl"))
        for problem_file in problem_files:
            try:
                parsed_problem = parse_problem(problem_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            sample_objects = [
                ObjectDeclaration(name=name, type_name=type_name) for name, type_name in parsed_problem.objects.items()
            ]
            try:
                grounded_predicates = build_grounded_predicates_for_objects(
                    domain_analysis.parsed_domain,
                    sample_objects,
                    problem_name=parsed_problem.problem_name,
                )
            except Exception:
                continue
            init_true_predicates = set(parsed_problem.init_state)
            predicate_truths = {
                grounded_predicate: (grounded_predicate in init_true_predicates)
                for grounded_predicate in grounded_predicates
            }
            object_names = set(parsed_problem.objects) | constant_names
            loaded.append((object_names, predicate_truths))
        self._historical_init_cache[root] = loaded
        return loaded

    @staticmethod
    def _group_scope_is_usable_in_sample(
        *,
        scope: list[Predicate],
        object_names: set[str],
    ) -> bool:
        return all(all(argument in object_names for argument in predicate.params) for predicate in scope)

    @staticmethod
    def _assignment_is_valid_for_group(assignment: tuple[bool, ...], *, group_kind: str) -> bool:
        if group_kind == "binary":
            return len(assignment) == 1
        if group_kind == "mutex":
            return sum(1 for item in assignment if item) == 1
        return True

    @staticmethod
    def _enumerate_valid_assignments(
        *,
        predicate_count: int,
        group_kind: str,
    ) -> list[tuple[bool, ...]]:
        if group_kind == "binary":
            if predicate_count != 1:
                raise ValueError("Binary groups must contain exactly one predicate.")
            return [(False,), (True,)]
        if group_kind == "mutex":
            return [
                tuple(index == true_index for index in range(predicate_count)) for true_index in range(predicate_count)
            ]
        assignments: list[tuple[bool, ...]] = []
        for mask in range(1 << predicate_count):
            assignment = tuple(bool((mask >> index) & 1) for index in range(predicate_count))
            if InitialBeliefGenerator._assignment_is_valid_for_group(
                assignment,
                group_kind=group_kind,
            ):
                assignments.append(assignment)
        return assignments

    def _uniform_group_cases(
        self,
        group: dict[str, Any],
    ) -> list[tuple[float, list[Predicate]]]:
        scope = list(group["predicates"])
        assignments = self._enumerate_valid_assignments(
            predicate_count=len(scope),
            group_kind=group["group_kind"],
        )
        if not assignments:
            raise ValueError(f"Unable to enumerate any valid assignments for group {group['name']!r}.")
        probability = 1.0 / len(assignments)
        return [
            (
                probability,
                [predicate for predicate, truth in zip(scope, assignment) if truth],
            )
            for assignment in assignments
        ]

    def _bayesian_update_singleton_distribution_from_observation_rules(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None,
        instruction: str,
        objects: list[ObjectDeclaration],
        manipulation_summary: list[dict[str, object]],
        predicate: Predicate,
        base_factor: BeliefFactor,
        observation_reliability: dict[str, dict[str, Any]],
        deterministic_truth_by_predicate: dict[Predicate, bool],
    ) -> tuple[BeliefFactor, dict[str, Any] | None]:
        reliability = observation_reliability.get(predicate.name)
        if reliability is None:
            return base_factor, None
        observable_name = self._resolve_observable_name_for_predicate(
            domain_analysis=domain_analysis,
            predicate_name=predicate.name,
        )
        if observable_name is None:
            return base_factor, {
                "predicate": predicate.to_pddl_str(),
                "changed": False,
                "reason": "No observable matched this predicate; kept base factor.",
            }
        applicable_rule_names = self._find_applicable_init_observation_rules(
            domain_analysis=domain_analysis,
            predicate=predicate,
            reliability_entry=reliability,
            deterministic_truth_by_predicate=deterministic_truth_by_predicate,
            objects=objects,
        )
        if not applicable_rule_names:
            return base_factor, {
                "predicate": predicate.to_pddl_str(),
                "changed": False,
                "reason": "No init observation rule condition matched on the deterministic subset of the init belief.",
                "observable_name": observable_name,
            }
        prior_true_probability = sum(
            probability for probability, true_predicates in base_factor.cases if predicate in true_predicates
        )
        if prior_true_probability <= 1e-9 or prior_true_probability >= 1.0 - 1e-9:
            return base_factor, {
                "predicate": predicate.to_pddl_str(),
                "changed": False,
                "reason": "Base factor is already deterministic; Bayesian update leaves it unchanged.",
                "observable_name": observable_name,
                "prior_true_probability": prior_true_probability,
            }
        observed_true = self._judge_observable_from_image(
            domain_analysis=domain_analysis,
            image_path=image_path,
            image_input_note=image_input_note,
            instruction=instruction,
            objects=objects,
            manipulation_summary=manipulation_summary,
            target_observable=Predicate(observable_name, list(predicate.params)),
            target_predicate=predicate,
        )
        true_likelihood = _normalize_probability(
            reliability.get("truth_true_observe_true_probability"),
            default=0.5,
        )
        false_likelihood = _normalize_probability(
            reliability.get("truth_false_observe_true_probability"),
            default=0.5,
        )
        if observed_true:
            likelihood_given_true = true_likelihood
            likelihood_given_false = false_likelihood
        else:
            likelihood_given_true = 1.0 - true_likelihood
            likelihood_given_false = 1.0 - false_likelihood
        denominator = likelihood_given_true * prior_true_probability + likelihood_given_false * (
            1.0 - prior_true_probability
        )
        if denominator <= 1e-12:
            posterior_true_probability = prior_true_probability
        else:
            posterior_true_probability = likelihood_given_true * prior_true_probability / denominator
        posterior_true_probability = _normalize_probability(
            posterior_true_probability,
            default=prior_true_probability,
        )
        cases = [
            (posterior_true_probability, [predicate]),
            (1.0 - posterior_true_probability, []),
        ]
        calibrated = BeliefFactor(
            name=base_factor.name,
            scope=list(base_factor.scope),
            cases=cases,
        )
        calibrated.validate()
        return calibrated, {
            "predicate": predicate.to_pddl_str(),
            "changed": cases != base_factor.cases,
            "base_cases": [
                {
                    "probability": probability,
                    "true_predicates": [item.to_pddl_str() for item in true_predicates],
                }
                for probability, true_predicates in base_factor.cases
            ],
            "calibrated_cases": [
                {
                    "probability": probability,
                    "true_predicates": [item.to_pddl_str() for item in true_predicates],
                }
                for probability, true_predicates in calibrated.cases
            ],
            "observation_reliability": reliability,
            "observable_name": observable_name,
            "observed_true": observed_true,
            "applicable_rule_names": applicable_rule_names,
            "prior_true_probability": prior_true_probability,
            "posterior_true_probability": posterior_true_probability,
            "bayesian_likelihood_given_true": likelihood_given_true,
            "bayesian_likelihood_given_false": likelihood_given_false,
        }

    def _bayesian_update_group_distribution_from_observation_rules(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None,
        instruction: str,
        objects: list[ObjectDeclaration],
        manipulation_summary: list[dict[str, object]],
        group: dict[str, Any],
        base_factor: BeliefFactor,
        observation_reliability: dict[str, dict[str, Any]],
        deterministic_truth_by_predicate: dict[Predicate, bool],
    ) -> tuple[BeliefFactor, dict[str, Any] | None]:
        predicate_updates: list[dict[str, Any]] = []
        observed_truth_by_predicate: dict[Predicate, bool] = {}
        applicable_rule_names: dict[str, list[str]] = {}
        for predicate in base_factor.scope:
            reliability = observation_reliability.get(predicate.name)
            if reliability is None:
                continue
            observable_name = self._resolve_observable_name_for_predicate(
                domain_analysis=domain_analysis,
                predicate_name=predicate.name,
            )
            if observable_name is None:
                continue
            predicate_rule_names = self._find_applicable_init_observation_rules(
                domain_analysis=domain_analysis,
                predicate=predicate,
                reliability_entry=reliability,
                deterministic_truth_by_predicate=deterministic_truth_by_predicate,
                objects=objects,
            )
            if not predicate_rule_names:
                continue
            observed_true = self._judge_observable_from_image(
                domain_analysis=domain_analysis,
                image_path=image_path,
                image_input_note=image_input_note,
                instruction=instruction,
                objects=objects,
                manipulation_summary=manipulation_summary,
                target_observable=Predicate(observable_name, list(predicate.params)),
                target_predicate=predicate,
            )
            observed_truth_by_predicate[predicate] = observed_true
            applicable_rule_names[predicate.to_pddl_str()] = predicate_rule_names
            predicate_updates.append(
                {
                    "predicate": predicate.to_pddl_str(),
                    "observable_name": observable_name,
                    "observed_true": observed_true,
                    "reliability": reliability,
                    "applicable_rule_names": predicate_rule_names,
                }
            )

        if not predicate_updates:
            return base_factor, None

        unnormalized_cases: list[tuple[float, list[Predicate]]] = []
        for prior_probability, true_predicates in base_factor.cases:
            if prior_probability <= 0.0:
                continue
            true_set = set(true_predicates)
            likelihood = 1.0
            for update in predicate_updates:
                predicate = next(item for item in base_factor.scope if item.to_pddl_str() == update["predicate"])
                observed_true = bool(update["observed_true"])
                reliability = dict(update["reliability"])
                truth_is_true = predicate in true_set
                if truth_is_true:
                    true_likelihood = _normalize_probability(
                        reliability.get("truth_true_observe_true_probability"),
                        default=0.5,
                    )
                    likelihood *= true_likelihood if observed_true else (1.0 - true_likelihood)
                else:
                    false_likelihood = _normalize_probability(
                        reliability.get("truth_false_observe_true_probability"),
                        default=0.5,
                    )
                    likelihood *= false_likelihood if observed_true else (1.0 - false_likelihood)
            unnormalized_cases.append((prior_probability * likelihood, list(true_predicates)))

        total_probability = sum(probability for probability, _case in unnormalized_cases)
        if total_probability <= 1e-12:
            return base_factor, {
                "group_name": str(group.get("name", "")),
                "group_kind": str(group.get("group_kind", "")),
                "changed": False,
                "reason": "Observation likelihoods collapsed to zero; kept base factor.",
                "predicate_updates": predicate_updates,
            }

        calibrated_cases = [
            (probability / total_probability, list(true_predicates))
            for probability, true_predicates in unnormalized_cases
            if probability > 0.0
        ]
        calibrated = BeliefFactor(
            name=base_factor.name,
            scope=list(base_factor.scope),
            cases=calibrated_cases,
        )
        calibrated.validate()
        return calibrated, {
            "group_name": str(group.get("name", "")),
            "group_kind": str(group.get("group_kind", "")),
            "changed": calibrated.cases != base_factor.cases,
            "base_cases": [
                {
                    "probability": probability,
                    "true_predicates": [item.to_pddl_str() for item in true_predicates],
                }
                for probability, true_predicates in base_factor.cases
            ],
            "calibrated_cases": [
                {
                    "probability": probability,
                    "true_predicates": [item.to_pddl_str() for item in true_predicates],
                }
                for probability, true_predicates in calibrated.cases
            ],
            "predicate_updates": predicate_updates,
            "applicable_rule_names": applicable_rule_names,
        }

    @staticmethod
    def _extract_init_observation_rule_summary(
        domain_analysis: DomainAnalysisResult,
    ) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for rule_schema in domain_analysis.parsed_domain.observation_rules:
            rule_name = rule_schema.rule.name
            if not rule_name.startswith("init_obs_"):
                continue
            observable = next(
                (
                    item
                    for item in rule_schema.rule.distribution
                    if item.name.startswith("obs-") or item.name.startswith("obs_")
                ),
                None,
            )
            if observable is None:
                continue
            if observable.name.startswith("obs-"):
                predicate_name = observable.name[len("obs-") :]
            else:
                predicate_name = observable.name[len("obs_") :]
            observe_true_probability = InitialBeliefGenerator._extract_positive_observable_probability(
                rule_schema.distribution_expr,
                observable.name,
            )
            if observe_true_probability is None:
                continue
            bucket_key: str | None = None
            if rule_name.endswith("_true"):
                bucket_key = "truth_true_observe_true_probability"
            elif rule_name.endswith("_false"):
                bucket_key = "truth_false_observe_true_probability"
            if bucket_key is None:
                continue
            entry = summary.setdefault(
                predicate_name,
                {
                    "predicate_name": predicate_name,
                    "rules": [],
                },
            )
            entry[bucket_key] = observe_true_probability
            entry["rules"].append(
                {
                    "rule_name": rule_name,
                    "truth_case": "true" if rule_name.endswith("_true") else "false",
                    "condition": rule_schema.condition,
                    "parameter_names": list(rule_schema.parameters),
                    "parameter_types": list(rule_schema.parameter_types),
                }
            )
        return summary

    def _find_applicable_init_observation_rules(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        predicate: Predicate,
        reliability_entry: dict[str, Any],
        deterministic_truth_by_predicate: dict[Predicate, bool],
        objects: list[ObjectDeclaration],
    ) -> list[str]:
        rule_entries = reliability_entry.get("rules", [])
        if not isinstance(rule_entries, list):
            return []
        applicable: list[str] = []
        object_type_by_name = self._build_object_type_index(
            domain_analysis=domain_analysis,
            objects=objects,
        )
        for entry in rule_entries:
            if not isinstance(entry, dict):
                continue
            parameter_names = list(entry.get("parameter_names", []))
            environment = {name: argument for name, argument in zip(parameter_names, predicate.params)}
            condition = entry.get("condition")
            outcome = self._evaluate_observation_condition_on_deterministic_truth(
                expr=condition,
                environment=environment,
                target_predicate=predicate,
                deterministic_truth_by_predicate=deterministic_truth_by_predicate,
                object_type_by_name=object_type_by_name,
                domain_analysis=domain_analysis,
            )
            if outcome is not False:
                applicable.append(str(entry.get("rule_name", "")))
        return applicable

    @staticmethod
    def _build_object_type_index(
        *,
        domain_analysis: DomainAnalysisResult,
        objects: list[ObjectDeclaration],
    ) -> dict[str, str]:
        index = dict(domain_analysis.parsed_domain.constants)
        for item in objects:
            index[item.name] = item.type_name
        return index

    def _evaluate_observation_condition_on_deterministic_truth(
        self,
        *,
        expr: object,
        environment: dict[str, str],
        target_predicate: Predicate,
        deterministic_truth_by_predicate: dict[Predicate, bool],
        object_type_by_name: dict[str, str],
        domain_analysis: DomainAnalysisResult,
    ) -> bool | None:
        if expr is None:
            return True
        if isinstance(expr, str):
            return None
        if not isinstance(expr, list) or not expr:
            return None
        head = expr[0]
        if not isinstance(head, str):
            return None
        if head == "and":
            saw_known = False
            for item in expr[1:]:
                value = self._evaluate_observation_condition_on_deterministic_truth(
                    expr=item,
                    environment=environment,
                    target_predicate=target_predicate,
                    deterministic_truth_by_predicate=deterministic_truth_by_predicate,
                    object_type_by_name=object_type_by_name,
                    domain_analysis=domain_analysis,
                )
                if value is False:
                    return False
                if value is True:
                    saw_known = True
            return True if saw_known else None
        if head == "not":
            if len(expr) != 2:
                return None
            value = self._evaluate_observation_condition_on_deterministic_truth(
                expr=expr[1],
                environment=environment,
                target_predicate=target_predicate,
                deterministic_truth_by_predicate=deterministic_truth_by_predicate,
                object_type_by_name=object_type_by_name,
                domain_analysis=domain_analysis,
            )
            if value is None:
                return None
            return not value
        if head == "forall":
            if len(expr) != 3 or not isinstance(expr[1], list):
                return None
            declarations = _parse_typed_symbol_sequence(expr[1], default_type="object")
            assignments = self._enumerate_forall_assignments(
                declarations=declarations,
                object_type_by_name=object_type_by_name,
                domain_analysis=domain_analysis,
            )
            saw_known = False
            for assignment in assignments:
                merged_env = dict(environment)
                merged_env.update(assignment)
                value = self._evaluate_observation_condition_on_deterministic_truth(
                    expr=expr[2],
                    environment=merged_env,
                    target_predicate=target_predicate,
                    deterministic_truth_by_predicate=deterministic_truth_by_predicate,
                    object_type_by_name=object_type_by_name,
                    domain_analysis=domain_analysis,
                )
                if value is False:
                    return False
                if value is True:
                    saw_known = True
            return True if saw_known or not assignments else None
        grounded_literal = self._ground_condition_literal(expr=expr, environment=environment)
        if grounded_literal is None:
            return None
        if grounded_literal.same_signature(target_predicate) and grounded_literal.params == target_predicate.params:
            return None
        truth_value = deterministic_truth_by_predicate.get(grounded_literal)
        return truth_value if truth_value is not None else None

    def _enumerate_forall_assignments(
        self,
        *,
        declarations: list[tuple[str, str]],
        object_type_by_name: dict[str, str],
        domain_analysis: DomainAnalysisResult,
    ) -> list[dict[str, str]]:
        if not declarations:
            return [{}]
        name, type_name = declarations[0]
        candidates = [
            object_name
            for object_name, object_type in object_type_by_name.items()
            if self._type_matches(
                object_type=object_type,
                required_type=type_name,
                domain_analysis=domain_analysis,
            )
        ]
        results: list[dict[str, str]] = []
        tail_assignments = self._enumerate_forall_assignments(
            declarations=declarations[1:],
            object_type_by_name=object_type_by_name,
            domain_analysis=domain_analysis,
        )
        for candidate in candidates:
            for tail in tail_assignments:
                assignment = dict(tail)
                assignment[name] = candidate
                results.append(assignment)
        return results

    @staticmethod
    def _type_matches(
        *,
        object_type: str,
        required_type: str,
        domain_analysis: DomainAnalysisResult,
    ) -> bool:
        object_node = domain_analysis.parsed_domain.types.get(object_type)
        if object_node is None:
            return object_type == required_type or required_type == "object"
        return object_node.is_subtype_of(required_type)

    @staticmethod
    def _ground_condition_literal(
        *,
        expr: object,
        environment: dict[str, str],
    ) -> Predicate | None:
        if not isinstance(expr, list) or not expr:
            return None
        head = expr[0]
        if not isinstance(head, str) or head in {"and", "not", "forall"}:
            return None
        grounded_args: list[str] = []
        for raw_argument in expr[1:]:
            if not isinstance(raw_argument, str):
                return None
            grounded_args.append(environment.get(raw_argument, raw_argument))
        return Predicate(head, grounded_args)

    @staticmethod
    def _resolve_observable_name_for_predicate(
        *,
        domain_analysis: DomainAnalysisResult,
        predicate_name: str,
    ) -> str | None:
        preferred_names = [f"obs-{predicate_name}", f"obs_{predicate_name}"]
        observable_names = {item.name for item in domain_analysis.parsed_domain.observables}
        for name in preferred_names:
            if name in observable_names:
                return name
        return None

    def _judge_observable_from_image(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None,
        instruction: str,
        objects: list[ObjectDeclaration],
        manipulation_summary: list[dict[str, object]],
        target_observable: Predicate,
        target_predicate: Predicate,
    ) -> bool:
        payload = {
            "domain_summary": domain_analysis.render_summary(),
            "instruction": instruction.strip(),
            "initial_state_hint": self._initial_state_hint,
            "image_input_note": image_input_note
            or "The provided image is a single-view snapshot of the initial scene.",
            "objects": [{"name": item.name, "type_name": item.type_name} for item in objects],
            "manipulation_records": manipulation_summary,
            "target_observable": target_observable.to_pddl_str(),
            "target_predicate": target_predicate.to_pddl_str(),
        }
        user_content = build_user_content(
            text=json.dumps(payload, ensure_ascii=False, indent=2),
            image_path=str(image_path),
        )
        client = make_client(api_key=self.api_key, base_url=self.base_url)
        reply = safe_chat(
            client,
            self._observable_judgment_prompt,
            user_content,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        observable_value = str(data.get("observable_value", "")).strip().lower()
        if observable_value not in {"true", "false"}:
            raise ValueError(f"Observable judgment must return observable_value=true/false, got {observable_value!r}.")
        return observable_value == "true"

    @staticmethod
    def _extract_positive_observable_probability(
        expr: object,
        observable_name: str,
    ) -> float | None:
        if not isinstance(expr, list) or not expr or expr[0] != "probabilistic":
            return None
        probability = 0.0
        index = 1
        while index + 1 < len(expr):
            raw_probability = expr[index]
            branch = expr[index + 1]
            try:
                branch_probability = float(raw_probability)
            except (TypeError, ValueError):
                return None
            if InitialBeliefGenerator._expr_contains_positive_observable(branch, observable_name):
                probability += branch_probability
            index += 2
        return probability

    @staticmethod
    def _expr_contains_positive_observable(expr: object, observable_name: str) -> bool:
        if isinstance(expr, str):
            return False
        if not isinstance(expr, list) or not expr:
            return False
        head = expr[0]
        if head == "and":
            return any(
                InitialBeliefGenerator._expr_contains_positive_observable(item, observable_name)
                for item in expr[1:]
            )
        if head == "not":
            return False
        return isinstance(head, str) and head == observable_name
