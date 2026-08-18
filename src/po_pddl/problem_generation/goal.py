from __future__ import annotations

import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from po_pddl.config import DEFAULT_MODEL

from ..core.models.predicate import Predicate
from ..core.parser.sexpr import SExpr, loads_sexpr
from ..domain_generation.infrastructure.llm_shared import build_user_content, make_client, safe_chat
from ..domain_generation.infrastructure.response_parsing import extract_json_object
from ..domain_generation.stages.problem_grounding.models import ObjectDeclaration
from ..prompts import load_prompt
from .domain_analysis import DomainAnalysisResult, build_grounded_predicates_for_objects
from .model_config import resolve_online_llm_config


def _load_prompt(name: str) -> str:
    return load_prompt(name)


def _parse_predicate_literal(text: str) -> Predicate:
    roots = loads_sexpr(text)
    if len(roots) != 1 or not isinstance(roots[0], list) or not roots[0]:
        raise ValueError(f"Expected exactly one grounded predicate literal, got {text!r}")
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


def _render_goal_expr(expr: SExpr) -> str:
    if isinstance(expr, list):
        if not expr:
            return "()"
        return "(" + " ".join(_render_goal_expr(item) for item in expr) + ")"
    return str(expr)


def _parse_goal_expr(text: str) -> SExpr:
    roots = loads_sexpr(text)
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one goal expression, got {text!r}")
    return roots[0]


def _is_observation_helper_predicate(predicate_name: str) -> bool:
    return (
        predicate_name.startswith("obs-")
        or predicate_name.startswith("obs_")
        or predicate_name.startswith("last_action_")
    )


def _action_constant_names(domain_analysis: DomainAnalysisResult) -> set[str]:
    result: set[str] = set()
    for constant_name, type_name in domain_analysis.parsed_domain.constants.items():
        if constant_name.startswith("action_") or "action" in type_name:
            result.add(constant_name)
    return result


def _render_goal_domain_summary(domain_analysis: DomainAnalysisResult) -> str:
    action_constant_names = _action_constant_names(domain_analysis)

    predicate_lines = []
    for predicate in domain_analysis.parsed_domain.predicates:
        if _is_observation_helper_predicate(predicate.name):
            continue
        parameter_types = domain_analysis.parsed_domain.predicate_parameter_types.get(predicate, [])
        if parameter_types:
            rendered_params = ", ".join(f"{name}: {type_name}" for name, type_name in parameter_types)
        else:
            rendered_params = "(no parameters)"
        predicate_lines.append(f"- {predicate.name}: {rendered_params}")
    if not predicate_lines:
        predicate_lines.append("- (no predicates declared)")

    filtered_constants = sorted(
        constant_name
        for constant_name in domain_analysis.parsed_domain.constants
        if constant_name not in action_constant_names
    )
    filtered_types = sorted(type_name for type_name in domain_analysis.parsed_domain.types if "action" not in type_name)

    action_names = (
        ", ".join(schema.action.name for schema in domain_analysis.parsed_domain.actions) or "(no actions declared)"
    )
    constant_names = ", ".join(filtered_constants) or "(no constants)"
    type_names = ", ".join(filtered_types) or "(no types)"
    return "\n".join(
        [
            f"Domain: {domain_analysis.parsed_domain.domain_name}",
            f"Types: {type_names}",
            f"Constants: {constant_names}",
            f"Actions: {action_names}",
            "Predicates:",
            *predicate_lines,
            f"Observation module present: {'yes' if domain_analysis.has_observation_module else 'no'}",
        ]
    )


def _predicate_mentions_action_constant(predicate: Predicate, action_constant_names: set[str]) -> bool:
    return any(argument in action_constant_names for argument in predicate.params)


def _conjunction_for_assignment(assignment: dict[Predicate, bool]) -> str:
    literals = []
    for predicate, value in sorted(assignment.items(), key=lambda item: item[0].to_pddl_str()):
        if value:
            literals.append(predicate.to_pddl_str())
        else:
            literals.append(f"(not {predicate.to_pddl_str()})")
    if not literals:
        return "(and)"
    if len(literals) == 1:
        return literals[0]
    return f"(and {' '.join(literals)})"


def _flatten_and_literals(expr: SExpr) -> list[SExpr]:
    if isinstance(expr, list) and expr and expr[0] == "and":
        return list(expr[1:])
    return [expr]


def _combine_assignment_with_goal_state(
    assignment: dict[Predicate, bool],
    goal_state: SExpr,
) -> SExpr:
    conjuncts: list[SExpr] = []
    assignment_expr = _parse_goal_expr(_conjunction_for_assignment(assignment))
    conjuncts.extend(_flatten_and_literals(assignment_expr))
    conjuncts.extend(_flatten_and_literals(goal_state))
    if not conjuncts:
        return ["and"]
    if len(conjuncts) == 1:
        return conjuncts[0]
    return ["and", *conjuncts]


@dataclass(frozen=True)
class GoalAssignmentEvaluation:
    assignment_id: str
    assignment: dict[Predicate, bool]
    satisfies_instruction: bool
    rationale: str | None = None


class GoalInferenceAgent:
    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        verbose: bool = False,
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
            default_temperature=0.1,
        )
        self.model = resolved.model
        self.base_url = resolved.base_url
        self.api_key = resolved.api_key
        self.temperature = resolved.temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        if inference_strategy not in {"batch", "parallel"}:
            raise ValueError("inference_strategy must be 'batch' or 'parallel'")
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        self.inference_strategy = inference_strategy
        self.inference_batch_size = inference_batch_size
        self._predicate_prompt = _load_prompt("relevant_predicates.md")
        self._ground_atoms_prompt = _load_prompt("relevant_ground_atoms.md")
        self._mutex_groups_prompt = _load_prompt("mutex_groups.md")
        self._semantic_prune_prompt = _load_prompt("semantic_pruning.md")
        self._goal_state_prompt = _load_prompt("assignment_satisfaction.md")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[goal_generation] {message}", flush=True)

    def infer_goal_expr(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        instruction: str,
        objects: list[ObjectDeclaration],
        max_workers: int = 8,
    ) -> SExpr:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive.")

        relevant_predicate_names = self._select_goal_relevant_predicate_names(
            domain_analysis=domain_analysis,
            instruction=instruction,
            objects=objects,
        )
        self._log("Selected relevant predicate schemas: " + ", ".join(relevant_predicate_names))
        relevant_grounded_atoms = self._select_goal_relevant_ground_atoms(
            domain_analysis=domain_analysis,
            instruction=instruction,
            objects=objects,
            relevant_predicate_names=relevant_predicate_names,
        )
        if not relevant_grounded_atoms:
            raise ValueError("Goal generation selected zero relevant grounded predicates.")
        self._log(
            "Selected relevant grounded predicates: "
            + ", ".join(predicate.to_pddl_str() for predicate in relevant_grounded_atoms)
        )
        mutex_groups = self._select_goal_mutex_groups(
            domain_analysis=domain_analysis,
            instruction=instruction,
            objects=objects,
            grounded_predicates=relevant_grounded_atoms,
        )
        if mutex_groups:
            rendered_groups = [
                "{" + ", ".join(predicate.to_pddl_str() for predicate in group) + "}" for group in mutex_groups
            ]
            self._log("Selected mutex groups: " + "; ".join(rendered_groups))
        else:
            self._log("Selected mutex groups: none")
        relevant_grounded_atoms, mutex_groups, semantically_pruned_atoms = (
            self._prune_semantically_invalid_ground_atoms(
                domain_analysis=domain_analysis,
                instruction=instruction,
                objects=objects,
                grounded_predicates=relevant_grounded_atoms,
                mutex_groups=mutex_groups,
            )
        )
        if semantically_pruned_atoms:
            self._log(
                "Pruned semantically invalid grounded predicates before enumeration: "
                + ", ".join(predicate.to_pddl_str() for predicate in semantically_pruned_atoms)
            )
        else:
            self._log("Pruned semantically invalid grounded predicates before enumeration: none")
        if not relevant_grounded_atoms:
            raise ValueError("Goal generation pruned all relevant grounded predicates before assignment enumeration.")

        assignments = self._enumerate_grounded_assignments(
            relevant_grounded_atoms,
            mutex_groups=mutex_groups,
        )
        self._log(f"Enumerated {len(assignments)} goal assignments for instruction checking after mutex filtering")
        evaluations = self._evaluate_goal_assignments(
            domain_analysis=domain_analysis,
            instruction=instruction,
            objects=objects,
            assignments=assignments,
            max_workers=max_workers,
        )
        minimized_assignments, irrelevant_atoms = self._project_semantically_irrelevant_atoms(
            grounded_predicates=relevant_grounded_atoms,
            evaluations=evaluations,
        )
        if irrelevant_atoms:
            self._log(
                "Removed predicates that do not affect instruction satisfaction: "
                + ", ".join(predicate.to_pddl_str() for predicate in irrelevant_atoms)
            )
        valid_goal_states: list[SExpr] = []
        seen_goal_pddl: set[str] = set()
        for assignment in minimized_assignments:
            combined_goal_state = _parse_goal_expr(_conjunction_for_assignment(assignment))
            rendered = _render_goal_expr(combined_goal_state)
            if rendered in seen_goal_pddl:
                continue
            seen_goal_pddl.add(rendered)
            valid_goal_states.append(combined_goal_state)
        if not valid_goal_states:
            raise ValueError("Goal agent could not construct any satisfying goal state from enumerated assignments.")
        self._log(f"Instruction satisfied by {len(valid_goal_states)} distinct goal assignments")
        if len(valid_goal_states) == 1:
            return valid_goal_states[0]
        return ["or", *valid_goal_states]

    def _select_goal_relevant_predicate_names(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        instruction: str,
        objects: list[ObjectDeclaration],
    ) -> list[str]:
        candidate_predicates = []
        for predicate in domain_analysis.parsed_domain.predicates:
            if _is_observation_helper_predicate(predicate.name):
                continue
            parameter_types = domain_analysis.parsed_domain.predicate_parameter_types.get(predicate, [])
            candidate_predicates.append(
                {
                    "predicate_name": predicate.name,
                    "signature": predicate.to_pddl_str(),
                    "parameter_types": [type_name for _name, type_name in parameter_types],
                }
            )
        payload = {
            "domain_summary": _render_goal_domain_summary(domain_analysis),
            "instruction": instruction.strip(),
            "objects": [{"name": item.name, "type_name": item.type_name} for item in objects],
            "candidate_predicates": candidate_predicates,
        }
        client = make_client(api_key=self.api_key, base_url=self.base_url)
        reply = safe_chat(
            client,
            self._predicate_prompt,
            build_user_content(text=json.dumps(payload, ensure_ascii=False, indent=2)),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        selected = data.get("goal_relevant_predicates", [])
        if not isinstance(selected, list) or not selected:
            raise ValueError("Goal predicate selector must return a non-empty goal_relevant_predicates list.")
        available = {item["predicate_name"] for item in candidate_predicates}
        normalized = []
        for item in selected:
            name = str(item).strip()
            if name not in available:
                raise ValueError(f"Goal predicate selector returned unknown predicate {name!r}.")
            if name not in normalized:
                normalized.append(name)
        return normalized

    def _select_goal_relevant_ground_atoms(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        instruction: str,
        objects: list[ObjectDeclaration],
        relevant_predicate_names: list[str],
    ) -> list[Predicate]:
        grounded_predicates = sorted(
            build_grounded_predicates_for_objects(
                domain_analysis.parsed_domain,
                objects,
            ),
            key=lambda predicate: predicate.to_pddl_str(),
        )
        action_constant_names = _action_constant_names(domain_analysis)
        if action_constant_names:
            self._log(
                "Filtering grounded predicates that mention action constants: "
                + ", ".join(sorted(action_constant_names))
            )
        coarse_candidates = [
            predicate
            for predicate in grounded_predicates
            if predicate.name in set(relevant_predicate_names)
            and not _predicate_mentions_action_constant(predicate, action_constant_names)
        ]
        if not coarse_candidates:
            return []

        payload = {
            "domain_summary": _render_goal_domain_summary(domain_analysis),
            "instruction": instruction.strip(),
            "objects": [{"name": item.name, "type_name": item.type_name} for item in objects],
            "goal_relevant_predicates": relevant_predicate_names,
            "candidate_grounded_predicates": [predicate.to_pddl_str() for predicate in coarse_candidates],
        }
        client = make_client(api_key=self.api_key, base_url=self.base_url)
        reply = safe_chat(
            client,
            self._ground_atoms_prompt,
            build_user_content(text=json.dumps(payload, ensure_ascii=False, indent=2)),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        selected = data.get("goal_relevant_grounded_predicates", [])
        if not isinstance(selected, list) or not selected:
            raise ValueError(
                "Goal grounded-predicate selector must return a non-empty goal_relevant_grounded_predicates list."
            )
        available = {predicate.to_pddl_str(): predicate for predicate in coarse_candidates}
        normalized: list[Predicate] = []
        seen: set[str] = set()
        for item in selected:
            predicate_text = str(item).strip()
            if predicate_text not in available:
                raise ValueError(
                    f"Goal grounded-predicate selector returned unknown grounded predicate {predicate_text!r}."
                )
            if predicate_text in seen:
                continue
            seen.add(predicate_text)
            normalized.append(available[predicate_text])
        return normalized

    def _select_goal_mutex_groups(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        instruction: str,
        objects: list[ObjectDeclaration],
        grounded_predicates: list[Predicate],
    ) -> list[list[Predicate]]:
        if len(grounded_predicates) < 2:
            return []
        payload = {
            "domain_summary": _render_goal_domain_summary(domain_analysis),
            "instruction": instruction.strip(),
            "objects": [{"name": item.name, "type_name": item.type_name} for item in objects],
            "candidate_mutex_group_predicates": [predicate.to_pddl_str() for predicate in grounded_predicates],
        }
        client = make_client(api_key=self.api_key, base_url=self.base_url)
        reply = safe_chat(
            client,
            self._mutex_groups_prompt,
            build_user_content(text=json.dumps(payload, ensure_ascii=False, indent=2)),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        raw_groups = data.get("mutex_groups", [])
        if not isinstance(raw_groups, list):
            raise ValueError("Goal mutex selector must return mutex_groups as a list.")
        available = {predicate.to_pddl_str(): predicate for predicate in grounded_predicates}
        normalized_groups: list[list[Predicate]] = []
        seen_groups: set[tuple[str, ...]] = set()
        for raw_group in raw_groups:
            if not isinstance(raw_group, list):
                raise ValueError("Each mutex group must be a list of grounded predicates.")
            group_predicates: list[Predicate] = []
            seen_members: set[str] = set()
            for item in raw_group:
                predicate_text = str(item).strip()
                if predicate_text not in available:
                    raise ValueError(f"Goal mutex selector returned unknown grounded predicate {predicate_text!r}.")
                if predicate_text in seen_members:
                    continue
                seen_members.add(predicate_text)
                group_predicates.append(available[predicate_text])
            if len(group_predicates) < 2:
                continue
            group_key = tuple(sorted(predicate.to_pddl_str() for predicate in group_predicates))
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            normalized_groups.append(sorted(group_predicates, key=lambda predicate: predicate.to_pddl_str()))
        return normalized_groups

    def _prune_semantically_invalid_ground_atoms(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        instruction: str,
        objects: list[ObjectDeclaration],
        grounded_predicates: list[Predicate],
        mutex_groups: list[list[Predicate]],
    ) -> tuple[list[Predicate], list[list[Predicate]], list[Predicate]]:
        if not grounded_predicates:
            return [], [], []
        payload = {
            "domain_summary": _render_goal_domain_summary(domain_analysis),
            "instruction": instruction.strip(),
            "objects": [{"name": item.name, "type_name": item.type_name} for item in objects],
            "candidate_grounded_predicates": [predicate.to_pddl_str() for predicate in grounded_predicates],
            "selected_mutex_groups": [[predicate.to_pddl_str() for predicate in group] for group in mutex_groups],
        }
        client = make_client(api_key=self.api_key, base_url=self.base_url)
        reply = safe_chat(
            client,
            self._semantic_prune_prompt,
            build_user_content(text=json.dumps(payload, ensure_ascii=False, indent=2)),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        raw_predicates = data.get("pruned_grounded_predicates", [])
        if not isinstance(raw_predicates, list):
            raise ValueError("Goal semantic-prune selector must return pruned_grounded_predicates as a list.")
        available = {predicate.to_pddl_str(): predicate for predicate in grounded_predicates}
        pruned_atoms: list[Predicate] = []
        seen: set[str] = set()
        for item in raw_predicates:
            predicate_text = str(item).strip()
            if predicate_text not in available:
                raise ValueError(
                    f"Goal semantic-prune selector returned unknown grounded predicate {predicate_text!r}."
                )
            if predicate_text in seen:
                continue
            seen.add(predicate_text)
            pruned_atoms.append(available[predicate_text])
        pruned_set = set(pruned_atoms)
        filtered_atoms = [predicate for predicate in grounded_predicates if predicate not in pruned_set]
        filtered_mutex_groups: list[list[Predicate]] = []
        seen_group_keys: set[tuple[str, ...]] = set()
        for group in mutex_groups:
            filtered_group = [predicate for predicate in group if predicate not in pruned_set]
            if len(filtered_group) < 2:
                continue
            group_key = tuple(sorted(predicate.to_pddl_str() for predicate in filtered_group))
            if group_key in seen_group_keys:
                continue
            seen_group_keys.add(group_key)
            filtered_mutex_groups.append(filtered_group)
        return filtered_atoms, filtered_mutex_groups, pruned_atoms

    def _enumerate_grounded_assignments(
        self,
        grounded_predicates: list[Predicate],
        *,
        mutex_groups: list[list[Predicate]] | None = None,
    ) -> list[dict[Predicate, bool]]:
        assignments: list[dict[Predicate, bool]] = []
        mutex_groups = mutex_groups or []
        group_sets = [set(group) for group in mutex_groups]
        for values in itertools.product([False, True], repeat=len(grounded_predicates)):
            assignment = {predicate: value for predicate, value in zip(grounded_predicates, values, strict=True)}
            if any(sum(1 for predicate in group if assignment.get(predicate, False)) > 1 for group in group_sets):
                continue
            assignments.append(assignment)
        return assignments

    def _evaluate_goal_assignments(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        instruction: str,
        objects: list[ObjectDeclaration],
        assignments: list[dict[Predicate, bool]],
        max_workers: int,
    ) -> list[GoalAssignmentEvaluation]:
        return self._evaluate_goal_assignments_in_chunks(
            domain_analysis=domain_analysis,
            instruction=instruction,
            objects=objects,
            assignments=assignments,
            max_workers=max_workers,
        )

    def _evaluate_goal_assignments_in_chunks(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        instruction: str,
        objects: list[ObjectDeclaration],
        assignments: list[dict[Predicate, bool]],
        max_workers: int,
    ) -> list[GoalAssignmentEvaluation]:
        indexed_assignments = [
            (f"assignment_{index + 1:04d}", assignment) for index, assignment in enumerate(assignments)
        ]
        if not indexed_assignments:
            return []
        chunks = [
            indexed_assignments[index : index + self.inference_batch_size]
            for index in range(0, len(indexed_assignments), self.inference_batch_size)
        ]
        self._log(
            f"Evaluating {len(indexed_assignments)} goal assignments in {len(chunks)} "
            f"chunk(s) of at most {self.inference_batch_size} using {self.inference_strategy} scheduling"
        )

        def _evaluate(
            chunk: list[tuple[str, dict[Predicate, bool]]],
        ) -> list[GoalAssignmentEvaluation]:
            return self._request_goal_assignment_evaluations(
                domain_analysis=domain_analysis,
                instruction=instruction,
                objects=objects,
                indexed_assignments=chunk,
            )

        if self.inference_strategy == "batch" or len(chunks) == 1:
            chunk_results = [_evaluate(chunk) for chunk in chunks]
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as executor:
                chunk_results = list(executor.map(_evaluate, chunks))
        return [evaluation for chunk in chunk_results for evaluation in chunk]

    def _request_goal_assignment_evaluations(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        instruction: str,
        objects: list[ObjectDeclaration],
        indexed_assignments: list[tuple[str, dict[Predicate, bool]]],
    ) -> list[GoalAssignmentEvaluation]:
        total = len(indexed_assignments)
        payload = {
            "domain_summary": _render_goal_domain_summary(domain_analysis),
            "instruction": instruction.strip(),
            "objects": [{"name": item.name, "type_name": item.type_name} for item in objects],
            "assignments": [
                {
                    "assignment_id": assignment_id,
                    "grounded_goal_relevant_assignment": [
                        {
                            "predicate": predicate.to_pddl_str(),
                            "truth_value": value,
                        }
                        for predicate, value in sorted(assignment.items(), key=lambda item: item[0].to_pddl_str())
                    ],
                    "assignment_conjunction_pddl": _conjunction_for_assignment(assignment),
                }
                for assignment_id, assignment in indexed_assignments
            ],
        }
        client = make_client(api_key=self.api_key, base_url=self.base_url)
        last_error: str | None = None
        assignment_by_id = dict(indexed_assignments)
        for attempt in range(2):
            if attempt:
                payload["correction"] = (
                    "The previous response was structurally invalid. Return every assignment_id "
                    "exactly once in the original order, without additions or omissions."
                )
            reply = safe_chat(
                client,
                self._goal_state_prompt,
                build_user_content(text=json.dumps(payload, ensure_ascii=False, indent=2)),
                model=self.model,
                temperature=self.temperature,
                max_tokens=max(self.max_tokens, 4096, 512 + total * 128),
                verbose=self.verbose,
            )
            data = extract_json_object(reply)
            rows = data.get("assignment_evaluations", [])
            if not isinstance(rows, list):
                last_error = "response did not contain an assignment_evaluations list"
                continue
            parsed: dict[str, GoalAssignmentEvaluation] = {}
            invalid_ids: list[str] = []
            for row in rows:
                if not isinstance(row, dict):
                    invalid_ids.append("<non-object>")
                    continue
                assignment_id = str(row.get("assignment_id", "")).strip()
                satisfies_instruction = row.get("satisfies_instruction")
                if (
                    assignment_id not in assignment_by_id
                    or assignment_id in parsed
                    or not isinstance(satisfies_instruction, bool)
                ):
                    invalid_ids.append(assignment_id)
                    continue
                parsed[assignment_id] = GoalAssignmentEvaluation(
                    assignment_id=assignment_id,
                    assignment=dict(assignment_by_id[assignment_id]),
                    satisfies_instruction=satisfies_instruction,
                    rationale=str(row.get("reasoning", "")).strip() or None,
                )
            missing_ids = [assignment_id for assignment_id, _assignment in indexed_assignments if assignment_id not in parsed]
            if invalid_ids or missing_ids or len(rows) != total:
                last_error = (
                    f"invalid={invalid_ids or 'none'}, missing={missing_ids or 'none'}, "
                    f"expected_count={total}, actual_count={len(rows)}"
                )
                continue
            return [parsed[assignment_id] for assignment_id, _assignment in indexed_assignments]
        raise ValueError(f"Invalid batched goal-state evaluation after retry: {last_error}.")

    @staticmethod
    def _project_semantically_irrelevant_atoms(
        *,
        grounded_predicates: list[Predicate],
        evaluations: list[GoalAssignmentEvaluation],
    ) -> tuple[list[dict[Predicate, bool]], list[Predicate]]:
        """Remove variables whose value never changes instruction satisfaction."""
        if not evaluations:
            return [], []
        satisfaction_by_key = {
            tuple(
                evaluation.assignment[predicate] for predicate in grounded_predicates
            ): evaluation.satisfies_instruction
            for evaluation in evaluations
        }
        irrelevant: list[Predicate] = []
        for index, predicate in enumerate(grounded_predicates):
            is_relevant = False
            for assignment_key, satisfies in satisfaction_by_key.items():
                flipped = list(assignment_key)
                flipped[index] = not flipped[index]
                flipped_key = tuple(flipped)
                if flipped_key in satisfaction_by_key and satisfaction_by_key[flipped_key] != satisfies:
                    is_relevant = True
                    break
            if not is_relevant:
                irrelevant.append(predicate)

        irrelevant_set = set(irrelevant)
        projected: list[dict[Predicate, bool]] = []
        seen: set[tuple[tuple[str, bool], ...]] = set()
        for evaluation in evaluations:
            if not evaluation.satisfies_instruction:
                continue
            assignment = {
                predicate: value
                for predicate, value in evaluation.assignment.items()
                if predicate not in irrelevant_set
            }
            key = tuple(
                sorted(
                    (
                        predicate.to_pddl_str(),
                        value,
                    )
                    for predicate, value in assignment.items()
                )
            )
            if key in seen:
                continue
            seen.add(key)
            projected.append(assignment)
        return projected, irrelevant
