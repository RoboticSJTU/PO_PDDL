from __future__ import annotations

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from po_pddl.domain_generation.infrastructure.artifact_io import load_json, load_json_object
from po_pddl.domain_generation.infrastructure.fact_utils import parse_symbolic_literal
from po_pddl.domain_generation.infrastructure.type_hierarchy import build_type_parent_map
from po_pddl.domain_generation.stages.manipulation_domain_learning.learner import load_raw_trajectory_steps
from po_pddl.domain_generation.stages.manipulation_domain_learning.models import PredicateSchema

from .models import (
    ConfirmedContradiction,
    DescriptionReviewResult,
    InitObservationCondition,
    InitObservationExample,
    InitObservationLearningResult,
    InitObservationRuleSchema,
    InitObservationSourceRecord,
    InitObservationUncertainPredicateDiscovery,
    VLMConfirmationResult,
)
from .modules import (
    InitDescriptionContradictionReviewModule,
    InitObservationUncertainPredicateDiscoveryModule,
    InitVLMContradictionConfirmationModule,
)
from .renderer import render_init_observation_module

EXCLUDED_INIT_OBSERVATION_PREDICATES = {"gripper_empty", "gripper_holding"}
_LAST_ACTION_RE = re.compile(r"\(\s*last_action_(\d+)_param\b")
_INIT_OBSERVATION_SUMMARY_FILENAME = "init_observation_learning_summary.json"


def _surface_name(name: str) -> str:
    return str(name).replace("_", " ").strip().lower()


def _predicate_name(literal: str) -> str:
    return parse_symbolic_literal(literal)[1]


def _literal_arguments(literal: str) -> list[str]:
    return parse_symbolic_literal(literal)[2]


def _instantiate_template(predicate_name: str, parameter_names: list[str]) -> str:
    return f"{predicate_name}({','.join(parameter_names)})" if parameter_names else f"{predicate_name}()"


def _target_literal_template_for_predicate(predicate_schema: PredicateSchema) -> str:
    return _instantiate_template(
        predicate_schema.predicate_name,
        [f"?obs{index}" for index in range(len(predicate_schema.parameter_types))],
    )


def _object_matches_type(
    object_type: str,
    required_type: str,
    parent_by_type: dict[str, str],
) -> bool:
    if required_type == "object":
        return True
    current = object_type
    seen: set[str] = set()
    while current and current not in seen:
        if current == required_type:
            return True
        seen.add(current)
        current = parent_by_type.get(current, "")
    return False


def _filter_init_facts(facts: list[str]) -> list[str]:
    filtered: list[str] = []
    for fact in facts:
        _negated, predicate, _arguments = parse_symbolic_literal(fact)
        if predicate in EXCLUDED_INIT_OBSERVATION_PREDICATES:
            continue
        filtered.append(fact)
    return sorted(dict.fromkeys(filtered))


def _build_complete_ground_truth_assignments(
    *,
    objects: dict[str, str],
    predicate_inventory: list[PredicateSchema],
    init_facts: list[str],
    parent_by_type: dict[str, str],
) -> list[str]:
    true_facts = set(_filter_init_facts(init_facts))
    assignments: list[str] = []
    for predicate in predicate_inventory:
        if predicate.predicate_name in EXCLUDED_INIT_OBSERVATION_PREDICATES:
            continue
        parameter_types = list(predicate.parameter_types)
        if not parameter_types:
            positive_literal = f"{predicate.predicate_name}()"
            assignments.append(positive_literal if positive_literal in true_facts else f"not {positive_literal}")
            continue
        candidate_object_lists: list[list[str]] = []
        for required_type in parameter_types:
            compatible_names = sorted(
                object_name
                for object_name, object_type in objects.items()
                if _object_matches_type(object_type, required_type, parent_by_type)
            )
            if not compatible_names:
                candidate_object_lists = []
                break
            candidate_object_lists.append(compatible_names)
        if not candidate_object_lists:
            continue
        tuples: list[list[str]] = [[]]
        for names in candidate_object_lists:
            next_rows: list[list[str]] = []
            for prefix in tuples:
                for name in names:
                    next_rows.append([*prefix, name])
            tuples = next_rows
        for argument_values in tuples:
            positive_literal = _instantiate_template(predicate.predicate_name, argument_values)
            assignments.append(positive_literal if positive_literal in true_facts else f"not {positive_literal}")
    return sorted(dict.fromkeys(assignments))


def _normalize_confirmed_contradictions(
    contradictions: list[ConfirmedContradiction],
    *,
    candidate_ground_truth_facts: list[str],
    predicate_by_name: dict[str, PredicateSchema],
    objects: dict[str, str],
    parent_by_type: dict[str, str],
    allowed_predicate_names: set[str] | None = None,
) -> list[ConfirmedContradiction]:
    ground_truth_by_literal: dict[str, bool] = {}
    for assignment in candidate_ground_truth_facts:
        negated, predicate_name, arguments = parse_symbolic_literal(assignment)
        ground_truth_by_literal[_instantiate_template(predicate_name, arguments)] = not negated

    normalized: list[ConfirmedContradiction] = []
    seen: set[str] = set()
    for contradiction in contradictions:
        _negated, predicate_name, arguments = parse_symbolic_literal(contradiction.grounded_literal)
        predicate_schema = predicate_by_name.get(predicate_name)
        if allowed_predicate_names is not None and predicate_name not in allowed_predicate_names:
            continue
        if predicate_schema is None or len(arguments) != len(predicate_schema.parameter_types):
            continue
        if any(
            argument not in objects
            or not _object_matches_type(
                objects[argument],
                required_type,
                parent_by_type,
            )
            for argument, required_type in zip(
                arguments,
                predicate_schema.parameter_types,
            )
        ):
            continue
        grounded_literal = _instantiate_template(predicate_name, arguments)
        ground_truth_value = ground_truth_by_literal.get(grounded_literal)
        if ground_truth_value is None:
            continue
        observed_value = bool(contradiction.observed_value)
        if observed_value == ground_truth_value or grounded_literal in seen:
            continue
        seen.add(grounded_literal)
        normalized.append(
            ConfirmedContradiction(
                predicate_name=predicate_name,
                grounded_literal=grounded_literal,
                observed_value=observed_value,
                ground_truth_value=ground_truth_value,
                rationale=contradiction.rationale,
            )
        )
    return normalized


def _last_action_false_conditions(source_text: str) -> list[str]:
    arities = sorted({int(match.group(1)) for match in _LAST_ACTION_RE.finditer(str(source_text or ""))})
    conditions: list[str] = []
    for arity in arities:
        quantifiers = ["?m - last_action_marker", *[f"?x{index} - object" for index in range(arity)]]
        predicate_terms = " ".join(["?m", *[f"?x{index}" for index in range(arity)]])
        conditions.append(f"(forall ({' '.join(quantifiers)}) (not (last_action_{arity}_param {predicate_terms})))")
    return conditions


def _load_camera_order_by_episode(scene_root: Path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for summary_file in sorted(scene_root.glob("*/scene_description_summary.json")):
        try:
            payload = load_json_object(summary_file)
        except Exception:
            continue
        episode_name = str(payload.get("episode_name") or summary_file.parent.name).strip()
        if not episode_name:
            continue
        raw_order = payload.get("camera_order_top_to_bottom")
        if isinstance(raw_order, list):
            mapping[episode_name] = [str(item) for item in raw_order if str(item).strip()]
    return mapping


@dataclass
class InitObservationLearner:
    uncertain_predicate_discovery_module: InitObservationUncertainPredicateDiscoveryModule
    description_review_module: InitDescriptionContradictionReviewModule
    vlm_confirmation_module: InitVLMContradictionConfirmationModule
    manipulation_artifact_dir: str | Path | None = None
    scene_description_dir: str | Path | None = None
    max_workers: int = 1

    def learn_from_source_records(
        self,
        *,
        source_records: list[InitObservationSourceRecord],
        predicate_inventory: list[PredicateSchema],
        predicate_comments: dict[str, str],
        type_parents: dict[str, str] | None = None,
        last_action_source_text: str | None = None,
    ) -> InitObservationLearningResult:
        parent_by_type = dict(type_parents or {})
        observation_predicate_inventory = [
            item for item in predicate_inventory if item.predicate_name not in EXCLUDED_INIT_OBSERVATION_PREDICATES
        ]
        filtered_predicate_comments = {
            name: comment
            for name, comment in predicate_comments.items()
            if name not in EXCLUDED_INIT_OBSERVATION_PREDICATES
        }
        predicate_by_name = {item.predicate_name: item for item in observation_predicate_inventory}
        last_action_false_conditions = _last_action_false_conditions(last_action_source_text or "")
        confirmed_contradiction_literals_by_record: dict[tuple[str, str], set[str]] = defaultdict(set)

        ordered_records = sorted(source_records, key=lambda item: item.episode_name)
        uncertain_predicate_discovery = (
            self.uncertain_predicate_discovery_module.discover(
                records=ordered_records,
                predicate_inventory=[item.to_dict() for item in observation_predicate_inventory],
                predicate_comments=filtered_predicate_comments,
            )
            if ordered_records
            else InitObservationUncertainPredicateDiscovery(predicate_names=[])
        )
        valid_uncertain_predicate_names = [
            name for name in uncertain_predicate_discovery.predicate_names if name in predicate_by_name
        ]
        uncertain_predicate_discovery = replace(
            uncertain_predicate_discovery,
            predicate_names=valid_uncertain_predicate_names,
            rationale_by_predicate={
                name: rationale
                for name, rationale in uncertain_predicate_discovery.rationale_by_predicate.items()
                if name in valid_uncertain_predicate_names
            },
        )
        review_results: list[dict[str, Any]] = [None] * len(ordered_records)  # type: ignore[list-item]
        if ordered_records:
            worker_count = max(1, min(self.max_workers, len(ordered_records)))
            if worker_count == 1:
                for index, record in enumerate(ordered_records):
                    review_results[index] = self._review_single_record(
                        record=record,
                        predicate_inventory=observation_predicate_inventory,
                        parent_by_type=parent_by_type,
                        predicate_comments=filtered_predicate_comments,
                        uncertain_predicate_names=valid_uncertain_predicate_names,
                    )
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_map = {
                        executor.submit(
                            self._review_single_record,
                            record=record,
                            predicate_inventory=observation_predicate_inventory,
                            parent_by_type=parent_by_type,
                            predicate_comments=filtered_predicate_comments,
                            uncertain_predicate_names=valid_uncertain_predicate_names,
                        ): index
                        for index, record in enumerate(ordered_records)
                    }
                    for future in as_completed(future_map):
                        review_results[future_map[future]] = future.result()

        seed_examples: list[InitObservationExample] = []
        grouped_seed_examples: dict[tuple[str, tuple[str, ...]], list[InitObservationExample]] = defaultdict(list)
        cleaned_review_results: list[dict[str, Any]] = []
        for payload in review_results:
            if payload is None:
                continue
            cleaned_review_results.append(payload["review_summary"])
            record = payload["record"]
            for contradiction in payload["confirmed_review"].confirmed_contradictions:
                confirmed_contradiction_literals_by_record[(record.episode_name, contradiction.predicate_name)].add(
                    contradiction.grounded_literal
                )
                predicate_schema = predicate_by_name.get(contradiction.predicate_name)
                if predicate_schema is None:
                    continue
                example = self._build_seed_example(
                    record=record,
                    contradiction=contradiction,
                    predicate_schema=predicate_schema,
                )
                seed_examples.append(example)
                grouped_seed_examples[(example.predicate_name, tuple(example.argument_types))].append(example)

        conditions: list[InitObservationCondition] = []
        for group_key, examples in sorted(grouped_seed_examples.items()):
            predicate_schema = predicate_by_name.get(group_key[0])
            if predicate_schema is None:
                continue
            examples_for_condition = list(examples)
            if predicate_schema.predicate_name == "in":
                # For containment, exclude hidden-but-true seeds from condition mining.
                # These samples represent world truth that is not visually observable yet,
                # so they should not weaken visibility conditions such as open(drawer).
                examples_for_condition = [
                    item for item in examples_for_condition if not (item.ground_truth_value and not item.observed_value)
                ]
            if not examples_for_condition:
                continue
            conditions.append(
                self._mine_condition(
                    examples=examples_for_condition,
                    predicate_schema=predicate_schema,
                )
            )
        expanded_examples: list[InitObservationExample] = []
        for condition in conditions:
            predicate_schema = predicate_by_name.get(condition.predicate_name)
            if predicate_schema is None:
                continue
            for record in ordered_records:
                expanded_examples.extend(
                    self._expand_examples_for_record(
                        record=record,
                        condition=condition,
                        predicate_schema=predicate_schema,
                        parent_by_type=parent_by_type,
                        confirmed_contradiction_literals=confirmed_contradiction_literals_by_record.get(
                            (record.episode_name, condition.predicate_name),
                            set(),
                        ),
                    )
                )

        schemas: list[InitObservationRuleSchema] = []
        for condition in conditions:
            predicate_schema = predicate_by_name.get(condition.predicate_name)
            if predicate_schema is None:
                continue
            related_examples = [
                item
                for item in [*seed_examples, *expanded_examples]
                if item.predicate_name == condition.predicate_name and item.argument_types == condition.argument_types
            ]
            if condition.predicate_name == "in":
                # Keep the containment observation model aligned with its mined
                # visibility conditions by excluding hidden-but-true positives
                # from probability estimation as well.
                related_examples = [
                    item for item in related_examples if not (item.ground_truth_value and not item.observed_value)
                ]
            if not related_examples:
                continue
            schemas.append(
                self._synthesize_schema(
                    condition=condition,
                    predicate_schema=predicate_schema,
                    examples=related_examples,
                    last_action_false_conditions=last_action_false_conditions,
                )
            )

        rendered_module_text = render_init_observation_module(schemas)
        return InitObservationLearningResult(
            source_records=source_records,
            review_results=cleaned_review_results,
            seed_examples=seed_examples,
            conditions=conditions,
            expanded_examples=expanded_examples,
            schemas=schemas,
            predicate_inventory=observation_predicate_inventory,
            predicate_comments=filtered_predicate_comments,
            rendered_module_text=rendered_module_text,
            uncertain_predicate_discovery=uncertain_predicate_discovery,
        )

    def _review_single_record(
        self,
        *,
        record: InitObservationSourceRecord,
        predicate_inventory: list[PredicateSchema],
        parent_by_type: dict[str, str],
        predicate_comments: dict[str, str],
        uncertain_predicate_names: list[str],
    ) -> dict[str, Any]:
        filtered_init_facts = _filter_init_facts(record.init_facts)
        candidate_ground_truth_facts = _build_complete_ground_truth_assignments(
            objects=record.objects,
            predicate_inventory=predicate_inventory,
            init_facts=record.init_facts,
            parent_by_type=parent_by_type,
        )
        uncertain_name_set = set(uncertain_predicate_names)
        candidate_ground_truth_facts = [
            assignment
            for assignment in candidate_ground_truth_facts
            if _predicate_name(assignment) in uncertain_name_set
        ]
        if not filtered_init_facts or not candidate_ground_truth_facts:
            description_review = DescriptionReviewResult(
                should_review_with_vlm=False,
                suspect_contradictions=[],
                summary=(
                    "Skipped because this episode has no grounded assignments for "
                    "the VLM-discovered uncertain predicates."
                ),
                raw_output=None,
            )
            confirmed_review = VLMConfirmationResult(
                confirmed_contradictions=[],
                summary="No VLM-discovered uncertain predicates apply to this episode.",
                raw_output=None,
            )
        else:
            description_review = DescriptionReviewResult(
                should_review_with_vlm=True,
                suspect_contradictions=[],
                summary=(
                    "Text-only gating disabled; the VLM independently reviews all grounded predicate assignments."
                ),
                raw_output=None,
            )
            confirmed_review = self.vlm_confirmation_module.confirm(
                record=record,
                filtered_init_facts=filtered_init_facts,
                candidate_ground_truth_facts=candidate_ground_truth_facts,
                suspects=[],
                predicate_comments=predicate_comments,
                uncertain_predicate_names=uncertain_predicate_names,
            )
            confirmed_review = replace(
                confirmed_review,
                confirmed_contradictions=_normalize_confirmed_contradictions(
                    confirmed_review.confirmed_contradictions,
                    candidate_ground_truth_facts=candidate_ground_truth_facts,
                    predicate_by_name={item.predicate_name: item for item in predicate_inventory},
                    objects=record.objects,
                    parent_by_type=parent_by_type,
                    allowed_predicate_names=uncertain_name_set,
                ),
            )
        return {
            "record": record,
            "confirmed_review": confirmed_review,
            "review_summary": {
                "episode_name": record.episode_name,
                "description_review": description_review.to_dict(),
                "vlm_confirmation": confirmed_review.to_dict(),
            },
        }

    def learn_from_pairs(
        self,
        *,
        domain_file: str | Path,
        episode_grounding_pairs: list[tuple[str | Path, str | Path]],
    ) -> InitObservationLearningResult:
        if self.manipulation_artifact_dir is None or self.scene_description_dir is None:
            raise ValueError(
                "InitObservationLearner requires manipulation_artifact_dir and scene_description_dir."
            )
        source_records, predicate_inventory, predicate_comments, parent_by_type = load_init_observation_inputs(
            manipulation_artifact_dir=self.manipulation_artifact_dir,
            scene_description_dir=self.scene_description_dir,
            episode_grounding_pairs=episode_grounding_pairs,
        )
        last_action_source_text = Path(domain_file).read_text(encoding="utf-8") if Path(domain_file).exists() else ""
        return self.learn_from_source_records(
            source_records=source_records,
            predicate_inventory=predicate_inventory,
            predicate_comments=predicate_comments,
            type_parents=parent_by_type,
            last_action_source_text=last_action_source_text,
        )

    def _build_seed_example(
        self,
        *,
        record: InitObservationSourceRecord,
        contradiction: ConfirmedContradiction,
        predicate_schema: PredicateSchema,
    ) -> InitObservationExample:
        literal_arguments = _literal_arguments(contradiction.grounded_literal)
        argument_types = [
            record.objects.get(
                argument,
                predicate_schema.parameter_types[index] if index < len(predicate_schema.parameter_types) else "object",
            )
            for index, argument in enumerate(literal_arguments)
        ]
        return InitObservationExample(
            episode_name=record.episode_name,
            predicate_name=contradiction.predicate_name,
            grounded_literal=contradiction.grounded_literal,
            argument_values=list(literal_arguments),
            argument_types=argument_types,
            ground_truth_value=contradiction.ground_truth_value,
            observed_value=contradiction.observed_value,
            init_facts=list(record.init_facts),
            source_kind="contradiction_seed",
        )

    def _mine_condition(
        self,
        *,
        examples: list[InitObservationExample],
        predicate_schema: PredicateSchema,
    ) -> InitObservationCondition:
        first = examples[0]
        binding = {argument: f"?obs{index}" for index, argument in enumerate(first.argument_values)}
        target_literal_template = _instantiate_template(
            first.predicate_name,
            [binding.get(argument, argument) for argument in first.argument_values],
        )
        intersections: set[str] | None = None
        for example in examples:
            example_binding = {argument: f"?obs{index}" for index, argument in enumerate(example.argument_values)}
            candidate_literals: set[str] = set()
            for fact in _filter_init_facts(example.init_facts):
                predicate_name = _predicate_name(fact)
                if predicate_name == example.predicate_name:
                    continue
                literal_arguments = _literal_arguments(fact)
                if literal_arguments and not any(argument in example.argument_values for argument in literal_arguments):
                    continue
                if not all(argument in example_binding for argument in literal_arguments):
                    continue
                candidate_literals.add(
                    _instantiate_template(
                        predicate_name,
                        [example_binding[argument] for argument in literal_arguments],
                    )
                )
            intersections = (
                candidate_literals if intersections is None else intersections.intersection(candidate_literals)
            )
        return InitObservationCondition(
            predicate_name=first.predicate_name,
            argument_types=list(predicate_schema.parameter_types),
            target_literal_template=target_literal_template,
            condition_literals=sorted(intersections or set()),
        )

    def _expand_examples_for_record(
        self,
        *,
        record: InitObservationSourceRecord,
        condition: InitObservationCondition,
        predicate_schema: PredicateSchema,
        parent_by_type: dict[str, str],
        confirmed_contradiction_literals: set[str],
    ) -> list[InitObservationExample]:
        candidate_object_lists: list[list[str]] = []
        for required_type in predicate_schema.parameter_types:
            candidate_object_lists.append(
                sorted(
                    object_name
                    for object_name, object_type in record.objects.items()
                    if _object_matches_type(object_type, required_type, parent_by_type)
                )
            )
        candidate_bindings: list[list[str]] = [[]]
        for object_names in candidate_object_lists:
            next_bindings: list[list[str]] = []
            for prefix in candidate_bindings:
                for object_name in object_names:
                    next_bindings.append([*prefix, object_name])
            candidate_bindings = next_bindings

        expanded: list[InitObservationExample] = []
        filtered_init_facts = _filter_init_facts(record.init_facts)
        init_fact_set = set(filtered_init_facts)
        for argument_values in candidate_bindings:
            if not self._condition_holds(
                init_facts=filtered_init_facts,
                condition_literals=condition.condition_literals,
                argument_values=argument_values,
            ):
                continue
            grounded_literal = _instantiate_template(condition.predicate_name, argument_values)
            if grounded_literal in confirmed_contradiction_literals:
                continue
            ground_truth_value = grounded_literal in init_fact_set
            expanded.append(
                InitObservationExample(
                    episode_name=record.episode_name,
                    predicate_name=condition.predicate_name,
                    grounded_literal=grounded_literal,
                    argument_values=list(argument_values),
                    argument_types=list(predicate_schema.parameter_types),
                    ground_truth_value=ground_truth_value,
                    observed_value=ground_truth_value,
                    init_facts=list(record.init_facts),
                    source_kind="expanded_correct_observation",
                )
            )
        return expanded

    def _synthesize_schema(
        self,
        *,
        condition: InitObservationCondition,
        predicate_schema: PredicateSchema,
        examples: list[InitObservationExample],
        last_action_false_conditions: list[str],
    ) -> InitObservationRuleSchema:
        observable_name = f"obs_{condition.predicate_name}"
        true_examples = [item for item in examples if item.ground_truth_value]
        false_examples = [item for item in examples if not item.ground_truth_value]
        observed_true_when_true = sum(1 for item in true_examples if item.observed_value)
        observed_false_when_true = len(true_examples) - observed_true_when_true
        observed_true_when_false = sum(1 for item in false_examples if item.observed_value)
        observed_false_when_false = len(false_examples) - observed_true_when_false
        prob_true_given_true = observed_true_when_true / len(true_examples) if true_examples else 0.0
        prob_false_given_true = observed_false_when_true / len(true_examples) if true_examples else 1.0
        prob_true_given_false = observed_true_when_false / len(false_examples) if false_examples else 0.0
        prob_false_given_false = observed_false_when_false / len(false_examples) if false_examples else 1.0
        if condition.predicate_name == "in":
            # For containment, when ground truth is false we deterministically emit
            # not-obs_in instead of estimating this side from sparse data.
            observed_true_when_false = 0
            observed_false_when_false = len(false_examples)
            prob_true_given_false = 0.0
            prob_false_given_false = 1.0
        return InitObservationRuleSchema(
            predicate_name=condition.predicate_name,
            observable_name=observable_name,
            parameter_types=list(predicate_schema.parameter_types),
            target_literal_template=condition.target_literal_template,
            condition_literals=list(condition.condition_literals),
            last_action_false_conditions=list(last_action_false_conditions),
            true_rule_name=f"init_obs_{condition.predicate_name}_true",
            false_rule_name=f"init_obs_{condition.predicate_name}_false",
            total_ground_truth_true=len(true_examples),
            observed_true_when_ground_truth_true=observed_true_when_true,
            observed_false_when_ground_truth_true=observed_false_when_true,
            prob_observable_true_given_ground_truth_true=prob_true_given_true,
            prob_observable_false_given_ground_truth_true=prob_false_given_true,
            total_ground_truth_false=len(false_examples),
            observed_true_when_ground_truth_false=observed_true_when_false,
            observed_false_when_ground_truth_false=observed_false_when_false,
            prob_observable_true_given_ground_truth_false=prob_true_given_false,
            prob_observable_false_given_ground_truth_false=prob_false_given_false,
        )

    def write_outputs(self, result: InitObservationLearningResult, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "init_observation_learning_summary.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_path / "init_observation_module.pddl").write_text(
            result.rendered_module_text,
            encoding="utf-8",
        )
        (output_path / "init_observation_schemas.json").write_text(
            json.dumps([item.to_dict() for item in result.schemas], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_path / "init_observation_rule_statistics.json").write_text(
            json.dumps(
                {
                    item.predicate_name: [
                        {
                            "bucket": "ground_truth_true",
                            "total_count": item.total_ground_truth_true,
                            "observed_positive_count": item.observed_true_when_ground_truth_true,
                            "observed_negative_count": item.observed_false_when_ground_truth_true,
                            "prob_observe_positive": item.prob_observable_true_given_ground_truth_true,
                            "prob_observe_negative": item.prob_observable_false_given_ground_truth_true,
                        },
                        {
                            "bucket": "ground_truth_false",
                            "total_count": item.total_ground_truth_false,
                            "observed_positive_count": item.observed_true_when_ground_truth_false,
                            "observed_negative_count": item.observed_false_when_ground_truth_false,
                            "prob_observe_positive": item.prob_observable_true_given_ground_truth_false,
                            "prob_observe_negative": item.prob_observable_false_given_ground_truth_false,
                        },
                    ]
                    for item in result.schemas
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_path / "init_observation_evidence.jsonl").write_text(
            "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in result.seed_examples),
            encoding="utf-8",
        )

    @staticmethod
    def _condition_holds(
        *,
        init_facts: list[str],
        condition_literals: list[str],
        argument_values: list[str],
    ) -> bool:
        binding = {f"?obs{index}": argument for index, argument in enumerate(argument_values)}
        grounded_state = set(init_facts)
        for literal in condition_literals:
            grounded_literal = literal
            for placeholder, value in binding.items():
                grounded_literal = grounded_literal.replace(placeholder, value)
            if grounded_literal not in grounded_state:
                return False
        return True


def load_init_observation_inputs(
    *,
    manipulation_artifact_dir: str | Path,
    scene_description_dir: str | Path,
    episode_grounding_pairs: list[tuple[str | Path, str | Path]],
) -> tuple[list[InitObservationSourceRecord], list[PredicateSchema], dict[str, str], dict[str, str]]:
    artifact_path = Path(manipulation_artifact_dir)
    scene_root = Path(scene_description_dir)
    predicate_inventory = [PredicateSchema(**row) for row in load_json(artifact_path / "predicate_inventory.json")]
    predicate_comments = dict(load_json_object(artifact_path / "predicate_comments.json"))
    camera_order_by_episode = _load_camera_order_by_episode(scene_root)
    object_type_rows = load_json(artifact_path / "object_types.json")
    parent_by_type = build_type_parent_map(object_type_rows)
    raw_steps = {
        (item.episode_name, item.step_index): item
        for item in load_raw_trajectory_steps(scene_root)
        if item.step_index == 0
    }
    source_records: list[InitObservationSourceRecord] = []
    for episode_file, grounding_dir in episode_grounding_pairs:
        episode_name = str(Path(episode_file).parent.name).strip()
        if not episode_name:
            continue
        grounding_path = Path(grounding_dir)
        problem_spec = load_json_object(grounding_path / "problem_summary.json")
        if not isinstance(problem_spec, dict):
            continue
        raw_step = raw_steps.get((episode_name, 0))
        if raw_step is None:
            continue
        objects = {
            str(item["name"]): str(item.get("type_name") or "object")
            for item in problem_spec.get("objects", [])
            if isinstance(item, dict) and item.get("name")
        }
        source_records.append(
            InitObservationSourceRecord(
                episode_name=str(episode_name),
                instruction=raw_step.instruction,
                init_facts=[str(item) for item in problem_spec.get("init_facts", [])],
                goal_facts=[str(item) for item in problem_spec.get("goal_facts", [])],
                scene_description=raw_step.observation_text,
                frame_paths=list(raw_step.frame_paths),
                objects=objects,
                camera_order_top_to_bottom=list(camera_order_by_episode.get(str(episode_name), [])),
            )
        )
    return source_records, predicate_inventory, predicate_comments, parent_by_type


def load_init_observation_learning_result(output_dir: str | Path) -> InitObservationLearningResult:
    payload = load_json_object(Path(output_dir) / _INIT_OBSERVATION_SUMMARY_FILENAME)
    return InitObservationLearningResult(
        source_records=[InitObservationSourceRecord(**item) for item in payload.get("source_records", [])],
        review_results=list(payload.get("review_results", [])),
        seed_examples=[InitObservationExample(**item) for item in payload.get("seed_examples", [])],
        conditions=[InitObservationCondition(**item) for item in payload.get("conditions", [])],
        expanded_examples=[InitObservationExample(**item) for item in payload.get("expanded_examples", [])],
        schemas=[InitObservationRuleSchema(**item) for item in payload.get("schemas", [])],
        predicate_inventory=[PredicateSchema(**item) for item in payload.get("predicate_inventory", [])],
        predicate_comments=dict(payload.get("predicate_comments", {})),
        rendered_module_text=str(payload.get("rendered_module_text") or ""),
        uncertain_predicate_discovery=(
            InitObservationUncertainPredicateDiscovery(**payload["uncertain_predicate_discovery"])
            if isinstance(payload.get("uncertain_predicate_discovery"), dict)
            else None
        ),
    )


__all__ = [
    "InitObservationLearner",
    "load_init_observation_learning_result",
    "load_init_observation_inputs",
]
