from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any

from po_pddl.config import DEFAULT_MODEL
from po_pddl.domain_generation.infrastructure.payload_utils import normalize_optional_text

from .models import ActionSchema, ActionTaxonomyRecord, ManipulationEffectRecord, PredicateSchema, RawTrajectoryStep
from .object_name_normalization import (
    coarsen_object_identifiers,
    coarsen_symbolic_literal_list,
)
from .shared import extract_json_object, load_prompt, make_client, safe_chat
from .structured_action_templates import (
    InducedActionTemplateArtifact,
    build_action_name_map,
    compile_template_regex,
    induced_template_from_dict,
    match_action_text_to_template,
    render_action_text_from_template,
)

logger = logging.getLogger(__name__)
_FAILURE_MARKERS = ("failure", "failed", "失败")
_MAX_TEMPLATE_RESPONSE_ATTEMPTS = 3


def _is_failure(extra_info: str | None) -> bool:
    normalized = (normalize_optional_text(extra_info) or "").lower()
    return any(marker in normalized for marker in _FAILURE_MARKERS)


@dataclass
class InducedTemplateRegistry:
    templates: list[InducedActionTemplateArtifact] = field(default_factory=list)

    def set_templates(self, templates: list[InducedActionTemplateArtifact]) -> None:
        self.templates = list(templates)

    def ensure_templates(self) -> list[InducedActionTemplateArtifact]:
        if not self.templates:
            raise RuntimeError("No induced action templates are available yet.")
        return list(self.templates)

    def update_template_categories(
        self, category_by_action_name: dict[str, str]
    ) -> list[InducedActionTemplateArtifact]:
        updated = [
            replace(
                template,
                action_category=category_by_action_name.get(template.canonical_action_name, template.action_category),
            )
            for template in self.templates
        ]
        self.templates = updated
        return list(updated)

    def export_artifacts(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        templates = self.ensure_templates()
        return [template.to_dict() for template in templates], build_action_name_map(templates)


@dataclass(frozen=True)
class ActionTextNormalizationRecord:
    episode_name: str
    step_index: int
    original_action_text: str
    normalized_action_text: str
    canonical_action_name: str
    template_id: str
    resolution_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_name": self.episode_name,
            "step_index": self.step_index,
            "original_action_text": self.original_action_text,
            "normalized_action_text": self.normalized_action_text,
            "canonical_action_name": self.canonical_action_name,
            "template_id": self.template_id,
            "resolution_kind": self.resolution_kind,
        }


@dataclass(frozen=True)
class ActionTextPreprocessingResult:
    normalized_steps: list[RawTrajectoryStep]
    normalization_records: list[ActionTextNormalizationRecord]


@dataclass
class LLMSemanticActionTextPreprocessingModule:
    registry: InducedTemplateRegistry
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1800
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._bootstrap_prompt = load_prompt("semantic_action_template_bootstrap_prompt.md")

    def preprocess_steps(self, steps: list[RawTrajectoryStep]) -> ActionTextPreprocessingResult:
        actionable_steps = [step for step in steps if step.action_text]
        if not actionable_steps:
            return ActionTextPreprocessingResult(normalized_steps=list(steps), normalization_records=[])

        templates = self._induce_templates_for_dataset(actionable_steps)
        self.registry.set_templates(templates)
        normalized_steps: list[RawTrajectoryStep] = []
        normalization_records: list[ActionTextNormalizationRecord] = []
        for step in steps:
            if not step.action_text:
                normalized_steps.append(step)
                continue
            parsed = match_action_text_to_template(step.action_text, templates)
            normalized_text = render_action_text_from_template(parsed.template_text, parsed.placeholder_values)
            normalized_steps.append(replace(step, action_text=normalized_text))
            normalization_records.append(
                ActionTextNormalizationRecord(
                    episode_name=step.episode_name,
                    step_index=step.step_index,
                    original_action_text=step.action_text,
                    normalized_action_text=normalized_text,
                    canonical_action_name=parsed.canonical_action_name,
                    template_id=parsed.template_id,
                    resolution_kind="global_induction",
                )
            )
        return ActionTextPreprocessingResult(
            normalized_steps=normalized_steps,
            normalization_records=normalization_records,
        )

    def _induce_templates_for_dataset(
        self,
        steps: list[RawTrajectoryStep],
    ) -> list[InducedActionTemplateArtifact]:
        action_texts = sorted({step.action_text for step in steps if step.action_text})
        payload = {
            "episode_name": steps[0].episode_name if steps else "",
            "action_texts": [{"action_text": text} for text in action_texts],
        }
        validation_error: str | None = None
        for _attempt in range(_MAX_TEMPLATE_RESPONSE_ATTEMPTS):
            request_payload = dict(payload)
            if validation_error:
                request_payload["validation_feedback"] = validation_error
            reply = safe_chat(
                self._client,
                self._bootstrap_prompt,
                json.dumps(request_payload, ensure_ascii=False, indent=2),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                verbose=self.verbose,
            )
            try:
                data = extract_json_object(reply)
                template_rows = data.get("action_templates")
                if not isinstance(template_rows, list) or not template_rows:
                    raise ValueError("response must contain a non-empty action_templates list")
                templates = [
                    induced_template_from_dict(item, require_action_category=True)
                    for item in template_rows
                    if isinstance(item, dict)
                ]
                if len(templates) != len(template_rows):
                    raise ValueError("all action_templates rows must be JSON objects")
                unmatched = [
                    action_text for action_text in action_texts if self._safe_match(action_text, templates) is None
                ]
                if unmatched:
                    raise ValueError(
                        "every action text must match a template without absorbing a relation "
                        f"clause into an entity parameter; unmatched={unmatched}"
                    )
                return templates
            except ValueError as exc:
                validation_error = str(exc)
                logger.warning("Rejected semantic template bootstrap response: %s", validation_error)
        raise ValueError(f"Semantic template bootstrap failed validation: {validation_error}")

    @staticmethod
    def _safe_match(
        action_text: str,
        templates: list[InducedActionTemplateArtifact],
    ) -> Any | None:
        try:
            return match_action_text_to_template(action_text, templates)
        except ValueError:
            return None



@dataclass
class LLMTemplateActionCategoryModule:
    registry: InducedTemplateRegistry
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1400
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("semantic_action_category_prompt.md")

    def classify_template_categories(
        self,
        *,
        steps: list[RawTrajectoryStep],
    ) -> list[InducedActionTemplateArtifact]:
        templates = self.registry.ensure_templates()
        if all(template.action_category in {"manipulation", "active_observation"} for template in templates):
            return templates
        examples_by_action: dict[str, list[str]] = {}
        for step in steps:
            if not step.action_text:
                continue
            parsed = match_action_text_to_template(step.action_text, templates)
            examples_by_action.setdefault(parsed.canonical_action_name, [])
            if step.action_text not in examples_by_action[parsed.canonical_action_name]:
                examples_by_action[parsed.canonical_action_name].append(step.action_text)
        payload = {
            "instruction": steps[0].instruction if steps else "",
            "action_templates": [
                {
                    "template_id": template.template_id,
                    "template_text": template.template_text,
                    "canonical_action_name": template.canonical_action_name,
                    "parameter_roles": list(template.parameter_roles),
                    "parameter_placeholders": list(template.parameter_placeholders),
                    "example_action_texts": examples_by_action.get(template.canonical_action_name, []),
                }
                for template in templates
            ],
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
        rows = data.get("action_categories", [])
        if not isinstance(rows, list) or not rows:
            raise ValueError("Semantic action category response must contain a non-empty action_categories list")
        category_by_action_name: dict[str, str] = {}
        allowed_names = {template.canonical_action_name for template in templates}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each action_categories row must be an object")
            action_name = str(row.get("canonical_action_name") or "").strip()
            category = str(row.get("action_category") or "").strip().lower()
            if action_name not in allowed_names:
                raise ValueError(f"Semantic action category response referenced unknown action {action_name!r}")
            if category not in {"manipulation", "active_observation"}:
                raise ValueError(f"Unsupported semantic action category {category!r}")
            category_by_action_name[action_name] = category
        missing = sorted(allowed_names - set(category_by_action_name))
        if missing:
            raise ValueError(f"Semantic action category response omitted actions: {missing}")
        return self.registry.update_template_categories(category_by_action_name)


@dataclass
class SemanticTemplateActionTaxonomyModule:
    registry: InducedTemplateRegistry
    category_module: LLMTemplateActionCategoryModule
    max_workers: int = 1

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {self.max_workers}")

    def classify_actions(self, steps: list[RawTrajectoryStep]) -> list[ActionTaxonomyRecord]:
        templates = self.category_module.classify_template_categories(steps=steps)
        return self._classify_with_templates(steps, templates)

    def _classify_with_templates(
        self,
        steps: list[RawTrajectoryStep],
        templates: list[InducedActionTemplateArtifact],
    ) -> list[ActionTaxonomyRecord]:
        actionable_steps = [step for step in steps if step.action_text]
        logger.info(
            "Action taxonomy (semantic_template_category): classifying %d normalized action steps with %d templates (max_workers=%d)",
            len(actionable_steps),
            len(templates),
            self.max_workers,
        )
        if self.max_workers == 1:
            return self._classify_actions_sequentially(actionable_steps, templates)
        return self._classify_actions_in_parallel(actionable_steps, templates)

    def _classify_actions_sequentially(
        self,
        actionable_steps: list[RawTrajectoryStep],
        templates: list[InducedActionTemplateArtifact],
    ) -> list[ActionTaxonomyRecord]:
        outputs: list[ActionTaxonomyRecord] = []
        total = len(actionable_steps)
        for index, step in enumerate(actionable_steps, start=1):
            logger.info(
                "Action taxonomy (semantic_template_category): step %d/%d [%s step %d]",
                index,
                total,
                step.episode_name,
                step.step_index,
            )
            outputs.append(self._classify_single_action(step, templates))
        return outputs

    def _classify_actions_in_parallel(
        self,
        actionable_steps: list[RawTrajectoryStep],
        templates: list[InducedActionTemplateArtifact],
    ) -> list[ActionTaxonomyRecord]:
        total = len(actionable_steps)
        if total == 0:
            return []
        ordered_records: list[ActionTaxonomyRecord | None] = [None] * total
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="semantic-taxonomy") as executor:
            future_to_index = {
                executor.submit(self._classify_single_action, step, templates): index
                for index, step in enumerate(actionable_steps)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                step = actionable_steps[index]
                try:
                    ordered_records[index] = future.result()
                except Exception as exc:
                    logger.error(
                        "Action taxonomy (semantic_template_category) failed at [%s step %d]: %s",
                        step.episode_name,
                        step.step_index,
                        exc,
                    )
                    raise
                completed += 1
                logger.debug(
                    "Action taxonomy (semantic_template_category): completed %d/%d [%s step %d]",
                    completed,
                    total,
                    step.episode_name,
                    step.step_index,
                )
        return [record for record in ordered_records if record is not None]

    @staticmethod
    def _classify_single_action(
        step: RawTrajectoryStep,
        templates: list[InducedActionTemplateArtifact],
    ) -> ActionTaxonomyRecord:
        parsed = match_action_text_to_template(step.action_text or "", templates)
        if parsed.action_category not in {"manipulation", "active_observation"}:
            raise ValueError(
                f"Semantic template taxonomy requires action_category for {parsed.canonical_action_name!r}, got {parsed.action_category!r}"
            )
        return ActionTaxonomyRecord(
            episode_name=step.episode_name,
            step_index=step.step_index,
            raw_action_text=step.action_text or "",
            proposed_action_name=parsed.canonical_action_name,
            canonical_action_name=parsed.canonical_action_name,
            action_category=parsed.action_category,
            action_arguments=coarsen_object_identifiers(list(parsed.action_arguments)),
            object_mentions=coarsen_object_identifiers(list(parsed.object_mentions)),
            observation_text=step.observation_text,
            extra_info=step.extra_info,
            template_text=parsed.template_text,
            parameter_placeholders=list(parsed.parameter_placeholders),
        )


@dataclass
class LLMActionTemplateInductionModule:
    registry: InducedTemplateRegistry
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1800
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("action_template_induction_prompt.md")
        self._repair_prompt = load_prompt("action_template_repair_prompt.md")

    def induce_from_steps(self, steps: list[RawTrajectoryStep]) -> list[InducedActionTemplateArtifact]:
        action_texts = sorted({step.action_text for step in steps if step.action_text})
        instruction = next((step.instruction for step in steps if step.instruction), "")
        payload = {
            "instruction": instruction,
            "action_texts": [{"action_text": text} for text in action_texts],
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
        template_rows = data.get("action_templates")
        if not isinstance(template_rows, list) or not template_rows:
            raise ValueError("Template induction response must contain a non-empty action_templates list")
        templates = [induced_template_from_dict(item) for item in template_rows if isinstance(item, dict)]
        if len(templates) != len(template_rows):
            raise ValueError("All action_templates rows must be JSON objects")
        self.registry.set_templates(templates)
        return templates

    def repair_unmatched_action_texts(
        self,
        *,
        steps: list[RawTrajectoryStep],
        unmatched_steps: list[RawTrajectoryStep],
        existing_templates: list[InducedActionTemplateArtifact],
    ) -> list[InducedActionTemplateArtifact]:
        if not unmatched_steps:
            return list(existing_templates)
        instruction = next((step.instruction for step in steps if step.instruction), "")
        payload = {
            "instruction": instruction,
            "existing_action_templates": [template.to_dict() for template in existing_templates],
            "unmatched_action_texts": [
                {
                    "episode_name": step.episode_name,
                    "step_index": step.step_index,
                    "action_text": step.action_text or "",
                }
                for step in unmatched_steps
            ],
        }
        reply = safe_chat(
            self._client,
            self._repair_prompt,
            json.dumps(payload, ensure_ascii=False, indent=2),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        template_rows = data.get("action_templates")
        if not isinstance(template_rows, list) or not template_rows:
            raise ValueError("Template repair response must contain a non-empty action_templates list")
        supplemental_templates = [induced_template_from_dict(item) for item in template_rows if isinstance(item, dict)]
        if len(supplemental_templates) != len(template_rows):
            raise ValueError("All repaired action_templates rows must be JSON objects")
        merged_templates = _merge_induced_templates(existing_templates, supplemental_templates)
        self.registry.set_templates(merged_templates)
        return merged_templates


@dataclass
class InducedTemplateActionTaxonomyModule:
    induction_module: LLMActionTemplateInductionModule
    registry: InducedTemplateRegistry
    max_workers: int = 1

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {self.max_workers}")

    def classify_actions(self, steps: list[RawTrajectoryStep]) -> list[ActionTaxonomyRecord]:
        templates = self.induction_module.induce_from_steps(steps)
        actionable_steps = [step for step in steps if step.action_text]
        unmatched_steps = self._find_unmatched_steps(actionable_steps, templates)
        if unmatched_steps:
            logger.warning(
                "Action taxonomy (llm_induced_templates): %d action texts did not match any induced template; requesting repair templates",
                len(unmatched_steps),
            )
            templates = self.induction_module.repair_unmatched_action_texts(
                steps=steps,
                unmatched_steps=unmatched_steps,
                existing_templates=templates,
            )
            remaining_unmatched = self._find_unmatched_steps(actionable_steps, templates)
            if remaining_unmatched:
                raise ValueError(
                    "Some action texts still did not match any induced template after repair: "
                    f"{[step.action_text for step in remaining_unmatched]}"
                )
        return self._classify_with_templates(steps, templates)

    def classify_actions_with_allowed_schemas(
        self,
        steps: list[RawTrajectoryStep],
        allowed_action_schemas: list[ActionSchema],
    ) -> list[ActionTaxonomyRecord]:
        templates = self.registry.ensure_templates()
        allowed_names = {schema.canonical_action_name for schema in allowed_action_schemas}
        outputs = self._classify_with_templates(steps, templates)
        for record in outputs:
            if record.canonical_action_name not in allowed_names:
                raise ValueError(
                    f"Induced template taxonomy produced unsupported action {record.canonical_action_name!r}. "
                    f"Allowed: {sorted(allowed_names)}"
                )
        return outputs

    def _classify_with_templates(
        self,
        steps: list[RawTrajectoryStep],
        templates: list[InducedActionTemplateArtifact],
    ) -> list[ActionTaxonomyRecord]:
        actionable_steps = [step for step in steps if step.action_text]
        logger.info(
            "Action taxonomy (llm_induced_templates): classifying %d action steps with %d induced templates (max_workers=%d)",
            len(actionable_steps),
            len(templates),
            self.max_workers,
        )
        if self.max_workers == 1:
            return self._classify_actions_sequentially(actionable_steps, templates)
        return self._classify_actions_in_parallel(actionable_steps, templates)

    @staticmethod
    def _find_unmatched_steps(
        actionable_steps: list[RawTrajectoryStep],
        templates: list[InducedActionTemplateArtifact],
    ) -> list[RawTrajectoryStep]:
        unmatched_steps: list[RawTrajectoryStep] = []
        for step in actionable_steps:
            try:
                match_action_text_to_template(step.action_text or "", templates)
            except ValueError:
                unmatched_steps.append(step)
        return unmatched_steps

    def _classify_actions_sequentially(
        self,
        actionable_steps: list[RawTrajectoryStep],
        templates: list[InducedActionTemplateArtifact],
    ) -> list[ActionTaxonomyRecord]:
        outputs: list[ActionTaxonomyRecord] = []
        total = len(actionable_steps)
        for index, step in enumerate(actionable_steps, start=1):
            logger.info(
                "Action taxonomy (llm_induced_templates): step %d/%d [%s step %d]",
                index,
                total,
                step.episode_name,
                step.step_index,
            )
            outputs.append(self._classify_single_action(step, templates))
        return outputs

    def _classify_actions_in_parallel(
        self,
        actionable_steps: list[RawTrajectoryStep],
        templates: list[InducedActionTemplateArtifact],
    ) -> list[ActionTaxonomyRecord]:
        total = len(actionable_steps)
        if total == 0:
            return []
        logger.info(
            "Action taxonomy (llm_induced_templates): submitting %d tasks to thread pool",
            total,
        )
        ordered_records: list[ActionTaxonomyRecord | None] = [None] * total
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="induced-taxonomy") as executor:
            future_to_index = {
                executor.submit(self._classify_single_action, step, templates): index
                for index, step in enumerate(actionable_steps)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                step = actionable_steps[index]
                try:
                    ordered_records[index] = future.result()
                except Exception as exc:
                    logger.error(
                        "Action taxonomy (llm_induced_templates) failed at [%s step %d]: %s",
                        step.episode_name,
                        step.step_index,
                        exc,
                    )
                    raise
                completed += 1
                logger.debug(
                    "Action taxonomy (llm_induced_templates): completed %d/%d [%s step %d]",
                    completed,
                    total,
                    step.episode_name,
                    step.step_index,
                )
        return [record for record in ordered_records if record is not None]

    @staticmethod
    def _classify_single_action(
        step: RawTrajectoryStep,
        templates: list[InducedActionTemplateArtifact],
    ) -> ActionTaxonomyRecord:
        parsed = match_action_text_to_template(step.action_text or "", templates)
        return ActionTaxonomyRecord(
            episode_name=step.episode_name,
            step_index=step.step_index,
            raw_action_text=step.action_text or "",
            proposed_action_name=parsed.canonical_action_name,
            canonical_action_name=parsed.canonical_action_name,
            action_category=parsed.action_category,
            action_arguments=coarsen_object_identifiers(list(parsed.action_arguments)),
            object_mentions=coarsen_object_identifiers(list(parsed.object_mentions)),
            observation_text=step.observation_text,
            extra_info=step.extra_info,
            template_text=parsed.template_text,
            parameter_placeholders=list(parsed.parameter_placeholders),
        )


@dataclass
class InducedTemplateActionSchemaConsolidationModule:
    registry: InducedTemplateRegistry

    def consolidate(
        self,
        steps: list[RawTrajectoryStep],
        taxonomy_records: list[ActionTaxonomyRecord],
        predicate_inventory: list[PredicateSchema] | None = None,
    ) -> tuple[list[ActionSchema], list[ActionTaxonomyRecord]]:
        del steps, predicate_inventory
        schema_by_name = {
            template.canonical_action_name: _template_to_action_schema(template)
            for template in self.registry.ensure_templates()
        }
        used_names = {record.canonical_action_name for record in taxonomy_records}
        schemas = [schema_by_name[name] for name in sorted(used_names) if name in schema_by_name]
        return schemas, taxonomy_records

    def consolidate_with_allowed_schemas(
        self,
        steps: list[RawTrajectoryStep],
        taxonomy_records: list[ActionTaxonomyRecord],
        allowed_action_schemas: list[ActionSchema],
        predicate_inventory: list[PredicateSchema] | None = None,
    ) -> tuple[list[ActionSchema], list[ActionTaxonomyRecord]]:
        del steps, predicate_inventory
        allowed_names = {schema.canonical_action_name for schema in allowed_action_schemas}
        for record in taxonomy_records:
            if record.canonical_action_name not in allowed_names:
                raise ValueError(
                    f"Induced template consolidation cannot add new action {record.canonical_action_name!r}. "
                    f"Allowed: {sorted(allowed_names)}"
                )
        used_names = {record.canonical_action_name for record in taxonomy_records}
        schemas = [schema for schema in allowed_action_schemas if schema.canonical_action_name in used_names]
        return schemas, taxonomy_records


@dataclass
class InducedTemplateManipulationEffectLearningModule:
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
        self._prompt = load_prompt("manipulation_delta_prompt.md")

    def learn_effects(
        self,
        steps: list[RawTrajectoryStep],
        action_schemas: list[ActionSchema],
        taxonomy_records: list[ActionTaxonomyRecord],
        predicate_inventory: list[PredicateSchema] | None = None,
    ) -> list[ManipulationEffectRecord]:
        step_map = {(step.episode_name, step.step_index): step for step in steps}
        manipulation_schemas = [
            schema.to_dict() for schema in action_schemas if schema.action_category == "manipulation"
        ]
        manipulation_records = [record for record in taxonomy_records if record.action_category == "manipulation"]
        logger.info(
            "Manipulation effect learning (LLM fixed-branch): learning from %d manipulation records (max_workers=%d)",
            len(manipulation_records),
            self.max_workers,
        )
        allowed_predicates = {item.predicate_name for item in predicate_inventory or []}
        if self.max_workers == 1:
            return self._learn_effects_sequentially(
                step_map,
                manipulation_records,
                manipulation_schemas,
                allowed_predicates,
            )
        return self._learn_effects_in_parallel(
            step_map,
            manipulation_records,
            manipulation_schemas,
            allowed_predicates,
        )

    def _learn_effects_sequentially(
        self,
        step_map: dict[tuple[str, int], RawTrajectoryStep],
        manipulation_records: list[ActionTaxonomyRecord],
        manipulation_schemas: list[dict[str, object]],
        allowed_predicates: set[str],
    ) -> list[ManipulationEffectRecord]:
        outputs: list[ManipulationEffectRecord] = []
        total = len(manipulation_records)
        for index, taxonomy_record in enumerate(manipulation_records, start=1):
            step = step_map[(taxonomy_record.episode_name, taxonomy_record.step_index)]
            logger.info(
                "Manipulation effect learning (LLM fixed-branch): record %d/%d [%s step %d]",
                index,
                total,
                step.episode_name,
                step.step_index,
            )
            outputs.append(self._learn_single_effect(step, taxonomy_record, manipulation_schemas, allowed_predicates))
        return outputs

    def _learn_effects_in_parallel(
        self,
        step_map: dict[tuple[str, int], RawTrajectoryStep],
        manipulation_records: list[ActionTaxonomyRecord],
        manipulation_schemas: list[dict[str, object]],
        allowed_predicates: set[str],
    ) -> list[ManipulationEffectRecord]:
        total = len(manipulation_records)
        if total == 0:
            return []
        logger.info(
            "Manipulation effect learning (LLM fixed-branch): submitting %d tasks to thread pool",
            total,
        )
        ordered_records: list[ManipulationEffectRecord | None] = [None] * total
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="induced-manip-effects") as executor:
            future_to_index = {
                executor.submit(
                    self._learn_single_effect,
                    step_map[(record.episode_name, record.step_index)],
                    record,
                    manipulation_schemas,
                    allowed_predicates,
                ): index
                for index, record in enumerate(manipulation_records)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                taxonomy_record = manipulation_records[index]
                step = step_map[(taxonomy_record.episode_name, taxonomy_record.step_index)]
                try:
                    ordered_records[index] = future.result()
                except Exception as exc:
                    logger.error(
                        "Manipulation effect learning (LLM fixed-branch) failed at [%s step %d]: %s",
                        step.episode_name,
                        step.step_index,
                        exc,
                    )
                    raise
                completed += 1
                logger.debug(
                    "Manipulation effect learning (LLM fixed-branch): completed %d/%d [%s step %d]",
                    completed,
                    total,
                    step.episode_name,
                    step.step_index,
                )
        return [record for record in ordered_records if record is not None]

    def _learn_single_effect(
        self,
        step: RawTrajectoryStep,
        taxonomy_record: ActionTaxonomyRecord,
        manipulation_schemas: list[dict[str, object]],
        allowed_predicates: set[str],
    ) -> ManipulationEffectRecord:
        expected_success = not _is_failure(step.extra_info)
        expected_branch = "success" if expected_success else "failure"
        payload = {
            "instruction": step.instruction,
            "manipulation_action_schemas": manipulation_schemas,
            "taxonomy_record": {
                **taxonomy_record.to_dict(),
                "observation_text": None,
            },
            "step": {
                **step.to_dict(),
                "observation_text": step.observation_text if _is_failure(step.extra_info) else None,
                "previous_observation_text": None,
                "previous_known_observation_text": None,
            },
            "fixed_action_name": taxonomy_record.canonical_action_name,
            "fixed_effect_branch": expected_branch,
            "fixed_success_value": expected_success,
        }
        if allowed_predicates:
            payload["allowed_predicates"] = sorted(allowed_predicates)
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
        nested = data.get("manipulation_record")
        if isinstance(nested, dict):
            data = nested
        delta_add = coarsen_symbolic_literal_list(_coerce_fact_list(data.get("delta_add")))
        delta_del = coarsen_symbolic_literal_list(_coerce_fact_list(data.get("delta_del")))
        if allowed_predicates:
            from .modules import _predicate_name_from_literal

            for literal in delta_add + delta_del:
                predicate_name = _predicate_name_from_literal(literal)
                if predicate_name not in allowed_predicates:
                    raise ValueError(
                        f"Structured manipulation effect learning produced predicate {predicate_name!r} not present in allowed_predicates."
                    )
        return ManipulationEffectRecord(
            episode_name=step.episode_name,
            step_index=step.step_index,
            raw_action_text=taxonomy_record.raw_action_text,
            canonical_action_name=taxonomy_record.canonical_action_name,
            action_arguments=coarsen_object_identifiers(list(taxonomy_record.action_arguments)),
            pre_observation_text=None,
            post_observation_text=None,
            extra_info=step.extra_info,
            delta_add=delta_add,
            delta_del=delta_del,
            effect_bucket=f"{taxonomy_record.canonical_action_name}_{expected_branch}",
            success=expected_success,
        )


def _template_to_action_schema(template: InducedActionTemplateArtifact) -> ActionSchema:
    return ActionSchema(
        canonical_action_name=template.canonical_action_name,
        action_category=template.action_category,
        parameter_count=len(template.parameter_roles),
        parameter_roles=list(template.parameter_roles),
        precondition_literals=[],
        schema_description=template.template_text,
    )


def _merge_induced_templates(
    existing_templates: list[InducedActionTemplateArtifact],
    supplemental_templates: list[InducedActionTemplateArtifact],
) -> list[InducedActionTemplateArtifact]:
    merged: list[InducedActionTemplateArtifact] = []
    seen_identity_keys: set[tuple[str, str, str]] = set()
    signature_by_template_id: dict[str, tuple[str, str, tuple[str, ...], tuple[str, ...], str]] = {}
    signature_by_action_name: dict[str, tuple[str, str, tuple[str, ...], tuple[str, ...], str]] = {}

    def _signature(template: InducedActionTemplateArtifact) -> tuple[str, str, tuple[str, ...], tuple[str, ...], str]:
        return (
            template.template_text,
            template.action_category,
            tuple(template.parameter_roles),
            tuple(template.parameter_placeholders),
            compile_template_regex(template.template_text).pattern,
        )

    def _add_template(template: InducedActionTemplateArtifact) -> None:
        identity_key = (template.template_id, template.canonical_action_name, template.template_text)
        if identity_key in seen_identity_keys:
            return
        signature = _signature(template)
        existing_by_id = signature_by_template_id.get(template.template_id)
        if existing_by_id is not None and existing_by_id != signature:
            raise ValueError(f"Template repair returned conflicting template_id {template.template_id!r}")
        existing_by_name = signature_by_action_name.get(template.canonical_action_name)
        if existing_by_name is not None and existing_by_name != signature:
            raise ValueError(
                f"Template repair returned conflicting canonical_action_name {template.canonical_action_name!r}"
            )
        signature_by_template_id[template.template_id] = signature
        signature_by_action_name[template.canonical_action_name] = signature
        seen_identity_keys.add(identity_key)
        merged.append(template)

    for template in existing_templates:
        _add_template(template)
    for template in supplemental_templates:
        _add_template(template)
    return merged


def _coerce_fact_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []
