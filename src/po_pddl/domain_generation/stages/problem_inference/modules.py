from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

from po_pddl.config import DEFAULT_MODEL
from po_pddl.core.parser import ParsedDomain
from po_pddl.core.parser.sexpr import SExpr
from po_pddl.domain_generation.infrastructure.payload_utils import (
    normalize_optional_text as _normalize_optional_text,
)
from po_pddl.domain_generation.infrastructure.payload_utils import (
    validate_snake_case as _validate_snake_case,
)
from po_pddl.domain_generation.stages.problem_grounding.models import EpisodeContext

from .models import (
    InferredInitFact,
    InitCompletionFact,
    InitStateRepairPlan,
    LatentObjectCandidate,
    VisibleObjectCandidate,
)
from .shared import extract_json_object, load_prompt, make_client, safe_chat

logger = logging.getLogger(__name__)
_FACT_PATTERN = re.compile(r"^(?:not\s+)?[a-z][a-z0-9_]*\([^()]*\)$|^(?:not\s+)?[a-z][a-z0-9_]*$")


class VisibleObjectExtractionModule(Protocol):
    def extract_visible_objects(
        self,
        *,
        episode: EpisodeContext,
        domain_summary: dict[str, object],
        allowed_object_names: list[str] | None = None,
        review_guidance: dict[str, object] | None = None,
    ) -> tuple[list[VisibleObjectCandidate], list[str]]: ...


class LatentObjectDiscoveryModule(Protocol):
    def discover_latent_objects(
        self,
        *,
        episode: EpisodeContext,
        domain_summary: dict[str, object],
        visible_objects: list[VisibleObjectCandidate],
        visible_facts: list[str],
        allowed_object_names: list[str] | None = None,
        review_guidance: dict[str, object] | None = None,
    ) -> list[LatentObjectCandidate]: ...


class InitialStateInferenceModule(Protocol):
    def infer_initial_state(
        self,
        *,
        episode: EpisodeContext,
        domain_summary: dict[str, object],
        visible_objects: list[VisibleObjectCandidate],
        visible_facts: list[str],
        latent_objects: list[LatentObjectCandidate],
        grounded_predicates: list[str] | None = None,
        review_guidance: dict[str, object] | None = None,
    ) -> list[InferredInitFact]: ...


class InitCompletionModule(Protocol):
    def complete_initial_state(
        self,
        *,
        episode: EpisodeContext,
        domain_summary: dict[str, object],
        existing_true_init_facts: list[str],
        grounded_steps: list[dict[str, object]],
        grounded_state_trace: list[dict[str, object]],
        review_guidance: dict[str, object] | None = None,
    ) -> list[InitCompletionFact]: ...


class InitStateRepairModule(Protocol):
    def review_initial_state(
        self,
        *,
        episode: EpisodeContext,
        domain_summary: dict[str, object],
        selected_objects: list[VisibleObjectCandidate],
        current_true_init_facts: list[str],
        grounded_predicates: list[str] | None = None,
        review_guidance: dict[str, object] | None = None,
    ) -> InitStateRepairPlan: ...


def summarize_parsed_domain(parsed_domain: ParsedDomain) -> dict[str, object]:
    def sexpr_to_text(expr: SExpr | None) -> str | None:
        if expr is None:
            return None
        if isinstance(expr, str):
            return expr
        return "(" + " ".join(sexpr_to_text(item) or "" for item in expr).strip() + ")"

    predicate_summaries: list[dict[str, object]] = []
    for predicate in parsed_domain.predicates:
        parameter_types = parsed_domain.predicate_parameter_types.get(predicate, [])
        predicate_summaries.append(
            {
                "name": predicate.name,
                "arity": len(predicate.params),
                "parameter_types": [
                    {"parameter": parameter_name, "type": type_name} for parameter_name, type_name in parameter_types
                ],
            }
        )
    action_summaries: list[dict[str, object]] = []
    for schema in parsed_domain.actions:
        action_summaries.append(
            {
                "name": schema.action.name,
                "parameters": list(schema.action.params),
                "parameter_types": [
                    {"parameter": parameter_name, "type": type_name}
                    for parameter_name, type_name in schema.parameter_types
                ],
                "precondition": sexpr_to_text(schema.precondition),
            }
        )
    return {
        "domain_name": parsed_domain.domain_name,
        "types": sorted(parsed_domain.types.keys()),
        "predicates": predicate_summaries,
        "actions": action_summaries,
    }


def _coerce_fact_list(items: object, *, field_name: str) -> list[str]:
    if items is None:
        return []
    if not isinstance(items, list):
        items = [items]
    normalized: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        if not _FACT_PATTERN.fullmatch(text):
            raise ValueError(f"Expected symbolic fact for {field_name}, got {text!r}")
        normalized.append(text)
    return normalized


def _allowed_predicate_arity_map(domain_summary: dict[str, object]) -> dict[str, int]:
    allowed: dict[str, int] = {}
    for row in domain_summary.get("predicates", []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        try:
            arity = int(row.get("arity", 0))
        except (TypeError, ValueError):
            arity = 0
        allowed[name] = arity
    return allowed


def _validate_facts_against_domain_summary(
    facts: list[str],
    *,
    domain_summary: dict[str, object],
    field_name: str,
) -> list[str]:
    allowed_predicates = _allowed_predicate_arity_map(domain_summary)
    validated: list[str] = []
    for fact in facts:
        text = _normalize_optional_text(fact)
        if text is None:
            continue
        normalized = _coerce_fact_list([text], field_name=field_name)[0]
        stripped = normalized.strip()
        negated = stripped.startswith("not ")
        core = stripped[4:].strip() if negated else stripped
        if "(" in core and core.endswith(")"):
            predicate_name, raw_args = core[:-1].split("(", 1)
            predicate_name = predicate_name.strip()
            raw_args = raw_args.strip()
            arguments = [arg.strip() for arg in raw_args.split(",")] if raw_args else []
        else:
            predicate_name = core[:-2].strip() if core.endswith("()") else core.strip()
            arguments = []
        if predicate_name not in allowed_predicates:
            raise ValueError(f"{field_name} uses predicate not declared in domain: {normalized!r}")
        expected_arity = allowed_predicates[predicate_name]
        if len(arguments) != expected_arity:
            raise ValueError(
                f"{field_name} uses predicate {predicate_name!r} with arity {len(arguments)}, "
                f"but the domain declares arity {expected_arity}: {normalized!r}"
            )
        validated.append(normalized)
    return validated


@dataclass
class LLMVisibleObjectExtractionModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2200
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("visible_object_extraction_prompt.md")
        self.last_raw_output: str | None = None

    def extract_visible_objects(
        self,
        *,
        episode: EpisodeContext,
        domain_summary: dict[str, object],
        allowed_object_names: list[str] | None = None,
        review_guidance: dict[str, object] | None = None,
    ) -> tuple[list[VisibleObjectCandidate], list[str]]:
        payload = {
            "domain_summary": domain_summary,
            "episode_name": episode.episode_name,
            "step0_observation_text": episode.step0_observation_text,
            "allowed_object_names": list(allowed_object_names or []),
            "review_guidance": review_guidance or {},
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
        self.last_raw_output = reply
        data = extract_json_object(reply)
        object_rows = data.get("visible_objects", [])
        visible_facts = _validate_facts_against_domain_summary(
            _coerce_fact_list(data.get("visible_facts"), field_name="visible_facts"),
            domain_summary=domain_summary,
            field_name="visible_facts",
        )
        visible_objects: list[VisibleObjectCandidate] = []
        for row in object_rows:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid visible object payload: {row!r}")
            visible_objects.append(
                VisibleObjectCandidate(
                    name=_validate_snake_case(str(row.get("name", "")), field_name="visible_object_name"),
                    type_name=_validate_snake_case(
                        str(row.get("type_name", "object")), field_name="visible_object_type"
                    ),
                    supporting_observation=_normalize_optional_text(row.get("supporting_observation")),
                    visible_facts=_validate_facts_against_domain_summary(
                        _coerce_fact_list(row.get("visible_facts"), field_name="visible_object.visible_facts"),
                        domain_summary=domain_summary,
                        field_name="visible_object.visible_facts",
                    ),
                )
            )
        return visible_objects, visible_facts


@dataclass
class LLMLatentObjectDiscoveryModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2600
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("latent_object_discovery_prompt.md")
        self.last_raw_output: str | None = None

    def discover_latent_objects(
        self,
        *,
        episode: EpisodeContext,
        domain_summary: dict[str, object],
        visible_objects: list[VisibleObjectCandidate],
        visible_facts: list[str],
        allowed_object_names: list[str] | None = None,
        review_guidance: dict[str, object] | None = None,
    ) -> list[LatentObjectCandidate]:
        payload = {
            "domain_summary": domain_summary,
            "episode": episode.to_dict(),
            "visible_objects": [item.to_dict() for item in visible_objects],
            "visible_facts": visible_facts,
            "allowed_object_names": list(allowed_object_names or []),
            "review_guidance": review_guidance or {},
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
        self.last_raw_output = reply
        data = extract_json_object(reply)
        latent_rows = data.get("additional_objects", [])
        latent_objects: list[LatentObjectCandidate] = []
        for row in latent_rows:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid latent object payload: {row!r}")
            evidence = row.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = [str(evidence)]
            latent_objects.append(
                LatentObjectCandidate(
                    name=_validate_snake_case(str(row.get("name", "")), field_name="latent_object_name"),
                    type_name=_validate_snake_case(
                        str(row.get("type_name", "object")), field_name="latent_object_type"
                    ),
                    evidence=[str(item).strip() for item in evidence if str(item).strip()],
                )
            )
        return latent_objects


@dataclass
class LLMInitialStateInferenceModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2800
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("initial_state_inference_prompt.md")
        self.last_raw_output: str | None = None

    def infer_initial_state(
        self,
        *,
        episode: EpisodeContext,
        domain_summary: dict[str, object],
        visible_objects: list[VisibleObjectCandidate],
        visible_facts: list[str],
        latent_objects: list[LatentObjectCandidate],
        grounded_predicates: list[str] | None = None,
        review_guidance: dict[str, object] | None = None,
    ) -> list[InferredInitFact]:
        payload = {
            "domain_summary": domain_summary,
            "episode": episode.to_dict(),
            "visible_objects": [item.to_dict() for item in visible_objects],
            "visible_facts": visible_facts,
            "latent_objects": [item.to_dict() for item in latent_objects],
            "grounded_predicates": list(grounded_predicates or []),
            "review_guidance": review_guidance or {},
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
        self.last_raw_output = reply
        data = extract_json_object(reply)
        candidate_predicates = _validate_facts_against_domain_summary(
            [str(item).strip() for item in grounded_predicates or [] if str(item).strip()],
            domain_summary=domain_summary,
            field_name="grounded_predicates",
        )
        candidate_set = set(candidate_predicates)
        inferred_facts: list[InferredInitFact] = []
        if candidate_predicates:
            location_classification_rows = data.get("location_predicate_classifications")
            other_classification_rows = data.get("other_predicate_classifications")
            classification_rows = data.get("predicate_classifications")
            if location_classification_rows is not None or other_classification_rows is not None:
                if location_classification_rows is None or other_classification_rows is None:
                    raise ValueError(
                        "Initial-state inference must return both `location_predicate_classifications` and "
                        "`other_predicate_classifications` together when using the split classification format."
                    )
                if not isinstance(location_classification_rows, list):
                    raise ValueError("Expected location_predicate_classifications to be a list.")
                if not isinstance(other_classification_rows, list):
                    raise ValueError("Expected other_predicate_classifications to be a list.")
                classification_rows = [*location_classification_rows, *other_classification_rows]
            if classification_rows is None:
                raise ValueError(
                    "Initial-state inference must return `location_predicate_classifications` plus "
                    "`other_predicate_classifications` or the legacy combined `predicate_classifications` "
                    "when grounded_predicates are provided. Legacy `init_facts` output is not accepted in classification mode."
                )

            if not isinstance(classification_rows, list):
                raise ValueError("Expected predicate_classifications to be a list.")
            seen_facts: set[str] = set()
            true_fact_payloads: dict[str, tuple[str, str | None]] = {}
            for row in classification_rows:
                if not isinstance(row, dict):
                    raise ValueError(f"Invalid predicate classification payload: {row!r}")
                fact = _validate_facts_against_domain_summary(
                    [str(row.get("fact", "")).strip()],
                    domain_summary=domain_summary,
                    field_name="predicate_classifications.fact",
                )[0]
                if fact not in candidate_set:
                    raise ValueError(
                        f"Predicate classification returned {fact!r}, which is not in grounded_predicates."
                    )
                if fact in seen_facts:
                    raise ValueError(f"Predicate classification repeated fact {fact!r}.")
                seen_facts.add(fact)
                raw_value = row.get("value", row.get("truth_value", row.get("is_true")))
                if isinstance(raw_value, bool):
                    is_true = raw_value
                else:
                    normalized_value = str(raw_value).strip().lower()
                    if normalized_value in {"true", "t", "yes", "positive"}:
                        is_true = True
                    elif normalized_value in {"false", "f", "no", "negative"}:
                        is_true = False
                    else:
                        raise ValueError(
                            f"Predicate classification for {fact!r} must specify value true/false, got {raw_value!r}."
                        )
                if is_true:
                    true_fact_payloads[fact] = (
                        str(row.get("confidence", "medium")).strip() or "medium",
                        _normalize_optional_text(row.get("justification")),
                    )
            missing_facts = [fact for fact in candidate_predicates if fact not in seen_facts]
            if missing_facts:
                raise ValueError(
                    "Predicate classification must assign every grounded predicate to true or false. "
                    f"Missing facts: {missing_facts[:10]!r}"
                )
            for fact in candidate_predicates:
                payload = true_fact_payloads.get(fact)
                if payload is None:
                    continue
                confidence, justification = payload
                inferred_facts.append(
                    InferredInitFact(
                        fact=fact,
                        confidence=confidence,
                        justification=justification,
                    )
                )
            return inferred_facts

        init_rows = data.get("init_facts", [])
        for row in init_rows:
            if isinstance(row, str):
                fact = _validate_facts_against_domain_summary(
                    [row.strip()],
                    domain_summary=domain_summary,
                    field_name="init_facts",
                )[0]
                inferred_facts.append(InferredInitFact(fact=fact))
                continue
            if not isinstance(row, dict):
                raise ValueError(f"Invalid init fact payload: {row!r}")
            fact = _validate_facts_against_domain_summary(
                [str(row.get("fact", "")).strip()],
                domain_summary=domain_summary,
                field_name="init_facts",
            )[0]
            inferred_facts.append(
                InferredInitFact(
                    fact=fact,
                    confidence=str(row.get("confidence", "medium")).strip() or "medium",
                    justification=_normalize_optional_text(row.get("justification")),
                )
            )
        return inferred_facts


@dataclass
class LLMInitCompletionModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2400
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("init_completion_prompt.md")
        self.last_raw_output: str | None = None

    def complete_initial_state(
        self,
        *,
        episode: EpisodeContext,
        domain_summary: dict[str, object],
        existing_true_init_facts: list[str],
        grounded_steps: list[dict[str, object]],
        grounded_state_trace: list[dict[str, object]],
        review_guidance: dict[str, object] | None = None,
    ) -> list[InitCompletionFact]:
        payload = {
            "domain_summary": domain_summary,
            "episode": episode.to_dict(),
            "existing_true_init_facts": list(existing_true_init_facts),
            "grounded_steps": grounded_steps,
            "grounded_state_trace": grounded_state_trace,
            "review_guidance": review_guidance or {},
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
        self.last_raw_output = reply
        data = extract_json_object(reply)
        rows = data.get("additional_init_facts", [])
        completed_facts: list[InitCompletionFact] = []
        for row in rows:
            if isinstance(row, str):
                fact = _validate_facts_against_domain_summary(
                    [row.strip()],
                    domain_summary=domain_summary,
                    field_name="additional_init_facts",
                )[0]
                completed_facts.append(InitCompletionFact(fact=fact))
                continue
            if not isinstance(row, dict):
                raise ValueError(f"Invalid init completion payload: {row!r}")
            fact = _validate_facts_against_domain_summary(
                [str(row.get("fact", "")).strip()],
                domain_summary=domain_summary,
                field_name="additional_init_facts",
            )[0]
            completed_facts.append(
                InitCompletionFact(
                    fact=fact,
                    confidence=str(row.get("confidence", "medium")).strip() or "medium",
                    justification=_normalize_optional_text(row.get("justification")),
                )
            )
        return completed_facts


@dataclass
class LLMInitStateRepairModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2400
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("init_state_location_repair_prompt.md")
        self.last_raw_output: str | None = None

    def review_initial_state(
        self,
        *,
        episode: EpisodeContext,
        domain_summary: dict[str, object],
        selected_objects: list[VisibleObjectCandidate],
        current_true_init_facts: list[str],
        grounded_predicates: list[str] | None = None,
        review_guidance: dict[str, object] | None = None,
    ) -> InitStateRepairPlan:
        payload = {
            "domain_summary": domain_summary,
            "episode": episode.to_dict(),
            "selected_objects": [item.to_dict() for item in selected_objects],
            "current_true_init_facts": list(current_true_init_facts),
            "grounded_predicates": list(grounded_predicates or []),
            "review_guidance": review_guidance or {},
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
        self.last_raw_output = reply
        data = extract_json_object(reply)
        candidate_predicates = _validate_facts_against_domain_summary(
            [str(item).strip() for item in grounded_predicates or [] if str(item).strip()],
            domain_summary=domain_summary,
            field_name="grounded_predicates",
        )
        candidate_set = set(candidate_predicates)
        current_fact_set = set(
            _validate_facts_against_domain_summary(
                [str(item).strip() for item in current_true_init_facts if str(item).strip()],
                domain_summary=domain_summary,
                field_name="current_true_init_facts",
            )
        )

        def _coerce_repair_fact_rows(raw_items: object, *, field_name: str) -> list[str]:
            if raw_items is None:
                return []
            if not isinstance(raw_items, list):
                raw_items = [raw_items]
            normalized: list[str] = []
            for row in raw_items:
                if isinstance(row, dict):
                    fact = str(row.get("fact", "")).strip()
                else:
                    fact = str(row).strip()
                if not fact:
                    continue
                validated = _validate_facts_against_domain_summary(
                    [fact],
                    domain_summary=domain_summary,
                    field_name=field_name,
                )[0]
                normalized.append(validated)
            return normalized

        init_facts_add = _coerce_repair_fact_rows(data.get("init_facts_add", []), field_name="init_facts_add")
        init_facts_remove = _coerce_repair_fact_rows(data.get("init_facts_remove", []), field_name="init_facts_remove")

        valid_init_facts_add: list[str] = []
        for fact in init_facts_add:
            if fact in candidate_set or fact in current_fact_set:
                valid_init_facts_add.append(fact)
                continue
            logger.warning(
                "Ignoring init-state repair addition %r because it is not in grounded_predicates or "
                "current_true_init_facts.",
                fact,
            )

        valid_init_facts_remove: list[str] = []
        for fact in init_facts_remove:
            if fact in current_fact_set:
                valid_init_facts_remove.append(fact)
                continue
            logger.warning(
                "Ignoring init-state repair removal %r because it is not currently true in the init facts.",
                fact,
            )

        should_repair_raw = data.get("should_repair")
        should_repair = (
            bool(should_repair_raw)
            if should_repair_raw is not None
            else bool(valid_init_facts_add or valid_init_facts_remove)
        )
        repair_summary = _normalize_optional_text(data.get("repair_summary"))
        if should_repair and not (valid_init_facts_add or valid_init_facts_remove):
            logger.warning(
                "Init-state repair requested changes, but all proposed additions/removals were invalid and were ignored."
            )
            should_repair = False
            if repair_summary is None:
                repair_summary = "Repair proposal contained no valid init-state fact changes."
        return InitStateRepairPlan(
            should_repair=should_repair,
            init_facts_add=valid_init_facts_add,
            init_facts_remove=valid_init_facts_remove,
            repair_summary=repair_summary,
            raw_llm_output=reply,
        )
