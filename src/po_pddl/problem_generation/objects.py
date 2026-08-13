from __future__ import annotations

import json
from pathlib import Path

from po_pddl.config import DEFAULT_MODEL

from ..domain_generation.infrastructure.llm_shared import build_user_content, make_client, safe_chat
from ..domain_generation.infrastructure.payload_utils import validate_snake_case
from ..domain_generation.infrastructure.response_parsing import extract_json_object
from ..domain_generation.stages.manipulation_domain_learning.grounding_update import load_manipulation_records
from ..domain_generation.stages.problem_grounding.models import ObjectDeclaration
from ..prompts import load_prompt
from .domain_analysis import DomainAnalysisResult
from .model_config import resolve_online_llm_config
from .models import VisibleObject

_DIRECTION_TOKENS = {
    "left",
    "right",
    "front",
    "back",
    "near",
    "far",
    "top",
    "bottom",
    "upper",
    "lower",
    "middle",
    "center",
    "centre",
}


def _load_prompt(name: str) -> str:
    return load_prompt(name)


def _normalize_object_name(name: str) -> str:
    cleaned = validate_snake_case(name.strip(), field_name="object_name")
    tokens = [token for token in cleaned.split("_") if token]
    kept_tokens = [token for token in tokens if token not in _DIRECTION_TOKENS]
    if kept_tokens:
        cleaned = "_".join(kept_tokens)
    return validate_snake_case(cleaned, field_name="normalized_object_name")


def _dedupe_name(name: str, seen: dict[str, int]) -> str:
    if name not in seen:
        seen[name] = 1
        return name
    suffix = seen[name]
    seen[name] += 1
    return f"{name}_{suffix}"


class VisibleObjectExtractionAgent:
    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        verbose: bool = False,
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
        self._prompt = _load_prompt("visible_objects.md")
        self._named_object_typing_prompt = _load_prompt("named_object_types.md")

    def infer_visible_objects(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None = None,
        instruction: str,
        manipulation_records_path: str | Path | None = None,
        known_object_names: set[str] | None = None,
    ) -> list[VisibleObject]:
        manipulation_records = (
            load_manipulation_records(manipulation_records_path)
            if manipulation_records_path is not None and Path(manipulation_records_path).exists()
            else []
        )
        manipulation_summary = [
            {
                "action": record.canonical_action_name,
                "arguments": list(record.action_arguments),
                "effect_bucket": record.effect_bucket,
                "success": record.success,
            }
            for record in manipulation_records
        ]
        payload = {
            "domain_summary": domain_analysis.render_summary(),
            "instruction": instruction.strip(),
            "image_input_note": image_input_note
            or "The provided image is a single-view snapshot of the initial scene.",
            "manipulation_records": manipulation_summary,
            "known_object_names": sorted(known_object_names or set()),
        }
        user_content = build_user_content(
            text=json.dumps(payload, ensure_ascii=False, indent=2),
            image_path=str(image_path),
        )
        client = make_client(api_key=self.api_key, base_url=self.base_url)
        reply = safe_chat(
            client,
            self._prompt,
            user_content,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        object_rows = data.get("objects", [])
        if not isinstance(object_rows, list):
            raise ValueError("Visible object agent response must contain an `objects` list.")

        seen_names: dict[str, int] = {}
        valid_types = set(domain_analysis.parsed_domain.types)
        results: list[VisibleObject] = []
        for row in object_rows:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid object row: {row!r}")
            raw_name = str(row.get("name", "")).strip()
            if not raw_name:
                continue
            normalized_name = _dedupe_name(_normalize_object_name(raw_name), seen_names)
            raw_type = str(row.get("type_name", row.get("type", "object"))).strip() or "object"
            type_name = raw_type if raw_type in valid_types else "object"
            results.append(
                VisibleObject(
                    name=normalized_name,
                    type_name=type_name,
                    justification=str(row.get("justification", "")).strip() or None,
                )
            )
        return results

    def infer_named_object_types(
        self,
        *,
        domain_analysis: DomainAnalysisResult,
        image_path: str | Path,
        image_input_note: str | None = None,
        instruction: str,
        object_names: set[str],
        manipulation_records_path: str | Path | None = None,
    ) -> list[VisibleObject]:
        if not object_names:
            return []
        manipulation_records = (
            load_manipulation_records(manipulation_records_path)
            if manipulation_records_path is not None and Path(manipulation_records_path).exists()
            else []
        )
        manipulation_summary = [
            {
                "action": record.canonical_action_name,
                "arguments": list(record.action_arguments),
                "effect_bucket": record.effect_bucket,
                "success": record.success,
            }
            for record in manipulation_records
        ]
        payload = {
            "domain_summary": domain_analysis.render_summary(),
            "instruction": instruction.strip(),
            "image_input_note": image_input_note
            or "The provided image is a single-view snapshot of the initial scene.",
            "candidate_object_names": sorted(object_names),
            "valid_domain_types": sorted(domain_analysis.parsed_domain.types),
            "manipulation_records": manipulation_summary,
        }
        user_content = build_user_content(
            text=json.dumps(payload, ensure_ascii=False, indent=2),
            image_path=str(image_path),
        )
        client = make_client(api_key=self.api_key, base_url=self.base_url)
        reply = safe_chat(
            client,
            self._named_object_typing_prompt,
            user_content,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        object_rows = data.get("objects", [])
        if not isinstance(object_rows, list):
            raise ValueError("Named object typing response must contain an `objects` list.")

        valid_types = set(domain_analysis.parsed_domain.types)
        expected_names = set(object_names)
        results_by_name: dict[str, VisibleObject] = {}
        for row in object_rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            if name not in expected_names:
                continue
            type_name = str(row.get("type_name", row.get("type", ""))).strip()
            if type_name not in valid_types:
                raise ValueError(f"Named object typing returned invalid domain type {type_name!r} for {name!r}.")
            results_by_name[name] = VisibleObject(
                name=name,
                type_name=type_name,
                justification=str(row.get("justification", "")).strip() or None,
            )
        missing_names = sorted(expected_names - set(results_by_name))
        if missing_names:
            raise ValueError("Named object typing did not classify every requested object: " + ", ".join(missing_names))
        return [results_by_name[name] for name in sorted(results_by_name)]

    @staticmethod
    def to_object_declarations(objects: list[VisibleObject]) -> list[ObjectDeclaration]:
        return [item.to_object_declaration() for item in objects]
