from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from po_pddl.config import DEFAULT_MODEL

from .models import (
    ConfirmedContradiction,
    DescriptionReviewResult,
    InitObservationSourceRecord,
    InitObservationUncertainPredicateDiscovery,
    SuspectContradiction,
    VLMConfirmationResult,
)
from .shared import (
    build_user_content,
    extract_json_object,
    load_prompt,
    make_client,
    safe_chat,
    sample_image_paths,
)


class InitObservationUncertainPredicateDiscoveryModule(Protocol):
    def discover(
        self,
        *,
        records: list[InitObservationSourceRecord],
        predicate_inventory: list[dict[str, object]],
        predicate_comments: dict[str, str],
    ) -> InitObservationUncertainPredicateDiscovery: ...


class InitDescriptionContradictionReviewModule(Protocol):
    def review(
        self,
        *,
        record: InitObservationSourceRecord,
        filtered_init_facts: list[str],
        candidate_ground_truth_facts: list[str],
        predicate_comments: dict[str, str],
    ) -> DescriptionReviewResult: ...


class InitVLMContradictionConfirmationModule(Protocol):
    def confirm(
        self,
        *,
        record: InitObservationSourceRecord,
        filtered_init_facts: list[str],
        candidate_ground_truth_facts: list[str],
        suspects: list[SuspectContradiction],
        predicate_comments: dict[str, str],
        uncertain_predicate_names: list[str],
    ) -> VLMConfirmationResult: ...


@dataclass
class VLMInitObservationUncertainPredicateDiscoveryModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2500
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("init_observation_uncertain_predicate_discovery_prompt.md")

    def discover(
        self,
        *,
        records: list[InitObservationSourceRecord],
        predicate_inventory: list[dict[str, object]],
        predicate_comments: dict[str, str],
    ) -> InitObservationUncertainPredicateDiscovery:
        payload = {
            "records": [
                {
                    "episode_name": record.episode_name,
                    "instruction": record.instruction,
                    "scene_description": record.scene_description,
                    "objects": dict(record.objects),
                }
                for record in records
            ],
            "predicate_inventory": list(predicate_inventory),
            "predicate_comments": dict(predicate_comments),
        }
        representative_images: list[str] = []
        for record in records:
            sampled = sample_image_paths(record.frame_paths)
            if sampled:
                representative_images.append(sampled[0])
        if len(representative_images) > 12:
            indices = {round(index * (len(representative_images) - 1) / 11) for index in range(12)}
            representative_images = [path for index, path in enumerate(representative_images) if index in indices]
        user_content = build_user_content(
            json.dumps(payload, ensure_ascii=False, indent=2),
            image_paths=representative_images or None,
        )
        reply = safe_chat(
            self._client,
            self._prompt,
            user_content,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        predicate_names: list[str] = []
        rationale_by_predicate: dict[str, str] = {}
        for item in data.get("observation_uncertain_predicates", []):
            if not isinstance(item, dict):
                continue
            predicate_name = str(item.get("predicate_name") or "").strip()
            if not predicate_name or predicate_name in predicate_names:
                continue
            predicate_names.append(predicate_name)
            rationale = str(item.get("rationale") or "").strip()
            if rationale:
                rationale_by_predicate[predicate_name] = rationale
        return InitObservationUncertainPredicateDiscovery(
            predicate_names=predicate_names,
            rationale_by_predicate=rationale_by_predicate,
            summary=str(data.get("summary")) if data.get("summary") is not None else None,
            raw_output=reply,
        )


@dataclass
class LLMInitDescriptionContradictionReviewModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2000
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("init_observation_description_review_prompt.md")

    def review(
        self,
        *,
        record: InitObservationSourceRecord,
        filtered_init_facts: list[str],
        candidate_ground_truth_facts: list[str],
        predicate_comments: dict[str, str],
    ) -> DescriptionReviewResult:
        payload = {
            "record": record.to_dict(),
            "filtered_init_facts": list(filtered_init_facts),
            "candidate_ground_truth_facts": list(candidate_ground_truth_facts),
            "predicate_comments": dict(predicate_comments),
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
        suspects = [
            SuspectContradiction(
                predicate_name=str(item["predicate_name"]),
                grounded_literal=str(item["grounded_literal"]),
                observed_value=bool(item["observed_value"]),
                ground_truth_value=bool(item["ground_truth_value"]),
                rationale=str(item.get("rationale")) if item.get("rationale") is not None else None,
            )
            for item in data.get("suspect_contradictions", [])
            if isinstance(item, dict)
        ]
        return DescriptionReviewResult(
            should_review_with_vlm=bool(data.get("should_review_with_vlm", bool(suspects))),
            suspect_contradictions=suspects,
            summary=str(data.get("summary")) if data.get("summary") is not None else None,
            raw_output=reply,
        )


@dataclass
class VLMInitContradictionConfirmationModuleLLM:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2500
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("init_observation_vlm_confirmation_prompt.md")

    def confirm(
        self,
        *,
        record: InitObservationSourceRecord,
        filtered_init_facts: list[str],
        candidate_ground_truth_facts: list[str],
        suspects: list[SuspectContradiction],
        predicate_comments: dict[str, str],
        uncertain_predicate_names: list[str],
    ) -> VLMConfirmationResult:
        payload = {
            "record": record.to_dict(),
            "filtered_init_facts": list(filtered_init_facts),
            "candidate_ground_truth_facts": list(candidate_ground_truth_facts),
            "suspects": [item.to_dict() for item in suspects],
            "predicate_comments": dict(predicate_comments),
            "uncertain_predicate_names": list(uncertain_predicate_names),
        }
        image_paths = sample_image_paths(record.frame_paths)
        prompt_payload = json.dumps(payload, ensure_ascii=False, indent=2)
        if record.camera_order_top_to_bottom:
            prompt_payload = (
                "Each input image is a vertical stack of synchronized camera views. "
                f"From top to bottom the camera order is: {', '.join(record.camera_order_top_to_bottom)}.\n\n"
                + prompt_payload
            )
        user_content = build_user_content(
            prompt_payload,
            image_paths=image_paths if image_paths else None,
        )
        reply = safe_chat(
            self._client,
            self._prompt,
            user_content,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        raw_confirmed_items = data.get("observation_evaluations")
        if not isinstance(raw_confirmed_items, list):
            raw_confirmed_items = data.get("confirmed_contradictions", [])
        confirmed = [
            ConfirmedContradiction(
                predicate_name=str(item["predicate_name"]),
                grounded_literal=str(item["grounded_literal"]),
                observed_value=bool(item["observed_value"]),
                ground_truth_value=bool(item["ground_truth_value"]),
                rationale=str(item.get("rationale")) if item.get("rationale") is not None else None,
            )
            for item in raw_confirmed_items
            if isinstance(item, dict)
            and bool(item.get("visually_assessable", True))
            and bool(item.get("observed_value")) != bool(item.get("ground_truth_value"))
        ]
        return VLMConfirmationResult(
            confirmed_contradictions=confirmed,
            summary=str(data.get("summary")) if data.get("summary") is not None else None,
            raw_output=reply,
        )
