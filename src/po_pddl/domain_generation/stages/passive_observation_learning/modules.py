from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from po_pddl.config import DEFAULT_MODEL

from .models import (
    ConfirmedContradiction,
    ContainableRevealActionSelectionResult,
    DescriptionReviewResult,
    PassiveObservationSourceRecord,
    SuspectContradiction,
    VisibleContainedObjectsResult,
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


class DescriptionContradictionReviewModule(Protocol):
    def review(
        self,
        *,
        record: PassiveObservationSourceRecord,
        filtered_state_before: list[str],
        candidate_ground_truth_facts: list[str],
        predicate_comments: dict[str, str],
        feature_predicate_names: list[str],
    ) -> DescriptionReviewResult: ...


class VLMContradictionConfirmationModule(Protocol):
    def confirm(
        self,
        *,
        record: PassiveObservationSourceRecord,
        filtered_state_before: list[str],
        candidate_ground_truth_facts: list[str],
        suspects: list[SuspectContradiction],
        predicate_comments: dict[str, str],
    ) -> VLMConfirmationResult: ...


class ContainableRevealActionSelectionModule(Protocol):
    def select_actions(
        self,
        *,
        action_summaries: list[dict[str, object]],
        predicate_comments: dict[str, str],
        containable_type_names: list[str],
    ) -> ContainableRevealActionSelectionResult: ...


class VisibleContainedObjectsModule(Protocol):
    def identify_visible_objects(
        self,
        *,
        record: PassiveObservationSourceRecord,
        container_object_name: str,
        candidate_object_names: list[str],
        predicate_comments: dict[str, str],
    ) -> VisibleContainedObjectsResult: ...


@dataclass
class LLMDescriptionContradictionReviewModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2000
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("passive_observation_description_review_prompt.md")

    def review(
        self,
        *,
        record: PassiveObservationSourceRecord,
        filtered_state_before: list[str],
        candidate_ground_truth_facts: list[str],
        predicate_comments: dict[str, str],
        feature_predicate_names: list[str],
    ) -> DescriptionReviewResult:
        payload = {
            "record": record.to_dict(),
            "filtered_state_before": list(filtered_state_before),
            "candidate_ground_truth_facts": list(candidate_ground_truth_facts),
            "predicate_comments": dict(predicate_comments),
            "feature_predicate_names": list(feature_predicate_names),
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
class VLMContradictionConfirmationModuleLLM:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2500
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("passive_observation_vlm_confirmation_prompt.md")

    def confirm(
        self,
        *,
        record: PassiveObservationSourceRecord,
        filtered_state_before: list[str],
        candidate_ground_truth_facts: list[str],
        suspects: list[SuspectContradiction],
        predicate_comments: dict[str, str],
    ) -> VLMConfirmationResult:
        payload = {
            "record": record.to_dict(),
            "filtered_state_before": list(filtered_state_before),
            "candidate_ground_truth_facts": list(candidate_ground_truth_facts),
            "suspects": [item.to_dict() for item in suspects],
            "predicate_comments": dict(predicate_comments),
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
        confirmed = [
            ConfirmedContradiction(
                predicate_name=str(item["predicate_name"]),
                grounded_literal=str(item["grounded_literal"]),
                observed_value=bool(item["observed_value"]),
                ground_truth_value=bool(item["ground_truth_value"]),
                rationale=str(item.get("rationale")) if item.get("rationale") is not None else None,
            )
            for item in data.get("confirmed_contradictions", [])
            if isinstance(item, dict)
        ]
        return VLMConfirmationResult(
            confirmed_contradictions=confirmed,
            summary=str(data.get("summary")) if data.get("summary") is not None else None,
            raw_output=reply,
        )


@dataclass
class LLMContainableRevealActionSelectionModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1800
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("passive_in_reveal_action_selection_prompt.md")

    def select_actions(
        self,
        *,
        action_summaries: list[dict[str, object]],
        predicate_comments: dict[str, str],
        containable_type_names: list[str],
    ) -> ContainableRevealActionSelectionResult:
        payload = {
            "action_summaries": list(action_summaries),
            "predicate_comments": dict(predicate_comments),
            "containable_type_names": list(containable_type_names),
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
        action_names = sorted({str(item).strip() for item in data.get("action_names", []) if str(item).strip()})
        return ContainableRevealActionSelectionResult(
            action_names=action_names,
            summary=str(data.get("summary")) if data.get("summary") is not None else None,
            raw_output=reply,
        )


@dataclass
class LLMVisibleContainedObjectsModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1800
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("passive_in_visible_objects_prompt.md")

    def identify_visible_objects(
        self,
        *,
        record: PassiveObservationSourceRecord,
        container_object_name: str,
        candidate_object_names: list[str],
        predicate_comments: dict[str, str],
    ) -> VisibleContainedObjectsResult:
        payload = {
            "record": record.to_dict(),
            "container_object_name": str(container_object_name),
            "candidate_object_names": [str(item) for item in candidate_object_names if str(item).strip()],
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
        visible_object_names = sorted(
            {str(item).strip() for item in data.get("visible_object_names", []) if str(item).strip()}
        )
        return VisibleContainedObjectsResult(
            visible_object_names=visible_object_names,
            summary=str(data.get("summary")) if data.get("summary") is not None else None,
            raw_output=reply,
        )
