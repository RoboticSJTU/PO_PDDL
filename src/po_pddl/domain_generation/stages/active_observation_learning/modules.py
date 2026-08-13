from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from po_pddl.config import DEFAULT_MODEL

from .models import (
    ActiveObservationDiscoveryResult,
    ActiveObservationSourceRecord,
    ConfirmedObservation,
    DiscoveredObservation,
    VLMObservationConfirmationResult,
)
from .shared import (
    build_user_content,
    extract_json_object,
    load_prompt,
    make_client,
    safe_chat,
    sample_image_paths,
)


class ActiveObservationDiscoveryModule(Protocol):
    def discover(
        self,
        *,
        record: ActiveObservationSourceRecord,
        filtered_current_state: list[str],
        candidate_ground_truth_facts: list[str],
        predicate_comments: dict[str, str],
        feature_predicate_names: list[str],
        available_observables: list[str],
    ) -> ActiveObservationDiscoveryResult: ...


class ActiveObservationValueConfirmationModule(Protocol):
    def confirm(
        self,
        *,
        record: ActiveObservationSourceRecord,
        filtered_current_state: list[str],
        candidate_ground_truth_facts: list[str],
        targets: list[dict[str, object]],
        predicate_comments: dict[str, str],
    ) -> VLMObservationConfirmationResult: ...


@dataclass
class LLMActiveObservationDiscoveryModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2500
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("active_observation_discovery_prompt.md")

    def discover(
        self,
        *,
        record: ActiveObservationSourceRecord,
        filtered_current_state: list[str],
        candidate_ground_truth_facts: list[str],
        predicate_comments: dict[str, str],
        feature_predicate_names: list[str],
        available_observables: list[str],
    ) -> ActiveObservationDiscoveryResult:
        payload = {
            "record": record.to_dict(),
            "filtered_current_state": list(filtered_current_state),
            "candidate_ground_truth_facts": list(candidate_ground_truth_facts),
            "predicate_comments": dict(predicate_comments),
            "feature_predicate_names": list(feature_predicate_names),
            "available_observables": list(available_observables),
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
        discovered = [
            DiscoveredObservation(
                predicate_name=str(item["predicate_name"]),
                grounded_literal=str(item["grounded_literal"]),
                observed_value=bool(item["observed_value"]),
                rationale=str(item.get("rationale")) if item.get("rationale") is not None else None,
            )
            for item in data.get("discovered_observations", [])
            if isinstance(item, dict)
        ]
        if not discovered:
            raise ValueError(
                "Active observation discovery must return at least one observed predicate for each active-observation step."
            )
        return ActiveObservationDiscoveryResult(
            discovered_observations=discovered,
            summary=str(data.get("summary")) if data.get("summary") is not None else None,
            raw_output=reply,
        )


@dataclass
class VLMActiveObservationValueConfirmationModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 3000
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("active_observation_vlm_confirmation_prompt.md")

    def confirm(
        self,
        *,
        record: ActiveObservationSourceRecord,
        filtered_current_state: list[str],
        candidate_ground_truth_facts: list[str],
        targets: list[dict[str, object]],
        predicate_comments: dict[str, str],
    ) -> VLMObservationConfirmationResult:
        payload = {
            "record": record.to_dict(),
            "filtered_current_state": list(filtered_current_state),
            "candidate_ground_truth_facts": list(candidate_ground_truth_facts),
            "targets": list(targets),
            "predicate_comments": dict(predicate_comments),
        }
        prompt_payload = json.dumps(payload, ensure_ascii=False, indent=2)
        if record.camera_order_top_to_bottom:
            prompt_payload = (
                "Each input image is a vertical stack of synchronized camera views. "
                f"From top to bottom the camera order is: {', '.join(record.camera_order_top_to_bottom)}.\n\n"
                + prompt_payload
            )
        user_content = build_user_content(
            prompt_payload,
            image_paths=sample_image_paths(record.frame_paths) or None,
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
            ConfirmedObservation(
                predicate_name=str(item["predicate_name"]),
                grounded_literal=str(item["grounded_literal"]),
                ground_truth_value=bool(item["ground_truth_value"]),
                observed_value=bool(item["observed_value"]),
                rationale=str(item.get("rationale")) if item.get("rationale") is not None else None,
            )
            for item in data.get("confirmed_observations", [])
            if isinstance(item, dict)
        ]
        return VLMObservationConfirmationResult(
            confirmed_observations=confirmed,
            summary=str(data.get("summary")) if data.get("summary") is not None else None,
            raw_output=reply,
        )
