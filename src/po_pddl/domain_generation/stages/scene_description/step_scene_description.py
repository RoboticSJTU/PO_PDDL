from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from po_pddl.config import DEFAULT_MODEL
from po_pddl.domain_generation.infrastructure.artifact_io import load_episode_payload
from po_pddl.domain_generation.infrastructure.episode_video import extract_single_episode_step_frames
from po_pddl.domain_generation.infrastructure.llm_shared import (
    build_user_content,
    load_llm_config,
    make_client,
    safe_chat,
)
from po_pddl.domain_generation.infrastructure.payload_utils import normalize_optional_text
from po_pddl.domain_generation.infrastructure.response_parsing import extract_json_object
from po_pddl.prompts import load_prompt

DEFAULT_PROMPT_NAME = "step_scene_description_prompt.md"
DEFAULT_FRAME_SELECTION_MODE = "first_and_last"
_VALID_FRAME_SELECTION_MODES = {
    "first_and_last",
    "all_sampled",
}
DEFAULT_SYSTEM_PROMPT = (
    "You generate concise, visually grounded English scene descriptions for one action step from ordered video frames. "
    "Only describe visually supported objects that appear in the provided allowed object list. "
    "Instruction, the current action, and the future action sequence only highlight which objects and regions deserve attention "
    "and how to name them; they are not evidence."
)
_REPAIR_SUFFIX = (
    "\n\nYour previous reply was invalid because `scene_description_text` was empty or missing. "
    "Return JSON only, and ensure `scene_description_text` is a non-empty single-sentence scene description. "
    "If evidence is limited, still provide a short conservative description mentioning only clearly visible allowed objects."
)

_SCENE_DESCRIPTION_RESPONSE_KEYS = (
    "scene_description_text",
    "observation_text",
    "scene_description",
    "observation",
    "description",
    "text",
)


@dataclass(frozen=True)
class PriorSceneDescriptionStep:
    step_index: int
    action_text: str | None
    extra_info: str | None
    scene_description_text: str | None

    def to_prompt_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StepSceneDescriptionRequest:
    instruction: str
    prior_steps: list[PriorSceneDescriptionStep]
    current_step_index: int
    current_action_text: str
    current_extra_info: str | None
    current_frame_paths: list[str]
    future_action_sequence: list[str]
    allowed_object_names: list[str]
    current_step_focus_objects: list[str]
    camera_order_top_to_bottom: list[str]

    def to_user_payload(self) -> str:
        payload = {
            "instruction": self.instruction,
            "prior_steps": [item.to_prompt_dict() for item in self.prior_steps],
            "current_step": {
                "step_index": self.current_step_index,
                "action_text": self.current_action_text,
                "extra_info": self.current_extra_info,
            },
            "future_action_sequence": [
                {"order": index + 1, "action_text": action_text}
                for index, action_text in enumerate(self.future_action_sequence)
            ],
            "allowed_object_names": list(self.allowed_object_names),
            "current_step_focus_objects": list(self.current_step_focus_objects),
        }
        if self.camera_order_top_to_bottom:
            payload["camera_order_top_to_bottom"] = list(self.camera_order_top_to_bottom)
        return json.dumps(payload, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class StepSceneDescriptionGenerationResult:
    episode_name: str
    step_index: int
    instruction: str
    current_action_text: str
    current_extra_info: str | None
    current_frame_paths: list[str]
    future_action_sequence: list[str]
    camera_order_top_to_bottom: list[str]
    scene_description_text: str
    raw_response: str
    model: str
    prompt_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StepSceneDescriptionGenerator:
    def __init__(
        self,
        *,
        config_path: str | None = None,
        config_name: str = "openai_config",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1600,
        prompt_name: str = DEFAULT_PROMPT_NAME,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        frame_selection_mode: str = DEFAULT_FRAME_SELECTION_MODE,
        image_detail: str = "high",
        verbose: bool = False,
        client: Any | None = None,
    ) -> None:
        self._config = load_llm_config(config_path=config_path, config_name=config_name)
        self.api_key = api_key or self._config.get("api_key")
        self.base_url = base_url or self._config.get("base_url")
        self.model = model or self._config.get("model") or DEFAULT_MODEL
        self.temperature = temperature if temperature is not None else (self._config.get("temperature") or 0.2)
        self.max_tokens = max_tokens
        self.prompt_name = prompt_name
        self.system_prompt = system_prompt
        self.frame_selection_mode = _normalize_frame_selection_mode(frame_selection_mode)
        self.image_detail = image_detail
        self.verbose = verbose
        self._prompt = load_prompt(prompt_name)
        self._client = client if client is not None else make_client(api_key=self.api_key, base_url=self.base_url)

    def build_request_from_episode(
        self,
        episode_file: str | Path,
        *,
        step_index: int,
        frame_paths: list[str],
        allowed_object_names: list[str] | None = None,
        current_step_focus_objects: list[str] | None = None,
        camera_order_top_to_bottom: list[str] | None = None,
    ) -> tuple[str, StepSceneDescriptionRequest]:
        episode_path = Path(episode_file).resolve()
        payload = load_episode_payload(episode_path)
        instruction = normalize_optional_text(payload.get("instruction"))
        if instruction is None:
            raise ValueError(f"{episode_path} is missing instruction")
        if step_index <= 0:
            raise ValueError("step_index must be > 0 for subsequent-step scene description generation")

        steps = payload.get("steps") or []
        if not isinstance(steps, list):
            raise ValueError(f"{episode_path} has non-list 'steps'")

        current_step: dict[str, Any] | None = None
        prior_steps: list[PriorSceneDescriptionStep] = []
        future_action_sequence: list[str] = []

        for item in steps:
            if not isinstance(item, dict):
                continue
            item_step_index = item.get("step_index")
            action_text = normalize_optional_text(item.get("action_text"))
            extra_info = normalize_optional_text(item.get("extra_info"))
            observation_text = normalize_optional_text(item.get("observation_text"))
            if item_step_index == step_index:
                current_step = item
            elif isinstance(item_step_index, int) and 0 <= item_step_index < step_index:
                prior_steps.append(
                    PriorSceneDescriptionStep(
                        step_index=item_step_index,
                        action_text=action_text,
                        extra_info=extra_info,
                        scene_description_text=observation_text,
                    )
                )
            elif isinstance(item_step_index, int) and item_step_index > step_index and action_text is not None:
                future_action_sequence.append(action_text)

        if current_step is None:
            raise ValueError(f"{episode_path} does not contain step_index={step_index}")

        current_action_text = normalize_optional_text(current_step.get("action_text"))
        if current_action_text is None:
            raise ValueError(f"{episode_path} step {step_index} is missing action_text")

        selected_frame_paths = select_step_scene_description_frame_paths(
            frame_paths,
            step_index=step_index,
            mode=self.frame_selection_mode,
        )
        request = StepSceneDescriptionRequest(
            instruction=instruction,
            prior_steps=prior_steps,
            current_step_index=step_index,
            current_action_text=current_action_text,
            current_extra_info=normalize_optional_text(current_step.get("extra_info")),
            current_frame_paths=selected_frame_paths,
            future_action_sequence=future_action_sequence,
            allowed_object_names=list(allowed_object_names or []),
            current_step_focus_objects=list(current_step_focus_objects or []),
            camera_order_top_to_bottom=list(camera_order_top_to_bottom or []),
        )
        episode_name = str(payload.get("episode_name") or episode_path.parent.name)
        return episode_name, request

    def generate(
        self,
        *,
        episode_name: str,
        request: StepSceneDescriptionRequest,
    ) -> StepSceneDescriptionGenerationResult:
        if not request.current_frame_paths:
            raise ValueError("current_frame_paths must not be empty")
        system_prompt = self.system_prompt
        if request.camera_order_top_to_bottom:
            system_prompt = (
                f"{system_prompt} Each input image is a vertical stack of synchronized camera views. "
                f"From top to bottom the camera order is: {', '.join(request.camera_order_top_to_bottom)}."
            )
        user_prompt = f"{self._prompt.strip()}\n\nInput JSON:\n{request.to_user_payload()}"
        user_content = build_user_content(
            user_prompt,
            image_paths=request.current_frame_paths,
            detail=self.image_detail,
        )
        raw_response = safe_chat(
            self._client,
            system_prompt,
            user_content,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        raw_response, scene_description_text = self._recover_scene_description_text(
            system_prompt=system_prompt,
            user_content=user_content,
            raw_response=raw_response,
        )
        return StepSceneDescriptionGenerationResult(
            episode_name=episode_name,
            step_index=request.current_step_index,
            instruction=request.instruction,
            current_action_text=request.current_action_text,
            current_extra_info=request.current_extra_info,
            current_frame_paths=list(request.current_frame_paths),
            future_action_sequence=list(request.future_action_sequence),
            camera_order_top_to_bottom=list(request.camera_order_top_to_bottom),
            scene_description_text=scene_description_text,
            raw_response=raw_response,
            model=self.model,
            prompt_name=self.prompt_name,
        )

    def _recover_scene_description_text(
        self,
        *,
        system_prompt: str,
        user_content,
        raw_response: str,
    ) -> tuple[str, str]:
        scene_description_text = _try_extract_scene_description_text(raw_response)
        if scene_description_text is not None:
            return raw_response, scene_description_text
        repair_user_content = _append_repair_suffix_to_user_content(user_content, _REPAIR_SUFFIX)
        repaired_raw_response = safe_chat(
            self._client,
            system_prompt,
            repair_user_content,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        repaired_scene_description_text = _try_extract_scene_description_text(repaired_raw_response)
        if repaired_scene_description_text is not None:
            return repaired_raw_response, repaired_scene_description_text
        raise ValueError(
            "LLM response is missing non-empty scene_description_text. "
            f"Raw response preview: {repaired_raw_response[:500]!r}"
        )

    def generate_from_episode(
        self,
        episode_file: str | Path,
        *,
        step_index: int,
        output_dir: str | Path,
        fps: float,
        video_types: list[str] | None = None,
    ) -> StepSceneDescriptionGenerationResult:
        episode_path = Path(episode_file).resolve()
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        extracted = extract_single_episode_step_frames(
            episode_path,
            step_index,
            destination / "frames",
            fps=fps,
            video_types=video_types,
        )
        episode_name, request = self.build_request_from_episode(
            episode_path,
            step_index=step_index,
            frame_paths=extracted.frame_paths,
            allowed_object_names=None,
            current_step_focus_objects=None,
            camera_order_top_to_bottom=video_types,
        )
        result = self.generate(episode_name=episode_name, request=request)
        self.write_outputs(destination, result)
        return result

    def write_outputs(
        self,
        output_dir: str | Path,
        result: StepSceneDescriptionGenerationResult,
    ) -> None:
        _write_generation_outputs(Path(output_dir).resolve(), result)


def _write_generation_outputs(output_dir: Path, result: StepSceneDescriptionGenerationResult) -> None:
    (output_dir / "step_scene_description.txt").write_text(result.scene_description_text + "\n", encoding="utf-8")
    (output_dir / "step_scene_description_generation.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def select_step_scene_description_frame_paths(
    frame_paths: list[str] | tuple[str, ...],
    *,
    step_index: int,
    mode: str = DEFAULT_FRAME_SELECTION_MODE,
) -> list[str]:
    normalized_mode = _normalize_frame_selection_mode(mode)
    selected = [str(path) for path in frame_paths]
    if step_index == 0 or normalized_mode == "all_sampled" or len(selected) <= 1:
        return selected
    return [selected[0], selected[-1]]


def _normalize_frame_selection_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in _VALID_FRAME_SELECTION_MODES:
        raise ValueError(
            "Unsupported step scene-description frame selection mode: "
            f"{mode!r}. Expected one of {sorted(_VALID_FRAME_SELECTION_MODES)}"
        )
    return normalized


def _extract_scene_description_text(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in _SCENE_DESCRIPTION_RESPONSE_KEYS:
            value = normalize_optional_text(payload.get(key))
            if value is not None:
                return value
        for nested_key in ("result", "output", "response", "data"):
            nested = payload.get(nested_key)
            value = _extract_scene_description_text(nested)
            if value is not None:
                return value
        for nested in payload.values():
            value = _extract_scene_description_text(nested)
            if value is not None:
                return value
        return None
    if isinstance(payload, list):
        for item in payload:
            value = _extract_scene_description_text(item)
            if value is not None:
                return value
    return None


def _try_extract_scene_description_text(raw_response: str) -> str | None:
    parsed = extract_json_object(raw_response)
    return _extract_scene_description_text(parsed)


def _append_repair_suffix_to_user_content(user_content: Any, suffix: str):
    if isinstance(user_content, str):
        return user_content + suffix
    repaired_content: list[dict[str, Any]] = []
    text_repaired = False
    for block in user_content:
        if isinstance(block, dict) and block.get("type") == "text" and not text_repaired:
            repaired_block = dict(block)
            repaired_block["text"] = str(block.get("text", "")) + suffix
            repaired_content.append(repaired_block)
            text_repaired = True
        else:
            repaired_content.append(block)
    if not text_repaired:
        repaired_content.append({"type": "text", "text": suffix.strip()})
    return repaired_content
