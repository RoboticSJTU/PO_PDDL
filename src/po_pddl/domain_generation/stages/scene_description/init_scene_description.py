from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from po_pddl.config import DEFAULT_MODEL
from po_pddl.domain_generation.infrastructure.artifact_io import load_episode_payload
from po_pddl.domain_generation.infrastructure.episode_video import (
    extract_episode_first_frame,
)
from po_pddl.domain_generation.infrastructure.llm_shared import (
    build_user_content,
    load_llm_config,
    make_client,
    safe_chat,
)
from po_pddl.domain_generation.infrastructure.payload_utils import normalize_optional_text
from po_pddl.domain_generation.infrastructure.response_parsing import extract_json_object
from po_pddl.prompts import load_prompt

from .text_normalization import remove_unseen_object_statements

DEFAULT_PROMPT_NAME = "init_scene_description_prompt.md"
DEFAULT_SYSTEM_PROMPT = (
    "You generate concise, visually grounded English scene descriptions from an initial frame. "
    "Only describe visually supported objects that appear in the provided allowed object list. "
    "Instruction and action context only tell you which objects and regions deserve attention and how to name them; "
    "they are not evidence."
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
class InitSceneDescriptionRequest:
    instruction: str
    first_frame_path: str
    action_text_sequence: list[str]
    allowed_object_names: list[str]
    camera_order_top_to_bottom: list[str]

    def to_user_payload(self) -> str:
        ordered_actions = [
            {"order": index + 1, "action_text": action_text}
            for index, action_text in enumerate(self.action_text_sequence)
        ]
        payload = {
            "instruction": self.instruction,
            "future_action_sequence": ordered_actions,
            "allowed_object_names": list(self.allowed_object_names),
        }
        if self.camera_order_top_to_bottom:
            payload["camera_order_top_to_bottom"] = list(self.camera_order_top_to_bottom)
        return json.dumps(payload, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class InitSceneDescriptionGenerationResult:
    episode_name: str
    instruction: str
    first_frame_path: str
    action_text_sequence: list[str]
    camera_order_top_to_bottom: list[str]
    scene_description_text: str
    raw_response: str
    model: str
    prompt_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InitSceneDescriptionGenerator:
    def __init__(
        self,
        *,
        config_path: str | None = None,
        config_name: str = "openai_config",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1200,
        prompt_name: str = DEFAULT_PROMPT_NAME,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
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
        self.image_detail = image_detail
        self.verbose = verbose
        self._prompt = load_prompt(prompt_name)
        self._client = client if client is not None else make_client(api_key=self.api_key, base_url=self.base_url)

    def build_request_from_episode(
        self,
        episode_file: str | Path,
        first_frame_path: str | Path,
        *,
        allowed_object_names: list[str] | None = None,
        camera_order_top_to_bottom: list[str] | None = None,
    ) -> tuple[str, InitSceneDescriptionRequest]:
        episode_path = Path(episode_file).resolve()
        payload = load_episode_payload(episode_path)
        instruction = normalize_optional_text(payload.get("instruction"))
        if instruction is None:
            raise ValueError(f"{episode_path} is missing instruction")
        action_text_sequence = _extract_action_sequence(payload)
        request = InitSceneDescriptionRequest(
            instruction=instruction,
            first_frame_path=str(Path(first_frame_path).resolve()),
            action_text_sequence=action_text_sequence,
            allowed_object_names=list(allowed_object_names or []),
            camera_order_top_to_bottom=list(camera_order_top_to_bottom or []),
        )
        episode_name = str(payload.get("episode_name") or episode_path.parent.name)
        return episode_name, request

    def generate(
        self,
        *,
        episode_name: str,
        request: InitSceneDescriptionRequest,
    ) -> InitSceneDescriptionGenerationResult:
        system_prompt = self.system_prompt
        if request.camera_order_top_to_bottom:
            system_prompt = (
                f"{system_prompt} Each input image is a vertical stack of synchronized camera views. "
                f"From top to bottom the camera order is: {', '.join(request.camera_order_top_to_bottom)}."
            )
        user_prompt = f"{self._prompt.strip()}\n\nInput JSON:\n{request.to_user_payload()}"
        user_content = build_user_content(
            user_prompt,
            image_path=request.first_frame_path,
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
        scene_description_text = remove_unseen_object_statements(
            scene_description_text,
            request.allowed_object_names,
        )
        return InitSceneDescriptionGenerationResult(
            episode_name=episode_name,
            instruction=request.instruction,
            first_frame_path=request.first_frame_path,
            action_text_sequence=list(request.action_text_sequence),
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
        user_content: list[dict[str, Any]],
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
        output_dir: str | Path,
        *,
        video_types: list[str] | None = None,
    ) -> InitSceneDescriptionGenerationResult:
        episode_path = Path(episode_file).resolve()
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        first_frame_path = _extract_init_reference_frame(
            episode_path,
            destination,
            video_types=video_types,
        )
        episode_name, request = self.build_request_from_episode(
            episode_path,
            first_frame_path,
            allowed_object_names=None,
            camera_order_top_to_bottom=[],
        )
        result = self.generate(episode_name=episode_name, request=request)
        self.write_outputs(destination, result)
        return result

    def write_outputs(
        self,
        output_dir: str | Path,
        result: InitSceneDescriptionGenerationResult,
    ) -> None:
        _write_generation_outputs(Path(output_dir).resolve(), result)


def _extract_action_sequence(payload: dict[str, Any]) -> list[str]:
    steps = payload.get("steps") or []
    if not isinstance(steps, list):
        raise ValueError("Episode payload has non-list steps")
    action_texts: list[str] = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        step_index = item.get("step_index")
        if step_index == 0:
            continue
        action_text = normalize_optional_text(item.get("action_text"))
        if action_text is not None:
            action_texts.append(action_text)
    return action_texts


def _extract_init_reference_frame(
    episode_path: Path,
    destination: Path,
    *,
    video_types: list[str] | None = None,
) -> Path:
    init_video_types = _select_init_scene_description_camera_types(video_types)
    return extract_episode_first_frame(
        episode_path,
        destination / "step0_first_frame.jpg",
        video_types=init_video_types,
    )


def _select_init_scene_description_camera_types(video_types: list[str] | None) -> list[str] | None:
    if not video_types:
        return None
    normalized = [str(item).strip() for item in video_types if str(item).strip()]
    if "camera_high" in normalized:
        return ["camera_high"]
    if "cam_high" in normalized:
        return ["cam_high"]
    return [normalized[0]] if normalized else None


def _write_generation_outputs(output_dir: Path, result: InitSceneDescriptionGenerationResult) -> None:
    (output_dir / "init_scene_description.txt").write_text(result.scene_description_text + "\n", encoding="utf-8")
    (output_dir / "init_scene_description_generation.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
