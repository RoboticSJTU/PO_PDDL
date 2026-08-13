from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from po_pddl.domain_generation.infrastructure.artifact_io import load_episode_payload
from po_pddl.domain_generation.infrastructure.episode_video import (
    EpisodeFrameExtractionManifest,
    extract_episode_step_frames,
)
from po_pddl.domain_generation.infrastructure.payload_utils import normalize_optional_text
from po_pddl.domain_generation.stages.scene_description.init_scene_description import (
    InitSceneDescriptionGenerator,
)
from po_pddl.domain_generation.stages.scene_description.step_scene_description import (
    StepSceneDescriptionGenerator,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneDescriptionResult:
    episode_name: str
    episode_file: str
    annotated_episode_file: str
    frame_manifest_file: str
    fps: float
    camera_order_top_to_bottom: list[str]
    generated_step_count: int
    step_output_dirs: dict[int, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SceneDescriptionRunner:
    def __init__(
        self,
        *,
        init_generator: InitSceneDescriptionGenerator,
        step_generator: StepSceneDescriptionGenerator,
        max_generation_attempts: int = 3,
    ) -> None:
        self.init_generator = init_generator
        self.step_generator = step_generator
        self.max_generation_attempts = max(1, int(max_generation_attempts))

    def run(
        self,
        *,
        episode_file: str | Path,
        output_dir: str | Path,
        fps: float,
        allowed_object_names: list[str] | None = None,
        focus_object_names_by_step: dict[int, list[str]] | None = None,
        video_types: list[str] | None = None,
    ) -> SceneDescriptionResult:
        episode_path = Path(episode_file).resolve()
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)

        frames_root = destination / "frames"
        manifest = extract_episode_step_frames(
            episode_path,
            frames_root,
            fps=fps,
            video_types=video_types,
            init_video_types=_select_init_scene_description_camera_types(video_types),
        )
        manifest_file = frames_root / "frame_manifest.json"

        annotated_payload = load_episode_payload(episode_path)
        episode_name = str(annotated_payload.get("episode_name") or episode_path.parent.name)
        if allowed_object_names:
            annotated_payload["allowed_object_names"] = list(allowed_object_names)
        annotated_episode_file = destination / "annotated_episode.json"

        step_output_dirs: dict[int, str] = {}

        init_manifest_step = _select_init_manifest_step(manifest)
        if init_manifest_step is None or not init_manifest_step.frame_paths:
            raise ValueError(f"Failed to extract init scene-description frame for {episode_path}")
        init_output_dir = destination / "step_000_init"
        init_output_dir.mkdir(parents=True, exist_ok=True)
        init_episode_name, init_request = self.init_generator.build_request_from_episode(
            episode_path,
            init_manifest_step.frame_paths[0],
            allowed_object_names=allowed_object_names,
            camera_order_top_to_bottom=[],
        )
        init_result = self._generate_with_retry(
            lambda: self.init_generator.generate(episode_name=init_episode_name, request=init_request),
            label=f"{episode_name} step 0",
        )
        self.init_generator.write_outputs(init_output_dir, init_result)
        _set_step_scene_description(annotated_payload, 0, init_result.scene_description_text)
        _write_episode_payload(annotated_episode_file, annotated_payload)
        step_output_dirs[0] = str(init_output_dir)
        logger.debug("Generated scene description for step 0: %s", init_result.scene_description_text)

        for manifest_step in manifest.steps:
            if manifest_step.step_index == 0:
                continue
            step_payload = _find_payload_step(annotated_payload, manifest_step.step_index)
            if step_payload is None:
                continue
            current_action_text = normalize_optional_text(step_payload.get("action_text"))
            if current_action_text is None:
                continue
            step_output_dir = destination / f"step_{manifest_step.step_index:03d}"
            step_output_dir.mkdir(parents=True, exist_ok=True)
            current_episode_name, request = self.step_generator.build_request_from_episode(
                annotated_episode_file,
                step_index=manifest_step.step_index,
                frame_paths=manifest_step.frame_paths,
                allowed_object_names=allowed_object_names,
                current_step_focus_objects=(focus_object_names_by_step or {}).get(manifest_step.step_index, []),
                camera_order_top_to_bottom=manifest.camera_order_top_to_bottom,
            )
            step_result = self._generate_with_retry(
                lambda: self.step_generator.generate(episode_name=current_episode_name, request=request),
                label=f"{episode_name} step {manifest_step.step_index}",
            )
            self.step_generator.write_outputs(step_output_dir, step_result)
            _set_step_scene_description(annotated_payload, manifest_step.step_index, step_result.scene_description_text)
            _write_episode_payload(annotated_episode_file, annotated_payload)
            step_output_dirs[manifest_step.step_index] = str(step_output_dir)
            logger.debug(
                "Generated scene description for step %d (%s): %s",
                manifest_step.step_index,
                current_action_text,
                step_result.scene_description_text,
            )

        result = SceneDescriptionResult(
            episode_name=episode_name,
            episode_file=str(episode_path),
            annotated_episode_file=str(annotated_episode_file),
            frame_manifest_file=str(manifest_file),
            fps=float(fps),
            camera_order_top_to_bottom=list(manifest.camera_order_top_to_bottom),
            generated_step_count=len(step_output_dirs),
            step_output_dirs=step_output_dirs,
        )
        (destination / "scene_description_summary.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    def _generate_with_retry(self, generate_fn, *, label: str):
        last_error: Exception | None = None
        for attempt in range(1, self.max_generation_attempts + 1):
            try:
                return generate_fn()
            except ValueError as exc:
                if not _is_retryable_scene_description_error(exc):
                    raise
                last_error = exc
                if attempt >= self.max_generation_attempts:
                    break
                logger.warning(
                    "Scene description generation parse failure for %s on attempt %d/%d: %s. Retrying.",
                    label,
                    attempt,
                    self.max_generation_attempts,
                    exc,
                )
        assert last_error is not None
        raise last_error


def _find_manifest_step(manifest: EpisodeFrameExtractionManifest, step_index: int):
    for step in manifest.steps:
        if step.step_index == step_index:
            return step
    return None


def _select_init_manifest_step(manifest: EpisodeFrameExtractionManifest):
    return _find_manifest_step(manifest, 0)


def _select_init_scene_description_camera_types(video_types: list[str] | None) -> list[str] | None:
    if not video_types:
        return None
    normalized = [str(item).strip() for item in video_types if str(item).strip()]
    if "camera_high" in normalized:
        return ["camera_high"]
    if "cam_high" in normalized:
        return ["cam_high"]
    return [normalized[0]] if normalized else None


def _find_payload_step(payload: dict[str, Any], step_index: int) -> dict[str, Any] | None:
    steps = payload.get("steps") or []
    if not isinstance(steps, list):
        return None
    for item in steps:
        if isinstance(item, dict) and item.get("step_index") == step_index:
            return item
    return None


def _set_step_scene_description(payload: dict[str, Any], step_index: int, scene_description_text: str) -> None:
    step = _find_payload_step(payload, step_index)
    if step is None:
        raise ValueError(f"Episode payload does not contain step_index={step_index}")
    step["observation_text"] = scene_description_text


def _write_episode_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _is_retryable_scene_description_error(error: ValueError) -> bool:
    message = str(error)
    return "scene_description_text" in message or "valid JSON object" in message or "Raw response preview" in message
