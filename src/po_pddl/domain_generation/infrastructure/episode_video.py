from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from po_pddl.domain_generation.infrastructure.artifact_io import load_episode_payload

_NAMED_CAMERA_VIDEO_FILE_ALIASES = {
    "camera_high": "cam_high",
    "cam_high": "cam_high",
    "left": "left",
    "right": "right",
}

_PREFERRED_CAMERA_ORDER = [
    "camera_high",
    "cam_high",
    "left",
    "right",
]


@dataclass(frozen=True)
class ExtractedStepFrames:
    step_index: int
    action_text: str | None
    start_time_sec: float | None
    end_time_sec: float | None
    frame_dir: str
    frame_paths: list[str]
    sample_timestamps_sec: list[float]


@dataclass(frozen=True)
class EpisodeFrameExtractionManifest:
    episode_name: str
    episode_file: str
    video_path: str | None
    fps: float
    image_extension: str
    steps: list[ExtractedStepFrames]
    video_paths: dict[str, str] = field(default_factory=dict)
    camera_order_top_to_bottom: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_video_frame(
    video_path: str | Path,
    output_path: str | Path,
    *,
    timestamp_sec: float = 0.0,
) -> Path:
    source = Path(video_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Video file not found: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {source}")
    try:
        frame = _read_frame_at_timestamp(capture, timestamp_sec)
    finally:
        capture.release()
    destination = Path(output_path).resolve()
    _write_frame(destination, frame)
    return destination


def resolve_episode_video_path(episode_file: str | Path) -> Path:
    episode_path = Path(episode_file)
    payload = load_episode_payload(episode_path)
    video_payload = payload.get("video")
    if not isinstance(video_payload, dict):
        raise ValueError(f"{episode_path} has invalid 'video' payload")
    relative_path = video_payload.get("file_path")
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{episode_path} is missing video.file_path")
    video_path = (episode_path.parent / relative_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found for {episode_path}: {video_path}")
    return video_path


def resolve_named_episode_video_paths(
    episode_file: str | Path,
    video_types: list[str],
) -> dict[str, Path]:
    episode_path = Path(episode_file).resolve()
    if not video_types:
        raise ValueError("video_types must not be empty")
    resolved: dict[str, Path] = {}
    for video_type in video_types:
        normalized = str(video_type).strip()
        if not normalized:
            raise ValueError("video_types must not contain empty items")
        alias = _NAMED_CAMERA_VIDEO_FILE_ALIASES.get(normalized, normalized)
        candidates = [
            episode_path.parent / f"video_{normalized}.mp4",
            episode_path.parent / f"video_{alias}.mp4",
        ]
        candidate = next((path for path in candidates if path.exists()), None)
        if candidate is None:
            raise FileNotFoundError(
                f"Video file not found for {episode_path}: expected one of "
                f"{', '.join(path.name for path in candidates)} for requested camera {normalized!r}"
            )
        resolved[normalized] = candidate.resolve()
    return resolved


def infer_named_episode_video_paths(episode_file: str | Path) -> dict[str, Path]:
    episode_path = Path(episode_file).resolve()
    resolved: dict[str, Path] = {}
    seen_paths: set[Path] = set()

    for camera_name in _PREFERRED_CAMERA_ORDER:
        alias = _NAMED_CAMERA_VIDEO_FILE_ALIASES.get(camera_name, camera_name)
        candidates = [
            episode_path.parent / f"video_{camera_name}.mp4",
            episode_path.parent / f"video_{alias}.mp4",
        ]
        candidate = next((path.resolve() for path in candidates if path.exists()), None)
        if candidate is None or candidate in seen_paths:
            continue
        resolved[camera_name] = candidate
        seen_paths.add(candidate)

    for candidate in sorted(episode_path.parent.glob("video_*.mp4")):
        resolved_name = candidate.stem.removeprefix("video_").strip()
        if not resolved_name:
            continue
        absolute_candidate = candidate.resolve()
        if absolute_candidate in seen_paths:
            continue
        resolved[resolved_name] = absolute_candidate
        seen_paths.add(absolute_candidate)

    return resolved


def extract_episode_first_frame(
    episode_file: str | Path,
    output_path: str | Path,
    *,
    video_types: list[str] | None = None,
) -> Path:
    if not video_types:
        return extract_video_frame(resolve_episode_video_path(episode_file), output_path, timestamp_sec=0.0)
    video_paths = resolve_named_episode_video_paths(episode_file, video_types)
    return extract_stacked_video_frame(video_paths, output_path, timestamp_sec=0.0)


def extract_stacked_video_frame(
    video_paths: dict[str, str | Path],
    output_path: str | Path,
    *,
    timestamp_sec: float = 0.0,
) -> Path:
    if not video_paths:
        raise ValueError("video_paths must not be empty")
    ordered_frames: list[Any] = []
    ordered_camera_names: list[str] = []
    captures: list[cv2.VideoCapture] = []
    try:
        for camera_name, path in video_paths.items():
            source = Path(path).resolve()
            if not source.exists():
                raise FileNotFoundError(f"Video file not found: {source}")
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise ValueError(f"Failed to open video: {source}")
            captures.append(capture)
            ordered_camera_names.append(str(camera_name))
            frame = _read_frame_at_timestamp(capture, timestamp_sec)
            ordered_frames.append(frame)
    finally:
        for capture in captures:
            capture.release()
    destination = Path(output_path).resolve()
    _write_frame(destination, _stack_frames_vertically(ordered_frames))
    return destination


def extract_episode_step_frames(
    episode_file: str | Path,
    output_dir: str | Path,
    *,
    fps: float,
    video_types: list[str] | None = None,
    init_video_types: list[str] | None = None,
    image_extension: str = ".jpg",
    manifest_file_name: str = "frame_manifest.json",
) -> EpisodeFrameExtractionManifest:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    normalized_ext = _normalize_image_extension(image_extension)
    episode_path = Path(episode_file).resolve()
    payload = load_episode_payload(episode_path)
    episode_name = str(payload.get("episode_name") or episode_path.parent.name)
    steps = payload.get("steps") or []
    if not isinstance(steps, list):
        raise ValueError(f"{episode_path} has non-list 'steps'")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    pre_extracted_manifest = _load_pre_extracted_frame_manifest(
        episode_path=episode_path,
        episode_payload=payload,
        destination=destination,
        manifest_file_name=manifest_file_name,
    )
    if pre_extracted_manifest is not None:
        return pre_extracted_manifest

    camera_order_top_to_bottom = list(video_types or [])
    if video_types:
        resolved_video_paths = resolve_named_episode_video_paths(episode_path, video_types)
        ordered_camera_names = list(resolved_video_paths.keys())
        captures = _open_video_captures(resolved_video_paths.values())
        video_path = None
    else:
        try:
            single_video_path = resolve_episode_video_path(episode_path)
        except (ValueError, FileNotFoundError):
            inferred_video_paths = infer_named_episode_video_paths(episode_path)
            if not inferred_video_paths:
                raise
            resolved_video_paths = inferred_video_paths
            ordered_camera_names = list(resolved_video_paths.keys())
            camera_order_top_to_bottom = ordered_camera_names
            captures = _open_video_captures(resolved_video_paths.values())
            video_path = None
        else:
            resolved_video_paths = {}
            ordered_camera_names = []
            captures = _open_video_captures([single_video_path])
            video_path = str(single_video_path)
    if init_video_types:
        resolved_init_video_paths = resolve_named_episode_video_paths(episode_path, init_video_types)
        ordered_init_camera_names = list(resolved_init_video_paths.keys())
        init_captures = _open_video_captures(resolved_init_video_paths.values())
    else:
        resolved_init_video_paths = {}
        ordered_init_camera_names = []
        init_captures = []

    try:
        extracted_steps: list[ExtractedStepFrames] = []
        for raw_step in steps:
            if not isinstance(raw_step, dict):
                raise ValueError(f"{episode_path} contains non-object step: {raw_step!r}")
            step_index = _parse_step_index(raw_step, episode_path)
            step_dir = destination / f"step_{step_index:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            end_time_sec = _coerce_optional_float(raw_step.get("end_time_sec"))

            timestamps = _build_step_sample_timestamps(
                raw_step,
                fps=fps,
            )
            frame_paths: list[str] = []
            for frame_index, timestamp_sec in enumerate(timestamps):
                current_captures = init_captures if step_index == 0 and init_captures else captures
                current_camera_names = (
                    ordered_init_camera_names if step_index == 0 and init_captures else ordered_camera_names
                )
                seek_mode = (
                    "forward" if end_time_sec is not None and abs(timestamp_sec - end_time_sec) <= 1e-6 else "default"
                )
                ordered_frames = [
                    _read_frame_at_timestamp(capture, timestamp_sec, seek_mode=seek_mode)
                    for capture, camera_name in zip(
                        current_captures,
                        current_camera_names or [None] * len(current_captures),
                        strict=False,
                    )
                ]
                frame = _stack_frames_vertically(ordered_frames) if len(ordered_frames) > 1 else ordered_frames[0]
                frame_name = f"frame_{frame_index:04d}_t{int(round(timestamp_sec * 1000)):08d}ms{normalized_ext}"
                frame_path = step_dir / frame_name
                _write_frame(frame_path, frame)
                frame_paths.append(str(frame_path))

            extracted_steps.append(
                ExtractedStepFrames(
                    step_index=step_index,
                    action_text=_coerce_optional_string(raw_step.get("action_text")),
                    start_time_sec=_coerce_optional_float(raw_step.get("start_time_sec")),
                    end_time_sec=_coerce_optional_float(raw_step.get("end_time_sec")),
                    frame_dir=str(step_dir),
                    frame_paths=frame_paths,
                    sample_timestamps_sec=timestamps,
                )
            )

        manifest = EpisodeFrameExtractionManifest(
            episode_name=episode_name,
            episode_file=str(episode_path),
            video_path=video_path,
            video_paths={key: str(path) for key, path in resolved_video_paths.items()},
            camera_order_top_to_bottom=camera_order_top_to_bottom,
            fps=float(fps),
            image_extension=normalized_ext,
            steps=extracted_steps,
        )
        manifest_path = destination / manifest_file_name
        manifest_path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest
    finally:
        for capture in captures:
            capture.release()
        for capture in init_captures:
            capture.release()


def extract_single_episode_step_frames(
    episode_file: str | Path,
    step_index: int,
    output_dir: str | Path,
    *,
    fps: float,
    video_types: list[str] | None = None,
    image_extension: str = ".jpg",
) -> ExtractedStepFrames:
    if step_index < 0:
        raise ValueError(f"step_index must be non-negative, got {step_index}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    normalized_ext = _normalize_image_extension(image_extension)
    episode_path = Path(episode_file).resolve()
    payload = load_episode_payload(episode_path)
    steps = payload.get("steps") or []
    if not isinstance(steps, list):
        raise ValueError(f"{episode_path} has non-list 'steps'")

    selected_step: dict[str, Any] | None = None
    for raw_step in steps:
        if isinstance(raw_step, dict) and raw_step.get("step_index") == step_index:
            selected_step = raw_step
            break
    if selected_step is None:
        raise ValueError(f"{episode_path} does not contain step_index={step_index}")

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    pre_extracted_manifest = _load_pre_extracted_frame_manifest(
        episode_path=episode_path,
        episode_payload=payload,
        destination=destination,
        manifest_file_name="frame_manifest.json",
    )
    if pre_extracted_manifest is not None:
        for extracted_step in pre_extracted_manifest.steps:
            if extracted_step.step_index == step_index:
                (destination / "step_frame_manifest.json").write_text(
                    json.dumps(asdict(extracted_step), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                return extracted_step
        raise ValueError(f"Pre-extracted frame manifest for {episode_path} does not contain step_index={step_index}")

    step_dir = destination / f"step_{step_index:03d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    if video_types:
        resolved_video_paths = resolve_named_episode_video_paths(episode_path, video_types)
        ordered_camera_names = list(resolved_video_paths.keys())
        captures = _open_video_captures(resolved_video_paths.values())
    else:
        ordered_camera_names = []
        captures = _open_video_captures([resolve_episode_video_path(episode_path)])
    try:
        timestamps = _build_step_sample_timestamps(selected_step, fps=fps)
        end_time_sec = _coerce_optional_float(selected_step.get("end_time_sec"))
        frame_paths: list[str] = []
        for frame_index, timestamp_sec in enumerate(timestamps):
            seek_mode = (
                "forward" if end_time_sec is not None and abs(timestamp_sec - end_time_sec) <= 1e-6 else "default"
            )
            ordered_frames = [
                _read_frame_at_timestamp(capture, timestamp_sec, seek_mode=seek_mode)
                for capture, camera_name in zip(captures, ordered_camera_names or [None] * len(captures), strict=False)
            ]
            frame = _stack_frames_vertically(ordered_frames) if len(ordered_frames) > 1 else ordered_frames[0]
            frame_name = f"frame_{frame_index:04d}_t{int(round(timestamp_sec * 1000)):08d}ms{normalized_ext}"
            frame_path = step_dir / frame_name
            _write_frame(frame_path, frame)
            frame_paths.append(str(frame_path))
    finally:
        for capture in captures:
            capture.release()

    extracted = ExtractedStepFrames(
        step_index=step_index,
        action_text=_coerce_optional_string(selected_step.get("action_text")),
        start_time_sec=_coerce_optional_float(selected_step.get("start_time_sec")),
        end_time_sec=_coerce_optional_float(selected_step.get("end_time_sec")),
        frame_dir=str(step_dir),
        frame_paths=frame_paths,
        sample_timestamps_sec=timestamps,
    )
    (destination / "step_frame_manifest.json").write_text(
        json.dumps(asdict(extracted), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return extracted


def _load_pre_extracted_frame_manifest(
    *,
    episode_path: Path,
    episode_payload: dict[str, Any],
    destination: Path,
    manifest_file_name: str,
) -> EpisodeFrameExtractionManifest | None:
    declaration = episode_payload.get("pre_extracted_frames")
    if declaration is None:
        return None
    if not isinstance(declaration, dict):
        raise ValueError(f"{episode_path} has invalid 'pre_extracted_frames' payload")
    relative_manifest_path = declaration.get("manifest_path")
    if not isinstance(relative_manifest_path, str) or not relative_manifest_path.strip():
        raise ValueError(f"{episode_path} is missing pre_extracted_frames.manifest_path")

    episode_root = episode_path.parent.resolve()
    source_manifest_path = (episode_root / relative_manifest_path).resolve()
    if not source_manifest_path.is_relative_to(episode_root):
        raise ValueError(f"Pre-extracted frame manifest must be inside the episode directory: {source_manifest_path}")
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Pre-extracted frame manifest not found for {episode_path}: {source_manifest_path}")

    manifest_payload = json.loads(source_manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest_payload, dict):
        raise ValueError(f"Expected JSON object at {source_manifest_path}")
    manifest_steps = manifest_payload.get("steps")
    if not isinstance(manifest_steps, list):
        raise ValueError(f"{source_manifest_path} has non-list 'steps'")

    annotation_steps = {
        _parse_step_index(step, episode_path): step
        for step in (episode_payload.get("steps") or [])
        if isinstance(step, dict)
    }
    manifest_rows: dict[int, dict[str, Any]] = {}
    for row in manifest_steps:
        if not isinstance(row, dict):
            raise ValueError(f"{source_manifest_path} contains a non-object step")
        row_index = _parse_step_index(row, source_manifest_path)
        if row_index in manifest_rows:
            raise ValueError(f"{source_manifest_path} contains duplicate step_index={row_index}")
        manifest_rows[row_index] = row

    missing_indices = sorted(set(annotation_steps) - set(manifest_rows))
    if missing_indices:
        raise ValueError(f"{source_manifest_path} is missing annotated steps: {missing_indices}")

    extracted_steps: list[ExtractedStepFrames] = []
    for step_index, annotation_step in sorted(annotation_steps.items()):
        row = manifest_rows[step_index]
        raw_frame_paths = row.get("frame_paths")
        if not isinstance(raw_frame_paths, list) or not raw_frame_paths:
            raise ValueError(f"{source_manifest_path} step {step_index} has no frame_paths")
        frame_paths = [
            str(
                _resolve_pre_extracted_frame_path(
                    raw_path=raw_path,
                    episode_root=episode_root,
                    manifest_path=source_manifest_path,
                )
            )
            for raw_path in raw_frame_paths
        ]
        raw_timestamps = row.get("sample_timestamps_sec")
        if raw_timestamps is None:
            sample_timestamps = []
        elif isinstance(raw_timestamps, list):
            sample_timestamps = [float(value) for value in raw_timestamps]
        else:
            raise ValueError(f"{source_manifest_path} step {step_index} has invalid sample_timestamps_sec")
        if sample_timestamps and len(sample_timestamps) != len(frame_paths):
            raise ValueError(f"{source_manifest_path} step {step_index} has mismatched frame and timestamp counts")
        extracted_steps.append(
            ExtractedStepFrames(
                step_index=step_index,
                action_text=_coerce_optional_string(annotation_step.get("action_text")),
                start_time_sec=_coerce_optional_float(annotation_step.get("start_time_sec")),
                end_time_sec=_coerce_optional_float(annotation_step.get("end_time_sec")),
                frame_dir=str(Path(frame_paths[0]).parent),
                frame_paths=frame_paths,
                sample_timestamps_sec=sample_timestamps,
            )
        )

    manifest = EpisodeFrameExtractionManifest(
        episode_name=str(episode_payload.get("episode_name") or episode_root.name),
        episode_file=str(episode_path),
        video_path=None,
        video_paths={},
        camera_order_top_to_bottom=[str(value) for value in manifest_payload.get("camera_order_top_to_bottom", [])],
        fps=float(manifest_payload.get("fps") or episode_payload.get("fps") or 1.0),
        image_extension=str(manifest_payload.get("image_extension") or ".jpg"),
        steps=extracted_steps,
    )
    (destination / manifest_file_name).write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _resolve_pre_extracted_frame_path(
    *,
    raw_path: Any,
    episode_root: Path,
    manifest_path: Path,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{manifest_path} contains an invalid frame path: {raw_path!r}")
    declared = Path(raw_path)
    candidates = (
        [declared.resolve()]
        if declared.is_absolute()
        else [
            (episode_root / declared).resolve(),
            (manifest_path.parent / declared).resolve(),
        ]
    )
    for candidate in candidates:
        if candidate.is_relative_to(episode_root) and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Frame {raw_path!r} declared by {manifest_path} was not found inside {episode_root}")


def _normalize_image_extension(image_extension: str) -> str:
    normalized = image_extension.strip().lower()
    if not normalized:
        raise ValueError("image_extension must not be empty")
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    if normalized not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unsupported image extension: {image_extension}")
    return normalized


def _parse_step_index(step: dict[str, Any], episode_path: Path) -> int:
    value = step.get("step_index")
    if not isinstance(value, int):
        raise ValueError(f"{episode_path} has step with invalid step_index: {value!r}")
    return value


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Expected float-compatible value, got {value!r}")


def _build_step_sample_timestamps(
    step: dict[str, Any],
    *,
    fps: float,
) -> list[float]:
    step_index = step.get("step_index")
    if step_index == 0:
        return [0.0]

    start = _coerce_optional_float(step.get("start_time_sec"))
    end = _coerce_optional_float(step.get("end_time_sec"))
    if start is None or end is None:
        raise ValueError(f"Step {step_index} is missing start_time_sec/end_time_sec")
    if end < start:
        raise ValueError(f"Step {step_index} has end_time_sec < start_time_sec")
    if end == start:
        return [start]

    step_size = 1.0 / fps
    timestamps: list[float] = []
    current = start
    epsilon = 1e-9
    while current < end - epsilon:
        timestamps.append(round(current, 6))
        current += step_size
    if not timestamps:
        timestamps.append(round(start, 6))
    end_rounded = round(end, 6)
    if abs(timestamps[-1] - end_rounded) > 1e-6:
        timestamps.append(end_rounded)
    return timestamps


def _read_frame_at_timestamp(
    capture: cv2.VideoCapture,
    timestamp_sec: float,
    *,
    seek_mode: str = "default",
) -> Any:
    if seek_mode == "forward":
        _set_capture_to_forward_frame_at_or_after_timestamp(capture, timestamp_sec)
    else:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(timestamp_sec, 0.0) * 1000.0)
    success, frame = capture.read()
    if not success or frame is None:
        raise ValueError(f"Failed to read frame at {timestamp_sec:.3f}s")
    return frame


def _set_capture_to_forward_frame_at_or_after_timestamp(
    capture: cv2.VideoCapture,
    timestamp_sec: float,
) -> None:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0.0 or frame_count <= 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(timestamp_sec, 0.0) * 1000.0)
        return
    epsilon = 1e-9
    frame_index = int(np.ceil(max(timestamp_sec, 0.0) * fps - epsilon))
    frame_index = max(0, min(frame_index, frame_count - 1))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)


def _write_frame(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), frame)
    if not success:
        raise ValueError(f"Failed to write frame to {path}")


def _open_video_captures(paths: list[Path] | list[str | Path]) -> list[cv2.VideoCapture]:
    captures: list[cv2.VideoCapture] = []
    for path in paths:
        source = Path(path).resolve()
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            for opened in captures:
                opened.release()
            raise ValueError(f"Failed to open video: {source}")
        captures.append(capture)
    return captures


def _stack_frames_vertically(frames: list[Any]) -> Any:
    if not frames:
        raise ValueError("frames must not be empty")
    max_width = max(int(frame.shape[1]) for frame in frames)
    padded_frames: list[Any] = []
    for frame in frames:
        width = int(frame.shape[1])
        if width == max_width:
            padded_frames.append(frame)
            continue
        height = int(frame.shape[0])
        pad_width = max_width - width
        padded_frames.append(
            np.pad(
                frame,
                ((0, 0), (0, pad_width), (0, 0)),
                mode="constant",
                constant_values=0,
            ).reshape(height, max_width, frame.shape[2])
        )
    return np.vstack(padded_frames)
