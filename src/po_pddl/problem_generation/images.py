from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class PreparedSceneImage:
    image_path: Path
    source_image_paths: list[Path]
    image_input_note: str
    is_stitched_multiview: bool


def _list_scene_images(image_input: Path) -> list[Path]:
    return sorted(
        [path for path in image_input.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES],
        key=lambda path: path.name.lower(),
    )


def _stitch_images_vertically(image_paths: list[Path], output_path: Path) -> None:
    images = [_prepare_stitched_image(path) for path in image_paths]
    try:
        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
        y_offset = 0
        for image in images:
            canvas.paste(image, (0, y_offset))
            y_offset += image.height
        canvas.save(output_path)
    finally:
        for image in images:
            image.close()


def _prepare_stitched_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _build_multiview_note(source_image_paths: list[Path]) -> str:
    ordered_names = [path.name for path in source_image_paths]
    ordered_text = ", ".join(ordered_names)
    return (
        "The provided image is a vertical top-to-bottom concatenation of multiple camera views. "
        f"Interpret the stitched image in this exact order from top to bottom: {ordered_text}. "
        "Each segment is a different viewpoint of the same initial scene."
    )


@contextmanager
def prepare_scene_image(image_input: str | Path):
    resolved_input = Path(image_input)
    if not resolved_input.exists():
        raise FileNotFoundError(f"Scene image input not found: {resolved_input}")

    if resolved_input.is_dir():
        source_image_paths = _list_scene_images(resolved_input)
        if not source_image_paths:
            raise FileNotFoundError(f"No image files found in scene image directory: {resolved_input}")
        with tempfile.TemporaryDirectory(prefix="online_planning_scene_") as temp_dir:
            stitched_path = Path(temp_dir) / "stitched_scene.png"
            _stitch_images_vertically(source_image_paths, stitched_path)
            yield PreparedSceneImage(
                image_path=stitched_path,
                source_image_paths=source_image_paths,
                image_input_note=_build_multiview_note(source_image_paths),
                is_stitched_multiview=True,
            )
        return

    yield PreparedSceneImage(
        image_path=resolved_input,
        source_image_paths=[resolved_input],
        image_input_note="The provided image is a single-view snapshot of the initial scene.",
        is_stitched_multiview=False,
    )
