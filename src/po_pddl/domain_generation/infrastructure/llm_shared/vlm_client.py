from __future__ import annotations

import base64
import mimetypes
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from po_pddl.config import DEFAULT_MODEL

from .config import load_llm_config
from .llm_client import (
    build_image_url_block,
    build_local_image_block,
    build_text_block,
    make_client,
    safe_chat,
)

DEFAULT_VLM_SYSTEM_PROMPT = "You are a helpful vision-language assistant."
MAX_VIDEO_DATA_URL_CHARS = 19_000_000
MAX_VIDEO_RAW_BYTES = 14_000_000


def build_video_url_block(url: str, *, fps: Optional[int] = None) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "video_url", "video_url": {"url": url}}
    if fps is not None:
        block["fps"] = fps
    return block


def encode_video_to_data_url(video_path: str) -> str:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "video/mp4"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _estimate_video_data_url_length(video_path: str) -> int:
    path = Path(video_path)
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "video/mp4"
    prefix = f"data:{mime};base64,"
    return len(prefix) + ((path.stat().st_size + 2) // 3) * 4


def _compress_video_for_data_url(video_path: str, *, max_raw_bytes: int = MAX_VIDEO_RAW_BYTES) -> str:
    path = Path(video_path)
    attempts = [
        ("960:-2", "32"),
        ("768:-2", "34"),
        ("640:-2", "36"),
        ("512:-2", "38"),
        ("426:-2", "40"),
    ]
    with tempfile.TemporaryDirectory(prefix="vlm_video_compress_") as tmpdir:
        for index, (scale, crf) in enumerate(attempts):
            output_path = Path(tmpdir) / f"compressed_{index:02d}.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(path),
                    "-an",
                    "-vf",
                    f"scale={scale}",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    crf,
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                capture_output=True,
                check=True,
            )
            if (
                output_path.stat().st_size <= max_raw_bytes
                and _estimate_video_data_url_length(str(output_path)) <= MAX_VIDEO_DATA_URL_CHARS
            ):
                return encode_video_to_data_url(str(output_path))
    raise ValueError(
        "Local video is too large to send as a data URL even after compression. "
        "Please host the video at an accessible URL and use video_url instead."
    )


def prepare_video_url_for_vlm(video_path: str) -> str:
    estimated_length = _estimate_video_data_url_length(video_path)
    if estimated_length <= MAX_VIDEO_DATA_URL_CHARS and Path(video_path).stat().st_size <= MAX_VIDEO_RAW_BYTES:
        return encode_video_to_data_url(video_path)
    return _compress_video_for_data_url(video_path)


def build_local_video_block(video_path: str, *, fps: Optional[int] = None) -> dict[str, Any]:
    return build_video_url_block(prepare_video_url_for_vlm(video_path), fps=fps)


def build_vlm_user_content(
    *,
    text: Optional[str] = None,
    image_url: Optional[str] = None,
    image_path: Optional[str] = None,
    video_url: Optional[str] = None,
    video_path: Optional[str] = None,
    fps: Optional[int] = None,
    extra_blocks: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if image_url and image_path:
        raise ValueError("Pass either image_url or image_path, not both.")
    if video_url and video_path:
        raise ValueError("Pass either video_url or video_path, not both.")
    if image_url is not None:
        blocks.append(build_image_url_block(image_url))
    if image_path is not None:
        blocks.append(build_local_image_block(image_path))
    if video_url is not None:
        blocks.append(build_video_url_block(video_url, fps=fps))
    if video_path is not None:
        blocks.append(build_local_video_block(video_path, fps=fps))
    if text:
        blocks.append(build_text_block(text))
    if extra_blocks:
        blocks.extend(extra_blocks)
    if not blocks:
        raise ValueError("VLM user content must include at least one text/image/video block.")
    return blocks


def make_vlm_client(
    *,
    config_path: Optional[str] = None,
    config_name: str = "openai_config",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    if api_key is None or base_url is None:
        config = load_llm_config(config_path=config_path, config_name=config_name)
        api_key = api_key or config.get("api_key")
        base_url = base_url or config.get("base_url")
    return make_client(api_key=api_key, base_url=base_url)


def safe_vlm_chat(
    client,
    *,
    user_content: list[dict[str, Any]],
    system_prompt: str = DEFAULT_VLM_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
    verbose: bool = False,
) -> str:
    return safe_chat(
        client,
        system_prompt,
        user_content,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        initial_backoff=initial_backoff,
        verbose=verbose,
    )


def ask_video_question(
    *,
    question: str,
    video_url: Optional[str] = None,
    video_path: Optional[str] = None,
    config_path: Optional[str] = None,
    config_name: str = "openai_config",
    model: Optional[str] = None,
    fps: Optional[int] = None,
    system_prompt: str = DEFAULT_VLM_SYSTEM_PROMPT,
    temperature: Optional[float] = None,
    max_tokens: int = 4096,
    verbose: bool = False,
) -> str:
    config = load_llm_config(config_path=config_path, config_name=config_name)
    client = make_vlm_client(
        config_path=config_path,
        config_name=config_name,
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
    )
    return safe_vlm_chat(
        client,
        user_content=build_vlm_user_content(text=question, video_url=video_url, video_path=video_path, fps=fps),
        system_prompt=system_prompt,
        model=model or config.get("model") or DEFAULT_MODEL,
        temperature=temperature if temperature is not None else (config.get("temperature") or 0.2),
        max_tokens=max_tokens,
        verbose=verbose,
    )
