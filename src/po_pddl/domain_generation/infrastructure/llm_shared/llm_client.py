"""
llm_client.py  –  Shared LLM call helpers with automatic retry

Provides a single ``safe_chat()`` function used by every LLM module in
``pomdpddl.llm``.  Retry logic handles:

  * Rate-limit errors (HTTP 429 / ``RateLimitError``)
  * Transient server errors (HTTP 5xx / ``APIStatusError``)
  * Connection / timeout errors (``APIConnectionError``, ``APITimeoutError``)
  * Empty responses from the model

All retries use exponential back-off with jitter.

Public helpers
--------------
    make_client(…)                -> OpenAI
    encode_image_to_data_url(…)   -> str
    build_text_block(…)           -> dict
    build_image_url_block(…)      -> dict
    build_local_image_block(…)    -> dict
    build_user_content(…)         -> str | list
    safe_chat(…)                  -> str
"""

from __future__ import annotations

import base64
import logging
import math
import mimetypes
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Optional

from po_pddl.config import DEFAULT_MODEL

logger = logging.getLogger(__name__)


_USAGE_LOCK = threading.Lock()
_USAGE_TOTALS: dict[str, int] = {
    "total_calls": 0,
    "llm_calls": 0,
    "vlm_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}


def _zero_usage_snapshot() -> dict[str, int]:
    return {
        "total_calls": 0,
        "llm_calls": 0,
        "vlm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def reset_usage_tracking() -> None:
    with _USAGE_LOCK:
        _USAGE_TOTALS["total_calls"] = 0
        _USAGE_TOTALS["llm_calls"] = 0
        _USAGE_TOTALS["vlm_calls"] = 0
        _USAGE_TOTALS["input_tokens"] = 0
        _USAGE_TOTALS["output_tokens"] = 0


def get_usage_snapshot() -> dict[str, int]:
    with _USAGE_LOCK:
        snapshot = dict(_USAGE_TOTALS)
    snapshot["total_tokens"] = snapshot["input_tokens"] + snapshot["output_tokens"]
    return snapshot


def diff_usage_snapshots(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> dict[str, int]:
    base_before = before or _zero_usage_snapshot()
    base_after = after or _zero_usage_snapshot()
    delta = {
        key: int(base_after.get(key, 0)) - int(base_before.get(key, 0))
        for key in ("total_calls", "llm_calls", "vlm_calls", "input_tokens", "output_tokens")
    }
    delta["total_tokens"] = delta["input_tokens"] + delta["output_tokens"]
    return delta


def _infer_modality_from_user_content(user_content: Any) -> str:
    if isinstance(user_content, list):
        for block in user_content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "").strip().lower()
            if block_type in {"image_url", "input_image", "video_url"}:
                return "vlm"
    return "llm"


def _record_usage_from_response(response: Any, *, user_content: Any) -> None:
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    modality = _infer_modality_from_user_content(user_content)
    with _USAGE_LOCK:
        _USAGE_TOTALS["total_calls"] += 1
        if modality == "vlm":
            _USAGE_TOTALS["vlm_calls"] += 1
        else:
            _USAGE_TOTALS["llm_calls"] += 1
        _USAGE_TOTALS["input_tokens"] += input_tokens
        _USAGE_TOTALS["output_tokens"] += output_tokens


def _prefers_max_completion_tokens(model: str) -> bool:
    """Return whether a model family is more likely to require
    ``max_completion_tokens`` instead of ``max_tokens``."""
    normalized = model.strip().lower()
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))


def _supports_custom_temperature(model: str) -> bool:
    """Return whether the chat endpoint accepts an explicit temperature."""
    return model.strip().lower() != "gpt-5.6-sol"


def _build_request_kwargs(
    *,
    model: str,
    system_prompt: str,
    user_content,
    temperature: float,
    max_tokens: int,
    use_max_completion_tokens: bool,
) -> dict:
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if _supports_custom_temperature(model):
        request_kwargs["temperature"] = temperature
    if use_max_completion_tokens:
        request_kwargs["max_completion_tokens"] = max_tokens
    else:
        request_kwargs["max_tokens"] = max_tokens
    return _sanitize_for_json(request_kwargs)


def _sanitize_string(value: str) -> str:
    sanitized_chars = []
    for ch in value:
        codepoint = ord(ch)
        if codepoint in (9, 10, 13):
            sanitized_chars.append(ch)
            continue
        if codepoint < 32 or 0xD800 <= codepoint <= 0xDFFF:
            sanitized_chars.append(" ")
            continue
        sanitized_chars.append(ch)
    return "".join(sanitized_chars)


def _sanitize_for_json(value):
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, dict):
        return {_sanitize_string(str(key)): _sanitize_for_json(item) for key, item in value.items()}
    return value


def _preview_text(value: str, *, max_chars: int = 600) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3] + "..."


# ---------------------------------------------------------------------------
#  Client constructor
# ---------------------------------------------------------------------------


def make_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """Create an OpenAI-compatible API client."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("The 'openai' package is required.  Install it with:\n    pip install openai")
    kwargs: dict = {}
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url is not None:
        kwargs["base_url"] = base_url
    timeout_seconds = float(
        os.getenv(
            "PO_PDDL_LLM_TIMEOUT_SECONDS",
            os.getenv("POMDPDDL_LLM_TIMEOUT_SECONDS", "180"),
        )
    )
    if timeout_seconds <= 0:
        raise ValueError("PO_PDDL_LLM_TIMEOUT_SECONDS must be positive")
    kwargs["timeout"] = timeout_seconds
    # safe_chat owns retry policy and logging; nested SDK retries make stalls opaque.
    kwargs["max_retries"] = 0
    return OpenAI(**kwargs)


# ---------------------------------------------------------------------------
#  Image / content helpers
# ---------------------------------------------------------------------------


def _resolve_image_path(image_path: str) -> Path:
    """Resolve an image path, allowing common extension fallback."""
    p = Path(image_path)
    if p.exists():
        return p

    suffix = p.suffix.lower()
    stem = p.with_suffix("")
    fallback_suffixes = [".png", ".jpg", ".jpeg"]
    if suffix in fallback_suffixes:
        fallback_suffixes = [s for s in fallback_suffixes if s != suffix]

    for alt_suffix in fallback_suffixes:
        alt = Path(str(stem) + alt_suffix)
        if alt.exists():
            return alt

    raise FileNotFoundError(f"Image file not found: {image_path}")


def encode_image_to_data_url(image_path: str) -> str:
    """Read an image file and return a ``data:<mime>;base64,…`` URL
    suitable for the OpenAI vision API."""
    p = _resolve_image_path(image_path)
    mime, _ = mimetypes.guess_type(str(p))
    if mime is None:
        suffix_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime = suffix_map.get(p.suffix.lower(), "image/jpeg")
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build_text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def build_image_url_block(url: str, *, detail: str = "high") -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url, "detail": detail}}


def build_local_image_block(image_path: str, *, detail: str = "high") -> dict[str, Any]:
    data_url = encode_image_to_data_url(image_path)
    return build_image_url_block(data_url, detail=detail)


def build_user_content(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    *,
    image_paths: Optional[list[str]] = None,
    image_url: Optional[str] = None,
    image_urls: Optional[list[str]] = None,
    detail: str = "high",
    extra_blocks: Optional[list[dict[str, Any]]] = None,
):
    """Build a user-message payload (text-only or multimodal).

    Backward-compatible behavior:
      * text only -> return a plain string
      * text + one image_path -> return [image, text]

    Extended behavior:
      * image_paths / image_urls preserve the order they are provided in
      * extra_blocks are appended after the convenience blocks
    """
    if image_path is not None and image_paths:
        raise ValueError("Pass either image_path or image_paths, not both.")
    if image_url is not None and image_urls:
        raise ValueError("Pass either image_url or image_urls, not both.")
    if image_path is not None and image_url is not None:
        raise ValueError("Pass either image_path or image_url, not both.")

    ordered_blocks: list[dict[str, Any]] = []

    for url in image_urls or []:
        ordered_blocks.append(build_image_url_block(url, detail=detail))
    for path in image_paths or []:
        ordered_blocks.append(build_local_image_block(path, detail=detail))
    if image_url is not None:
        ordered_blocks.append(build_image_url_block(image_url, detail=detail))
    if image_path is not None:
        ordered_blocks.append(build_local_image_block(image_path, detail=detail))
    if text is not None:
        ordered_blocks.append(build_text_block(text))
    if extra_blocks:
        ordered_blocks.extend(extra_blocks)

    if not ordered_blocks:
        raise ValueError("User content must include at least text, an image, or extra_blocks.")

    if len(ordered_blocks) == 1 and ordered_blocks[0]["type"] == "text":
        return ordered_blocks[0]["text"]
    return ordered_blocks


def sample_image_paths(
    image_paths: list[str],
    *,
    max_images: int | None = None,
) -> list[str]:
    """Select an evenly spaced temporal sample while retaining both endpoints."""
    valid_paths = [str(path) for path in image_paths if str(path).strip()]
    if max_images is None:
        max_images = int(
            os.getenv(
                "PO_PDDL_VLM_MAX_IMAGES",
                os.getenv("POMDPDDL_VLM_MAX_IMAGES", "8"),
            )
        )
    if max_images < 1:
        raise ValueError("PO_PDDL_VLM_MAX_IMAGES must be at least 1")
    if len(valid_paths) <= max_images:
        return valid_paths
    if max_images == 1:
        return [valid_paths[-1]]
    indices = [round(index * (len(valid_paths) - 1) / (max_images - 1)) for index in range(max_images)]
    return [valid_paths[index] for index in indices]


# ---------------------------------------------------------------------------
#  Safe chat completion with retry
# ---------------------------------------------------------------------------


def safe_chat(
    client,
    system_prompt: str,
    user_content,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 16384,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
    verbose: bool = False,
) -> str:
    """Send a chat completion and return the reply text.

    Automatically retries on transient / rate-limit errors with
    exponential back-off + jitter.

    Parameters
    ----------
    client : openai.OpenAI
        An OpenAI client instance (created by :func:`make_client`).
    system_prompt : str
        System-level instruction.
    user_content : str | list
        User-message payload (plain text or multimodal list).
    model, temperature, max_tokens : forwarded to the API.
    max_retries : int
        Maximum number of retry attempts (default 5).
    initial_backoff : float
        Initial sleep duration in seconds; doubles each retry.
    verbose : bool
        If ``True``, log a short reply preview without dumping the request payload.

    Returns
    -------
    str
        The assistant's reply text.

    Raises
    ------
    RuntimeError
        If the API returns an empty response after all retries.
    openai exceptions
        Re-raised after exhausting retries.
    """
    # Lazy-import so users who don't have openai installed see a clear
    # error only when they actually try to call this function.
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
        )
    except ImportError:
        # The filesystem-backed agent client does not require the OpenAI SDK.
        # Keep distinct placeholders so its AgentTaskPending control-flow
        # exception is not mistaken for every API error category.
        class APIConnectionError(Exception):
            pass

        class APIStatusError(Exception):
            pass

        class APITimeoutError(Exception):
            pass

        class RateLimitError(Exception):
            pass

    last_exc: Optional[Exception] = None
    backoff = initial_backoff
    configured_max_retries = os.getenv(
        "PO_PDDL_LLM_MAX_RETRIES",
        os.getenv("POMDPDDL_LLM_MAX_RETRIES"),
    )
    if configured_max_retries is not None:
        max_retries = int(configured_max_retries)
    if max_retries < 1:
        raise ValueError("PO_PDDL_LLM_MAX_RETRIES must be at least 1")

    for attempt in range(1, max_retries + 1):
        try:
            use_max_completion_tokens = _prefers_max_completion_tokens(model)
            request_kwargs = _build_request_kwargs(
                model=model,
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=temperature,
                max_tokens=max_tokens,
                use_max_completion_tokens=use_max_completion_tokens,
            )
            try:
                response = client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                message = str(exc)
                unsupported_max_tokens = (
                    "Unsupported parameter: 'max_tokens'" in message
                    or "\"param': 'max_tokens'" in message
                    or '"param": "max_tokens"' in message
                )
                unsupported_max_completion_tokens = (
                    "Unsupported parameter: 'max_completion_tokens'" in message
                    or "\"param': 'max_completion_tokens'" in message
                    or '"param": "max_completion_tokens"' in message
                )
                if use_max_completion_tokens and not unsupported_max_completion_tokens:
                    raise
                if not use_max_completion_tokens and not unsupported_max_tokens:
                    raise

                logger.debug(
                    "Switching token limit parameter for model %s after API rejection.",
                    model,
                )
                request_kwargs = _build_request_kwargs(
                    model=model,
                    system_prompt=system_prompt,
                    user_content=user_content,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_max_completion_tokens=not use_max_completion_tokens,
                )
                response = client.chat.completions.create(**request_kwargs)

            _record_usage_from_response(response, user_content=user_content)

            reply = response.choices[0].message.content

            if reply is None or (isinstance(reply, str) and not reply.strip()):
                # Empty / whitespace-only response — treat as retryable.
                logger.warning(
                    "LLM returned an empty or blank response (attempt %d/%d).",
                    attempt,
                    max_retries,
                )
                last_exc = RuntimeError("LLM returned an empty or blank response.")
                if attempt < max_retries:
                    _sleep_with_jitter(backoff)
                    backoff *= 2
                    continue
                raise last_exc

            if verbose:
                logger.info("LLM reply preview: %s", _preview_text(str(reply)))

            return reply

        except RateLimitError as exc:
            last_exc = exc
            wait = _extract_retry_after(exc) or backoff
            logger.warning(
                "Rate-limited (429) on attempt %d/%d.  Waiting %.1f s before retry …",
                attempt,
                max_retries,
                wait,
            )
            if attempt < max_retries:
                _sleep_with_jitter(wait)
                backoff = max(backoff, wait) * 2
                continue
            raise

        except (APIConnectionError, APITimeoutError) as exc:
            last_exc = exc
            logger.warning(
                "Connection/timeout error on attempt %d/%d: %s.  Waiting %.1f s before retry …",
                attempt,
                max_retries,
                exc,
                backoff,
            )
            if attempt < max_retries:
                _sleep_with_jitter(backoff)
                backoff *= 2
                continue
            raise

        except APIStatusError as exc:
            last_exc = exc
            retryable = False

            # 5xx — genuine server errors, always retry
            if exc.status_code and exc.status_code >= 500:
                retryable = True

            # Some providers wrap server-side failures in a 400 status
            # code (e.g. "The model was unable to complete inference due
            # to an internal error").  Detect these by inspecting the
            # error body and retry them as well.
            if exc.status_code == 400 and _is_server_side_400(exc):
                retryable = True

            if retryable:
                logger.warning(
                    "Server error (HTTP %s) on attempt %d/%d.  Waiting %.1f s before retry …",
                    exc.status_code,
                    attempt,
                    max_retries,
                    backoff,
                )
                if attempt < max_retries:
                    _sleep_with_jitter(backoff)
                    backoff *= 2
                    continue
            raise

    # Should not be reached, but just in case
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM call failed after all retries.")


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

# Phrases in a 400 error body that indicate a *server-side* failure
# rather than a genuine client mistake (e.g. bad prompt format).
_SERVER_SIDE_400_PHRASES = (
    "internal error",
    "unable to complete inference",
    "temporarily unavailable",
    "overloaded",
    "capacity",
)


def _is_server_side_400(exc: Exception) -> bool:
    """Return True if a 400 ``APIStatusError`` looks like a server-side
    failure disguised as a client error."""
    try:
        msg = str(exc).lower()
        return any(phrase in msg for phrase in _SERVER_SIDE_400_PHRASES)
    except Exception:
        return False


def _sleep_with_jitter(seconds: float) -> None:
    """Sleep for *seconds* ± 25 % jitter."""
    jitter = seconds * 0.25 * (2 * random.random() - 1)
    time.sleep(max(0, seconds + jitter))


def _extract_retry_after(exc: Exception) -> Optional[float]:
    """Try to pull a ``Retry-After`` value (seconds) from the exception."""
    # The openai SDK stores response headers when available.
    try:
        headers = exc.response.headers  # type: ignore[union-attr]
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is not None:
            return float(retry_after)
    except Exception:
        pass
    return None
