from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CONFIG_NAME = "openai_config"
DEFAULT_CONFIG_FILENAME = "large_model_config.private.json"


def _discover_default_config_path() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / DEFAULT_CONFIG_FILENAME
        if candidate.exists():
            return candidate
    return current.parents[4] / DEFAULT_CONFIG_FILENAME


DEFAULT_CONFIG_PATH = _discover_default_config_path()


def _clean_config_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.startswith("YOUR_"):
        return None
    return stripped


def _select_profile(data: Dict[str, Any], config_name: str) -> Dict[str, Any]:
    if any(key in data for key in ("provider", "api_key", "base_url", "model", "temperature")):
        return data
    selected = data.get(config_name)
    if isinstance(selected, dict):
        return selected
    return {}


def load_llm_config(config_path: Optional[str] = None, config_name: Optional[str] = None) -> Dict[str, Any]:
    """Load local LLM config from JSON, falling back to environment variables.

    Supported file formats:
    1. Legacy flat format:
       {"api_key": "...", "base_url": "...", ...}
    2. Named-profile format:
       {"openai_config": {"api_key": "...", "base_url": "...", ...}, ...}
    """
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    selected_config_name = config_name or DEFAULT_CONFIG_NAME

    data: Dict[str, Any] = {}
    path_exists = path.exists()
    if path_exists:
        data = json.loads(path.read_text(encoding="utf-8"))
    selected = _select_profile(data, selected_config_name)

    provider = _clean_config_value(selected.get("provider")) or "openai"
    selected_base_url = _clean_config_value(selected.get("base_url"))
    if provider == "codex_cli":
        base_url = selected_base_url or "codex-cli://local"
    else:
        base_url = selected_base_url or _clean_config_value(os.getenv("OPENAI_BASE_URL"))

    return {
        "config_path": str(path),
        "config_exists": path_exists,
        "config_name": selected_config_name,
        "provider": provider,
        "api_key": _clean_config_value(selected.get("api_key")) or _clean_config_value(os.getenv("OPENAI_API_KEY")),
        "base_url": base_url,
        "model": _clean_config_value(selected.get("model")),
        "temperature": selected.get("temperature"),
    }
