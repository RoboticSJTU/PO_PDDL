"""Typed configuration objects for the public generation APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DEFAULT_MODEL = "gpt-5.6-sol"


def _positive(value: int | float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Provider settings shared by language and vision model calls."""

    config_path: Path | None = None
    config_name: str = "openai_config"
    model: str | None = DEFAULT_MODEL
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int = 5000

    def __post_init__(self) -> None:
        _positive(self.max_tokens, "max_tokens")
        if not self.config_name.strip():
            raise ValueError("config_name must not be empty")

    def runner_kwargs(self) -> dict[str, object]:
        """Return arguments accepted by the domain pipeline runners."""
        return {
            "config": str(self.config_path) if self.config_path else None,
            "config_name": self.config_name,
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True, slots=True)
class DomainGenerationConfig:
    input_dir: Path
    output_dir: Path
    llm: LLMSettings = field(default_factory=LLMSettings)
    max_workers: int = 1
    max_iterations: int = 3
    smoothing: float = 0.0
    annotation_fps: float = 2.0
    annotation_video_types: tuple[str, ...] = ()
    run_stages: tuple[str, ...] | None = None
    keep_all_intersection_preconditions: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        _positive(self.max_workers, "max_workers")
        _positive(self.max_iterations, "max_iterations")
        _positive(self.annotation_fps, "annotation_fps")
        if self.smoothing < 0:
            raise ValueError("smoothing must be non-negative")


@dataclass(frozen=True, slots=True)
class DomainExtensionConfig:
    bundle_dir: Path
    input_dir: Path
    output_dir: Path
    llm: LLMSettings = field(default_factory=LLMSettings)
    max_workers: int = 1
    max_iterations: int = 3
    smoothing: float = 0.0
    annotation_fps: float = 2.0
    verbose: bool = False

    def __post_init__(self) -> None:
        _positive(self.max_workers, "max_workers")
        _positive(self.max_iterations, "max_iterations")
        _positive(self.annotation_fps, "annotation_fps")
        if self.smoothing < 0:
            raise ValueError("smoothing must be non-negative")


@dataclass(frozen=True, slots=True)
class ProblemGenerationConfig:
    domain_file: Path
    image_path: Path
    instruction: str
    initial_state_hint: str | None = None
    output_file: Path | None = None
    final_bundle_dir: Path | None = None
    reuse_problem_file: Path | None = None
    problem_name: str | None = None
    llm: LLMSettings = field(default_factory=lambda: LLMSettings(max_tokens=4096))
    max_workers: int = 8
    inference_strategy: Literal["batch", "parallel"] = "batch"
    prior_data_confidence: float = 0.0
    close_domain: bool = False
    skip_init_observation: bool = False
    verbose: bool = True

    def __post_init__(self) -> None:
        _positive(self.max_workers, "max_workers")
        if self.inference_strategy not in {"batch", "parallel"}:
            raise ValueError("inference_strategy must be 'batch' or 'parallel'")
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        if not 0.0 <= self.prior_data_confidence <= 1.0:
            raise ValueError("prior_data_confidence must be between 0 and 1")

    @property
    def resolved_output_file(self) -> Path:
        return self.output_file or self.domain_file.with_name("problem_online.pddl")
