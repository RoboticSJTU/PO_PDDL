"""Stable service layer around the internal domain pipeline runners."""

from __future__ import annotations

from po_pddl.config import DomainExtensionConfig, DomainGenerationConfig

from .extension.models import BundleUpdateResult
from .extension.runner import BundleUpdateRunner
from .pipeline.runner import LearningPipelineResult, LearningPipelineRunner


def generate_domain(config: DomainGenerationConfig) -> LearningPipelineResult:
    runner = LearningPipelineRunner(
        **config.llm.runner_kwargs(),
        max_workers=config.max_workers,
        max_iterations=config.max_iterations,
        smoothing=config.smoothing,
        annotation_fps=config.annotation_fps,
        annotation_video_types=list(config.annotation_video_types),
        run_stages=list(config.run_stages) if config.run_stages else None,
        precondition_learning_keep_all_intersection_preconditions=(config.keep_all_intersection_preconditions),
        verbose=config.verbose,
    )
    return runner.run(input_dir=config.input_dir, output_dir=config.output_dir)


def extend_domain(config: DomainExtensionConfig) -> BundleUpdateResult:
    runner = BundleUpdateRunner(
        **config.llm.runner_kwargs(),
        max_workers=config.max_workers,
        max_iterations=config.max_iterations,
        smoothing=config.smoothing,
        annotation_fps=config.annotation_fps,
        verbose=config.verbose,
    )
    return runner.run(
        bundle_dir=config.bundle_dir,
        input_dir=config.input_dir,
        output_dir=config.output_dir,
    )
