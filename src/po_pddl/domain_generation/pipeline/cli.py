from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from po_pddl.config import DomainGenerationConfig, LLMSettings

from ..service import generate_domain
from .runner import PIPELINE_STAGE_ORDER


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end learning pipeline including scene description, manipulation learning, grounding, passive/init/active observation learning, and merge.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing episode subdirectories with episode.json or annotations.json files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory where all pipeline artifacts will be written."
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of repair/review iterations for modules that support iterative refinement.",
    )
    parser.add_argument("--config", default=None, help="Optional path to the local OpenAI JSON config file.")
    parser.add_argument("--config-name", default="openai_config", help="Named profile in the model config file.")
    parser.add_argument("--model", default=None, help="Override model from config.")
    parser.add_argument("--api-key", default=None, help="Override API key.")
    parser.add_argument("--base-url", default=None, help="Override base URL.")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=5000, help="Max completion tokens for LLM modules.")
    parser.add_argument(
        "--max-workers", type=int, default=1, help="Maximum number of concurrent worker threads where supported."
    )
    parser.add_argument(
        "--precondition-learning-keep-all-intersection-preconditions",
        action="store_true",
        help=(
            "Skip the final LLM precondition-selection pass and keep every candidate predicate "
            "that appears in all grounded examples for an action."
        ),
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.0, help="Optional Laplace smoothing for observation distributions."
    )
    parser.add_argument(
        "--annotation-fps",
        type=float,
        default=2.0,
        help="Frame sampling rate used by the initial scene-description stage.",
    )
    parser.add_argument(
        "--annotation-video-types",
        nargs="*",
        default=None,
        help="Optional ordered list of camera types to use during scene description. Multi-camera frames are stacked top-to-bottom in this order.",
    )
    parser.add_argument(
        "--run-stages",
        nargs="+",
        choices=PIPELINE_STAGE_ORDER,
        default=None,
        help="Optional subset of pipeline stages to run. Unselected predecessor stages will be loaded from existing outputs under --output-dir.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose LLM client logs.")
    return parser.parse_args(argv)


def configure_logging(*, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    if not verbose:
        for logger_name in ("httpx", "httpcore", "openai", "urllib3"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def main() -> int:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    logger.info(
        "[domain] start model=%s input=%s output=%s workers=%d",
        args.model or "config/default",
        args.input_dir,
        args.output_dir,
        args.max_workers,
    )

    config = DomainGenerationConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        llm=LLMSettings(
            config_path=Path(args.config) if args.config else None,
            config_name=args.config_name,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        max_workers=args.max_workers,
        max_iterations=args.max_iterations,
        smoothing=args.smoothing,
        annotation_fps=args.annotation_fps,
        annotation_video_types=tuple(args.annotation_video_types or ()),
        run_stages=tuple(args.run_stages) if args.run_stages else None,
        keep_all_intersection_preconditions=(args.precondition_learning_keep_all_intersection_preconditions),
        verbose=args.verbose,
    )
    result = generate_domain(config)

    logger.info(
        "[domain] complete episodes=%d domain=%s bundle=%s",
        result.total_episode_count,
        result.merged_domain_file,
        result.final_bundle_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
