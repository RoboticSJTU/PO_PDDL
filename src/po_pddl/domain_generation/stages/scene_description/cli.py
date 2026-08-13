from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from po_pddl.domain_generation.stages.scene_description.init_scene_description import InitSceneDescriptionGenerator
from po_pddl.domain_generation.stages.scene_description.runner import SceneDescriptionRunner
from po_pddl.domain_generation.stages.scene_description.step_scene_description import StepSceneDescriptionGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe one episode end-to-end by extracting frames and generating scene descriptions for step 0 and all subsequent steps.",
    )
    parser.add_argument("--episode-file", type=Path, required=True, help="Path to one episode.json file.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory where scene-description outputs will be written."
    )
    parser.add_argument(
        "--fps", type=float, default=2.0, help="Frame sampling rate for the full episode step extraction."
    )
    parser.add_argument("--config", default=None, help="Optional path to the local OpenAI JSON config file.")
    parser.add_argument("--config-name", default="openai_config", help="Config profile name inside the JSON config.")
    parser.add_argument("--model", default=None, help="Override model from config.")
    parser.add_argument("--api-key", default=None, help="Override API key.")
    parser.add_argument("--base-url", default=None, help="Override base URL.")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=1600, help="Max completion tokens for the generators.")
    parser.add_argument(
        "--frame-selection-mode",
        default="first_and_last",
        choices=["first_and_last", "all_sampled"],
        help="How many sampled frames to send for each non-initial step scene description.",
    )
    parser.add_argument(
        "--image-detail",
        default="high",
        choices=["low", "high", "auto"],
        help="Image detail level for the VLM request.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print concise debug logs.")
    return parser.parse_args()


def configure_logging(*, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for logger_name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def main() -> int:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Running full scene-description pipeline")
    logger.info("Episode file: %s", args.episode_file)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Sampling fps: %s", args.fps)

    init_generator = InitSceneDescriptionGenerator(
        config_path=args.config,
        config_name=args.config_name,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        image_detail=args.image_detail,
        verbose=args.verbose,
    )
    step_generator = StepSceneDescriptionGenerator(
        config_path=args.config,
        config_name=args.config_name,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        frame_selection_mode=args.frame_selection_mode,
        image_detail=args.image_detail,
        verbose=args.verbose,
    )
    runner = SceneDescriptionRunner(
        init_generator=init_generator,
        step_generator=step_generator,
    )
    result = runner.run(
        episode_file=args.episode_file,
        output_dir=args.output_dir,
        fps=args.fps,
    )

    print(f"Episode: {result.episode_name}")
    print(f"Generated steps: {result.generated_step_count}")
    print(f"Wrote: {result.annotated_episode_file}")
    print(f"Wrote: {result.frame_manifest_file}")
    print(f"Wrote: {Path(args.output_dir) / 'scene_description_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
