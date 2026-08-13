from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from po_pddl.domain_generation.stages.scene_description.init_scene_description import InitSceneDescriptionGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the initial scene description from an episode's first video frame and future action sequence.",
    )
    parser.add_argument("--episode-file", type=Path, required=True, help="Path to one episode.json file.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory where generation outputs will be written."
    )
    parser.add_argument("--config", default=None, help="Optional path to the local OpenAI JSON config file.")
    parser.add_argument("--config-name", default="openai_config", help="Config profile name inside the JSON config.")
    parser.add_argument("--model", default=None, help="Override model from config.")
    parser.add_argument("--api-key", default=None, help="Override API key.")
    parser.add_argument("--base-url", default=None, help="Override base URL.")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Max completion tokens for the generator.")
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

    logger.info("Generating initial scene description")
    logger.info("Episode file: %s", args.episode_file)
    logger.info("Output dir: %s", args.output_dir)

    generator = InitSceneDescriptionGenerator(
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
    result = generator.generate_from_episode(args.episode_file, args.output_dir)

    print(f"Episode: {result.episode_name}")
    print(f"Scene description: {result.scene_description_text}")
    print(f"Wrote: {args.output_dir / 'step0_first_frame.jpg'}")
    print(f"Wrote: {args.output_dir / 'init_scene_description.txt'}")
    print(f"Wrote: {args.output_dir / 'init_scene_description_generation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
