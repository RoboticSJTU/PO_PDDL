from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .factory import build_learner_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer problem objects and init facts from initial observation and trajectory steps.",
    )
    parser.add_argument("--domain-file", type=Path, required=True, help="Path to the domain file.")
    parser.add_argument("--episode-file", type=Path, required=True, help="Path to one episode.json file.")
    parser.add_argument(
        "--domain-learning-dir",
        type=Path,
        default=None,
        help="Optional domain-learning artifact directory for grounded init completion.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory where inferred problem artifacts will be written."
    )
    parser.add_argument("--config", default=None, help="Optional path to the local OpenAI JSON config file.")
    parser.add_argument("--model", default=None, help="Override model from config.")
    parser.add_argument("--api-key", default=None, help="Override API key.")
    parser.add_argument("--base-url", default=None, help="Override base URL.")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=3000, help="Max completion tokens for LLM modules.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose LLM client logs.")
    return parser.parse_args()


def configure_logging(*, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )


def main() -> int:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Starting problem inference run")
    logger.info("Domain file: %s", args.domain_file)
    logger.info("Episode file: %s", args.episode_file)
    logger.info("Domain-learning dir: %s", args.domain_learning_dir or "<disabled>")
    logger.info("Output dir: %s", args.output_dir)

    learner, module_modes = build_learner_from_args(args)
    logger.info(
        "Modules selected: visible=%s latent=%s initial_state=%s",
        module_modes.visible_object_module,
        module_modes.latent_object_module,
        module_modes.initial_state_module,
    )
    logger.info("Init completion module: %s", module_modes.init_completion_module)
    result = learner.infer_from_files(
        domain_file=args.domain_file,
        episode_file=args.episode_file,
        domain_learning_dir=args.domain_learning_dir,
    )
    summary = learner.write_outputs(result, args.output_dir)

    print(f"Visible objects: {summary.visible_object_count}")
    print(f"Latent objects: {summary.latent_object_count}")
    print(f"Init facts: {summary.init_fact_count}")
    print(f"Wrote: {args.output_dir / 'problem_summary.json'}")
    print(f"Wrote: {args.output_dir / 'problem.pddl'}")
    print(f"Wrote: {args.output_dir / 'visible_object_extraction.json'}")
    print(f"Wrote: {args.output_dir / 'latent_object_discovery.json'}")
    print(f"Wrote: {args.output_dir / 'initial_state_inference.json'}")
    print(f"Wrote: {args.output_dir / 'init_completion.json'}")
    print(f"Wrote: {args.output_dir / 'problem_inference_diagnostics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
