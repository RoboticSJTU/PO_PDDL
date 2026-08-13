from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .factory import build_manipulation_domain_learner_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manipulation-domain learning over full-chain web collector data.",
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Directory containing full-chain episode folders."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory where learned artifacts will be written."
    )
    parser.add_argument("--config", default=None, help="Optional path to the local OpenAI JSON config file.")
    parser.add_argument("--model", default=None, help="Override model from config.")
    parser.add_argument("--api-key", default=None, help="Override API key.")
    parser.add_argument("--base-url", default=None, help="Override base URL.")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Max completion tokens for LLM modules.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of concurrent worker threads for LLM taxonomy/effect extraction.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum repair/replay iterations for episode-grounded effect learning.",
    )
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

    logger.info("Starting manipulation-domain learning run")
    logger.info("Input directory: %s", args.input_dir)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Requested max_workers: %d", args.max_workers)
    logger.info("Requested max_iterations: %d", args.max_iterations)
    learner, module_modes = build_manipulation_domain_learner_from_args(args)
    logger.info(
        "Modules selected: action_taxonomy=%s, action_schema_consolidation=%s, manipulation_effect=%s",
        module_modes.action_taxonomy_module,
        module_modes.action_schema_consolidation_module,
        module_modes.manipulation_effect_module,
    )
    result = learner.learn_from_directory(args.input_dir)
    learner.write_outputs(result, args.output_dir)

    print(f"Action taxonomy records: {len(result.taxonomy_records)}")
    print(f"Action schemas: {len(result.action_schemas)}")
    print(f"Manipulation effect records: {len(result.manipulation_records)}")
    print(f"Wrote: {args.output_dir / 'action_taxonomy.jsonl'}")
    print(f"Wrote: {args.output_dir / 'action_schemas.json'}")
    print(f"Wrote: {args.output_dir / 'action_schemas.pddl'}")
    print(f"Wrote: {args.output_dir / 'manipulation_records.jsonl'}")
    print(f"Wrote: {args.output_dir / 'manipulation_effect_statistics.json'}")
    print(f"Wrote: {args.output_dir / 'manipulation_actions.pddl'}")
    print(f"Wrote: {args.output_dir / 'observation_action_learning_status.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
