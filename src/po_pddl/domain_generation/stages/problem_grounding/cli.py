from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .factory import build_learner_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ground a problem and trajectory from a learned domain plus one episode.",
    )
    parser.add_argument("--domain-file", type=Path, required=True, help="Path to the generated domain file.")
    parser.add_argument("--episode-file", type=Path, required=True, help="Path to one episode.json file.")
    parser.add_argument(
        "--domain-learning-dir",
        type=Path,
        required=True,
        help="Directory containing action_schemas.json, action_taxonomy.jsonl, and manipulation_records.jsonl.",
    )
    parser.add_argument(
        "--review-guidance-file",
        type=Path,
        default=None,
        help="Optional JSON file containing review guidance for problem grounding.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory where grounded artifacts will be written."
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

    logger.info("Starting problem grounding run")
    logger.info("Domain file: %s", args.domain_file)
    logger.info("Episode file: %s", args.episode_file)
    logger.info("Domain-learning dir: %s", args.domain_learning_dir)
    logger.info("Review guidance: %s", args.review_guidance_file or "<none>")
    logger.info("Output dir: %s", args.output_dir)

    learner, module_modes = build_learner_from_args(args)
    logger.info(
        "Modules selected: object_init=%s, goal=%s, assembly=%s, trajectory_grounding=%s, validator=%s",
        module_modes.object_init_module,
        module_modes.goal_inference_module,
        module_modes.assembly_module,
        module_modes.trajectory_grounding_module,
        module_modes.validator_module,
    )
    result = learner.learn_from_files(
        domain_file=args.domain_file,
        episode_file=args.episode_file,
        domain_learning_dir=args.domain_learning_dir,
        review_guidance=(
            json.loads(args.review_guidance_file.read_text(encoding="utf-8"))
            if args.review_guidance_file is not None
            else None
        ),
    )
    learner.write_outputs(result, args.output_dir)

    print(f"Objects: {len(result.problem_spec.objects)}")
    print(f"Init facts: {len(result.problem_spec.init_facts)}")
    print(f"Goal facts: {len(result.problem_spec.goal_facts)}")
    print(f"Grounded steps: {len(result.grounded_steps)}")
    print(f"Validation issues: {len(result.validation_issues)}")
    print(f"Goal satisfied: {result.goal_satisfied}")
    print(f"Wrote: {args.output_dir / 'problem_summary.json'}")
    print(f"Wrote: {args.output_dir / 'problem.pddl'}")
    print(f"Wrote: {args.output_dir / 'grounded_trajectory.jsonl'}")
    print(f"Wrote: {args.output_dir / 'validation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
