from __future__ import annotations

import argparse
from pathlib import Path

from po_pddl.config import LLMSettings, ProblemGenerationConfig

from .service import generate_problem


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a POMDPDDL problem file for online planning from a real scene image.",
    )
    parser.add_argument("domain", help="Path to the learned POMDPDDL domain file.")
    parser.add_argument(
        "image",
        help="Path to the current scene image, or a directory containing multiple view images to stitch vertically.",
    )
    parser.add_argument("instruction", help="Natural-language task instruction.")
    parser.add_argument(
        "--initial-state-hint",
        default=None,
        help="Optional natural-language facts about the initial scene; used only for initial-belief inference.",
    )
    parser.add_argument(
        "--final-bundle-dir",
        default=None,
        help="Optional path to the offline learning final bundle directory.",
    )
    parser.add_argument(
        "--objects-file",
        default=None,
        help="Optional explicit objects.txt allowlist; overrides lookup next to the domain file.",
    )
    parser.add_argument(
        "--reuse-problem-file",
        default=None,
        help=(
            "Optional existing online problem file whose objects and goal should be reused. "
            "When set, object generation and goal generation are skipped and only init belief is regenerated."
        ),
    )
    parser.add_argument(
        "--close-domain",
        "--close_domain",
        dest="close_domain",
        action="store_true",
        help=(
            "Skip LLM object extraction and build a closed-domain object inventory by merging objects "
            "from `4_problem_grounding_all/*/problem.pddl`."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the rendered problem file.",
    )
    parser.add_argument(
        "--problem-name",
        default=None,
        help="Optional explicit problem name.",
    )
    parser.add_argument("--config", default=None, help="Optional path to a local model configuration JSON file.")
    parser.add_argument(
        "--config-name",
        default="openai_config",
        help="Named profile inside the model configuration file.",
    )
    parser.add_argument("--model", default=None, help="Override model from config.")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=None, help="Optional API key override.")
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum completion tokens.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum workers used by the parallel inference strategy.",
    )
    parser.add_argument(
        "--inference-strategy",
        choices=("batch", "parallel"),
        default="batch",
        help=(
            "Scheduling for batched predicate and goal-assignment judgments. "
            "The default 'batch' processes chunks sequentially; 'parallel' processes chunks concurrently."
        ),
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=20,
        help="Maximum deterministic predicates or goal assignments included in one model call.",
    )
    parser.add_argument(
        "--prior-data-confidence",
        type=float,
        default=0.0,
        help=(
            "Weight used to blend uncertain-group priors from historical grounding data with uniform priors. "
            "The default 0.0 uses code-generated uniform distributions and disables "
            "historical priors and observation-based probability reweighting."
        ),
    )
    parser.add_argument(
        "--skip-init-observation",
        action="store_true",
        help=(
            "Skip observation-aware initial belief calibration even when the domain has an observation module. "
            "When enabled, directly use deterministic init-belief inference without initial observation / belief update."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed pipeline diagnostics.")
    parser.add_argument("--quiet", action="store_true", help="Suppress all console output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    verbose = args.verbose and not args.quiet

    def _log(message: str) -> None:
        if verbose:
            print(f"[generate_online_problem] {message}", flush=True)

    if not args.quiet:
        print(
            f"[problem] start input={args.image} output={args.output or '<default>'}",
            flush=True,
        )
    config = ProblemGenerationConfig(
        domain_file=Path(args.domain),
        image_path=Path(args.image),
        instruction=args.instruction,
        initial_state_hint=args.initial_state_hint,
        output_file=Path(args.output) if args.output else None,
        final_bundle_dir=Path(args.final_bundle_dir) if args.final_bundle_dir else None,
        objects_file=Path(args.objects_file) if args.objects_file else None,
        reuse_problem_file=Path(args.reuse_problem_file) if args.reuse_problem_file else None,
        problem_name=args.problem_name,
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
        inference_strategy=args.inference_strategy,
        inference_batch_size=args.inference_batch_size,
        prior_data_confidence=args.prior_data_confidence,
        close_domain=args.close_domain,
        skip_init_observation=args.skip_init_observation,
        verbose=verbose,
    )
    result = generate_problem(config, logger=_log)
    if not args.quiet:
        print(
            f"[problem] complete name={result.spec.problem_name} "
            f"objects={len(result.visible_objects)} predicates={len(result.predicate_judgments)} "
            f"output={config.resolved_output_file}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
