"""Command-line entry point for terminal PO-PDDL execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..planning import parse_pomdpddl_runtime_from_files
from .feedback import ConsoleFeedback, ScriptedFeedback
from .session import TerminalSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a PO-PDDL planner with terminal or scripted feedback.")
    parser.add_argument("domain_file", type=Path)
    parser.add_argument("problem_file", type=Path)
    parser.add_argument("default_policy_file", nargs="?", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runtime"))
    parser.add_argument("--feedback-script", type=Path)
    parser.add_argument("--planner-seed", type=int)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--enable-report-goal-action", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--belief-update-log", action="store_true")
    parser.add_argument("--force-recompile-despot", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Parse and compile the model without planning.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = parse_pomdpddl_runtime_from_files(
        args.domain_file,
        args.problem_file,
        args.default_policy_file,
        output_dir=args.output_dir,
        planner_seed=args.planner_seed,
        gamma=args.gamma,
        max_step=args.max_steps,
        skip_despot_build=args.validate_only,
        belief_update_debug=args.belief_update_log,
        force_recompile_despot_cpp=args.force_recompile_despot,
        enable_report_goal_action=args.enable_report_goal_action,
    )
    print(f"Validated domain={runtime.parsed_domain.domain_name} problem={runtime.parsed_problem.problem_name}")
    print(f"Grounded actions={len(runtime.pomdp_planner.pomdp_model.actions)}")
    if args.validate_only:
        return 0

    feedback = ConsoleFeedback()
    if args.feedback_script is not None:
        payload = json.loads(args.feedback_script.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Feedback script must contain a JSON array of step objects.")
        feedback = ScriptedFeedback(payload)
    schemas = {schema.action.name: schema for schema in runtime.parsed_domain.actions}
    result = TerminalSession(
        planner=runtime.pomdp_planner,
        action_schemas=schemas,
        feedback=feedback,
        max_steps=args.max_steps,
        gamma=args.gamma,
        output_dir=args.output_dir,
    ).run()
    print(f"termination={result.termination_reason}")
    print(f"steps={len(result.steps)} cumulative_reward={result.cumulative_reward:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
