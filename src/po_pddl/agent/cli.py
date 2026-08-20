"""Command-line interface for Codex/agent-driven PO-PDDL workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from po_pddl.config import DEFAULT_MODEL
from po_pddl.domain_generation.pipeline.runner import PIPELINE_STAGE_ORDER

from .worker_pool import DEFAULT_POOL_SIZE, DEFAULT_TASKS_PER_WORKER
from .workflow import AgentWorkflow, domain_arguments, extension_arguments, problem_arguments


def _add_model_options(
    parser: argparse.ArgumentParser,
    *,
    default_max_tokens: int,
    default_workers: int = DEFAULT_POOL_SIZE,
) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=default_max_tokens)
    parser.add_argument("--max-workers", type=int, default=default_workers)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")


def _add_init_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--no-start", action="store_true", help="Initialize without advancing to the first task.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PO-PDDL generation through a resumable coding-agent workflow.")
    commands = parser.add_subparsers(dest="command", required=True)

    domain = commands.add_parser("init-domain", help="Initialize domain learning from demonstrations.")
    _add_init_options(domain)
    _add_model_options(domain, default_max_tokens=5000)
    domain.add_argument("--input-dir", type=Path, required=True)
    domain.add_argument("--output-dir", type=Path, required=True)
    domain.add_argument("--smoothing", type=float, default=0.0)
    domain.add_argument("--annotation-fps", type=float, default=2.0)
    domain.add_argument("--annotation-video-types", nargs="*", default=[])
    domain.add_argument("--run-stages", nargs="+", choices=PIPELINE_STAGE_ORDER, default=None)
    domain.add_argument("--keep-all-intersection-preconditions", action="store_true")

    extension = commands.add_parser("init-extension", help="Initialize incremental domain extension.")
    _add_init_options(extension)
    _add_model_options(extension, default_max_tokens=5000)
    extension.add_argument("--bundle-dir", type=Path, required=True)
    extension.add_argument("--input-dir", type=Path, required=True)
    extension.add_argument("--output-dir", type=Path, required=True)
    extension.add_argument("--smoothing", type=float, default=0.0)
    extension.add_argument("--annotation-fps", type=float, default=2.0)

    problem = commands.add_parser("init-problem", help="Initialize problem generation from a scene.")
    _add_init_options(problem)
    _add_model_options(problem, default_max_tokens=4096)
    problem.add_argument("domain_file", type=Path)
    problem.add_argument("image_path", type=Path)
    problem.add_argument("instruction")
    problem.add_argument("--output-file", type=Path, required=True)
    problem.add_argument("--initial-state-hint", default=None)
    problem.add_argument("--final-bundle-dir", type=Path, default=None)
    problem.add_argument("--objects-file", type=Path, default=None)
    problem.add_argument("--reuse-problem-file", type=Path, default=None)
    problem.add_argument("--problem-name", default=None)
    problem.add_argument("--inference-strategy", choices=("batch", "parallel"), default="batch")
    problem.add_argument("--inference-batch-size", type=int, default=20)
    problem.add_argument("--location-visibility-batch-size", type=int, default=30)
    problem.add_argument("--prior-data-confidence", type=float, default=0.0)
    problem.add_argument("--close-domain", action="store_true")
    problem.add_argument("--skip-init-observation", action="store_true")

    for name in ("next", "status"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)

    dispatch = commands.add_parser(
        "dispatch",
        help="Advance the workflow and assign pending tasks to persistent worker slots.",
    )
    dispatch.add_argument("--run-dir", type=Path, required=True)
    dispatch.add_argument("--workers", type=int, default=DEFAULT_POOL_SIZE)
    dispatch.add_argument("--tasks-per-worker", type=int, default=DEFAULT_TASKS_PER_WORKER)

    run_pool = commands.add_parser(
        "run-pool",
        help="Run a workflow to completion with persistent Codex app-server workers.",
    )
    run_pool.add_argument("--run-dir", type=Path, required=True)
    run_pool.add_argument("--workers", type=int, default=DEFAULT_POOL_SIZE)
    run_pool.add_argument("--tasks-per-worker", type=int, default=DEFAULT_TASKS_PER_WORKER)
    run_pool.add_argument("--codex-bin", default="codex")
    run_pool.add_argument("--task-timeout-seconds", type=float, default=300.0)

    submit = commands.add_parser("submit")
    submit.add_argument("--run-dir", type=Path, required=True)
    submit.add_argument("--task-id", default=None)
    response = submit.add_mutually_exclusive_group(required=True)
    response.add_argument("--response")
    response.add_argument("--response-file", type=Path)

    reopen = commands.add_parser("reopen")
    reopen.add_argument("--run-dir", type=Path, required=True)
    reopen.add_argument("--task-id", required=True)
    reopen.add_argument("--reason", required=True)
    return parser


def _common_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_workers": args.max_workers,
        "max_iterations": args.max_iterations,
        "verbose": args.verbose,
    }


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workflow = AgentWorkflow(args.run_dir)
    if args.command == "status":
        _print(workflow.status())
        return 0
    if args.command == "dispatch":
        _print(
            workflow.dispatch(
                worker_count=args.workers,
                tasks_per_worker=args.tasks_per_worker,
            )
        )
        return 0
    if args.command == "run-pool":
        from .pool_runner import run_persistent_pool

        _print(
            run_persistent_pool(
                workflow,
                worker_count=args.workers,
                tasks_per_worker=args.tasks_per_worker,
                codex_executable=args.codex_bin,
                timeout_seconds=args.task_timeout_seconds,
            )
        )
        return 0
    if args.command == "next":
        _print(workflow.advance())
        return 0
    if args.command == "submit":
        response = args.response
        if args.response_file is not None:
            response = args.response_file.read_text(encoding="utf-8")
        assert response is not None
        _print(workflow.submit(response, task_id=args.task_id))
        return 0
    if args.command == "reopen":
        _print(workflow.reopen(args.task_id, args.reason))
        return 0

    common = _common_arguments(args)
    if args.command == "init-domain":
        arguments = domain_arguments(
            **common,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            smoothing=args.smoothing,
            annotation_fps=args.annotation_fps,
            annotation_video_types=args.annotation_video_types,
            run_stages=args.run_stages,
            keep_all_intersection_preconditions=args.keep_all_intersection_preconditions,
        )
        workflow.initialize("domain", arguments, replace=args.replace)
    elif args.command == "init-extension":
        arguments = extension_arguments(
            **common,
            bundle_dir=args.bundle_dir,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            smoothing=args.smoothing,
            annotation_fps=args.annotation_fps,
        )
        workflow.initialize("extension", arguments, replace=args.replace)
    else:
        arguments = problem_arguments(
            **common,
            domain_file=args.domain_file,
            image_path=args.image_path,
            instruction=args.instruction,
            output_file=args.output_file,
            initial_state_hint=args.initial_state_hint,
            final_bundle_dir=args.final_bundle_dir,
            objects_file=args.objects_file,
            reuse_problem_file=args.reuse_problem_file,
            problem_name=args.problem_name,
            inference_strategy=args.inference_strategy,
            inference_batch_size=args.inference_batch_size,
            location_visibility_batch_size=args.location_visibility_batch_size,
            prior_data_confidence=args.prior_data_confidence,
            close_domain=args.close_domain,
            skip_init_observation=args.skip_init_observation,
        )
        workflow.initialize("problem", arguments, replace=args.replace)
    _print(workflow.status() if args.no_start else workflow.advance())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
