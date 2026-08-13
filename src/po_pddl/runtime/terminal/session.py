"""Planner session orchestration independent of terminal input details."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .feedback import FeedbackProvider


@dataclass(frozen=True)
class StepRecord:
    step: int
    action: str
    effect_bucket: str | None
    success: bool | None
    observations: dict[str, bool]
    reward: float
    search_seconds: float
    belief_update_seconds: float
    discounted_total: float
    cumulative_reward: float


@dataclass(frozen=True)
class SessionResult:
    termination_reason: str
    steps: tuple[StepRecord, ...]
    discounted_total: float
    cumulative_reward: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "termination_reason": self.termination_reason,
            "steps": [asdict(step) for step in self.steps],
            "discounted_total": self.discounted_total,
            "cumulative_reward": self.cumulative_reward,
        }


@dataclass
class TerminalSession:
    planner: Any
    action_schemas: dict[str, Any]
    feedback: FeedbackProvider
    max_steps: int = 30
    gamma: float = 0.95
    output_dir: Path | None = None
    emit: Callable[[str], None] = print
    _records: list[StepRecord] = field(default_factory=list, init=False)

    def run(self) -> SessionResult:
        discounted_total = 0.0
        cumulative_reward = 0.0
        termination = "max_steps_reached"
        model = self.planner.pomdp_model
        report_enabled = bool(getattr(model, "enable_report_goal_action", False))
        is_report_goal = getattr(model, "is_report_goal_action", lambda _action: False)

        while self.planner.current_step < self.max_steps:
            started = time.perf_counter()
            action = self.planner.search()
            search_seconds = time.perf_counter() - started
            if isinstance(action, int) and action == len(model.actions):
                termination = "no_feasible_action"
                break
            self.emit(f"chosen_action={action.to_pddl_str()}")

            schema = self.action_schemas.get(action.name)
            annotations = schema.effect_bucket_annotations if schema is not None else []
            effect = self.feedback.select_effect(
                action,
                annotations,
                report_goal_enabled=report_enabled,
            )
            reward = float(self.planner.estimate_expected_reward(action, effect_bucket=effect))
            applicable = self.planner.applicable_observables(action, effect_bucket=effect)
            observation, skip_observation = self.feedback.select_observation(applicable, model.observables)
            report_failed = bool(report_enabled and is_report_goal(action) and effect is not None and effect.success is False)

            update_started = time.perf_counter()
            self.planner.belief_update(
                action,
                observation,
                effect_bucket=effect,
                simulator_reported_not_goal=report_failed,
                skip_observation_update=skip_observation,
            )
            update_seconds = time.perf_counter() - update_started
            step = int(self.planner.current_step)
            cumulative_reward += reward
            discounted_total += (self.gamma ** (step - 1)) * reward
            record = StepRecord(
                step=step,
                action=action.to_pddl_str(),
                effect_bucket=effect.bucket_name if effect is not None else None,
                success=effect.success if effect is not None else None,
                observations={item.to_pddl_str(): value for item, value in observation.items()},
                reward=reward,
                search_seconds=search_seconds,
                belief_update_seconds=update_seconds,
                discounted_total=discounted_total,
                cumulative_reward=cumulative_reward,
            )
            self._records.append(record)
            self._write_step(record)
            self.emit(
                f"step={step} reward={reward:.6f} cumulative_reward={cumulative_reward:.6f} "
                f"search_seconds={search_seconds:.6f} belief_update_seconds={update_seconds:.6f}"
            )
            if report_enabled and is_report_goal(action) and effect is not None and effect.success:
                termination = "report_goal_success"
                break

        result = SessionResult(termination, tuple(self._records), discounted_total, cumulative_reward)
        self._write_summary(result)
        return result

    def _write_step(self, record: StepRecord) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "steps.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def _write_summary(self, result: SessionResult) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "session_summary.json").write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
