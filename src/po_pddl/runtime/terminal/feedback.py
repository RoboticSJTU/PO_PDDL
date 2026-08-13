"""Human and scripted feedback providers for terminal execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..planning.data_structures import Action, EffectBucket, Observable
from ..planning.data_structures.effect_bucket import ParsedEffectBucketAnnotation
from ..planning.engine.observation_semantics import EMPTY_OBSERVABLE_NAME, normalize_semantic_observation_entry
from ..planning.engine.report_goal_action import REPORT_GOAL_ACTION_NAME, REPORT_GOAL_BUCKET_NAME


class FeedbackProvider(Protocol):
    def select_effect(
        self,
        action: Action,
        annotations: Sequence[ParsedEffectBucketAnnotation],
        *,
        report_goal_enabled: bool,
    ) -> EffectBucket | None: ...

    def select_observation(
        self,
        applicable: Sequence[Observable],
        all_observables: Sequence[Observable],
    ) -> tuple[dict[Observable, bool], bool]: ...


def _effect_options(
    action: Action,
    annotations: Sequence[ParsedEffectBucketAnnotation],
    *,
    report_goal_enabled: bool,
) -> list[EffectBucket]:
    if report_goal_enabled and action.name == REPORT_GOAL_ACTION_NAME:
        return [
            EffectBucket(REPORT_GOAL_BUCKET_NAME, True, 0, 0),
            EffectBucket(REPORT_GOAL_BUCKET_NAME, False, 1, 0),
        ]
    return [
        EffectBucket(item.bucket_name, item.success, index, item.variant_rank)
        for index, item in enumerate(annotations)
    ]


@dataclass
class ConsoleFeedback:
    """Read action outcomes and observations from a human operator."""

    read: Callable[[str], str] = input
    write: Callable[[str], None] = print

    def select_effect(
        self,
        action: Action,
        annotations: Sequence[ParsedEffectBucketAnnotation],
        *,
        report_goal_enabled: bool,
    ) -> EffectBucket | None:
        options = _effect_options(action, annotations, report_goal_enabled=report_goal_enabled)
        if not options:
            return None

        self.write("Effect outcomes:")
        for index, option in enumerate(options):
            outcome = "success" if option.success else "failure"
            variant = f", variant={option.variant_rank}" if option.variant_rank is not None else ""
            self.write(f"  {index}: {outcome}, bucket={option.bucket_name}{variant}")
        while True:
            raw = self.read("Select effect outcome: ").strip()
            try:
                return options[int(raw)]
            except (ValueError, IndexError):
                self.write(f"Enter an integer from 0 to {len(options) - 1}.")

    def select_observation(
        self,
        applicable: Sequence[Observable],
        all_observables: Sequence[Observable],
    ) -> tuple[dict[Observable, bool], bool]:
        empty = next((item for item in all_observables if item.name == EMPTY_OBSERVABLE_NAME), None)
        observables = sorted(
            (item for item in applicable if item.name != EMPTY_OBSERVABLE_NAME),
            key=lambda item: item.to_pddl_str(),
        )
        if not observables:
            return ({empty: True}, False) if empty is not None else ({}, True)

        literals = [(item, value) for item in observables for value in (True, False)]
        self.write("Observation literals:")
        for index, (observable, value) in enumerate(literals):
            text = observable.to_pddl_str()
            self.write(f"  {index}: {text if value else f'(not {text})'}")
        self.write("  blank: no informative observation")
        while True:
            raw = self.read("Select comma-separated observation ids: ").strip()
            if not raw:
                return ({empty: True}, False) if empty is not None else ({}, True)
            try:
                indices = [int(token.strip()) for token in raw.split(",")]
                selected = [literals[index] for index in indices]
            except (ValueError, IndexError):
                self.write("Enter valid comma-separated integer ids.")
                continue
            observation: dict[Observable, bool] = {}
            conflict = False
            for item, value in selected:
                if item in observation and observation[item] != value:
                    conflict = True
                    break
                observation[item] = value
            if conflict:
                self.write("An observable cannot be selected as both true and false.")
                continue
            return normalize_semantic_observation_entry(observation), False


class ScriptedFeedback:
    """Replay structured feedback, useful for deterministic regression tests."""

    def __init__(self, steps: Sequence[dict[str, Any]]) -> None:
        self._steps = iter(steps)
        self._current: dict[str, Any] | None = None

    def select_effect(
        self,
        action: Action,
        annotations: Sequence[ParsedEffectBucketAnnotation],
        *,
        report_goal_enabled: bool,
    ) -> EffectBucket | None:
        try:
            self._current = next(self._steps)
        except StopIteration as exc:
            raise RuntimeError(f"Feedback script ended before action {action.to_pddl_str()}.") from exc
        expected = self._current.get("action")
        if expected and expected not in {action.name, action.to_pddl_str()}:
            raise ValueError(f"Expected action {expected!r}, planner chose {action.to_pddl_str()!r}.")

        options = _effect_options(action, annotations, report_goal_enabled=report_goal_enabled)
        if not options:
            return None
        success = bool(self._current.get("success", True))
        rank = self._current.get("variant_rank")
        matches = [item for item in options if item.success is success and (rank is None or item.variant_rank == rank)]
        if not matches:
            raise ValueError(f"No effect branch matches success={success}, variant_rank={rank} for {action.name}.")
        return matches[0]

    def select_observation(
        self,
        applicable: Sequence[Observable],
        all_observables: Sequence[Observable],
    ) -> tuple[dict[Observable, bool], bool]:
        assert self._current is not None
        requested = self._current.get("observations", [])
        by_text = {item.to_pddl_str(): item for item in applicable}
        observation: dict[Observable, bool] = {}
        for raw in requested:
            text = str(raw).strip()
            negative = text.startswith("(not ") and text.endswith(")")
            key = text[5:-1].strip() if negative else text
            observable = by_text.get(key)
            if observable is None:
                raise ValueError(f"Scripted observation is not applicable: {text}")
            observation[observable] = not negative
        if observation:
            return normalize_semantic_observation_entry(observation), False
        empty = next((item for item in all_observables if item.name == EMPTY_OBSERVABLE_NAME), None)
        return ({empty: True}, False) if empty is not None else ({}, True)
