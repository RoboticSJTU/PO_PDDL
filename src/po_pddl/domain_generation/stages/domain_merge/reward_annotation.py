from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from po_pddl.domain_generation.infrastructure.artifact_io import load_json, load_json_object, load_jsonl, write_jsonl

from .merger import _find_top_level_forms


@dataclass(frozen=True)
class RewardAnnotationSummary:
    manipulation_action_rewards: dict[str, float]
    observation_action_rewards: dict[str, float]
    updated_manipulation_record_count: int
    updated_observation_record_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fmt_number(value: float) -> str:
    text = f"{value:.6f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def _step_duration_seconds(step_payload: dict[str, Any]) -> float | None:
    start = step_payload.get("start_time_sec")
    end = step_payload.get("end_time_sec")
    if start is None or end is None:
        return None
    try:
        duration = float(end) - float(start)
    except (TypeError, ValueError):
        return None
    return max(duration, 0.0)


def build_episode_duration_index(episode_files: list[Path]) -> dict[tuple[str, int], float]:
    durations: dict[tuple[str, int], float] = {}
    for episode_file in episode_files:
        payload = load_json_object(episode_file)
        episode_name = str(payload.get("episode_name", episode_file.parent.name))
        for step_payload in payload.get("steps", []):
            if not isinstance(step_payload, dict):
                continue
            try:
                step_index = int(step_payload.get("step_index"))
            except (TypeError, ValueError):
                continue
            duration = _step_duration_seconds(step_payload)
            if duration is None:
                continue
            durations[(episode_name, step_index)] = duration
    return durations


def annotate_manipulation_artifacts(
    *,
    domain_learning_dir: str | Path,
    duration_index: dict[tuple[str, int], float],
) -> dict[str, float]:
    root = Path(domain_learning_dir)
    records_path = root / "manipulation_records.jsonl"
    statistics_path = root / "manipulation_effect_statistics.json"
    if not records_path.exists() or not statistics_path.exists():
        return {}

    records = load_jsonl(records_path)
    durations_by_action: dict[str, list[float]] = {}
    durations_by_bucket: dict[tuple[str, str], list[float]] = {}
    for row in records:
        duration = duration_index.get((str(row.get("episode_name")), int(row.get("step_index", -1))))
        if duration is None:
            try:
                existing_duration = row.get("execution_time_sec")
                duration = float(existing_duration) if existing_duration is not None else None
            except (TypeError, ValueError):
                duration = None
        row["execution_time_sec"] = duration
        if duration is None:
            continue
        action_name = str(row.get("canonical_action_name") or "")
        effect_bucket = str(row.get("effect_bucket") or "")
        durations_by_action.setdefault(action_name, []).append(duration)
        durations_by_bucket.setdefault((action_name, effect_bucket), []).append(duration)
    write_jsonl(records_path, records)

    statistics_payload = load_json(statistics_path)
    if isinstance(statistics_payload, dict):
        for action_name, entries in statistics_payload.items():
            if not isinstance(entries, list):
                continue
            action_durations = durations_by_action.get(str(action_name), [])
            action_avg = sum(action_durations) / len(action_durations) if action_durations else None
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                bucket_name = str(entry.get("effect_bucket") or "")
                bucket_durations = durations_by_bucket.get((str(action_name), bucket_name), [])
                bucket_avg = sum(bucket_durations) / len(bucket_durations) if bucket_durations else None
                entry["avg_execution_time_sec"] = bucket_avg
                entry["reward"] = (-bucket_avg) if bucket_avg is not None else None
                entry["action_avg_execution_time_sec"] = action_avg
                entry["action_reward"] = (-action_avg) if action_avg is not None else None
        statistics_path.write_text(
            json.dumps(statistics_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    action_rewards = {
        action_name: -(sum(values) / len(values)) for action_name, values in durations_by_action.items() if values
    }

    taxonomy_path = root / "action_taxonomy.jsonl"
    if taxonomy_path.exists():
        observation_durations_by_action: dict[str, list[float]] = {}
        for row in load_jsonl(taxonomy_path):
            action_category = str(row.get("action_category") or "").strip()
            if action_category not in {"active_observation", "observation"}:
                continue
            action_name = str(row.get("canonical_action_name") or "").strip()
            if not action_name:
                continue
            try:
                step_index = int(row.get("step_index", -1))
            except (TypeError, ValueError):
                continue
            duration = duration_index.get((str(row.get("episode_name") or ""), step_index))
            if duration is None:
                continue
            observation_durations_by_action.setdefault(action_name, []).append(duration)
        for action_name, values in observation_durations_by_action.items():
            if values:
                action_rewards[action_name] = -(sum(values) / len(values))

    return action_rewards


def annotate_observation_artifacts(
    *,
    observation_learning_dir: str | Path | None,
    duration_index: dict[tuple[str, int], float],
) -> dict[str, float]:
    if observation_learning_dir is None:
        return {}
    root = Path(observation_learning_dir)
    records_path = root / "observation_evidence.jsonl"
    if not records_path.exists():
        return {}

    records = load_jsonl(records_path)
    step_duration_by_action: dict[str, dict[tuple[str, int], float]] = {}
    for row in records:
        duration = duration_index.get((str(row.get("episode_name")), int(row.get("step_index", -1))))
        row["execution_time_sec"] = duration
        action_name = str(row.get("canonical_action_name") or "")
        if duration is None or not action_name:
            continue
        unique_key = (str(row.get("episode_name")), int(row.get("step_index", -1)))
        step_duration_by_action.setdefault(action_name, {})[unique_key] = duration
    write_jsonl(records_path, records)

    action_avg_by_name = {
        action_name: (sum(step_durations.values()) / len(step_durations))
        for action_name, step_durations in step_duration_by_action.items()
        if step_durations
    }

    (root / "observation_action_rewards.json").write_text(
        json.dumps(
            {
                action_name: {
                    "avg_execution_time_sec": avg_duration,
                    "reward": -avg_duration,
                }
                for action_name, avg_duration in sorted(action_avg_by_name.items())
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return {action_name: -avg_duration for action_name, avg_duration in action_avg_by_name.items()}


def _ensure_requirements_has_fluents(domain_text: str) -> str:
    forms = _find_top_level_forms(domain_text)
    requirements = next((form for form in forms if form.keyword == "requirements"), None)
    if requirements is None or ":fluents" in requirements.text:
        return domain_text
    updated = requirements.text.rstrip()
    if updated.endswith(")"):
        updated = updated[:-1].rstrip() + "\n    :fluents\n  )"
    return domain_text[: requirements.start] + updated + domain_text[requirements.end :]


def _ensure_total_reward_function(domain_text: str) -> str:
    forms = _find_top_level_forms(domain_text)
    functions_form = next((form for form in forms if form.keyword == "functions"), None)
    function_entry = "    (total-reward)                    ; cumulative reward"
    if functions_form is not None:
        if "total-reward" in functions_form.text:
            return domain_text
        updated = functions_form.text.rstrip()
        if updated.endswith(")"):
            updated = updated[:-1].rstrip() + "\n" + function_entry + "\n  )"
        return domain_text[: functions_form.start] + updated + domain_text[functions_form.end :]

    insert_after = None
    for keyword in ("observables", "predicates", "constants", "types", "requirements"):
        candidate = next((form for form in forms if form.keyword == keyword), None)
        if candidate is not None:
            insert_after = candidate
    functions_block = "  (:functions\n" + function_entry + "\n  )\n\n"
    if insert_after is None:
        return domain_text.rstrip() + "\n\n" + functions_block
    return domain_text[: insert_after.end] + "\n\n" + functions_block + domain_text[insert_after.end :].lstrip("\n")


def _find_section_expr_span(form_text: str, section_name: str) -> tuple[int, int] | None:
    keyword_index = form_text.find(section_name)
    if keyword_index < 0:
        return None
    pos = keyword_index + len(section_name)
    while pos < len(form_text) and form_text[pos].isspace():
        pos += 1
    if pos >= len(form_text) or form_text[pos] != "(":
        return None
    depth = 0
    end = pos
    while end < len(form_text):
        if form_text[end] == "(":
            depth += 1
        elif form_text[end] == ")":
            depth -= 1
            if depth == 0:
                end += 1
                break
        end += 1
    return pos, end


def _inject_reward_into_effect(form_text: str, reward_value: float) -> str:
    effect_span = _find_section_expr_span(form_text, ":effect")
    if effect_span is None or "total-reward" in form_text:
        return form_text
    effect_start, effect_end = effect_span
    original_effect = form_text[effect_start:effect_end].strip()
    reward_effect = f"(decrease (total-reward) {_fmt_number(abs(reward_value))})"
    new_effect = f"(and\n        {reward_effect}\n        {original_effect}\n    )"
    return form_text[:effect_start] + new_effect + form_text[effect_end:]


def apply_action_rewards_to_domain_text(
    domain_text: str,
    action_rewards: dict[str, float],
) -> str:
    if not action_rewards:
        return domain_text
    updated = _ensure_total_reward_function(_ensure_requirements_has_fluents(domain_text))
    forms = _find_top_level_forms(updated)
    replacements: list[tuple[int, int, str]] = []
    for form in forms:
        if form.keyword not in {"action", "observation"}:
            continue
        name_tokens = form.text.split()
        if len(name_tokens) < 2:
            continue
        action_name = name_tokens[1]
        reward_value = action_rewards.get(action_name)
        if reward_value is None:
            continue
        replacements.append((form.start, form.end, _inject_reward_into_effect(form.text, reward_value)))
    for start, end, replacement in sorted(replacements, reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    return updated


def annotate_rewards_and_apply_to_domain(
    *,
    episode_files: list[Path],
    domain_learning_dir: str | Path,
    observation_learning_dir: str | Path | None,
    merged_domain_file: str | Path,
) -> RewardAnnotationSummary:
    duration_index = build_episode_duration_index(episode_files)
    manipulation_action_rewards = annotate_manipulation_artifacts(
        domain_learning_dir=domain_learning_dir,
        duration_index=duration_index,
    )
    observation_action_rewards = annotate_observation_artifacts(
        observation_learning_dir=observation_learning_dir,
        duration_index=duration_index,
    )
    all_action_rewards = dict(manipulation_action_rewards)
    all_action_rewards.update(observation_action_rewards)
    merged_domain_path = Path(merged_domain_file)
    merged_domain_path.write_text(
        apply_action_rewards_to_domain_text(
            merged_domain_path.read_text(encoding="utf-8"),
            all_action_rewards,
        ),
        encoding="utf-8",
    )

    manipulation_record_count = (
        len(load_jsonl(Path(domain_learning_dir) / "manipulation_records.jsonl"))
        if Path(domain_learning_dir, "manipulation_records.jsonl").exists()
        else 0
    )
    observation_record_count = (
        len(load_jsonl(Path(observation_learning_dir) / "observation_evidence.jsonl"))
        if observation_learning_dir is not None
        and Path(observation_learning_dir, "observation_evidence.jsonl").exists()
        else 0
    )
    return RewardAnnotationSummary(
        manipulation_action_rewards=manipulation_action_rewards,
        observation_action_rewards=observation_action_rewards,
        updated_manipulation_record_count=manipulation_record_count,
        updated_observation_record_count=observation_record_count,
    )
