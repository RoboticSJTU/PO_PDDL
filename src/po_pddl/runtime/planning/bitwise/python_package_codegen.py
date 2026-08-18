"""Emit explicit bitwise Python model packages from explicit semantic packages."""

from __future__ import annotations

import json
import re
import shutil
from importlib import import_module
from pathlib import Path
from typing import Iterable

from ..base.pomdp_model import POMDPModelBase as SemanticPOMDPModelBase
from .indexed_belief_json import load_indexed_factorized_belief_json
from .indexed_particle_belief import IndexedParticleBelief
from .indexed_particle_belief_json import dump_indexed_particle_belief_json
from .parser import BitwiseEffectBranch, BitwiseGoalCheck, BitwiseModelSkeleton, BitwiseObservationBranch
from .parser import (
    build_bitwise_model_skeleton_from_module,
    convert_indexed_factorized_belief_to_particles,
)


def emit_bitwise_python_model_package_from_module(
    module_name: str,
    output_dir: str | Path,
    *,
    builder_name: str = "build_model",
) -> Path:
    """Compile an explicit semantic package into a pure bitwise package."""
    module = import_module(module_name)
    factory = getattr(module, builder_name)
    if not callable(factory):
        raise TypeError(f"`{module_name}.{builder_name}` is not callable.")
    semantic_model = factory()
    if not isinstance(semantic_model, SemanticPOMDPModelBase):
        raise TypeError(
            f"`{module_name}.{builder_name}()` did not return a semantic POMDPModelBase instance."
        )

    shared_module = import_module(f"{module_name}.shared")
    problem_name = getattr(shared_module, "PROBLEM_NAME", None)
    bitwise_model = build_bitwise_model_skeleton_from_module(module_name, builder_name=builder_name)
    return emit_bitwise_python_model_package(
        bitwise_model,
        semantic_model,
        output_dir,
        class_name=_make_class_name(problem_name or Path(str(output_dir)).name),
    )


def emit_bitwise_python_model_package(
    bitwise_model: BitwiseModelSkeleton,
    semantic_model: SemanticPOMDPModelBase,
    output_dir: str | Path,
    *,
    class_name: str = "GeneratedBitwiseModel",
    init_belief: IndexedParticleBelief | None = None,
) -> Path:
    """Write one explicit bitwise model package with only ids, masks, and bit operations."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actions_dir = out_dir / "actions"
    observation_rules_dir = out_dir / "observation_rules"
    default_policy_rules_dir = out_dir / "default_policy_rules"
    for directory in (actions_dir, observation_rules_dir, default_policy_rules_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _write_shared_module(out_dir / "shared.py", bitwise_model)
    _write_index_layout_json(out_dir / "index_layout.json", semantic_model)
    _write_init_belief_json(out_dir / "init_belief.json", semantic_model, init_belief)
    _write_action_modules(actions_dir, semantic_model, bitwise_model)
    _write_observation_rule_modules(observation_rules_dir, semantic_model, bitwise_model)
    _write_default_policy_rule_modules(default_policy_rules_dir, semantic_model, bitwise_model)
    _write_model_module(out_dir / "model.py", class_name, semantic_model, bitwise_model)
    _write_package_init(out_dir / "__init__.py", class_name)
    for subdir in (actions_dir, observation_rules_dir, default_policy_rules_dir):
        (subdir / "__init__.py").write_text('"""Auto-generated bitwise helper modules."""\n', encoding="utf-8")
    return out_dir


def _write_shared_module(path: Path, model: BitwiseModelSkeleton) -> None:
    lines = [
        '"""Auto-generated shared constants for an explicit bitwise model package."""',
        "",
        "from __future__ import annotations",
        "",
        f"GROUNDED_PREDICATES_COUNT = {model.grounded_predicates_count}",
        f"GROUNDED_OBSERVABLES_COUNT = {model.grounded_observables_count}",
        f"TOTAL_ACTIONS = {model.total_actions}",
        f"MAXIMIZE_REWARD = {model.maximize_reward!r}",
        f"GOAL_REWARD = {model.goal_reward!r}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_init_belief_json(
    path: Path,
    semantic_model: SemanticPOMDPModelBase,
    init_belief: IndexedParticleBelief | None = None,
) -> None:
    if init_belief is not None:
        dump_indexed_particle_belief_json(init_belief, path)
        return
    module_name = semantic_model.__class__.__module__.rsplit(".", 1)[0]
    package = import_module(module_name)
    source_path = Path(package.__file__).with_name("init_belief.json")
    if not source_path.exists():
        raise FileNotFoundError(
            f"Expected explicit package init belief JSON at {source_path}, but it does not exist."
        )
    factorized_belief = load_indexed_factorized_belief_json(source_path)
    particle_belief = convert_indexed_factorized_belief_to_particles(factorized_belief)
    dump_indexed_particle_belief_json(particle_belief, path)


def _write_index_layout_json(path: Path, semantic_model: SemanticPOMDPModelBase) -> None:
    data = {
        "predicates": [_render_pomdpddl_atom(predicate.name, predicate.params) for predicate in semantic_model.predicates],
        "observables": [_render_pomdpddl_atom(observable.name, observable.params) for observable in semantic_model.observables],
        "actions": [_render_pomdpddl_atom(action.name, action.params) for action in semantic_model.actions],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _render_pomdpddl_atom(name: str, params: list[str]) -> str:
    if not params:
        return f"({name})"
    return "(" + " ".join([name, *params]) + ")"


def _write_model_module(
    path: Path,
    class_name: str,
    semantic_model: SemanticPOMDPModelBase,
    bitwise_model: BitwiseModelSkeleton,
) -> None:
    grouped_actions = _group_indices_by_name([action.name for action in semantic_model.actions])
    grouped_observation_rules = _group_indices_by_name(
        [rule.name for rule in semantic_model.observation_rules]
    )
    grouped_default_policy_rules = _group_indices_by_name(
        [rule.name for rule in semantic_model.default_policy_rules]
    )

    lines: list[str] = []
    append = lines.append
    append('"""Auto-generated explicit bitwise model package entrypoint."""')
    append("")
    append("from __future__ import annotations")
    append("")
    append("from pathlib import Path")
    append("")
    append("from po_pddl.runtime.planning.bitwise import POMDPModelBase")
    append("")
    append("from .shared import (")
    append("    GOAL_REWARD,")
    append("    GROUNDED_OBSERVABLES_COUNT,")
    append("    GROUNDED_PREDICATES_COUNT,")
    append("    MAXIMIZE_REWARD,")
    append("    TOTAL_ACTIONS,")
    append(")")
    if grouped_actions:
        append(
            "from .actions import "
            + ", ".join(
                f"{_safe_module_name(name)} as action_group_{_safe_module_name(name)}"
                for name in grouped_actions
            )
        )
    if grouped_observation_rules:
        append(
            "from .observation_rules import "
            + ", ".join(
                f"{_safe_module_name(name)} as observation_rule_group_{_safe_module_name(name)}"
                for name in grouped_observation_rules
            )
        )
    if grouped_default_policy_rules:
        append(
            "from .default_policy_rules import "
            + ", ".join(
                f"{_safe_module_name(name)} as default_policy_rule_group_{_safe_module_name(name)}"
                for name in grouped_default_policy_rules
            )
        )
    append("")
    append("CHECK_ACTION_PRECONDITIONS = [")
    for idx, action in enumerate(semantic_model.actions):
        mod = f"action_group_{_safe_module_name(action.name)}"
        append(f"    {mod}._check_action_precondition_{idx},")
    append("]")
    append("FORWARD_ACTIONS = [")
    for idx, action in enumerate(semantic_model.actions):
        mod = f"action_group_{_safe_module_name(action.name)}"
        append(f"    {mod}._forward_action_{idx},")
    append("]")
    append("GET_ACTION_REWARDS = [")
    for idx, action in enumerate(semantic_model.actions):
        mod = f"action_group_{_safe_module_name(action.name)}"
        append(f"    {mod}._get_action_reward_{idx},")
    append("]")
    append("")
    append("CHECK_OBSERVATION_RULE_CONDITIONS = [")
    for idx, rule in enumerate(semantic_model.observation_rules):
        mod = f"observation_rule_group_{_safe_module_name(rule.name)}"
        append(f"    {mod}._check_observation_rule_condition_{idx},")
    append("]")
    append("OBSERVE_WITH_RULES = [")
    for idx, rule in enumerate(semantic_model.observation_rules):
        mod = f"observation_rule_group_{_safe_module_name(rule.name)}"
        append(f"    {mod}._observe_with_rule_{idx},")
    append("]")
    append("")
    append(
        f"OBS_RULE_SKIP_BEFORE_ON_ACTIVE_PERCEPTION = {repr([rule.name.startswith('before_') for rule in semantic_model.observation_rules])}"
    )
    append(
        f"ACTION_IS_ACTIVE_PERCEPTION = {repr([action.name.startswith('active_obs_') for action in semantic_model.actions])}"
    )
    append("")
    append("CHECK_DEFAULT_POLICY_RULE_CONDITIONS = [")
    for idx, rule in enumerate(semantic_model.default_policy_rules):
        mod = f"default_policy_rule_group_{_safe_module_name(rule.name)}"
        append(f"    {mod}._check_default_policy_rule_condition_{idx},")
    append("]")
    append("GET_DEFAULT_POLICY_RULE_ACTIONS = [")
    for idx, rule in enumerate(semantic_model.default_policy_rules):
        mod = f"default_policy_rule_group_{_safe_module_name(rule.name)}"
        append(f"    {mod}._get_default_policy_rule_action_{idx},")
    append("]")
    append("")
    append(f"class {class_name}(POMDPModelBase):")
    append('    """Concrete explicit bitwise model."""')
    append("")
    append("    def __init__(self) -> None:")
    append("        super().__init__(")
    append("            grounded_predicates_count=GROUNDED_PREDICATES_COUNT,")
    append("            grounded_observables_count=GROUNDED_OBSERVABLES_COUNT,")
    append("            total_actions=TOTAL_ACTIONS,")
    append("            maximize_reward=MAXIMIZE_REWARD,")
    append("            goal_reward=GOAL_REWARD,")
    append("        )")
    append("        self.observation_rule_skip_before_for_active_perception = list(OBS_RULE_SKIP_BEFORE_ON_ACTIVE_PERCEPTION)")
    append("        self.action_is_active_perception = list(ACTION_IS_ACTIVE_PERCEPTION)")
    append("")
    append("    def _sample_branch_index(self, weights: list[float]) -> int | None:")
    append("        cleaned = [max(weight, 0.0) for weight in weights]")
    append("        total = sum(cleaned)")
    append("        if total <= 0.0:")
    append("            return None")
    append("        draw = self._rng.random() * total")
    append("        cumulative = 0.0")
    append("        for idx, weight in enumerate(cleaned):")
    append("            cumulative += weight")
    append("            if draw <= cumulative:")
    append("                return idx")
    append("        return len(cleaned) - 1")
    append("")
    append("    def _cache_transition_reward(")
    append("        self, action: int, state: int, next_state: int, reward: float")
    append("    ) -> None:")
    append("        self._last_transition_reward_cache[(action, state, next_state)] = reward")
    append("")
    append("    def _lookup_cached_reward(")
    append("        self, action: int, state: int, next_state: int")
    append("    ) -> float | None:")
    append("        return self._last_transition_reward_cache.get((action, state, next_state))")
    append("")
    append("    def is_goal(self, state: int) -> bool:")
    append(f"        return {_render_goal_check_expr(bitwise_model.goal_check, 'state')}")
    append("")
    append("    def check_action_precondition(self, action: int, state: int) -> bool:")
    append("        if action < 0 or action >= len(CHECK_ACTION_PRECONDITIONS):")
    append("            raise ValueError(f\"Unknown grounded action id: {action}\")")
    append("        return CHECK_ACTION_PRECONDITIONS[action](self, state)")
    append("")
    append("    def forward_action(self, action: int, state: int) -> tuple[int, float]:")
    append("        if action < 0 or action >= len(FORWARD_ACTIONS):")
    append("            raise ValueError(f\"Unknown grounded action id: {action}\")")
    append("        return FORWARD_ACTIONS[action](self, state)")
    append("")
    append("    def get_action_reward(self, action: int, state: int, next_state: int) -> float:")
    append("        if action < 0 or action >= len(GET_ACTION_REWARDS):")
    append("            raise ValueError(f\"Unknown grounded action id: {action}\")")
    append("        return GET_ACTION_REWARDS[action](self, state, next_state)")
    append("")
    append("    def is_active_perception_action(self, action: int | None) -> bool:")
    append("        if action is None or action < 0 or action >= len(self.action_is_active_perception):")
    append("            return False")
    append("        return bool(self.action_is_active_perception[action])")
    append("")
    append("    def should_skip_observation_rule_before_check(")
    append("        self, observation_rule: int, current_action: int | None = None")
    append("    ) -> bool:")
    append("        if observation_rule < 0 or observation_rule >= len(self.observation_rule_skip_before_for_active_perception):")
    append("            return False")
    append("        return bool(")
    append("            self.observation_rule_skip_before_for_active_perception[observation_rule]")
    append("            and self.is_active_perception_action(current_action)")
    append("        )")
    append("")
    append("    def check_observation_rule_condition(")
    append("        self, observation_rule: int, state: int, current_action: int | None = None")
    append("    ) -> bool:")
    append("        if observation_rule < 0 or observation_rule >= len(CHECK_OBSERVATION_RULE_CONDITIONS):")
    append("            raise ValueError(f\"Unknown grounded observation rule id: {observation_rule}\")")
    append("        if self.should_skip_observation_rule_before_check(observation_rule, current_action):")
    append("            return False")
    append("        return CHECK_OBSERVATION_RULE_CONDITIONS[observation_rule](self, state)")
    append("")
    append("    def observe_with_rule(")
    append("        self, observation_rule: int, state: int, current_action: int | None = None")
    append("    ) -> tuple[int, int]:")
    append("        if observation_rule < 0 or observation_rule >= len(OBSERVE_WITH_RULES):")
    append("            raise ValueError(f\"Unknown grounded observation rule id: {observation_rule}\")")
    append("        if self.should_skip_observation_rule_before_check(observation_rule, current_action):")
    append("            return 0, 0")
    append("        return OBSERVE_WITH_RULES[observation_rule](self, state)")
    append("")
    append("    def check_default_policy_rule_condition(self, default_policy_rule: int, state: int) -> bool:")
    append("        if default_policy_rule < 0 or default_policy_rule >= len(CHECK_DEFAULT_POLICY_RULE_CONDITIONS):")
    append("            raise ValueError(f\"Unknown grounded default policy rule id: {default_policy_rule}\")")
    append("        return CHECK_DEFAULT_POLICY_RULE_CONDITIONS[default_policy_rule](self, state)")
    append("")
    append("    def get_default_policy_rule_action(self, default_policy_rule: int, state: int) -> int:")
    append("        if default_policy_rule < 0 or default_policy_rule >= len(GET_DEFAULT_POLICY_RULE_ACTIONS):")
    append("            raise ValueError(f\"Unknown grounded default policy rule id: {default_policy_rule}\")")
    append("        return GET_DEFAULT_POLICY_RULE_ACTIONS[default_policy_rule](self, state)")
    append("")
    append("")
    append("def build_model() -> POMDPModelBase:")
    append(f"    return {class_name}()")
    append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_action_modules(
    actions_dir: Path,
    semantic_model: SemanticPOMDPModelBase,
    bitwise_model: BitwiseModelSkeleton,
) -> None:
    grouped = _group_indices_by_name([action.name for action in semantic_model.actions])
    for action_name, indices in grouped.items():
        module_lines = _emit_group_module_header()
        for idx in indices:
            _emit_grounded_action_case(
                module_lines,
                idx,
                bitwise_model.action_precondition_checks[idx],
                bitwise_model.action_effect_distributions[idx],
                bitwise_model.action_effect_condition_checks[idx],
                bitwise_model.action_conditional_effect_distributions[idx],
            )
        (actions_dir / f"{_safe_module_name(action_name)}.py").write_text(
            "\n".join(module_lines),
            encoding="utf-8",
        )


def _write_observation_rule_modules(
    observation_rules_dir: Path,
    semantic_model: SemanticPOMDPModelBase,
    bitwise_model: BitwiseModelSkeleton,
) -> None:
    grouped = _group_indices_by_name([rule.name for rule in semantic_model.observation_rules])
    for rule_name, indices in grouped.items():
        module_lines = _emit_group_module_header()
        for idx in indices:
            _emit_grounded_observation_rule_case(
                module_lines,
                idx,
                bitwise_model.observation_rule_condition_checks[idx],
                bitwise_model.observation_rule_distributions[idx],
            )
        (observation_rules_dir / f"{_safe_module_name(rule_name)}.py").write_text(
            "\n".join(module_lines),
            encoding="utf-8",
        )


def _write_default_policy_rule_modules(
    default_policy_rules_dir: Path,
    semantic_model: SemanticPOMDPModelBase,
    bitwise_model: BitwiseModelSkeleton,
) -> None:
    grouped = _group_indices_by_name([rule.name for rule in semantic_model.default_policy_rules])
    for rule_name, indices in grouped.items():
        module_lines = _emit_group_module_header()
        for idx in indices:
            _emit_grounded_default_policy_rule_case(
                module_lines,
                idx,
                bitwise_model.default_policy_rule_condition_checks[idx],
                bitwise_model.default_policy_rule_action_ids[idx],
            )
        (default_policy_rules_dir / f"{_safe_module_name(rule_name)}.py").write_text(
            "\n".join(module_lines),
            encoding="utf-8",
        )


def _emit_group_module_header() -> list[str]:
    return [
        '"""Auto-generated grouped bitwise helper functions."""',
        "",
        "from __future__ import annotations",
        "",
        "from po_pddl.runtime.planning.bitwise import POMDPModelBase",
        "",
    ]


def _emit_grounded_action_case(
    lines: list[str],
    idx: int,
    precondition_check: BitwiseGoalCheck,
    unconditional_branches: list[BitwiseEffectBranch],
    conditional_checks: list[BitwiseGoalCheck],
    conditional_branches: list[list[BitwiseEffectBranch]],
) -> None:
    lines.append(f"def _check_action_precondition_{idx}(model: POMDPModelBase, state: int) -> bool:")
    lines.append(f"    return {_render_goal_check_expr(precondition_check, 'state')}")
    lines.append("")
    lines.append(f"def _forward_action_{idx}(model: POMDPModelBase, state: int) -> tuple[int, float]:")
    lines.append(f"    if not _check_action_precondition_{idx}(model, state):")
    lines.append(f"        raise ValueError(f\"Grounded action id `{idx}` is not applicable in the given state.\")")
    lines.append("    next_state = state")
    lines.append("    reward = 0.0")
    _emit_effect_sampling_block(lines, unconditional_branches, state_var="next_state", level=1)
    for guard_idx, (guard_check, branches) in enumerate(zip(conditional_checks, conditional_branches)):
        guard_expr = _render_goal_check_expr(guard_check, "state")
        lines.append(f"    if {guard_expr}:")
        _emit_effect_sampling_block(lines, branches, state_var="next_state", level=2)
        if not branches:
            lines.append("        pass")
    lines.append("    if model.is_goal(next_state):")
    lines.append("        reward += model.goal_reward")
    lines.append(f"    model._cache_transition_reward({idx}, state, next_state, reward)")
    lines.append("    return next_state, reward")
    lines.append("")
    lines.append(
        f"def _get_action_reward_{idx}(model: POMDPModelBase, state: int, next_state: int) -> float:"
    )
    lines.append(f"    cached_reward = model._lookup_cached_reward({idx}, state, next_state)")
    lines.append("    if cached_reward is not None:")
    lines.append("        return cached_reward")
    lines.append("    reward = 0.0")
    _emit_effect_reward_sampling_block(lines, unconditional_branches, level=1)
    for guard_check, branches in zip(conditional_checks, conditional_branches):
        guard_expr = _render_goal_check_expr(guard_check, "state")
        lines.append(f"    if {guard_expr}:")
        _emit_effect_reward_sampling_block(lines, branches, level=2)
        if not branches:
            lines.append("        pass")
    lines.append("    if model.is_goal(next_state):")
    lines.append("        reward += model.goal_reward")
    lines.append("    return reward")
    lines.append("")


def _emit_grounded_observation_rule_case(
    lines: list[str],
    idx: int,
    condition_check: BitwiseGoalCheck,
    branches: list[BitwiseObservationBranch],
) -> None:
    lines.append(
        f"def _check_observation_rule_condition_{idx}(model: POMDPModelBase, state: int) -> bool:"
    )
    lines.append(f"    return {_render_goal_check_expr(condition_check, 'state')}")
    lines.append("")
    lines.append(f"def _observe_with_rule_{idx}(model: POMDPModelBase, state: int) -> tuple[int, int]:")
    lines.append(f"    if not _check_observation_rule_condition_{idx}(model, state):")
    lines.append("        return 0, 0")
    _emit_observation_sampling_block(lines, branches, level=1)
    if not branches:
        lines.append("    return 0, 0")
    lines.append("")


def _emit_grounded_default_policy_rule_case(
    lines: list[str],
    idx: int,
    condition_check: BitwiseGoalCheck,
    action_id: int,
) -> None:
    lines.append(
        f"def _check_default_policy_rule_condition_{idx}(model: POMDPModelBase, state: int) -> bool:"
    )
    lines.append(f"    return {_render_goal_check_expr(condition_check, 'state')}")
    lines.append("")
    lines.append(
        f"def _get_default_policy_rule_action_{idx}(model: POMDPModelBase, state: int) -> int:"
    )
    lines.append(f"    if not _check_default_policy_rule_condition_{idx}(model, state):")
    lines.append(
        f"        raise ValueError(f\"Grounded default policy rule id `{idx}` is not applicable in the given state.\")"
    )
    lines.append(f"    return {action_id}")
    lines.append("")


def _emit_effect_sampling_block(
    lines: list[str],
    branches: list[BitwiseEffectBranch],
    *,
    state_var: str,
    level: int,
) -> None:
    indent = "    " * level
    if not branches:
        return
    if len(branches) == 1:
        wrote = _emit_effect_branch_apply(
            lines,
            branches[0],
            state_var=state_var,
            reward_var="reward",
            level=level,
        )
        if not wrote:
            lines.append(f"{indent}pass")
        return
    probs = [branch.probability for branch in branches]
    lines.append(f"{indent}branch_index = model._sample_branch_index({probs!r})")
    lines.append(f"{indent}if branch_index is None:")
    lines.append(f"{indent}    pass")
    for branch_idx, branch in enumerate(branches):
        keyword = "elif" if branch_idx > 0 else "elif"
        lines.append(f"{indent}{keyword} branch_index == {branch_idx}:")
        wrote = _emit_effect_branch_apply(
            lines,
            branch,
            state_var=state_var,
            reward_var="reward",
            level=level + 1,
        )
        if not wrote:
            lines.append(f"{indent}    pass")


def _emit_effect_reward_sampling_block(
    lines: list[str],
    branches: list[BitwiseEffectBranch],
    *,
    level: int,
) -> None:
    indent = "    " * level
    if not branches:
        return
    if len(branches) == 1:
        wrote = _emit_effect_reward_apply(
            lines,
            branches[0],
            reward_var="reward",
            level=level,
        )
        if not wrote:
            lines.append(f"{indent}pass")
        return
    probs = [branch.probability for branch in branches]
    lines.append(f"{indent}branch_index = model._sample_branch_index({probs!r})")
    lines.append(f"{indent}if branch_index is None:")
    lines.append(f"{indent}    pass")
    for branch_idx, branch in enumerate(branches):
        keyword = "elif" if branch_idx > 0 else "elif"
        lines.append(f"{indent}{keyword} branch_index == {branch_idx}:")
        wrote = _emit_effect_reward_apply(
            lines,
            branch,
            reward_var="reward",
            level=level + 1,
        )
        if not wrote:
            lines.append(f"{indent}    pass")


def _emit_effect_branch_apply(
    lines: list[str],
    branch: BitwiseEffectBranch,
    *,
    state_var: str,
    reward_var: str,
    level: int,
) -> bool:
    indent = "    " * level
    wrote = False
    if branch.set_mask != 0 or branch.clear_mask != 0:
        lines.append(f"{indent}{state_var} = ({state_var} | {branch.set_mask}) & ~{branch.clear_mask}")
        wrote = True
    if branch.reward_delta != 0.0:
        lines.append(f"{indent}{reward_var} += {branch.reward_delta!r}")
        wrote = True
    return wrote


def _emit_effect_reward_apply(
    lines: list[str],
    branch: BitwiseEffectBranch,
    *,
    reward_var: str,
    level: int,
) -> bool:
    indent = "    " * level
    if branch.reward_delta != 0.0:
        lines.append(f"{indent}{reward_var} += {branch.reward_delta!r}")
        return True
    return False

def _emit_observation_sampling_block(
    lines: list[str],
    branches: list[BitwiseObservationBranch],
    *,
    level: int,
) -> None:
    indent = "    " * level
    if not branches:
        return
    if len(branches) == 1:
        branch = branches[0]
        lines.append(f"{indent}return {branch.observation_bits}, {branch.observation_mask}")
        return
    probs = [branch.probability for branch in branches]
    lines.append(f"{indent}branch_index = model._sample_branch_index({probs!r})")
    lines.append(f"{indent}if branch_index is None:")
    lines.append(f"{indent}    return 0, 0")
    for branch_idx, branch in enumerate(branches):
        keyword = "elif" if branch_idx > 0 else "if"
        lines.append(f"{indent}{keyword} branch_index == {branch_idx}:")
        lines.append(f"{indent}    return {branch.observation_bits}, {branch.observation_mask}")
    lines.append(f"{indent}return 0, 0")


def _render_goal_check_expr(goal_check: BitwiseGoalCheck | None, state_var: str) -> str:
    if goal_check is None:
        return "False"
    if not goal_check.clauses:
        return "False"
    rendered_clauses = [_render_clause_expr(clause, state_var) for clause in goal_check.clauses]
    if len(rendered_clauses) == 1:
        return rendered_clauses[0]
    return "(" + " or ".join(rendered_clauses) + ")"


def _render_clause_expr(clause: tuple[int, int], state_var: str) -> str:
    required_true_mask, required_false_mask = clause
    if required_true_mask == 0 and required_false_mask == 0:
        return "True"
    parts: list[str] = []
    if required_true_mask != 0:
        parts.append(f"(({state_var} & {required_true_mask}) == {required_true_mask})")
    if required_false_mask != 0:
        parts.append(f"(({state_var} & {required_false_mask}) == 0)")
    if not parts:
        return "True"
    if len(parts) == 1:
        return parts[0]
    return "(" + " and ".join(parts) + ")"


def _group_indices_by_name(names: Iterable[str]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        grouped.setdefault(name, []).append(idx)
    return grouped


def _safe_module_name(name: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_").lower()
    if not text:
        return "group"
    if text[0].isdigit():
        return f"group_{text}"
    return text


def _make_class_name(name: str) -> str:
    parts = [part for part in re.split(r"[^0-9A-Za-z]+", name) if part]
    if not parts:
        return "GeneratedBitwiseModel"
    return "".join(part[:1].upper() + part[1:] for part in parts) + "BitwiseModel"
def _write_package_init(path: Path, class_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                '"""Auto-generated explicit bitwise model package."""',
                "",
                f"from .model import {class_name}, build_model",
                "",
                f"__all__ = [{class_name!r}, 'build_model']",
                "",
            ]
        ),
        encoding="utf-8",
    )
