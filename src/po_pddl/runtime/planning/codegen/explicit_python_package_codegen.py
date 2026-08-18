"""Emit explicit grounded Python model code from a code-generation plan."""

from __future__ import annotations

import json
import re
from pathlib import Path
from pprint import pformat

from ..data_structures import Action, DefaultPolicyRule, Observable, ObservationRule, Predicate
from ..parser.sexpr import SExpr
from .python_model_codegen import build_codegen_plan_from_texts
from .schemas import (
    GroundedActionCase,
    GroundedDefaultPolicyRuleCase,
    GroundedObservationRuleCase,
    PythonModelCodegenPlan,
)


def _is_last_action_predicate(predicate: Predicate) -> bool:
    return predicate.name.startswith("last_action_")


def _catalog_predicates(
    plan: PythonModelCodegenPlan,
    *,
    suppress_last_action_catalog: bool,
) -> list[Predicate]:
    if not suppress_last_action_catalog:
        return list(plan.predicates)
    return [predicate for predicate in plan.predicates if not _is_last_action_predicate(predicate)]


def emit_explicit_python_model_package(
    plan: PythonModelCodegenPlan,
    output_dir: str | Path,
    *,
    domain_text: str | None = None,
    problem_text: str | None = None,
    default_policy_text: str | None = None,
    suppress_last_action_catalog: bool = False,
) -> Path:
    """Write an explicit grounded Python model package split by action/rule kind."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actions_dir = out_dir / "actions"
    observation_rules_dir = out_dir / "observation_rules"
    default_policy_rules_dir = out_dir / "default_policy_rules"
    for directory in (actions_dir, observation_rules_dir, default_policy_rules_dir):
        directory.mkdir(parents=True, exist_ok=True)

    class_name = _make_class_name(plan.problem_name or plan.domain_name or "GeneratedModel")
    catalog_predicates = _catalog_predicates(
        plan,
        suppress_last_action_catalog=suppress_last_action_catalog,
    )
    type_parents = {
        name: (node.parent.name if node.parent is not None else None)
        for name, node in plan.types.items()
    }

    _write_shared_module(
        out_dir / "shared.py",
        plan,
        catalog_predicates,
        type_parents,
        domain_text=domain_text,
        problem_text=problem_text,
        default_policy_text=default_policy_text,
    )
    _write_init_belief_json(out_dir / "init_belief.json", plan, catalog_predicates)
    _write_action_modules(actions_dir, plan)
    _write_observation_rule_modules(observation_rules_dir, plan)
    _write_default_policy_rule_modules(default_policy_rules_dir, plan)
    _write_model_module(out_dir / "model.py", class_name, plan, catalog_predicates)
    _write_package_init(out_dir / "__init__.py", class_name)
    for subdir in (actions_dir, observation_rules_dir, default_policy_rules_dir):
        (subdir / "__init__.py").write_text('"""Auto-generated helper modules."""\n', encoding="utf-8")
    return out_dir


def emit_explicit_python_model_package_from_texts(
    domain_text: str,
    problem_text: str,
    output_dir: str | Path,
    *,
    default_policy_text: str | None = None,
    suppress_last_action_catalog: bool = False,
) -> Path:
    """Convenience wrapper: parse, ground, and emit one explicit Python model package."""
    plan = build_codegen_plan_from_texts(domain_text, problem_text, default_policy_text)
    return emit_explicit_python_model_package(
        plan,
        output_dir,
        domain_text=domain_text,
        problem_text=problem_text,
        default_policy_text=default_policy_text,
        suppress_last_action_catalog=suppress_last_action_catalog,
    )


def _emit_runtime_helpers(lines: list[str]) -> None:
    lines.extend(
        [
            "    def _copy_state(self, state: StateEntry) -> StateEntry:",
            "        return dict(state)",
            "",
            "    def _objects_of_type(self, type_name: str) -> list[str]:",
            "        universe = dict(CONSTANTS)",
            "        universe.update(OBJECTS)",
            "        if type_name == 'object':",
            "            return list(universe.keys())",
            "        return [",
            "            obj_name",
            "            for obj_name, obj_type in universe.items()",
            "            if self._is_subtype_name(obj_type, type_name)",
            "        ]",
            "",
            "    def _is_subtype_name(self, child_type: str, target_type: str) -> bool:",
            "        current = child_type",
            "        while current is not None:",
            "            if current == target_type:",
            "                return True",
            "            current = TYPE_PARENTS.get(current)",
            "        return False",
            "",
            "    def _state_value(self, state: StateEntry, predicate: Predicate) -> bool:",
            "        return state.get(predicate, False)",
            "",
            "    def _sample_branch_index(self, weights: list[float]) -> int | None:",
            "        cleaned = [max(weight, 0.0) for weight in weights]",
            "        total = sum(cleaned)",
            "        if total <= 0.0:",
            "            return None",
            "        draw = self._rng.random() * total",
            "        cumulative = 0.0",
            "        for idx, weight in enumerate(cleaned):",
            "            cumulative += weight",
            "            if draw <= cumulative:",
            "                return idx",
            "        return len(cleaned) - 1",
            "",
            "    def _cache_transition_reward(",
            "        self, action: Action, state: StateEntry, next_state: StateEntry, reward: float",
            "    ) -> None:",
            "        cache_key = (action, frozenset(state.items()), frozenset(next_state.items()))",
            "        self._last_transition_reward_cache[cache_key] = reward",
            "",
            "    def _lookup_cached_reward(",
            "        self, action: Action, state: StateEntry, next_state: StateEntry",
            "    ) -> float | None:",
            "        cache_key = (action, frozenset(state.items()), frozenset(next_state.items()))",
            "        return self._last_transition_reward_cache.get(cache_key)",
            "",
        ]
    )


def _write_shared_module(
    path: Path,
    plan: PythonModelCodegenPlan,
    catalog_predicates: list[Predicate],
    type_parents: dict[str, str | None],
    *,
    domain_text: str | None,
    problem_text: str | None,
    default_policy_text: str | None,
) -> None:
    lines: list[str] = []
    append = lines.append
    append('"""Auto-generated shared literals for an explicit grounded Python model package."""')
    append("")
    append("from __future__ import annotations")
    append("")
    append("from po_pddl.runtime.planning.data_structures import (")
    append("    Predicate,")
    append("    Observable,")
    append("    Action,")
    append("    ObservationRule,")
    append("    DefaultPolicyRule,")
    append(")")
    append("")
    if domain_text is not None:
        append(f"DOMAIN_TEXT = {domain_text!r}")
        append("")
    if problem_text is not None:
        append(f"PROBLEM_TEXT = {problem_text!r}")
        append("")
    if default_policy_text is not None:
        append(f"DEFAULT_POLICY_TEXT = {default_policy_text!r}")
        append("")
    append(f"DOMAIN_NAME = {plan.domain_name!r}")
    append(f"PROBLEM_NAME = {plan.problem_name!r}")
    append(f"MAXIMIZE_REWARD = {plan.maximize_reward!r}")
    append(f"GOAL_REWARD = {plan.goal_reward!r}")
    append("")
    append(f"TYPE_PARENTS = {pformat(type_parents, width=100)}")
    append(f"CONSTANTS = {pformat(plan.constants, width=100)}")
    append(f"OBJECTS = {pformat(plan.objects, width=100)}")
    append(f"GOAL_EXPR = {pformat(plan.goal_expr, width=100)}")
    append(f"GROUNDING_SUMMARY = {pformat(plan.summary(), width=100)}")
    append("")
    append(f"PREDICATES = {repr(catalog_predicates)}")
    append("")
    append(f"OBSERVABLES = {repr(plan.observables)}")
    append("")
    append(f"ACTIONS = {repr([case.action for case in plan.grounded_actions])}")
    append("")
    append(f"OBSERVATION_RULES = {repr([case.rule for case in plan.grounded_observation_rules])}")
    append("")
    append(f"DEFAULT_POLICY_RULES = {repr([case.rule for case in plan.grounded_default_policy_rules])}")
    append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_model_module(
    path: Path,
    class_name: str,
    plan: PythonModelCodegenPlan,
    catalog_predicates: list[Predicate],
) -> None:
    lines: list[str] = []
    append = lines.append
    append('"""Auto-generated explicit grounded Python model package entrypoint."""')
    append("")
    append("from __future__ import annotations")
    append("")
    append("from po_pddl.runtime.planning.base import POMDPModelBase")
    append("from po_pddl.runtime.planning.data_structures import (")
    append("    Action,")
    append("    Observable,")
    append("    ObservationRule,")
    append("    DefaultPolicyRule,")
    append("    Predicate,")
    append("    StateEntry,")
    append("    ObservationEntry,")
    append(")")
    append("")
    append("from .shared import (")
    append("    PREDICATES,")
    append("    OBSERVABLES,")
    append("    ACTIONS,")
    append("    OBSERVATION_RULES,")
    append("    DEFAULT_POLICY_RULES,")
    append("    MAXIMIZE_REWARD,")
    append("    GOAL_REWARD,")
    append("    TYPE_PARENTS,")
    append("    CONSTANTS,")
    append("    OBJECTS,")
    append("    GOAL_EXPR,")
    append(")")
    append("from .actions import " + ", ".join(
        f"{_safe_module_name(name)} as action_group_{_safe_module_name(name)}"
        for name in _ordered_unique(case.schema.action.name for case in plan.grounded_actions)
    ) if plan.grounded_actions else "")
    if plan.grounded_observation_rules:
        append("from .observation_rules import " + ", ".join(
            f"{_safe_module_name(case.schema.rule.name)} as observation_rule_group_{_safe_module_name(case.schema.rule.name)}"
            for case in _ordered_unique_cases(plan.grounded_observation_rules, lambda c: c.schema.rule.name)
        ))
    if plan.grounded_default_policy_rules:
        append("from .default_policy_rules import " + ", ".join(
            f"{_safe_module_name(case.schema.rule.name)} as default_policy_rule_group_{_safe_module_name(case.schema.rule.name)}"
            for case in _ordered_unique_cases(plan.grounded_default_policy_rules, lambda c: c.schema.rule.name)
        ))
    append("")
    if plan.grounded_actions:
        append(f"ACTION_INDEX = {repr({case.action: idx for idx, case in enumerate(plan.grounded_actions)})}")
        append("CHECK_ACTION_PRECONDITIONS = [")
        for idx, case in enumerate(plan.grounded_actions):
            mod = f"action_group_{_safe_module_name(case.schema.action.name)}"
            append(f"    {mod}._check_action_precondition_{idx},")
        append("]")
        append("FORWARD_ACTIONS = [")
        for idx, case in enumerate(plan.grounded_actions):
            mod = f"action_group_{_safe_module_name(case.schema.action.name)}"
            append(f"    {mod}._forward_action_{idx},")
        append("]")
        append("GET_ACTION_REWARDS = [")
        for idx, case in enumerate(plan.grounded_actions):
            mod = f"action_group_{_safe_module_name(case.schema.action.name)}"
            append(f"    {mod}._get_action_reward_{idx},")
        append("]")
        append("")
    if plan.grounded_observation_rules:
        append(
            f"OBSERVATION_RULE_INDEX = {repr({case.rule: idx for idx, case in enumerate(plan.grounded_observation_rules)})}"
        )
        append("CHECK_OBSERVATION_RULE_CONDITIONS = [")
        for idx, case in enumerate(plan.grounded_observation_rules):
            mod = f"observation_rule_group_{_safe_module_name(case.schema.rule.name)}"
            append(f"    {mod}._check_observation_rule_condition_{idx},")
        append("]")
        append("OBSERVE_WITH_RULES = [")
        for idx, case in enumerate(plan.grounded_observation_rules):
            mod = f"observation_rule_group_{_safe_module_name(case.schema.rule.name)}"
            append(f"    {mod}._observe_with_rule_{idx},")
        append("]")
        append("")
    if plan.grounded_default_policy_rules:
        append(
            "DEFAULT_POLICY_RULE_INDEX = "
            + repr({case.rule: idx for idx, case in enumerate(plan.grounded_default_policy_rules)})
        )
        append("CHECK_DEFAULT_POLICY_RULE_CONDITIONS = [")
        for idx, case in enumerate(plan.grounded_default_policy_rules):
            mod = f"default_policy_rule_group_{_safe_module_name(case.schema.rule.name)}"
            append(f"    {mod}._check_default_policy_rule_condition_{idx},")
        append("]")
        append("GET_DEFAULT_POLICY_RULE_ACTIONS = [")
        for idx, case in enumerate(plan.grounded_default_policy_rules):
            mod = f"default_policy_rule_group_{_safe_module_name(case.schema.rule.name)}"
            append(f"    {mod}._get_default_policy_rule_action_{idx},")
        append("]")
        append("")
    append(f"class {class_name}(POMDPModelBase):")
    append('    """Concrete explicit grounded model class."""')
    append("")
    append("    def __init__(self) -> None:")
    append("        super().__init__(")
    append("            predicates=list(PREDICATES),")
    append("            observables=list(OBSERVABLES),")
    append("            actions=list(ACTIONS),")
    append("            observation_rules=list(OBSERVATION_RULES),")
    append("            default_policy_rules=list(DEFAULT_POLICY_RULES),")
    append("            maximize_reward=MAXIMIZE_REWARD,")
    append("            goal_reward=GOAL_REWARD,")
    append("        )")
    append("")
    _emit_runtime_helpers(lines)
    append("    def is_goal(self, state: StateEntry) -> bool:")
    if plan.goal_expr is None:
        append("        return False")
    else:
        goal_lines = _compile_bool_expr_lines(
            plan.goal_expr, {}, plan.types, level=2, target_name="goal_reached", state_map_name="state"
        )
        lines.extend(goal_lines)
        append("        return goal_reached")
    append("")
    append("    def check_action_precondition(self, action: Action, state: StateEntry) -> bool:")
    if plan.grounded_actions:
        append("        action_index = ACTION_INDEX.get(action)")
        append("        if action_index is None:")
        append("            raise ValueError(f'Unknown action: {action}')")
        append("        return CHECK_ACTION_PRECONDITIONS[action_index](self, state)")
    else:
        append("        raise ValueError(f'Unknown action: {action}')")
    append("")
    append("    def forward_action(self, action: Action, state: StateEntry) -> tuple[StateEntry, float]:")
    if plan.grounded_actions:
        append("        action_index = ACTION_INDEX.get(action)")
        append("        if action_index is None:")
        append("            raise ValueError(f'Unknown action: {action}')")
        append("        return FORWARD_ACTIONS[action_index](self, state)")
    else:
        append("        raise ValueError(f'Unknown action: {action}')")
    append("")
    append("    def get_action_reward(")
    append("        self, action: Action, state: StateEntry, next_state: StateEntry")
    append("    ) -> float:")
    if plan.grounded_actions:
        append("        action_index = ACTION_INDEX.get(action)")
        append("        if action_index is None:")
        append("            raise ValueError(f'Unknown action: {action}')")
        append("        return GET_ACTION_REWARDS[action_index](self, state, next_state)")
    else:
        append("        raise ValueError(f'Unknown action: {action}')")
    append("")
    append("    def check_observation_rule_condition(")
    append("        self, observation_rule: ObservationRule, state: StateEntry, current_action: Action | None = None")
    append("    ) -> bool:")
    if plan.grounded_observation_rules:
        append("        if self.should_skip_observation_rule_before_check(observation_rule, current_action):")
        append("            return False")
        append("        rule_index = OBSERVATION_RULE_INDEX.get(observation_rule)")
        append("        if rule_index is None:")
        append("            raise ValueError(f'Unknown observation rule: {observation_rule}')")
        append("        return CHECK_OBSERVATION_RULE_CONDITIONS[rule_index](self, state)")
    else:
        append("        raise ValueError(f'Unknown observation rule: {observation_rule}')")
    append("")
    append("    def observe_with_rule(")
    append("        self, observation_rule: ObservationRule, state: StateEntry, current_action: Action | None = None")
    append("    ) -> ObservationEntry:")
    if plan.grounded_observation_rules:
        append("        if self.should_skip_observation_rule_before_check(observation_rule, current_action):")
        append("            return {}")
        append("        rule_index = OBSERVATION_RULE_INDEX.get(observation_rule)")
        append("        if rule_index is None:")
        append("            raise ValueError(f'Unknown observation rule: {observation_rule}')")
        append("        return OBSERVE_WITH_RULES[rule_index](self, state)")
    else:
        append("        raise ValueError(f'Unknown observation rule: {observation_rule}')")
    append("")
    append("    def check_default_policy_rule_condition(")
    append("        self, default_policy_rule: DefaultPolicyRule, state: StateEntry")
    append("    ) -> bool:")
    if plan.grounded_default_policy_rules:
        append("        rule_index = DEFAULT_POLICY_RULE_INDEX.get(default_policy_rule)")
        append("        if rule_index is None:")
        append("            raise ValueError(f'Unknown default policy rule: {default_policy_rule}')")
        append("        return CHECK_DEFAULT_POLICY_RULE_CONDITIONS[rule_index](self, state)")
    else:
        append("        raise ValueError(f'Unknown default policy rule: {default_policy_rule}')")
    append("")
    append("    def get_default_policy_rule_action(")
    append("        self, default_policy_rule: DefaultPolicyRule, state: StateEntry")
    append("    ) -> Action:")
    if plan.grounded_default_policy_rules:
        append("        rule_index = DEFAULT_POLICY_RULE_INDEX.get(default_policy_rule)")
        append("        if rule_index is None:")
        append("            raise ValueError(f'Unknown default policy rule: {default_policy_rule}')")
        append("        return GET_DEFAULT_POLICY_RULE_ACTIONS[rule_index](self, state)")
    else:
        append("        raise ValueError(f'Unknown default policy rule: {default_policy_rule}')")
    append("")
    append("")
    append("def build_model() -> POMDPModelBase:")
    append(f"    return {class_name}()")
    append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_init_belief_json(
    path: Path,
    plan: PythonModelCodegenPlan,
    catalog_predicates: list[Predicate],
) -> None:
    plan.init_belief.validate()
    predicate_index = {
        predicate: index for index, predicate in enumerate(catalog_predicates)
    }
    factors: list[dict[str, object]] = []
    for factor in plan.init_belief.factors:
        retained_scope = [
            predicate for predicate in factor.scope if predicate in predicate_index
        ]
        if not retained_scope:
            continue

        # Marginalize suppressed internal predicates and coalesce projected cases.
        projected_cases: dict[tuple[Predicate, ...], float] = {}
        for probability, true_predicates in factor.cases:
            true_set = set(true_predicates)
            projected_true = tuple(
                predicate for predicate in retained_scope if predicate in true_set
            )
            projected_cases[projected_true] = (
                projected_cases.get(projected_true, 0.0) + probability
            )
        factors.append(
            {
                "scope": [predicate_index[predicate] for predicate in retained_scope],
                "cases": [
                    {
                        "probability": probability,
                        "true_indices": [
                            predicate_index[predicate] for predicate in true_predicates
                        ],
                    }
                    for true_predicates, probability in projected_cases.items()
                ],
            }
        )

    json_data = {
        "grounded_predicates_count": len(catalog_predicates),
        "known_true": [
            predicate_index[predicate]
            for predicate in plan.init_belief.known_true
            if predicate in predicate_index
        ],
        "known_false": [
            predicate_index[predicate]
            for predicate in plan.init_belief.known_false
            if predicate in predicate_index
        ],
        "factors": factors,
    }
    path.write_text(
        json.dumps(json_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_action_modules(actions_dir: Path, plan: PythonModelCodegenPlan) -> None:
    grouped: dict[str, list[tuple[int, GroundedActionCase]]] = {}
    for idx, case in enumerate(plan.grounded_actions):
        grouped.setdefault(case.schema.action.name, []).append((idx, case))
    for action_name, indexed_cases in grouped.items():
        module_lines = _emit_group_module_header()
        for idx, case in indexed_cases:
            temp: list[str] = []
            _emit_grounded_action_case(temp, idx, case, plan)
            module_lines.extend(_convert_method_block_to_module_functions(temp))
        (actions_dir / f"{_safe_module_name(action_name)}.py").write_text("\n".join(module_lines) + "\n", encoding="utf-8")


def _write_observation_rule_modules(observation_rules_dir: Path, plan: PythonModelCodegenPlan) -> None:
    grouped: dict[str, list[tuple[int, GroundedObservationRuleCase]]] = {}
    for idx, case in enumerate(plan.grounded_observation_rules):
        grouped.setdefault(case.schema.rule.name, []).append((idx, case))
    for rule_name, indexed_cases in grouped.items():
        module_lines = _emit_group_module_header()
        for idx, case in indexed_cases:
            temp: list[str] = []
            _emit_grounded_observation_rule_case(temp, idx, case, plan)
            module_lines.extend(_convert_method_block_to_module_functions(temp))
        (observation_rules_dir / f"{_safe_module_name(rule_name)}.py").write_text("\n".join(module_lines) + "\n", encoding="utf-8")


def _write_default_policy_rule_modules(default_policy_rules_dir: Path, plan: PythonModelCodegenPlan) -> None:
    grouped: dict[str, list[tuple[int, GroundedDefaultPolicyRuleCase]]] = {}
    for idx, case in enumerate(plan.grounded_default_policy_rules):
        grouped.setdefault(case.schema.rule.name, []).append((idx, case))
    for rule_name, indexed_cases in grouped.items():
        module_lines = _emit_group_module_header()
        for idx, case in indexed_cases:
            temp: list[str] = []
            _emit_grounded_default_policy_rule_case(temp, idx, case, plan)
            module_lines.extend(_convert_method_block_to_module_functions(temp))
        (default_policy_rules_dir / f"{_safe_module_name(rule_name)}.py").write_text("\n".join(module_lines) + "\n", encoding="utf-8")


def _write_package_init(path: Path, class_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                '"""Auto-generated explicit grounded Python model package."""',
                "",
                f"from .model import {class_name}, build_model",
                "",
                f"__all__ = [{class_name!r}, 'build_model']",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _emit_group_module_header() -> list[str]:
    return [
        '"""Auto-generated grouped helper functions."""',
        "",
        "from __future__ import annotations",
        "",
        "from po_pddl.runtime.planning.base import POMDPModelBase",
        "from po_pddl.runtime.planning.data_structures import (",
        "    Predicate,",
        "    Observable,",
        "    Action,",
        "    ObservationRule,",
        "    DefaultPolicyRule,",
        "    StateEntry,",
        "    ObservationEntry,",
        ")",
        "",
    ]


def _convert_method_block_to_module_functions(lines: list[str]) -> list[str]:
    converted: list[str] = []
    for line in lines:
        if not line:
            converted.append("")
            continue
        if line.startswith("    def "):
            line = line[4:]
            line = re.sub(r"\bself\b", "model", line)
            line = re.sub(r"\(model,", "(model: POMDPModelBase,", line, count=1)
            line = re.sub(r"\(model\)", "(model: POMDPModelBase)", line, count=1)
            converted.append(line)
            continue
        if line.startswith("    "):
            line = line[4:]
        line = re.sub(r"\bself\b", "model", line)
        converted.append(line)
    return converted


def _safe_module_name(name: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_").lower()
    if not text:
        return "group"
    if text[0].isdigit():
        return f"group_{text}"
    return text


def _ordered_unique(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _ordered_unique_cases(cases, key_fn):
    seen: set[str] = set()
    result = []
    for case in cases:
        key = key_fn(case)
        if key not in seen:
            seen.add(key)
            result.append(case)
    return result


def _emit_grounded_action_case(
    lines: list[str],
    idx: int,
    case: GroundedActionCase,
    plan: PythonModelCodegenPlan,
) -> None:
    action = case.action
    schema = case.schema
    bindings = case.bindings
    condition_lines = _compile_bool_expr_lines(
        schema.precondition,
        bindings,
        plan.types,
        level=2,
        target_name="precondition_holds",
        state_map_name="state",
    ) if schema.precondition is not None else ["        precondition_holds = True"]

    lines.append(f"    def _check_action_precondition_{idx}(self, state: StateEntry) -> bool:")
    lines.extend(condition_lines)
    lines.append("        return precondition_holds")
    lines.append("")

    lines.append(f"    def _forward_action_{idx}(self, state: StateEntry) -> tuple[StateEntry, float]:")
    lines.append(f"        action = {repr(action)}")
    lines.extend(condition_lines)
    lines.append("        if not precondition_holds:")
    lines.append("            raise ValueError(f'Action `{action.name}` is not applicable in the given state.')")
    lines.append("        next_state = self._copy_state(state)")
    lines.append("        reward = 0.0")
    if schema.effect is not None:
        lines.extend(
            _compile_effect_statements(
                schema.effect,
                bindings,
                plan.types,
                level=2,
                effect_bucket_annotations=schema.effect_bucket_annotations,
                track_effect_bucket=True,
            )
        )
    lines.append("        if self.is_goal(next_state):")
    lines.append("            reward += self.goal_reward")
    lines.append("        self._cache_transition_reward(action, state, next_state, reward)")
    lines.append("        return next_state, reward")
    lines.append("")

    lines.append(
        f"    def _get_action_reward_{idx}(self, state: StateEntry, next_state: StateEntry) -> float:"
    )
    lines.append(f"        action = {repr(action)}")
    lines.append("        cached_reward = self._lookup_cached_reward(action, state, next_state)")
    lines.append("        if cached_reward is not None:")
    lines.append("            return cached_reward")
    lines.append("        reward = 0.0")
    if schema.effect is not None:
        lines.extend(
            _compile_effect_statements(
                schema.effect,
                bindings,
                plan.types,
                level=2,
                apply_state_updates=False,
                effect_bucket_annotations=schema.effect_bucket_annotations,
                track_effect_bucket=False,
            )
        )
    lines.append("        if self.is_goal(next_state):")
    lines.append("            reward += self.goal_reward")
    lines.append("        return reward")
    lines.append("")


def _emit_grounded_observation_rule_case(
    lines: list[str],
    idx: int,
    case: GroundedObservationRuleCase,
    plan: PythonModelCodegenPlan,
) -> None:
    rule = case.rule
    schema = case.schema
    bindings = case.bindings
    condition_lines = _compile_bool_expr_lines(
        schema.condition,
        bindings,
        plan.types,
        level=2,
        target_name="condition_holds",
        state_map_name="state",
    ) if schema.condition is not None else ["        condition_holds = True"]

    lines.append(
        f"    def _check_observation_rule_condition_{idx}(self, state: StateEntry) -> bool:"
    )
    lines.extend(condition_lines)
    lines.append("        return condition_holds")
    lines.append("")

    lines.append(f"    def _observe_with_rule_{idx}(self, state: StateEntry) -> ObservationEntry:")
    lines.extend(condition_lines)
    lines.append("        if not condition_holds:")
    lines.append("            return {}")
    if schema.distribution_expr is None:
        lines.append("        return {}")
    else:
        lines.extend(
            _compile_observation_distribution(
                schema.distribution_expr,
                bindings,
                plan.types,
                level=2,
                target_name="observation_result",
            )
        )
        lines.append("        return observation_result")
    lines.append("")


def _emit_grounded_default_policy_rule_case(
    lines: list[str],
    idx: int,
    case: GroundedDefaultPolicyRuleCase,
    plan: PythonModelCodegenPlan,
) -> None:
    rule = case.rule
    schema = case.schema
    bindings = case.bindings
    condition_lines = _compile_bool_expr_lines(
        schema.precondition,
        bindings,
        plan.types,
        level=2,
        target_name="condition_holds",
        state_map_name="state",
    ) if schema.precondition is not None else ["        condition_holds = True"]

    lines.append(
        f"    def _check_default_policy_rule_condition_{idx}(self, state: StateEntry) -> bool:"
    )
    lines.extend(condition_lines)
    lines.append("        return condition_holds")
    lines.append("")

    lines.append(
        f"    def _get_default_policy_rule_action_{idx}(self, state: StateEntry) -> Action:"
    )
    lines.extend(condition_lines)
    lines.append("        if not condition_holds:")
    lines.append(
        f"            raise ValueError('Default policy rule `{rule.name}` is not applicable in the given state.')"
    )
    if schema.action_expr is None or not isinstance(schema.action_expr, list) or not schema.action_expr:
        lines.append("        raise ValueError('Malformed default policy action expression.')")
    else:
        action_name = schema.action_expr[0]
        action_params = [
            _compile_term(token, bindings, {})
            for token in schema.action_expr[1:]
        ]
        lines.append(f"        return Action({action_name!r}, [{', '.join(action_params)}])")
    lines.append("")


def _compile_bool_expr_lines(
    expr: SExpr | None,
    fixed_bindings: dict[str, str],
    types: dict[str, object],
    *,
    level: int,
    target_name: str,
    state_map_name: str,
) -> list[str]:
    indent = "    " * level
    if expr is None:
        return [f"{indent}{target_name} = True"]
    expression = _compile_bool_expr(expr, fixed_bindings, {}, types, state_map_name)
    return [f"{indent}{target_name} = {expression}"]


def _compile_bool_expr(
    expr: SExpr,
    fixed_bindings: dict[str, str],
    local_vars: dict[str, str],
    types: dict[str, object],
    state_map_name: str,
) -> str:
    if isinstance(expr, str):
        raise ValueError(f"Unexpected bare symbol in boolean expression: {expr}")
    if not expr:
        return "True"
    head = expr[0]
    if head == "and":
        if len(expr) == 1:
            return "True"
        return "(" + " and ".join(
            _compile_bool_expr(subexpr, fixed_bindings, local_vars, types, state_map_name)
            for subexpr in expr[1:]
        ) + ")"
    if head == "or":
        if len(expr) == 1:
            return "False"
        return "(" + " or ".join(
            _compile_bool_expr(subexpr, fixed_bindings, local_vars, types, state_map_name)
            for subexpr in expr[1:]
        ) + ")"
    if head == "not":
        return f"(not {_compile_bool_expr(expr[1], fixed_bindings, local_vars, types, state_map_name)})"
    if head == "imply":
        left = _compile_bool_expr(expr[1], fixed_bindings, local_vars, types, state_map_name)
        right = _compile_bool_expr(expr[2], fixed_bindings, local_vars, types, state_map_name)
        return f"((not {left}) or {right})"
    if head == "=":
        left = _compile_term(expr[1], fixed_bindings, local_vars)
        right = _compile_term(expr[2], fixed_bindings, local_vars)
        return f"({left} == {right})"
    if head in {"forall", "all", "exists"}:
        typed_vars = _parse_typed_variables(expr[1])
        iterators = []
        extended_locals = dict(local_vars)
        for raw_name, type_name in typed_vars:
            py_var = _safe_var_name(raw_name)
            suffix = 1
            while py_var in extended_locals.values():
                suffix += 1
                py_var = f"{_safe_var_name(raw_name)}_{suffix}"
            extended_locals[raw_name] = py_var
            iterators.append((py_var, type_name))
        body = _compile_bool_expr(expr[2], fixed_bindings, extended_locals, types, state_map_name)
        generators = " ".join(
            f"for {py_var} in self._objects_of_type({type_name!r})"
            for py_var, type_name in iterators
        )
        aggregator = "all" if head in {"forall", "all"} else "any"
        return f"{aggregator}(({body}) {generators})"

    params = [
        _compile_term(token, fixed_bindings, local_vars)
        for token in expr[1:]
    ]
    predicate_expr = f"Predicate({head!r}, [{', '.join(params)}])"
    return f"self._state_value({state_map_name}, {predicate_expr})"


def _compile_effect_statements(
    expr: SExpr,
    fixed_bindings: dict[str, str],
    types: dict[str, object],
    *,
    level: int,
    apply_state_updates: bool = True,
    effect_bucket_annotations=None,
    track_effect_bucket: bool = False,
) -> list[str]:
    lines: list[str] = []
    _emit_effect(
        expr,
        fixed_bindings,
        {},
        types,
        level,
        lines,
        apply_state_updates,
        effect_bucket_annotations=effect_bucket_annotations,
        track_effect_bucket=track_effect_bucket,
        probabilistic_depth=0,
    )
    return lines


def _emit_effect(
    expr: SExpr,
    fixed_bindings: dict[str, str],
    local_vars: dict[str, str],
    types: dict[str, object],
    level: int,
    lines: list[str],
    apply_state_updates: bool,
    *,
    effect_bucket_annotations=None,
    track_effect_bucket: bool,
    probabilistic_depth: int,
) -> None:
    indent = "    " * level
    if isinstance(expr, str):
        raise ValueError(f"Unexpected bare symbol in effect: {expr}")
    if not expr:
        return
    head = expr[0]
    if head == "and":
        for subexpr in expr[1:]:
            _emit_effect(
                subexpr,
                fixed_bindings,
                local_vars,
                types,
                level,
                lines,
                apply_state_updates,
                effect_bucket_annotations=effect_bucket_annotations,
                track_effect_bucket=track_effect_bucket,
                probabilistic_depth=probabilistic_depth,
            )
        return
    if head == "when":
        condition = _compile_bool_expr(expr[1], fixed_bindings, local_vars, types, "state")
        lines.append(f"{indent}if {condition}:")
        before_len = len(lines)
        _emit_effect(
            expr[2],
            fixed_bindings,
            local_vars,
            types,
            level + 1,
            lines,
            apply_state_updates,
            effect_bucket_annotations=effect_bucket_annotations,
            track_effect_bucket=track_effect_bucket,
            probabilistic_depth=probabilistic_depth,
        )
        if len(lines) == before_len:
            lines.append(f"{indent}    pass")
        return
    if head == "forall":
        typed_vars = _parse_typed_variables(expr[1])
        extended_locals = dict(local_vars)
        for raw_name, type_name in typed_vars:
            py_var = _unique_local_name(raw_name, extended_locals)
            extended_locals[raw_name] = py_var
            lines.append(f"{indent}for {py_var} in self._objects_of_type({type_name!r}):")
            indent += "    "
        before_len = len(lines)
        _emit_effect(
            expr[2],
            fixed_bindings,
            extended_locals,
            types,
            level + len(typed_vars),
            lines,
            apply_state_updates,
            effect_bucket_annotations=effect_bucket_annotations,
            track_effect_bucket=track_effect_bucket,
            probabilistic_depth=probabilistic_depth,
        )
        if len(lines) == before_len:
            lines.append(f"{indent}pass")
        return
    if head == "probabilistic":
        weights = [
            _compile_numeric_expr(expr[index], fixed_bindings, local_vars)
            for index in range(1, len(expr), 2)
        ]
        branch_count = len(range(2, len(expr), 2))
        if track_effect_bucket and probabilistic_depth == 0:
            bucket_names = [
                (
                    effect_bucket_annotations[branch_idx].bucket_name
                    if effect_bucket_annotations is not None and branch_idx < len(effect_bucket_annotations)
                    else None
                )
                for branch_idx in range(branch_count)
            ]
            successes = [
                (
                    effect_bucket_annotations[branch_idx].success
                    if effect_bucket_annotations is not None and branch_idx < len(effect_bucket_annotations)
                    else None
                )
                for branch_idx in range(branch_count)
            ]
            lines.append(
                f"{indent}_branch_index = self._select_effect_branch_index("
                f"action=action, "
                f"weights=[{', '.join(weights)}], "
                f"bucket_names={bucket_names!r}, "
                f"successes={successes!r}, "
                f"record_sample=True)"
            )
        else:
            lines.append(f"{indent}_branch_index = self._sample_branch_index([{', '.join(weights)}])")
        for branch_idx, branch_pos in enumerate(range(2, len(expr), 2)):
            keyword = "if" if branch_idx == 0 else "elif"
            lines.append(f"{indent}{keyword} _branch_index == {branch_idx}:")
            branch_lines: list[str] = []
            _emit_effect(
                expr[branch_pos],
                fixed_bindings,
                local_vars,
                types,
                level + 1,
                branch_lines,
                apply_state_updates,
                effect_bucket_annotations=effect_bucket_annotations,
                track_effect_bucket=track_effect_bucket,
                probabilistic_depth=probabilistic_depth + 1,
            )
            if branch_lines:
                lines.extend(branch_lines)
            else:
                lines.append(f"{indent}    pass")
        return
    if head == "not":
        if apply_state_updates:
            predicate_expr = _compile_ground_predicate_expr(expr[1], fixed_bindings, local_vars)
            lines.append(f"{indent}next_state[{predicate_expr}] = False")
        return
    if head in {"increase", "decrease", "assign"}:
        function_expr = expr[1]
        function_name = function_expr[0] if isinstance(function_expr, list) and function_expr else None
        if function_name is not None and "reward" in function_name:
            amount = _compile_numeric_expr(expr[2], fixed_bindings, local_vars)
            if head == "increase":
                lines.append(f"{indent}reward += {amount}")
            elif head == "decrease":
                lines.append(f"{indent}reward -= {amount}")
            else:
                lines.append(f"{indent}reward = {amount}")
        return
    if apply_state_updates:
        predicate_expr = _compile_ground_predicate_expr(expr, fixed_bindings, local_vars)
        lines.append(f"{indent}next_state[{predicate_expr}] = True")


def _compile_numeric_expr(
    expr: SExpr,
    fixed_bindings: dict[str, str],
    local_vars: dict[str, str],
) -> str:
    if isinstance(expr, str):
        return expr
    head = expr[0]
    if head == "+":
        return "(" + " + ".join(_compile_numeric_expr(subexpr, fixed_bindings, local_vars) for subexpr in expr[1:]) + ")"
    if head == "-":
        if len(expr) == 2:
            return f"(-{_compile_numeric_expr(expr[1], fixed_bindings, local_vars)})"
        return "(" + " - ".join(_compile_numeric_expr(subexpr, fixed_bindings, local_vars) for subexpr in expr[1:]) + ")"
    if head == "*":
        return "(" + " * ".join(_compile_numeric_expr(subexpr, fixed_bindings, local_vars) for subexpr in expr[1:]) + ")"
    if head == "/":
        return "(" + " / ".join(_compile_numeric_expr(subexpr, fixed_bindings, local_vars) for subexpr in expr[1:]) + ")"
    raise ValueError(f"Unsupported numeric expression: {expr}")


def _compile_observation_distribution(
    expr: SExpr,
    fixed_bindings: dict[str, str],
    types: dict[str, object],
    *,
    level: int,
    target_name: str,
) -> list[str]:
    lines: list[str] = []
    _emit_observation_distribution(
        expr,
        fixed_bindings,
        {},
        types,
        level,
        lines,
        target_name,
    )
    return lines


def _emit_observation_distribution(
    expr: SExpr,
    fixed_bindings: dict[str, str],
    local_vars: dict[str, str],
    types: dict[str, object],
    level: int,
    lines: list[str],
    target_name: str,
) -> None:
    indent = "    " * level
    if isinstance(expr, str):
        raise ValueError(f"Unexpected bare symbol in observation distribution: {expr}")
    if not expr:
        lines.append(f"{indent}{target_name} = {{}}")
        return
    head = expr[0]
    if head == "probabilistic":
        weights = [
            _compile_numeric_expr(expr[index], fixed_bindings, local_vars)
            for index in range(1, len(expr), 2)
        ]
        lines.append(f"{indent}_obs_branch_index = self._sample_branch_index([{', '.join(weights)}])")
        for branch_idx, branch_pos in enumerate(range(2, len(expr), 2)):
            keyword = "if" if branch_idx == 0 else "elif"
            lines.append(f"{indent}{keyword} _obs_branch_index == {branch_idx}:")
            _emit_observation_distribution(
                expr[branch_pos],
                fixed_bindings,
                local_vars,
                types,
                level + 1,
                lines,
                target_name,
            )
        lines.append(f"{indent}else:")
        lines.append(f"{indent}    {target_name} = {{}}")
        return
    if head == "and":
        entries: list[str] = []
        for subexpr in expr[1:]:
            entries.extend(_flatten_observation_entries(subexpr, fixed_bindings, local_vars))
        lines.append(f"{indent}{target_name} = {{{', '.join(entries)}}}")
        return
    entries = _flatten_observation_entries(expr, fixed_bindings, local_vars)
    lines.append(f"{indent}{target_name} = {{{', '.join(entries)}}}")


def _flatten_observation_entries(
    expr: SExpr,
    fixed_bindings: dict[str, str],
    local_vars: dict[str, str],
) -> list[str]:
    if isinstance(expr, str):
        raise ValueError(f"Unexpected bare symbol in observation expression: {expr}")
    if not expr:
        return []
    head = expr[0]
    if head == "and":
        entries: list[str] = []
        for subexpr in expr[1:]:
            entries.extend(_flatten_observation_entries(subexpr, fixed_bindings, local_vars))
        return entries
    if head == "not":
        observable_expr = _compile_ground_observable_expr(expr[1], fixed_bindings, local_vars)
        return [f"{observable_expr}: False"]
    observable_expr = _compile_ground_observable_expr(expr, fixed_bindings, local_vars)
    return [f"{observable_expr}: True"]


def _compile_ground_predicate_expr(
    expr: SExpr,
    fixed_bindings: dict[str, str],
    local_vars: dict[str, str],
) -> str:
    if isinstance(expr, str):
        raise ValueError(f"Expected predicate expression, got bare symbol: {expr}")
    name = expr[0]
    params = ", ".join(_compile_term(token, fixed_bindings, local_vars) for token in expr[1:])
    return f"Predicate({name!r}, [{params}])"


def _compile_ground_observable_expr(
    expr: SExpr,
    fixed_bindings: dict[str, str],
    local_vars: dict[str, str],
) -> str:
    if isinstance(expr, str):
        raise ValueError(f"Expected observable expression, got bare symbol: {expr}")
    name = expr[0]
    params = ", ".join(_compile_term(token, fixed_bindings, local_vars) for token in expr[1:])
    return f"Observable({name!r}, [{params}])"


def _compile_term(
    token,
    fixed_bindings: dict[str, str],
    local_vars: dict[str, str],
) -> str:
    if not isinstance(token, str):
        raise ValueError(f"Only flat symbol terms are supported, got: {token!r}")
    if token.startswith("?"):
        if token in local_vars:
            return local_vars[token]
        if token in fixed_bindings:
            return repr(fixed_bindings[token])
        raise ValueError(f"Unbound variable in code emission: {token}")
    return repr(token)


def _parse_typed_variables(tokens_expr) -> list[tuple[str, str]]:
    raw_tokens = []
    for token in tokens_expr:
        if not isinstance(token, str):
            raise ValueError("Quantified variable declarations must be flat symbol lists.")
        raw_tokens.append(token)
    declarations = []
    pending_names = []
    index = 0
    while index < len(raw_tokens):
        token = raw_tokens[index]
        if token == "-":
            if not pending_names or index + 1 >= len(raw_tokens):
                raise ValueError("Malformed quantified variable list.")
            type_name = raw_tokens[index + 1]
            for name in pending_names:
                declarations.append((name, type_name))
            pending_names.clear()
            index += 2
            continue
        pending_names.append(token)
        index += 1
    for name in pending_names:
        declarations.append((name, "object"))
    return declarations


def _safe_var_name(raw_name: str) -> str:
    cleaned = raw_name.lstrip("?")
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", cleaned)
    if not cleaned:
        cleaned = "var"
    if cleaned[0].isdigit():
        cleaned = f"v_{cleaned}"
    return cleaned


def _unique_local_name(raw_name: str, local_vars: dict[str, str]) -> str:
    base_name = _safe_var_name(raw_name)
    candidate = base_name
    suffix = 1
    while candidate in local_vars.values():
        suffix += 1
        candidate = f"{base_name}_{suffix}"
    return candidate


def _make_class_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", " ", name).title().replace(" ", "")
    if not cleaned:
        cleaned = "GeneratedPOMDPModel"
    if cleaned[0].isdigit():
        cleaned = f"Model{cleaned}"
    if not cleaned.endswith("Model"):
        cleaned = f"{cleaned}Model"
    return cleaned


def _render_predicate(predicate: Predicate) -> str:
    return f"Predicate(name={predicate.name!r}, params={predicate.params!r})"


def _render_predicate_list(predicates: list[Predicate], indent: int) -> str:
    if not predicates:
        return "[]"
    prefix = " " * indent
    inner = " " * (indent + 4)
    lines = ["["]
    for predicate in predicates:
        lines.append(f"{inner}{_render_predicate(predicate)},")
    lines.append(f"{prefix}]")
    return "\n".join(lines)


def _render_belief_factor(factor, indent: int) -> str:
    prefix = " " * indent
    inner = " " * (indent + 4)
    scope_text = _render_predicate_list(factor.scope, indent + 4)
    case_lines = ["["]
    for probability, true_predicates in factor.cases:
        rendered_case = _render_predicate_list(true_predicates, indent + 12)
        case_lines.append(f"{inner}    ({probability!r}, {rendered_case}),")
    case_lines.append(f"{inner}]")
    cases_text = "\n".join(case_lines)
    return (
        "BeliefFactor(\n"
        f"{inner}name={factor.name!r},\n"
        f"{inner}scope={scope_text},\n"
        f"{inner}cases={cases_text},\n"
        f"{prefix})"
    )


def _render_factorized_belief(belief, indent: int) -> str:
    prefix = " " * indent
    inner = " " * (indent + 4)
    known_true_text = _render_predicate_list(belief.known_true, indent + 4)
    known_false_text = _render_predicate_list(belief.known_false, indent + 4)
    factor_lines = ["["]
    for factor in belief.factors:
        factor_text = _render_belief_factor(factor, indent + 8)
        factor_lines.append(f"{inner}    {factor_text},")
    factor_lines.append(f"{inner}]")
    factors_text = "\n".join(factor_lines)
    return (
        "FactorizedBelief(\n"
        f"{inner}known_true={known_true_text},\n"
        f"{inner}known_false={known_false_text},\n"
        f"{inner}factors={factors_text},\n"
        f"{prefix})"
    )
