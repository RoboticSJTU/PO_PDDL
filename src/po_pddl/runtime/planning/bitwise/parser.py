"""Parse/convert semantic Python POMDP models into bitwise model skeletons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import import_module
from typing import Any, Callable

from ..base.pomdp_model import POMDPModelBase as SemanticPOMDPModelBase
from ..data_structures import (
    Action,
    BeliefFactor as SemanticBeliefFactor,
    DefaultPolicyRule,
    FactorizedBelief as SemanticFactorizedBelief,
    ObservationEntry as SemanticObservationEntry,
    Observable,
    ObservationRule,
    ParsedEffectBucketAnnotation,
    Predicate,
    StateEntry as SemanticStateEntry,
)
from .indexed_belief_factor import IndexedBeliefFactor
from .indexed_factorized_belief import IndexedFactorizedBelief
from .indexed_particle_belief import IndexedParticleBelief
from ..parser import ParsedDefaultPolicy, ParsedDomain, ParsedProblem, parse_default_policy, parse_domain
from .pomdp_model import POMDPModelBase as BitwisePOMDPModelBase

REPORT_GOAL_ACTION_ID_SENTINEL = -1


@dataclass(frozen=True)
class BitwiseModelMetadata:
    """The minimal member-parameter view required by the bitwise model base."""

    grounded_predicates_count: int
    grounded_observables_count: int
    total_actions: int
    maximize_reward: bool
    goal_reward: float


@dataclass
class BitwiseModelSkeleton(BitwisePOMDPModelBase):
    """Concrete placeholder bitwise model with only member parameters populated.

    This class intentionally does not implement any real bitwise transition or
    observation semantics yet. It exists so the parser/converter can already
    produce a valid bitwise-model object while the remaining conversion stages
    are built incrementally.
    """

    goal_check: "BitwiseGoalCheck | None" = field(default=None, repr=False)
    action_precondition_checks: list["BitwiseGoalCheck"] = field(default_factory=list, repr=False)
    action_effect_condition_checks: list[list["BitwiseGoalCheck"]] = field(default_factory=list, repr=False)
    action_effect_distributions: list[list["BitwiseEffectBranch"]] = field(default_factory=list, repr=False)
    action_conditional_effect_distributions: list[list[list["BitwiseEffectBranch"]]] = field(default_factory=list, repr=False)
    observation_rule_condition_checks: list["BitwiseGoalCheck"] = field(default_factory=list, repr=False)
    observation_rule_distributions: list[list["BitwiseObservationBranch"]] = field(default_factory=list, repr=False)
    observation_rule_skip_before_for_active_perception: list[bool] = field(default_factory=list, repr=False)
    action_is_active_perception: list[bool] = field(default_factory=list, repr=False)
    default_policy_rule_condition_checks: list["BitwiseGoalCheck"] = field(default_factory=list, repr=False)
    default_policy_rule_action_ids: list[int] = field(default_factory=list, repr=False)

    def is_goal(self, state: int) -> bool:
        if self.goal_check is None:
            raise NotImplementedError("Bitwise goal conversion has not been attached to this model yet.")
        return evaluate_goal_check(self.goal_check, state)

    def _sample_branch_index(self, weights: list[float]) -> int | None:
        cleaned = [max(weight, 0.0) for weight in weights]
        total = sum(cleaned)
        if total <= 0.0:
            return None
        draw = self._rng.random() * total
        cumulative = 0.0
        for idx, weight in enumerate(cleaned):
            cumulative += weight
            if draw <= cumulative:
                return idx
        return len(cleaned) - 1

    def check_action_precondition(self, action: int, state: int) -> bool:
        if action < 0 or action >= len(self.action_precondition_checks):
            raise ValueError(f"Unknown grounded action id: {action}")
        return evaluate_goal_check(self.action_precondition_checks[action], state)

    def forward_action(self, action: int, state: int) -> tuple[int, float]:
        if action < 0 or action >= len(self.action_precondition_checks):
            raise ValueError(f"Unknown grounded action id: {action}")
        if not self.check_action_precondition(action, state):
            raise ValueError(f"Grounded action id `{action}` is not applicable in the given state.")

        self.clear_last_effect_bucket()
        next_state = state
        reward = 0.0

        unconditional_branches = self.action_effect_distributions[action]
        if unconditional_branches:
            branch = self._sample_effect_branch(
                action,
                unconditional_branches,
                record_sample=True,
            )
            next_state = self._apply_effect_branch(next_state, branch)
            reward += branch.reward_delta

        for guard_check, conditional_branches in zip(
            self.action_effect_condition_checks[action],
            self.action_conditional_effect_distributions[action],
        ):
            if not evaluate_goal_check(guard_check, state):
                continue
            if not conditional_branches:
                continue
            branch = self._sample_effect_branch(
                action,
                conditional_branches,
                record_sample=False,
            )
            next_state = self._apply_effect_branch(next_state, branch)
            reward += branch.reward_delta

        if self.is_goal(next_state):
            reward += self.goal_reward
        self._cache_transition_reward(action, state, next_state, reward)
        return next_state, reward

    def get_action_reward(self, action: int, state: int, next_state: int) -> float:
        cached_reward = self._lookup_cached_reward(action, state, next_state)
        if cached_reward is not None:
            return cached_reward

        reward = 0.0
        unconditional_branches = self.action_effect_distributions[action]
        if unconditional_branches:
            reward += self._sample_effect_branch(
                action,
                unconditional_branches,
                record_sample=False,
            ).reward_delta
        for guard_check, conditional_branches in zip(
            self.action_effect_condition_checks[action],
            self.action_conditional_effect_distributions[action],
        ):
            if not evaluate_goal_check(guard_check, state):
                continue
            if not conditional_branches:
                continue
            reward += self._sample_effect_branch(
                action,
                conditional_branches,
                record_sample=False,
            ).reward_delta
        if self.is_goal(next_state):
            reward += self.goal_reward
        return reward

    def is_active_perception_action(self, action: int | None) -> bool:
        if action is None:
            return False
        action_index = int(action)
        if action_index < 0 or action_index >= len(self.action_is_active_perception):
            return False
        return bool(self.action_is_active_perception[action_index])

    def should_skip_observation_rule_before_check(
        self,
        observation_rule: int,
        current_action: int | None = None,
    ) -> bool:
        rule_index = int(observation_rule)
        if rule_index < 0 or rule_index >= len(self.observation_rule_skip_before_for_active_perception):
            return False
        return bool(
            self.observation_rule_skip_before_for_active_perception[rule_index]
            and self.is_active_perception_action(current_action)
        )

    def check_observation_rule_condition(
        self,
        observation_rule: int,
        state: int,
        current_action: int | None = None,
    ) -> bool:
        if observation_rule < 0 or observation_rule >= len(self.observation_rule_condition_checks):
            raise ValueError(f"Unknown grounded observation rule id: {observation_rule}")
        if self.should_skip_observation_rule_before_check(observation_rule, current_action):
            return False
        return evaluate_goal_check(self.observation_rule_condition_checks[observation_rule], state)

    def observe_with_rule(
        self,
        observation_rule: int,
        state: int,
        current_action: int | None = None,
    ) -> tuple[int, int]:
        if observation_rule < 0 or observation_rule >= len(self.observation_rule_condition_checks):
            raise ValueError(f"Unknown grounded observation rule id: {observation_rule}")
        if not self.check_observation_rule_condition(
            observation_rule,
            state,
            current_action=current_action,
        ):
            return 0, 0
        branches = self.observation_rule_distributions[observation_rule]
        if not branches:
            return 0, 0
        weights = [max(branch.probability, 0.0) for branch in branches]
        branch_index = self._sample_branch_index(weights)
        if branch_index is None:
            return 0, 0
        branch = branches[branch_index]
        return branch.observation_bits, branch.observation_mask

    def check_default_policy_rule_condition(self, default_policy_rule: int, state: int) -> bool:
        if default_policy_rule < 0 or default_policy_rule >= len(self.default_policy_rule_condition_checks):
            raise ValueError(f"Unknown grounded default policy rule id: {default_policy_rule}")
        return evaluate_goal_check(self.default_policy_rule_condition_checks[default_policy_rule], state)

    def get_default_policy_rule_action(self, default_policy_rule: int, state: int) -> int:
        if default_policy_rule < 0 or default_policy_rule >= len(self.default_policy_rule_action_ids):
            raise ValueError(f"Unknown grounded default policy rule id: {default_policy_rule}")
        if not self.check_default_policy_rule_condition(default_policy_rule, state):
            raise ValueError(
                f"Grounded default policy rule id `{default_policy_rule}` is not applicable in the given state."
            )
        return self.default_policy_rule_action_ids[default_policy_rule]

    def _sample_effect_branch(
        self,
        action: int,
        branches: list["BitwiseEffectBranch"],
        *,
        record_sample: bool,
    ) -> "BitwiseEffectBranch":
        weights = [branch.probability for branch in branches]
        branch_index = self._select_effect_branch_index(
            action=action,
            weights=weights,
            bucket_names=[branch.bucket_name for branch in branches],
            successes=[branch.success for branch in branches],
            branch_indices=[branch.branch_index for branch in branches],
            variant_ranks=[branch.variant_rank for branch in branches],
            record_sample=record_sample,
        )
        if branch_index is None:
            return BitwiseEffectBranch(probability=1.0)
        return branches[branch_index]

    def _apply_effect_branch(self, state_bits: int, branch: "BitwiseEffectBranch") -> int:
        return (state_bits | branch.set_mask) & ~branch.clear_mask

    def _cache_transition_reward(
        self,
        action: int,
        state: int,
        next_state: int,
        reward: float,
    ) -> None:
        self._last_transition_reward_cache[(action, state, next_state)] = reward

    def _lookup_cached_reward(
        self,
        action: int,
        state: int,
        next_state: int,
    ) -> float | None:
        return self._last_transition_reward_cache.get((action, state, next_state))


@dataclass(frozen=True)
class BitwiseIndexLayout:
    """Index layout induced directly by list order in the semantic model.

    The order is intentionally taken from:
    - model.predicates
    - model.observables
    - model.actions
    - model.observation_rules
    - model.default_policy_rules
    """

    predicates: list[Predicate]
    observables: list[Observable]
    actions: list[Action]
    observation_rules: list[ObservationRule]
    default_policy_rules: list[DefaultPolicyRule]
    predicate_index: dict[Predicate, int]
    observable_index: dict[Observable, int]
    action_index: dict[Action, int]
    observation_rule_index: dict[ObservationRule, int]
    default_policy_rule_index: dict[DefaultPolicyRule, int]


@dataclass(frozen=True)
class BitwiseGoalCheck:
    """Bitwise goal check compiled to DNF mask clauses.

    Each clause is `(required_true_mask, required_false_mask)`.
    The goal is satisfied iff any clause is satisfied:

        (state & required_true_mask) == required_true_mask
        and
        (state & required_false_mask) == 0
    """

    clauses: list[tuple[int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class BitwiseEffectBranch:
    """One probabilistic effect branch represented only by bit masks."""

    probability: float
    set_mask: int = 0
    clear_mask: int = 0
    reward_delta: float = 0.0
    bucket_name: str | None = None
    success: bool | None = None
    branch_index: int | None = None
    variant_rank: int | None = None


@dataclass(frozen=True)
class BitwiseObservationBranch:
    """One probabilistic observation branch represented by observation bits and mask."""

    probability: float
    observation_bits: int = 0
    observation_mask: int = 0


def _substitute_symbol(symbol: str, bindings: dict[str, str]) -> str:
    if symbol.startswith("?") and symbol in bindings:
        return bindings[symbol]
    return symbol


def _substitute_expr(expr: Any, bindings: dict[str, str]) -> Any:
    if isinstance(expr, list):
        return [_substitute_expr(part, bindings) for part in expr]
    if isinstance(expr, str):
        return _substitute_symbol(expr, bindings)
    return expr


def _iter_typed_variables(typed_vars: Any) -> list[tuple[str, str]]:
    if not isinstance(typed_vars, list):
        return []

    result: list[tuple[str, str]] = []
    idx = 0
    while idx < len(typed_vars):
        item = typed_vars[idx]
        if isinstance(item, list):
            if len(item) == 3 and isinstance(item[0], str) and item[1] == "-" and isinstance(item[2], str):
                result.append((item[0], item[2]))
            elif len(item) == 2 and all(isinstance(part, str) for part in item):
                result.append((item[0], item[1]))
            idx += 1
            continue
        if not isinstance(item, str):
            idx += 1
            continue
        var_names = [item]
        idx += 1
        while idx < len(typed_vars) and isinstance(typed_vars[idx], str) and typed_vars[idx] != "-":
            var_names.append(typed_vars[idx])
            idx += 1
        if idx < len(typed_vars) and typed_vars[idx] == "-" and idx + 1 < len(typed_vars):
            type_name = typed_vars[idx + 1]
            if isinstance(type_name, str):
                for var_name in var_names:
                    result.append((var_name, type_name))
            idx += 2
            continue
        for var_name in var_names:
            result.append((var_name, "object"))
    return result


def _is_subtype_name(child_type: str, target_type: str, type_parents: dict[str, str | None]) -> bool:
    current = child_type
    while current is not None:
        if current == target_type:
            return True
        current = type_parents.get(current)
    return False


def _objects_of_type(
    type_name: str,
    *,
    constants: dict[str, str],
    objects: dict[str, str],
    type_parents: dict[str, str | None],
) -> list[str]:
    universe = dict(constants)
    universe.update(objects)
    if type_name == "object":
        return list(universe.keys())
    return [
        obj_name
        for obj_name, obj_type in universe.items()
        if _is_subtype_name(obj_type, type_name, type_parents)
    ]


def _atom_to_predicate(expr: list[Any]) -> Predicate:
    return Predicate(
        name=str(expr[0]),
        params=[str(param) for param in expr[1:]],
    )


def _merge_clause_pair(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int] | None:
    left_true, left_false = left
    right_true, right_false = right
    merged_true = left_true | right_true
    merged_false = left_false | right_false
    if (merged_true & merged_false) != 0:
        return None
    return merged_true, merged_false


def _normalize_dnf_clauses(clauses: list[tuple[int, int]]) -> list[tuple[int, int]]:
    unique: list[tuple[int, int]] = []
    for clause in clauses:
        if clause not in unique:
            unique.append(clause)

    simplified: list[tuple[int, int]] = []
    for idx, clause in enumerate(unique):
        clause_true, clause_false = clause
        absorbed = False
        for other_idx, other in enumerate(unique):
            if idx == other_idx:
                continue
            other_true, other_false = other
            true_subset = (other_true & ~clause_true) == 0
            false_subset = (other_false & ~clause_false) == 0
            if true_subset and false_subset:
                absorbed = True
                break
        if not absorbed:
            simplified.append(clause)
    return simplified


def _conjoin_clause_sets(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not left or not right:
        return []
    merged: list[tuple[int, int]] = []
    for left_clause in left:
        for right_clause in right:
            combined = _merge_clause_pair(left_clause, right_clause)
            if combined is not None:
                merged.append(combined)
    return _normalize_dnf_clauses(merged)


def _disjoin_clause_sets(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    return _normalize_dnf_clauses([*left, *right])


def _compile_goal_expr_to_dnf(
    expr: Any,
    *,
    layout: BitwiseIndexLayout,
    constants: dict[str, str],
    objects: dict[str, str],
    type_parents: dict[str, str | None],
    bindings: dict[str, str],
    negated: bool = False,
) -> list[tuple[int, int]]:
    expr = _substitute_expr(expr, bindings)
    if expr is None or expr == []:
        return [] if negated else [(0, 0)]
    if not isinstance(expr, list) or not expr:
        truth_value = bool(expr)
        truth_value = not truth_value if negated else truth_value
        return [(0, 0)] if truth_value else []

    operator = expr[0]
    if not isinstance(operator, str):
        raise TypeError(f"Unsupported goal operator type: {operator!r}")

    if operator == "not":
        child = expr[1] if len(expr) > 1 else []
        return _compile_goal_expr_to_dnf(
            child,
            layout=layout,
            constants=constants,
            objects=objects,
            type_parents=type_parents,
            bindings=bindings,
            negated=not negated,
        )

    if operator == "and":
        child_clause_sets = [
            _compile_goal_expr_to_dnf(
                child,
                layout=layout,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
                bindings=bindings,
                negated=negated,
            )
            for child in expr[1:]
        ]
        if negated:
            merged: list[tuple[int, int]] = []
            for clause_set in child_clause_sets:
                merged = _disjoin_clause_sets(merged, clause_set)
            return merged
        if not child_clause_sets:
            return [(0, 0)]
        merged = child_clause_sets[0]
        for clause_set in child_clause_sets[1:]:
            merged = _conjoin_clause_sets(merged, clause_set)
        return merged

    if operator == "or":
        child_clause_sets = [
            _compile_goal_expr_to_dnf(
                child,
                layout=layout,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
                bindings=bindings,
                negated=negated,
            )
            for child in expr[1:]
        ]
        if negated:
            if not child_clause_sets:
                return []
            merged = child_clause_sets[0]
            for clause_set in child_clause_sets[1:]:
                merged = _conjoin_clause_sets(merged, clause_set)
            return merged
        merged: list[tuple[int, int]] = []
        for clause_set in child_clause_sets:
            merged = _disjoin_clause_sets(merged, clause_set)
        return merged

    if operator == "imply":
        antecedent = expr[1] if len(expr) > 1 else []
        consequent = expr[2] if len(expr) > 2 else []
        rewritten = (
            ["and", antecedent, ["not", consequent]]
            if negated
            else ["or", ["not", antecedent], consequent]
        )
        return _compile_goal_expr_to_dnf(
            rewritten,
            layout=layout,
            constants=constants,
            objects=objects,
            type_parents=type_parents,
            bindings=bindings,
            negated=False,
        )

    if operator == "=":
        left = _substitute_symbol(str(expr[1]), bindings) if len(expr) > 1 else ""
        right = _substitute_symbol(str(expr[2]), bindings) if len(expr) > 2 else ""
        is_equal = left == right
        is_equal = not is_equal if negated else is_equal
        return [(0, 0)] if is_equal else []

    if operator in {"forall", "all", "exists"}:
        typed_vars = expr[1] if len(expr) > 1 else []
        body = expr[2] if len(expr) > 2 else []
        var_specs = _iter_typed_variables(typed_vars)
        if not var_specs:
            return _compile_goal_expr_to_dnf(
                body,
                layout=layout,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
                bindings=bindings,
                negated=negated,
            )

        def expand(idx: int, current_bindings: dict[str, str]) -> list[list[tuple[int, int]]]:
            if idx >= len(var_specs):
                return [
                    _compile_goal_expr_to_dnf(
                        body,
                        layout=layout,
                        constants=constants,
                        objects=objects,
                        type_parents=type_parents,
                        bindings=current_bindings,
                        negated=negated,
                    )
                ]
            var_name, type_name = var_specs[idx]
            expanded: list[list[tuple[int, int]]] = []
            for obj_name in _objects_of_type(
                type_name,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
            ):
                next_bindings = dict(current_bindings)
                next_bindings[var_name] = obj_name
                expanded.extend(expand(idx + 1, next_bindings))
            return expanded

        expanded_clause_sets = expand(0, dict(bindings))
        quantifier_is_and = (operator in {"forall", "all"} and not negated) or (
            operator == "exists" and negated
        )
        if quantifier_is_and:
            if not expanded_clause_sets:
                return [(0, 0)]
            merged = expanded_clause_sets[0]
            for clause_set in expanded_clause_sets[1:]:
                merged = _conjoin_clause_sets(merged, clause_set)
            return merged
        merged: list[tuple[int, int]] = []
        for clause_set in expanded_clause_sets:
            merged = _disjoin_clause_sets(merged, clause_set)
        return merged

    predicate = _atom_to_predicate(expr)
    if predicate not in layout.predicate_index:
        raise KeyError(f"Goal predicate `{predicate}` was not found in grounded predicates.")
    predicate_mask = 1 << layout.predicate_index[predicate]
    return [(0, predicate_mask)] if negated else [(predicate_mask, 0)]


def compile_goal_check(
    goal_expr: Any,
    layout: BitwiseIndexLayout,
    *,
    constants: dict[str, str] | None = None,
    objects: dict[str, str] | None = None,
    type_parents: dict[str, str | None] | None = None,
) -> BitwiseGoalCheck:
    """Compile one semantic goal expression into DNF bit-mask clauses."""

    return BitwiseGoalCheck(
        clauses=_normalize_dnf_clauses(
            _compile_goal_expr_to_dnf(
                goal_expr,
                layout=layout,
                constants=dict(constants or {}),
                objects=dict(objects or {}),
                type_parents=dict(type_parents or {"object": None}),
                bindings={},
                negated=False,
            )
        )
    )


def _goal_clause_holds(clause: tuple[int, int], state_bits: int) -> bool:
    required_true_mask, required_false_mask = clause
    required_true_ok = (state_bits & required_true_mask) == required_true_mask
    required_false_ok = (state_bits & required_false_mask) == 0
    return required_true_ok and required_false_ok


def evaluate_goal_check(goal_check: BitwiseGoalCheck, state_bits: int) -> bool:
    """Evaluate one compiled DNF bitwise goal-check on a state bit vector."""

    return any(_goal_clause_holds(clause, state_bits) for clause in goal_check.clauses)


def _merge_effect_branch_pair(
    left: BitwiseEffectBranch,
    right: BitwiseEffectBranch,
) -> BitwiseEffectBranch | None:
    # Apply `left` first, then let `right` override overlapping bits.
    combined_set = (left.set_mask & ~right.clear_mask) | right.set_mask
    combined_clear = (left.clear_mask & ~right.set_mask) | right.clear_mask
    if (combined_set & combined_clear) != 0:
        return None
    return BitwiseEffectBranch(
        probability=left.probability * right.probability,
        set_mask=combined_set,
        clear_mask=combined_clear,
        reward_delta=left.reward_delta + right.reward_delta,
        bucket_name=right.bucket_name if right.bucket_name is not None else left.bucket_name,
        success=right.success if right.success is not None else left.success,
        branch_index=right.branch_index if right.branch_index is not None else left.branch_index,
        variant_rank=right.variant_rank if right.variant_rank is not None else left.variant_rank,
    )


def _merge_observation_branch_pair(
    left: BitwiseObservationBranch,
    right: BitwiseObservationBranch,
) -> BitwiseObservationBranch | None:
    overlapping_mask = left.observation_mask & right.observation_mask
    conflicting_bits = (left.observation_bits ^ right.observation_bits) & overlapping_mask
    if conflicting_bits != 0:
        return None
    return BitwiseObservationBranch(
        probability=left.probability * right.probability,
        observation_bits=left.observation_bits | right.observation_bits,
        observation_mask=left.observation_mask | right.observation_mask,
    )


def _normalize_effect_branches(branches: list[BitwiseEffectBranch]) -> list[BitwiseEffectBranch]:
    merged: dict[
        tuple[int, int, float, str | None, bool | None, int | None, int | None],
        float,
    ] = {}
    for branch in branches:
        key = (
            branch.set_mask,
            branch.clear_mask,
            branch.reward_delta,
            branch.bucket_name,
            branch.success,
            branch.branch_index,
            branch.variant_rank,
        )
        merged[key] = merged.get(key, 0.0) + branch.probability
    return [
        BitwiseEffectBranch(
            probability=probability,
            set_mask=set_mask,
            clear_mask=clear_mask,
            reward_delta=reward_delta,
            bucket_name=bucket_name,
            success=success,
            branch_index=branch_index,
            variant_rank=variant_rank,
        )
        for (
            set_mask,
            clear_mask,
            reward_delta,
            bucket_name,
            success,
            branch_index,
            variant_rank,
        ), probability in merged.items()
        if probability > 0.0
    ]


def _normalize_observation_branches(
    branches: list[BitwiseObservationBranch],
) -> list[BitwiseObservationBranch]:
    merged: dict[tuple[int, int], float] = {}
    for branch in branches:
        key = (branch.observation_bits, branch.observation_mask)
        merged[key] = merged.get(key, 0.0) + branch.probability
    return [
        BitwiseObservationBranch(
            probability=probability,
            observation_bits=observation_bits,
            observation_mask=observation_mask,
        )
        for (observation_bits, observation_mask), probability in merged.items()
        if probability > 0.0
    ]


def extract_bitwise_metadata(model: SemanticPOMDPModelBase) -> BitwiseModelMetadata:
    """Extract the bitwise-base member parameters from a semantic POMDP model."""

    return BitwiseModelMetadata(
        grounded_predicates_count=len(model.predicates),
        grounded_observables_count=len(model.observables),
        total_actions=len(model.actions),
        maximize_reward=model.maximize_reward,
        goal_reward=model.goal_reward,
    )


def build_bitwise_model_skeleton(model: SemanticPOMDPModelBase) -> BitwiseModelSkeleton:
    """Build a concrete bitwise-model placeholder from an existing semantic model."""

    metadata = extract_bitwise_metadata(model)
    return BitwiseModelSkeleton(**asdict(metadata))


def convert_pomdp_model_to_bitwise_model(
    model: SemanticPOMDPModelBase,
) -> BitwiseModelSkeleton:
    """Convert one semantic POMDP model directly to a bitwise model object.

    If the semantic model comes from a generated explicit package, this returns a
    fully compiled bitwise model with goal/action/observation/default-policy
    structures attached. Otherwise it falls back to the metadata-only skeleton.
    """

    module_name = type(model).__module__
    bitwise_model: BitwiseModelSkeleton
    if module_name.endswith(".model"):
        package_module_name = module_name.rsplit(".model", 1)[0]
        try:
            bitwise_model = build_bitwise_model_skeleton_from_module(package_module_name)
        except Exception:  # noqa: BLE001
            bitwise_model = build_bitwise_model_skeleton(model)
    else:
        bitwise_model = build_bitwise_model_skeleton(model)

    setattr(bitwise_model, "_source_semantic_model", model)
    return bitwise_model


def build_bitwise_model_from_parsed(
    model: SemanticPOMDPModelBase,
    *,
    parsed_domain: ParsedDomain,
    parsed_problem: ParsedProblem,
    parsed_default_policy: ParsedDefaultPolicy | None = None,
) -> BitwiseModelSkeleton:
    """Compile one fully-populated bitwise model directly from parsed POMDPDDL data."""

    metadata = extract_bitwise_metadata(model)
    layout = build_bitwise_index_layout(model)
    type_parents = {
        name: (node.parent.name if node.parent is not None else None)
        for name, node in parsed_domain.types.items()
    }
    default_policy = parsed_default_policy or ParsedDefaultPolicy(policy_name="default-policy")

    goal_check = compile_goal_check(
        parsed_problem.goal,
        layout,
        constants=dict(parsed_domain.constants),
        objects=dict(parsed_problem.objects),
        type_parents=type_parents,
    )
    action_precondition_checks = compile_action_precondition_checks(
        actions=list(model.actions),
        parsed_domain=parsed_domain,
        layout=layout,
        constants=dict(parsed_domain.constants),
        objects=dict(parsed_problem.objects),
        type_parents=type_parents,
    )
    action_effect_condition_checks = compile_action_effect_condition_checks(
        actions=list(model.actions),
        parsed_domain=parsed_domain,
        layout=layout,
        constants=dict(parsed_domain.constants),
        objects=dict(parsed_problem.objects),
        type_parents=type_parents,
    )
    action_effect_distributions, action_conditional_effect_distributions = compile_action_effect_distributions(
        actions=list(model.actions),
        parsed_domain=parsed_domain,
        layout=layout,
        constants=dict(parsed_domain.constants),
        objects=dict(parsed_problem.objects),
        type_parents=type_parents,
    )
    observation_rule_condition_checks = compile_observation_rule_condition_checks(
        observation_rules=list(model.observation_rules),
        parsed_domain=parsed_domain,
        layout=layout,
        constants=dict(parsed_domain.constants),
        objects=dict(parsed_problem.objects),
        type_parents=type_parents,
    )
    observation_rule_distributions = compile_observation_rule_distributions(
        observation_rules=list(model.observation_rules),
        parsed_domain=parsed_domain,
        layout=layout,
        constants=dict(parsed_domain.constants),
        objects=dict(parsed_problem.objects),
        type_parents=type_parents,
    )
    default_policy_rule_condition_checks = compile_default_policy_rule_condition_checks(
        default_policy_rules=list(model.default_policy_rules),
        parsed_default_policy=default_policy,
        layout=layout,
        constants=dict(parsed_domain.constants),
        objects=dict(parsed_problem.objects),
        type_parents=type_parents,
    )
    default_policy_rule_action_ids = compile_default_policy_rule_action_ids(
        default_policy_rules=list(model.default_policy_rules),
        parsed_default_policy=default_policy,
        layout=layout,
    )
    bitwise_model = BitwiseModelSkeleton(
        **asdict(metadata),
        goal_check=goal_check,
        action_precondition_checks=action_precondition_checks,
        action_effect_condition_checks=action_effect_condition_checks,
        action_effect_distributions=action_effect_distributions,
        action_conditional_effect_distributions=action_conditional_effect_distributions,
        observation_rule_condition_checks=observation_rule_condition_checks,
        observation_rule_distributions=observation_rule_distributions,
        observation_rule_skip_before_for_active_perception=[
            observation_rule.name.startswith("before_")
            for observation_rule in model.observation_rules
        ],
        action_is_active_perception=[
            action.name.startswith("active_obs_")
            for action in model.actions
        ],
        default_policy_rule_condition_checks=default_policy_rule_condition_checks,
        default_policy_rule_action_ids=default_policy_rule_action_ids,
    )
    setattr(bitwise_model, "_source_semantic_model", model)
    return bitwise_model


def build_bitwise_index_layout(model: SemanticPOMDPModelBase) -> BitwiseIndexLayout:
    """Build index tables using the current order of semantic-model lists."""

    predicates = list(model.predicates)
    observables = list(model.observables)
    actions = list(model.actions)
    observation_rules = list(model.observation_rules)
    default_policy_rules = list(model.default_policy_rules)

    return BitwiseIndexLayout(
        predicates=predicates,
        observables=observables,
        actions=actions,
        observation_rules=observation_rules,
        default_policy_rules=default_policy_rules,
        predicate_index={predicate: idx for idx, predicate in enumerate(predicates)},
        observable_index={observable: idx for idx, observable in enumerate(observables)},
        action_index={action: idx for idx, action in enumerate(actions)},
        observation_rule_index={
            observation_rule: idx for idx, observation_rule in enumerate(observation_rules)
        },
        default_policy_rule_index={
            default_policy_rule: idx
            for idx, default_policy_rule in enumerate(default_policy_rules)
        },
    )


def _convert_belief_factor_to_indexed(
    factor: SemanticBeliefFactor,
    layout: BitwiseIndexLayout,
) -> IndexedBeliefFactor:
    return IndexedBeliefFactor(
        scope=[layout.predicate_index[predicate] for predicate in factor.scope],
        cases=[
            (
                probability,
                [layout.predicate_index[predicate] for predicate in true_predicates],
            )
            for probability, true_predicates in factor.cases
        ],
    )


def convert_factorized_belief_to_indexed(
    belief: SemanticFactorizedBelief,
    layout: BitwiseIndexLayout,
) -> IndexedFactorizedBelief:
    """Replace semantic predicates in a factorized belief with grounded predicate indices."""
    indexed = IndexedFactorizedBelief(
        grounded_predicates_count=len(layout.predicates),
        known_true=[layout.predicate_index[predicate] for predicate in belief.known_true],
        known_false=[layout.predicate_index[predicate] for predicate in belief.known_false],
        factors=[_convert_belief_factor_to_indexed(factor, layout) for factor in belief.factors],
    )
    indexed.validate()
    return indexed


def convert_indexed_factorized_belief_to_particles(
    belief: IndexedFactorizedBelief,
) -> IndexedParticleBelief:
    """Expand one indexed factorized belief into its exact weighted support states."""
    belief.validate()
    base_bits = 0
    for index in belief.known_true:
        base_bits |= 1 << index

    state_weights: dict[int, float] = {base_bits: 1.0}

    def _indexed_case_bits(factor_scope: list[int], true_indices: list[int]) -> int:
        bits = 0
        true_set = set(true_indices)
        for index in factor_scope:
            if index in true_set:
                bits |= 1 << index
        return bits

    for factor in belief.factors:
        next_state_weights: dict[int, float] = {}
        for state_bits, probability in state_weights.items():
            if probability <= 0.0:
                continue
            for case_probability, true_indices in factor.cases:
                next_probability = probability * case_probability
                if next_probability <= 0.0:
                    continue
                next_bits = state_bits | _indexed_case_bits(factor.scope, true_indices)
                next_state_weights[next_bits] = (
                    next_state_weights.get(next_bits, 0.0) + next_probability
                )
        state_weights = next_state_weights

    particles = [
        (state_bits, probability)
        for state_bits, probability in sorted(state_weights.items(), key=lambda item: item[0])
        if probability > 0.0
    ]
    indexed_particles = IndexedParticleBelief(
        grounded_predicates_count=belief.grounded_predicates_count,
        particles=particles,
    )
    indexed_particles.validate()
    return indexed_particles


def convert_factorized_belief_to_indexed_particles(
    belief: SemanticFactorizedBelief,
    layout: BitwiseIndexLayout,
) -> IndexedParticleBelief:
    """Convert a semantic factorized belief into an indexed particle belief."""
    return convert_indexed_factorized_belief_to_particles(
        convert_factorized_belief_to_indexed(belief, layout)
    )


def state_to_bitvector(
    state: SemanticStateEntry,
    layout: BitwiseIndexLayout,
) -> int:
    """Convert a semantic state map into one predicate bit vector."""

    bits = 0
    for predicate, value in state.items():
        if value and predicate in layout.predicate_index:
            bits |= 1 << layout.predicate_index[predicate]
    return bits


def observation_to_bitvector(
    observation: SemanticObservationEntry,
    layout: BitwiseIndexLayout,
) -> int:
    """Convert a semantic observation map into one observable bit vector."""

    bits = 0
    for observable, value in observation.items():
        if value and observable in layout.observable_index:
            bits |= 1 << layout.observable_index[observable]
    return bits


def action_to_index(action: Action, layout: BitwiseIndexLayout) -> int:
    """Convert one grounded action object to its integer id."""

    return layout.action_index[action]


def observation_rule_to_index(observation_rule: ObservationRule, layout: BitwiseIndexLayout) -> int:
    """Convert one grounded observation rule to its integer id."""

    return layout.observation_rule_index[observation_rule]


def default_policy_rule_to_index(
    default_policy_rule: DefaultPolicyRule,
    layout: BitwiseIndexLayout,
) -> int:
    """Convert one grounded default-policy rule to its integer id."""

    return layout.default_policy_rule_index[default_policy_rule]


def _action_bindings_from_instance(
    action: Action,
    parameter_types: list[tuple[str, str]],
) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for idx, (param_name, _type_name) in enumerate(parameter_types):
        if idx >= len(action.params):
            break
        bindings[param_name] = action.params[idx]
    return bindings


def _enumerate_typed_bindings(
    parameter_types: list[tuple[str, str]],
    *,
    constants: dict[str, str],
    objects: dict[str, str],
    type_parents: dict[str, str | None],
) -> list[dict[str, str]]:
    if not parameter_types:
        return [{}]

    bindings: list[dict[str, str]] = [{}]
    for param_name, type_name in parameter_types:
        next_bindings: list[dict[str, str]] = []
        for partial_binding in bindings:
            for obj_name in _objects_of_type(
                type_name,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
            ):
                updated = dict(partial_binding)
                updated[param_name] = obj_name
                next_bindings.append(updated)
        bindings = next_bindings
    return bindings


def _ground_observable_with_bindings(
    observable: Observable,
    bindings: dict[str, str],
) -> Observable:
    return Observable(
        name=observable.name,
        params=[_substitute_symbol(param, bindings) for param in observable.params],
    )


def _ground_observation_rule_with_bindings(
    rule_schema: Any,
    bindings: dict[str, str],
) -> ObservationRule:
    grounded_distribution = [
        _ground_observable_with_bindings(observable, bindings)
        for observable in rule_schema.rule.distribution
    ]
    grounded_params = [
        bindings[param_name]
        for param_name, _type_name in rule_schema.parameter_types
        if param_name in bindings
    ]
    if grounded_params:
        grounded_name = f"{rule_schema.rule.name}[{','.join(grounded_params)}]"
    else:
        grounded_name = rule_schema.rule.name
    return ObservationRule(name=grounded_name, distribution=grounded_distribution)


def compile_action_precondition_checks(
    *,
    actions: list[Action],
    parsed_domain: Any,
    layout: BitwiseIndexLayout,
    constants: dict[str, str],
    objects: dict[str, str],
    type_parents: dict[str, str | None],
) -> list[BitwiseGoalCheck]:
    """Compile one DNF mask precondition for each grounded action, by action list order."""

    schema_by_signature: dict[tuple[str, int], list[Any]] = {}
    for action_schema in parsed_domain.actions:
        signature = (action_schema.action.name, len(action_schema.action.params))
        schema_by_signature.setdefault(signature, []).append(action_schema)

    compiled: list[BitwiseGoalCheck] = []
    for action in actions:
        signature = (action.name, len(action.params))
        candidate_schemas = schema_by_signature.get(signature, [])
        if not candidate_schemas:
            raise KeyError(f"No action schema found for grounded action `{action}`.")
        action_schema = candidate_schemas[0]
        bindings = _action_bindings_from_instance(action, action_schema.parameter_types)
        precondition_expr = _substitute_expr(action_schema.precondition, bindings)
        compiled.append(
            compile_goal_check(
                precondition_expr,
                layout,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
            )
        )
    return compiled


def _collect_effect_condition_exprs(
    effect_expr: Any,
    *,
    constants: dict[str, str],
    objects: dict[str, str],
    type_parents: dict[str, str | None],
    bindings: dict[str, str],
) -> list[Any]:
    effect_expr = _substitute_expr(effect_expr, bindings)
    if effect_expr is None or not isinstance(effect_expr, list) or not effect_expr:
        return []

    operator = effect_expr[0]
    if not isinstance(operator, str):
        return []

    if operator == "and":
        collected: list[Any] = []
        for subexpr in effect_expr[1:]:
            collected.extend(
                _collect_effect_condition_exprs(
                    subexpr,
                    constants=constants,
                    objects=objects,
                    type_parents=type_parents,
                    bindings=bindings,
                )
            )
        return collected

    if operator == "probabilistic":
        collected: list[Any] = []
        index = 1
        while index + 1 < len(effect_expr):
            branch_effect = effect_expr[index + 1]
            collected.extend(
                _collect_effect_condition_exprs(
                    branch_effect,
                    constants=constants,
                    objects=objects,
                    type_parents=type_parents,
                    bindings=bindings,
                )
            )
            index += 2
        return collected

    if operator == "forall":
        typed_vars = effect_expr[1] if len(effect_expr) > 1 else []
        body = effect_expr[2] if len(effect_expr) > 2 else []
        var_specs = _iter_typed_variables(typed_vars)
        if not var_specs:
            return _collect_effect_condition_exprs(
                body,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
                bindings=bindings,
            )
        collected: list[Any] = []
        for quantified_bindings in _enumerate_typed_bindings(
            var_specs,
            constants=constants,
            objects=objects,
            type_parents=type_parents,
        ):
            merged_bindings = dict(bindings)
            merged_bindings.update(quantified_bindings)
            collected.extend(
                _collect_effect_condition_exprs(
                    body,
                    constants=constants,
                    objects=objects,
                    type_parents=type_parents,
                    bindings=merged_bindings,
                )
            )
        return collected

    if operator == "when":
        condition_expr = effect_expr[1] if len(effect_expr) > 1 else []
        nested_effect = effect_expr[2] if len(effect_expr) > 2 else []
        collected = [_substitute_expr(condition_expr, bindings)]
        collected.extend(
            _collect_effect_condition_exprs(
                nested_effect,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
                bindings=bindings,
            )
        )
        return collected

    return []


def _compile_effect_expr_to_branches(
    effect_expr: Any,
    *,
    layout: BitwiseIndexLayout,
    constants: dict[str, str],
    objects: dict[str, str],
    type_parents: dict[str, str | None],
    bindings: dict[str, str],
    ignore_when: bool,
    effect_bucket_annotations: list[ParsedEffectBucketAnnotation] | None = None,
    probabilistic_depth: int = 0,
) -> list[BitwiseEffectBranch]:
    effect_expr = _substitute_expr(effect_expr, bindings)
    if effect_expr is None or effect_expr == []:
        return [BitwiseEffectBranch(probability=1.0)]
    if not isinstance(effect_expr, list) or not effect_expr:
        return [BitwiseEffectBranch(probability=1.0)]

    operator = effect_expr[0]
    if not isinstance(operator, str):
        return [BitwiseEffectBranch(probability=1.0)]

    if operator == "and":
        child_branches = [
            _compile_effect_expr_to_branches(
                subexpr,
                layout=layout,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
                bindings=bindings,
                ignore_when=ignore_when,
                effect_bucket_annotations=effect_bucket_annotations,
                probabilistic_depth=probabilistic_depth,
            )
            for subexpr in effect_expr[1:]
        ]
        if not child_branches:
            return [BitwiseEffectBranch(probability=1.0)]
        merged = child_branches[0]
        for branch_group in child_branches[1:]:
            next_merged: list[BitwiseEffectBranch] = []
            for left_branch in merged:
                for right_branch in branch_group:
                    combined = _merge_effect_branch_pair(left_branch, right_branch)
                    if combined is not None:
                        next_merged.append(combined)
            merged = _normalize_effect_branches(next_merged)
        return merged

    if operator == "probabilistic":
        branches: list[BitwiseEffectBranch] = []
        index = 1
        branch_slot = 0
        while index + 1 < len(effect_expr):
            probability = float(effect_expr[index])
            branch_effect = effect_expr[index + 1]
            for branch in _compile_effect_expr_to_branches(
                branch_effect,
                layout=layout,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
                bindings=bindings,
                ignore_when=ignore_when,
                effect_bucket_annotations=None,
                probabilistic_depth=probabilistic_depth + 1,
            ):
                annotation = (
                    effect_bucket_annotations[branch_slot]
                    if effect_bucket_annotations is not None
                    and probabilistic_depth == 0
                    and branch_slot < len(effect_bucket_annotations)
                    else None
                )
                branches.append(
                    BitwiseEffectBranch(
                        probability=probability * branch.probability,
                        set_mask=branch.set_mask,
                        clear_mask=branch.clear_mask,
                        reward_delta=branch.reward_delta,
                        bucket_name=annotation.bucket_name if annotation is not None else branch.bucket_name,
                        success=annotation.success if annotation is not None else branch.success,
                        branch_index=branch_slot if probabilistic_depth == 0 else branch.branch_index,
                        variant_rank=(
                            annotation.variant_rank
                            if annotation is not None
                            else branch.variant_rank
                        ),
                    )
                )
            branch_slot += 1
            index += 2
        return _normalize_effect_branches(branches)

    if operator == "forall":
        typed_vars = effect_expr[1] if len(effect_expr) > 1 else []
        body = effect_expr[2] if len(effect_expr) > 2 else []
        var_specs = _iter_typed_variables(typed_vars)
        if not var_specs:
            return _compile_effect_expr_to_branches(
                body,
                layout=layout,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
                bindings=bindings,
                ignore_when=ignore_when,
                effect_bucket_annotations=effect_bucket_annotations,
                probabilistic_depth=probabilistic_depth,
            )
        expanded_groups = [
            _compile_effect_expr_to_branches(
                body,
                layout=layout,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
                bindings={**bindings, **quantified_bindings},
                ignore_when=ignore_when,
                effect_bucket_annotations=effect_bucket_annotations,
                probabilistic_depth=probabilistic_depth,
            )
            for quantified_bindings in _enumerate_typed_bindings(
                var_specs,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
            )
        ]
        if not expanded_groups:
            return [BitwiseEffectBranch(probability=1.0)]
        merged = expanded_groups[0]
        for branch_group in expanded_groups[1:]:
            next_merged: list[BitwiseEffectBranch] = []
            for left_branch in merged:
                for right_branch in branch_group:
                    combined = _merge_effect_branch_pair(left_branch, right_branch)
                    if combined is not None:
                        next_merged.append(combined)
            merged = _normalize_effect_branches(next_merged)
        return merged

    if operator == "when":
        if ignore_when:
            return [BitwiseEffectBranch(probability=1.0)]
        nested_effect = effect_expr[2] if len(effect_expr) > 2 else []
        return _compile_effect_expr_to_branches(
            nested_effect,
            layout=layout,
            constants=constants,
            objects=objects,
            type_parents=type_parents,
            bindings=bindings,
            ignore_when=ignore_when,
            effect_bucket_annotations=effect_bucket_annotations,
            probabilistic_depth=probabilistic_depth,
        )

    if operator in {"increase", "decrease", "assign"}:
        target = effect_expr[1] if len(effect_expr) > 1 else []
        value_expr = effect_expr[2] if len(effect_expr) > 2 else 0

        target_name: str | None = None
        if isinstance(target, list) and target:
            head = target[0]
            if isinstance(head, str):
                target_name = head
        elif isinstance(target, str):
            target_name = target

        numeric_value = 0.0
        if isinstance(value_expr, (int, float)):
            numeric_value = float(value_expr)
        elif isinstance(value_expr, str):
            try:
                numeric_value = float(value_expr)
            except ValueError:
                numeric_value = 0.0

        reward_delta = 0.0
        if target_name is not None and "reward" in target_name:
            if operator == "increase":
                reward_delta = numeric_value
            elif operator == "decrease":
                reward_delta = -numeric_value
            else:
                reward_delta = numeric_value
        return [BitwiseEffectBranch(probability=1.0, reward_delta=reward_delta)]

    if operator == "not":
        atom = effect_expr[1] if len(effect_expr) > 1 else []
        if isinstance(atom, list) and atom:
            predicate = _atom_to_predicate(atom)
            if predicate not in layout.predicate_index:
                raise KeyError(f"Effect predicate `{predicate}` was not found in grounded predicates.")
            bit = 1 << layout.predicate_index[predicate]
            return [BitwiseEffectBranch(probability=1.0, clear_mask=bit)]
        return [BitwiseEffectBranch(probability=1.0)]

    predicate = _atom_to_predicate(effect_expr)
    if predicate not in layout.predicate_index:
        raise KeyError(f"Effect predicate `{predicate}` was not found in grounded predicates.")
    bit = 1 << layout.predicate_index[predicate]
    return [BitwiseEffectBranch(probability=1.0, set_mask=bit)]


def _compile_observation_distribution_to_branches(
    distribution_expr: Any,
    *,
    layout: BitwiseIndexLayout,
    constants: dict[str, str],
    objects: dict[str, str],
    type_parents: dict[str, str | None],
    bindings: dict[str, str],
) -> list[BitwiseObservationBranch]:
    distribution_expr = _substitute_expr(distribution_expr, bindings)
    if distribution_expr is None or distribution_expr == []:
        return [BitwiseObservationBranch(probability=1.0)]
    if not isinstance(distribution_expr, list) or not distribution_expr:
        return [BitwiseObservationBranch(probability=1.0)]

    operator = distribution_expr[0]
    if not isinstance(operator, str):
        return [BitwiseObservationBranch(probability=1.0)]

    if operator == "and":
        child_branches = [
            _compile_observation_distribution_to_branches(
                subexpr,
                layout=layout,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
                bindings=bindings,
            )
            for subexpr in distribution_expr[1:]
        ]
        if not child_branches:
            return [BitwiseObservationBranch(probability=1.0)]
        merged = child_branches[0]
        for branch_group in child_branches[1:]:
            next_merged: list[BitwiseObservationBranch] = []
            for left_branch in merged:
                for right_branch in branch_group:
                    combined = _merge_observation_branch_pair(left_branch, right_branch)
                    if combined is not None:
                        next_merged.append(combined)
            merged = _normalize_observation_branches(next_merged)
        return merged

    if operator == "probabilistic":
        branches: list[BitwiseObservationBranch] = []
        index = 1
        while index + 1 < len(distribution_expr):
            probability = float(distribution_expr[index])
            branch_expr = distribution_expr[index + 1]
            for branch in _compile_observation_distribution_to_branches(
                branch_expr,
                layout=layout,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
                bindings=bindings,
            ):
                branches.append(
                    BitwiseObservationBranch(
                        probability=probability * branch.probability,
                        observation_bits=branch.observation_bits,
                        observation_mask=branch.observation_mask,
                    )
                )
            index += 2
        return _normalize_observation_branches(branches)

    if operator == "forall":
        typed_vars = distribution_expr[1] if len(distribution_expr) > 1 else []
        body = distribution_expr[2] if len(distribution_expr) > 2 else []
        var_specs = _iter_typed_variables(typed_vars)
        if not var_specs:
            return _compile_observation_distribution_to_branches(
                body,
                layout=layout,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
                bindings=bindings,
            )
        expanded_groups = [
            _compile_observation_distribution_to_branches(
                body,
                layout=layout,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
                bindings={**bindings, **quantified_bindings},
            )
            for quantified_bindings in _enumerate_typed_bindings(
                var_specs,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
            )
        ]
        if not expanded_groups:
            return [BitwiseObservationBranch(probability=1.0)]
        merged = expanded_groups[0]
        for branch_group in expanded_groups[1:]:
            next_merged: list[BitwiseObservationBranch] = []
            for left_branch in merged:
                for right_branch in branch_group:
                    combined = _merge_observation_branch_pair(left_branch, right_branch)
                    if combined is not None:
                        next_merged.append(combined)
            merged = _normalize_observation_branches(next_merged)
        return merged

    if operator == "not":
        atom = distribution_expr[1] if len(distribution_expr) > 1 else []
        if isinstance(atom, list) and atom:
            observable = Observable(
                name=str(atom[0]),
                params=[str(_substitute_symbol(str(param), bindings)) for param in atom[1:]],
            )
            if observable not in layout.observable_index:
                raise KeyError(f"Observation `{observable}` was not found in grounded observables.")
            bit = 1 << layout.observable_index[observable]
            return [BitwiseObservationBranch(probability=1.0, observation_mask=bit)]
        return [BitwiseObservationBranch(probability=1.0)]

    observable = Observable(
        name=operator,
        params=[str(_substitute_symbol(str(param), bindings)) for param in distribution_expr[1:]],
    )
    if observable not in layout.observable_index:
        raise KeyError(f"Observation `{observable}` was not found in grounded observables.")
    bit = 1 << layout.observable_index[observable]
    return [BitwiseObservationBranch(probability=1.0, observation_bits=bit, observation_mask=bit)]


def compile_action_effect_condition_checks(
    *,
    actions: list[Action],
    parsed_domain: Any,
    layout: BitwiseIndexLayout,
    constants: dict[str, str],
    objects: dict[str, str],
    type_parents: dict[str, str | None],
) -> list[list[BitwiseGoalCheck]]:
    """Compile `when` guard conditions from action effects for each grounded action."""

    schema_by_signature: dict[tuple[str, int], list[Any]] = {}
    for action_schema in parsed_domain.actions:
        signature = (action_schema.action.name, len(action_schema.action.params))
        schema_by_signature.setdefault(signature, []).append(action_schema)

    compiled: list[list[BitwiseGoalCheck]] = []
    for action in actions:
        signature = (action.name, len(action.params))
        candidate_schemas = schema_by_signature.get(signature, [])
        if not candidate_schemas:
            raise KeyError(f"No action schema found for grounded action `{action}`.")
        action_schema = candidate_schemas[0]
        bindings = _action_bindings_from_instance(action, action_schema.parameter_types)
        condition_exprs = _collect_effect_condition_exprs(
            action_schema.effect,
            constants=constants,
            objects=objects,
            type_parents=type_parents,
            bindings=bindings,
        )
        compiled.append(
            [
                compile_goal_check(
                    condition_expr,
                    layout,
                    constants=constants,
                    objects=objects,
                    type_parents=type_parents,
                )
                for condition_expr in condition_exprs
            ]
        )
    return compiled


def _collect_effect_condition_payload_exprs(
    effect_expr: Any,
    *,
    constants: dict[str, str],
    objects: dict[str, str],
    type_parents: dict[str, str | None],
    bindings: dict[str, str],
) -> list[Any]:
    effect_expr = _substitute_expr(effect_expr, bindings)
    if effect_expr is None or not isinstance(effect_expr, list) or not effect_expr:
        return []

    operator = effect_expr[0]
    if not isinstance(operator, str):
        return []

    if operator == "and":
        collected: list[Any] = []
        for subexpr in effect_expr[1:]:
            collected.extend(
                _collect_effect_condition_payload_exprs(
                    subexpr,
                    constants=constants,
                    objects=objects,
                    type_parents=type_parents,
                    bindings=bindings,
                )
            )
        return collected

    if operator == "probabilistic":
        collected: list[Any] = []
        index = 1
        while index + 1 < len(effect_expr):
            branch_effect = effect_expr[index + 1]
            collected.extend(
                _collect_effect_condition_payload_exprs(
                    branch_effect,
                    constants=constants,
                    objects=objects,
                    type_parents=type_parents,
                    bindings=bindings,
                )
            )
            index += 2
        return collected

    if operator == "forall":
        typed_vars = effect_expr[1] if len(effect_expr) > 1 else []
        body = effect_expr[2] if len(effect_expr) > 2 else []
        var_specs = _iter_typed_variables(typed_vars)
        if not var_specs:
            return _collect_effect_condition_payload_exprs(
                body,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
                bindings=bindings,
            )
        collected: list[Any] = []
        for quantified_bindings in _enumerate_typed_bindings(
            var_specs,
            constants=constants,
            objects=objects,
            type_parents=type_parents,
        ):
            merged_bindings = dict(bindings)
            merged_bindings.update(quantified_bindings)
            collected.extend(
                _collect_effect_condition_payload_exprs(
                    body,
                    constants=constants,
                    objects=objects,
                    type_parents=type_parents,
                    bindings=merged_bindings,
                )
            )
        return collected

    if operator == "when":
        nested_effect = effect_expr[2] if len(effect_expr) > 2 else []
        collected = [_substitute_expr(nested_effect, bindings)]
        collected.extend(
            _collect_effect_condition_payload_exprs(
                nested_effect,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
                bindings=bindings,
            )
        )
        return collected

    return []


def compile_action_effect_distributions(
    *,
    actions: list[Action],
    parsed_domain: Any,
    layout: BitwiseIndexLayout,
    constants: dict[str, str],
    objects: dict[str, str],
    type_parents: dict[str, str | None],
) -> tuple[list[list[BitwiseEffectBranch]], list[list[list[BitwiseEffectBranch]]]]:
    """Compile grounded action effect branches.

    Returns:
    - unconditional/probabilistic effect branches for each grounded action
    - conditional payload branches for each grounded action, aligned with the
      previously compiled `when`-guard order
    """

    schema_by_signature: dict[tuple[str, int], list[Any]] = {}
    for action_schema in parsed_domain.actions:
        signature = (action_schema.action.name, len(action_schema.action.params))
        schema_by_signature.setdefault(signature, []).append(action_schema)

    unconditional_distributions: list[list[BitwiseEffectBranch]] = []
    conditional_distributions: list[list[list[BitwiseEffectBranch]]] = []
    for action in actions:
        signature = (action.name, len(action.params))
        candidate_schemas = schema_by_signature.get(signature, [])
        if not candidate_schemas:
            raise KeyError(f"No action schema found for grounded action `{action}`.")
        action_schema = candidate_schemas[0]
        bindings = _action_bindings_from_instance(action, action_schema.parameter_types)
        unconditional_distributions.append(
            _normalize_effect_branches(
                _compile_effect_expr_to_branches(
                    action_schema.effect,
                    layout=layout,
                    constants=constants,
                    objects=objects,
                    type_parents=type_parents,
                    bindings=bindings,
                    ignore_when=True,
                    effect_bucket_annotations=action_schema.effect_bucket_annotations,
                )
            )
        )
        conditional_payload_exprs = _collect_effect_condition_payload_exprs(
            action_schema.effect,
            constants=constants,
            objects=objects,
            type_parents=type_parents,
            bindings=bindings,
        )
        conditional_distributions.append(
            [
                _normalize_effect_branches(
                    _compile_effect_expr_to_branches(
                        payload_expr,
                        layout=layout,
                        constants=constants,
                        objects=objects,
                        type_parents=type_parents,
                        bindings=bindings,
                        ignore_when=False,
                        effect_bucket_annotations=None,
                    )
                )
                for payload_expr in conditional_payload_exprs
            ]
        )
    return unconditional_distributions, conditional_distributions


def compile_action_precondition_checks_from_module(
    module_name: str,
    *,
    builder_name: str = "build_model",
    shared_module_name: str = "shared",
) -> tuple[BitwiseIndexLayout, list[BitwiseGoalCheck]]:
    """Compile grounded action preconditions from a generated explicit model package."""

    module = import_module(module_name)
    factory = getattr(module, builder_name)
    if not callable(factory):
        raise TypeError(f"`{module_name}.{builder_name}` is not callable.")
    model = factory()
    if not isinstance(model, SemanticPOMDPModelBase):
        raise TypeError(
            f"`{module_name}.{builder_name}()` did not return a semantic POMDPModelBase instance."
        )

    shared_module = import_module(f"{module_name}.{shared_module_name}")
    domain_text = getattr(shared_module, "DOMAIN_TEXT")
    constants = getattr(shared_module, "CONSTANTS", {})
    objects = getattr(shared_module, "OBJECTS", {})
    type_parents = getattr(shared_module, "TYPE_PARENTS", {"object": None})

    parsed_domain = parse_domain(domain_text)
    layout = build_bitwise_index_layout(model)
    compiled = compile_action_precondition_checks(
        actions=list(model.actions),
        parsed_domain=parsed_domain,
        layout=layout,
        constants=constants,
        objects=objects,
        type_parents=type_parents,
    )
    return layout, compiled


def compile_action_effect_condition_checks_from_module(
    module_name: str,
    *,
    builder_name: str = "build_model",
    shared_module_name: str = "shared",
) -> tuple[BitwiseIndexLayout, list[list[BitwiseGoalCheck]]]:
    """Compile grounded conditional-effect guards from a generated explicit model package."""

    module = import_module(module_name)
    factory = getattr(module, builder_name)
    if not callable(factory):
        raise TypeError(f"`{module_name}.{builder_name}` is not callable.")
    model = factory()
    if not isinstance(model, SemanticPOMDPModelBase):
        raise TypeError(
            f"`{module_name}.{builder_name}()` did not return a semantic POMDPModelBase instance."
        )

    shared_module = import_module(f"{module_name}.{shared_module_name}")
    domain_text = getattr(shared_module, "DOMAIN_TEXT")
    constants = getattr(shared_module, "CONSTANTS", {})
    objects = getattr(shared_module, "OBJECTS", {})
    type_parents = getattr(shared_module, "TYPE_PARENTS", {"object": None})

    parsed_domain = parse_domain(domain_text)
    layout = build_bitwise_index_layout(model)
    compiled = compile_action_effect_condition_checks(
        actions=list(model.actions),
        parsed_domain=parsed_domain,
        layout=layout,
        constants=constants,
        objects=objects,
        type_parents=type_parents,
    )
    return layout, compiled


def compile_action_effect_distributions_from_module(
    module_name: str,
    *,
    builder_name: str = "build_model",
    shared_module_name: str = "shared",
) -> tuple[BitwiseIndexLayout, list[list[BitwiseEffectBranch]], list[list[list[BitwiseEffectBranch]]]]:
    """Compile grounded effect distributions from a generated explicit model package."""

    module = import_module(module_name)
    factory = getattr(module, builder_name)
    if not callable(factory):
        raise TypeError(f"`{module_name}.{builder_name}` is not callable.")
    model = factory()
    if not isinstance(model, SemanticPOMDPModelBase):
        raise TypeError(
            f"`{module_name}.{builder_name}()` did not return a semantic POMDPModelBase instance."
        )

    shared_module = import_module(f"{module_name}.{shared_module_name}")
    domain_text = getattr(shared_module, "DOMAIN_TEXT")
    constants = getattr(shared_module, "CONSTANTS", {})
    objects = getattr(shared_module, "OBJECTS", {})
    type_parents = getattr(shared_module, "TYPE_PARENTS", {"object": None})

    parsed_domain = parse_domain(domain_text)
    layout = build_bitwise_index_layout(model)
    unconditional, conditional = compile_action_effect_distributions(
        actions=list(model.actions),
        parsed_domain=parsed_domain,
        layout=layout,
        constants=constants,
        objects=objects,
        type_parents=type_parents,
    )
    return layout, unconditional, conditional


def compile_observation_rule_condition_checks(
    *,
    observation_rules: list[ObservationRule],
    parsed_domain: Any,
    layout: BitwiseIndexLayout,
    constants: dict[str, str],
    objects: dict[str, str],
    type_parents: dict[str, str | None],
) -> list[BitwiseGoalCheck]:
    """Compile one DNF mask condition for each grounded observation rule, by rule list order."""

    compiled_map: dict[ObservationRule, list[BitwiseGoalCheck]] = {}
    for rule_schema in parsed_domain.observation_rules:
        for bindings in _enumerate_typed_bindings(
            rule_schema.parameter_types,
            constants=constants,
            objects=objects,
            type_parents=type_parents,
        ):
            grounded_rule = _ground_observation_rule_with_bindings(rule_schema, bindings)
            condition_expr = _substitute_expr(rule_schema.condition, bindings)
            compiled_check = compile_goal_check(
                condition_expr,
                layout,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
            )
            compiled_map.setdefault(grounded_rule, []).append(compiled_check)

    compiled: list[BitwiseGoalCheck] = []
    for grounded_rule in observation_rules:
        checks = compiled_map.get(grounded_rule, [])
        if not checks:
            raise KeyError(f"No observation-rule condition found for grounded rule `{grounded_rule}`.")
        compiled.append(checks.pop(0))
    return compiled


def compile_observation_rule_distributions(
    *,
    observation_rules: list[ObservationRule],
    parsed_domain: Any,
    layout: BitwiseIndexLayout,
    constants: dict[str, str],
    objects: dict[str, str],
    type_parents: dict[str, str | None],
) -> list[list[BitwiseObservationBranch]]:
    """Compile observation distributions for each grounded observation rule, by rule list order."""

    compiled_map: dict[ObservationRule, list[list[BitwiseObservationBranch]]] = {}
    for rule_schema in parsed_domain.observation_rules:
        for bindings in _enumerate_typed_bindings(
            rule_schema.parameter_types,
            constants=constants,
            objects=objects,
            type_parents=type_parents,
        ):
            grounded_rule = _ground_observation_rule_with_bindings(rule_schema, bindings)
            compiled_distribution = _normalize_observation_branches(
                _compile_observation_distribution_to_branches(
                    rule_schema.distribution_expr,
                    layout=layout,
                    constants=constants,
                    objects=objects,
                    type_parents=type_parents,
                    bindings=bindings,
                )
            )
            compiled_map.setdefault(grounded_rule, []).append(compiled_distribution)

    compiled: list[list[BitwiseObservationBranch]] = []
    for grounded_rule in observation_rules:
        distributions = compiled_map.get(grounded_rule, [])
        if not distributions:
            raise KeyError(f"No observation distribution found for grounded rule `{grounded_rule}`.")
        compiled.append(distributions.pop(0))
    return compiled


def compile_observation_rule_condition_checks_from_module(
    module_name: str,
    *,
    builder_name: str = "build_model",
    shared_module_name: str = "shared",
) -> tuple[BitwiseIndexLayout, list[BitwiseGoalCheck]]:
    """Compile grounded observation-rule conditions from a generated explicit model package."""

    module = import_module(module_name)
    factory = getattr(module, builder_name)
    if not callable(factory):
        raise TypeError(f"`{module_name}.{builder_name}` is not callable.")
    model = factory()
    if not isinstance(model, SemanticPOMDPModelBase):
        raise TypeError(
            f"`{module_name}.{builder_name}()` did not return a semantic POMDPModelBase instance."
        )

    shared_module = import_module(f"{module_name}.{shared_module_name}")
    domain_text = getattr(shared_module, "DOMAIN_TEXT")
    constants = getattr(shared_module, "CONSTANTS", {})
    objects = getattr(shared_module, "OBJECTS", {})
    type_parents = getattr(shared_module, "TYPE_PARENTS", {"object": None})

    parsed_domain = parse_domain(domain_text)
    layout = build_bitwise_index_layout(model)
    compiled = compile_observation_rule_condition_checks(
        observation_rules=list(model.observation_rules),
        parsed_domain=parsed_domain,
        layout=layout,
        constants=constants,
        objects=objects,
        type_parents=type_parents,
    )
    return layout, compiled


def compile_observation_rule_distributions_from_module(
    module_name: str,
    *,
    builder_name: str = "build_model",
    shared_module_name: str = "shared",
) -> tuple[BitwiseIndexLayout, list[list[BitwiseObservationBranch]]]:
    """Compile grounded observation distributions from a generated explicit model package."""

    module = import_module(module_name)
    factory = getattr(module, builder_name)
    if not callable(factory):
        raise TypeError(f"`{module_name}.{builder_name}` is not callable.")
    model = factory()
    if not isinstance(model, SemanticPOMDPModelBase):
        raise TypeError(
            f"`{module_name}.{builder_name}()` did not return a semantic POMDPModelBase instance."
        )

    shared_module = import_module(f"{module_name}.{shared_module_name}")
    domain_text = getattr(shared_module, "DOMAIN_TEXT")
    constants = getattr(shared_module, "CONSTANTS", {})
    objects = getattr(shared_module, "OBJECTS", {})
    type_parents = getattr(shared_module, "TYPE_PARENTS", {"object": None})

    parsed_domain = parse_domain(domain_text)
    layout = build_bitwise_index_layout(model)
    compiled = compile_observation_rule_distributions(
        observation_rules=list(model.observation_rules),
        parsed_domain=parsed_domain,
        layout=layout,
        constants=constants,
        objects=objects,
        type_parents=type_parents,
    )
    return layout, compiled


def _default_policy_bindings_from_instance(
    default_policy_rule: DefaultPolicyRule,
    parameter_types: list[tuple[str, str]],
) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for idx, (param_name, _type_name) in enumerate(parameter_types):
        if idx >= len(default_policy_rule.params):
            break
        bindings[param_name] = default_policy_rule.params[idx]
    return bindings


def _ground_action_expr_to_action(
    action_expr: Any,
    bindings: dict[str, str],
) -> Action:
    grounded_expr = _substitute_expr(action_expr, bindings)
    if not isinstance(grounded_expr, list) or not grounded_expr:
        raise ValueError(f"Malformed default policy action expression: {grounded_expr!r}")
    head = grounded_expr[0]
    if not isinstance(head, str):
        raise ValueError(f"Malformed default policy action head: {head!r}")
    return Action(name=head, params=[str(param) for param in grounded_expr[1:]])


def compile_default_policy_rule_condition_checks(
    *,
    default_policy_rules: list[DefaultPolicyRule],
    parsed_default_policy: Any,
    layout: BitwiseIndexLayout,
    constants: dict[str, str],
    objects: dict[str, str],
    type_parents: dict[str, str | None],
) -> list[BitwiseGoalCheck]:
    """Compile one DNF mask condition for each grounded default-policy rule, by rule list order."""

    schema_by_signature: dict[tuple[str, int], list[Any]] = {}
    for rule_schema in parsed_default_policy.rules:
        signature = (rule_schema.rule.name, len(rule_schema.rule.params))
        schema_by_signature.setdefault(signature, []).append(rule_schema)

    compiled: list[BitwiseGoalCheck] = []
    for grounded_rule in default_policy_rules:
        signature = (grounded_rule.name, len(grounded_rule.params))
        candidate_schemas = schema_by_signature.get(signature, [])
        if not candidate_schemas:
            raise KeyError(f"No default-policy schema found for grounded rule `{grounded_rule}`.")
        rule_schema = candidate_schemas[0]
        bindings = _default_policy_bindings_from_instance(grounded_rule, rule_schema.parameter_types)
        precondition_expr = _substitute_expr(rule_schema.precondition, bindings)
        compiled.append(
            compile_goal_check(
                precondition_expr,
                layout,
                constants=constants,
                objects=objects,
                type_parents=type_parents,
            )
        )
    return compiled


def compile_default_policy_rule_action_ids(
    *,
    default_policy_rules: list[DefaultPolicyRule],
    parsed_default_policy: Any,
    layout: BitwiseIndexLayout,
) -> list[int]:
    """Compile one grounded action id for each grounded default-policy rule, by rule list order."""

    schema_by_signature: dict[tuple[str, int], list[Any]] = {}
    for rule_schema in parsed_default_policy.rules:
        signature = (rule_schema.rule.name, len(rule_schema.rule.params))
        schema_by_signature.setdefault(signature, []).append(rule_schema)

    compiled: list[int] = []
    for grounded_rule in default_policy_rules:
        signature = (grounded_rule.name, len(grounded_rule.params))
        candidate_schemas = schema_by_signature.get(signature, [])
        if not candidate_schemas:
            raise KeyError(f"No default-policy schema found for grounded rule `{grounded_rule}`.")
        rule_schema = candidate_schemas[0]
        bindings = _default_policy_bindings_from_instance(grounded_rule, rule_schema.parameter_types)
        grounded_action = _ground_action_expr_to_action(rule_schema.action_expr, bindings)
        if grounded_action not in layout.action_index:
            if grounded_action.name == "report-goal" and not grounded_action.params:
                compiled.append(REPORT_GOAL_ACTION_ID_SENTINEL)
                continue
            raise KeyError(
                f"Default-policy grounded action `{grounded_action}` was not found in grounded actions."
            )
        compiled.append(layout.action_index[grounded_action])
    return compiled


def compile_default_policy_rule_condition_checks_from_module(
    module_name: str,
    *,
    builder_name: str = "build_model",
    shared_module_name: str = "shared",
) -> tuple[BitwiseIndexLayout, list[BitwiseGoalCheck]]:
    """Compile grounded default-policy rule conditions from a generated explicit model package."""

    module = import_module(module_name)
    factory = getattr(module, builder_name)
    if not callable(factory):
        raise TypeError(f"`{module_name}.{builder_name}` is not callable.")
    model = factory()
    if not isinstance(model, SemanticPOMDPModelBase):
        raise TypeError(
            f"`{module_name}.{builder_name}()` did not return a semantic POMDPModelBase instance."
        )

    shared_module = import_module(f"{module_name}.{shared_module_name}")
    default_policy_text = getattr(shared_module, "DEFAULT_POLICY_TEXT")
    constants = getattr(shared_module, "CONSTANTS", {})
    objects = getattr(shared_module, "OBJECTS", {})
    type_parents = getattr(shared_module, "TYPE_PARENTS", {"object": None})

    parsed_default_policy = parse_default_policy(default_policy_text)
    layout = build_bitwise_index_layout(model)
    compiled = compile_default_policy_rule_condition_checks(
        default_policy_rules=list(model.default_policy_rules),
        parsed_default_policy=parsed_default_policy,
        layout=layout,
        constants=constants,
        objects=objects,
        type_parents=type_parents,
    )
    return layout, compiled


def compile_default_policy_rule_action_ids_from_module(
    module_name: str,
    *,
    builder_name: str = "build_model",
    shared_module_name: str = "shared",
) -> tuple[BitwiseIndexLayout, list[int]]:
    """Compile grounded default-policy actions to integer action ids."""

    module = import_module(module_name)
    factory = getattr(module, builder_name)
    if not callable(factory):
        raise TypeError(f"`{module_name}.{builder_name}` is not callable.")
    model = factory()
    if not isinstance(model, SemanticPOMDPModelBase):
        raise TypeError(
            f"`{module_name}.{builder_name}()` did not return a semantic POMDPModelBase instance."
        )

    shared_module = import_module(f"{module_name}.{shared_module_name}")
    default_policy_text = getattr(shared_module, "DEFAULT_POLICY_TEXT")
    parsed_default_policy = parse_default_policy(default_policy_text)
    layout = build_bitwise_index_layout(model)
    compiled = compile_default_policy_rule_action_ids(
        default_policy_rules=list(model.default_policy_rules),
        parsed_default_policy=parsed_default_policy,
        layout=layout,
    )
    return layout, compiled


def build_bitwise_model_skeleton_from_factory(
    factory: Callable[[], SemanticPOMDPModelBase],
) -> BitwiseModelSkeleton:
    """Instantiate a semantic model from a factory and convert its member parameters."""

    return build_bitwise_model_skeleton(factory())


def build_bitwise_model_skeleton_from_module(
    module_name: str,
    *,
    builder_name: str = "build_model",
) -> BitwiseModelSkeleton:
    """Import a generated explicit Python model package and convert its member parameters.

    Parameters
    ----------
    module_name:
        Import path of the generated explicit Python model package, such as
        ``POMDPDDL.tiger_explicit_package``.
    builder_name:
        Name of the factory function that returns the semantic model instance.
    """

    module = import_module(module_name)
    factory = getattr(module, builder_name)
    if not callable(factory):
        raise TypeError(f"`{module_name}.{builder_name}` is not callable.")
    model = factory()
    if not isinstance(model, SemanticPOMDPModelBase):
        raise TypeError(
            f"`{module_name}.{builder_name}()` did not return a semantic POMDPModelBase instance."
        )
    shared_module = import_module(f"{module_name}.shared")
    goal_expr = getattr(shared_module, "GOAL_EXPR")
    constants = getattr(shared_module, "CONSTANTS", {})
    objects = getattr(shared_module, "OBJECTS", {})
    type_parents = getattr(shared_module, "TYPE_PARENTS", {"object": None})

    metadata = extract_bitwise_metadata(model)
    layout = build_bitwise_index_layout(model)
    goal_check = compile_goal_check(
        goal_expr,
        layout,
        constants=constants,
        objects=objects,
        type_parents=type_parents,
    )
    parsed_domain = parse_domain(getattr(shared_module, "DOMAIN_TEXT"))
    parsed_default_policy = parse_default_policy(getattr(shared_module, "DEFAULT_POLICY_TEXT"))
    action_precondition_checks = compile_action_precondition_checks(
        actions=list(model.actions),
        parsed_domain=parsed_domain,
        layout=layout,
        constants=constants,
        objects=objects,
        type_parents=type_parents,
    )
    action_effect_condition_checks = compile_action_effect_condition_checks(
        actions=list(model.actions),
        parsed_domain=parsed_domain,
        layout=layout,
        constants=constants,
        objects=objects,
        type_parents=type_parents,
    )
    action_effect_distributions, action_conditional_effect_distributions = compile_action_effect_distributions(
        actions=list(model.actions),
        parsed_domain=parsed_domain,
        layout=layout,
        constants=constants,
        objects=objects,
        type_parents=type_parents,
    )
    observation_rule_condition_checks = compile_observation_rule_condition_checks(
        observation_rules=list(model.observation_rules),
        parsed_domain=parsed_domain,
        layout=layout,
        constants=constants,
        objects=objects,
        type_parents=type_parents,
    )
    observation_rule_distributions = compile_observation_rule_distributions(
        observation_rules=list(model.observation_rules),
        parsed_domain=parsed_domain,
        layout=layout,
        constants=constants,
        objects=objects,
        type_parents=type_parents,
    )
    default_policy_rule_condition_checks = compile_default_policy_rule_condition_checks(
        default_policy_rules=list(model.default_policy_rules),
        parsed_default_policy=parsed_default_policy,
        layout=layout,
        constants=constants,
        objects=objects,
        type_parents=type_parents,
    )
    default_policy_rule_action_ids = compile_default_policy_rule_action_ids(
        default_policy_rules=list(model.default_policy_rules),
        parsed_default_policy=parsed_default_policy,
        layout=layout,
    )
    return BitwiseModelSkeleton(
        **asdict(metadata),
        goal_check=goal_check,
        action_precondition_checks=action_precondition_checks,
        action_effect_condition_checks=action_effect_condition_checks,
        action_effect_distributions=action_effect_distributions,
        action_conditional_effect_distributions=action_conditional_effect_distributions,
        observation_rule_condition_checks=observation_rule_condition_checks,
        observation_rule_distributions=observation_rule_distributions,
        observation_rule_skip_before_for_active_perception=[
            observation_rule.name.startswith("before_")
            for observation_rule in model.observation_rules
        ],
        action_is_active_perception=[
            action.name.startswith("active_obs_")
            for action in model.actions
        ],
        default_policy_rule_condition_checks=default_policy_rule_condition_checks,
        default_policy_rule_action_ids=default_policy_rule_action_ids,
    )


def compile_goal_check_from_model(
    model: SemanticPOMDPModelBase,
    goal_expr: Any,
    *,
    constants: dict[str, str] | None = None,
    objects: dict[str, str] | None = None,
    type_parents: dict[str, str | None] | None = None,
) -> BitwiseGoalCheck:
    """Compile a semantic-model goal expression using list-order bit indices."""

    layout = build_bitwise_index_layout(model)
    return compile_goal_check(
        goal_expr,
        layout,
        constants=constants,
        objects=objects,
        type_parents=type_parents,
    )


def compile_goal_check_from_module(
    module_name: str,
    *,
    builder_name: str = "build_model",
    shared_module_name: str = "shared",
) -> tuple[BitwiseIndexLayout, BitwiseGoalCheck]:
    """Compile the generated package goal expression into a bitwise goal check."""

    module = import_module(module_name)
    factory = getattr(module, builder_name)
    if not callable(factory):
        raise TypeError(f"`{module_name}.{builder_name}` is not callable.")
    model = factory()
    if not isinstance(model, SemanticPOMDPModelBase):
        raise TypeError(
            f"`{module_name}.{builder_name}()` did not return a semantic POMDPModelBase instance."
        )

    shared_module = import_module(f"{module_name}.{shared_module_name}")
    goal_expr = getattr(shared_module, "GOAL_EXPR")
    constants = getattr(shared_module, "CONSTANTS", {})
    objects = getattr(shared_module, "OBJECTS", {})
    type_parents = getattr(shared_module, "TYPE_PARENTS", {"object": None})

    layout = build_bitwise_index_layout(model)
    goal_check = compile_goal_check(
        goal_expr,
        layout,
        constants=constants,
        objects=objects,
        type_parents=type_parents,
    )
    return layout, goal_check


def describe_goal_check(goal_check: BitwiseGoalCheck) -> dict[str, Any]:
    """Return a serializable view of one compiled bitwise goal check."""

    return asdict(goal_check)


def describe_action_precondition_checks(
    action_precondition_checks: list[BitwiseGoalCheck],
) -> list[dict[str, Any]]:
    """Return a serializable view of compiled grounded action preconditions."""

    return [describe_goal_check(goal_check) for goal_check in action_precondition_checks]


def describe_action_effect_condition_checks(
    action_effect_condition_checks: list[list[BitwiseGoalCheck]],
) -> list[list[dict[str, Any]]]:
    """Return a serializable view of compiled grounded conditional-effect guards."""

    return [
        [describe_goal_check(goal_check) for goal_check in action_checks]
        for action_checks in action_effect_condition_checks
    ]


def describe_action_effect_distributions(
    action_effect_distributions: list[list[BitwiseEffectBranch]],
) -> list[list[dict[str, Any]]]:
    """Return a serializable view of unconditional grounded action effect branches."""

    return [[asdict(branch) for branch in branches] for branches in action_effect_distributions]


def describe_action_conditional_effect_distributions(
    action_conditional_effect_distributions: list[list[list[BitwiseEffectBranch]]],
) -> list[list[list[dict[str, Any]]]]:
    """Return a serializable view of conditional payload branches aligned with `when` guards."""

    return [
        [[asdict(branch) for branch in branches] for branches in guarded_branches]
        for guarded_branches in action_conditional_effect_distributions
    ]


def describe_observation_rule_condition_checks(
    observation_rule_condition_checks: list[BitwiseGoalCheck],
) -> list[dict[str, Any]]:
    """Return a serializable view of compiled grounded observation-rule conditions."""

    return [describe_goal_check(goal_check) for goal_check in observation_rule_condition_checks]


def describe_observation_rule_distributions(
    observation_rule_distributions: list[list[BitwiseObservationBranch]],
) -> list[list[dict[str, Any]]]:
    """Return a serializable view of grounded observation distributions."""

    return [[asdict(branch) for branch in branches] for branches in observation_rule_distributions]


def describe_default_policy_rule_condition_checks(
    default_policy_rule_condition_checks: list[BitwiseGoalCheck],
) -> list[dict[str, Any]]:
    """Return a serializable view of compiled grounded default-policy rule conditions."""

    return [describe_goal_check(goal_check) for goal_check in default_policy_rule_condition_checks]


def describe_default_policy_rule_action_ids(
    default_policy_rule_action_ids: list[int],
) -> list[int]:
    """Return the grounded action id chosen by each grounded default-policy rule."""

    return list(default_policy_rule_action_ids)


def describe_bitwise_metadata(model: SemanticPOMDPModelBase | BitwisePOMDPModelBase) -> dict[str, Any]:
    """Return a small serializable summary of the member-parameter conversion."""

    return {
        "grounded_predicates_count": model.grounded_predicates_count,
        "grounded_observables_count": model.grounded_observables_count,
        "total_actions": model.total_actions,
        "maximize_reward": model.maximize_reward,
        "goal_reward": model.goal_reward,
    }
