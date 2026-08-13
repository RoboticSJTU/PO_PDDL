"""Runtime planner wrapper for Python POMDP models."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
import tempfile
import time
from collections import defaultdict
from typing import Iterator, cast

from ..base.pomdp_model import POMDPModelBase
from ..bitwise import (
    IndexedParticleBelief,
    POMDPModelBase as BitwisePOMDPModelBase,
    build_bitwise_index_layout,
    convert_factorized_belief_to_indexed_particles,
    convert_pomdp_model_to_bitwise_model,
)
from ..data_structures import (
    Action,
    BeliefFactor,
    DespotCppModelCode,
        EffectBucket,
        FactorizedBelief,
        ObservationEntry,
        Predicate,
)
from .probability_registry import (
    build_probability_registry,
    extract_updated_probabilities,
    update_bitwise_model_probabilities,
)
from .despot_runtime_support import (
    configure_and_build_despot_cpp_project,
    convert_bitwise_model_to_despot_cpp_model_code,
    instantiate_despot_planner,
    write_despot_cpp_model_code,
    write_runtime_belief_json,
)
from .observation_semantics import normalize_bitwise_observation_bits, normalize_semantic_observation_entry
from .paths import despot_root


class POMDPPlanner:
    """Minimal runtime planner shell around semantic and bitwise POMDP models."""

    def __init__(
        self,
        pomdp_model: POMDPModelBase,
        init_belief: FactorizedBelief | IndexedParticleBelief,
        bitwise_pomdp_model: BitwisePOMDPModelBase | None = None,
        despot_cpp_model_code: DespotCppModelCode | None = None,
        despot_shared_library_path: str | Path | None = None,
        random_seed: int | None = None,
        skip_despot_build: bool = False,
        belief_update_debug: bool = False,
    ) -> None:
        self.pomdp_model = pomdp_model
        self.random_seed = random_seed
        self.current_step = 0
        self.belief_update_debug = belief_update_debug
        self.last_belief_update_profile: dict[str, object] | None = None
        self._bitwise_layout = build_bitwise_index_layout(pomdp_model)
        self._despot_root = despot_root()
        self.bitwise_pomdp_model = (
            bitwise_pomdp_model
            if bitwise_pomdp_model is not None
            else convert_pomdp_model_to_bitwise_model(pomdp_model)
        )
        self.current_belief = self._normalize_init_belief(init_belief)
        self._last_runtime_observation_bits = 0
        self._has_last_runtime_observation = False
        self._semantic_state_cache: dict[int, dict] = {}
        self._observation_rule_condition_cache: dict[tuple[int, int, int], bool] = {}
        self._applicable_observable_bits_cache: dict[tuple[int, int], tuple[int, ...]] = {}
        self._action_outcome_bits_cache: dict[
            tuple[int, int, str | None, bool | None, int | None, int | None],
            list[tuple[int, float]],
        ] = {}
        self._action_outcome_bits_with_reward_cache: dict[
            tuple[int, int, str | None, bool | None, int | None, int | None],
            list[tuple[int, float, float]],
        ] = {}
        self._observation_rule_support_masks = self._build_observation_rule_support_masks()
        self._observation_rules_by_support_mask = self._build_observation_rules_by_support_mask()
        self._observation_action_anchor_masks = self._build_observation_action_anchor_masks()
        (
            self._observation_rules_by_anchor_mask,
            self._generic_observation_rule_indices,
        ) = self._build_observation_rules_by_anchor_mask()
        (
            self._observation_rules_by_action_id,
            self._generic_action_observation_rule_indices,
        ) = self._build_observation_rules_by_action_id()
        self._init_observation_rule_indices = self._build_init_observation_rule_indices()
        self.despot_cpp_model_code = (
            despot_cpp_model_code
            if despot_cpp_model_code is not None
            else convert_bitwise_model_to_despot_cpp_model_code(
                self.bitwise_pomdp_model,
                semantic_model=self.pomdp_model,
                despot_root=self._despot_root,
            )
        )
        self._runtime_root = Path(tempfile.mkdtemp(prefix="pomdpddl_despot_runtime_"))
        self._despot_cpp_dir = self._runtime_root / "despot_cpp"
        self._despot_build_dir = self._despot_cpp_dir / "build"
        self._runtime_belief_json_path = self._despot_build_dir / "init_belief.json"
        self._despot_shared_library_path = None
        self._despot_planner_module = None
        self._despot_planner_instance = None
        if despot_shared_library_path is not None:
            self._despot_shared_library_path = Path(despot_shared_library_path)
            self._despot_build_dir = self._despot_shared_library_path.parent
            self._runtime_belief_json_path = self._despot_build_dir / "init_belief.json"
            (
                self._despot_planner_module,
                self._despot_planner_instance,
            ) = instantiate_despot_planner(self._despot_shared_library_path)
        elif not skip_despot_build:
            write_despot_cpp_model_code(self.despot_cpp_model_code, self._despot_cpp_dir)
            self._despot_shared_library_path = configure_and_build_despot_cpp_project(
                self._despot_cpp_dir,
                self._despot_build_dir,
                despot_root=self._despot_root,
            )
            (
                self._despot_planner_module,
                self._despot_planner_instance,
            ) = instantiate_despot_planner(self._despot_shared_library_path)
        else:
            self._despot_shared_library_path = None

    def _belief_debug(self, message: str) -> None:
        if self.belief_update_debug:
            print(f"[belief_update] {message}", flush=True)

    def _normalize_init_belief(
        self,
        init_belief: FactorizedBelief | IndexedParticleBelief,
    ) -> IndexedParticleBelief:
        """Normalize one planner init-belief into indexed particle-belief form."""
        if isinstance(init_belief, IndexedParticleBelief):
            belief = deepcopy(init_belief)
            belief.validate()
            return belief
        if not isinstance(init_belief, FactorizedBelief):
            raise TypeError(
                "POMDPPlanner expects init_belief to be a FactorizedBelief or IndexedParticleBelief."
            )
        belief = deepcopy(init_belief)
        belief.validate()
        return convert_factorized_belief_to_indexed_particles(belief, self._bitwise_layout)

    def _step_seed(self) -> int | None:
        if self.random_seed is None:
            return None
        return int(self.random_seed) + int(self.current_step)

    def _despot_root_seed(self) -> int | None:
        step_seed = self._step_seed()
        if step_seed is None:
            return None
        return int(step_seed) & 0xFFFFFFFF

    def _observation_to_bits_and_mask(self, observation: ObservationEntry) -> tuple[int, int]:
        observation = normalize_semantic_observation_entry(observation)
        observation_bits = 0
        observation_mask = 0
        for observable, value in observation.items():
            if observable not in self._bitwise_layout.observable_index:
                continue
            bit_index = self._bitwise_layout.observable_index[observable]
            bit_mask = 1 << bit_index
            observation_mask |= bit_mask
            if value:
                observation_bits |= bit_mask
        return normalize_bitwise_observation_bits(
            observation_bits,
            observation_mask,
            self._bitwise_layout.observables,
        )

    def _build_state_from_case_indices(
        self,
        belief: FactorizedBelief,
        case_indices: tuple[int, ...],
    ) -> dict:
        state = {predicate: True for predicate in belief.known_true}
        state.update({predicate: False for predicate in belief.known_false})
        for factor, case_index in zip(belief.factors, case_indices):
            _probability, true_predicates = factor.cases[case_index]
            true_set = set(true_predicates)
            for predicate in factor.scope:
                state[predicate] = predicate in true_set
        return state

    def _semantic_state_from_bits(self, state_bits: int) -> dict:
        cached = self._semantic_state_cache.get(state_bits)
        if cached is not None:
            return dict(cached)
        state = {
            predicate: True
            for index, predicate in enumerate(self._bitwise_layout.predicates)
            if (state_bits >> index) & 1
        }
        self._semantic_state_cache[state_bits] = dict(state)
        return dict(state)

    def _state_to_bits(self, state: dict) -> int:
        bits = 0
        for predicate, value in state.items():
            if not value:
                continue
            if predicate not in self._bitwise_layout.predicate_index:
                continue
            bits |= 1 << self._bitwise_layout.predicate_index[predicate]
        return bits

    def _resolve_predicate_from_observable(self, observable: Predicate) -> Predicate | None:
        name = observable.name
        if name.startswith("obs_"):
            predicate_name = name[len("obs_") :]
        elif name.startswith("obs-"):
            predicate_name = name[len("obs-") :]
        else:
            return None
        return Predicate(predicate_name, list(observable.params))

    def _predicate_marginal_probability(
        self,
        belief: IndexedParticleBelief,
        predicate: Predicate,
    ) -> float | None:
        predicate_index = self._bitwise_layout.predicate_index.get(predicate)
        if predicate_index is None:
            return None
        probability = 0.0
        for state_bits, state_probability in belief.particles:
            if state_probability <= 0.0:
                continue
            if (state_bits >> predicate_index) & 1:
                probability += state_probability
        return probability

    def _build_observation_rule_support_masks(self) -> list[int]:
        distributions = getattr(self.bitwise_pomdp_model, "observation_rule_distributions", None)
        if distributions is None:
            return []
        support_masks: list[int] = []
        for branches in distributions:
            support_mask = 0
            for branch in branches:
                support_mask |= branch.observation_mask
            support_masks.append(support_mask)
        return support_masks

    def _rule_support_mask(self, observation_rule_index: int) -> int:
        if observation_rule_index < len(self._observation_rule_support_masks):
            return self._observation_rule_support_masks[observation_rule_index]
        raise IndexError(f"observation rule index out of range: {observation_rule_index}")

    def _build_observation_rules_by_support_mask(self) -> dict[int, list[int]]:
        mapping: dict[int, list[int]] = defaultdict(list)
        for rule_index, support_mask in enumerate(self._observation_rule_support_masks):
            mapping[support_mask].append(rule_index)
        return dict(mapping)

    def _extract_common_required_true_mask(self, rule_index: int) -> int:
        checks = getattr(self.bitwise_pomdp_model, "observation_rule_condition_checks", None)
        if checks is None or rule_index >= len(checks):
            return 0
        goal_check = checks[rule_index]
        clauses = getattr(goal_check, "clauses", None)
        if not clauses:
            return 0

        common_true_mask: int | None = None
        for required_true_mask, _required_false_mask in clauses:
            common_true_mask = (
                required_true_mask
                if common_true_mask is None
                else (common_true_mask & required_true_mask)
            )
            if common_true_mask == 0:
                return 0
        return common_true_mask or 0

    def _extract_common_true_anchor_mask(self, rule_index: int) -> int:
        common_true_mask = self._extract_common_required_true_mask(rule_index)
        if not common_true_mask:
            return 0
        return common_true_mask & -common_true_mask

    def _build_observation_action_anchor_masks(self) -> dict[int, int]:
        action_masks: dict[int, int] = {action_id: 0 for action_id in range(len(self.pomdp_model.actions))}
        action_str_to_id = {
            action.to_pddl_str(): action_id
            for action, action_id in self._bitwise_layout.action_index.items()
        }
        action_name_to_id = {
            action.to_pddl_str()[1:-1].split()[0]: action_id
            for action, action_id in self._bitwise_layout.action_index.items()
        }

        for bit_index, predicate in enumerate(self._bitwise_layout.predicates):
            predicate_str = predicate.to_pddl_str()
            action_id: int | None = None
            if predicate_str.startswith("(last-") and predicate_str.endswith(")"):
                candidate_action = "(" + predicate_str[len("(last-"):-1] + ")"
                action_id = action_str_to_id.get(candidate_action)
            elif predicate_str.startswith("(last-action ") and predicate_str.endswith(")"):
                payload = predicate_str[len("(last-action "):-1]
                if payload.startswith("act_"):
                    candidate_name = payload[len("act_"):].replace("_", "-")
                    action_id = action_name_to_id.get(candidate_name)
            if action_id is None:
                continue
            action_masks[action_id] |= 1 << bit_index

        return {action_id: mask for action_id, mask in action_masks.items() if mask != 0}

    def _clause_action_ids(self, required_true_mask: int) -> set[int]:
        action_ids: set[int] = set()
        if required_true_mask == 0:
            return action_ids
        for action_id, action_mask in self._observation_action_anchor_masks.items():
            if required_true_mask & action_mask:
                action_ids.add(action_id)
        return action_ids

    def _build_observation_rules_by_anchor_mask(self) -> tuple[dict[int, list[int]], list[int]]:
        mapping: dict[int, list[int]] = defaultdict(list)
        generic_rules: list[int] = []
        for rule_index in range(len(self._observation_rule_support_masks)):
            anchor_mask = self._extract_common_true_anchor_mask(rule_index)
            if anchor_mask == 0:
                generic_rules.append(rule_index)
            else:
                mapping[anchor_mask].append(rule_index)
        return dict(mapping), generic_rules

    def _build_observation_rules_by_action_id(self) -> tuple[dict[int, list[int]], list[int]]:
        mapping: dict[int, list[int]] = defaultdict(list)
        generic_rules: list[int] = []
        if not self._observation_action_anchor_masks:
            return {}, list(range(len(self._observation_rule_support_masks)))

        checks = getattr(self.bitwise_pomdp_model, "observation_rule_condition_checks", None)
        if checks is None:
            return {}, list(range(len(self._observation_rule_support_masks)))

        for rule_index in range(len(self._observation_rule_support_masks)):
            clauses = getattr(checks[rule_index], "clauses", None)
            if not clauses:
                generic_rules.append(rule_index)
                continue

            clause_action_sets = [
                self._clause_action_ids(required_true_mask)
                for required_true_mask, _required_false_mask in clauses
            ]
            clause_action_sets = [action_ids for action_ids in clause_action_sets if action_ids]
            if not clause_action_sets:
                generic_rules.append(rule_index)
                continue

            allowed_action_ids = set(clause_action_sets[0])
            for clause_action_ids in clause_action_sets[1:]:
                allowed_action_ids &= clause_action_ids

            if not allowed_action_ids:
                allowed_action_ids = set().union(*clause_action_sets)
                if len(allowed_action_ids) > 3:
                    generic_rules.append(rule_index)
                    continue

            for action_id in sorted(allowed_action_ids):
                mapping[action_id].append(rule_index)
        return dict(mapping), generic_rules

    def _build_init_observation_rule_indices(self) -> set[int]:
        return {
            rule_index
            for rule_index, observation_rule in enumerate(self.pomdp_model.observation_rules)
            if str(getattr(observation_rule, "name", "") or "").strip().startswith("init_obs_")
        }

    def _candidate_observation_rule_indices(
        self,
        action_id: int,
        state_bits: int,
        observation_mask: int,
    ) -> list[int] | range:
        if observation_mask == 0:
            zero_mask_rules = self._observation_rules_by_support_mask.get(0)
            if zero_mask_rules:
                seed_candidates = zero_mask_rules
            else:
                seed_candidates = range(len(self._observation_rule_support_masks))
        else:
            seed_candidates = []
            for rule_index, support_mask in enumerate(self._observation_rule_support_masks):
                if support_mask & observation_mask:
                    seed_candidates.append(rule_index)

        candidate_set = set(self._generic_observation_rule_indices)
        pending_anchor_candidates: set[int] = set()
        remaining_state_bits = state_bits
        while remaining_state_bits:
            anchor_mask = remaining_state_bits & -remaining_state_bits
            remaining_state_bits &= remaining_state_bits - 1
            for rule_index in self._observation_rules_by_anchor_mask.get(anchor_mask, []):
                pending_anchor_candidates.add(rule_index)

        if pending_anchor_candidates:
            candidate_set.update(pending_anchor_candidates)

        action_candidate_set = set(self._generic_action_observation_rule_indices)
        action_candidate_set.update(self._observation_rules_by_action_id.get(action_id, []))

        if not candidate_set and not action_candidate_set:
            return list(seed_candidates)

        candidates: list[int] = []
        for rule_index in seed_candidates:
            if candidate_set and rule_index not in candidate_set:
                continue
            if action_candidate_set and rule_index not in action_candidate_set:
                continue
            candidates.append(rule_index)
        if candidates:
            return candidates
        if observation_mask == 0:
            return []
        for rule_index, support_mask in enumerate(self._observation_rule_support_masks):
            if support_mask & observation_mask:
                if candidate_set and rule_index not in candidate_set:
                    continue
                if action_candidate_set and rule_index not in action_candidate_set:
                    continue
                candidates.append(rule_index)
        return candidates

    def _check_observation_rule_condition_cached(
        self,
        rule_index: int,
        state_bits: int,
        action_id: int,
    ) -> bool:
        cache_key = (rule_index, state_bits, action_id)
        cached = self._observation_rule_condition_cache.get(cache_key)
        if cached is not None:
            return cached
        result = self.bitwise_pomdp_model.check_observation_rule_condition(
            rule_index,
            state_bits,
            current_action=action_id,
        )
        self._observation_rule_condition_cache[cache_key] = result
        return result

    def _compute_observation_likelihood(
        self,
        action_id: int,
        observation_bits: int,
        observation_mask: int,
        state_bits: int,
        profile: dict[str, float | int] | None = None,
    ) -> float:
        distributions = getattr(self.bitwise_pomdp_model, "observation_rule_distributions", None)
        if distributions is None:
            raise NotImplementedError(
                "bitwise_pomdp_model does not expose observation_rule_distributions needed for belief update."
            )

        total_likelihood = 1.0
        support_mask_seconds = 0.0
        condition_check_seconds = 0.0
        branch_match_seconds = 0.0
        active_rule_count = 0
        branch_count_checked = 0
        candidate_rule_indices = self._candidate_observation_rule_indices(action_id, state_bits, observation_mask)

        candidate_rule_count = 0
        active_rule_indices: list[int] = []
        for rule_index in candidate_rule_indices:
            if rule_index in self._init_observation_rule_indices:
                continue
            candidate_rule_count += 1
            support_mask_start = time.perf_counter()
            support_mask_seconds += time.perf_counter() - support_mask_start

            condition_start = time.perf_counter()
            is_active = self._check_observation_rule_condition_cached(
                rule_index,
                state_bits,
                action_id,
            )
            condition_check_seconds += time.perf_counter() - condition_start
            if not is_active:
                continue

            active_rule_count += 1
            active_rule_indices.append(rule_index)

        remaining_mask = observation_mask
        while remaining_mask:
            observable_mask = remaining_mask & -remaining_mask
            remaining_mask &= remaining_mask - 1
            observed_positive = bool(observation_bits & observable_mask)
            observable_likelihood = 1.0
            matched_any_precondition = False

            for rule_index in active_rule_indices:
                support_mask_start = time.perf_counter()
                support_mask = self._rule_support_mask(rule_index)
                support_mask_seconds += time.perf_counter() - support_mask_start
                if not (support_mask & observable_mask):
                    continue

                matched_any_precondition = True
                branches = distributions[rule_index]
                rule_likelihood = 0.0
                branch_match_start = time.perf_counter()
                for branch in branches:
                    if not (branch.observation_mask & observable_mask):
                        continue
                    branch_count_checked += 1
                    branch_positive = bool(branch.observation_bits & observable_mask)
                    if branch_positive == observed_positive:
                        rule_likelihood += max(branch.probability, 0.0)
                branch_match_seconds += time.perf_counter() - branch_match_start
                observable_likelihood *= rule_likelihood
                if observable_likelihood <= 0.0:
                    break

            if not matched_any_precondition:
                continue

            total_likelihood *= observable_likelihood
            if total_likelihood <= 0.0:
                if profile is not None:
                    profile["rule_count"] = profile.get("rule_count", 0) + candidate_rule_count
                    profile["active_rule_count"] = profile.get("active_rule_count", 0) + active_rule_count
                    profile["branch_count_checked"] = profile.get("branch_count_checked", 0) + branch_count_checked
                    profile["candidate_rule_count"] = profile.get("candidate_rule_count", 0) + candidate_rule_count
                    profile["support_mask_seconds"] = profile.get("support_mask_seconds", 0.0) + support_mask_seconds
                    profile["condition_check_seconds"] = profile.get("condition_check_seconds", 0.0) + condition_check_seconds
                    profile["branch_match_seconds"] = profile.get("branch_match_seconds", 0.0) + branch_match_seconds
                return 0.0
        if profile is not None:
            profile["rule_count"] = profile.get("rule_count", 0) + candidate_rule_count
            profile["active_rule_count"] = profile.get("active_rule_count", 0) + active_rule_count
            profile["branch_count_checked"] = profile.get("branch_count_checked", 0) + branch_count_checked
            profile["candidate_rule_count"] = profile.get("candidate_rule_count", 0) + candidate_rule_count
            profile["support_mask_seconds"] = profile.get("support_mask_seconds", 0.0) + support_mask_seconds
            profile["condition_check_seconds"] = profile.get("condition_check_seconds", 0.0) + condition_check_seconds
            profile["branch_match_seconds"] = profile.get("branch_match_seconds", 0.0) + branch_match_seconds
        return total_likelihood

    def _predict_update_pass(
        self,
        belief: FactorizedBelief,
        action: Action | int,
        action_id: int,
        observation_bits: int,
        observation_mask: int,
        effect_bucket: EffectBucket | None,
        skip_observation_update: bool,
        should_prune_goal_particles: bool,
        debug_label: str = "primary",
    ) -> dict[str, object]:
        posterior_state_weights: dict[int, float] = {}
        normalization = 0.0
        applicable_mass = 0.0
        observation_profile: dict[str, float | int] = {
            "rule_count": 0,
            "candidate_rule_count": 0,
            "active_rule_count": 0,
            "branch_count_checked": 0,
            "support_mask_seconds": 0.0,
            "condition_check_seconds": 0.0,
            "branch_match_seconds": 0.0,
        }
        action_outcome_seconds = 0.0
        observation_likelihood_seconds = 0.0
        posterior_aggregation_seconds = 0.0
        total_transition_outcomes = 0
        applicable_prior_states = 0
        prior_state_count = 0
        goal_pruned_particle_count = 0
        goal_pruned_mass = 0.0

        loop_start = time.perf_counter()
        self._belief_debug(f"predict_update_loop_start label={debug_label}")
        prior_states = self._iter_indexed_joint_prior_states(belief)
        for prior_index, (prior_state_bits, prior_probability) in enumerate(prior_states, start=1):
            prior_state_count = prior_index
            action_start = time.perf_counter()
            transition_outcomes = self._enumerate_action_outcomes_bits(
                action,
                prior_state_bits,
                effect_bucket=effect_bucket,
            )
            action_outcome_seconds += time.perf_counter() - action_start
            if not transition_outcomes:
                if self.belief_update_debug and (prior_index <= 3 or prior_index % 1000 == 0):
                    self._belief_debug(
                        f"prior_progress label={debug_label} index={prior_index} "
                        f"applicable={applicable_prior_states} outcomes={total_transition_outcomes} "
                        "skipped_no_transition=1"
                    )
                continue
            applicable_prior_states += 1
            applicable_mass += prior_probability
            for next_state_bits, transition_probability in transition_outcomes:
                if transition_probability <= 0.0:
                    continue
                total_transition_outcomes += 1
                likelihood_start = time.perf_counter()
                if skip_observation_update:
                    likelihood = 1.0
                else:
                    likelihood = self._compute_observation_likelihood(
                        action_id,
                        observation_bits,
                        observation_mask,
                        next_state_bits,
                        profile=observation_profile,
                    )
                observation_likelihood_seconds += time.perf_counter() - likelihood_start
                if likelihood <= 0.0:
                    continue
                posterior_weight = prior_probability * transition_probability * likelihood
                if posterior_weight <= 0.0:
                    continue
                if should_prune_goal_particles and self._state_satisfies_raw_goal_bits(next_state_bits):
                    goal_pruned_particle_count += 1
                    goal_pruned_mass += posterior_weight
                    continue
                normalization += posterior_weight
                aggregation_start = time.perf_counter()
                posterior_state_weights[next_state_bits] = (
                    posterior_state_weights.get(next_state_bits, 0.0) + posterior_weight
                )
                posterior_aggregation_seconds += time.perf_counter() - aggregation_start
            if self.belief_update_debug and (prior_index <= 3 or prior_index % 1000 == 0):
                self._belief_debug(
                    f"prior_progress label={debug_label} index={prior_index} "
                    f"applicable={applicable_prior_states} outcomes={total_transition_outcomes} "
                    f"posterior_states={len(posterior_state_weights)}"
                )
        loop_seconds = time.perf_counter() - loop_start
        self._belief_debug(
            "predict_update_loop_done "
            f"label={debug_label} seconds={loop_seconds:.6f} "
            f"applicable={applicable_prior_states} outcomes={total_transition_outcomes} "
            f"posterior_states={len(posterior_state_weights)}"
        )
        return {
            "posterior_state_weights": posterior_state_weights,
            "normalization": normalization,
            "applicable_mass": applicable_mass,
            "observation_profile": observation_profile,
            "action_outcome_seconds": action_outcome_seconds,
            "observation_likelihood_seconds": observation_likelihood_seconds,
            "posterior_aggregation_seconds": posterior_aggregation_seconds,
            "total_transition_outcomes": total_transition_outcomes,
            "applicable_prior_states": applicable_prior_states,
            "prior_state_count": prior_state_count,
            "goal_pruned_particle_count": goal_pruned_particle_count,
            "goal_pruned_mass": goal_pruned_mass,
            "loop_seconds": loop_seconds,
        }

    def _force_predicate_value_on_particles(
        self,
        state_weights: dict[int, float],
        predicate: Predicate,
        value: bool,
    ) -> dict[int, float]:
        predicate_index = self._bitwise_layout.predicate_index.get(predicate)
        if predicate_index is None:
            return dict(state_weights)
        predicate_mask = 1 << predicate_index
        forced_state_weights: dict[int, float] = {}
        for state_bits, probability in state_weights.items():
            if probability <= 0.0:
                continue
            forced_bits = (
                state_bits | predicate_mask
                if value
                else state_bits & ~predicate_mask
            )
            forced_state_weights[forced_bits] = (
                forced_state_weights.get(forced_bits, 0.0) + probability
            )
        return forced_state_weights

    def applicable_observables(
        self,
        action: Action | int,
        *,
        effect_bucket: EffectBucket | None = None,
    ) -> list:
        """Return observables that may be produced after applying ``action``.

        The result is conditioned on the current belief and the selected
        effect bucket/variant, and excludes ``obs-nothing``.
        """

        if self._is_no_feasible_action(action):
            return []

        action_id = (
            int(action)
            if isinstance(action, int)
            else self._bitwise_layout.action_index[action]
        )
        observable_bit_indices: set[int] = set()
        distributions = getattr(
            self.bitwise_pomdp_model,
            "observation_rule_distributions",
            [],
        )
        for prior_state_bits, prior_probability in self.current_belief.particles:
            if prior_probability <= 0.0:
                continue
            transition_outcomes = self._enumerate_action_outcomes_bits(
                action if not isinstance(action, int) else self.pomdp_model.actions[action_id],
                prior_state_bits,
                effect_bucket=effect_bucket,
            )
            for next_state_bits, transition_probability in transition_outcomes:
                if transition_probability <= 0.0:
                    continue
                cache_key = (action_id, next_state_bits)
                cached_bits = self._applicable_observable_bits_cache.get(cache_key)
                if cached_bits is None:
                    local_bit_indices: set[int] = set()
                    candidate_rule_indices = self._candidate_observation_rule_indices(
                        action_id,
                        next_state_bits,
                        0,
                    )
                    for rule_index in candidate_rule_indices:
                        if rule_index in self._init_observation_rule_indices:
                            continue
                        if not self._check_observation_rule_condition_cached(
                            rule_index,
                            next_state_bits,
                            action_id,
                        ):
                            continue
                        branches = distributions[rule_index]
                        for branch in branches:
                            mask = int(branch.observation_mask)
                            while mask:
                                observable_mask = mask & -mask
                                mask &= mask - 1
                                local_bit_indices.add(observable_mask.bit_length() - 1)
                    cached_bits = tuple(sorted(local_bit_indices))
                    self._applicable_observable_bits_cache[cache_key] = cached_bits
                observable_bit_indices.update(cached_bits)

        observables = []
        for bit_index in sorted(observable_bit_indices):
            observable = self._bitwise_layout.observables[bit_index]
            if observable.name == "obs-nothing":
                continue
            observables.append(observable)
        return observables

    def _indexed_case_bits(self, factor_scope: list[int], true_indices: list[int]) -> int:
        bits = 0
        true_set = set(true_indices)
        for index in factor_scope:
            if index in true_set:
                bits |= 1 << index
        return bits

    def _iter_indexed_joint_prior_states(
        self,
        belief: IndexedParticleBelief,
    ) -> Iterator[tuple[int, float]]:
        yield from belief.particles

    def _state_key(self, state: dict) -> tuple[tuple[object, bool], ...]:
        return tuple(
            sorted(
                state.items(),
                key=lambda item: item[0].to_pddl_str(),
            )
        )

    def _enumerate_joint_prior_states(
        self,
        belief: FactorizedBelief,
    ) -> list[tuple[dict, float]]:
        if not belief.factors:
            return [(self._build_state_from_case_indices(belief, ()), 1.0)]

        case_ranges = [range(len(factor.cases)) for factor in belief.factors]
        joint_states: list[tuple[dict, float]] = []
        for case_indices in product(*case_ranges):
            prior_probability = 1.0
            for factor_index, case_index in enumerate(case_indices):
                prior_probability *= belief.factors[factor_index].cases[case_index][0]
            if prior_probability <= 0.0:
                continue
            joint_states.append(
                (self._build_state_from_case_indices(belief, case_indices), prior_probability)
            )
        return joint_states

    def _state_satisfies_raw_goal_bits(self, state_bits: int) -> bool:
        raw_goal_checker = getattr(self.bitwise_pomdp_model, "raw_is_goal", None)
        if callable(raw_goal_checker):
            return bool(raw_goal_checker(state_bits))
        return bool(self.bitwise_pomdp_model.is_goal(state_bits))

    def _enumerate_action_outcomes_bits(
        self,
        action: Action,
        state_bits: int,
        effect_bucket: EffectBucket | None = None,
    ) -> list[tuple[int, float]]:
        action_id = self._bitwise_layout.action_index[action]
        cache_key = (
            action_id,
            state_bits,
            effect_bucket.bucket_name if effect_bucket is not None else None,
            effect_bucket.success if effect_bucket is not None else None,
            effect_bucket.branch_index if effect_bucket is not None else None,
            effect_bucket.variant_rank if effect_bucket is not None else None,
        )
        cached = self._action_outcome_bits_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        if effect_bucket is not None:
            state_bucket_matcher = getattr(self.bitwise_pomdp_model, "effect_bucket_matches_state", None)
            if callable(state_bucket_matcher) and not bool(
                state_bucket_matcher(action_id, state_bits, effect_bucket)
            ):
                self._action_outcome_bits_cache[cache_key] = []
                return []

        if not self.bitwise_pomdp_model.check_action_precondition(action_id, state_bits):
            self._action_outcome_bits_cache[cache_key] = []
            return []

        class _BranchRequest(Exception):
            def __init__(self, weights: list[float]) -> None:
                self.weights = weights

        outcomes_by_bits: dict[int, float] = {}

        def _run_with_choices(prefix: list[int]) -> tuple[str, object]:
            original_sampler = getattr(self.bitwise_pomdp_model, "_sample_branch_index", None)
            original_effect_sampler = getattr(self.bitwise_pomdp_model, "_select_effect_branch_index", None)
            depth = 0

            def _deterministic_sampler(weights: list[float]) -> int:
                nonlocal depth
                cleaned = [max(weight, 0.0) for weight in weights]
                if depth < len(prefix):
                    branch_index = prefix[depth]
                    depth += 1
                    return branch_index
                raise _BranchRequest(cleaned)

            def _deterministic_effect_sampler(
                *,
                action: int,
                weights: list[float],
                bucket_names: list[str | None] | None = None,
                successes: list[bool | None] | None = None,
                branch_indices: list[int | None] | None = None,
                variant_ranks: list[int | None] | None = None,
                record_sample: bool = True,
            ) -> int:
                del action, record_sample
                nonlocal depth
                cleaned = [max(weight, 0.0) for weight in weights]
                if effect_bucket is not None:
                    for index, weight in enumerate(list(cleaned)):
                        if weight <= 0.0:
                            continue
                        if (
                            effect_bucket.bucket_name is not None
                            and bucket_names is not None
                            and bucket_names[index] != effect_bucket.bucket_name
                        ):
                            cleaned[index] = 0.0
                            continue
                        if (
                            effect_bucket.success is not None
                            and successes is not None
                            and successes[index] != effect_bucket.success
                        ):
                            cleaned[index] = 0.0
                            continue
                        candidate_branch_index = (
                            branch_indices[index]
                            if branch_indices is not None
                            else index
                        )
                        if (
                            effect_bucket.branch_index is not None
                            and candidate_branch_index != effect_bucket.branch_index
                        ):
                            cleaned[index] = 0.0
                            continue
                        candidate_variant_rank = (
                            variant_ranks[index]
                            if variant_ranks is not None
                            else None
                        )
                        if (
                            effect_bucket.variant_rank is not None
                            and candidate_variant_rank != effect_bucket.variant_rank
                        ):
                            cleaned[index] = 0.0
                if depth < len(prefix):
                    branch_index = prefix[depth]
                    depth += 1
                    return branch_index
                raise _BranchRequest(cleaned)

            setattr(self.bitwise_pomdp_model, "_sample_branch_index", _deterministic_sampler)
            setattr(self.bitwise_pomdp_model, "_select_effect_branch_index", _deterministic_effect_sampler)
            try:
                next_state_bits, _reward = self.bitwise_pomdp_model.forward_action(action_id, state_bits)
                return "done", next_state_bits
            except _BranchRequest as branch_request:
                return "branch", branch_request.weights
            finally:
                if original_sampler is not None:
                    setattr(self.bitwise_pomdp_model, "_sample_branch_index", original_sampler)
                if original_effect_sampler is not None:
                    setattr(self.bitwise_pomdp_model, "_select_effect_branch_index", original_effect_sampler)

        def _enumerate(prefix: list[int], path_probability: float) -> None:
            status, payload = _run_with_choices(prefix)
            if status == "done":
                next_state_bits = cast(int, payload)
                outcomes_by_bits[next_state_bits] = outcomes_by_bits.get(next_state_bits, 0.0) + path_probability
                return

            weights = cast(list[float], payload)
            total = sum(weights)
            if total <= 0.0:
                return
            for branch_index, weight in enumerate(weights):
                if weight <= 0.0:
                    continue
                _enumerate(prefix + [branch_index], path_probability * (weight / total))

        _enumerate([], 1.0)
        outcomes = [(bits, probability) for bits, probability in outcomes_by_bits.items() if probability > 0.0]
        self._action_outcome_bits_cache[cache_key] = list(outcomes)
        return outcomes

    def _enumerate_action_outcomes_bits_with_rewards(
        self,
        action: Action,
        state_bits: int,
        effect_bucket: EffectBucket | None = None,
    ) -> list[tuple[int, float, float]]:
        action_id = self._bitwise_layout.action_index[action]
        cache_key = (
            action_id,
            state_bits,
            effect_bucket.bucket_name if effect_bucket is not None else None,
            effect_bucket.success if effect_bucket is not None else None,
            effect_bucket.branch_index if effect_bucket is not None else None,
            effect_bucket.variant_rank if effect_bucket is not None else None,
        )
        cached = self._action_outcome_bits_with_reward_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        if effect_bucket is not None:
            state_bucket_matcher = getattr(self.bitwise_pomdp_model, "effect_bucket_matches_state", None)
            if callable(state_bucket_matcher) and not bool(
                state_bucket_matcher(action_id, state_bits, effect_bucket)
            ):
                self._action_outcome_bits_with_reward_cache[cache_key] = []
                return []

        if not self.bitwise_pomdp_model.check_action_precondition(action_id, state_bits):
            self._action_outcome_bits_with_reward_cache[cache_key] = []
            return []

        class _BranchRequest(Exception):
            def __init__(self, weights: list[float]) -> None:
                self.weights = weights

        outcomes_by_key: dict[tuple[int, float], float] = {}

        def _run_with_choices(prefix: list[int]) -> tuple[str, object]:
            original_sampler = getattr(self.bitwise_pomdp_model, "_sample_branch_index", None)
            original_effect_sampler = getattr(self.bitwise_pomdp_model, "_select_effect_branch_index", None)
            depth = 0

            def _deterministic_sampler(weights: list[float]) -> int:
                nonlocal depth
                cleaned = [max(weight, 0.0) for weight in weights]
                if depth < len(prefix):
                    branch_index = prefix[depth]
                    depth += 1
                    return branch_index
                raise _BranchRequest(cleaned)

            def _deterministic_effect_sampler(
                *,
                action: int,
                weights: list[float],
                bucket_names: list[str | None] | None = None,
                successes: list[bool | None] | None = None,
                branch_indices: list[int | None] | None = None,
                variant_ranks: list[int | None] | None = None,
                record_sample: bool = True,
            ) -> int:
                del action, record_sample
                nonlocal depth
                cleaned = [max(weight, 0.0) for weight in weights]
                if effect_bucket is not None:
                    for index, weight in enumerate(list(cleaned)):
                        if weight <= 0.0:
                            continue
                        if (
                            effect_bucket.bucket_name is not None
                            and bucket_names is not None
                            and bucket_names[index] != effect_bucket.bucket_name
                        ):
                            cleaned[index] = 0.0
                            continue
                        if (
                            effect_bucket.success is not None
                            and successes is not None
                            and successes[index] != effect_bucket.success
                        ):
                            cleaned[index] = 0.0
                            continue
                        candidate_branch_index = (
                            branch_indices[index]
                            if branch_indices is not None
                            else index
                        )
                        if (
                            effect_bucket.branch_index is not None
                            and candidate_branch_index != effect_bucket.branch_index
                        ):
                            cleaned[index] = 0.0
                            continue
                        candidate_variant_rank = (
                            variant_ranks[index]
                            if variant_ranks is not None
                            else None
                        )
                        if (
                            effect_bucket.variant_rank is not None
                            and candidate_variant_rank != effect_bucket.variant_rank
                        ):
                            cleaned[index] = 0.0
                if depth < len(prefix):
                    branch_index = prefix[depth]
                    depth += 1
                    return branch_index
                raise _BranchRequest(cleaned)

            setattr(self.bitwise_pomdp_model, "_sample_branch_index", _deterministic_sampler)
            setattr(self.bitwise_pomdp_model, "_select_effect_branch_index", _deterministic_effect_sampler)
            try:
                next_state_bits, reward = self.bitwise_pomdp_model.forward_action(action_id, state_bits)
                return "done", (next_state_bits, float(reward))
            except _BranchRequest as branch_request:
                return "branch", branch_request.weights
            finally:
                if original_sampler is not None:
                    setattr(self.bitwise_pomdp_model, "_sample_branch_index", original_sampler)
                if original_effect_sampler is not None:
                    setattr(self.bitwise_pomdp_model, "_select_effect_branch_index", original_effect_sampler)

        def _enumerate(prefix: list[int], path_probability: float) -> None:
            status, payload = _run_with_choices(prefix)
            if status == "done":
                next_state_bits, reward = cast(tuple[int, float], payload)
                outcome_key = (next_state_bits, reward)
                outcomes_by_key[outcome_key] = outcomes_by_key.get(outcome_key, 0.0) + path_probability
                return

            weights = cast(list[float], payload)
            total = sum(weights)
            if total <= 0.0:
                return
            for branch_index, weight in enumerate(weights):
                if weight <= 0.0:
                    continue
                _enumerate(prefix + [branch_index], path_probability * (weight / total))

        _enumerate([], 1.0)
        outcomes = [
            (bits, reward, probability)
            for (bits, reward), probability in outcomes_by_key.items()
            if probability > 0.0
        ]
        self._action_outcome_bits_with_reward_cache[cache_key] = list(outcomes)
        return outcomes

    def _enumerate_action_outcomes(
        self,
        action: Action,
        state: dict,
        effect_bucket: EffectBucket | None = None,
    ) -> list[tuple[dict, float]]:
        if effect_bucket is not None:
            state_bucket_matcher = getattr(self.pomdp_model, "effect_bucket_matches_state", None)
            if callable(state_bucket_matcher) and not bool(
                state_bucket_matcher(action, state, effect_bucket)
            ):
                return []
        if not self.pomdp_model.check_action_precondition(action, state):
            return []

        class _BranchRequest(Exception):
            def __init__(self, weights: list[float]) -> None:
                self.weights = weights

        outcomes_by_key: dict[tuple[tuple[object, bool], ...], tuple[dict, float]] = {}

        def _run_with_choices(prefix: list[int]) -> tuple[str, object]:
            original_sampler = getattr(self.pomdp_model, "_sample_branch_index", None)
            original_effect_sampler = getattr(self.pomdp_model, "_select_effect_branch_index", None)
            depth = 0

            def _deterministic_sampler(weights: list[float]) -> int:
                nonlocal depth
                cleaned = [max(weight, 0.0) for weight in weights]
                if depth < len(prefix):
                    branch_index = prefix[depth]
                    depth += 1
                    return branch_index
                raise _BranchRequest(cleaned)

            def _deterministic_effect_sampler(
                *,
                action: Action,
                weights: list[float],
                bucket_names: list[str | None] | None = None,
                successes: list[bool | None] | None = None,
                branch_indices: list[int | None] | None = None,
                variant_ranks: list[int | None] | None = None,
                record_sample: bool = True,
            ) -> int:
                del action, record_sample
                nonlocal depth
                cleaned = [max(weight, 0.0) for weight in weights]
                if effect_bucket is not None:
                    for index, weight in enumerate(list(cleaned)):
                        if weight <= 0.0:
                            continue
                        if (
                            effect_bucket.bucket_name is not None
                            and bucket_names is not None
                            and bucket_names[index] != effect_bucket.bucket_name
                        ):
                            cleaned[index] = 0.0
                            continue
                        if (
                            effect_bucket.success is not None
                            and successes is not None
                            and successes[index] != effect_bucket.success
                        ):
                            cleaned[index] = 0.0
                            continue
                        candidate_branch_index = (
                            branch_indices[index]
                            if branch_indices is not None
                            else index
                        )
                        if (
                            effect_bucket.branch_index is not None
                            and candidate_branch_index != effect_bucket.branch_index
                        ):
                            cleaned[index] = 0.0
                            continue
                        candidate_variant_rank = (
                            variant_ranks[index]
                            if variant_ranks is not None
                            else None
                        )
                        if (
                            effect_bucket.variant_rank is not None
                            and candidate_variant_rank != effect_bucket.variant_rank
                        ):
                            cleaned[index] = 0.0
                if depth < len(prefix):
                    branch_index = prefix[depth]
                    depth += 1
                    return branch_index
                raise _BranchRequest(cleaned)

            setattr(self.pomdp_model, "_sample_branch_index", _deterministic_sampler)
            setattr(self.pomdp_model, "_select_effect_branch_index", _deterministic_effect_sampler)
            try:
                next_state, _reward = self.pomdp_model.forward_action(action, dict(state))
                return "done", next_state
            except _BranchRequest as branch_request:
                return "branch", branch_request.weights
            finally:
                if original_sampler is not None:
                    setattr(self.pomdp_model, "_sample_branch_index", original_sampler)
                if original_effect_sampler is not None:
                    setattr(self.pomdp_model, "_select_effect_branch_index", original_effect_sampler)

        def _enumerate(prefix: list[int], path_probability: float) -> None:
            status, payload = _run_with_choices(prefix)
            if status == "done":
                next_state = cast(dict, payload)
                state_key = self._state_key(next_state)
                existing_state, existing_probability = outcomes_by_key.get(
                    state_key,
                    (next_state, 0.0),
                )
                outcomes_by_key[state_key] = (
                    existing_state,
                    existing_probability + path_probability,
                )
                return

            weights = cast(list[float], payload)
            total = sum(weights)
            if total <= 0.0:
                return
            for branch_index, weight in enumerate(weights):
                if weight <= 0.0:
                    continue
                _enumerate(prefix + [branch_index], path_probability * (weight / total))

        _enumerate([], 1.0)
        return [
            (dict(next_state), probability)
            for next_state, probability in outcomes_by_key.values()
            if probability > 0.0
        ]

    def _rebuild_particle_belief_from_bits_distribution(
        self,
        state_distribution: list[tuple[int, float]],
        grounded_predicates_count: int,
    ) -> IndexedParticleBelief:
        state_weights: dict[int, float] = {}
        for state_bits, probability in state_distribution:
            if probability <= 0.0:
                continue
            state_weights[state_bits] = state_weights.get(state_bits, 0.0) + probability

        rebuilt = IndexedParticleBelief(
            grounded_predicates_count=grounded_predicates_count,
            particles=[
                (state_bits, probability)
                for state_bits, probability in sorted(state_weights.items(), key=lambda item: item[0])
                if probability > 0.0
            ],
        )
        rebuilt.validate()
        return rebuilt

    def estimate_expected_reward(
        self,
        action: Action | int,
        *,
        effect_bucket: EffectBucket | None = None,
    ) -> float:
        """Estimate the one-step reward under the current belief.

        When ``effect_bucket`` is provided, the estimate is conditioned on the
        human/simulator feedback selecting that top-level effect family.
        """

        if self._is_no_feasible_action(action):
            return 0.0

        normalization = 0.0
        reward_mass = 0.0
        for prior_state_bits, prior_probability in self.current_belief.particles:
            transition_outcomes = self._enumerate_action_outcomes_bits_with_rewards(
                action,
                prior_state_bits,
                effect_bucket=effect_bucket,
            )
            for _next_state_bits, reward, transition_probability in transition_outcomes:
                if transition_probability <= 0.0:
                    continue
                weight = prior_probability * transition_probability
                if weight <= 0.0:
                    continue
                normalization += weight
                reward_mass += weight * reward
        if normalization <= 0.0:
            raise ValueError("Action is not applicable under the current belief.")
        return reward_mass / normalization

    def _rebuild_factorized_belief_from_state_distribution(
        self,
        state_distribution: list[tuple[dict, float]],
        template_belief: FactorizedBelief,
    ) -> FactorizedBelief:
        covered_predicates = {
            predicate for factor in template_belief.factors for predicate in factor.scope
        }

        updated_factors: list[BeliefFactor] = []
        for factor in template_belief.factors:
            pattern_masses: dict[tuple[bool, ...], float] = {}
            for state, probability in state_distribution:
                pattern = tuple(bool(state.get(predicate, False)) for predicate in factor.scope)
                pattern_masses[pattern] = pattern_masses.get(pattern, 0.0) + probability

            cases = []
            for pattern, probability in pattern_masses.items():
                if probability <= 0.0:
                    continue
                true_predicates = [
                    predicate for predicate, is_true in zip(factor.scope, pattern) if is_true
                ]
                cases.append((probability, true_predicates))

            updated_factors.append(
                BeliefFactor(
                    name=factor.name,
                    scope=list(factor.scope),
                    cases=cases,
                )
            )

        known_true = []
        known_false = []
        singleton_counter = 0
        all_predicates = list(self.pomdp_model.predicates)
        for predicate in all_predicates:
            if predicate in covered_predicates:
                continue
            true_probability = 0.0
            for state, probability in state_distribution:
                if state.get(predicate, False):
                    true_probability += probability
            false_probability = 1.0 - true_probability
            if true_probability >= 1.0 - 1e-9:
                known_true.append(predicate)
            elif true_probability <= 1e-9:
                known_false.append(predicate)
            else:
                updated_factors.append(
                    BeliefFactor(
                        name=f"singleton_factor_{singleton_counter}",
                        scope=[predicate],
                        cases=[
                            (true_probability, [predicate]),
                            (false_probability, []),
                        ],
                    )
                )
                singleton_counter += 1

        rebuilt = FactorizedBelief(
            known_true=known_true,
            known_false=known_false,
            factors=updated_factors,
        )
        rebuilt.validate()
        return rebuilt

    def _no_feasible_action_id(self) -> int:
        return len(self.pomdp_model.actions)

    def _is_no_feasible_action(self, action: Action | int) -> bool:
        return isinstance(action, int) and action == self._no_feasible_action_id()

    def belief_update(
        self,
        action: Action | int,
        observation: ObservationEntry,
        effect_bucket: EffectBucket | None = None,
        simulator_reported_not_goal: bool = False,
        skip_observation_update: bool = False,
    ) -> FactorizedBelief:
        """Predict with one semantic action, then update with one observation."""
        if self._is_no_feasible_action(action):
            self.current_step += 1
            self.last_belief_update_profile = {
                "no_feasible_action": True,
                "total_seconds": 0.0,
            }
            return self.current_belief

        total_start = time.perf_counter()
        self._belief_debug("start")
        belief = self.current_belief
        indexed_seconds = 0.0
        self._belief_debug("indexed_ready seconds=0.000000")
        obs_bits_start = time.perf_counter()
        observation_bits, observation_mask = self._observation_to_bits_and_mask(observation)
        recorded_observation_bits = observation_bits
        if skip_observation_update:
            observation_bits = 0
            observation_mask = 0
        obs_bits_seconds = time.perf_counter() - obs_bits_start
        self._belief_debug(
            f"observation_bits_ready seconds={obs_bits_seconds:.6f} "
            f"bits={observation_bits} mask={observation_mask}"
        )
        observed_predicates = [
            predicate
            for predicate in (
                self._resolve_predicate_from_observable(observable)
                for observable in observation.keys()
            )
            if predicate is not None
        ]
        observable_names_by_mask: dict[int, str] = {}
        remaining_observation_mask = observation_mask
        while remaining_observation_mask:
            observable_mask = remaining_observation_mask & -remaining_observation_mask
            remaining_observation_mask &= remaining_observation_mask - 1
            bit_index = observable_mask.bit_length() - 1
            observable_names_by_mask[observable_mask] = self._bitwise_layout.observables[
                bit_index
            ].to_pddl_str()
        pre_update_marginals: dict[str, float] = {}
        if self.belief_update_debug and observed_predicates:
            for predicate in observed_predicates:
                marginal = self._predicate_marginal_probability(belief, predicate)
                if marginal is None:
                    continue
                key = predicate.to_pddl_str()
                pre_update_marginals[key] = marginal
                self._belief_debug(
                    f"observable_target_before predicate={key} true_probability={marginal:.6f}"
                )

        enumerate_start = time.perf_counter()
        prior_states = self._iter_indexed_joint_prior_states(belief)
        enumerate_seconds = time.perf_counter() - enumerate_start
        self._belief_debug(
            f"prior_state_iterator_ready seconds={enumerate_seconds:.6f}"
        )
        enable_report_goal_action = bool(
            getattr(self.pomdp_model, "enable_report_goal_action", False)
        )
        should_prune_goal_particles = (not enable_report_goal_action) or simulator_reported_not_goal
        action_id = self._bitwise_layout.action_index[action]
        del prior_states
        pass_result = self._predict_update_pass(
            belief=belief,
            action=action,
            action_id=action_id,
            observation_bits=observation_bits,
            observation_mask=observation_mask,
            effect_bucket=effect_bucket,
            skip_observation_update=skip_observation_update,
            should_prune_goal_particles=should_prune_goal_particles,
        )
        overridden_observation_masks: list[int] = []
        forced_predicate_assignments: list[tuple[Predicate, bool]] = []
        fallback_applied = False
        fallback_attempt_count = 0
        fallback_selected_bits = observation_bits
        fallback_selected_mask = observation_mask

        normalization = cast(float, pass_result["normalization"])
        if (
            not skip_observation_update
            and normalization <= 0.0
            and observation_mask != 0
        ):
            active_bits = observation_bits
            active_mask = observation_mask
            while normalization <= 0.0 and active_mask != 0:
                recovered = False
                remaining_mask = active_mask
                while remaining_mask:
                    observable_mask = remaining_mask & -remaining_mask
                    remaining_mask &= remaining_mask - 1
                    candidate_mask = active_mask & ~observable_mask
                    candidate_bits = active_bits & candidate_mask
                    fallback_attempt_count += 1
                    candidate_result = self._predict_update_pass(
                        belief=belief,
                        action=action,
                        action_id=action_id,
                        observation_bits=candidate_bits,
                        observation_mask=candidate_mask,
                        effect_bucket=effect_bucket,
                        skip_observation_update=False,
                        should_prune_goal_particles=should_prune_goal_particles,
                        debug_label=f"fallback_drop_{observable_mask}",
                    )
                    candidate_normalization = cast(float, candidate_result["normalization"])
                    if candidate_normalization <= 0.0:
                        continue
                    overridden_observation_masks.append(observable_mask)
                    observable = self._bitwise_layout.observables[
                        observable_mask.bit_length() - 1
                    ]
                    predicate = self._resolve_predicate_from_observable(observable)
                    if predicate is not None:
                        forced_predicate_assignments.append(
                            (predicate, bool(observation_bits & observable_mask))
                        )
                    active_mask = candidate_mask
                    active_bits = candidate_bits
                    pass_result = candidate_result
                    normalization = candidate_normalization
                    fallback_selected_mask = active_mask
                    fallback_selected_bits = active_bits
                    fallback_applied = True
                    recovered = True
                    self._belief_debug(
                        f"fallback_override_observable mask={observable_mask} "
                        f"remaining_mask={active_mask}"
                    )
                    break
                if not recovered:
                    break

        posterior_state_weights = cast(dict[int, float], pass_result["posterior_state_weights"])
        for predicate, value in forced_predicate_assignments:
            posterior_state_weights = self._force_predicate_value_on_particles(
                posterior_state_weights,
                predicate,
                value,
            )
        applicable_mass = cast(float, pass_result["applicable_mass"])
        observation_profile = cast(dict[str, float | int], pass_result["observation_profile"])
        action_outcome_seconds = cast(float, pass_result["action_outcome_seconds"])
        observation_likelihood_seconds = cast(float, pass_result["observation_likelihood_seconds"])
        posterior_aggregation_seconds = cast(float, pass_result["posterior_aggregation_seconds"])
        total_transition_outcomes = cast(int, pass_result["total_transition_outcomes"])
        applicable_prior_states = cast(int, pass_result["applicable_prior_states"])
        prior_state_count = cast(int, pass_result["prior_state_count"])
        goal_pruned_particle_count = cast(int, pass_result["goal_pruned_particle_count"])
        goal_pruned_mass = cast(float, pass_result["goal_pruned_mass"])
        loop_seconds = cast(float, pass_result["loop_seconds"])

        if applicable_mass <= 0.0:
            raise ValueError("Action is not applicable under the current belief.")
        if normalization <= 0.0:
            if should_prune_goal_particles and goal_pruned_mass > 0.0:
                raise ValueError(
                    "All posterior particles satisfy the goal and were pruned from belief update."
                )
            raise ValueError("Observation has zero likelihood after action prediction.")

        normalized_state_distribution = [
            (state_bits, probability / normalization)
            for state_bits, probability in posterior_state_weights.items()
            if probability > 0.0
        ]

        rebuild_start = time.perf_counter()
        self._belief_debug(
            f"rebuild_start posterior_state_count={len(normalized_state_distribution)}"
        )
        updated_belief = self._rebuild_particle_belief_from_bits_distribution(
            normalized_state_distribution,
            belief.grounded_predicates_count,
        )
        rebuild_seconds = time.perf_counter() - rebuild_start
        self.current_belief = updated_belief
        self.current_step += 1
        total_seconds = time.perf_counter() - total_start
        self.last_belief_update_profile = {
            "indexed_seconds": indexed_seconds,
            "observation_bits_seconds": obs_bits_seconds,
            "enumerate_prior_states_seconds": enumerate_seconds,
            "predict_update_loop_seconds": loop_seconds,
            "action_outcomes_seconds": action_outcome_seconds,
            "observation_likelihood_seconds": observation_likelihood_seconds,
            "observation_rule_count": observation_profile["rule_count"],
            "observation_candidate_rule_count": observation_profile["candidate_rule_count"],
            "observation_active_rule_count": observation_profile["active_rule_count"],
            "observation_branch_count_checked": observation_profile["branch_count_checked"],
            "observation_support_mask_seconds": observation_profile["support_mask_seconds"],
            "observation_condition_check_seconds": observation_profile["condition_check_seconds"],
            "observation_branch_match_seconds": observation_profile["branch_match_seconds"],
            "posterior_aggregation_seconds": posterior_aggregation_seconds,
            "rebuild_belief_seconds": rebuild_seconds,
            "total_seconds": total_seconds,
            "prior_state_count": prior_state_count,
            "applicable_prior_state_count": applicable_prior_states,
            "transition_outcome_count": total_transition_outcomes,
            "posterior_state_count": len(normalized_state_distribution),
            "action_outcome_cache_size": len(self._action_outcome_bits_cache),
            "semantic_state_cache_size": len(self._semantic_state_cache),
            "goal_prune_applied": should_prune_goal_particles,
            "goal_pruned_particle_count": goal_pruned_particle_count,
            "goal_pruned_mass": goal_pruned_mass,
            "skip_observation_update": skip_observation_update,
            "observation_fallback_applied": fallback_applied,
            "observation_fallback_attempt_count": fallback_attempt_count,
            "overridden_observation_masks": overridden_observation_masks,
            "overridden_observations": [
                observable_names_by_mask[mask]
                for mask in overridden_observation_masks
                if mask in observable_names_by_mask
            ],
            "forced_predicate_assignments": [
                {
                    "predicate": predicate.to_pddl_str(),
                    "value": value,
                }
                for predicate, value in forced_predicate_assignments
            ],
            "effective_observation_bits": fallback_selected_bits,
            "effective_observation_mask": fallback_selected_mask,
            "pre_update_observed_predicate_marginals": pre_update_marginals,
        }
        if self.belief_update_debug and observed_predicates:
            post_update_marginals: dict[str, float] = {}
            for predicate in observed_predicates:
                marginal = self._predicate_marginal_probability(updated_belief, predicate)
                if marginal is None:
                    continue
                key = predicate.to_pddl_str()
                post_update_marginals[key] = marginal
                self._belief_debug(
                    f"observable_target_after predicate={key} true_probability={marginal:.6f}"
                )
            self.last_belief_update_profile["post_update_observed_predicate_marginals"] = (
                post_update_marginals
            )
        if self.belief_update_debug:
            print("[belief_update] summary", flush=True)
            for key, value in self.last_belief_update_profile.items():
                if isinstance(value, float):
                    print(f"  {key}={value:.6f}", flush=True)
                else:
                    print(f"  {key}={value}", flush=True)
        self._last_runtime_observation_bits = int(recorded_observation_bits)
        self._has_last_runtime_observation = True
        return deepcopy(self.current_belief)

    def search(self) -> Action | int:
        """Call DESPOT and return the next best semantic action."""
        if self._despot_planner_instance is None:
            raise RuntimeError("DESPOT planner shared library is not built.")
        step_seed = self._step_seed()
        despot_root_seed = self._despot_root_seed()
        if step_seed is not None:
            self.pomdp_model._rng.seed(int(step_seed))
            self.bitwise_pomdp_model._rng.seed(int(step_seed))
        if despot_root_seed is not None and hasattr(self._despot_planner_instance, "SetRootSeed"):
            self._despot_planner_instance.SetRootSeed(int(despot_root_seed))
        write_runtime_belief_json(
            self.current_belief,
            self._runtime_belief_json_path,
            last_observation_bits=self._last_runtime_observation_bits,
            has_last_observation=self._has_last_runtime_observation,
        )
        self._despot_planner_instance.SetBeliefJsonPath(str(self._runtime_belief_json_path))
        action_index = int(self._despot_planner_instance.MakePlanning())
        if action_index == self._no_feasible_action_id():
            return action_index
        if action_index < 0 or action_index >= len(self.pomdp_model.actions):
            raise ValueError(f"DESPOT returned invalid action index: {action_index}")
        return self.pomdp_model.actions[action_index]

    def reload_probabilities_from_domain(
        self,
        domain_text: str,
        problem_text: str,
        default_policy_text: str | None = None,
    ) -> None:
        """Hot-reload probabilities without recompiling C++."""
        registry = extract_updated_probabilities(
            domain_text,
            problem_text,
            default_policy_text,
            self.bitwise_pomdp_model,
        )
        if self._despot_planner_instance is not None:
            for action_id, weights in registry.action_weights.items():
                self._despot_planner_instance.UpdateActionWeights(action_id, weights)
            for rule_id, weights in registry.observation_weights.items():
                self._despot_planner_instance.UpdateObsWeights(rule_id, weights)
        update_bitwise_model_probabilities(self.bitwise_pomdp_model, registry)

    def reload_hyperparams_from_yaml(self, yaml_path: str | Path) -> None:
        """Hot-reload hyperparameters from a YAML file without recompiling C++."""
        params = _parse_hyperparams_yaml(Path(yaml_path))
        self.pomdp_model.goal_reward = float(
            params.get("goal_reward", getattr(self.pomdp_model, "goal_reward", 0.0))
        )
        self.bitwise_pomdp_model.goal_reward = float(
            params.get("goal_reward", getattr(self.bitwise_pomdp_model, "goal_reward", 0.0))
        )
        report_goal_failure_penalty = float(
            params.get(
                "report_goal_failure_penalty",
                getattr(self.pomdp_model, "report_goal_failure_penalty", -10.0),
            )
        )
        if hasattr(self.pomdp_model, "report_goal_failure_penalty"):
            self.pomdp_model.report_goal_failure_penalty = report_goal_failure_penalty
        if hasattr(self.bitwise_pomdp_model, "report_goal_failure_penalty"):
            self.bitwise_pomdp_model.report_goal_failure_penalty = report_goal_failure_penalty
        if self._despot_planner_instance is not None and hasattr(
            self._despot_planner_instance, "UpdateHyperparams"
        ):
            self._despot_planner_instance.UpdateHyperparams(
                float(params.get("goal_reward", 100.0)),
                report_goal_failure_penalty,
                float(params.get("action_invalid_penalty", -10.0)),
                float(params.get("action_no_change_penalty", -10.0)),
                float(params.get("dead_end_penalty", -500.0)),
                float(params.get("dp_exploit_prob", 0.9)),
                float(params.get("time_per_move", 1.0)),
                int(params.get("num_scenarios", 512)),
                int(params.get("search_depth", 90)),
                int(params.get("max_policy_sim_len", 20)),
                int(params.get("sim_len", 20)),
                float(params.get("discount", 0.95)),
                float(params.get("pruning_constant", 0.0)),
                float(params.get("xi", 0.95)),
                int(params.get("root_seed", 42)),
                bool(params.get("silence", True)),
            )


def _parse_hyperparams_yaml(yaml_path: Path) -> dict:
    """Parse a simple flat key-value YAML file without requiring pyyaml."""
    params: dict = {}
    text = yaml_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        # Parse typed values
        if value.lower() == "true":
            params[key] = True
        elif value.lower() == "false":
            params[key] = False
        else:
            try:
                if "." in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                params[key] = value
    return params
