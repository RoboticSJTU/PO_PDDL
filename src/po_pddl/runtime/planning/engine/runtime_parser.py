"""Parse POMDPDDL files into ready-to-run runtime objects."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType

from ..base.pomdp_model import POMDPModelBase
from ..bitwise import (
    IndexedParticleBelief,
    POMDPModelBase as BitwisePOMDPModelBase,
    build_bitwise_index_layout,
    build_bitwise_model_from_parsed,
    convert_factorized_belief_to_indexed_particles,
    dump_indexed_particle_belief_json,
    emit_bitwise_python_model_package,
)
from ..codegen import (
    build_codegen_plan_from_parsed,
    emit_explicit_python_model_package,
)
from ..data_structures import DespotCppModelCode, FactorizedBelief, Predicate, StateEntry
from ..parser import (
    ParsedDefaultPolicy,
    ParsedDomain,
    ParsedProblem,
    parse_default_policy,
    parse_domain,
    parse_problem,
)
from .despot_runtime_support import (
    configure_and_build_despot_cpp_project,
    convert_bitwise_model_to_despot_cpp_model_code,
    find_existing_despot_shared_library,
    write_despot_cpp_model_code,
)
from .bitwise_pomdp_world import BitwisePOMDPWorld
from .pomdp_planner import POMDPPlanner, _parse_hyperparams_yaml
from .probability_registry import ProbabilityRegistry, build_probability_registry
from .pomdp_world import POMDPWorld
from .report_goal_action import (
    ReportGoalActionBitwiseModel,
    ReportGoalActionSemanticModel,
    build_report_goal_planner_metadata_model,
)
from .semantic_metadata_model import RuntimeSemanticMetadataModel
from .paths import despot_root, hyperparams_path

RUNTIME_ARTIFACT_SCHEMA_VERSION = "2026-04-28-runtime-last-action-specialization-v2-observation-anchor-fix"
DEFAULT_HYPERPARAMS_YAML_PATH = hyperparams_path()
DESPOT_CPP_CODEGEN_PATH = Path(__file__).resolve().parents[1] / "bitwise" / "despot_cpp_codegen.py"


def _is_last_action_predicate(predicate: Predicate) -> bool:
    return predicate.name.startswith("last_action_")


@dataclass(frozen=True)
class _SpecializedLastActionPredicate:
    original_name: str
    marker: str
    specialized_name: str
    parameter_types: tuple[tuple[str, str], ...]


def _specialized_last_action_name(original_name: str, marker: str) -> str:
    sanitized = "".join(char if char.isalnum() else "_" for char in marker).strip("_")
    return f"last_action__{original_name}__{sanitized}"


def _flat_typed_vars(parameter_types: tuple[tuple[str, str], ...], *, prefix: str) -> list[str]:
    rendered: list[str] = []
    for index, (_var_name, type_name) in enumerate(parameter_types):
        rendered.extend([f"?{prefix}{index}", "-", type_name])
    return rendered


def _last_action_specialization_key(expr: list) -> tuple[str, str] | None:
    if not expr or not isinstance(expr[0], str):
        return None
    name = expr[0]
    if not name.startswith("last_action_") or len(expr) < 2:
        return None
    marker = expr[1]
    if not isinstance(marker, str) or marker.startswith("?"):
        return None
    return name, marker


def _merge_parameter_types(
    existing: tuple[tuple[str, str], ...],
    new_value: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if not existing:
        return new_value
    merged: list[tuple[str, str]] = []
    for idx, (existing_name, existing_type) in enumerate(existing):
        new_name, new_type = new_value[idx]
        merged_name = existing_name or new_name
        merged_type = existing_type
        if merged_type == "object" and new_type != "object":
            merged_type = new_type
        merged.append((merged_name, merged_type))
    return tuple(merged)


def _collect_specialized_last_action_predicates(
    expr,
    *,
    variable_types: dict[str, str],
    registry: dict[tuple[str, str], _SpecializedLastActionPredicate],
) -> None:
    if not isinstance(expr, list) or not expr:
        return

    specialization_key = _last_action_specialization_key(expr)
    if specialization_key is not None:
        original_name, marker = specialization_key
        parameter_types: tuple[tuple[str, str], ...] = tuple(
            (
                arg if isinstance(arg, str) and arg.startswith("?") else f"?la{index}",
                variable_types.get(arg, "object") if isinstance(arg, str) else "object",
            )
            for index, arg in enumerate(expr[2:])
        )
        current = registry.get((original_name, marker))
        if current is None:
            registry[(original_name, marker)] = _SpecializedLastActionPredicate(
                original_name=original_name,
                marker=marker,
                specialized_name=_specialized_last_action_name(original_name, marker),
                parameter_types=parameter_types,
            )
        else:
            registry[(original_name, marker)] = _SpecializedLastActionPredicate(
                original_name=current.original_name,
                marker=current.marker,
                specialized_name=current.specialized_name,
                parameter_types=_merge_parameter_types(current.parameter_types, parameter_types),
            )

    for item in expr:
        _collect_specialized_last_action_predicates(
            item,
            variable_types=variable_types,
            registry=registry,
        )


def _is_last_action_clear_quantifier(expr: list) -> str | None:
    if not expr or expr[0] not in {"forall", "all"} or len(expr) < 3:
        return None
    body = expr[2]
    if not isinstance(body, list) or len(body) != 2 or body[0] != "not":
        return None
    atom = body[1]
    if not isinstance(atom, list) or not atom:
        return None
    head = atom[0]
    if not isinstance(head, str) or not head.startswith("last_action_"):
        return None
    if len(atom) < 2 or not isinstance(atom[1], str) or not atom[1].startswith("?"):
        return None
    return head


def _transform_last_action_expr(
    expr,
    *,
    registry: dict[tuple[str, str], _SpecializedLastActionPredicate],
) :
    if not isinstance(expr, list) or not expr:
        return expr

    clear_original_name = _is_last_action_clear_quantifier(expr)
    if clear_original_name is not None:
        matching_specs = [
            spec
            for spec in registry.values()
            if spec.original_name == clear_original_name
        ]
        if not matching_specs:
            return ["and"]
        rewritten_terms = []
        for spec in matching_specs:
            fresh_vars = [f"?la{index}" for index, _ in enumerate(spec.parameter_types)]
            atom = [spec.specialized_name, *fresh_vars]
            if spec.parameter_types:
                rewritten_terms.append(
                    [
                        "forall",
                        _flat_typed_vars(spec.parameter_types, prefix="la"),
                        ["not", atom],
                    ]
                )
            else:
                rewritten_terms.append(["not", atom])
        if len(rewritten_terms) == 1:
            return rewritten_terms[0]
        return ["and", *rewritten_terms]

    specialization_key = _last_action_specialization_key(expr)
    if specialization_key is not None:
        spec = registry.get(specialization_key)
        if spec is None:
            return expr
        return [spec.specialized_name, *expr[2:]]

    return [
        _transform_last_action_expr(item, registry=registry)
        for item in expr
    ]


def _transform_last_action_predicate(
    predicate: Predicate,
    *,
    registry: dict[tuple[str, str], _SpecializedLastActionPredicate],
) -> Predicate:
    if not predicate.name.startswith("last_action_") or not predicate.params:
        return predicate
    marker = predicate.params[0]
    if marker.startswith("?"):
        return predicate
    spec = registry.get((predicate.name, marker))
    if spec is None:
        return predicate
    return Predicate(spec.specialized_name, list(predicate.params[1:]))


def _specialize_last_action_domain_and_problem(
    parsed_domain: ParsedDomain,
    parsed_problem: ParsedProblem,
    parsed_default_policy: ParsedDefaultPolicy | None,
) -> tuple[ParsedDomain, ParsedProblem, ParsedDefaultPolicy | None]:
    transformed_domain = deepcopy(parsed_domain)
    transformed_problem = deepcopy(parsed_problem)
    transformed_default_policy = deepcopy(parsed_default_policy) if parsed_default_policy is not None else None

    registry: dict[tuple[str, str], _SpecializedLastActionPredicate] = {}
    for action_schema in transformed_domain.actions:
        variable_types = dict(action_schema.parameter_types)
        _collect_specialized_last_action_predicates(
            action_schema.precondition,
            variable_types=variable_types,
            registry=registry,
        )
        _collect_specialized_last_action_predicates(
            action_schema.effect,
            variable_types=variable_types,
            registry=registry,
        )
    for observation_rule in transformed_domain.observation_rules:
        variable_types = dict(observation_rule.parameter_types)
        _collect_specialized_last_action_predicates(
            observation_rule.condition,
            variable_types=variable_types,
            registry=registry,
        )
    if transformed_default_policy is not None:
        for rule in transformed_default_policy.rules:
            variable_types = dict(rule.parameter_types)
            _collect_specialized_last_action_predicates(
                rule.precondition,
                variable_types=variable_types,
                registry=registry,
            )
            _collect_specialized_last_action_predicates(
                rule.action_expr,
                variable_types=variable_types,
                registry=registry,
            )

    if not registry:
        return transformed_domain, transformed_problem, transformed_default_policy

    transformed_predicates = [
        predicate
        for predicate in transformed_domain.predicates
        if not predicate.name.startswith("last_action_")
    ]
    transformed_predicate_parameter_types = {
        predicate: parameter_types
        for predicate, parameter_types in transformed_domain.predicate_parameter_types.items()
        if not predicate.name.startswith("last_action_")
    }
    for spec in sorted(registry.values(), key=lambda item: item.specialized_name):
        predicate = Predicate(
            spec.specialized_name,
            [var_name for var_name, _ in spec.parameter_types],
        )
        transformed_predicates.append(predicate)
        transformed_predicate_parameter_types[predicate] = list(spec.parameter_types)
    transformed_domain.predicates = transformed_predicates
    transformed_domain.predicate_parameter_types = transformed_predicate_parameter_types

    for action_schema in transformed_domain.actions:
        action_schema.precondition = _transform_last_action_expr(
            action_schema.precondition,
            registry=registry,
        )
        action_schema.effect = _transform_last_action_expr(
            action_schema.effect,
            registry=registry,
        )
    for observation_rule in transformed_domain.observation_rules:
        observation_rule.condition = _transform_last_action_expr(
            observation_rule.condition,
            registry=registry,
        )
        observation_rule.distribution_expr = _transform_last_action_expr(
            observation_rule.distribution_expr,
            registry=registry,
        )
    if transformed_default_policy is not None:
        for rule in transformed_default_policy.rules:
            rule.precondition = _transform_last_action_expr(
                rule.precondition,
                registry=registry,
            )
            rule.action_expr = _transform_last_action_expr(
                rule.action_expr,
                registry=registry,
            )

    transformed_problem.init_state = {
        _transform_last_action_predicate(predicate, registry=registry): value
        for predicate, value in transformed_problem.init_state.items()
    }
    transformed_problem.init_belief = FactorizedBelief(
        known_true=[
            _transform_last_action_predicate(predicate, registry=registry)
            for predicate in transformed_problem.init_belief.known_true
        ],
        known_false=[
            _transform_last_action_predicate(predicate, registry=registry)
            for predicate in transformed_problem.init_belief.known_false
        ],
        factors=[
            factor.__class__(
                name=factor.name,
                scope=[
                    _transform_last_action_predicate(predicate, registry=registry)
                    for predicate in factor.scope
                ],
                cases=[
                    (
                        probability,
                        [
                            _transform_last_action_predicate(predicate, registry=registry)
                            for predicate in true_predicates
                        ],
                    )
                    for probability, true_predicates in factor.cases
                ],
            )
            for factor in transformed_problem.init_belief.factors
        ],
    )
    transformed_problem.goal = _transform_last_action_expr(
        transformed_problem.goal,
        registry=registry,
    )
    return transformed_domain, transformed_problem, transformed_default_policy


def _complete_missing_last_action_false_belief(
    belief: FactorizedBelief,
    grounded_predicates: list[Predicate],
) -> FactorizedBelief:
    covered = set(belief.known_true) | set(belief.known_false)
    for factor in belief.factors:
        covered.update(factor.scope)
    missing_last_action = [
        predicate
        for predicate in grounded_predicates
        if _is_last_action_predicate(predicate) and predicate not in covered
    ]
    if not missing_last_action:
        return belief
    completed = FactorizedBelief(
        known_true=list(belief.known_true),
        known_false=sorted(
            [*belief.known_false, *missing_last_action],
            key=lambda item: item.to_pddl_str(),
        ),
        factors=list(belief.factors),
    )
    completed.validate()
    return completed


@dataclass
class ParsedRuntimeArtifacts:
    """All runtime objects built from one POMDPDDL problem bundle."""

    semantic_pomdp_model: POMDPModelBase
    simulator_pomdp_model: POMDPModelBase
    planner_pomdp_model: POMDPModelBase
    init_belief: IndexedParticleBelief
    init_state: StateEntry
    bitwise_pomdp_model: BitwisePOMDPModelBase
    despot_cpp_model_code: DespotCppModelCode
    pomdp_world: POMDPWorld | BitwisePOMDPWorld
    bitwise_pomdp_world: BitwisePOMDPWorld
    pomdp_planner: POMDPPlanner
    simulator_seed: int
    planner_seed: int
    parsed_domain: ParsedDomain
    parsed_problem: ParsedProblem
    parsed_default_policy: ParsedDefaultPolicy | None
    artifact_root: Path
    explicit_package_dir: Path | None
    bitwise_package_dir: Path | None
    despot_cpp_dir: Path | None
    probability_registry: ProbabilityRegistry | None = None
    reused_explicit_package: bool = False
    reused_bitwise_package: bool = False
    reused_despot_cpp: bool = False


def parse_pomdpddl_runtime_from_files(
    domain_file: str | Path,
    problem_file: str | Path,
    default_policy_file: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    simulator_seed: int | None = None,
    planner_seed: int | None = None,
    gamma: float = 0.95,
    max_step: int = 30,
    skip_despot_build: bool = False,
    use_bitwise_world: bool = False,
    belief_update_debug: bool = False,
    emit_explicit_package: bool = False,
    force_reuse_despot_cpp: bool = False,
    force_recompile_despot_cpp: bool = False,
    enable_report_goal_action: bool | None = None,
    report_goal_failure_penalty: float | None = None,
) -> ParsedRuntimeArtifacts:
    """Parse one POMDPDDL file bundle into world/planner runtime objects."""
    domain_path = Path(domain_file)
    problem_path = Path(problem_file)
    default_policy_path = Path(default_policy_file) if default_policy_file is not None else None
    return parse_pomdpddl_runtime_from_texts(
        domain_path.read_text(encoding="utf-8"),
        problem_path.read_text(encoding="utf-8"),
        default_policy_path.read_text(encoding="utf-8")
        if default_policy_path is not None
        else None,
        output_dir=output_dir,
        simulator_seed=simulator_seed,
        planner_seed=planner_seed,
        gamma=gamma,
        max_step=max_step,
        skip_despot_build=skip_despot_build,
        use_bitwise_world=use_bitwise_world,
        belief_update_debug=belief_update_debug,
        emit_explicit_package=emit_explicit_package,
        force_reuse_despot_cpp=force_reuse_despot_cpp,
        force_recompile_despot_cpp=force_recompile_despot_cpp,
        enable_report_goal_action=enable_report_goal_action,
        report_goal_failure_penalty=report_goal_failure_penalty,
    )


def parse_pomdpddl_runtime_from_texts(
    domain_text: str,
    problem_text: str,
    default_policy_text: str | None = None,
    *,
    output_dir: str | Path | None = None,
    simulator_seed: int | None = None,
    planner_seed: int | None = None,
    gamma: float = 0.95,
    max_step: int = 30,
    skip_despot_build: bool = False,
    use_bitwise_world: bool = False,
    belief_update_debug: bool = False,
    emit_explicit_package: bool = False,
    force_reuse_despot_cpp: bool = False,
    force_recompile_despot_cpp: bool = False,
    enable_report_goal_action: bool | None = None,
    report_goal_failure_penalty: float | None = None,
) -> ParsedRuntimeArtifacts:
    """Parse POMDPDDL texts and build semantic/bitwise/runtime objects."""
    parsed_domain = parse_domain(domain_text)
    parsed_problem = parse_problem(problem_text)
    parsed_default_policy = (
        parse_default_policy(default_policy_text) if default_policy_text is not None else None
    )
    parsed_domain, parsed_problem, parsed_default_policy = _specialize_last_action_domain_and_problem(
        parsed_domain,
        parsed_problem,
        parsed_default_policy,
    )

    artifact_root = _prepare_artifact_root(output_dir)
    package_base_name = _sanitize_package_base_name(
        parsed_problem.problem_name or parsed_domain.domain_name or "generated_pomdp"
    )
    explicit_package_dir = artifact_root / f"{package_base_name}_explicit_package" if output_dir is not None and emit_explicit_package else None
    bitwise_package_dir: Path | None = None
    despot_cpp_dir: Path | None = None
    if output_dir is not None:
        bitwise_package_dir = artifact_root / f"{package_base_name}_bitwise_package"
        despot_cpp_dir = artifact_root / "despot_cpp"

    artifact_metadata_path = artifact_root / "runtime_artifact_metadata.json"

    resolved_simulator_seed, resolved_planner_seed = _resolve_runtime_seeds(
        simulator_seed,
        planner_seed,
    )
    codegen_plan = build_codegen_plan_from_parsed(
        parsed_domain,
        parsed_problem,
        parsed_default_policy,
    )
    parsed_problem.init_belief = _complete_missing_last_action_false_belief(
        parsed_problem.init_belief,
        codegen_plan.predicates,
    )
    planner_model = _build_runtime_metadata_model(codegen_plan)
    runtime_goal_reward, runtime_max_step = _resolve_runtime_hyperparams(
        default_goal_reward=float(planner_model.goal_reward),
        default_max_step=max_step,
    )
    resolved_enable_report_goal_action, resolved_report_goal_failure_penalty = _resolve_report_goal_hyperparams(
        enable_report_goal_action=enable_report_goal_action,
        report_goal_failure_penalty=report_goal_failure_penalty,
    )
    artifact_fingerprint = _compute_artifact_fingerprint(
        domain_text,
        problem_text,
        default_policy_text,
        enable_report_goal_action=resolved_enable_report_goal_action,
        report_goal_failure_penalty=resolved_report_goal_failure_penalty,
    )
    artifact_metadata = _load_artifact_metadata(artifact_metadata_path)
    can_reuse_artifacts = (
        output_dir is not None
        and artifact_metadata is not None
        and artifact_metadata.get("fingerprint") == artifact_fingerprint
        and artifact_metadata.get("package_base_name") == package_base_name
    )
    reused_explicit_package = False
    reused_bitwise_package = False
    reused_despot_cpp = False
    planner_model.goal_reward = runtime_goal_reward
    planner_model._rng.seed(resolved_planner_seed)
    initial_particle_belief = convert_factorized_belief_to_indexed_particles(
        parsed_problem.init_belief,
        build_bitwise_index_layout(planner_model),
    )

    if explicit_package_dir is not None:
        if can_reuse_artifacts and _is_explicit_package_dir_valid(explicit_package_dir):
            reused_explicit_package = True
        else:
            emit_explicit_python_model_package(
                codegen_plan,
                explicit_package_dir,
                domain_text=domain_text,
                problem_text=problem_text,
                default_policy_text=default_policy_text,
                suppress_last_action_catalog=True,
            )

    if use_bitwise_world:
        simulator_model = planner_model
    else:
        simulator_explicit_dir = explicit_package_dir
        if simulator_explicit_dir is None:
            simulator_explicit_dir = artifact_root / f"{package_base_name}_semantic_runtime_package"
        if not _is_explicit_package_dir_valid(simulator_explicit_dir):
            emit_explicit_python_model_package(
                codegen_plan,
                simulator_explicit_dir,
                domain_text=domain_text,
                problem_text=problem_text,
                default_policy_text=default_policy_text,
                suppress_last_action_catalog=True,
            )
        semantic_module = _import_generated_package(simulator_explicit_dir)
        simulator_model = semantic_module.build_model()
        simulator_model.goal_reward = runtime_goal_reward
        simulator_model._rng.seed(resolved_simulator_seed)

    raw_bitwise_model = build_bitwise_model_from_parsed(
        planner_model,
        parsed_domain=parsed_domain,
        parsed_problem=parsed_problem,
        parsed_default_policy=parsed_default_policy,
    )
    raw_bitwise_model._rng.seed(resolved_planner_seed)

    despot_shared_library_path: Path | None = None
    if output_dir is not None:
        if can_reuse_artifacts and _is_bitwise_package_dir_valid(bitwise_package_dir):
            reused_bitwise_package = True
        else:
            emit_bitwise_python_model_package(
                raw_bitwise_model,
                planner_model,
                bitwise_package_dir,
                init_belief=initial_particle_belief,
            )

    wrapped_planner_model = planner_model
    wrapped_simulator_model = simulator_model
    bitwise_model = raw_bitwise_model
    if resolved_enable_report_goal_action:
        wrapped_planner_model = build_report_goal_planner_metadata_model(
            planner_model,
            report_goal_failure_penalty=resolved_report_goal_failure_penalty,
        )
        wrapped_simulator_model = ReportGoalActionSemanticModel(
            simulator_model,
            report_goal_failure_penalty=resolved_report_goal_failure_penalty,
        )
        bitwise_model = ReportGoalActionBitwiseModel(
            raw_bitwise_model,
            report_goal_failure_penalty=resolved_report_goal_failure_penalty,
        )
        setattr(bitwise_model, "_source_semantic_model", wrapped_planner_model)
    else:
        setattr(bitwise_model, "_source_semantic_model", planner_model)

    despot_cpp_model_code = convert_bitwise_model_to_despot_cpp_model_code(
        bitwise_model,
        semantic_model=wrapped_planner_model,
        despot_root=despot_root(),
    )
    if output_dir is not None:
        if (can_reuse_artifacts or force_reuse_despot_cpp) and _is_despot_cpp_dir_valid(despot_cpp_dir):
            reused_despot_cpp = True
        else:
            write_despot_cpp_model_code(despot_cpp_model_code, despot_cpp_dir)
        if reused_despot_cpp:
            despot_shared_library_path = find_existing_despot_shared_library(despot_cpp_dir / "build")
        if force_recompile_despot_cpp or (despot_shared_library_path is None and not skip_despot_build):
            despot_shared_library_path = configure_and_build_despot_cpp_project(
                despot_cpp_dir,
                despot_cpp_dir / "build",
                despot_root=despot_root(),
            )
        _write_init_belief_json(
            artifact_root / "init_belief.json",
            initial_particle_belief,
        )
        _write_init_state_json(artifact_root / "init_state.json", parsed_problem.init_state)
        _write_artifact_metadata(
            artifact_metadata_path,
            package_base_name=package_base_name,
            fingerprint=artifact_fingerprint,
        )

    world = POMDPWorld(
        init_state=parsed_problem.init_state,
        pomdp_model=wrapped_simulator_model,
        gamma=gamma,
        max_step=runtime_max_step,
        random_seed=resolved_simulator_seed,
    )
    bitwise_world = BitwisePOMDPWorld(
        init_state=parsed_problem.init_state,
        bitwise_pomdp_model=bitwise_model,
        layout=build_bitwise_index_layout(wrapped_planner_model),
        gamma=gamma,
        max_step=runtime_max_step,
        random_seed=resolved_simulator_seed,
    )
    planner = POMDPPlanner(
        wrapped_planner_model,
        initial_particle_belief,
        bitwise_pomdp_model=bitwise_model,
        despot_cpp_model_code=despot_cpp_model_code,
        despot_shared_library_path=despot_shared_library_path,
        random_seed=resolved_planner_seed,
        skip_despot_build=skip_despot_build,
        belief_update_debug=belief_update_debug,
    )

    probability_registry = build_probability_registry(bitwise_model)
    return ParsedRuntimeArtifacts(
        semantic_pomdp_model=wrapped_planner_model,
        simulator_pomdp_model=wrapped_simulator_model,
        planner_pomdp_model=wrapped_planner_model,
        init_belief=initial_particle_belief,
        init_state=dict(parsed_problem.init_state),
        bitwise_pomdp_model=bitwise_model,
        despot_cpp_model_code=despot_cpp_model_code,
        pomdp_world=bitwise_world if use_bitwise_world else world,
        bitwise_pomdp_world=bitwise_world,
        pomdp_planner=planner,
        simulator_seed=resolved_simulator_seed,
        planner_seed=resolved_planner_seed,
        parsed_domain=parsed_domain,
        parsed_problem=parsed_problem,
        parsed_default_policy=parsed_default_policy,
        artifact_root=artifact_root,
        explicit_package_dir=explicit_package_dir,
        bitwise_package_dir=bitwise_package_dir,
        despot_cpp_dir=despot_cpp_dir,
        reused_explicit_package=reused_explicit_package,
        reused_bitwise_package=reused_bitwise_package,
        reused_despot_cpp=reused_despot_cpp,
        probability_registry=probability_registry,
    )


def _prepare_artifact_root(output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.mkdtemp(prefix="pomdpddl_runtime_parse_"))


def _build_runtime_metadata_model(plan) -> RuntimeSemanticMetadataModel:
    return RuntimeSemanticMetadataModel(
        predicates=list(plan.predicates),
        observables=list(plan.observables),
        actions=[case.action for case in plan.grounded_actions],
        observation_rules=[case.rule for case in plan.grounded_observation_rules],
        default_policy_rules=[case.rule for case in plan.grounded_default_policy_rules],
        maximize_reward=plan.maximize_reward,
        goal_reward=plan.goal_reward,
    )


def _compute_artifact_fingerprint(
    domain_text: str,
    problem_text: str,
    default_policy_text: str | None,
    *,
    enable_report_goal_action: bool | None,
    report_goal_failure_penalty: float | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(RUNTIME_ARTIFACT_SCHEMA_VERSION.encode("utf-8"))
    digest.update(b"\n===RUNTIME_SCHEMA===\n")
    digest.update(domain_text.encode("utf-8"))
    digest.update(b"\n===PROBLEM===\n")
    digest.update(problem_text.encode("utf-8"))
    digest.update(b"\n===DEFAULT_POLICY===\n")
    digest.update((default_policy_text or "").encode("utf-8"))
    digest.update(b"\n===REPORT_GOAL_ACTION===\n")
    digest.update(str(enable_report_goal_action).encode("utf-8"))
    digest.update(b"\n===REPORT_GOAL_PENALTY===\n")
    digest.update(str(report_goal_failure_penalty).encode("utf-8"))
    digest.update(b"\n===DESPOT_CPP_CODEGEN===\n")
    try:
        digest.update(DESPOT_CPP_CODEGEN_PATH.read_bytes())
    except OSError:
        digest.update(str(DESPOT_CPP_CODEGEN_PATH).encode("utf-8"))
    return digest.hexdigest()


def _load_artifact_metadata(path: Path) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_artifact_metadata(
    path: Path,
    *,
    package_base_name: str,
    fingerprint: str,
) -> None:
    data = {
        "package_base_name": package_base_name,
        "fingerprint": fingerprint,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_explicit_package_dir_valid(package_dir: Path) -> bool:
    return (
        package_dir.exists()
        and (package_dir / "__init__.py").exists()
        and (package_dir / "model.py").exists()
        and (package_dir / "shared.py").exists()
        and (package_dir / "init_belief.json").exists()
    )


def _is_bitwise_package_dir_valid(package_dir: Path | None) -> bool:
    if package_dir is None:
        return False
    return (
        package_dir.exists()
        and (package_dir / "__init__.py").exists()
        and (package_dir / "model.py").exists()
        and (package_dir / "shared.py").exists()
        and (package_dir / "init_belief.json").exists()
        and (package_dir / "index_layout.json").exists()
    )


def _is_despot_cpp_dir_valid(despot_cpp_dir: Path | None) -> bool:
    if despot_cpp_dir is None:
        return False
    return (
        despot_cpp_dir.exists()
        and (despot_cpp_dir / "bitwise_pomdp_model.h").exists()
        and (despot_cpp_dir / "bitwise_pomdp_model.cpp").exists()
        and (despot_cpp_dir / "biwise_pomdp_planner.cpp").exists()
        and (despot_cpp_dir / "CMakeLists.txt").exists()
    )


def _resolve_runtime_seeds(
    simulator_seed: int | None,
    planner_seed: int | None,
) -> tuple[int, int]:
    if simulator_seed is None and planner_seed is None:
        simulator_seed = time.time_ns()
        planner_seed = simulator_seed + 1
        return simulator_seed, planner_seed
    if simulator_seed is None:
        planner_seed = int(planner_seed)
        simulator_seed = planner_seed + 1
        return simulator_seed, planner_seed
    if planner_seed is None:
        simulator_seed = int(simulator_seed)
        planner_seed = simulator_seed + 1
        return simulator_seed, planner_seed
    simulator_seed = int(simulator_seed)
    planner_seed = int(planner_seed)
    if simulator_seed == planner_seed:
        planner_seed += 1
    return simulator_seed, planner_seed


def _resolve_runtime_hyperparams(
    *,
    default_goal_reward: float,
    default_max_step: int,
    yaml_path: str | Path | None = None,
) -> tuple[float, int]:
    """Resolve simulator-facing hyperparameters from config/hyperparams.yaml when present."""
    path = Path(yaml_path) if yaml_path is not None else DEFAULT_HYPERPARAMS_YAML_PATH
    if not path.exists():
        return float(default_goal_reward), int(default_max_step)
    params = _parse_hyperparams_yaml(path)
    return (
        float(params.get("goal_reward", default_goal_reward)),
        int(params.get("max_steps", default_max_step)),
    )


def _resolve_report_goal_hyperparams(
    *,
    enable_report_goal_action: bool | None,
    report_goal_failure_penalty: float | None,
    yaml_path: str | Path | None = None,
) -> tuple[bool, float]:
    path = Path(yaml_path) if yaml_path is not None else DEFAULT_HYPERPARAMS_YAML_PATH
    params = _parse_hyperparams_yaml(path) if path is not None and path.exists() else {}
    resolved_enable = (
        bool(enable_report_goal_action)
        if enable_report_goal_action is not None
        else bool(params.get("enable_report_goal_action", False))
    )
    resolved_penalty = (
        float(report_goal_failure_penalty)
        if report_goal_failure_penalty is not None
        else float(params.get("report_goal_failure_penalty", -10.0))
    )
    return resolved_enable, resolved_penalty


def _sanitize_package_base_name(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in name)
    sanitized = sanitized.strip("_") or "generated_pomdp"
    if sanitized[0].isdigit():
        sanitized = f"p_{sanitized}"
    return sanitized


def _import_generated_package(package_dir: Path) -> ModuleType:
    package_root = package_dir.parent
    package_name = package_dir.name
    sys.path.insert(0, str(package_root))
    try:
        importlib.invalidate_caches()
        for module_name in list(sys.modules):
            if module_name == package_name or module_name.startswith(f"{package_name}."):
                del sys.modules[module_name]
        return importlib.import_module(package_name)
    finally:
        sys.path = [entry for entry in sys.path if entry != str(package_root)]


def _write_init_belief_json(
    output_path: Path,
    init_belief: IndexedParticleBelief,
) -> None:
    dump_indexed_particle_belief_json(init_belief, output_path)


def _write_init_state_json(output_path: Path, init_state: StateEntry) -> None:
    data = {
        "true": sorted(
            predicate.to_pddl_str()
            for predicate, value in init_state.items()
            if value
        ),
        "false": sorted(
            predicate.to_pddl_str()
            for predicate, value in init_state.items()
            if not value
        ),
    }
    output_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
