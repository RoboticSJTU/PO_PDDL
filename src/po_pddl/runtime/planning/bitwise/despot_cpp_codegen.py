"""Generate DESPOT C++ source files from a bitwise POMDP model package.

Usage
-----
    python -m POMDPDDL.bitwise.despot_cpp_codegen \
        POMDPDDL.examples.operator_coverage_mission.bitwise_package \
        --output-dir /tmp/despot_gen
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import re
import textwrap
import warnings
from importlib import import_module
from pathlib import Path
from typing import Any

from ..data_structures import DespotCppModelCode
from ..base.pomdp_model import POMDPModelBase as SemanticPOMDPModelBase
from .parser import (
    BitwiseEffectBranch,
    BitwiseGoalCheck,
    BitwiseModelSkeleton,
    BitwiseObservationBranch,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

DESPOT_CORE_SOURCE_REL_PATHS = [
    "core/builtin_lower_bounds.cpp",
    "core/builtin_policy.cpp",
    "core/builtin_upper_bounds.cpp",
    "core/globals.cpp",
    "core/mdp.cpp",
    "core/node.cpp",
    "core/particle_belief.cpp",
    "core/solver.cpp",
    "interface/belief.cpp",
    "interface/default_policy.cpp",
    "interface/lower_bound.cpp",
    "interface/pomdp.cpp",
    "interface/upper_bound.cpp",
    "interface/world.cpp",
    "random_streams.cpp",
    "solver/despot.cpp",
    "solver/pomcp.cpp",
    "util/coord.cpp",
    "util/dirichlet.cpp",
    "util/exec_tracker.cpp",
    "util/floor.cpp",
    "util/gamma.cpp",
    "util/logging.cpp",
    "util/random.cpp",
    "util/seeds.cpp",
    "util/util.cpp",
]

def build_despot_cpp_model_code_from_bitwise_package(
    module_name: str,
    *,
    class_name: str = "BitwisePOMDPModel",
    state_class_name: str = "BitwisePOMDPState",
    despot_root: str | Path | None = None,
) -> DespotCppModelCode:
    """Build in-memory DESPOT C++ source texts from one bitwise model package."""
    data = _load_package_data(module_name)
    if despot_root is None:
        from ..engine.paths import despot_root as resolve_despot_root

        despot_root = resolve_despot_root()
    despot_root_str = Path(despot_root).as_posix()
    return DespotCppModelCode(
        bitwise_pomdp_model_h=_gen_header(data, class_name, state_class_name),
        bitwise_pomdp_model_cpp=_gen_source(data, class_name, state_class_name),
        main_cpp="",
        planner_binding_cpp=_gen_planner_pybind(class_name),
        cmakelists_txt=_gen_pybind_cmakelists(despot_root_str),
        build_sh=_gen_build_script(),
    )


def build_despot_cpp_model_code_from_bitwise_model(
    bitwise_model: BitwiseModelSkeleton,
    *,
    semantic_model: SemanticPOMDPModelBase,
    class_name: str = "BitwisePOMDPModel",
    state_class_name: str = "BitwisePOMDPState",
    despot_root: str | Path | None = None,
) -> DespotCppModelCode:
    """Build in-memory DESPOT C++ source texts directly from one bitwise model object."""
    if despot_root is None:
        from ..engine.paths import despot_root as resolve_despot_root

        despot_root = resolve_despot_root()
    despot_root_str = Path(despot_root).as_posix()
    data = {
        "grounded_predicates_count": bitwise_model.grounded_predicates_count,
        "grounded_observables_count": bitwise_model.grounded_observables_count,
        "total_actions": bitwise_model.total_actions,
        "maximize_reward": bitwise_model.maximize_reward,
        "goal_reward": bitwise_model.goal_reward,
        "enable_report_goal_action": bool(getattr(bitwise_model, "enable_report_goal_action", False)),
        "report_goal_failure_penalty": float(
            getattr(bitwise_model, "report_goal_failure_penalty", -10.0)
        ),
        "check_obs_conditions": [None] * len(bitwise_model.observation_rule_condition_checks),
        "check_dp_conditions": [None] * len(bitwise_model.default_policy_rule_condition_checks),
        "index_layout": {
            "predicates": [_render_pomdpddl_atom(predicate.name, predicate.params) for predicate in semantic_model.predicates],
            "observables": [_render_pomdpddl_atom(observable.name, observable.params) for observable in semantic_model.observables],
            "actions": [_render_pomdpddl_atom(action.name, action.params) for action in semantic_model.actions],
        },
    }
    return DespotCppModelCode(
        bitwise_pomdp_model_h=_gen_header(data, class_name, state_class_name),
        bitwise_pomdp_model_cpp=_gen_source_from_bitwise_model(
            data,
            bitwise_model,
            class_name,
            state_class_name,
        ),
        main_cpp="",
        planner_binding_cpp=_gen_planner_pybind(class_name),
        cmakelists_txt=_gen_pybind_cmakelists(despot_root_str),
        build_sh=_gen_build_script(),
    )

def emit_despot_cpp_from_bitwise_package(
    module_name: str,
    output_dir: str | Path,
    *,
    class_name: str = "BitwisePOMDPModel",
    state_class_name: str = "BitwisePOMDPState",
    despot_root: str | Path | None = None,
) -> Path:
    """Load a bitwise model Python package and emit DESPOT C++ source files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    code = build_despot_cpp_model_code_from_bitwise_package(
        module_name,
        class_name=class_name,
        state_class_name=state_class_name,
        despot_root=despot_root,
    )
    (out / "bitwise_pomdp_model.h").write_text(code.bitwise_pomdp_model_h, encoding="utf-8")
    (out / "bitwise_pomdp_model.cpp").write_text(
        code.bitwise_pomdp_model_cpp,
        encoding="utf-8",
    )
    (out / "biwise_pomdp_planner.cpp").write_text(
        code.planner_binding_cpp or code.main_cpp,
        encoding="utf-8",
    )
    (out / "CMakeLists.txt").write_text(code.cmakelists_txt, encoding="utf-8")
    build_sh_path = out / "build.sh"
    build_sh_path.write_text(code.build_sh, encoding="utf-8")
    build_sh_path.chmod(0o755)
    # Auto-compile
    import subprocess
    result = subprocess.run(
        ["bash", str(build_sh_path.resolve())],
        cwd=str(out),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Build STDOUT:", result.stdout)
        print("Build STDERR:", result.stderr)
        raise RuntimeError(f"build.sh failed with exit code {result.returncode}")
    # Find and return .so path
    so_files = list((out / "build").glob("despot_planner*.so")) + list((out / "build").glob("despot_planner*.dylib"))
    so_path = so_files[0] if so_files else None
    if so_path:
        print(f"Shared library: {so_path.resolve()}")
    else:
        print("WARNING: No .so/.dylib found after build")
    return out


# ---------------------------------------------------------------------------
# Package data loader
# ---------------------------------------------------------------------------

def _load_package_data(module_name: str) -> dict[str, Any]:
    mod = import_module(f"{module_name}.model")
    shared = import_module(f"{module_name}.shared")

    belief_json_path = Path(mod.__file__).with_name("init_belief.json")
    with open(belief_json_path, encoding="utf-8") as f:
        belief_data = json.load(f)

    layout_json_path = Path(mod.__file__).with_name("index_layout.json")
    if layout_json_path.exists():
        with open(layout_json_path, encoding="utf-8") as f:
            layout_data = json.load(f)
    else:
        layout_data = None

    data: dict[str, Any] = {
        "grounded_predicates_count": getattr(shared, "GROUNDED_PREDICATES_COUNT"),
        "grounded_observables_count": getattr(shared, "GROUNDED_OBSERVABLES_COUNT"),
        "total_actions": getattr(shared, "TOTAL_ACTIONS"),
        "maximize_reward": getattr(shared, "MAXIMIZE_REWARD"),
        "goal_reward": getattr(shared, "GOAL_REWARD"),
        "init_belief": belief_data,
        "model_instance": getattr(mod, "build_model")(),
        "module_name": module_name,
        "check_preconditions": getattr(mod, "CHECK_ACTION_PRECONDITIONS", []),
        "forward_actions": getattr(mod, "FORWARD_ACTIONS", []),
        "get_action_rewards": getattr(mod, "GET_ACTION_REWARDS", []),
        "check_obs_conditions": getattr(mod, "CHECK_OBSERVATION_RULE_CONDITIONS", []),
        "observe_with_rules": getattr(mod, "OBSERVE_WITH_RULES", []),
        "check_dp_conditions": getattr(mod, "CHECK_DEFAULT_POLICY_RULE_CONDITIONS", []),
        "get_dp_actions": getattr(mod, "GET_DEFAULT_POLICY_RULE_ACTIONS", []),
        "index_layout": layout_data,
    }
    return data


# ---------------------------------------------------------------------------
# Multi-word helpers
# ---------------------------------------------------------------------------

def _n_words(pred_count: int) -> int:
    return max(1, math.ceil(pred_count / 64))

def _int_to_words(value: int, n_words: int) -> list[int]:
    mask = (1 << 64) - 1
    words = []
    for _ in range(n_words):
        words.append(value & mask)
        value >>= 64
    return words

def _words_to_c_array(words: list[int]) -> str:
    return "{" + ", ".join(f"0x{w:016X}ULL" for w in words) + "}"

def _words_to_c_decl(name: str, words: list[int]) -> str:
    n = len(words)
    return f"static const uint64_t {name}[{n}] = {_words_to_c_array(words)};"


# ---------------------------------------------------------------------------
# AST helpers for parsing Python bitwise functions
# ---------------------------------------------------------------------------

def _eval_const(node: Any) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _eval_const(node.operand)
        if v is not None:
            return -v
    return None

def _emit_check_true(mask_words: list[int], sv: str) -> str:
    checks = []
    for i, w in enumerate(mask_words):
        if w == 0:
            continue
        checks.append(f"(({sv}[{i}] & 0x{w:016X}ULL) == 0x{w:016X}ULL)")
    return "(" + " && ".join(checks) + ")" if checks else "true"

def _emit_check_false(mask_words: list[int], sv: str) -> str:
    checks = []
    for i, w in enumerate(mask_words):
        if w == 0:
            continue
        checks.append(f"(({sv}[{i}] & 0x{w:016X}ULL) == 0)")
    return "(" + " && ".join(checks) + ")" if checks else "true"

def _compile_bool_expr(node: Any, sv: str, nw: int) -> str:
    if isinstance(node, ast.Constant):
        if node.value is True:
            return "true"
        elif node.value is False:
            return "false"
    if isinstance(node, ast.BoolOp):
        parts = [_compile_bool_expr(v, sv, nw) for v in node.values]
        op = " && " if isinstance(node.op, ast.And) else " || "
        return "(" + op.join(parts) + ")"
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        left = node.left
        right = node.comparators[0]
        if isinstance(left, ast.BinOp) and isinstance(left.op, ast.BitAnd):
            mask_val = _eval_const(left.right)
            cmp_val = _eval_const(right)
            if mask_val is not None and cmp_val is not None:
                mw = _int_to_words(mask_val, nw)
                if cmp_val == mask_val:
                    return _emit_check_true(mw, sv)
                elif cmp_val == 0:
                    return _emit_check_false(mw, sv)
    val = _eval_const(node)
    if val is True:
        return "true"
    if val is False:
        return "false"
    return "false /* unrecognized */"


def _extract_precondition_logic(func: Any, nw: int) -> str:
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    fd = tree.body[0]
    for stmt in fd.body:
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            expr_str = _compile_bool_expr(stmt.value, "state", nw)
            return f"return {expr_str};"
    return "return false;"


def _extract_goal_check_from_callable(func: Any, nw: int) -> BitwiseGoalCheck | None:
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    fd = tree.body[0]
    for stmt in fd.body:
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            return BitwiseGoalCheck(clauses=_extract_dnf_clauses(stmt.value, nw))
    return None


def _extract_dnf_clauses(node: Any, nw: int) -> list[tuple[int, int]]:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        clauses = []
        for v in node.values:
            clauses.extend(_extract_dnf_clauses(v, nw))
        return clauses
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        tm, fm = 0, 0
        for v in node.values:
            sub = _extract_single_clause(v)
            if sub is not None:
                t, f = sub
                tm |= t
                fm |= f
        return [(tm, fm)]
    sub = _extract_single_clause(node)
    if sub is not None:
        return [sub]
    return [(0, 0)]

def _extract_single_clause(node: Any) -> tuple[int, int] | None:
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        left = node.left
        right = node.comparators[0]
        if isinstance(left, ast.BinOp) and isinstance(left.op, ast.BitAnd):
            mask_val = _eval_const(left.right)
            cmp_val = _eval_const(right)
            if mask_val is not None and cmp_val is not None:
                if cmp_val == mask_val:
                    return (mask_val, 0)
                elif cmp_val == 0:
                    return (0, mask_val)
    return None


def _extract_branch_weights(node: Any) -> list[float] | None:
    if isinstance(node, ast.Call) and node.args:
        arg = node.args[0]
        if isinstance(arg, ast.List):
            ws = []
            for elt in arg.elts:
                v = _eval_const(elt)
                if v is None:
                    return None
                ws.append(float(v))
            return ws
    return None


def _extract_set_clear(node: Any) -> tuple[int, int]:
    """Extract (set_mask, clear_mask) from an expression like (x | S) & ~C."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
        cm = 0
        if isinstance(node.right, ast.UnaryOp) and isinstance(node.right.op, ast.Invert):
            cm = _eval_const(node.right.operand)
            if cm is None:
                cm = 0
        sm = 0
        if isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.BitOr):
            sm = _eval_const(node.left.right)
            if sm is None:
                sm = 0
        return (sm, cm)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        sm = _eval_const(node.right)
        if sm is None:
            sm = 0
        return (sm, 0)
    return (None, None)


# ---------------------------------------------------------------------------
# Forward action logic extraction (FIXED: proper branch handling)
# ---------------------------------------------------------------------------

def _get_branch_index_from_test(test: Any) -> int | None:
    """Extract integer branch index from 'branch_index == N'.
    Returns None for 'branch_index is None' or unrecognized patterns."""
    if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name):
        if test.left.id == "branch_index" and len(test.ops) == 1:
            if isinstance(test.ops[0], ast.Is):
                return None  # branch_index is None -> skip
            if isinstance(test.ops[0], ast.Eq):
                val = _eval_const(test.comparators[0])
                if val is not None:
                    return int(val)
    return None


def _parse_branch_effect(body: list) -> tuple[int, int]:
    """Extract (set_mask, clear_mask) from a branch body."""
    for s in body:
        if isinstance(s, ast.Assign) and isinstance(s.value, ast.BinOp):
            sm, cm = _extract_set_clear(s.value)
            if sm is not None:
                return (sm, cm)
        elif isinstance(s, ast.Pass):
            return (0, 0)
    return (0, 0)


def _collect_if_chain_branches(stmt: Any) -> dict[int, tuple[int, int]]:
    """Walk the if/elif chain for branch_index, collect {index: (set, clear)}."""
    raw: dict[int, tuple[int, int]] = {}
    _walk_branch_chain(stmt, raw)
    return raw


def _walk_branch_chain(stmt: Any, raw: dict) -> None:
    if not isinstance(stmt, ast.If):
        return
    idx = _get_branch_index_from_test(stmt.test)
    if idx is not None and idx >= 0:
        raw[idx] = _parse_branch_effect(stmt.body)
    for elif_s in stmt.orelse:
        if isinstance(elif_s, ast.If):
            _walk_branch_chain(elif_s, raw)


def _is_branch_var_ref(stmt: Any) -> bool:
    """Check if this if-statement references branch_index."""
    if isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare):
        if isinstance(stmt.test.left, ast.Name) and stmt.test.left.id == "branch_index":
            return True
    return False


def _parse_cond_reward(
    stmt: Any,
    nw: int,
    state_var: str = "__STATE__",
) -> tuple[str, float] | None:
    """Extract a reward condition as (C++ expression string, delta).
    Uses _compile_bool_expr for adaptive handling of any condition shape."""
    delta = _get_reward_delta(stmt.body)
    if delta is None:
        return None
    expr = _compile_bool_expr(stmt.test, state_var, nw)
    if "unrecognized" in expr:
        return None
    return (expr, delta)


def _get_reward_delta(stmts: list) -> float | None:
    for s in stmts:
        if isinstance(s, ast.AugAssign) and isinstance(s.op, ast.Add):
            if isinstance(s.target, ast.Name) and s.target.id == "reward":
                return _eval_const(s.value)
    return None


def _extract_forward_action_logic(func: Any, action_id: int, nw: int) -> dict:
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    fd = tree.body[0]
    result: dict[str, Any] = {
        "prob_branches": [],   # list of (set_mask, clear_mask)
        "cond_rewards": [],    # list of (expr_str, delta)
        "has_goal_check": False,
        "prob_weights": [],
        "deterministic_effect": None,  # (set_mask, clear_mask) if no branch
    }
    for stmt in fd.body:
        _visit_fwd(stmt, result, nw)
    return result


def _visit_fwd(stmt: Any, result: dict, nw: int) -> None:
    if isinstance(stmt, ast.If):
        test = stmt.test
        # skip precondition block
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return
        # goal check
        if "is_goal" in ast.dump(test):
            result["has_goal_check"] = True
            return
        # branch_index check chain
        if _is_branch_var_ref(stmt):
            raw = _collect_if_chain_branches(stmt)
            if raw:
                max_idx = max(raw.keys())
                branches = []
                for i in range(max_idx + 1):
                    branches.append(raw.get(i, (0, 0)))
                result["prob_branches"] = branches
            return
        # conditional reward (adaptive: uses _compile_bool_expr)
        cr = _parse_cond_reward(stmt, nw)
        if cr is not None:
            result["cond_rewards"].append(cr)
            return
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            if name == "branch_index":
                ws = _extract_branch_weights(stmt.value)
                if ws:
                    result["prob_weights"] = ws
            elif name == "next_state" and isinstance(stmt.value, ast.BinOp):
                sm, cm = _extract_set_clear(stmt.value)
                if sm is not None:
                    result["deterministic_effect"] = (sm, cm)


# ---------------------------------------------------------------------------
# Observation and default policy extraction
# ---------------------------------------------------------------------------

def _extract_observe_logic(func: Any, nw: int) -> dict:
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    fd = tree.body[0]
    result = {"prob_weights": [], "branches": []}
    for stmt in fd.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            if isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "branch_index":
                ws = _extract_branch_weights(stmt.value)
                if ws:
                    result["prob_weights"] = ws
        if isinstance(stmt, ast.If):
            _extract_obs_branches(stmt, result)
    return result


def _extract_obs_branches(stmt: Any, result: dict) -> None:
    if not isinstance(stmt, ast.If):
        return
    test = stmt.test
    if isinstance(test, ast.Compare):
        if isinstance(test.left, ast.Name) and test.left.id == "branch_index":
            for s in stmt.body:
                if isinstance(s, ast.Return) and isinstance(s.value, ast.Tuple):
                    elts = s.value.elts
                    if len(elts) == 2:
                        ob = _eval_const(elts[0])
                        om = _eval_const(elts[1])
                        if ob is not None and om is not None:
                            result["branches"].append((ob, om))
    for elif_s in stmt.orelse:
        if isinstance(elif_s, ast.If):
            _extract_obs_branches(elif_s, result)


def _extract_dp_action_id(func: Any) -> int:
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    fd = tree.body[0]
    for stmt in fd.body:
        if isinstance(stmt, ast.Return):
            val = _eval_const(stmt.value)
            if val is not None:
                return int(val)
    return 0


# ---------------------------------------------------------------------------
# Header gen
# ---------------------------------------------------------------------------

def _gen_header(data: dict, cn: str, sn: str) -> str:
    nw = _n_words(data["grounded_predicates_count"])
    ta = data["total_actions"]
    nor = len(data["check_obs_conditions"])
    ndp = len(data["check_dp_conditions"])
    guard = cn.upper() + "_H"
    L = []
    L.append(f"#ifndef {guard}")
    L.append(f"#define {guard}")
    L.append("")
    L.append('#include "bitwise_despot.h"')
    L.append('#include <vector>')
    L.append("")
    L.append("namespace despot {")
    L.append("")
    L.append(f"static const int N_WORDS = {nw};")
    L.append(f"using {sn} = BitwiseState<N_WORDS>;")
    L.append("")
    L.append(f"class {cn} : public DSPOMDP {{")
    L.append("protected:")
    L.append(f"    mutable MemoryPool<{sn}> memory_pool_;")
    L.append("public:")
    L.append(f"    static constexpr int TOTAL_ACTIONS = {ta};")
    L.append(f"    static constexpr int N_OBS_RULES = {nor};")
    L.append(f"    static constexpr int N_DP_RULES = {ndp};")
    L.append(f"    static double GOAL_REWARD;")
    L.append(f"    static double REPORT_GOAL_FAILURE_PENALTY;")
    L.append(f"    static double ACTION_INVALID_PENALTY;")
    L.append(f"    static double ACTION_NO_CHANGE_PENALTY;")
    L.append(f"    static double DEAD_END_PENALTY;")
    L.append(f"    static constexpr int N_PREDICATES = {data['grounded_predicates_count']};")
    L.append(f"    static constexpr int N_OBSERVABLES = {data['grounded_observables_count']};")
    L.append(f"    static double DP_EXPLOIT_PROB;")
    L.append("")
    L.append(f"    {cn}();")
    L.append(f"    ~{cn}() override {{}};")
    L.append("    int NumActions() const override;")
    L.append("    bool Step(State& state, double rand_num, ACT_TYPE action,")
    L.append("             double& reward, OBS_TYPE& obs) const override;")
    L.append("    bool Step(State& state, ACT_TYPE action,")
    L.append("             double& reward, OBS_TYPE& obs) const override;")
    L.append("    double ObsProb(OBS_TYPE obs, const State& state,")
    L.append("                   ACT_TYPE action) const override;")
    L.append("    Belief* InitialBelief(const State* start,")
    L.append('                          std::string type = "DEFAULT") const override;')
    L.append("    double GetMaxReward() const override;")
    L.append("    ValuedAction GetBestAction() const override;")
    L.append('    ScenarioUpperBound* CreateScenarioUpperBound(std::string name = "DEFAULT",')
    L.append('        std::string particle_bound_name = "DEFAULT") const override;')
    L.append('    ScenarioLowerBound* CreateScenarioLowerBound(std::string name = "DEFAULT",')
    L.append('        std::string particle_bound_name = "DEFAULT") const override;')
    L.append("    State* Allocate(int state_id, double weight) const;")
    L.append("    State* Copy(const State* particle) const;")
    L.append("    void Free(State* particle) const;")
    L.append("    int NumActiveParticles() const;")
    L.append("    void PrintState(const State& state, std::ostream& out = std::cout) const;")
    L.append("    void PrintObs(const State& state, OBS_TYPE observation,")
    L.append("                  std::ostream& out = std::cout) const;")
    L.append("    void PrintAction(ACT_TYPE action, std::ostream& out = std::cout) const;")
    L.append("    void PrintBelief(const Belief& belief, std::ostream& out = std::cout) const;")
    L.append("    std::vector<ACT_TYPE> GetLegalActions(std::vector<State*> particles) const override;")
    L.append("    bool CheckActionPrecondition(int action, const uint64_t* bits) const;")
    L.append("    int GetDefaultPolicyAction(const uint64_t* bits) const;")
    L.append("    void ResetProfileStats() const;")
    L.append("    void PrintProfileStats(std::ostream& out = std::cout) const;")
    L.append("private:")
    L.append("    bool IsGoal(const uint64_t* bits) const;")
    L.append("    void ForwardAction(int action, uint64_t* bits, uint64_t& rng_seed,")
    L.append("                       double& reward) const;")
    L.append("    OBS_TYPE Observe(ACT_TYPE action, const uint64_t* bits, uint64_t& rng_seed) const;")
    L.append("    bool DeterministicActionWouldChangeState(int action,")
    L.append("        const uint64_t* bits) const;")
    L.append("    OBS_TYPE PredictDeterministicExplicitObservation(int action,")
    L.append("        const uint64_t* bits, uint64_t rng_seed) const;")
    L.append("    bool ShouldPruneRepeatedDeterministicObservation(int action,")
    L.append("        const uint64_t* bits, OBS_TYPE last_obs, bool has_last_obs,")
    L.append("        uint64_t rng_seed) const;")
    L.append("};")
    L.append("")
    L.append("void set_init_belief_json_path(const std::string& path);")
    L.append("void update_action_weights(int action_id, const std::vector<double>& weights);")
    L.append("void update_obs_weights(int rule_id, const std::vector<double>& weights);")
    L.append("void update_hyperparams(double goal_reward, double report_goal_failure_penalty,")
    L.append("    double action_invalid_penalty,")
    L.append("    double action_no_change_penalty,")
    L.append("    double dead_end_penalty,")
    L.append("    double dp_exploit_prob, double time_per_move, int num_scenarios,")
    L.append("    int search_depth, int max_policy_sim_len, int sim_len,")
    L.append("    double discount, double pruning_constant, double xi,")
    L.append("    unsigned int root_seed, bool silence);")
    L.append("")
    L.append("} // namespace despot")
    L.append(f"#endif // {guard}")
    L.append("")
    return "\n".join(L)


def _gen_profile_helpers(cn: str) -> str:
    stats_name = f"{cn}ProfileStats"
    stats_var = f"G_{cn.upper()}_PROFILE_STATS"
    enabled_fn = f"{cn}ProfileEnabled"
    L = []
    L.append(f"struct {stats_name} {{")
    L.append("    double step_seconds = 0.0;")
    L.append("    double forward_seconds = 0.0;")
    L.append("    double observe_seconds = 0.0;")
    L.append("    double legal_seconds = 0.0;")
    L.append("    double default_policy_seconds = 0.0;")
    L.append("    double default_policy_select_seconds = 0.0;")
    L.append("    double default_policy_action_lookup_seconds = 0.0;")
    L.append("    uint64_t step_calls = 0;")
    L.append("    uint64_t step_invalid_precondition = 0;")
    L.append("    uint64_t step_no_change_penalty_applied = 0;")
    L.append("    uint64_t observe_calls = 0;")
    L.append("    uint64_t observe_candidate_rules = 0;")
    L.append("    uint64_t observe_cond_checks = 0;")
    L.append("    uint64_t observe_fired_rules = 0;")
    L.append("    uint64_t legal_calls = 0;")
    L.append("    uint64_t legal_particles = 0;")
    L.append("    uint64_t legal_precondition_checks = 0;")
    L.append("    uint64_t default_policy_calls = 0;")
    L.append("    uint64_t default_policy_particles = 0;")
    L.append("    uint64_t default_policy_step_calls = 0;")
    L.append("    uint64_t default_policy_select_calls = 0;")
    L.append("    uint64_t default_policy_rule_checks = 0;")
    L.append("    uint64_t default_policy_action_lookup_calls = 0;")
    L.append("};")
    L.append(f"static {stats_name} {stats_var};")
    L.append(f"static inline bool {enabled_fn}() {{")
    L.append("    return !Globals::config.silence;")
    L.append("}")
    L.append("")
    L.append(f"void {cn}::ResetProfileStats() const {{")
    L.append(f"    if (!{enabled_fn}()) return;")
    L.append(f"    {stats_var} = {stats_name}{{}};")
    L.append("}")
    L.append("")
    L.append(f"void {cn}::PrintProfileStats(std::ostream& out) const {{")
    L.append(f"    if (!{enabled_fn}()) return;")
    L.append(f'    out << "[PROFILE model] step_calls=" << {stats_var}.step_calls')
    L.append(f'        << " step_seconds=" << {stats_var}.step_seconds')
    L.append(f'        << " forward_seconds=" << {stats_var}.forward_seconds')
    L.append(f'        << " observe_seconds=" << {stats_var}.observe_seconds')
    L.append(f'        << " invalid_preconditions=" << {stats_var}.step_invalid_precondition')
    L.append(f'        << " no_change_penalties=" << {stats_var}.step_no_change_penalty_applied << std::endl;')
    L.append(f'    out << "[PROFILE legal] calls=" << {stats_var}.legal_calls')
    L.append(f'        << " seconds=" << {stats_var}.legal_seconds')
    L.append(f'        << " particles=" << {stats_var}.legal_particles')
    L.append(f'        << " precondition_checks=" << {stats_var}.legal_precondition_checks << std::endl;')
    L.append(f'    out << "[PROFILE observe] calls=" << {stats_var}.observe_calls')
    L.append(f'        << " seconds=" << {stats_var}.observe_seconds')
    L.append(f'        << " candidate_rules=" << {stats_var}.observe_candidate_rules')
    L.append(f'        << " cond_checks=" << {stats_var}.observe_cond_checks')
    L.append(f'        << " fired_rules=" << {stats_var}.observe_fired_rules << std::endl;')
    L.append(f'    out << "[PROFILE default_policy] calls=" << {stats_var}.default_policy_calls')
    L.append(f'        << " seconds=" << {stats_var}.default_policy_seconds')
    L.append(f'        << " particles=" << {stats_var}.default_policy_particles')
    L.append(f'        << " step_calls=" << {stats_var}.default_policy_step_calls << std::endl;')
    L.append(f'    out << "[PROFILE default_policy_select] calls=" << {stats_var}.default_policy_select_calls')
    L.append(f'        << " seconds=" << {stats_var}.default_policy_select_seconds')
    L.append(f'        << " rule_checks=" << {stats_var}.default_policy_rule_checks << std::endl;')
    L.append(f'    out << "[PROFILE default_policy_action] calls=" << {stats_var}.default_policy_action_lookup_calls')
    L.append(f'        << " seconds=" << {stats_var}.default_policy_action_lookup_seconds << std::endl;')
    L.append("}")
    L.append("")
    return "\n".join(L)

# ---------------------------------------------------------------------------
# Source gen
# ---------------------------------------------------------------------------


def _gen_no_change_ignored_mask_data(data: dict, nw: int) -> str:
    """Emit a bit-mask for predicates ignored by no-change world-state checks."""
    ignored_mask = _no_change_ignored_mask(data)
    words = _int_to_words(ignored_mask, nw)
    rendered_words = ", ".join(f"0x{word:016x}ULL" for word in words)
    return (
        f"static const uint64_t NO_CHANGE_IGNORED_PREDICATE_MASK[N_WORDS] = "
        f"{{ {rendered_words} }};"
    )


def _no_change_ignored_mask(data: dict) -> int:
    layout = data.get("index_layout") or {}
    preds = layout.get("predicates") or []
    ignored_mask = 0
    for predicate_index, atom in enumerate(preds[: data["grounded_predicates_count"]]):
        if atom.startswith("(last_action_"):
            ignored_mask |= 1 << predicate_index
    return ignored_mask


def _parse_rendered_atom(atom: str) -> tuple[str, list[str]] | None:
    atom = atom.strip()
    if not atom.startswith("(") or not atom.endswith(")"):
        return None
    payload = atom[1:-1].strip()
    if not payload:
        return None
    parts = payload.split()
    return parts[0], parts[1:]


def _observable_predicate_keys(observables: list[str]) -> set[tuple[str, tuple[str, ...]]]:
    keys: set[tuple[str, tuple[str, ...]]] = set()
    for observable in observables:
        parsed = _parse_rendered_atom(observable)
        if parsed is None:
            continue
        name, params = parsed
        if name in {"obs-nothing", "obs_nothing"}:
            continue
        candidates = [name]
        if name.startswith("obs_"):
            candidates.append(name[len("obs_"):])
        if name.startswith("obs-"):
            candidates.append(name[len("obs-"):])
        for candidate in candidates:
            keys.add((candidate, tuple(params)))
            keys.add((candidate.replace("-", "_"), tuple(params)))
            keys.add((candidate.replace("_", "-"), tuple(params)))
    return keys


def _is_last_action_predicate_atom(atom: str) -> bool:
    parsed = _parse_rendered_atom(atom)
    if parsed is None:
        return False
    name, _params = parsed
    return (
        name.startswith("last_action_")
        or name.startswith("last-action-")
        or name == "last-action"
        or name.startswith("last_")
    )


def _fully_observed_predicate_indices(data: dict) -> list[int]:
    """Return predicate indices that should be folded into DESPOT observation keys."""
    layout = data.get("index_layout") or {}
    predicates = list(layout.get("predicates") or [])
    observables = list(layout.get("observables") or [])
    observable_keys = _observable_predicate_keys(observables)

    indices: list[int] = []
    for predicate_index, predicate in enumerate(predicates[: data["grounded_predicates_count"]]):
        if _is_last_action_predicate_atom(predicate):
            continue
        parsed = _parse_rendered_atom(predicate)
        if parsed is None:
            continue
        name, params = parsed
        candidate_keys = {
            (name, tuple(params)),
            (name.replace("-", "_"), tuple(params)),
            (name.replace("_", "-"), tuple(params)),
        }
        if candidate_keys & observable_keys:
            continue
        indices.append(predicate_index)
    return indices


def _positive_branch_count(branches: list[Any]) -> int:
    return sum(1 for branch in branches if float(getattr(branch, "probability", 0.0)) > 0.0)


def _effect_branch_real_change_mask(branch: BitwiseEffectBranch, ignored_mask: int) -> int:
    return (int(branch.set_mask) | int(branch.clear_mask)) & ~ignored_mask


def _is_deterministic_no_real_change_action(
    action_id: int,
    *,
    action_effect_distributions: list[list[BitwiseEffectBranch]],
    action_effect_condition_checks: list[list[BitwiseGoalCheck]],
    action_conditional_effect_distributions: list[list[list[BitwiseEffectBranch]]],
    ignored_mask: int,
) -> bool:
    if action_id < 0 or action_id >= len(action_effect_distributions):
        return False
    unconditional = action_effect_distributions[action_id]
    if _positive_branch_count(unconditional) != 1:
        return False
    positive_unconditional = [
        branch for branch in unconditional if float(branch.probability) > 0.0
    ]
    if not positive_unconditional:
        return False
    if _effect_branch_real_change_mask(positive_unconditional[0], ignored_mask) != 0:
        return False

    condition_checks = action_effect_condition_checks[action_id]
    conditional_distributions = action_conditional_effect_distributions[action_id]
    if condition_checks or conditional_distributions:
        return False
    return True


def _deterministic_unconditional_effect_branch(
    action_id: int,
    action_effect_distributions: list[list[BitwiseEffectBranch]],
    action_effect_condition_checks: list[list[BitwiseGoalCheck]],
    action_conditional_effect_distributions: list[list[list[BitwiseEffectBranch]]],
) -> BitwiseEffectBranch | None:
    if action_id < 0 or action_id >= len(action_effect_distributions):
        return None
    unconditional = action_effect_distributions[action_id]
    if _positive_branch_count(unconditional) != 1:
        return None
    if action_effect_condition_checks[action_id] or action_conditional_effect_distributions[action_id]:
        return None
    positive_branches = [
        branch for branch in unconditional if float(branch.probability) > 0.0
    ]
    return positive_branches[0] if positive_branches else None


def _observation_branches_deterministic(branches: list[Any]) -> bool:
    return _positive_branch_count(branches) <= 1


def _observation_rule_ids_by_action(
    *,
    data: dict,
    observation_rule_condition_checks: list[BitwiseGoalCheck],
) -> tuple[list[set[int]], set[int]]:
    layout = data.get("index_layout") or {}
    predicate_names = list(layout.get("predicates", []))
    action_names = list(layout.get("actions", []))
    action_anchor_masks = _build_observation_action_anchor_masks(predicate_names, action_names)
    action_rule_ids: list[set[int]] = [set() for _ in range(data["total_actions"])]
    generic_rule_ids: set[int] = set()
    for rule_id, check in enumerate(observation_rule_condition_checks):
        allowed_action_ids = _classify_observation_rule_action_ids(check, action_anchor_masks)
        if not allowed_action_ids:
            generic_rule_ids.add(rule_id)
        else:
            for action_id in allowed_action_ids:
                if 0 <= action_id < len(action_rule_ids):
                    action_rule_ids[action_id].add(rule_id)
    return action_rule_ids, generic_rule_ids


def _gen_fully_observed_observation_data(data: dict, nw: int) -> str:
    """Emit data/helper that appends fully observed state predicates into OBS_TYPE."""
    del nw
    predicate_indices = _fully_observed_predicate_indices(data)
    observable_count = int(data["grounded_observables_count"])
    if observable_count > 64:
        raise ValueError(
            "DESPOT OBS_TYPE is uint64_t, but the domain requires "
            f"{observable_count} explicit observation bits."
        )
    available_fully_observed_bits = 64 - observable_count
    if len(predicate_indices) > available_fully_observed_bits:
        warnings.warn(
            "DESPOT OBS_TYPE has no room for all fully observed state evidence; "
            f"keeping {available_fully_observed_bits} of {len(predicate_indices)} "
            "auxiliary predicate bits after preserving every explicit observable.",
            RuntimeWarning,
            stacklevel=2,
        )
        predicate_indices = predicate_indices[:available_fully_observed_bits]
    extended_obs_bits = observable_count + len(predicate_indices)

    L: list[str] = []
    L.append(f"static const int N_EXPLICIT_OBS_BITS = {observable_count};")
    L.append(f"static const int N_FULLY_OBSERVED_PREDICATES = {len(predicate_indices)};")
    L.append(f"static const int N_EXTENDED_OBS_BITS = {extended_obs_bits};")
    if predicate_indices:
        predicate_text = ", ".join(str(index) for index in predicate_indices)
        obs_bit_text = ", ".join(str(observable_count + offset) for offset in range(len(predicate_indices)))
        L.append(f"static const int FULLY_OBSERVED_PREDICATE_INDEX[{len(predicate_indices)}] = {{{predicate_text}}};")
        L.append(f"static const int FULLY_OBSERVED_OBS_BIT[{len(predicate_indices)}] = {{{obs_bit_text}}};")
    else:
        L.append("static const int* FULLY_OBSERVED_PREDICATE_INDEX = nullptr;")
        L.append("static const int* FULLY_OBSERVED_OBS_BIT = nullptr;")
    L.append("static OBS_TYPE AppendFullyObservedStateToObs(const uint64_t* bits, OBS_TYPE obs) {")
    L.append("    for (int i = 0; i < N_FULLY_OBSERVED_PREDICATES; ++i) {")
    L.append("        int predicate_index = FULLY_OBSERVED_PREDICATE_INDEX[i];")
    L.append("        int obs_bit = FULLY_OBSERVED_OBS_BIT[i];")
    L.append("        uint64_t predicate_mask = 1ULL << (predicate_index % 64);")
    L.append("        if ((bits[predicate_index / 64] & predicate_mask) != 0ULL) {")
    L.append("            obs |= (1ULL << obs_bit);")
    L.append("        } else {")
    L.append("            obs &= ~(1ULL << obs_bit);")
    L.append("        }")
    L.append("    }")
    L.append("    return obs;")
    L.append("}")
    L.append("static OBS_TYPE ExplicitObservationMask() {")
    L.append("    if (N_EXPLICIT_OBS_BITS >= 64) return ~static_cast<OBS_TYPE>(0);")
    L.append("    return (static_cast<OBS_TYPE>(1) << N_EXPLICIT_OBS_BITS) - 1;")
    L.append("}")
    return "\n".join(L)


def _gen_deterministic_noop_filter_data_from_bitwise_model(
    data: dict,
    model: BitwiseModelSkeleton,
    nw: int,
    *,
    action_effect_distributions: list[list[BitwiseEffectBranch]] | None = None,
    action_effect_condition_checks: list[list[BitwiseGoalCheck]] | None = None,
    action_conditional_effect_distributions: list[list[list[BitwiseEffectBranch]]] | None = None,
) -> str:
    ignored_mask = _no_change_ignored_mask(data)
    action_effect_distributions = (
        model.action_effect_distributions
        if action_effect_distributions is None
        else action_effect_distributions
    )
    action_effect_condition_checks = (
        model.action_effect_condition_checks
        if action_effect_condition_checks is None
        else action_effect_condition_checks
    )
    action_conditional_effect_distributions = (
        model.action_conditional_effect_distributions
        if action_conditional_effect_distributions is None
        else action_conditional_effect_distributions
    )
    action_rule_ids, generic_rule_ids = _observation_rule_ids_by_action(
        data=data,
        observation_rule_condition_checks=model.observation_rule_condition_checks,
    )
    noop_filter_flags: list[bool] = []
    set_masks: list[int] = []
    clear_masks: list[int] = []
    for action_id in range(data["total_actions"]):
        effect_branch = _deterministic_unconditional_effect_branch(
            action_id,
            action_effect_distributions,
            action_effect_condition_checks,
            action_conditional_effect_distributions,
        )
        if effect_branch is None:
            noop_filter_flags.append(False)
            set_masks.append(0)
            clear_masks.append(0)
            continue
        relevant_rule_ids = action_rule_ids[action_id] | generic_rule_ids
        obs_is_deterministic = all(
            _observation_branches_deterministic(model.observation_rule_distributions[rule_id])
            for rule_id in relevant_rule_ids
        )
        real_effect_mask = _effect_branch_real_change_mask(effect_branch, ignored_mask)
        noop_filter_flags.append(obs_is_deterministic and real_effect_mask != 0)
        set_masks.append(int(effect_branch.set_mask) & ~ignored_mask)
        clear_masks.append(int(effect_branch.clear_mask) & ~ignored_mask)

    L: list[str] = []
    flags_text = ", ".join("true" if flag else "false" for flag in noop_filter_flags)
    L.append(
        f"static const bool ACTION_DETERMINISTIC_NOOP_FILTER[{data['total_actions']}] = "
        f"{{{flags_text}}};"
    )
    for action_id in range(data["total_actions"]):
        L.append(_words_to_c_decl(
            f"ACTION_DETERMINISTIC_SET_{action_id}",
            _int_to_words(set_masks[action_id], nw),
        ))
        L.append(_words_to_c_decl(
            f"ACTION_DETERMINISTIC_CLEAR_{action_id}",
            _int_to_words(clear_masks[action_id], nw),
        ))
    L.append(f"static const uint64_t* ACTION_DETERMINISTIC_SET_MASKS[{data['total_actions']}] = {{")
    for action_id in range(data["total_actions"]):
        comma = "," if action_id < data["total_actions"] - 1 else ""
        L.append(f"    ACTION_DETERMINISTIC_SET_{action_id}{comma}")
    L.append("};")
    L.append(f"static const uint64_t* ACTION_DETERMINISTIC_CLEAR_MASKS[{data['total_actions']}] = {{")
    for action_id in range(data["total_actions"]):
        comma = "," if action_id < data["total_actions"] - 1 else ""
        L.append(f"    ACTION_DETERMINISTIC_CLEAR_{action_id}{comma}")
    L.append("};")
    return "\n".join(L)


def _gen_disabled_deterministic_noop_filter_data(data: dict, nw: int) -> str:
    L: list[str] = []
    flags_text = ", ".join("false" for _ in range(data["total_actions"]))
    L.append(
        f"static const bool ACTION_DETERMINISTIC_NOOP_FILTER[{data['total_actions']}] = "
        f"{{{flags_text}}};"
    )
    zero_words = [0] * nw
    for action_id in range(data["total_actions"]):
        L.append(_words_to_c_decl(f"ACTION_DETERMINISTIC_SET_{action_id}", zero_words))
        L.append(_words_to_c_decl(f"ACTION_DETERMINISTIC_CLEAR_{action_id}", zero_words))
    L.append(f"static const uint64_t* ACTION_DETERMINISTIC_SET_MASKS[{data['total_actions']}] = {{")
    for action_id in range(data["total_actions"]):
        comma = "," if action_id < data["total_actions"] - 1 else ""
        L.append(f"    ACTION_DETERMINISTIC_SET_{action_id}{comma}")
    L.append("};")
    L.append(f"static const uint64_t* ACTION_DETERMINISTIC_CLEAR_MASKS[{data['total_actions']}] = {{")
    for action_id in range(data["total_actions"]):
        comma = "," if action_id < data["total_actions"] - 1 else ""
        L.append(f"    ACTION_DETERMINISTIC_CLEAR_{action_id}{comma}")
    L.append("};")
    return "\n".join(L)


def _gen_state_compare_helpers() -> str:
    """Emit helpers for world-state change checks under ignored predicate masks."""
    L: list[str] = []
    L.append("template<int NW>")
    L.append(
        "inline bool bw_any_change_outside_mask(const uint64_t* lhs, const uint64_t* rhs, const uint64_t* ignore_mask) {"
    )
    L.append("    for (int i = 0; i < NW; ++i) {")
    L.append("        if (((lhs[i] ^ rhs[i]) & ~ignore_mask[i]) != 0ULL) return true;")
    L.append("    }")
    L.append("    return false;")
    L.append("}")
    return "\n".join(L)

def _gen_source(data: dict, cn: str, sn: str) -> str:
    nw = _n_words(data["grounded_predicates_count"])
    model = data["model_instance"]
    P = []
    P.append(f'#include "bitwise_pomdp_model.h"')
    P.append('#include <despot/core/builtin_lower_bounds.h>')
    P.append('#include <despot/core/builtin_policy.h>')
    P.append('#include <despot/core/builtin_upper_bounds.h>')
    P.append('#include <despot/core/particle_belief.h>')
    P.append('#include <despot/core/globals.h>')
    P.append('#include <despot/util/logging.h>')
    P.append('#include <despot/util/random.h>')
    P.append('#include <despot/util/seeds.h>')
    P.append('#include <iostream>')
    P.append('#include <cstring>')
    P.append('#include <cstdlib>')
    P.append('#include <fstream>')
    P.append('#include <sstream>')
    P.append('#include <functional>')
    P.append('#include <cctype>')
    P.append('#include <stdexcept>')
    P.append('#include <random>')
    P.append('#include <map>')
    P.append('#include <vector>')
    P.append('#include <chrono>')
    P.append("using namespace std;")
    P.append("namespace despot {")
    P.append("")
    P.append(f"double {cn}::GOAL_REWARD = {data.get('goal_reward', 100.0)};")
    P.append(
        f"double {cn}::REPORT_GOAL_FAILURE_PENALTY = "
        f"{data.get('report_goal_failure_penalty', -10.0)};"
    )
    P.append(f"double {cn}::ACTION_INVALID_PENALTY = -10.0;")
    P.append(f"double {cn}::ACTION_NO_CHANGE_PENALTY = -10.0;")
    P.append(f"double {cn}::DEAD_END_PENALTY = -500.0;")
    P.append(f"double {cn}::DP_EXPLOIT_PROB = 0.9;")
    P.append(_gen_no_change_ignored_mask_data(data, nw))
    P.append(_gen_state_compare_helpers())
    P.append("")
    P.append(_gen_profile_helpers(cn))

    # Semantic name tables (for debug printing)
    P.append(_gen_semantic_data(data))
    P.append(_gen_fully_observed_observation_data(data, nw))
    # Goal check data
    P.append(_gen_goal_data(model, nw))
    # Precondition functions
    P.append(_gen_precond_fns(data, nw))
    # Action effect data
    P.append(_gen_effect_data(data, nw))
    P.append(_gen_disabled_deterministic_noop_filter_data(data, nw))
    # Observation data
    P.append(_gen_obs_data(data, nw))
    # Default policy data
    P.append(_gen_dp_data(data, nw))
    # Init belief data
    P.append(_gen_belief_data(data, nw))
    # Constructor
    P.append(f"{cn}::{cn}() {{}}")
    P.append(f"int {cn}::NumActions() const {{ return TOTAL_ACTIONS + 1; }}")
    P.append("")
    # IsGoal
    P.append(f"bool {cn}::IsGoal(const uint64_t* bits) const {{")
    P.append(f"    for (int i = 0; i < GOAL_N_CLAUSES; ++i)")
    P.append(f"        if (bw_check_clause<N_WORDS>(bits, GOAL_TRUE_MASKS[i], GOAL_FALSE_MASKS[i])) return true;")
    P.append(f"    return false;")
    P.append("}")
    P.append("")
    # CheckActionPrecondition
    P.append(f"bool {cn}::CheckActionPrecondition(int action, const uint64_t* bits) const {{")
    P.append(f"    if (action < 0 || action >= TOTAL_ACTIONS) return false;")
    if data.get("enable_report_goal_action", False):
        P.append(f"    if (action == 0) return IsGoal(bits);")
    P.append(f"    return PRECONDITION_TABLE[action](bits);")
    P.append("}")
    P.append("")
    # GetLegalActions
    P.append(_gen_legal_actions(cn, sn))
    P.append(_gen_deterministic_action_change_fn(cn))
    P.append(_gen_repeated_observation_prune_fn(cn))
    # ForwardAction
    P.append(_gen_fwd_fn(data, cn, nw))
    # Observe
    P.append(_gen_obs_fn(data, cn, nw))
    # Step (5-arg override: ignores rand_num, delegates to 4-arg)
    P.append(f"bool {cn}::Step(State& state, double rand_num, ACT_TYPE action,")
    P.append(f"        double& reward, OBS_TYPE& obs) const {{")
    P.append(f"    (void)rand_num;")
    P.append(f"    return Step(state, action, reward, obs);")
    P.append("}")
    P.append("")
    # Step (4-arg: uses state's internal rng_seed)
    P.append(f"bool {cn}::Step(State& state, ACT_TYPE action,")
    P.append(f"        double& reward, OBS_TYPE& obs) const {{")
    P.append(f"    bool profile_enabled = {cn}ProfileEnabled();")
    P.append(f"    auto step_start = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{{}};")
    P.append(f"    {sn}& s = static_cast<{sn}&>(state);")
    P.append(f"    if (profile_enabled) {{")
    P.append(f"        G_{cn.upper()}_PROFILE_STATS.step_calls++;")
    P.append(f"    }}")
    if data.get("enable_report_goal_action", False):
        P.append(f"    if (action == 0) {{")
        P.append(f"        bool report_success = IsGoal(s.bits);")
        P.append(f"        reward = report_success ? GOAL_REWARD : REPORT_GOAL_FAILURE_PENALTY;")
        P.append(f"        obs = Observe(action, s.bits, s.rng_seed);")
        P.append(f"        s.last_obs = obs;")
        P.append(f"        s.has_last_obs = true;")
        P.append(f"        if (profile_enabled) {{")
        P.append(f"            G_{cn.upper()}_PROFILE_STATS.step_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();")
        P.append(f"        }}")
        P.append(f"        return report_success;")
        P.append(f"    }}")
    P.append(f"    if (action == TOTAL_ACTIONS) {{")
    P.append(f"        reward = DEAD_END_PENALTY; obs = 0;")
    P.append(f"        if (profile_enabled) {{")
    P.append(f"            G_{cn.upper()}_PROFILE_STATS.step_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();")
    P.append(f"        }}")
    P.append(f"        return true;")
    P.append(f"    }}")
    P.append(f"    if (!CheckActionPrecondition(action, s.bits)) {{")
    P.append(f"        if (profile_enabled) {{")
    P.append(f"            G_{cn.upper()}_PROFILE_STATS.step_invalid_precondition++;")
    P.append(f"            G_{cn.upper()}_PROFILE_STATS.step_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();")
    P.append(f"        }}")
    P.append(f"        reward = ACTION_INVALID_PENALTY; obs = 0;")
    P.append(f"        return false;")
    P.append(f"    }}")
    P.append(f"    uint64_t orig_bits[N_WORDS]; bw_copy<N_WORDS>(orig_bits, s.bits);")
    P.append(f"    OBS_TYPE prev_obs = s.last_obs;")
    P.append(f"    bool had_prev_obs = s.has_last_obs;")
    P.append(f"    auto forward_start = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{{}};")
    P.append(f"    ForwardAction(action, s.bits, s.rng_seed, reward);")
    P.append(f"    if (profile_enabled) {{")
    P.append(f"        G_{cn.upper()}_PROFILE_STATS.forward_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - forward_start).count();")
    P.append(f"    }}")
    P.append(f"    obs = Observe(action, s.bits, s.rng_seed);")
    P.append(
        f"    bool world_state_changed = bw_any_change_outside_mask<N_WORDS>(orig_bits, s.bits, NO_CHANGE_IGNORED_PREDICATE_MASK);"
    )
    P.append(f"    bool world_state_unchanged = !world_state_changed;")
    P.append(f"    bool obs_unchanged = had_prev_obs && (obs == prev_obs);")
    P.append(f"    if (action >= 0 && action < TOTAL_ACTIONS && ACTION_HAS_OBSERVATION_MODEL[action] && world_state_unchanged && obs_unchanged) {{")
    P.append(f"        reward += ACTION_NO_CHANGE_PENALTY;")
    P.append(f"        if (profile_enabled) {{")
    P.append(f"            G_{cn.upper()}_PROFILE_STATS.step_no_change_penalty_applied++;")
    P.append(f"        }}")
    P.append(f"    }}")
    P.append(f"    s.last_obs = obs;")
    P.append(f"    s.has_last_obs = true;")
    P.append(f"    if (profile_enabled) {{")
    P.append(f"        G_{cn.upper()}_PROFILE_STATS.step_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();")
    P.append(f"    }}")
    P.append(f"    return IsGoal(s.bits);")
    P.append("}")
    P.append("")
    # ObsProb
    P.append(f"double {cn}::ObsProb(OBS_TYPE obs, const State& state, ACT_TYPE action) const {{")
    P.append(f"    return 1.0; // particle-based; not needed for DESPOT")
    P.append("}")
    P.append("")
    # InitialBelief
    P.append(_gen_init_belief(data, cn, sn, nw))
    # GetMaxReward
    P.append(f"double {cn}::GetMaxReward() const {{ return 1e6; }}")
    P.append("")
    # GetBestAction
    P.append(f"ValuedAction {cn}::GetBestAction() const {{ return ValuedAction(TOTAL_ACTIONS, 0.0); }}")
    P.append("")
    # GetDefaultPolicyAction
    P.append(_gen_dp_action_fn(data, cn))
    # Bounds
    P.append(_gen_upper(cn))
    P.append(_gen_lower(data, cn, sn))
    # Memory
    P.append(_gen_mem(cn, sn))
    # Display
    P.append(_gen_disp(cn, sn))
    # Belief JSON path setter (for pybind integration)
    P.append(f"void set_init_belief_json_path(const std::string& path) {{")
    P.append(f"    INIT_BELIEF_JSON_PATH = path;")
    P.append(f"}}")
    P.append("")
    P.append(f"void update_hyperparams(double goal_reward, double report_goal_failure_penalty,")
    P.append(f"    double action_invalid_penalty,")
    P.append(f"    double action_no_change_penalty,")
    P.append(f"    double dead_end_penalty,")
    P.append(f"    double dp_exploit_prob, double time_per_move, int num_scenarios,")
    P.append(f"    int search_depth, int max_policy_sim_len, int sim_len,")
    P.append(f"    double discount, double pruning_constant, double xi,")
    P.append(f"    unsigned int root_seed, bool silence) {{")
    P.append(f"    {cn}::GOAL_REWARD = goal_reward;")
    P.append(f"    {cn}::REPORT_GOAL_FAILURE_PENALTY = report_goal_failure_penalty;")
    P.append(f"    {cn}::ACTION_INVALID_PENALTY = action_invalid_penalty;")
    P.append(f"    {cn}::ACTION_NO_CHANGE_PENALTY = action_no_change_penalty;")
    P.append(f"    {cn}::DEAD_END_PENALTY = dead_end_penalty;")
    P.append(f"    {cn}::DP_EXPLOIT_PROB = dp_exploit_prob;")
    P.append(f"    Globals::config.time_per_move = time_per_move;")
    P.append(f"    Globals::config.num_scenarios = num_scenarios;")
    P.append(f"    Globals::config.search_depth = search_depth;")
    P.append(f"    Globals::config.max_policy_sim_len = max_policy_sim_len;")
    P.append(f"    Globals::config.sim_len = sim_len;")
    P.append(f"    Globals::config.discount = discount;")
    P.append(f"    Globals::config.pruning_constant = pruning_constant;")
    P.append(f"    Globals::config.xi = xi;")
    P.append(f"    Globals::config.root_seed = root_seed;")
    P.append(f"    Seeds::root_seed(Globals::config.root_seed);")
    P.append(f"    Random::RANDOM = Random(Seeds::Next());")
    P.append(f"    Globals::config.silence = silence;")
    P.append(f"    if (silence) logging::level(logging::ERROR);")
    P.append(f"    else logging::level(logging::INFO);")
    P.append(f"}}")
    P.append("")
    # Base class GetLegalActions fallback (not defined in DESPOT source)
    P.append("std::vector<ACT_TYPE> DSPOMDP::GetLegalActions(std::vector<State*> particles) const {")
    P.append("    std::vector<ACT_TYPE> actions;")
    P.append("    for (int a = 0; a < NumActions(); ++a) actions.push_back(a);")
    P.append("    return actions;")
    P.append("}")
    P.append("")
    P.append("} // namespace despot")
    P.append("")
    return "\n".join(P)


def _gen_source_from_bitwise_model(
    data: dict,
    model: BitwiseModelSkeleton,
    cn: str,
    sn: str,
) -> str:
    nw = _n_words(data["grounded_predicates_count"])
    expanded_actions = _expand_bitwise_model_codegen_actions(model)
    enable_report_goal_action = expanded_actions["enable_report_goal_action"]
    raw_goal_check = expanded_actions["raw_goal_check"]
    action_precondition_checks = expanded_actions["action_precondition_checks"]
    action_effect_condition_checks = expanded_actions["action_effect_condition_checks"]
    action_effect_distributions = expanded_actions["action_effect_distributions"]
    action_conditional_effect_distributions = expanded_actions["action_conditional_effect_distributions"]
    default_policy_action_ids = expanded_actions["default_policy_action_ids"]
    P = []
    P.append(f'#include "bitwise_pomdp_model.h"')
    P.append('#include <despot/core/builtin_lower_bounds.h>')
    P.append('#include <despot/core/builtin_policy.h>')
    P.append('#include <despot/core/builtin_upper_bounds.h>')
    P.append('#include <despot/core/particle_belief.h>')
    P.append('#include <despot/core/globals.h>')
    P.append('#include <despot/util/logging.h>')
    P.append('#include <despot/util/random.h>')
    P.append('#include <despot/util/seeds.h>')
    P.append('#include <iostream>')
    P.append('#include <cstring>')
    P.append('#include <cstdlib>')
    P.append('#include <fstream>')
    P.append('#include <sstream>')
    P.append('#include <functional>')
    P.append('#include <cctype>')
    P.append('#include <stdexcept>')
    P.append('#include <random>')
    P.append('#include <map>')
    P.append('#include <vector>')
    P.append('#include <chrono>')
    P.append("using namespace std;")
    P.append("namespace despot {")
    P.append("")
    P.append(f"double {cn}::GOAL_REWARD = {data.get('goal_reward', 100.0)};")
    P.append(
        f"double {cn}::REPORT_GOAL_FAILURE_PENALTY = "
        f"{data.get('report_goal_failure_penalty', -10.0)};"
    )
    P.append(f"double {cn}::ACTION_INVALID_PENALTY = -10.0;")
    P.append(f"double {cn}::ACTION_NO_CHANGE_PENALTY = -10.0;")
    P.append(f"double {cn}::DEAD_END_PENALTY = -500.0;")
    P.append(f"double {cn}::DP_EXPLOIT_PROB = 0.9;")
    P.append("")
    P.append(_gen_profile_helpers(cn))
    P.append(_gen_semantic_data(data))
    P.append(_gen_fully_observed_observation_data(data, nw))
    P.append(_gen_no_change_ignored_mask_data(data, nw))
    P.append(_gen_state_compare_helpers())
    P.append(_gen_deterministic_noop_filter_data_from_bitwise_model(
        data,
        model,
        nw,
        action_effect_distributions=action_effect_distributions,
        action_effect_condition_checks=action_effect_condition_checks,
        action_conditional_effect_distributions=action_conditional_effect_distributions,
    ))
    P.append(_gen_precond_fns_from_goal_checks(action_precondition_checks, nw))
    P.append(_gen_obs_data_from_bitwise_model(data, model, nw))
    P.append(
        _gen_dp_data_from_bitwise_model(
            model,
            nw,
            default_policy_action_ids=default_policy_action_ids,
            total_actions=data["total_actions"],
        )
    )
    P.append(_gen_belief_data(data, nw))
    P.append(f"{cn}::{cn}() {{}}")
    P.append(f"int {cn}::NumActions() const {{ return TOTAL_ACTIONS + 1; }}")
    P.append("")
    P.append(_gen_is_goal_fn_from_goal_check(raw_goal_check, cn, nw))
    P.append("")
    P.append(f"bool {cn}::CheckActionPrecondition(int action, const uint64_t* bits) const {{")
    P.append(f"    if (action < 0 || action >= TOTAL_ACTIONS) return false;")
    if enable_report_goal_action:
        P.append(f"    if (action == 0) return IsGoal(bits);")
    P.append(f"    return PRECONDITION_TABLE[action](bits);")
    P.append("}")
    P.append("")
    P.append(_gen_legal_actions(cn, sn))
    P.append(_gen_deterministic_action_change_fn(cn))
    P.append(_gen_repeated_observation_prune_fn(cn))
    P.append(
        _gen_fwd_fn_from_bitwise_model(
            action_effect_distributions=action_effect_distributions,
            action_effect_condition_checks=action_effect_condition_checks,
            action_conditional_effect_distributions=action_conditional_effect_distributions,
            raw_goal_check=raw_goal_check,
            enable_report_goal_action=enable_report_goal_action,
            cn=cn,
            nw=nw,
        )
    )
    P.append(_gen_obs_fn_from_bitwise_model(model, cn))
    P.append(
        _gen_probability_update_tables(
            action_effect_distributions=action_effect_distributions,
            observation_rule_distributions=model.observation_rule_distributions,
        )
    )
    P.append(f"bool {cn}::Step(State& state, double rand_num, ACT_TYPE action,")
    P.append(f"        double& reward, OBS_TYPE& obs) const {{")
    P.append(f"    (void)rand_num;")
    P.append(f"    return Step(state, action, reward, obs);")
    P.append("}")
    P.append("")
    P.append(f"bool {cn}::Step(State& state, ACT_TYPE action,")
    P.append(f"        double& reward, OBS_TYPE& obs) const {{")
    P.append(f"    bool profile_enabled = {cn}ProfileEnabled();")
    P.append(f"    auto step_start = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{{}};")
    P.append(f"    {sn}& s = static_cast<{sn}&>(state);")
    P.append(f"    if (profile_enabled) {{")
    P.append(f"        G_{cn.upper()}_PROFILE_STATS.step_calls++;")
    P.append(f"    }}")
    if enable_report_goal_action:
        P.append(f"    if (action == 0) {{")
        P.append(f"        bool report_success = IsGoal(s.bits);")
        P.append(f"        reward = report_success ? GOAL_REWARD : REPORT_GOAL_FAILURE_PENALTY;")
        P.append(f"        obs = Observe(action, s.bits, s.rng_seed);")
        P.append(f"        s.last_obs = obs;")
        P.append(f"        s.has_last_obs = true;")
        P.append(f"        if (profile_enabled) {{")
        P.append(f"            G_{cn.upper()}_PROFILE_STATS.step_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();")
        P.append(f"        }}")
        P.append(f"        return report_success;")
        P.append(f"    }}")
    P.append(f"    if (action == TOTAL_ACTIONS) {{")
    P.append(f"        reward = DEAD_END_PENALTY; obs = 0;")
    P.append(f"        if (profile_enabled) {{")
    P.append(f"            G_{cn.upper()}_PROFILE_STATS.step_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();")
    P.append(f"        }}")
    P.append(f"        return true;")
    P.append(f"    }}")
    P.append(f"    if (!CheckActionPrecondition(action, s.bits)) {{")
    P.append(f"        if (profile_enabled) {{")
    P.append(f"            G_{cn.upper()}_PROFILE_STATS.step_invalid_precondition++;")
    P.append(f"            G_{cn.upper()}_PROFILE_STATS.step_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();")
    P.append(f"        }}")
    P.append(f"        reward = ACTION_INVALID_PENALTY; obs = 0;")
    P.append(f"        return false;")
    P.append(f"    }}")
    P.append(f"    uint64_t orig_bits[N_WORDS]; bw_copy<N_WORDS>(orig_bits, s.bits);")
    P.append(f"    OBS_TYPE prev_obs = s.last_obs;")
    P.append(f"    bool had_prev_obs = s.has_last_obs;")
    P.append(f"    auto forward_start = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{{}};")
    P.append(f"    ForwardAction(action, s.bits, s.rng_seed, reward);")
    P.append(f"    if (profile_enabled) {{")
    P.append(f"        G_{cn.upper()}_PROFILE_STATS.forward_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - forward_start).count();")
    P.append(f"    }}")
    P.append(f"    obs = Observe(action, s.bits, s.rng_seed);")
    P.append(
        f"    bool world_state_changed = bw_any_change_outside_mask<N_WORDS>(orig_bits, s.bits, NO_CHANGE_IGNORED_PREDICATE_MASK);"
    )
    P.append(f"    bool world_state_unchanged = !world_state_changed;")
    P.append(f"    bool obs_unchanged = had_prev_obs && (obs == prev_obs);")
    P.append(f"    if (action >= 0 && action < TOTAL_ACTIONS && ACTION_HAS_OBSERVATION_MODEL[action] && world_state_unchanged && obs_unchanged) {{")
    P.append(f"        reward += ACTION_NO_CHANGE_PENALTY;")
    P.append(f"        if (profile_enabled) {{")
    P.append(f"            G_{cn.upper()}_PROFILE_STATS.step_no_change_penalty_applied++;")
    P.append(f"        }}")
    P.append(f"    }}")
    P.append(f"    s.last_obs = obs;")
    P.append(f"    s.has_last_obs = true;")
    P.append(f"    if (profile_enabled) {{")
    P.append(f"        G_{cn.upper()}_PROFILE_STATS.step_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();")
    P.append(f"    }}")
    if enable_report_goal_action:
        P.append(f"    return false;")
    else:
        P.append(f"    return IsGoal(s.bits);")
    P.append("}")
    P.append("")
    P.append(f"double {cn}::ObsProb(OBS_TYPE obs, const State& state, ACT_TYPE action) const {{")
    P.append(f"    return 1.0; // particle-based; not needed for DESPOT")
    P.append("}")
    P.append("")
    P.append(_gen_init_belief(data, cn, sn, nw))
    P.append(f"double {cn}::GetMaxReward() const {{ return 1e6; }}")
    P.append("")
    P.append(f"ValuedAction {cn}::GetBestAction() const {{ return ValuedAction(TOTAL_ACTIONS, 0.0); }}")
    P.append("")
    P.append(_gen_dp_action_fn(data, cn))
    P.append(_gen_upper(cn))
    P.append(_gen_lower(data, cn, sn))
    P.append(_gen_mem(cn, sn))
    P.append(_gen_disp(cn, sn))
    P.append(f"void set_init_belief_json_path(const std::string& path) {{")
    P.append(f"    INIT_BELIEF_JSON_PATH = path;")
    P.append(f"}}")
    P.append("")
    P.append(f"void update_hyperparams(double goal_reward, double report_goal_failure_penalty,")
    P.append(f"    double action_invalid_penalty,")
    P.append(f"    double action_no_change_penalty,")
    P.append(f"    double dead_end_penalty,")
    P.append(f"    double dp_exploit_prob, double time_per_move, int num_scenarios,")
    P.append(f"    int search_depth, int max_policy_sim_len, int sim_len,")
    P.append(f"    double discount, double pruning_constant, double xi,")
    P.append(f"    unsigned int root_seed, bool silence) {{")
    P.append(f"    {cn}::GOAL_REWARD = goal_reward;")
    P.append(f"    {cn}::REPORT_GOAL_FAILURE_PENALTY = report_goal_failure_penalty;")
    P.append(f"    {cn}::ACTION_INVALID_PENALTY = action_invalid_penalty;")
    P.append(f"    {cn}::ACTION_NO_CHANGE_PENALTY = action_no_change_penalty;")
    P.append(f"    {cn}::DEAD_END_PENALTY = dead_end_penalty;")
    P.append(f"    {cn}::DP_EXPLOIT_PROB = dp_exploit_prob;")
    P.append(f"    Globals::config.time_per_move = time_per_move;")
    P.append(f"    Globals::config.num_scenarios = num_scenarios;")
    P.append(f"    Globals::config.search_depth = search_depth;")
    P.append(f"    Globals::config.max_policy_sim_len = max_policy_sim_len;")
    P.append(f"    Globals::config.sim_len = sim_len;")
    P.append(f"    Globals::config.discount = discount;")
    P.append(f"    Globals::config.pruning_constant = pruning_constant;")
    P.append(f"    Globals::config.xi = xi;")
    P.append(f"    Globals::config.root_seed = root_seed;")
    P.append(f"    Seeds::root_seed(Globals::config.root_seed);")
    P.append(f"    Random::RANDOM = Random(Seeds::Next());")
    P.append(f"    Globals::config.silence = silence;")
    P.append(f"    if (silence) logging::level(logging::ERROR);")
    P.append(f"    else logging::level(logging::INFO);")
    P.append(f"}}")
    P.append("")
    P.append("std::vector<ACT_TYPE> DSPOMDP::GetLegalActions(std::vector<State*> particles) const {")
    P.append("    std::vector<ACT_TYPE> actions;")
    P.append("    for (int a = 0; a < NumActions(); ++a) actions.push_back(a);")
    P.append("    return actions;")
    P.append("}")
    P.append("")
    P.append("} // namespace despot")
    P.append("")
    return "\n".join(P)


def _gen_semantic_data(data: dict) -> str:
    """Emit static string arrays for predicate/action/observable names."""
    layout = data.get("index_layout")
    L = []

    # Predicate names
    pc = data["grounded_predicates_count"]
    if layout and "predicates" in layout:
        preds = layout["predicates"]
    else:
        preds = [f"pred_{i}" for i in range(pc)]
    L.append(f"static const char* PREDICATE_NAMES[{pc}] = {{")
    for i, name in enumerate(preds[:pc]):
        escaped = name.replace('\\', '\\\\').replace('"', '\\"')
        comma = "," if i < pc - 1 else ""
        L.append(f'    "{escaped}"{comma}')
    L.append("};")

    # Action names
    ta = data["total_actions"]
    if layout and "actions" in layout:
        acts = layout["actions"]
    else:
        acts = [f"action_{i}" for i in range(ta)]
    L.append(f"static const char* ACTION_NAMES[{ta}] = {{")
    for i, name in enumerate(acts[:ta]):
        escaped = name.replace('\\', '\\\\').replace('"', '\\"')
        comma = "," if i < ta - 1 else ""
        L.append(f'    "{escaped}"{comma}')
    L.append("};")

    # Observable names
    oc = data["grounded_observables_count"]
    if layout and "observables" in layout:
        obs = layout["observables"]
    else:
        obs = [f"obs_{i}" for i in range(oc)]
    L.append(f"static const char* OBSERVABLE_NAMES[{oc}] = {{")
    for i, name in enumerate(obs[:oc]):
        escaped = name.replace('\\', '\\\\').replace('"', '\\"')
        comma = "," if i < oc - 1 else ""
        L.append(f'    "{escaped}"{comma}')
    L.append("};")
    L.append("")
    return "\n".join(L)


def _render_pomdpddl_atom(name: str, params: list[str]) -> str:
    if not params:
        return f"({name})"
    return "(" + " ".join([name, *params]) + ")"


def _render_cpp_clause_expr(clause: tuple[int, int], state_var: str, nw: int) -> str:
    required_true_mask, required_false_mask = clause
    if required_true_mask == 0 and required_false_mask == 0:
        return "true"
    parts: list[str] = []
    if required_true_mask != 0:
        parts.append(_emit_check_true(_int_to_words(required_true_mask, nw), state_var))
    if required_false_mask != 0:
        parts.append(_emit_check_false(_int_to_words(required_false_mask, nw), state_var))
    if len(parts) == 1:
        return parts[0]
    return "(" + " && ".join(parts) + ")"


def _render_cpp_goal_check_expr(
    goal_check: BitwiseGoalCheck | None,
    state_var: str,
    nw: int,
) -> str:
    if goal_check is None or not goal_check.clauses:
        return "false"
    rendered_clauses = [_render_cpp_clause_expr(clause, state_var, nw) for clause in goal_check.clauses]
    if len(rendered_clauses) == 1:
        return rendered_clauses[0]
    return "(" + " || ".join(rendered_clauses) + ")"


def _gen_is_goal_fn_from_goal_check(
    goal_check: BitwiseGoalCheck | None,
    cn: str,
    nw: int,
) -> str:
    expr = _render_cpp_goal_check_expr(goal_check, "bits", nw)
    return f"bool {cn}::IsGoal(const uint64_t* bits) const {{ return {expr}; }}"


def _extract_common_true_anchor_signature(
    goal_check: BitwiseGoalCheck | None,
    *,
    max_bits: int = 3,
) -> tuple[int, int] | None:
    if goal_check is None or not goal_check.clauses:
        return None
    common_true_mask: int | None = None
    for required_true_mask, _required_false_mask in goal_check.clauses:
        common_true_mask = (
            required_true_mask
            if common_true_mask is None
            else (common_true_mask & required_true_mask)
        )
        if common_true_mask == 0:
            return None
    if not common_true_mask:
        return None
    best_word_index: int | None = None
    best_word_mask = 0
    best_popcount = 0
    word_index = 0
    remaining_mask = common_true_mask
    while remaining_mask:
        word_mask = remaining_mask & ((1 << 64) - 1)
        if word_mask:
            popcount = word_mask.bit_count()
            if popcount > best_popcount:
                best_word_index = word_index
                best_word_mask = word_mask
                best_popcount = popcount
        remaining_mask >>= 64
        word_index += 1
    if best_word_index is None or best_word_mask == 0:
        return None
    selected_mask = 0
    while best_word_mask and selected_mask.bit_count() < max_bits:
        lowest_bit = best_word_mask & -best_word_mask
        selected_mask |= lowest_bit
        best_word_mask ^= lowest_bit
    return best_word_index, selected_mask


def _select_low_bits(mask: int, limit: int) -> int:
    selected = 0
    while mask and selected.bit_count() < limit:
        lowest_bit = mask & -mask
        selected |= lowest_bit
        mask ^= lowest_bit
    return selected


def _extract_common_bucket_signature(
    goal_check: BitwiseGoalCheck | None,
    *,
    max_true_bits: int = 2,
    max_false_bits: int = 2,
) -> tuple[int, int, int] | None:
    if goal_check is None or not goal_check.clauses:
        return None
    common_true_mask: int | None = None
    common_false_mask: int | None = None
    for required_true_mask, required_false_mask in goal_check.clauses:
        common_true_mask = (
            required_true_mask
            if common_true_mask is None
            else (common_true_mask & required_true_mask)
        )
        common_false_mask = (
            required_false_mask
            if common_false_mask is None
            else (common_false_mask & required_false_mask)
        )
    true_mask_full = common_true_mask or 0
    false_mask_full = common_false_mask or 0
    if true_mask_full == 0 and false_mask_full == 0:
        return None

    best_word_index: int | None = None
    best_true_word_mask = 0
    best_false_word_mask = 0
    best_score = 0
    word_index = 0
    true_remaining = true_mask_full
    false_remaining = false_mask_full
    while true_remaining or false_remaining:
        true_word_mask = true_remaining & ((1 << 64) - 1)
        false_word_mask = false_remaining & ((1 << 64) - 1)
        score = true_word_mask.bit_count() + false_word_mask.bit_count()
        if score > best_score:
            best_word_index = word_index
            best_true_word_mask = true_word_mask
            best_false_word_mask = false_word_mask
            best_score = score
        true_remaining >>= 64
        false_remaining >>= 64
        word_index += 1

    if best_word_index is None or best_score == 0:
        return None

    selected_true = _select_low_bits(best_true_word_mask, max_true_bits)
    selected_false = _select_low_bits(best_false_word_mask, max_false_bits)
    if selected_true == 0 and selected_false == 0:
        return None
    return best_word_index, selected_true, selected_false


def _extract_common_bucket_signature_parts(
    goal_check: BitwiseGoalCheck | None,
    *,
    max_words: int = 2,
    max_true_bits: int = 2,
    max_false_bits: int = 2,
) -> list[tuple[int, int, int]]:
    if goal_check is None or not goal_check.clauses:
        return []
    common_true_mask: int | None = None
    common_false_mask: int | None = None
    for required_true_mask, required_false_mask in goal_check.clauses:
        common_true_mask = (
            required_true_mask
            if common_true_mask is None
            else (common_true_mask & required_true_mask)
        )
        common_false_mask = (
            required_false_mask
            if common_false_mask is None
            else (common_false_mask & required_false_mask)
        )
    true_mask_full = common_true_mask or 0
    false_mask_full = common_false_mask or 0
    if true_mask_full == 0 and false_mask_full == 0:
        return []

    parts: list[tuple[int, int, int, int]] = []
    word_index = 0
    true_remaining = true_mask_full
    false_remaining = false_mask_full
    while true_remaining or false_remaining:
        true_word_mask = true_remaining & ((1 << 64) - 1)
        false_word_mask = false_remaining & ((1 << 64) - 1)
        score = true_word_mask.bit_count() + false_word_mask.bit_count()
        if score > 0:
            parts.append((score, word_index, true_word_mask, false_word_mask))
        true_remaining >>= 64
        false_remaining >>= 64
        word_index += 1

    parts.sort(key=lambda item: (-item[0], item[1]))
    selected: list[tuple[int, int, int]] = []
    for _score, selected_word_index, true_word_mask, false_word_mask in parts[:max_words]:
        selected_true = _select_low_bits(true_word_mask, max_true_bits)
        selected_false = _select_low_bits(false_word_mask, max_false_bits)
        if selected_true == 0 and selected_false == 0:
            continue
        selected.append((selected_word_index, selected_true, selected_false))
    return selected


def _extract_common_clause_masks(
    goal_check: BitwiseGoalCheck | None,
) -> tuple[int, int]:
    if goal_check is None or not goal_check.clauses:
        return 0, 0
    common_true_mask: int | None = None
    common_false_mask: int | None = None
    for required_true_mask, required_false_mask in goal_check.clauses:
        common_true_mask = (
            required_true_mask
            if common_true_mask is None
            else (common_true_mask & required_true_mask)
        )
        common_false_mask = (
            required_false_mask
            if common_false_mask is None
            else (common_false_mask & required_false_mask)
        )
    return common_true_mask or 0, common_false_mask or 0

_LAST_ACTION_MARKER_SUFFIX_RE = re.compile(r"_(?:success|fail|failure)(?:_\d+)?$")


def _marker_token_to_action_schema_name(marker_token: str) -> str | None:
    base_name = _LAST_ACTION_MARKER_SUFFIX_RE.sub("", marker_token.strip())
    if not base_name:
        return None
    return base_name


def _build_observation_action_anchor_masks(
    predicate_names: list[str],
    action_names: list[str],
) -> dict[int, int]:
    action_masks: dict[int, int] = {action_id: 0 for action_id in range(len(action_names))}
    action_str_to_id = {action_name: action_id for action_id, action_name in enumerate(action_names)}
    action_name_to_id: dict[str, int] = {}
    for action_id, action_name in enumerate(action_names):
        if not action_name.startswith("(") or not action_name.endswith(")"):
            continue
        payload = action_name[1:-1].split()
        if not payload:
            continue
        action_name_to_id[payload[0]] = action_id

    for bit_index, predicate_name in enumerate(predicate_names):
        action_id: int | None = None
        if predicate_name.startswith("(last-") and predicate_name.endswith(")"):
            candidate_action = "(" + predicate_name[len("(last-"):-1] + ")"
            action_id = action_str_to_id.get(candidate_action)
        elif predicate_name.startswith("(last-action ") and predicate_name.endswith(")"):
            payload = predicate_name[len("(last-action "):-1]
            if payload.startswith("act_"):
                candidate_name = payload[len("act_"):].replace("_", "-")
                action_id = action_name_to_id.get(candidate_name)
        elif predicate_name.startswith("(") and predicate_name.endswith(")"):
            payload = predicate_name[1:-1].split()
            if len(payload) >= 2 and payload[0] in {"last_action_1_param", "last_action_2_param"}:
                action_schema_name = _marker_token_to_action_schema_name(payload[1])
                if action_schema_name is not None:
                    grounded_action = "(" + " ".join([action_schema_name, *payload[2:]]) + ")"
                    action_id = action_str_to_id.get(grounded_action)
                    if action_id is None:
                        action_id = action_name_to_id.get(action_schema_name)
            elif len(payload) >= 2 and payload[0].startswith("last_action__last_action_"):
                marker_token = payload[0].removeprefix("last_action__")
                marker_parts = marker_token.split("__", 1)
                if len(marker_parts) == 2:
                    _last_action_kind, raw_marker_name = marker_parts
                    action_schema_name = _marker_token_to_action_schema_name(raw_marker_name)
                    if action_schema_name is not None:
                        grounded_action = "(" + " ".join([action_schema_name, *payload[1:]]) + ")"
                        action_id = action_str_to_id.get(grounded_action)
                        if action_id is None:
                            action_id = action_name_to_id.get(action_schema_name)
        if action_id is None:
            continue
        action_masks[action_id] |= 1 << bit_index

    return {action_id: mask for action_id, mask in action_masks.items() if mask != 0}


def _clause_action_ids_from_mask(
    required_true_mask: int,
    action_anchor_masks: dict[int, int],
) -> set[int]:
    action_ids: set[int] = set()
    if required_true_mask == 0:
        return action_ids
    for action_id, action_mask in action_anchor_masks.items():
        if required_true_mask & action_mask:
            action_ids.add(action_id)
    return action_ids


def _classify_observation_rule_action_ids(
    goal_check: BitwiseGoalCheck | None,
    action_anchor_masks: dict[int, int],
) -> set[int]:
    if goal_check is None or not goal_check.clauses or not action_anchor_masks:
        return set()

    clause_action_sets = [
        _clause_action_ids_from_mask(required_true_mask, action_anchor_masks)
        for required_true_mask, _required_false_mask in goal_check.clauses
    ]
    clause_action_sets = [action_ids for action_ids in clause_action_sets if action_ids]
    if not clause_action_sets:
        return set()

    allowed_action_ids = set(clause_action_sets[0])
    for clause_action_ids in clause_action_sets[1:]:
        allowed_action_ids &= clause_action_ids

    if allowed_action_ids:
        return allowed_action_ids

    union_action_ids = set().union(*clause_action_sets)
    if len(union_action_ids) <= 3:
        return union_action_ids
    return set()


def _gen_precond_fns_from_goal_checks(checks: list[BitwiseGoalCheck], nw: int) -> str:
    L = []
    for i, check in enumerate(checks):
        expr = _render_cpp_goal_check_expr(check, "state", nw)
        L.append(f"static bool _check_precond_{i}(const uint64_t* state) {{ return {expr}; }}")
    L.append("typedef bool (*PrecondFn)(const uint64_t*);")
    L.append(f"static PrecondFn PRECONDITION_TABLE[{len(checks)}] = {{")
    for i in range(len(checks)):
        L.append(f"    _check_precond_{i},")
    L.append("};")
    L.append("")
    return "\n".join(L)


def _expand_bitwise_model_codegen_actions(model: BitwiseModelSkeleton) -> dict[str, Any]:
    enable_report_goal_action = bool(getattr(model, "enable_report_goal_action", False))
    action_id_offset = int(getattr(model, "action_id_offset", 0)) if enable_report_goal_action else 0
    raw_goal_check = getattr(model, "raw_goal_check", getattr(model, "goal_check", None))

    action_precondition_checks = list(getattr(model, "action_precondition_checks", []))
    action_effect_condition_checks = [
        list(condition_checks)
        for condition_checks in getattr(model, "action_effect_condition_checks", [])
    ]
    action_effect_distributions = [
        list(branches)
        for branches in getattr(model, "action_effect_distributions", [])
    ]
    action_conditional_effect_distributions = [
        [list(branches) for branches in conditional_branches]
        for conditional_branches in getattr(model, "action_conditional_effect_distributions", [])
    ]
    default_policy_action_ids = list(getattr(model, "default_policy_rule_action_ids", []))

    if enable_report_goal_action:
        action_precondition_checks = [
            BitwiseGoalCheck(clauses=[(0, 0)]),
            *action_precondition_checks,
        ]
        action_effect_condition_checks = [
            [],
            *action_effect_condition_checks,
        ]
        action_effect_distributions = [
            [BitwiseEffectBranch(probability=1.0)],
            *action_effect_distributions,
        ]
        action_conditional_effect_distributions = [
            [],
            *action_conditional_effect_distributions,
        ]
        default_policy_action_ids = [
            0 if int(action_id) < 0 else int(action_id) + 1
            for action_id in default_policy_action_ids
        ]
        default_policy_action_ids = [
            action_id + action_id_offset
            for action_id in default_policy_action_ids
        ]

    return {
        "enable_report_goal_action": enable_report_goal_action,
        "raw_goal_check": raw_goal_check,
        "action_precondition_checks": action_precondition_checks,
        "action_effect_condition_checks": action_effect_condition_checks,
        "action_effect_distributions": action_effect_distributions,
        "action_conditional_effect_distributions": action_conditional_effect_distributions,
        "default_policy_action_ids": default_policy_action_ids,
    }


def _expand_observation_rule_codegen_metadata(
    model: Any,
    *,
    total_actions: int,
    observation_rule_count: int,
) -> tuple[list[bool], list[bool]]:
    rule_skip_flags = list(
        getattr(model, "observation_rule_skip_before_for_active_perception", [])
    )
    action_active_flags = list(getattr(model, "action_is_active_perception", []))
    enable_report_goal_action = bool(getattr(model, "enable_report_goal_action", False))

    if len(rule_skip_flags) < observation_rule_count:
        rule_skip_flags.extend([False] * (observation_rule_count - len(rule_skip_flags)))
    else:
        rule_skip_flags = rule_skip_flags[:observation_rule_count]

    if enable_report_goal_action and len(action_active_flags) + 1 == total_actions:
        action_active_flags = [False, *action_active_flags]
    if len(action_active_flags) < total_actions:
        action_active_flags.extend([False] * (total_actions - len(action_active_flags)))
    else:
        action_active_flags = action_active_flags[:total_actions]

    return rule_skip_flags, action_active_flags


def _gen_obs_data_from_bitwise_model(data: dict, model: BitwiseModelSkeleton, nw: int) -> str:
    L = []
    layout = data.get("index_layout") or {}
    predicate_names = list(layout.get("predicates", []))
    action_names = list(layout.get("actions", []))
    rule_skip_flags, action_active_flags = _expand_observation_rule_codegen_metadata(
        model,
        total_actions=data["total_actions"],
        observation_rule_count=len(model.observation_rule_condition_checks),
    )
    action_anchor_masks = _build_observation_action_anchor_masks(predicate_names, action_names)
    rule_must_true_masks: list[int] = []
    rule_must_false_masks: list[int] = []
    action_bucket_mapping: dict[int, dict[tuple[int, int, int, int, int, int], list[int]]] = {
        action_id: {} for action_id in range(data["total_actions"])
    }
    generic_bucket_mapping: dict[tuple[int, int, int, int, int, int], list[int]] = {}
    action_has_observation_model_flags = [False] * data["total_actions"]
    action_observation_rule_ids: list[set[int]] = [set() for _ in range(data["total_actions"])]
    generic_observation_rule_ids: set[int] = set()
    for i, check in enumerate(model.observation_rule_condition_checks):
        logic = _render_cpp_goal_check_expr(check, "state", nw)
        L.append(f"static bool _check_obs_{i}(const uint64_t* state) {{ return {logic}; }}")
    for i, check in enumerate(model.observation_rule_condition_checks):
        signature_parts = _extract_common_bucket_signature_parts(check)
        signature_tuple = [-1, 0, 0, -1, 0, 0]
        for part_index, (word_index, true_mask, false_mask) in enumerate(signature_parts[:2]):
            base = part_index * 3
            signature_tuple[base] = word_index
            signature_tuple[base + 1] = true_mask
            signature_tuple[base + 2] = false_mask
        must_true_mask, must_false_mask = _extract_common_clause_masks(check)
        rule_must_true_masks.append(must_true_mask)
        rule_must_false_masks.append(must_false_mask)
        allowed_action_ids = _classify_observation_rule_action_ids(check, action_anchor_masks)
        bucket_key = tuple(signature_tuple)
        if not allowed_action_ids:
            generic_bucket_mapping.setdefault(bucket_key, []).append(i)
            generic_observation_rule_ids.add(i)
        else:
            for action_id in sorted(allowed_action_ids):
                if 0 <= action_id < len(action_has_observation_model_flags):
                    action_has_observation_model_flags[action_id] = True
                    action_observation_rule_ids[action_id].add(i)
                action_bucket_mapping[action_id].setdefault(bucket_key, []).append(i)
    n = len(model.observation_rule_condition_checks)
    L.append("typedef bool (*ObsCFn)(const uint64_t*);")
    L.append(f"static ObsCFn OBS_COND_TABLE[{n}] = {{")
    for i in range(n):
        L.append(f"    _check_obs_{i},")
    L.append("};")
    for i in range(n):
        L.append(_words_to_c_decl(f"OBS_RULE_MUST_TRUE_{i}", _int_to_words(rule_must_true_masks[i], nw)))
        L.append(_words_to_c_decl(f"OBS_RULE_MUST_FALSE_{i}", _int_to_words(rule_must_false_masks[i], nw)))
    L.append(f"static const uint64_t* OBS_RULE_MUST_TRUE[{n}] = {{")
    for i in range(n):
        L.append(f"    OBS_RULE_MUST_TRUE_{i},")
    L.append("};")
    L.append(f"static const uint64_t* OBS_RULE_MUST_FALSE[{n}] = {{")
    for i in range(n):
        L.append(f"    OBS_RULE_MUST_FALSE_{i},")
    L.append("};")
    rule_skip_text = ", ".join("true" if flag else "false" for flag in rule_skip_flags)
    action_active_text = ", ".join("true" if flag else "false" for flag in action_active_flags)
    L.append(f"static const bool OBS_RULE_SKIP_BEFORE_ON_ACTIVE_PERCEPTION[{n}] = {{{rule_skip_text}}};")
    L.append(
        f"static const bool ACTION_IS_ACTIVE_PERCEPTION[{data['total_actions']}] = "
        f"{{{action_active_text}}};"
    )
    action_has_observation_model_text = ", ".join(
        "true" if flag else "false" for flag in action_has_observation_model_flags
    )
    L.append(
        f"static const bool ACTION_HAS_OBSERVATION_MODEL[{data['total_actions']}] = "
        f"{{{action_has_observation_model_text}}};"
    )
    ignored_mask = _no_change_ignored_mask(data)
    deterministic_observation_filter_flags: list[bool] = []
    for action_id in range(data["total_actions"]):
        if not action_has_observation_model_flags[action_id]:
            deterministic_observation_filter_flags.append(False)
            continue
        # Only action-anchored observation rules should decide whether an action is a
        # deterministic information-gathering action. Generic rules such as init/passive
        # observation modules can apply globally, but they should not disable repeated-
        # observation pruning for a specific action.
        relevant_rule_ids = action_observation_rule_ids[action_id]
        if not relevant_rule_ids:
            deterministic_observation_filter_flags.append(False)
            continue
        obs_is_deterministic = all(
            _observation_branches_deterministic(model.observation_rule_distributions[rule_id])
            for rule_id in relevant_rule_ids
        )
        effect_is_no_real_change = _is_deterministic_no_real_change_action(
            action_id,
            action_effect_distributions=model.action_effect_distributions,
            action_effect_condition_checks=model.action_effect_condition_checks,
            action_conditional_effect_distributions=model.action_conditional_effect_distributions,
            ignored_mask=ignored_mask,
        )
        deterministic_observation_filter_flags.append(
            obs_is_deterministic and effect_is_no_real_change
        )
    deterministic_observation_filter_text = ", ".join(
        "true" if flag else "false" for flag in deterministic_observation_filter_flags
    )
    L.append(
        f"static const bool ACTION_DETERMINISTIC_OBSERVATION_FILTER[{data['total_actions']}] = "
        f"{{{deterministic_observation_filter_text}}};"
    )
    L.append("struct ObsBucketDesc { int word0; uint64_t true_mask0; uint64_t false_mask0; int word1; uint64_t true_mask1; uint64_t false_mask1; const int* rules; int count; };")
    for action_id in range(data["total_actions"]):
        buckets = action_bucket_mapping.get(action_id, {})
        for bucket_index, ((word0, true0, false0, word1, true1, false1), rule_indices) in enumerate(sorted(buckets.items())):
            rule_text = ", ".join(str(rule_index) for rule_index in rule_indices)
            L.append(
                f"static const int OBS_ACTION_BUCKET_RULES_{action_id}_{bucket_index}[{len(rule_indices)}] = "
                f"{{{rule_text}}};"
            )
        if buckets:
            L.append(
                f"static const ObsBucketDesc OBS_ACTION_BUCKETS_{action_id}[{len(buckets)}] = {{"
            )
            for bucket_index, ((word0, true0, false0, word1, true1, false1), rule_indices) in enumerate(sorted(buckets.items())):
                comma = "," if bucket_index < len(buckets) - 1 else ""
                L.append(
                    f"    {{{word0}, 0x{true0:016X}ULL, 0x{false0:016X}ULL, {word1}, 0x{true1:016X}ULL, 0x{false1:016X}ULL, OBS_ACTION_BUCKET_RULES_{action_id}_{bucket_index}, {len(rule_indices)}}}{comma}"
                )
            L.append("};")
    L.append(f"static const int OBS_ACTION_BUCKET_COUNT[{data['total_actions']}] = {{")
    for action_id in range(data["total_actions"]):
        bucket_count = len(action_bucket_mapping.get(action_id, {}))
        comma = "," if action_id < data["total_actions"] - 1 else ""
        L.append(f"    {bucket_count}{comma}")
    L.append("};")
    L.append(f"static const ObsBucketDesc* OBS_ACTION_BUCKET_TABLE[{data['total_actions']}] = {{")
    for action_id in range(data["total_actions"]):
        entry = (
            f"OBS_ACTION_BUCKETS_{action_id}"
            if action_bucket_mapping.get(action_id)
            else "nullptr"
        )
        comma = "," if action_id < data["total_actions"] - 1 else ""
        L.append(f"    {entry}{comma}")
    L.append("};")
    generic_buckets = sorted(generic_bucket_mapping.items())
    for bucket_index, ((word0, true0, false0, word1, true1, false1), rule_indices) in enumerate(generic_buckets):
        rule_text = ", ".join(str(rule_index) for rule_index in rule_indices)
        L.append(
            f"static const int OBS_GENERIC_BUCKET_RULES_{bucket_index}[{len(rule_indices)}] = "
            f"{{{rule_text}}};"
        )
    if generic_buckets:
        L.append(f"static const ObsBucketDesc OBS_GENERIC_BUCKETS[{len(generic_buckets)}] = {{")
        for bucket_index, ((word0, true0, false0, word1, true1, false1), rule_indices) in enumerate(generic_buckets):
            comma = "," if bucket_index < len(generic_buckets) - 1 else ""
            L.append(
                f"    {{{word0}, 0x{true0:016X}ULL, 0x{false0:016X}ULL, {word1}, 0x{true1:016X}ULL, 0x{false1:016X}ULL, OBS_GENERIC_BUCKET_RULES_{bucket_index}, {len(rule_indices)}}}{comma}"
            )
        L.append("};")
    else:
        L.append("static const ObsBucketDesc* OBS_GENERIC_BUCKETS = nullptr;")
    L.append(f"static const int OBS_GENERIC_BUCKET_COUNT = {len(generic_buckets)};")
    L.append("")
    return "\n".join(L)


def _gen_dp_data_from_bitwise_model(
    model: BitwiseModelSkeleton,
    nw: int,
    *,
    default_policy_action_ids: list[int] | None = None,
    total_actions: int | None = None,
) -> str:
    L = []
    action_bucket_mapping: dict[int, dict[tuple[int, int], list[int]]] = {}
    rule_signature_words: list[int] = []
    rule_signature_masks: list[int] = []
    rule_must_true_masks: list[int] = []
    rule_must_false_masks: list[int] = []
    action_ids = (
        default_policy_action_ids
        if default_policy_action_ids is not None
        else model.default_policy_rule_action_ids
    )
    for i, check in enumerate(model.default_policy_rule_condition_checks):
        expr = _render_cpp_goal_check_expr(check, "state", nw)
        L.append(f"static bool _check_dp_{i}(const uint64_t* state) {{ return {expr}; }}")
        signature = _extract_common_true_anchor_signature(check)
        if signature is None:
            rule_signature_words.append(-1)
            rule_signature_masks.append(0)
        else:
            word_index, word_mask = signature
            rule_signature_words.append(word_index)
            rule_signature_masks.append(word_mask)
        must_true_mask, must_false_mask = _extract_common_clause_masks(check)
        rule_must_true_masks.append(must_true_mask)
        rule_must_false_masks.append(must_false_mask)
        action_id = action_ids[i]
        action_bucket_mapping.setdefault(action_id, {}).setdefault(
            (rule_signature_words[-1], rule_signature_masks[-1]),
            [],
        ).append(i)
    aids = action_ids
    ids_str = ", ".join(str(a) for a in aids)
    L.append(f"static const int DP_AIDS[{len(aids)}] = {{{ids_str}}};")
    L.append("typedef bool (*DPCFn)(const uint64_t*);")
    n = len(model.default_policy_rule_condition_checks)
    L.append(f"static DPCFn DP_COND_TABLE[{n}] = {{")
    for i in range(n):
        L.append(f"    _check_dp_{i},")
    L.append("};")
    signature_words_text = ", ".join(str(word_index) for word_index in rule_signature_words)
    L.append(f"static const int DP_RULE_SIGNATURE_WORD[{n}] = {{{signature_words_text}}};")
    signature_masks_text = ", ".join(f"0x{word_mask:016X}ULL" for word_mask in rule_signature_masks)
    L.append(f"static const uint64_t DP_RULE_SIGNATURE_MASK[{n}] = {{{signature_masks_text}}};")
    for i in range(n):
        L.append(_words_to_c_decl(f"DP_RULE_MUST_TRUE_{i}", _int_to_words(rule_must_true_masks[i], nw)))
        L.append(_words_to_c_decl(f"DP_RULE_MUST_FALSE_{i}", _int_to_words(rule_must_false_masks[i], nw)))
    L.append(f"static const uint64_t* DP_RULE_MUST_TRUE[{n}] = {{")
    for i in range(n):
        L.append(f"    DP_RULE_MUST_TRUE_{i},")
    L.append("};")
    L.append(f"static const uint64_t* DP_RULE_MUST_FALSE[{n}] = {{")
    for i in range(n):
        L.append(f"    DP_RULE_MUST_FALSE_{i},")
    L.append("};")
    L.append("struct DpBucketDesc { int word; uint64_t mask; const int* rules; int count; };")
    total_actions = model.total_actions if total_actions is None else total_actions
    for action_id in range(total_actions):
        buckets = action_bucket_mapping.get(action_id, {})
        for bucket_index, ((word_index, word_mask), rule_indices) in enumerate(sorted(buckets.items())):
            rule_text = ", ".join(str(rule_index) for rule_index in rule_indices)
            L.append(
                f"static const int DP_BUCKET_RULES_{action_id}_{bucket_index}[{len(rule_indices)}] = "
                f"{{{rule_text}}};"
            )
        if buckets:
            L.append(f"static const DpBucketDesc DP_BUCKETS_{action_id}[{len(buckets)}] = {{")
            for bucket_index, ((word_index, word_mask), rule_indices) in enumerate(sorted(buckets.items())):
                comma = "," if bucket_index < len(buckets) - 1 else ""
                L.append(
                    f"    {{{word_index}, 0x{word_mask:016X}ULL, DP_BUCKET_RULES_{action_id}_{bucket_index}, {len(rule_indices)}}}{comma}"
                )
            L.append("};")
    L.append(f"static const int DP_BUCKET_COUNT[{total_actions if total_actions > 0 else 1}] = {{")
    if total_actions == 0:
        L.append("    0")
    else:
        for action_id in range(total_actions):
            comma = "," if action_id < total_actions - 1 else ""
            L.append(f"    {len(action_bucket_mapping.get(action_id, {}))}{comma}")
    L.append("};")
    L.append(f"static const DpBucketDesc* DP_BUCKET_TABLE[{total_actions if total_actions > 0 else 1}] = {{")
    if total_actions == 0:
        L.append("    nullptr")
    else:
        for action_id in range(total_actions):
            entry = f"DP_BUCKETS_{action_id}" if action_bucket_mapping.get(action_id) else "nullptr"
            comma = "," if action_id < total_actions - 1 else ""
            L.append(f"    {entry}{comma}")
    L.append("};")
    L.append("")
    return "\n".join(L)


def _emit_cpp_apply_effect_lines(
    lines: list[str],
    branch: BitwiseEffectBranch,
    *,
    indent: str,
    state_var: str,
    reward_var: str,
    nw: int,
) -> None:
    set_words = _int_to_words(branch.set_mask, nw)
    clear_words = _int_to_words(branch.clear_mask, nw)
    wrote = False
    for idx, (set_word, clear_word) in enumerate(zip(set_words, clear_words)):
        if set_word == 0 and clear_word == 0:
            continue
        if set_word != 0 and clear_word != 0:
            lines.append(
                f"{indent}{state_var}[{idx}] = ({state_var}[{idx}] | 0x{set_word:016X}ULL) & ~0x{clear_word:016X}ULL;"
            )
        elif set_word != 0:
            lines.append(f"{indent}{state_var}[{idx}] |= 0x{set_word:016X}ULL;")
        else:
            lines.append(f"{indent}{state_var}[{idx}] &= ~0x{clear_word:016X}ULL;")
        wrote = True
    if branch.reward_delta != 0.0:
        lines.append(f"{indent}{reward_var} += {branch.reward_delta};")
        wrote = True
    if not wrote:
        lines.append(f"{indent}/* no effect */")


def _extract_single_true_guard_bit(
    guard_check: BitwiseGoalCheck,
) -> tuple[int, int, int] | None:
    if guard_check is None or len(guard_check.clauses) != 1:
        return None
    required_true_mask, required_false_mask = guard_check.clauses[0]
    if required_false_mask != 0 or required_true_mask == 0:
        return None
    if required_true_mask & (required_true_mask - 1):
        return None
    global_bit_index = required_true_mask.bit_length() - 1
    word_index = global_bit_index // 64
    word_mask = 1 << (global_bit_index % 64)
    return word_index, word_mask, global_bit_index


def _extract_single_effect_bit(mask: int) -> tuple[int, int, int] | None:
    if mask == 0 or (mask & (mask - 1)) != 0:
        return None
    global_bit_index = mask.bit_length() - 1
    word_index = global_bit_index // 64
    word_mask = 1 << (global_bit_index % 64)
    return word_index, word_mask, global_bit_index


def _analyze_conditional_effect_optimizations(
    conditional_checks: list[BitwiseGoalCheck],
    conditional_branches: list[list[BitwiseEffectBranch]],
) -> tuple[
    dict[tuple[int, int, int, float], tuple[int, set[int]]],
    dict[tuple[int, float], int],
    dict[tuple[int, int, str, int], tuple[int, int]],
    set[int],
]:
    identical_effect_groups: dict[tuple[int, int, int, float], tuple[int, set[int]]] = {}
    reward_groups: dict[tuple[int, float], int] = {}
    # NOTE:
    # We intentionally keep move_groups disabled for now.
    #
    # A previous optimization tried to batch many single-bit guarded moves into
    # one shift expression:
    #
    #   bits[word] = (bits[word] & ~source_mask & ~target_mask) | shifted_bits
    #
    # That is only semantics-preserving if clearing target_mask cannot erase any
    # location bit that should remain true under a different guard. RockSample
    # move-west on the western boundary is a counterexample: west-edge guards are
    # reward-only/no-op, while target_mask still contains those west-edge rover
    # bits as destinations of non-boundary moves. Clearing target_mask therefore
    # deletes the rover location for boundary states and creates dead-end states
    # with no legal actions.
    #
    # Until we re-introduce this optimization with stronger safety checks, keep
    # move_groups empty and let the per-guard fallback preserve semantics.
    move_groups: dict[tuple[int, int, str, int], tuple[int, int]] = {}
    optimized_indices: set[int] = set()

    for guard_index, guard_check in enumerate(conditional_checks):
        if guard_index >= len(conditional_branches):
            continue
        branches = conditional_branches[guard_index]
        if len(branches) != 1:
            continue
        guard_info = _extract_single_true_guard_bit(guard_check)
        if guard_info is None:
            continue
        source_word_index, source_word_mask, source_global_bit = guard_info
        branch = branches[0]

        if branch.set_mask == 0 and branch.clear_mask == 0 and branch.reward_delta != 0.0:
            key = (source_word_index, branch.reward_delta)
            reward_groups[key] = reward_groups.get(key, 0) | source_word_mask
            continue

        # Grouping identical conditional effects is safe for idempotent effects
        # such as "set these bits" or "clear these bits" or "set bits + add
        # reward". It is not safe for move-like effects that both set and clear
        # bits. When multiple guarded single-bit moves are merged into one large
        # branch, the combined clear-mask can erase location bits that should stay
        # true for reward-only/no-op boundary guards (RockSample move-west is the
        # concrete counterexample).
        #
        # Keep the per-guard fallback for branches that have both set and clear
        # masks. That preserves the original semantics while still allowing the
        # safe grouped optimizations above.
        if branch.set_mask == 0 or branch.clear_mask == 0:
            identical_key = (
                source_word_index,
                branch.set_mask,
                branch.clear_mask,
                branch.reward_delta,
            )
            existing_group = identical_effect_groups.get(identical_key)
            if existing_group is None:
                identical_effect_groups[identical_key] = (source_word_mask, {guard_index})
            else:
                identical_effect_groups[identical_key] = (
                    existing_group[0] | source_word_mask,
                    existing_group[1] | {guard_index},
                )

        if branch.reward_delta != 0.0:
            continue

        # move_groups intentionally disabled; see note above.

    for (_word_index, _set_mask, _clear_mask, _reward_delta), (source_mask, indices) in identical_effect_groups.items():
        if source_mask != 0 and (source_mask & (source_mask - 1)) != 0:
            optimized_indices.update(indices)
    for (_word_index, _reward_delta), source_mask in reward_groups.items():
        if source_mask != 0 and (source_mask & (source_mask - 1)) != 0:
            for index, branch_list in enumerate(conditional_branches):
                if len(branch_list) != 1:
                    continue
                branch = branch_list[0]
                if branch.set_mask != 0 or branch.clear_mask != 0 or branch.reward_delta != _reward_delta:
                    continue
                guard_info = _extract_single_true_guard_bit(conditional_checks[index])
                if guard_info is None:
                    continue
                guard_word_index, guard_word_mask, _guard_global_bit = guard_info
                if guard_word_index == _word_index and (guard_word_mask & source_mask) != 0:
                    optimized_indices.add(index)

    return identical_effect_groups, reward_groups, move_groups, optimized_indices


def _emit_cpp_optimized_conditional_effects(
    lines: list[str],
    conditional_checks: list[BitwiseGoalCheck],
    conditional_branches: list[list[BitwiseEffectBranch]],
    nw: int,
) -> set[int]:
    identical_effect_groups, reward_groups, move_groups, optimized_indices = _analyze_conditional_effect_optimizations(
        conditional_checks,
        conditional_branches,
    )

    for (word_index, set_mask, clear_mask, reward_delta), (source_mask, indices) in identical_effect_groups.items():
        if source_mask == 0 or len(indices) <= 1:
            continue
        var_name = f"hits_{word_index}_{source_mask:016X}"
        lines.append(f"        const uint64_t {var_name} = orig[{word_index}] & 0x{source_mask:016X}ULL;")
        lines.append(f"        if ({var_name}) {{")
        _emit_cpp_apply_effect_lines(
            lines,
            BitwiseEffectBranch(
                probability=1.0,
                set_mask=set_mask,
                clear_mask=clear_mask,
                reward_delta=0.0,
            ),
            indent="            ",
            state_var="bits",
            reward_var="reward",
            nw=nw,
        )
        if reward_delta != 0.0:
            lines.append(
                "            "
                f"reward += {reward_delta} * static_cast<double>(__builtin_popcountll({var_name}));"
            )
        lines.append("        }")

    for (word_index, reward_delta), source_mask in reward_groups.items():
        lines.append(
            "        "
            f"reward += {reward_delta} * static_cast<double>(__builtin_popcountll(orig[{word_index}] & 0x{source_mask:016X}ULL));"
        )

    for (source_word_index, target_word_index, direction, shift), (source_mask, target_mask) in move_groups.items():
        if source_word_index == target_word_index:
            shift_expr = (
                f"((orig[{source_word_index}] & 0x{source_mask:016X}ULL) >> {shift})"
                if direction == "right"
                else f"((orig[{source_word_index}] & 0x{source_mask:016X}ULL) << {shift})"
            )
            lines.append(
                "        "
                f"bits[{source_word_index}] = (bits[{source_word_index}] & ~0x{source_mask:016X}ULL & ~0x{target_mask:016X}ULL) | ({shift_expr} & 0x{target_mask:016X}ULL);"
            )
        else:
            shift_expr = (
                f"((orig[{source_word_index}] & 0x{source_mask:016X}ULL) >> {shift})"
                if direction == "right"
                else f"((orig[{source_word_index}] & 0x{source_mask:016X}ULL) << {shift})"
            )
            lines.append(f"        bits[{source_word_index}] &= ~0x{source_mask:016X}ULL;")
            lines.append(
                "        "
                f"bits[{target_word_index}] = (bits[{target_word_index}] & ~0x{target_mask:016X}ULL) | ({shift_expr} & 0x{target_mask:016X}ULL);"
            )

    return optimized_indices


def _gen_fwd_fn_from_bitwise_model(
    *,
    action_effect_distributions: list[list[BitwiseEffectBranch]],
    action_effect_condition_checks: list[list[BitwiseGoalCheck]],
    action_conditional_effect_distributions: list[list[list[BitwiseEffectBranch]]],
    raw_goal_check: BitwiseGoalCheck | None,
    enable_report_goal_action: bool,
    cn: str,
    nw: int,
) -> str:
    L = []
    # Emit file-scope mutable probability arrays for hot-reload support.
    total_actions = len(action_effect_distributions)
    for _i in range(total_actions):
        _ub = action_effect_distributions[_i]
        if len(_ub) > 1:
            _probs = ", ".join(str(b.probability) for b in _ub)
            L.append(f"static double ACT_{_i}_W[] = {{{_probs}}};")  # mutable
        _cb = action_conditional_effect_distributions[_i]
        for _gi, _branches in enumerate(_cb):
            if len(_branches) > 1:
                _probs = ", ".join(str(b.probability) for b in _branches)
                L.append(f"static double ACT_{_i}_G_{_gi}_W[] = {{{_probs}}};")  # mutable
    L.append("")
    L.append(f"void {cn}::ForwardAction(int action, uint64_t* bits, uint64_t& rng_seed, double& reward) const {{")
    L.append("    reward = 0.0;")
    L.append("    switch (action) {")
    for i in range(total_actions):
        unconditional_branches = action_effect_distributions[i]
        conditional_checks = action_effect_condition_checks[i]
        conditional_branches = action_conditional_effect_distributions[i]
        has_saved_state = bool(conditional_checks)
        L.append(f"    case {i}: {{")
        if enable_report_goal_action and i == 0:
            goal_expr = _render_cpp_goal_check_expr(raw_goal_check, "bits", nw)
            L.append(f"        reward = {goal_expr} ? GOAL_REWARD : REPORT_GOAL_FAILURE_PENALTY;")
            L.append("        break; }")
            continue
        if has_saved_state:
            L.append("        uint64_t orig[N_WORDS]; bw_copy<N_WORDS>(orig, bits);")
        if len(unconditional_branches) == 1:
            _emit_cpp_apply_effect_lines(
                L,
                unconditional_branches[0],
                indent="        ",
                state_var="bits",
                reward_var="reward",
                nw=nw,
            )
        elif len(unconditional_branches) > 1:
            L.append(f"        int bi = bw_sample_branch(Random::NextGoldenRandom(rng_seed), ACT_{i}_W, {len(unconditional_branches)});")
            for j, branch in enumerate(unconditional_branches):
                prefix = "if" if j == 0 else "else if"
                L.append(f"        {prefix} (bi == {j}) {{")
                _emit_cpp_apply_effect_lines(
                    L,
                    branch,
                    indent="            ",
                    state_var="bits",
                    reward_var="reward",
                    nw=nw,
                )
                L.append("        }")
        optimized_guard_indices: set[int] = set()
        if has_saved_state:
            optimized_guard_indices = _emit_cpp_optimized_conditional_effects(
                L,
                conditional_checks,
                conditional_branches,
                nw,
            )
        for guard_index, guard_check in enumerate(conditional_checks):
            if guard_index in optimized_guard_indices:
                continue
            guard_expr = _render_cpp_goal_check_expr(guard_check, "orig" if has_saved_state else "bits", nw)
            branches = conditional_branches[guard_index]
            L.append(f"        if ({guard_expr}) {{")
            if len(branches) == 1:
                _emit_cpp_apply_effect_lines(
                    L,
                    branches[0],
                    indent="            ",
                    state_var="bits",
                    reward_var="reward",
                    nw=nw,
                )
            elif len(branches) > 1:
                L.append(
                    f"            int gbi = bw_sample_branch(Random::NextGoldenRandom(rng_seed), "
                    f"ACT_{i}_G_{guard_index}_W, {len(branches)});"
                )
                for j, branch in enumerate(branches):
                    prefix = "if" if j == 0 else "else if"
                    L.append(f"            {prefix} (gbi == {j}) {{")
                    _emit_cpp_apply_effect_lines(
                        L,
                        branch,
                        indent="                ",
                        state_var="bits",
                        reward_var="reward",
                        nw=nw,
                    )
                    L.append("            }")
            else:
                L.append("            /* no conditional branches */")
            L.append("        }")
        if (
            not enable_report_goal_action
            and raw_goal_check is not None
            and raw_goal_check.clauses
        ):
            goal_expr = _render_cpp_goal_check_expr(raw_goal_check, "bits", nw)
            L.append(f"        if ({goal_expr}) reward += GOAL_REWARD;")
        L.append("        break; }")
    L.append("    default: break; }")
    L.append("}")
    L.append("")
    return "\n".join(L)


def _gen_obs_fn_from_bitwise_model(model: BitwiseModelSkeleton, cn: str) -> str:
    L = []
    # Emit file-scope mutable observation probability arrays for hot-reload.
    for _i, _branches in enumerate(model.observation_rule_distributions):
        if len(_branches) > 1:
            _probs = ", ".join(str(b.probability) for b in _branches)
            L.append(f"static double OBS_{_i}_W[] = {{{_probs}}};")  # mutable
    L.append("")
    L.append(f"OBS_TYPE {cn}::Observe(ACT_TYPE action, const uint64_t* bits, uint64_t& rng_seed) const {{")
    L.append(f"    bool profile_enabled = {cn}ProfileEnabled();")
    L.append("    auto observe_start = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};")
    L.append("    OBS_TYPE final_obs = 0;")
    L.append("    uint64_t candidate_rules = 0;")
    L.append("    uint64_t cond_checks = 0;")
    L.append("    uint64_t fired_rules = 0;")
    L.append("    auto eval_rule = [&](int r) {")
    L.append("        if (action >= 0 && action < TOTAL_ACTIONS && ACTION_IS_ACTIVE_PERCEPTION[action] && OBS_RULE_SKIP_BEFORE_ON_ACTIVE_PERCEPTION[r]) return;")
    L.append("        if (!bw_check_clause<N_WORDS>(bits, OBS_RULE_MUST_TRUE[r], OBS_RULE_MUST_FALSE[r])) return;")
    L.append("        cond_checks++;")
    L.append("        if (!OBS_COND_TABLE[r](bits)) return;")
    L.append("        fired_rules++;")
    L.append("        OBS_TYPE ob = 0, om = 0;")
    L.append("        switch (r) {")
    for i, branches in enumerate(model.observation_rule_distributions):
        L.append(f"        case {i}: {{")
        if len(branches) == 1:
            branch = branches[0]
            L.append(f"            ob = {branch.observation_bits};")
            L.append(f"            om = {branch.observation_mask};")
        elif len(branches) > 1:
            L.append(f"            int bi = bw_sample_branch(Random::NextGoldenRandom(rng_seed), OBS_{i}_W, {len(branches)});")
            for j, branch in enumerate(branches):
                prefix = "if" if j == 0 else "else if"
                L.append(f"            {prefix} (bi == {j}) {{ ob = {branch.observation_bits}; om = {branch.observation_mask}; }}")
        L.append("            break; }")
    L.append("        default: break; }")
    L.append("        final_obs |= (ob & om);")
    L.append("    };")
    L.append("    auto eval_bucket = [&](const ObsBucketDesc& bucket) {")
    L.append("        if (bucket.word0 >= 0) {")
    L.append("            uint64_t word_bits0 = bits[bucket.word0];")
    L.append("            if ((word_bits0 & bucket.true_mask0) != bucket.true_mask0) return;")
    L.append("            if ((word_bits0 & bucket.false_mask0) != 0) return;")
    L.append("        }")
    L.append("        if (bucket.word1 >= 0) {")
    L.append("            uint64_t word_bits1 = bits[bucket.word1];")
    L.append("            if ((word_bits1 & bucket.true_mask1) != bucket.true_mask1) return;")
    L.append("            if ((word_bits1 & bucket.false_mask1) != 0) return;")
    L.append("        }")
    L.append("        candidate_rules += bucket.count;")
    L.append("        for (int i = 0; i < bucket.count; ++i) eval_rule(bucket.rules[i]);")
    L.append("    };")
    L.append("    if (action >= 0 && action < TOTAL_ACTIONS) {")
    L.append("        const ObsBucketDesc* action_buckets = OBS_ACTION_BUCKET_TABLE[action];")
    L.append("        int action_bucket_count = OBS_ACTION_BUCKET_COUNT[action];")
    L.append("        for (int i = 0; i < action_bucket_count; ++i) eval_bucket(action_buckets[i]);")
    L.append("    }")
    L.append("    for (int i = 0; i < OBS_GENERIC_BUCKET_COUNT; ++i) eval_bucket(OBS_GENERIC_BUCKETS[i]);")
    L.append("    if (profile_enabled) {")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.observe_calls++;")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.observe_candidate_rules += candidate_rules;")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.observe_cond_checks += cond_checks;")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.observe_fired_rules += fired_rules;")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.observe_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - observe_start).count();")
    L.append("    }")
    L.append("    final_obs = AppendFullyObservedStateToObs(bits, final_obs);")
    L.append("    return final_obs;")
    L.append("}")
    L.append("")
    return "\n".join(L)


def _gen_goal_data(model: Any, nw: int) -> str:
    source = textwrap.dedent(inspect.getsource(model.is_goal))
    tree = ast.parse(source)
    fd = tree.body[0]
    ret_stmt = None
    for stmt in fd.body:
        if isinstance(stmt, ast.Return):
            ret_stmt = stmt
            break
    clauses = _extract_dnf_clauses(ret_stmt.value, nw) if ret_stmt else []
    L = [f"static const int GOAL_N_CLAUSES = {len(clauses)};"]
    for i, (tm, fm) in enumerate(clauses):
        L.append(_words_to_c_decl(f"GOAL_TRUE_{i}", _int_to_words(tm, nw)))
        L.append(_words_to_c_decl(f"GOAL_FALSE_{i}", _int_to_words(fm, nw)))
    L.append(f"static const uint64_t* GOAL_TRUE_MASKS[{max(len(clauses),1)}] = {{")
    for i in range(len(clauses)):
        L.append(f"    GOAL_TRUE_{i},")
    L.append("};")
    L.append(f"static const uint64_t* GOAL_FALSE_MASKS[{max(len(clauses),1)}] = {{")
    for i in range(len(clauses)):
        L.append(f"    GOAL_FALSE_{i},")
    L.append("};")
    L.append("")
    return "\n".join(L)


def _gen_precond_fns(data: dict, nw: int) -> str:
    funcs = data["check_preconditions"]
    L = []
    for i, func in enumerate(funcs):
        logic = _extract_precondition_logic(func, nw)
        L.append(f"static bool _check_precond_{i}(const uint64_t* state) {{ {logic} }}")
    L.append(f"typedef bool (*PrecondFn)(const uint64_t*);")
    L.append(f"static PrecondFn PRECONDITION_TABLE[{len(funcs)}] = {{")
    for i in range(len(funcs)):
        L.append(f"    _check_precond_{i},")
    L.append("};")
    L.append("")
    return "\n".join(L)


def _gen_effect_data(data: dict, nw: int) -> str:
    """Generate static data for action effects (branch masks only, no reward condition masks)."""
    funcs = data["forward_actions"]
    L = []
    for i, func in enumerate(funcs):
        info = _extract_forward_action_logic(func, i, nw)
        weights = info.get("prob_weights", [])
        branches = info.get("prob_branches", [])
        det = info.get("deterministic_effect")
        if weights and branches:
            w_str = ", ".join(str(w) for w in weights)
            L.append(f"static const double ACT_{i}_W[] = {{{w_str}}};")
            L.append(f"static const int ACT_{i}_NB = {len(weights)};")
            for j, (sm, cm) in enumerate(branches):
                L.append(_words_to_c_decl(f"ACT_{i}_S_{j}", _int_to_words(sm, nw)))
                L.append(_words_to_c_decl(f"ACT_{i}_C_{j}", _int_to_words(cm, nw)))
        elif det is not None:
            sm, cm = det
            if sm != 0 or cm != 0:
                L.append(_words_to_c_decl(f"ACT_{i}_S_0", _int_to_words(sm, nw)))
                L.append(_words_to_c_decl(f"ACT_{i}_C_0", _int_to_words(cm, nw)))
    L.append("")
    return "\n".join(L)


def _gen_obs_data(data: dict, nw: int) -> str:
    cfs = data["check_obs_conditions"]
    ofs = data["observe_with_rules"]
    model_instance = data.get("model_instance")
    rule_skip_flags, action_active_flags = _expand_observation_rule_codegen_metadata(
        model_instance,
        total_actions=data["total_actions"],
        observation_rule_count=len(cfs),
    )
    layout = data.get("index_layout") or {}
    predicate_names = list(layout.get("predicates", []))
    action_names = list(layout.get("actions", []))
    action_anchor_masks = _build_observation_action_anchor_masks(predicate_names, action_names)
    action_has_observation_model_flags = [False] * data["total_actions"]
    L = []
    for i, func in enumerate(cfs):
        logic = _extract_precondition_logic(func, nw)
        L.append(f"static bool _check_obs_{i}(const uint64_t* state) {{ {logic} }}")
        goal_check = _extract_goal_check_from_callable(func, nw)
        allowed_action_ids = _classify_observation_rule_action_ids(goal_check, action_anchor_masks)
        for action_id in sorted(allowed_action_ids):
            if 0 <= action_id < len(action_has_observation_model_flags):
                action_has_observation_model_flags[action_id] = True
    for i, func in enumerate(ofs):
        info = _extract_observe_logic(func, nw)
        ws = info.get("prob_weights", [])
        bs = info.get("branches", [])
        if ws:
            w_str = ", ".join(str(w) for w in ws)
            L.append(f"static const double OBS_{i}_W[] = {{{w_str}}};")
            L.append(f"static const int OBS_{i}_NB = {len(ws)};")
        else:
            L.append(f"static const double* OBS_{i}_W = nullptr;")
            L.append(f"static const int OBS_{i}_NB = 0;")
        for j, (ob, om) in enumerate(bs):
            L.append(f"static const OBS_TYPE OBS_{i}_BITS_{j} = {ob};")
            L.append(f"static const OBS_TYPE OBS_{i}_MASK_{j} = {om};")
    L.append(f"typedef bool (*ObsCFn)(const uint64_t*);")
    n = len(cfs)
    L.append(f"static ObsCFn OBS_COND_TABLE[{n}] = {{")
    for i in range(n):
        L.append(f"    _check_obs_{i},")
    L.append("};")
    rule_skip_text = ", ".join("true" if flag else "false" for flag in rule_skip_flags)
    action_active_text = ", ".join("true" if flag else "false" for flag in action_active_flags)
    L.append(f"static const bool OBS_RULE_SKIP_BEFORE_ON_ACTIVE_PERCEPTION[{n}] = {{{rule_skip_text}}};")
    L.append(
        f"static const bool ACTION_IS_ACTIVE_PERCEPTION[{data['total_actions']}] = "
        f"{{{action_active_text}}};"
    )
    action_has_observation_model_text = ", ".join(
        "true" if flag else "false" for flag in action_has_observation_model_flags
    )
    L.append(
        f"static const bool ACTION_HAS_OBSERVATION_MODEL[{data['total_actions']}] = "
        f"{{{action_has_observation_model_text}}};"
    )
    L.append(
        f"static const bool ACTION_DETERMINISTIC_OBSERVATION_FILTER[{data['total_actions']}] = "
        f"{{{', '.join('false' for _ in range(data['total_actions']))}}};"
    )
    L.append("")
    return "\n".join(L)


def _gen_dp_data(data: dict, nw: int) -> str:
    cfs = data["check_dp_conditions"]
    afs = data["get_dp_actions"]
    L = []
    for i, func in enumerate(cfs):
        logic = _extract_precondition_logic(func, nw)
        L.append(f"static bool _check_dp_{i}(const uint64_t* state) {{ {logic} }}")
    aids = [_extract_dp_action_id(f) for f in afs]
    ids_str = ", ".join(str(a) for a in aids)
    L.append(f"static const int DP_AIDS[{len(aids)}] = {{{ids_str}}};")
    L.append(f"typedef bool (*DPCFn)(const uint64_t*);")
    n = len(cfs)
    L.append(f"static DPCFn DP_COND_TABLE[{n}] = {{")
    for i in range(n):
        L.append(f"    _check_dp_{i},")
    L.append("};")
    L.append("")
    return "\n".join(L)


def _gen_belief_data(data: dict, nw: int) -> str:
    L = []
    L.append('static std::string INIT_BELIEF_JSON_PATH = "init_belief.json";')
    L.append("struct JsonBeliefParticle {")
    L.append("    double probability = 0.0;")
    L.append("    std::vector<int> true_indices;")
    L.append("};")
    L.append("struct JsonBeliefData {")
    L.append("    int grounded_predicates_count = 0;")
    L.append("    uint64_t last_observation_bits = 0;")
    L.append("    int has_last_observation = 0;")
    L.append("    std::vector<JsonBeliefParticle> particles;")
    L.append("};")
    L.append("")
    L.append("static std::string _read_text_file(const std::string& path) {")
    L.append("    std::ifstream in(path);")
    L.append("    if (!in) throw std::runtime_error(\"Failed to open init belief json: \" + path);")
    L.append("    std::ostringstream buffer;")
    L.append("    buffer << in.rdbuf();")
    L.append("    return buffer.str();")
    L.append("}")
    L.append("")
    L.append("static size_t _skip_ws(const std::string& text, size_t pos) {")
    L.append("    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;")
    L.append("    return pos;")
    L.append("}")
    L.append("")
    L.append("static size_t _find_matching(const std::string& text, size_t open_pos, char open_c, char close_c) {")
    L.append("    int depth = 0;")
    L.append("    bool in_string = false;")
    L.append("    bool escaped = false;")
    L.append("    for (size_t i = open_pos; i < text.size(); ++i) {")
    L.append("        char ch = text[i];")
    L.append("        if (in_string) {")
    L.append("            if (escaped) { escaped = false; continue; }")
    L.append("            if (ch == '\\\\') { escaped = true; continue; }")
    L.append("            if (ch == '\"') in_string = false;")
    L.append("            continue;")
    L.append("        }")
    L.append("        if (ch == '\"') { in_string = true; continue; }")
    L.append("        if (ch == open_c) ++depth;")
    L.append("        else if (ch == close_c) {")
    L.append("            --depth;")
    L.append("            if (depth == 0) return i;")
    L.append("        }")
    L.append("    }")
    L.append("    throw std::runtime_error(\"Unmatched JSON bracket while parsing init belief.\");")
    L.append("}")
    L.append("")
    L.append("static size_t _find_key(const std::string& text, const std::string& key) {")
    L.append("    std::string needle = std::string(\"\\\"\") + key + \"\\\"\";")
    L.append("    size_t pos = text.find(needle);")
    L.append("    if (pos == std::string::npos) throw std::runtime_error(\"Missing key in init belief json: \" + key);")
    L.append("    return pos + needle.size();")
    L.append("}")
    L.append("")
    L.append("static size_t _find_key_optional(const std::string& text, const std::string& key) {")
    L.append("    std::string needle = std::string(\"\\\"\") + key + \"\\\"\";")
    L.append("    size_t pos = text.find(needle);")
    L.append("    if (pos == std::string::npos) return std::string::npos;")
    L.append("    return pos + needle.size();")
    L.append("}")
    L.append("")
    L.append("static std::string _extract_bracket_payload(const std::string& text, const std::string& key, char open_c, char close_c) {")
    L.append("    size_t pos = _find_key(text, key);")
    L.append("    pos = text.find(':', pos);")
    L.append("    if (pos == std::string::npos) throw std::runtime_error(\"Malformed JSON near key: \" + key);")
    L.append("    pos = _skip_ws(text, pos + 1);")
    L.append("    if (pos >= text.size() || text[pos] != open_c) throw std::runtime_error(\"Expected bracket after key: \" + key);")
    L.append("    size_t end = _find_matching(text, pos, open_c, close_c);")
    L.append("    return text.substr(pos + 1, end - pos - 1);")
    L.append("}")
    L.append("")
    L.append("static double _extract_number_field(const std::string& text, const std::string& key) {")
    L.append("    size_t pos = _find_key(text, key);")
    L.append("    pos = text.find(':', pos);")
    L.append("    if (pos == std::string::npos) throw std::runtime_error(\"Malformed numeric JSON field: \" + key);")
    L.append("    pos = _skip_ws(text, pos + 1);")
    L.append("    size_t end = pos;")
    L.append("    while (end < text.size() && (std::isdigit(static_cast<unsigned char>(text[end])) || text[end] == '-' || text[end] == '+' || text[end] == '.' || text[end] == 'e' || text[end] == 'E')) ++end;")
    L.append("    return std::stod(text.substr(pos, end - pos));")
    L.append("}")
    L.append("")
    L.append("static int _extract_int_field(const std::string& text, const std::string& key) {")
    L.append("    return static_cast<int>(_extract_number_field(text, key));")
    L.append("}")
    L.append("")
    L.append("static uint64_t _extract_uint64_field_or_default(const std::string& text, const std::string& key, uint64_t default_value) {")
    L.append("    size_t pos = _find_key_optional(text, key);")
    L.append("    if (pos == std::string::npos) return default_value;")
    L.append("    pos = text.find(':', pos);")
    L.append("    if (pos == std::string::npos) throw std::runtime_error(\"Malformed numeric JSON field: \" + key);")
    L.append("    pos = _skip_ws(text, pos + 1);")
    L.append("    size_t end = pos;")
    L.append("    while (end < text.size() && (std::isdigit(static_cast<unsigned char>(text[end])) || text[end] == '-' || text[end] == '+' || text[end] == '.' || text[end] == 'e' || text[end] == 'E')) ++end;")
    L.append("    return static_cast<uint64_t>(std::stoull(text.substr(pos, end - pos)));")
    L.append("}")
    L.append("")
    L.append("static int _extract_int_field_or_default(const std::string& text, const std::string& key, int default_value) {")
    L.append("    size_t pos = _find_key_optional(text, key);")
    L.append("    if (pos == std::string::npos) return default_value;")
    L.append("    pos = text.find(':', pos);")
    L.append("    if (pos == std::string::npos) throw std::runtime_error(\"Malformed numeric JSON field: \" + key);")
    L.append("    pos = _skip_ws(text, pos + 1);")
    L.append("    size_t end = pos;")
    L.append("    while (end < text.size() && (std::isdigit(static_cast<unsigned char>(text[end])) || text[end] == '-' || text[end] == '+' || text[end] == '.' || text[end] == 'e' || text[end] == 'E')) ++end;")
    L.append("    return static_cast<int>(std::stoi(text.substr(pos, end - pos)));")
    L.append("}")
    L.append("")
    L.append("static std::vector<int> _parse_int_array(const std::string& payload) {")
    L.append("    std::vector<int> values;")
    L.append("    size_t pos = 0;")
    L.append("    while (true) {")
    L.append("        pos = _skip_ws(payload, pos);")
    L.append("        if (pos >= payload.size()) break;")
    L.append("        size_t end = pos;")
    L.append("        while (end < payload.size() && payload[end] != ',') ++end;")
    L.append("        std::string token = payload.substr(pos, end - pos);")
    L.append("        size_t token_pos = _skip_ws(token, 0);")
    L.append("        if (token_pos < token.size()) values.push_back(std::stoi(token.substr(token_pos)));")
    L.append("        if (end >= payload.size()) break;")
    L.append("        pos = end + 1;")
    L.append("    }")
    L.append("    return values;")
    L.append("}")
    L.append("")
    L.append("static std::vector<std::string> _split_top_level_objects(const std::string& payload) {")
    L.append("    std::vector<std::string> objects;")
    L.append("    size_t pos = 0;")
    L.append("    while (true) {")
    L.append("        pos = _skip_ws(payload, pos);")
    L.append("        if (pos >= payload.size()) break;")
    L.append("        if (payload[pos] == ',') { ++pos; continue; }")
    L.append("        if (payload[pos] != '{') throw std::runtime_error(\"Expected JSON object while parsing init belief.\");")
    L.append("        size_t end = _find_matching(payload, pos, '{', '}');")
    L.append("        objects.push_back(payload.substr(pos, end - pos + 1));")
    L.append("        pos = end + 1;")
    L.append("    }")
    L.append("    return objects;")
    L.append("}")
    L.append("")
    L.append("static JsonBeliefData _load_init_belief_json(const std::string& path) {")
    L.append("    std::string text = _read_text_file(path);")
    L.append("    JsonBeliefData belief;")
    L.append("    belief.grounded_predicates_count = _extract_int_field(text, \"grounded_predicates_count\");")
    L.append("    belief.last_observation_bits = _extract_uint64_field_or_default(text, \"last_observation_bits\", 0ULL);")
    L.append("    belief.has_last_observation = _extract_int_field_or_default(text, \"has_last_observation\", 0);")
    L.append("    std::string particles_payload = _extract_bracket_payload(text, \"particles\", '[', ']');")
    L.append("    for (const std::string& particle_text : _split_top_level_objects(particles_payload)) {")
    L.append("        JsonBeliefParticle particle;")
    L.append("        particle.probability = _extract_number_field(particle_text, \"probability\");")
    L.append("        particle.true_indices = _parse_int_array(_extract_bracket_payload(particle_text, \"true_indices\", '[', ']'));")
    L.append("        belief.particles.push_back(particle);")
    L.append("    }")
    L.append("    return belief;")
    L.append("}")
    L.append("")
    return "\n".join(L)


def _gen_fwd_fn(data: dict, cn: str, nw: int) -> str:
    """Generate ForwardAction - FIXED version.
    
    Fixes:
    1. Proper branch index mapping (skip branch_index is None guard)
    2. Deterministic effects handled (no branch_index)
    3. Reward conditions use original state (save before effect)
    4. Adaptive inline expressions for reward conditions
    5. Uses bw_sample_branch instead of DerivedRandom
    """
    funcs = data["forward_actions"]
    L = []
    L.append(f"void {cn}::ForwardAction(int action, uint64_t* bits, uint64_t& rng_seed, double& reward) const {{")
    L.append(f"    reward = 0.0;")
    L.append(f"    switch (action) {{")
    for i, func in enumerate(funcs):
        info = _extract_forward_action_logic(func, i, nw)
        L.append(f"    case {i}: {{")
        ws = info.get("prob_weights", [])
        branches = info.get("prob_branches", [])
        det = info.get("deterministic_effect")
        crs = info.get("cond_rewards", [])
        has_conds = len(crs) > 0
        has_effect = bool(branches) or (det is not None and (det[0] != 0 or det[1] != 0))
        # Save original state if we have reward conditions AND an effect
        if has_conds and has_effect:
            L.append(f"        uint64_t orig[N_WORDS]; bw_copy<N_WORDS>(orig, bits);")
        # Apply action effect
        if ws and branches:
            L.append(f"        double _r = Random::NextGoldenRandom(rng_seed);")
            L.append(f"        int bi = bw_sample_branch(_r, ACT_{i}_W, ACT_{i}_NB);")
            for j in range(len(branches)):
                sm, cm = branches[j]
                if sm == 0 and cm == 0:
                    L.append(f"        if (bi == {j}) {{ /* no effect */ }}")
                else:
                    L.append(f"        if (bi == {j}) bw_apply_effect<N_WORDS>(bits, ACT_{i}_S_{j}, ACT_{i}_C_{j});")
        elif det is not None:
            sm, cm = det
            if sm != 0 or cm != 0:
                L.append(f"        bw_apply_effect<N_WORDS>(bits, ACT_{i}_S_0, ACT_{i}_C_0);")
        # Reward conditions (inline expressions, use orig if saved)
        state_var = "orig" if (has_conds and has_effect) else "bits"
        for (expr_str, delta) in crs:
            rendered_expr = expr_str.replace("__STATE__", state_var)
            L.append(f"        if ({rendered_expr}) reward += {delta};")
        if info.get("has_goal_check"):
            L.append(f"        if (IsGoal(bits)) reward += GOAL_REWARD;")
        L.append(f"        break; }}")
    L.append(f"    default: break; }}")
    L.append("}")
    L.append("")
    return "\n".join(L)


def _gen_obs_fn(data: dict, cn: str, nw: int) -> str:
    """Generate Observe - uses bw_sample_branch + bw_derive_rand (no DerivedRandom)."""
    ofs = data["observe_with_rules"]
    n = len(ofs)
    empty_obs_index = next(
        (
            index
            for index, observable in enumerate(data.get("observables", []))
            if observable == "(obs-nothing)"
        ),
        None,
    )
    L = []
    L.append(f"OBS_TYPE {cn}::Observe(ACT_TYPE action, const uint64_t* bits, uint64_t& rng_seed) const {{")
    L.append(f"    bool profile_enabled = {cn}ProfileEnabled();")
    L.append(f"    auto observe_start = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{{}};")
    L.append(f"    OBS_TYPE final_obs = 0;")
    L.append(f"    uint64_t candidate_rules = 0;")
    L.append(f"    uint64_t cond_checks = 0;")
    L.append(f"    uint64_t fired_rules = 0;")
    L.append(f"    for (int r = 0; r < {n}; ++r) {{")
    L.append(f"        candidate_rules++;")
    L.append(f"        if (action >= 0 && action < TOTAL_ACTIONS && ACTION_IS_ACTIVE_PERCEPTION[action] && OBS_RULE_SKIP_BEFORE_ON_ACTIVE_PERCEPTION[r]) continue;")
    L.append(f"        cond_checks++;")
    L.append(f"        if (!OBS_COND_TABLE[r](bits)) continue;")
    L.append(f"        fired_rules++;")
    L.append(f"        OBS_TYPE ob = 0, om = 0;")
    L.append(f"        double obs_rand = Random::NextGoldenRandom(rng_seed);")
    L.append(f"        switch (r) {{")
    for i, func in enumerate(ofs):
        info = _extract_observe_logic(func, nw)
        ws = info.get("prob_weights", [])
        bs = info.get("branches", [])
        L.append(f"        case {i}: {{")
        if ws:
            L.append(f"            int bi = bw_sample_branch(obs_rand, OBS_{i}_W, OBS_{i}_NB);")
            for j in range(len(bs)):
                L.append(f"            if (bi == {j}) {{ ob = OBS_{i}_BITS_{j}; om = OBS_{i}_MASK_{j}; }}")
        L.append(f"            break; }}")
    L.append(f"        default: break; }}")
    L.append(f"        final_obs |= (ob & om);")
    L.append(f"    }}")
    if empty_obs_index is not None and empty_obs_index < 64:
        L.append(f"    const OBS_TYPE empty_obs_bit = static_cast<OBS_TYPE>(1ULL << {empty_obs_index});")
        L.append(f"    if ((final_obs & empty_obs_bit) && (final_obs & ~empty_obs_bit)) {{")
        L.append(f"        final_obs &= ~empty_obs_bit;")
        L.append(f"    }}")
    L.append(f"    if (profile_enabled) {{")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.observe_calls++;")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.observe_candidate_rules += candidate_rules;")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.observe_cond_checks += cond_checks;")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.observe_fired_rules += fired_rules;")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.observe_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - observe_start).count();")
    L.append(f"    }}")
    L.append(f"    final_obs = AppendFullyObservedStateToObs(bits, final_obs);")
    L.append(f"    return final_obs;")
    L.append("}")
    L.append("")
    return "\n".join(L)


def _gen_init_belief(data: dict, cn: str, sn: str, nw: int) -> str:
    L = []
    L.append(f'Belief* {cn}::InitialBelief(const State* start, string type) const {{')
    L.append(f'    if (type == "DEFAULT" || type == "PARTICLE") {{')
    L.append(f'        JsonBeliefData belief = _load_init_belief_json(INIT_BELIEF_JSON_PATH);')
    L.append(f'        if (belief.grounded_predicates_count != N_PREDICATES) {{')
    L.append(f'            cerr << "[{cn}::InitialBelief] grounded_predicates_count mismatch." << endl;')
    L.append(f'            exit(1);')
    L.append(f'        }}')
    L.append(f'        vector<State*> particles;')
    L.append(f'        for (int pi = 0; pi < static_cast<int>(belief.particles.size()); ++pi) {{')
    L.append(f'            const auto& belief_particle = belief.particles[pi];')
    L.append(f'            if (belief_particle.probability <= 0.0) continue;')
    L.append(f'            {sn}* p = static_cast<{sn}*>(Allocate(-1, belief_particle.probability));')
    L.append(f'            for (int w = 0; w < N_WORDS; ++w) p->bits[w] = 0;')
    L.append(f'            for (int idx : belief_particle.true_indices) {{')
    L.append(f'                int word = idx / 64;')
    L.append(f'                int bit = idx % 64;')
    L.append(f'                if (word >= 0 && word < N_WORDS) p->bits[word] |= (1ULL << bit);')
    L.append(f'            }}')
    L.append(f'            p->rng_seed = (uint64_t)(pi + 1) * 0x9E3779B97F4A7C15ULL;')
    L.append(f'            p->last_obs = static_cast<OBS_TYPE>(belief.last_observation_bits);')
    L.append(f'            p->has_last_obs = belief.has_last_observation != 0;')
    L.append(f'            particles.push_back(p);')
    L.append(f'        }}')
    L.append(f'        if (particles.empty()) {{')
    L.append(f'            cerr << "[{cn}::InitialBelief] particle belief is empty." << endl;')
    L.append(f'            exit(1);')
    L.append(f'        }}')
    L.append(f'        return new ParticleBelief(particles, this, NULL, false);')
    L.append(f'    }} else {{')
    L.append(f'        cerr << "[{cn}::InitialBelief] Unsupported: " << type << endl; exit(1);')
    L.append(f'    }}')
    L.append(f'}}')
    L.append("")
    return "\n".join(L)


def _gen_dp_action_fn(data: dict, cn: str) -> str:
    n = len(data["check_dp_conditions"])
    L = []
    L.append(f"int {cn}::GetDefaultPolicyAction(const uint64_t* bits) const {{")
    L.append(f"    bool profile_enabled = {cn}ProfileEnabled();")
    L.append("    auto dp_action_start = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};")
    L.append("    if (profile_enabled) {")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.default_policy_action_lookup_calls++;")
    L.append("    }")
    L.append(f"    for (int r = 0; r < {n}; ++r) {{")
    L.append(f"        if (DP_COND_TABLE[r](bits)) {{")
    L.append("            if (profile_enabled) {")
    L.append(f"                G_{cn.upper()}_PROFILE_STATS.default_policy_action_lookup_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - dp_action_start).count();")
    L.append("            }")
    L.append(f"            return DP_AIDS[r];")
    L.append(f"        }}")
    L.append(f"    }}")
    L.append("    if (profile_enabled) {")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.default_policy_action_lookup_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - dp_action_start).count();")
    L.append("    }")
    L.append(f"    return 0;")
    L.append("}")
    L.append("")
    return "\n".join(L)


def _gen_upper(cn: str) -> str:
    L = []
    L.append(f"ScenarioUpperBound* {cn}::CreateScenarioUpperBound(string name, string pbname) const {{")
    L.append(f'    if (name == "TRIVIAL" || name == "DEFAULT") return new TrivialParticleUpperBound(this);')
    L.append(f'    cerr << "Unsupported upper bound: " << name << endl; exit(1);')
    L.append("}")
    L.append("")
    return "\n".join(L)


def _gen_legal_actions(cn: str, sn: str) -> str:
    L = []
    L.append(f"vector<ACT_TYPE> {cn}::GetLegalActions(vector<State*> particles) const {{")
    L.append(f"    bool profile_enabled = {cn}ProfileEnabled();")
    L.append("    auto legal_start = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};")
    L.append(f"    vector<ACT_TYPE> legal;")
    L.append(f"    vector<const {sn}*> unique_particles;")
    L.append("    unique_particles.reserve(particles.size());")
    L.append(f"    for (int i = 0; i < (int)particles.size(); ++i) {{")
    L.append(f"        const {sn}& s = static_cast<const {sn}&>(*particles[i]);")
    L.append("        bool duplicate = false;")
    L.append("        for (int j = 0; j < (int)unique_particles.size(); ++j) {")
    L.append("            if (memcmp(unique_particles[j]->bits, s.bits, sizeof(s.bits)) == 0) {")
    L.append("                duplicate = true;")
    L.append("                break;")
    L.append("            }")
    L.append("        }")
    L.append("        if (!duplicate) unique_particles.push_back(&s);")
    L.append("    }")
    L.append("    uint64_t precondition_checks = 0;")
    L.append(f"    for (int a = 0; a < TOTAL_ACTIONS; ++a) {{")
    L.append("        bool found_legal_particle = false;")
    L.append("        bool should_prune_repeated_observation = ACTION_DETERMINISTIC_OBSERVATION_FILTER[a];")
    L.append("        bool should_prune_uninformative_observation = ACTION_DETERMINISTIC_OBSERVATION_FILTER[a];")
    L.append("        bool has_reference_explicit_obs = false;")
    L.append("        OBS_TYPE reference_explicit_obs = 0;")
    L.append(f"        for (int i = 0; i < (int)unique_particles.size(); ++i) {{")
    L.append(f"            const {sn}& s = *unique_particles[i];")
    L.append("            precondition_checks++;")
    L.append(f"            if (CheckActionPrecondition(a, s.bits)) {{")
    L.append(f"                if (ACTION_DETERMINISTIC_NOOP_FILTER[a] && !DeterministicActionWouldChangeState(a, s.bits)) {{")
    L.append(f"                    continue;")
    L.append(f"                }}")
    L.append("                found_legal_particle = true;")
    L.append("                if (should_prune_repeated_observation &&")
    L.append(f"                        !ShouldPruneRepeatedDeterministicObservation(a, s.bits, s.last_obs, s.has_last_obs, s.rng_seed)) {{")
    L.append("                    should_prune_repeated_observation = false;")
    L.append("                }")
    L.append("                if (should_prune_uninformative_observation) {")
    L.append("                    OBS_TYPE predicted_explicit_obs = PredictDeterministicExplicitObservation(a, s.bits, s.rng_seed);")
    L.append("                    if (!has_reference_explicit_obs) {")
    L.append("                        has_reference_explicit_obs = true;")
    L.append("                        reference_explicit_obs = predicted_explicit_obs;")
    L.append("                    } else if (predicted_explicit_obs != reference_explicit_obs) {")
    L.append("                        should_prune_uninformative_observation = false;")
    L.append("                    }")
    L.append("                }")
    L.append(f"            }}")
    L.append(f"        }}")
    L.append("        if (!found_legal_particle) continue;")
    L.append("        if (should_prune_uninformative_observation) continue;")
    L.append("        if (should_prune_repeated_observation) continue;")
    L.append(f"        legal.push_back(a);")
    L.append(f"    }}")
    L.append(f"    if (legal.empty())")
    L.append(f"        legal.push_back(TOTAL_ACTIONS);")
    L.append("    if (profile_enabled) {")
    L.append(f"        auto legal_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - legal_start).count();")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.legal_calls++;")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.legal_seconds += legal_elapsed;")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.legal_particles += particles.size();")
    L.append(f"        G_{cn.upper()}_PROFILE_STATS.legal_precondition_checks += precondition_checks;")
    L.append("    }")
    L.append(f"    return legal;")
    L.append("}")
    L.append("")
    return "\n".join(L)


def _gen_deterministic_action_change_fn(cn: str) -> str:
    L: list[str] = []
    L.append(f"bool {cn}::DeterministicActionWouldChangeState(int action,")
    L.append("        const uint64_t* bits) const {")
    L.append("    if (action < 0 || action >= TOTAL_ACTIONS) return true;")
    L.append("    if (!ACTION_DETERMINISTIC_NOOP_FILTER[action]) return true;")
    L.append("    const uint64_t* set_masks = ACTION_DETERMINISTIC_SET_MASKS[action];")
    L.append("    const uint64_t* clear_masks = ACTION_DETERMINISTIC_CLEAR_MASKS[action];")
    L.append("    for (int w = 0; w < N_WORDS; ++w) {")
    L.append("        uint64_t set_changes = set_masks[w] & ~bits[w];")
    L.append("        uint64_t clear_changes = clear_masks[w] & bits[w];")
    L.append("        if ((set_changes | clear_changes) != 0ULL) return true;")
    L.append("    }")
    L.append("    return false;")
    L.append("}")
    L.append("")
    return "\n".join(L)


def _gen_repeated_observation_prune_fn(cn: str) -> str:
    L: list[str] = []
    L.append(f"OBS_TYPE {cn}::PredictDeterministicExplicitObservation(int action,")
    L.append("        const uint64_t* bits, uint64_t rng_seed) const {")
    L.append("    if (action < 0 || action >= TOTAL_ACTIONS) return 0;")
    L.append("    if (!ACTION_DETERMINISTIC_OBSERVATION_FILTER[action]) return 0;")
    L.append("    uint64_t predicted_bits[N_WORDS];")
    L.append("    bw_copy<N_WORDS>(predicted_bits, bits);")
    L.append("    uint64_t local_seed = rng_seed;")
    L.append("    double ignored_reward = 0.0;")
    L.append("    ForwardAction(action, predicted_bits, local_seed, ignored_reward);")
    L.append("    OBS_TYPE predicted_obs = Observe(action, predicted_bits, local_seed);")
    L.append("    OBS_TYPE explicit_obs_mask = ExplicitObservationMask();")
    L.append("    return predicted_obs & explicit_obs_mask;")
    L.append("}")
    L.append("")
    L.append(f"bool {cn}::ShouldPruneRepeatedDeterministicObservation(int action,")
    L.append("        const uint64_t* bits, OBS_TYPE last_obs, bool has_last_obs,")
    L.append("        uint64_t rng_seed) const {")
    L.append("    if (!has_last_obs) return false;")
    L.append("    if (action < 0 || action >= TOTAL_ACTIONS) return false;")
    L.append("    if (!ACTION_DETERMINISTIC_OBSERVATION_FILTER[action]) return false;")
    L.append("    OBS_TYPE explicit_obs_mask = ExplicitObservationMask();")
    L.append("    OBS_TYPE predicted_obs = PredictDeterministicExplicitObservation(action, bits, rng_seed);")
    L.append("    return (predicted_obs & explicit_obs_mask) == (last_obs & explicit_obs_mask);")
    L.append("}")
    L.append("")
    return "\n".join(L)

def _gen_lower(data: dict, cn: str, sn: str) -> str:
    """Generate ScenarioLowerBound subclass with RecursiveValue().

    Follows the same pattern as DefaultPolicy::RecursiveValue in DESPOT:
    at each depth all particles share one action, are stepped together,
    then partitioned by observation for recursive evaluation.
    Uses state rng_seed (4-arg Step) instead of RandomStreams."""
    ndp = len(data["check_dp_conditions"])
    L = []
    L.append(f"class {cn}DefLowerBound : public ScenarioLowerBound {{")
    L.append(f"    const {cn}* m_;")
    L.append(f"public:")
    L.append(f"    {cn}DefLowerBound(const DSPOMDP* model)")
    L.append(f"        : ScenarioLowerBound(model), m_(static_cast<const {cn}*>(model)) {{}}")
    # ---------- Value (entry point) ----------
    L.append(f"    ValuedAction Value(const vector<State*>& particles,")
    L.append(f"            RandomStreams& streams, History& history) const override {{")
    L.append(f"        bool profile_enabled = {cn}ProfileEnabled();")
    L.append(f"        auto dp_value_start = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{{}};")
    L.append(f"        vector<State*> copy;")
    L.append(f"        for (int i = 0; i < (int)particles.size(); ++i)")
    L.append(f"            copy.push_back(model_->Copy(particles[i]));")
    L.append(f"        int initial_depth = history.Size();")
    L.append(f"        ValuedAction va = RecursiveValue(copy, initial_depth);")
    L.append(f"        for (int i = 0; i < (int)copy.size(); ++i)")
    L.append(f"            model_->Free(copy[i]);")
    L.append(f"        if (profile_enabled) {{")
    L.append(f"            G_{cn.upper()}_PROFILE_STATS.default_policy_calls++;")
    L.append(f"            G_{cn.upper()}_PROFILE_STATS.default_policy_particles += particles.size();")
    L.append(f"            G_{cn.upper()}_PROFILE_STATS.default_policy_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - dp_value_start).count();")
    L.append(f"        }}")
    L.append(f"        return va;")
    L.append(f"    }}")
    # ---------- SelectAction ----------
    L.append(f"    ACT_TYPE SelectAction(const vector<State*>& particles, int depth) const {{")
    L.append(f"        bool profile_enabled = {cn}ProfileEnabled();")
    L.append(f"        auto dp_select_start = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{{}};")
    L.append(f"        vector<ACT_TYPE> legal = m_->GetLegalActions(particles);")
    L.append(f"        if (profile_enabled) {{")
    L.append(f"            G_{cn.upper()}_PROFILE_STATS.default_policy_select_calls++;")
    L.append(f"        }}")
    L.append(f"        if (legal.size() <= 1) {{")
    L.append(f"            if (profile_enabled) {{")
    L.append(f"                G_{cn.upper()}_PROFILE_STATS.default_policy_select_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - dp_select_start).count();")
    L.append(f"            }}")
    L.append(f"            return legal[0];")
    L.append(f"        }}")
    L.append(f"        // Preserve original rule-priority semantics while scanning fewer candidate rules.")
    L.append(f"        ACT_TYPE best_dp = -1;")
    L.append(f"        int best_rule = {ndp};")
    L.append(f"        uint64_t dp_rule_checks = 0;")
    L.append(f"        vector<const {sn}*> unique_particles;")
    L.append(f"        unique_particles.reserve(particles.size());")
    L.append(f"        for (int i = 0; i < (int)particles.size(); ++i) {{")
    L.append(f"            const {sn}& s = static_cast<const {sn}&>(*particles[i]);")
    L.append(f"            bool duplicate = false;")
    L.append(f"            for (int j = 0; j < (int)unique_particles.size(); ++j) {{")
    L.append(f"                if (memcmp(unique_particles[j]->bits, s.bits, sizeof(s.bits)) == 0) {{")
    L.append(f"                    duplicate = true;")
    L.append(f"                    break;")
    L.append(f"                }}")
    L.append(f"            }}")
    L.append(f"            if (!duplicate) unique_particles.push_back(&s);")
    L.append(f"        }}")
    L.append(f"        for (int i = 0; i < (int)unique_particles.size(); ++i) {{")
    L.append(f"            const {sn}& s = *unique_particles[i];")
    L.append(f"            for (int j = 0; j < (int)legal.size(); ++j) {{")
    L.append(f"                int action = legal[j];")
    L.append(f"                const DpBucketDesc* buckets = DP_BUCKET_TABLE[action];")
    L.append(f"                int bucket_count = DP_BUCKET_COUNT[action];")
    L.append(f"                for (int b = 0; b < bucket_count; ++b) {{")
    L.append(f"                    const DpBucketDesc& bucket = buckets[b];")
    L.append(f"                    if (bucket.word >= 0 && (s.bits[bucket.word] & bucket.mask) != bucket.mask) continue;")
    L.append(f"                    for (int k = 0; k < bucket.count; ++k) {{")
    L.append(f"                        int r = bucket.rules[k];")
    L.append(f"                        if (r >= best_rule) break;")
    L.append(f"                        if (!bw_check_clause<N_WORDS>(s.bits, DP_RULE_MUST_TRUE[r], DP_RULE_MUST_FALSE[r])) continue;")
    L.append(f"                        dp_rule_checks++;")
    L.append(f"                        if (DP_COND_TABLE[r](s.bits)) {{")
    L.append(f"                            best_rule = r;")
    L.append(f"                            best_dp = action;")
    L.append(f"                            if (best_rule == 0) break;")
    L.append(f"                        }}")
    L.append(f"                    }}")
    L.append(f"                    if (best_rule == 0) break;")
    L.append(f"                }}")
    L.append(f"                if (best_rule == 0) break;")
    L.append(f"            }}")
    L.append(f"            if (best_rule == 0) break;")
    L.append(f"        }}")
    L.append(f"        if (best_dp < 0) {{")
    L.append(f"            // No rule matched: pick uniformly at random from legal actions")
    L.append(f"            static thread_local std::mt19937 rng_fallback(std::random_device{{}}());")
    L.append(f"            std::uniform_int_distribution<int> fallback_pick(0, (int)legal.size() - 1);")
    L.append(f"            if (profile_enabled) {{")
    L.append(f"                G_{cn.upper()}_PROFILE_STATS.default_policy_rule_checks += dp_rule_checks;")
    L.append(f"                G_{cn.upper()}_PROFILE_STATS.default_policy_select_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - dp_select_start).count();")
    L.append(f"            }}")
    L.append(f"            return legal[fallback_pick(rng_fallback)];")
    L.append(f"        }}")
    L.append(f"        // Exploit best_dp with DP_EXPLOIT_PROB, else random from others")
    L.append(f"        static thread_local std::mt19937 rng(std::random_device{{}}());")
    L.append(f"        std::uniform_real_distribution<double> dist01(0.0, 1.0);")
    L.append(f"        double sample01 = dist01(rng);")
    L.append(f"        if (sample01 < {cn}::DP_EXPLOIT_PROB) {{")
    L.append(f"            if (profile_enabled) {{")
    L.append(f"                G_{cn.upper()}_PROFILE_STATS.default_policy_rule_checks += dp_rule_checks;")
    L.append(f"                G_{cn.upper()}_PROFILE_STATS.default_policy_select_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - dp_select_start).count();")
    L.append(f"            }}")
    L.append(f"            return best_dp;")
    L.append(f"        }}")
    L.append(f"        // Pick uniformly from legal actions excluding best_dp")
    L.append(f"        vector<ACT_TYPE> others;")
    L.append(f"        for (int j = 0; j < (int)legal.size(); ++j)")
    L.append(f"            if (legal[j] != best_dp) others.push_back(legal[j]);")
    L.append(f"        if (others.empty()) {{")
    L.append(f"            if (profile_enabled) {{")
    L.append(f"                G_{cn.upper()}_PROFILE_STATS.default_policy_rule_checks += dp_rule_checks;")
    L.append(f"                G_{cn.upper()}_PROFILE_STATS.default_policy_select_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - dp_select_start).count();")
    L.append(f"            }}")
    L.append(f"            return best_dp;")
    L.append(f"        }}")
    L.append(f"        std::uniform_int_distribution<int> pick(0, (int)others.size() - 1);")
    L.append(f"        ACT_TYPE selected = others[pick(rng)];")
    L.append(f"        if (profile_enabled) {{")
    L.append(f"            G_{cn.upper()}_PROFILE_STATS.default_policy_rule_checks += dp_rule_checks;")
    L.append(f"            G_{cn.upper()}_PROFILE_STATS.default_policy_select_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - dp_select_start).count();")
    L.append(f"        }}")
    L.append(f"        return selected;")
    L.append(f"    }}")
    # ---------- RecursiveValue ----------
    L.append(f"    ValuedAction RecursiveValue(vector<State*>& particles, int depth) const {{")
    L.append(f"        if (particles.empty() || depth >= Globals::config.max_policy_sim_len) {{")
    L.append(f"            double val = 0.0;")
    L.append(f"            ValuedAction best = m_->GetBestAction();")
    L.append(f"            for (int i = 0; i < (int)particles.size(); ++i)")
    L.append(f"                val += best.value * particles[i]->weight;")
    L.append(f"            return ValuedAction(best.action, val);")
    L.append(f"        }}")
    L.append(f"        ACT_TYPE action = SelectAction(particles, depth);")
    L.append(f"        double value = 0.0;")
    L.append(f"        double immediate_value = 0.0;")
    L.append(f"        map<OBS_TYPE, vector<State*>> partitions;")
    L.append(f"        uint64_t dp_step_calls = 0;")
    L.append(f"        for (int i = 0; i < (int)particles.size(); ++i) {{")
    L.append(f"            State* particle = particles[i];")
    L.append(f"            double reward; OBS_TYPE obs;")
    L.append(f"            dp_step_calls++;")
    L.append(f"            bool terminal = m_->Step(*particle, action, reward, obs);")
    L.append(f"            immediate_value += reward * particle->weight;")
    L.append(f"            value += reward * particle->weight;")
    L.append(f"            if (!terminal) partitions[obs].push_back(particle);")
    L.append(f"        }}")
    L.append(f"        for (map<OBS_TYPE, vector<State*>>::iterator it = partitions.begin();")
    L.append(f"                it != partitions.end(); ++it) {{")
    L.append(f"            ValuedAction va = RecursiveValue(it->second, depth + 1);")
    L.append(f"            value += Globals::Discount() * va.value;")
    L.append(f"        }}")
    L.append(f"        if ({cn}ProfileEnabled()) {{")
    L.append(f"            G_{cn.upper()}_PROFILE_STATS.default_policy_step_calls += dp_step_calls;")
    L.append(f"        }}")
    L.append(f"        return ValuedAction(action, value);")
    L.append(f"    }}")
    L.append(f"}};")
    # CreateScenarioLowerBound
    L.append(f"ScenarioLowerBound* {cn}::CreateScenarioLowerBound(string name, string pbname) const {{")
    L.append(f'    if (name == "TRIVIAL" || name == "DEFAULT") return new TrivialParticleLowerBound(this);')
    L.append(f'    if (name == "POLICY") return new {cn}DefLowerBound(this);')
    L.append(f'    cerr << "Unsupported lower bound: " << name << endl; exit(1);')
    L.append("}")
    L.append("")
    return "\n".join(L)


def _gen_mem(cn: str, sn: str) -> str:
    L = []
    L.append(f"State* {cn}::Allocate(int state_id, double weight) const {{")
    L.append(f"    {sn}* s = memory_pool_.Allocate();")
    L.append(f"    s->state_id = state_id; s->weight = weight;")
    L.append(f"    memset(s->bits, 0, sizeof(s->bits));")
    L.append(f"    s->rng_seed = 0;")
    L.append(f"    return s;")
    L.append("}")
    L.append(f"State* {cn}::Copy(const State* particle) const {{")
    L.append(f"    {sn}* s = memory_pool_.Allocate();")
    L.append(f"    *s = *static_cast<const {sn}*>(particle);")
    L.append(f"    s->SetAllocated(); return s;")
    L.append("}")
    L.append(f"void {cn}::Free(State* particle) const {{")
    L.append(f"    memory_pool_.Free(static_cast<{sn}*>(particle));")
    L.append("}")
    L.append(f"int {cn}::NumActiveParticles() const {{ return memory_pool_.num_allocated(); }}")
    L.append("")
    return "\n".join(L)


def _gen_disp(cn: str, sn: str) -> str:
    L = []
    # PrintState: translate bits to predicate names
    L.append(f"void {cn}::PrintState(const State& state, ostream& out) const {{")
    L.append(f"    const {sn}& s = static_cast<const {sn}&>(state);")
    L.append(f'    out << "State [true predicates]:";')
    L.append(f"    for (int i = 0; i < N_PREDICATES; ++i) {{")
    L.append(f"        int w = i / 64, b = i % 64;")
    L.append(f"        if ((s.bits[w] >> b) & 1ULL)")
    L.append(f'            out << " " << PREDICATE_NAMES[i];')
    L.append(f"    }}")
    L.append(f'    out << endl;')
    L.append("}")
    # PrintAction: translate index to action name
    L.append(f"void {cn}::PrintAction(ACT_TYPE action, ostream& out) const {{")
    L.append(f"    if (action == TOTAL_ACTIONS)")
    L.append(f'        out << "Action(" << action << "): [no feasible action]" << endl;')
    L.append(f"    else if (action >= 0 && action < TOTAL_ACTIONS)")
    L.append(f'        out << "Action(" << action << "): " << ACTION_NAMES[action] << endl;')
    L.append(f"    else")
    L.append(f'        out << "Action: " << action << " [invalid]" << endl;')
    L.append("}")
    # PrintObs: translate obs bits to observable names
    L.append(f"void {cn}::PrintObs(const State& state, OBS_TYPE obs, ostream& out) const {{")
    L.append(f'    out << "Obs(" << obs << ") [true]:";')
    L.append(f"    for (int i = 0; i < N_OBSERVABLES; ++i) {{")
    L.append(f"        if ((obs >> i) & 1ULL)")
    L.append(f'            out << " " << OBSERVABLE_NAMES[i];')
    L.append(f"    }}")
    L.append(f"    for (int i = 0; i < N_FULLY_OBSERVED_PREDICATES; ++i) {{")
    L.append(f"        int obs_bit = FULLY_OBSERVED_OBS_BIT[i];")
    L.append(f"        if ((obs >> obs_bit) & 1ULL)")
    L.append(f'            out << " " << PREDICATE_NAMES[FULLY_OBSERVED_PREDICATE_INDEX[i]];')
    L.append(f"    }}")
    L.append(f'    out << endl;')
    L.append("}")
    # PrintBelief
    L.append(f"void {cn}::PrintBelief(const Belief& belief, ostream& out) const {{")
    L.append(f'    out << "Belief" << endl;')
    L.append("}")
    L.append("")
    return "\n".join(L)



# ---------------------------------------------------------------------------
# Planner with pybind11 binding
# ---------------------------------------------------------------------------


def _gen_probability_update_tables(
    *,
    action_effect_distributions: list[list[BitwiseEffectBranch]],
    observation_rule_distributions: list[list[BitwiseObservationBranch]],
) -> str:
    """Generate file-scope probability update lookup tables and C functions."""
    L = []
    act_entries = []
    act_counts = []
    for i, branches in enumerate(action_effect_distributions):
        if len(branches) > 1:
            act_entries.append(f"ACT_{i}_W")
            act_counts.append(str(len(branches)))
        else:
            act_entries.append("nullptr")
            act_counts.append("0")
    L.append(f"static const int N_PROB_ACTIONS = {len(action_effect_distributions)};")
    L.append("static double* ACTION_WEIGHT_TABLE[] = {" + ", ".join(act_entries) + "};")
    L.append("static const int ACTION_BRANCH_COUNT[] = {" + ", ".join(act_counts) + "};")
    L.append("")
    obs_entries = []
    obs_counts = []
    for i, branches in enumerate(observation_rule_distributions):
        if len(branches) > 1:
            obs_entries.append(f"OBS_{i}_W")
            obs_counts.append(str(len(branches)))
        else:
            obs_entries.append("nullptr")
            obs_counts.append("0")
    n_obs = len(observation_rule_distributions)
    L.append(f"static const int N_PROB_OBS_RULES = {n_obs};")
    if obs_entries:
        L.append("static double* OBS_WEIGHT_TABLE[] = {" + ", ".join(obs_entries) + "};")
        L.append("static const int OBS_BRANCH_COUNT[] = {" + ", ".join(obs_counts) + "};")
    else:
        L.append("static double** OBS_WEIGHT_TABLE = nullptr;")
        L.append("static const int* OBS_BRANCH_COUNT = nullptr;")
    L.append("")
    L.append("void update_action_weights(int action_id, const std::vector<double>& weights) {")
    L.append("    if (action_id < 0 || action_id >= N_PROB_ACTIONS) return;")
    L.append("    double* arr = ACTION_WEIGHT_TABLE[action_id];")
    L.append("    if (!arr) return;")
    L.append("    int n = ACTION_BRANCH_COUNT[action_id];")
    L.append("    for (int i = 0; i < n && i < (int)weights.size(); ++i) {")
    L.append("        arr[i] = weights[i];")
    L.append("    }")
    L.append("}")
    L.append("")
    L.append("void update_obs_weights(int rule_id, const std::vector<double>& weights) {")
    L.append("    if (rule_id < 0 || rule_id >= N_PROB_OBS_RULES) return;")
    L.append("    if (!OBS_WEIGHT_TABLE) return;")
    L.append("    double* arr = OBS_WEIGHT_TABLE[rule_id];")
    L.append("    if (!arr) return;")
    L.append("    int n = OBS_BRANCH_COUNT[rule_id];")
    L.append("    for (int i = 0; i < n && i < (int)weights.size(); ++i) {")
    L.append("        arr[i] = weights[i];")
    L.append("    }")
    L.append("}")
    L.append("")
    return "\n".join(L)



def _gen_hyperparams_yaml(data: dict) -> str:
    """Generate a YAML file with all hot-reloadable hyperparameters."""
    goal_reward = data.get("goal_reward", 100.0)
    lines = [
        "# DESPOT Hyperparameters (hot-reloadable)",
        "# Edit values below and they will be picked up at each planning step.",
        "",
        "# Model rewards",
        f"goal_reward: {goal_reward}",
        f"report_goal_failure_penalty: {data.get('report_goal_failure_penalty', -10.0)}",
        "action_invalid_penalty: -10.0",
        "action_no_change_penalty: -10.0",
        "",
        "# Default policy",
        "dp_exploit_prob: 0.9",
        "",
        "# DESPOT planner config",
        "time_per_move: 1.0",
        "num_scenarios: 512",
        "search_depth: 90",
        "max_policy_sim_len: 20",
        "sim_len: 20",
        "discount: 0.95",
        "pruning_constant: 0.0",
        "xi: 0.95",
        "root_seed: 42",
        "silence: true",
        "",
    ]
    return "\n".join(lines)


def _gen_planner_pybind(cn: str) -> str:
    """Generate biwise_pomdp_planner.cpp with pybind11 Python binding."""
    L = []
    L.append(f'#include "bitwise_pomdp_model.h"')
    L.append("")
    L.append("#include <despot/core/globals.h>")
    L.append("#include <despot/core/solver.h>")
    L.append("#include <despot/interface/belief.h>")
    L.append("#include <despot/solver/despot.h>")
    L.append("#include <despot/core/globals.h>")
    L.append("#include <despot/util/logging.h>")
    L.append("#include <despot/util/random.h>")
    L.append("#include <despot/util/seeds.h>")
    L.append("")
    L.append("#include <pybind11/pybind11.h>")
    L.append("#include <pybind11/stl.h>")
    L.append("")
    L.append("namespace py = pybind11;")
    L.append("using namespace despot;")
    L.append("")
    L.append(f"class {cn}Planner {{")
    L.append("private:")
    L.append(f"    {cn}* model_;")
    L.append("")
    L.append("public:")
    L.append(f"    {cn}Planner() : model_(nullptr) {{")
    L.append("        Globals::config.time_per_move = 1.0;")
    L.append("        Globals::config.num_scenarios = 512;")
    L.append("        Globals::config.search_depth = 90;")
    L.append("        Globals::config.max_policy_sim_len = 20;")
    L.append("        Globals::config.sim_len = 20;")
    L.append("        Globals::config.discount = 0.95;")
    L.append("        Globals::config.pruning_constant = 0.0;")
    L.append("        Globals::config.xi = 0.95;")
    L.append("        Globals::config.root_seed = 42;")
    L.append("        Globals::config.silence = true;")
    L.append("")
    L.append("        Seeds::root_seed(Globals::config.root_seed);")
    L.append("        Random::RANDOM = Random(Seeds::Next());")
    L.append("        logging::level(logging::ERROR);")
    L.append("")
    L.append(f"        model_ = new {cn}();")
    L.append("    }")
    L.append("")
    L.append(f"    ~{cn}Planner() {{")
    L.append("        delete model_;")
    L.append("    }")
    L.append("")
    L.append("    int MakePlanning() {")
    L.append("        model_->ResetProfileStats();")
    L.append('        Belief* belief = model_->InitialBelief(nullptr, "DEFAULT");')
    L.append('        ScenarioLowerBound* lower = model_->CreateScenarioLowerBound("POLICY", "DEFAULT");')
    L.append('        ScenarioUpperBound* upper = model_->CreateScenarioUpperBound("TRIVIAL", "DEFAULT");')
    L.append("")
    L.append("        DESPOT solver(model_, lower, upper, belief);")
    L.append("        ValuedAction result = solver.Search();")
    L.append("        if (!Globals::config.silence) model_->PrintProfileStats();")
    L.append("        return result.action;")
    L.append("    }")
    L.append("")
    L.append("    void SetBeliefJsonPath(const std::string& path) {")
    L.append("        set_init_belief_json_path(path);")
    L.append("    }")
    L.append("")
    L.append("    void SetRootSeed(unsigned int seed) {")
    L.append("        Globals::config.root_seed = seed;")
    L.append("        Seeds::root_seed(Globals::config.root_seed);")
    L.append("        Random::RANDOM = Random(Seeds::Next());")
    L.append("    }")
    L.append("")
    L.append("    void UpdateActionWeights(int action_id, const std::vector<double>& weights) {")
    L.append("        update_action_weights(action_id, weights);")
    L.append("    }")
    L.append("")
    L.append("    void UpdateObsWeights(int rule_id, const std::vector<double>& weights) {")
    L.append("        update_obs_weights(rule_id, weights);")
    L.append("    }")
    L.append("")
    L.append("    void UpdateHyperparams(double goal_reward, double report_goal_failure_penalty,")
    L.append("        double action_invalid_penalty,")
    L.append("        double action_no_change_penalty,")
    L.append("        double dead_end_penalty,")
    L.append("        double dp_exploit_prob, double time_per_move, int num_scenarios,")
    L.append("        int search_depth, int max_policy_sim_len, int sim_len,")
    L.append("        double discount, double pruning_constant, double xi,")
    L.append("        unsigned int root_seed, bool silence) {")
    L.append("        update_hyperparams(goal_reward, report_goal_failure_penalty,")
    L.append("            action_invalid_penalty, action_no_change_penalty, dead_end_penalty, dp_exploit_prob, time_per_move, num_scenarios,")
    L.append("            search_depth, max_policy_sim_len, sim_len,")
    L.append("            discount, pruning_constant, xi, root_seed, silence);")
    L.append("    }")
    L.append("};")
    L.append("")
    L.append("PYBIND11_MODULE(despot_planner, m) {")
    L.append('    m.doc() = "DESPOT POMDP Planner Python Binding";')
    L.append("")
    L.append(f'    py::class_<{cn}Planner>(m, "{cn}Planner")')
    L.append("        .def(py::init<>())")
    L.append(f'        .def("MakePlanning", &{cn}Planner::MakePlanning,')
    L.append('             "Run DESPOT planning and return best action index")')
    L.append(f'        .def("SetBeliefJsonPath", &{cn}Planner::SetBeliefJsonPath,')
    L.append('             "Set path to init_belief.json for next planning call")')
    L.append(f'        .def("SetRootSeed", &{cn}Planner::SetRootSeed,')
    L.append('             "Set DESPOT root seed for the next planning call")')
    L.append(f'        .def("UpdateActionWeights", &{cn}Planner::UpdateActionWeights,')
    L.append('             "Update probability weights for one action by ID")')
    L.append(f'        .def("UpdateObsWeights", &{cn}Planner::UpdateObsWeights,')
    L.append('             "Update probability weights for one observation rule by ID")')
    L.append(f'        .def("UpdateHyperparams", &{cn}Planner::UpdateHyperparams,')
    L.append('             "Update model and planner hyperparameters");')
    L.append("}")
    L.append("")
    return "\n".join(L)


def _gen_build_script() -> str:
    """Generate build.sh that compiles the pybind11 shared library."""
    return textwrap.dedent("""        #!/usr/bin/env bash
        set -e
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        cd "$SCRIPT_DIR"

        PYTHON_BIN="${PYTHON_BIN:-python3}"

        # Auto-detect pybind11 cmake dir from Python
        PYBIND11_DIR=$("$PYTHON_BIN" -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null || true)
        CMAKE_EXTRA=""
        if [ -n "$PYBIND11_DIR" ]; then
            CMAKE_EXTRA="-Dpybind11_DIR=$PYBIND11_DIR"
        fi
        if [ -n "${DESPOT_CORE_LIB:-}" ]; then
            CMAKE_EXTRA="$CMAKE_EXTRA -DDESPOT_PREBUILT_CORE_LIB=$DESPOT_CORE_LIB"
        fi

        cmake -S . -B build \
            -DPython3_EXECUTABLE="$PYTHON_BIN" \
            -DPython_EXECUTABLE="$PYTHON_BIN" \
            $CMAKE_EXTRA
        cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)

        SO_FILE=$(find build -maxdepth 1 -name 'despot_planner*.so' -o -name 'despot_planner*.dylib' | head -1)
        if [ -z "$SO_FILE" ]; then
            echo "ERROR: build failed, no .so/.dylib found" >&2
            exit 1
        fi
        echo "Built shared library: $SCRIPT_DIR/$SO_FILE"
    """).lstrip()


def _gen_pybind_cmakelists(despot_root_str: str) -> str:
    """Generate CMakeLists.txt for building a pybind11 shared library."""
    L = []
    L.append("cmake_minimum_required(VERSION 3.16)")
    L.append("project(despot_planner LANGUAGES CXX)")
    L.append("")
    L.append("set(CMAKE_CXX_STANDARD 17)")
    L.append("set(CMAKE_CXX_STANDARD_REQUIRED ON)")
    L.append("set(CMAKE_CXX_EXTENSIONS OFF)")
    L.append("")
    L.append(f'set(DESPOT_ROOT "{despot_root_str}")')
    L.append("")
    L.append("set(PYBIND11_FINDPYTHON ON)")
    L.append("find_package(Python COMPONENTS Interpreter Development REQUIRED)")
    L.append("find_package(pybind11 REQUIRED)")
    L.append("")
    L.append("pybind11_add_module(despot_planner")
    L.append("    bitwise_pomdp_model.cpp")
    L.append("    biwise_pomdp_planner.cpp")
    L.append(")")
    L.append("")
    L.append("target_include_directories(despot_planner PRIVATE")
    L.append('    "${DESPOT_ROOT}"')
    L.append('    "${DESPOT_ROOT}/despot/interface"')
    L.append('    "${CMAKE_CURRENT_SOURCE_DIR}"')
    L.append(")")
    L.append("")
    L.append("find_package(Threads REQUIRED)")
    L.append("if(DESPOT_PREBUILT_CORE_LIB)")
    L.append('    message(STATUS "Using prebuilt DESPOT core library: ${DESPOT_PREBUILT_CORE_LIB}")')
    L.append("    target_link_libraries(despot_planner PRIVATE Threads::Threads ${DESPOT_PREBUILT_CORE_LIB})")
    L.append("else()")
    L.append("    set(DESPOT_SOURCES")
    for s in DESPOT_CORE_SOURCE_REL_PATHS:
        L.append(f'        "${{DESPOT_ROOT}}/despot/{s}"')
    L.append("    )")
    L.append("    target_sources(despot_planner PRIVATE ${DESPOT_SOURCES})")
    L.append("    target_link_libraries(despot_planner PRIVATE Threads::Threads)")
    L.append("endif()")
    L.append("")
    return "\n".join(L)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Emit DESPOT C++ from bitwise model package.")
    parser.add_argument("module", help="Python import path to the bitwise_package")
    parser.add_argument("--output-dir", "-o", default="./despot_gen")
    parser.add_argument("--class-name", default="BitwisePOMDPModel")
    parser.add_argument("--despot-root", default=None,
                        help="Path to POMDPDDL_ccplus directory (auto-detected if omitted)")
    args = parser.parse_args()
    out = emit_despot_cpp_from_bitwise_package(
        args.module, args.output_dir,
        class_name=args.class_name,
        despot_root=args.despot_root,
    )
    print(f"Generated DESPOT C++ files in: {out}")
    so_files = list((Path(out) / "build").glob("despot_planner*.so")) + list((Path(out) / "build").glob("despot_planner*.dylib"))
    if so_files:
        print(f"Compiled shared library: {so_files[0].resolve()}")


if __name__ == "__main__":
    main()
