"""Configurable linter for the supported POMDPDDL subset."""

from __future__ import annotations

from dataclasses import dataclass

from .config import LinterConfig
from .diagnostics import LintResult
from ..data_structures.action import Action
from ..data_structures.default_policy_rule import DefaultPolicyRule
from ..data_structures.observable import Observable
from ..data_structures.observation_rule import ObservationRule
from ..data_structures.predicate import Predicate
from ..parser.default_policy_parser import _parse_default_policy_raw
from ..parser.domain_parser import _parse_domain_raw
from ..parser.problem_parser import _parse_problem_raw
from ..parser.schemas import (
    ParsedActionSchema,
    ParsedDefaultPolicy,
    ParsedDefaultPolicyRuleSchema,
    ParsedDomain,
    ParsedObservationRuleSchema,
    ParsedProblem,
)
from ..parser.sexpr import ParseError, SExpr


@dataclass
class _SymbolTable:
    types: dict[str, object]
    constants: dict[str, str]
    objects: dict[str, str]
    predicates: dict[str, list[tuple[str, str]]]
    observables: dict[str, list[tuple[str, str]]]
    actions: dict[str, list[tuple[str, str]]]


_BUILTIN_OPS = {
    "and",
    "or",
    "not",
    "imply",
    "forall",
    "all",
    "exists",
    "=",
    "when",
    "probabilistic",
    "increase",
    "decrease",
    "assign",
}


def lint_texts(
    domain_text: str,
    problem_text: str | None = None,
    default_policy_text: str | None = None,
    config: LinterConfig | None = None,
) -> LintResult:
    """Lint domain/problem/default-policy texts together."""

    cfg = config or LinterConfig()
    result = LintResult()

    parsed_domain: ParsedDomain | None = None
    parsed_problem: ParsedProblem | None = None
    parsed_default_policy: ParsedDefaultPolicy | None = None

    try:
        parsed_domain = _parse_domain_raw(domain_text)
    except ParseError as exc:
        result.add("error", "syntax.parse_error", str(exc), "domain", "domain")
        return result

    if problem_text is not None:
        try:
            parsed_problem = _parse_problem_raw(problem_text)
        except ParseError as exc:
            result.add("error", "syntax.parse_error", str(exc), "problem", "problem")
            return result

    if default_policy_text is not None:
        try:
            parsed_default_policy = _parse_default_policy_raw(default_policy_text)
        except ParseError as exc:
            result.add("error", "syntax.parse_error", str(exc), "default_policy", "default_policy")
            return result

    _lint_parsed(parsed_domain, parsed_problem, parsed_default_policy, result, cfg)
    if cfg.treat_warnings_as_errors and result.warnings:
        promoted = [
            diagnostic
            for diagnostic in result.diagnostics
            if diagnostic.severity == "warning"
        ]
        for diagnostic in promoted:
            result.add(
                "error",
                diagnostic.code,
                diagnostic.message,
                diagnostic.module,
                diagnostic.source,
                diagnostic.context_name,
            )
    return result


def lint_domain_text(
    domain_text: str,
    config: LinterConfig | None = None,
) -> LintResult:
    """Lint a domain text in isolation."""

    cfg = config or LinterConfig()
    result = LintResult()
    try:
        parsed_domain = _parse_domain_raw(domain_text)
    except ParseError as exc:
        result.add("error", "syntax.parse_error", str(exc), "domain", "domain")
        return result

    _lint_parsed(parsed_domain, None, None, result, cfg)
    if cfg.treat_warnings_as_errors and result.warnings:
        promoted = [
            diagnostic
            for diagnostic in result.diagnostics
            if diagnostic.severity == "warning"
        ]
        for diagnostic in promoted:
            result.add(
                "error",
                diagnostic.code,
                diagnostic.message,
                diagnostic.module,
                diagnostic.source,
                diagnostic.context_name,
            )
    return result


def lint_problem_text(
    problem_text: str,
    config: LinterConfig | None = None,
) -> LintResult:
    """Lint a problem text in isolation."""

    cfg = config or LinterConfig()
    result = LintResult()
    try:
        parsed_problem = _parse_problem_raw(problem_text)
    except ParseError as exc:
        result.add("error", "syntax.parse_error", str(exc), "problem", "problem")
        return result

    if cfg.enable_belief_checks:
        try:
            parsed_problem.init_belief.validate()
        except ValueError as exc:
            result.add(
                "error",
                "belief.invalid_structure",
                str(exc),
                "belief",
                "problem",
                ":init-belief",
            )
    return result


def lint_default_policy_text(
    default_policy_text: str,
    config: LinterConfig | None = None,
) -> LintResult:
    """Lint a default policy text in isolation."""

    cfg = config or LinterConfig()
    result = LintResult()
    try:
        parsed_default_policy = _parse_default_policy_raw(default_policy_text)
    except ParseError as exc:
        result.add(
            "error",
            "syntax.parse_error",
            str(exc),
            "default_policy",
            "default_policy",
        )
        return result

    if cfg.enable_duplicate_name_checks:
        _report_duplicates(
            [schema.rule.name for schema in parsed_default_policy.rules],
            "default_policy_rule",
            "default_policy",
            result,
        )
    return result


def _lint_parsed(
    parsed_domain: ParsedDomain,
    parsed_problem: ParsedProblem | None,
    parsed_default_policy: ParsedDefaultPolicy | None,
    result: LintResult,
    config: LinterConfig,
) -> None:
    symbols = _build_symbol_table(parsed_domain, parsed_problem)

    if config.enable_duplicate_name_checks:
        _check_duplicate_names(parsed_domain, parsed_default_policy, result)
    _check_declared_types(parsed_domain, parsed_problem, parsed_default_policy, result)

    for action_schema in parsed_domain.actions:
        _check_expr(
            action_schema.precondition,
            "action_precondition",
            "domain",
            f"action:{action_schema.action.name}",
            symbols,
            dict(action_schema.parameter_types),
            result,
            config,
            expected_atoms="predicate",
        )
        _check_expr(
            action_schema.effect,
            "action_effect",
            "domain",
            f"action:{action_schema.action.name}",
            symbols,
            dict(action_schema.parameter_types),
            result,
            config,
            expected_atoms="predicate",
        )

    for rule_schema in parsed_domain.observation_rules:
        scope = dict(rule_schema.parameter_types)
        _check_expr(
            rule_schema.condition,
            "observation_condition",
            "domain",
            f"observation:{rule_schema.rule.name}",
            symbols,
            scope,
            result,
            config,
            expected_atoms="predicate",
        )
        _check_expr(
            rule_schema.distribution_expr,
            "observation_distribution",
            "domain",
            f"observation:{rule_schema.rule.name}",
            symbols,
            scope,
            result,
            config,
            expected_atoms="observable",
        )

    if parsed_problem is not None:
        if (
            parsed_problem.domain_name is not None
            and parsed_problem.domain_name != parsed_domain.domain_name
        ):
            result.add(
                "error",
                "semantic.problem_domain_mismatch",
                (
                    f"Problem declares domain `{parsed_problem.domain_name}`, but parsed domain is "
                    f"`{parsed_domain.domain_name}`."
                ),
                "problem",
                "problem",
                parsed_problem.problem_name,
            )
        _check_problem(parsed_problem, symbols, result, config)
        _check_reward_metric_consistency(parsed_domain, parsed_problem, result)

    if parsed_default_policy is not None:
        for rule_schema in parsed_default_policy.rules:
            scope = dict(rule_schema.parameter_types)
            _check_expr(
                rule_schema.precondition,
                "default_policy_precondition",
                "default_policy",
                f"policy:{rule_schema.rule.name}",
                symbols,
                scope,
                result,
                config,
                expected_atoms="predicate",
            )
            _check_default_policy_action(rule_schema, parsed_domain, symbols, result, config)


def _build_symbol_table(
    parsed_domain: ParsedDomain,
    parsed_problem: ParsedProblem | None,
) -> _SymbolTable:
    return _SymbolTable(
        types=parsed_domain.types,
        constants=dict(parsed_domain.constants),
        objects=dict(parsed_problem.objects) if parsed_problem is not None else {},
        predicates={predicate.name: parsed_domain.predicate_parameter_types[predicate] for predicate in parsed_domain.predicates},
        observables={observable.name: parsed_domain.observable_parameter_types[observable] for observable in parsed_domain.observables},
        actions={schema.action.name: schema.parameter_types for schema in parsed_domain.actions},
    )


def _check_duplicate_names(
    parsed_domain: ParsedDomain,
    parsed_default_policy: ParsedDefaultPolicy | None,
    result: LintResult,
) -> None:
    _report_duplicates([predicate.name for predicate in parsed_domain.predicates], "predicate", "domain", result)
    _report_duplicates([observable.name for observable in parsed_domain.observables], "observable", "domain", result)
    _report_duplicates([schema.action.name for schema in parsed_domain.actions], "action", "domain", result)
    _report_duplicates(
        [schema.rule.name for schema in parsed_domain.observation_rules],
        "observation_rule",
        "domain",
        result,
    )
    if parsed_default_policy is not None:
        _report_duplicates(
            [schema.rule.name for schema in parsed_default_policy.rules],
            "default_policy_rule",
            "default_policy",
            result,
        )


def _check_declared_types(
    parsed_domain: ParsedDomain,
    parsed_problem: ParsedProblem | None,
    parsed_default_policy: ParsedDefaultPolicy | None,
    result: LintResult,
) -> None:
    declared_types = set(parsed_domain.types)

    for constant_name, type_name in parsed_domain.constants.items():
        if type_name not in declared_types:
            result.add(
                "error",
                "symbol.undefined_type",
                f"Constant `{constant_name}` uses undefined type `{type_name}`.",
                "constants",
                "domain",
                constant_name,
            )

    for predicate, typed_params in parsed_domain.predicate_parameter_types.items():
        for _param_name, type_name in typed_params:
            if type_name not in declared_types:
                result.add(
                    "error",
                    "symbol.undefined_type",
                    f"Predicate `{predicate.name}` uses undefined type `{type_name}`.",
                    "predicates",
                    "domain",
                    predicate.name,
                )

    for observable, typed_params in parsed_domain.observable_parameter_types.items():
        for _param_name, type_name in typed_params:
            if type_name not in declared_types:
                result.add(
                    "error",
                    "symbol.undefined_type",
                    f"Observable `{observable.name}` uses undefined type `{type_name}`.",
                    "observables",
                    "domain",
                    observable.name,
                )

    for action_schema in parsed_domain.actions:
        for _param_name, type_name in action_schema.parameter_types:
            if type_name not in declared_types:
                result.add(
                    "error",
                    "symbol.undefined_type",
                    f"Action `{action_schema.action.name}` uses undefined type `{type_name}`.",
                    "action",
                    "domain",
                    action_schema.action.name,
                )

    for function_name, typed_params in parsed_domain.functions.items():
        for _param_name, type_name in typed_params:
            if type_name not in declared_types:
                result.add(
                    "error",
                    "symbol.undefined_type",
                    f"Function `{function_name}` uses undefined type `{type_name}`.",
                    "functions",
                    "domain",
                    function_name,
                )

    for rule_schema in parsed_domain.observation_rules:
        for _param_name, type_name in rule_schema.parameter_types:
            if type_name not in declared_types:
                result.add(
                    "error",
                    "symbol.undefined_type",
                    f"Observation rule `{rule_schema.rule.name}` uses undefined type `{type_name}`.",
                    "observation",
                    "domain",
                    rule_schema.rule.name,
                )

    if parsed_problem is not None:
        for object_name, type_name in parsed_problem.objects.items():
            if type_name not in declared_types:
                result.add(
                    "error",
                    "symbol.undefined_type",
                    f"Object `{object_name}` uses undefined type `{type_name}`.",
                    "objects",
                    "problem",
                    object_name,
                )

    if parsed_default_policy is not None:
        for rule_schema in parsed_default_policy.rules:
            for _param_name, type_name in rule_schema.parameter_types:
                if type_name not in declared_types:
                    result.add(
                        "error",
                        "symbol.undefined_type",
                        f"Default policy rule `{rule_schema.rule.name}` uses undefined type `{type_name}`.",
                        "default_policy",
                        "default_policy",
                        rule_schema.rule.name,
                    )


def _report_duplicates(names: list[str], kind: str, source: str, result: LintResult) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            result.add(
                "error",
                "symbol.duplicate_name",
                f"Duplicate {kind} name `{name}`.",
                kind,
                source,
                name,
            )
        seen.add(name)


def _check_problem(
    parsed_problem: ParsedProblem,
    symbols: _SymbolTable,
    result: LintResult,
    config: LinterConfig,
) -> None:
    if parsed_problem.domain_name is None:
        result.add(
            "warning",
            "semantic.problem_missing_domain",
            "Problem file does not declare `:domain`.",
            "problem",
            "problem",
        )
    for predicate, _value in parsed_problem.init_state.items():
        _check_atom_symbol(
            predicate.name,
            predicate.params,
            expected="predicate",
            module="problem_init",
            source="problem",
            context_name=":init",
            symbols=symbols,
            scope={},
            result=result,
            config=config,
        )
    _check_expr(
        parsed_problem.goal,
        "goal",
        "problem",
        ":goal",
        symbols,
        {},
        result,
        config,
        expected_atoms="predicate",
    )
    if config.enable_belief_checks:
        _check_belief(parsed_problem, symbols, result, config)


def _check_reward_metric_consistency(
    parsed_domain: ParsedDomain,
    parsed_problem: ParsedProblem,
    result: LintResult,
) -> None:
    declared_functions = set(parsed_domain.functions)
    updated_functions: set[str] = set()
    for action_schema in parsed_domain.actions:
        updated_functions |= _collect_numeric_targets(action_schema.effect)

    for function_name in updated_functions:
        if function_name not in declared_functions:
            result.add(
                "error",
                "semantic.reward_function_undeclared",
                f"Action effects update function `{function_name}`, but it is not declared in `:functions`.",
                "reward_metric",
                "domain",
                function_name,
            )

    metric_target = parsed_problem.metric_target_function
    if metric_target is not None and metric_target not in declared_functions:
        result.add(
            "error",
            "semantic.metric_function_undeclared",
            f"Problem metric references function `{metric_target}`, but it is not declared in the domain.",
            "reward_metric",
            "problem",
            metric_target,
        )

    if len(updated_functions) > 1:
        result.add(
            "error",
            "semantic.multiple_reward_functions",
            (
                "Action effects update multiple numeric functions: "
                + ", ".join(sorted(updated_functions))
                + "."
            ),
            "reward_metric",
            "domain",
        )

    if metric_target is not None and len(updated_functions) == 1:
        updated_name = next(iter(updated_functions))
        if metric_target != updated_name:
            result.add(
                "error",
                "semantic.metric_reward_mismatch",
                (
                    f"Problem metric uses `{metric_target}`, but action effects update "
                    f"`{updated_name}`."
                ),
                "reward_metric",
                "problem",
                metric_target,
            )

    if metric_target is None and updated_functions:
        result.add(
            "warning",
            "semantic.metric_missing",
            "Action effects update numeric reward-like functions, but the problem does not declare a metric.",
            "reward_metric",
            "problem",
        )


def _check_belief(
    parsed_problem: ParsedProblem,
    symbols: _SymbolTable,
    result: LintResult,
    config: LinterConfig,
) -> None:
    try:
        parsed_problem.init_belief.validate()
    except ValueError as exc:
        result.add(
            "error",
            "belief.invalid_structure",
            str(exc),
            "belief",
            "problem",
            ":init-belief",
        )
    for predicate in parsed_problem.init_belief.known_true + parsed_problem.init_belief.known_false:
        _check_atom_symbol(
            predicate.name,
            predicate.params,
            expected="predicate",
            module="problem_init_belief",
            source="problem",
            context_name=":init-belief",
            symbols=symbols,
            scope={},
            result=result,
            config=config,
        )
    for factor in parsed_problem.init_belief.factors:
        for predicate in factor.scope:
            _check_atom_symbol(
                predicate.name,
                predicate.params,
                expected="predicate",
                module="problem_init_belief",
                source="problem",
                context_name=factor.name,
                symbols=symbols,
                scope={},
                result=result,
                config=config,
            )


def _check_default_policy_action(
    rule_schema: ParsedDefaultPolicyRuleSchema,
    parsed_domain: ParsedDomain,
    symbols: _SymbolTable,
    result: LintResult,
    config: LinterConfig,
) -> None:
    action_expr = rule_schema.action_expr
    if action_expr is None:
        result.add(
            "error",
            "symbol.missing_policy_action",
            f"Default policy rule `{rule_schema.rule.name}` is missing `:action`.",
            "default_policy_action",
            "default_policy",
            rule_schema.rule.name,
        )
        return
    if not isinstance(action_expr, list) or not action_expr or not isinstance(action_expr[0], str):
        result.add(
            "error",
            "syntax.invalid_policy_action",
            f"Default policy rule `{rule_schema.rule.name}` has malformed `:action`.",
            "default_policy_action",
            "default_policy",
            rule_schema.rule.name,
        )
        return
    _check_action_invocation(
        action_expr[0],
        [token for token in action_expr[1:] if isinstance(token, str)],
        "default_policy_action",
        "default_policy",
        rule_schema.rule.name,
        symbols,
        dict(rule_schema.parameter_types),
        result,
        config,
    )
    action_schema = _find_action_schema(parsed_domain, action_expr[0])
    if action_schema is not None:
        _check_default_policy_action_legality(rule_schema, action_schema, result)


def _check_expr(
    expr: SExpr | None,
    context: str,
    source: str,
    context_name: str,
    symbols: _SymbolTable,
    scope: dict[str, str],
    result: LintResult,
    config: LinterConfig,
    *,
    expected_atoms: str,
) -> None:
    if expr is None:
        return
    _walk_expr(expr, context, source, context_name, symbols, scope, result, config, expected_atoms)


def _walk_expr(
    expr: SExpr,
    context: str,
    source: str,
    context_name: str,
    symbols: _SymbolTable,
    scope: dict[str, str],
    result: LintResult,
    config: LinterConfig,
    expected_atoms: str,
) -> None:
    if isinstance(expr, str):
        return
    if not expr:
        return
    head = expr[0]
    if not isinstance(head, str):
        return

    if head in _BUILTIN_OPS:
        allowed = config.allowed_ops_by_context.get(context)
        if allowed is not None and head not in allowed:
            result.add(
                "error",
                "operator.not_allowed",
                f"Operator `{head}` is not allowed in context `{context}`.",
                context,
                source,
                context_name,
            )

        if head in {"forall", "all", "exists"}:
            if len(expr) < 3 or not isinstance(expr[1], list):
                result.add(
                    "error",
                    "syntax.invalid_quantifier",
                    f"Malformed quantifier `{head}`.",
                    context,
                    source,
                    context_name,
                )
                return
            quant_scope = dict(scope)
            for var_name, type_name in _parse_typed_bindings(expr[1], result, context, source, context_name):
                quant_scope[var_name] = type_name
            for subexpr in expr[2:]:
                _walk_expr(
                    subexpr,
                    context,
                    source,
                    context_name,
                    symbols,
                    quant_scope,
                    result,
                    config,
                    expected_atoms,
                )
            return

        if head in {"increase", "decrease", "assign"}:
            if len(expr) >= 2 and isinstance(expr[1], list) and expr[1]:
                target = expr[1][0]
                if not isinstance(target, str):
                    result.add(
                        "error",
                        "syntax.invalid_numeric_target",
                        f"Malformed numeric target in `{head}`.",
                        context,
                        source,
                        context_name,
                    )
            for subexpr in expr[2:]:
                _walk_expr(
                    subexpr,
                    context,
                    source,
                    context_name,
                    symbols,
                    scope,
                    result,
                    config,
                    expected_atoms,
                )
            return

        for subexpr in expr[1:]:
            _walk_expr(
                subexpr,
                context,
                source,
                context_name,
                symbols,
                scope,
                result,
                config,
                expected_atoms,
            )
        return

    args = [token for token in expr[1:] if isinstance(token, str)]
    if context == "default_policy_action":
        _check_action_invocation(
            head,
            args,
            context,
            source,
            context_name,
            symbols,
            scope,
            result,
            config,
        )
        return
    _check_atom_symbol(
        head,
        args,
        expected=expected_atoms,
        module=context,
        source=source,
        context_name=context_name,
        symbols=symbols,
        scope=scope,
        result=result,
        config=config,
    )


def _parse_typed_bindings(
    tokens: list[SExpr],
    result: LintResult,
    module: str,
    source: str,
    context_name: str,
) -> list[tuple[str, str]]:
    raw: list[str] = []
    for token in tokens:
        if not isinstance(token, str):
            result.add(
                "error",
                "syntax.invalid_typed_binding",
                "Quantifier bindings must be flat symbols.",
                module,
                source,
                context_name,
            )
            return []
        raw.append(token)
    bindings: list[tuple[str, str]] = []
    pending: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if token == "-":
            if not pending or index + 1 >= len(raw):
                result.add(
                    "error",
                    "syntax.invalid_typed_binding",
                    "Malformed typed binding list.",
                    module,
                    source,
                    context_name,
                )
                return bindings
            type_name = raw[index + 1]
            for name in pending:
                bindings.append((name, type_name))
            pending.clear()
            index += 2
            continue
        pending.append(token)
        index += 1
    for name in pending:
        bindings.append((name, "object"))
    return bindings


def _check_atom_symbol(
    name: str,
    args: list[str],
    *,
    expected: str,
    module: str,
    source: str,
    context_name: str,
    symbols: _SymbolTable,
    scope: dict[str, str],
    result: LintResult,
    config: LinterConfig,
) -> None:
    table = symbols.predicates if expected == "predicate" else symbols.observables
    kind = "predicate" if expected == "predicate" else "observable"
    signature = table.get(name)
    if signature is None:
        result.add(
            "error",
            f"symbol.undefined_{kind}",
            f"Undefined {kind} `{name}`.",
            module,
            source,
            context_name,
        )
        return
    if len(signature) != len(args):
        result.add(
            "error",
            f"symbol.{kind}_arity_mismatch",
            f"{kind.capitalize()} `{name}` expects {len(signature)} arguments, got {len(args)}.",
            module,
            source,
            context_name,
        )
        return
    if not config.enable_type_checks:
        return
    for index, (arg, (_param_name, expected_type)) in enumerate(zip(args, signature), start=1):
        actual_type = _resolve_arg_type(arg, scope, symbols)
        if actual_type is None:
            if arg.startswith("?"):
                result.add(
                    "error",
                    "symbol.undefined_variable",
                    f"Undefined variable `{arg}`.",
                    module,
                    source,
                    context_name,
                )
            else:
                result.add(
                    "error",
                    "symbol.undefined_object",
                    f"Undefined object or constant `{arg}`.",
                    module,
                    source,
                    context_name,
                )
            continue
        if not _is_type_compatible(actual_type, expected_type, symbols):
            result.add(
                "error",
                f"symbol.{kind}_argument_type_mismatch",
                (
                    f"{kind.capitalize()} `{name}` expects argument {index} to have type "
                    f"`{expected_type}`, but `{arg}` has type `{actual_type}`."
                ),
                module,
                source,
                context_name,
            )


def _check_action_invocation(
    action_name: str,
    args: list[str],
    module: str,
    source: str,
    context_name: str,
    symbols: _SymbolTable,
    scope: dict[str, str],
    result: LintResult,
    config: LinterConfig,
) -> None:
    signature = symbols.actions.get(action_name)
    if signature is None:
        result.add(
            "error",
            "symbol.undefined_action",
            f"Undefined action `{action_name}`.",
            module,
            source,
            context_name,
        )
        return
    if len(signature) != len(args):
        result.add(
            "error",
            "symbol.action_arity_mismatch",
            f"Action `{action_name}` expects {len(signature)} arguments, got {len(args)}.",
            module,
            source,
            context_name,
        )
        return
    if not config.enable_type_checks:
        return
    for index, (arg, (_param_name, expected_type)) in enumerate(zip(args, signature), start=1):
        actual_type = _resolve_arg_type(arg, scope, symbols)
        if actual_type is None:
            if arg.startswith("?"):
                result.add(
                    "error",
                    "symbol.undefined_variable",
                    f"Undefined variable `{arg}`.",
                    module,
                    source,
                    context_name,
                )
            else:
                result.add(
                    "error",
                    "symbol.undefined_object",
                    f"Undefined object or constant `{arg}`.",
                    module,
                    source,
                    context_name,
                )
            continue
        if not _is_type_compatible(actual_type, expected_type, symbols):
            result.add(
                "error",
                "symbol.action_argument_type_mismatch",
                (
                    f"Action `{action_name}` expects argument {index} to have type "
                    f"`{expected_type}`, but `{arg}` has type `{actual_type}`."
                ),
                module,
                source,
                context_name,
            )


def _find_action_schema(parsed_domain: ParsedDomain, action_name: str) -> ParsedActionSchema | None:
    for action_schema in parsed_domain.actions:
        if action_schema.action.name == action_name:
            return action_schema
    return None


def _check_default_policy_action_legality(
    rule_schema: ParsedDefaultPolicyRuleSchema,
    action_schema: ParsedActionSchema,
    result: LintResult,
) -> None:
    if rule_schema.action_expr is None or action_schema.precondition is None:
        return
    if not isinstance(rule_schema.action_expr, list):
        return
    args = [token for token in rule_schema.action_expr[1:] if isinstance(token, str)]
    if len(args) != len(action_schema.parameter_types):
        return

    substitution = {
        param_name: arg
        for (param_name, _type_name), arg in zip(action_schema.parameter_types, args)
    }
    substituted_action_pre = _substitute_expr(action_schema.precondition, substitution)

    required_true, required_false = _extract_conjunctive_literals(substituted_action_pre)
    policy_true, policy_false = _extract_conjunctive_literals(rule_schema.precondition)

    conflicts: list[str] = []
    for literal in sorted(required_true & policy_false):
        conflicts.append(
            f"action precondition requires `{literal}`, but policy precondition explicitly negates it"
        )
    for literal in sorted(required_false & policy_true):
        conflicts.append(
            f"action precondition requires `not {literal}`, but policy precondition explicitly asserts it"
        )
    if conflicts:
        result.add(
            "error",
            "semantic.default_policy_action_precondition_conflict",
            (
                f"Default policy rule `{rule_schema.rule.name}` selects action "
                f"`{action_schema.action.name}` under a conflicting precondition: "
                + "; ".join(conflicts)
                + "."
            ),
            "default_policy_action",
            "default_policy",
            rule_schema.rule.name,
        )


def _substitute_expr(expr: SExpr | None, substitution: dict[str, str]) -> SExpr | None:
    if expr is None:
        return None
    if isinstance(expr, str):
        return substitution.get(expr, expr)
    return [_substitute_expr(subexpr, substitution) for subexpr in expr]


def _extract_conjunctive_literals(expr: SExpr | None) -> tuple[set[str], set[str]]:
    if expr is None:
        return set(), set()
    if isinstance(expr, str):
        return set(), set()
    if not expr:
        return set(), set()
    head = expr[0]
    if not isinstance(head, str):
        return set(), set()

    if head == "and":
        positive: set[str] = set()
        negative: set[str] = set()
        for subexpr in expr[1:]:
            sub_positive, sub_negative = _extract_conjunctive_literals(subexpr)
            positive |= sub_positive
            negative |= sub_negative
        return positive, negative

    if head == "not":
        if len(expr) == 2:
            literal = _literal_string(expr[1])
            if literal is not None:
                return set(), {literal}
        return set(), set()

    literal = _literal_string(expr)
    if literal is not None:
        return {literal}, set()
    return set(), set()


def _literal_string(expr: SExpr) -> str | None:
    if isinstance(expr, str):
        return None
    if not expr:
        return None
    head = expr[0]
    if not isinstance(head, str):
        return None
    if head == "=":
        args = [token for token in expr[1:] if isinstance(token, str)]
        if len(args) == 2:
            return f"(= {args[0]} {args[1]})"
        return None
    if head in _BUILTIN_OPS:
        return None
    args = [token for token in expr[1:] if isinstance(token, str)]
    if args:
        return f"({head} {' '.join(args)})"
    return f"({head})"


def _resolve_arg_type(arg: str, scope: dict[str, str], symbols: _SymbolTable) -> str | None:
    if arg.startswith("?"):
        return scope.get(arg)
    if arg in symbols.objects:
        return symbols.objects[arg]
    if arg in symbols.constants:
        return symbols.constants[arg]
    return None


def _is_type_compatible(actual_type: str, expected_type: str, symbols: _SymbolTable) -> bool:
    if actual_type == expected_type:
        return True
    node = symbols.types.get(actual_type)
    while node is not None and getattr(node, "parent", None) is not None:
        node = node.parent
        if node is not None and node.name == expected_type:
            return True
    return expected_type == "object"


def _collect_numeric_targets(expr: SExpr | None) -> set[str]:
    if expr is None:
        return set()
    if isinstance(expr, str):
        return set()
    if not expr:
        return set()
    head = expr[0]
    if not isinstance(head, str):
        return set()
    targets: set[str] = set()
    if head in {"increase", "decrease", "assign"}:
        if len(expr) >= 2 and isinstance(expr[1], list) and expr[1]:
            target_head = expr[1][0]
            if isinstance(target_head, str):
                targets.add(target_head)
        for subexpr in expr[2:]:
            targets |= _collect_numeric_targets(subexpr)
        return targets
    for subexpr in expr[1:]:
        targets |= _collect_numeric_targets(subexpr)
    return targets
