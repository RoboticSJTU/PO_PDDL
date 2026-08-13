from __future__ import annotations

import json
import re
from dataclasses import dataclass

from po_pddl.domain_generation.infrastructure.fact_utils import parse_symbolic_literal
from po_pddl.domain_generation.stages.manipulation_domain_learning.models import ActionSchema

from .models import ActionPreconditionCandidateBundle
from .shared import extract_json_object, load_prompt, make_client, safe_chat

_IDENTITY_TOKEN_STOPWORDS = {
    "a",
    "an",
    "at",
    "in",
    "is",
    "item",
    "object",
    "of",
    "on",
    "state",
    "the",
    "to",
}

_DIRECTIONAL_VARIANT_TOKEN_GROUPS = (
    frozenset({"left", "right"}),
    frozenset({"front", "back"}),
    frozenset({"upper", "lower"}),
    frozenset({"up", "down"}),
    frozenset({"inside", "outside"}),
    frozenset({"clockwise", "counterclockwise"}),
)


def _directional_family_key(action_name: str) -> tuple[str, ...] | None:
    tokens = tuple(filter(None, re.split(r"[^a-z0-9]+", action_name.lower())))
    replaced: list[str] = []
    found_variant = False
    for token in tokens:
        group_index = next(
            (index for index, group in enumerate(_DIRECTIONAL_VARIANT_TOKEN_GROUPS) if token in group),
            None,
        )
        if group_index is None:
            replaced.append(token)
            continue
        found_variant = True
        replaced.append(f"<direction-{group_index}>")
    return tuple(replaced) if found_variant else None


def _literal_variant_tokens(literal: str) -> set[str]:
    _negated, predicate_name, _arguments = parse_symbolic_literal(literal)
    return {
        token
        for token in predicate_name.lower().split("_")
        if any(token in group for group in _DIRECTIONAL_VARIANT_TOKEN_GROUPS)
    }


def _normalized_literal_key(literal: str) -> tuple[bool, str, tuple[str, ...]]:
    negated, predicate_name, arguments = parse_symbolic_literal(literal)
    normalized_tokens = [
        next(
            (f"<direction-{index}>" for index, group in enumerate(_DIRECTIONAL_VARIANT_TOKEN_GROUPS) if token in group),
            token,
        )
        for token in predicate_name.lower().split("_")
    ]
    return negated, "_".join(normalized_tokens), tuple(arguments)


def prune_directional_variant_contradictions(
    *,
    selected_preconditions_by_action: dict[str, list[str]],
    universally_supported_by_action: dict[str, list[str]],
    parameter_types_by_action: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, list[str]]:
    """Drop non-directional conditions contradicted by an equivalent directional variant."""
    families: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = {}
    for action_name in selected_preconditions_by_action:
        family_key = _directional_family_key(action_name)
        if family_key is not None:
            parameter_types = (parameter_types_by_action or {}).get(action_name, ())
            families.setdefault((family_key, parameter_types), []).append(action_name)

    universal_keys = {
        action_name: {_normalized_literal_key(literal) for literal in literals}
        for action_name, literals in universally_supported_by_action.items()
    }
    pruned = {action_name: list(literals) for action_name, literals in selected_preconditions_by_action.items()}
    for sibling_names in families.values():
        if len(sibling_names) < 2:
            continue
        for action_name in sibling_names:
            kept: list[str] = []
            for literal in pruned[action_name]:
                if _literal_variant_tokens(literal):
                    kept.append(literal)
                    continue
                negated, predicate_name, arguments = _normalized_literal_key(literal)
                complement = (not negated, predicate_name, arguments)
                contradicted = any(
                    complement in universal_keys.get(sibling_name, set())
                    for sibling_name in sibling_names
                    if sibling_name != action_name
                )
                if not contradicted:
                    kept.append(literal)
            pruned[action_name] = kept
    return pruned


def retain_identity_anchored_preconditions(
    *,
    action_name: str,
    universally_supported_literals: list[str],
    selected_literals: list[str],
    effect_add_literals: list[str] | None = None,
) -> list[str]:
    """Keep supported literals that encode otherwise-unrepresented action qualifiers."""
    action_tokens = set(filter(None, re.split(r"[^a-z0-9]+", action_name.lower())))
    action_verb = next(iter(re.split(r"[^a-z0-9]+", action_name.lower())), "")
    effect_add_keys = {
        (predicate_name, tuple(arguments))
        for literal in (effect_add_literals or [])
        for _negated, predicate_name, arguments in [parse_symbolic_literal(literal)]
    }

    def is_redundant_noop_check(literal: str) -> bool:
        negated, predicate_name, arguments = parse_symbolic_literal(literal)
        predicate_tokens = set(predicate_name.lower().split("_"))
        return negated and (predicate_name, tuple(arguments)) in effect_add_keys and action_verb not in predicate_tokens

    def is_dependent_literal(literal: str) -> bool:
        _negated, _predicate_name, arguments = parse_symbolic_literal(literal)
        return any(argument.startswith("?dep") for argument in arguments)

    def is_supported_dependency_shape(literal: str) -> bool:
        negated, _predicate_name, arguments = parse_symbolic_literal(literal)
        dependency_arguments = {argument for argument in arguments if argument.startswith("?dep")}
        return (
            negated
            and bool(dependency_arguments)
            and any(argument.startswith("?arg") for argument in arguments)
        )

    held_arguments = {
        arguments[0]
        for literal in selected_literals
        for negated, predicate_name, arguments in [parse_symbolic_literal(literal)]
        if not negated and predicate_name == "gripper_holding" and len(arguments) == 1
    }

    def is_redundant_held_item_location_absence(literal: str) -> bool:
        """Holding an item already establishes that its previous location was cleared."""
        negated, _predicate_name, arguments = parse_symbolic_literal(literal)
        action_arguments = [argument for argument in arguments if argument.startswith("?arg")]
        return (
            negated
            and len(action_arguments) == 1
            and action_arguments[0] in held_arguments
            and arguments
            and arguments[0] == action_arguments[0]
        )

    # Dependency candidates are eligible only when a demonstrated earlier
    # transition removed a matching fact. Requiring an action-argument anchor
    # keeps that causal evidence local instead of turning scene-wide absences
    # into global gates.
    causal_dependency_literals = [
        literal
        for literal in universally_supported_literals
        if is_supported_dependency_shape(literal) and not is_redundant_held_item_location_absence(literal)
    ]
    mandatory_resource_literals = []
    for literal in universally_supported_literals:
        negated, _predicate_name, arguments = parse_symbolic_literal(literal)
        if not negated and not arguments:
            mandatory_resource_literals.append(literal)

    selected_literals = [
        literal
        for literal in dict.fromkeys(
            [*selected_literals, *mandatory_resource_literals, *causal_dependency_literals]
        )
        if not is_redundant_noop_check(literal)
        and (not is_dependent_literal(literal) or is_supported_dependency_shape(literal))
    ]
    parsed_selected = [(literal, *parse_symbolic_literal(literal)) for literal in selected_literals]
    selected_literals = [
        literal
        for literal, negated, predicate_name, arguments in parsed_selected
        if negated
        or not any(
            not other_negated
            and {token for token in predicate_name.lower().split("_") if token not in _IDENTITY_TOKEN_STOPWORDS}
            < {token for token in other_predicate.lower().split("_") if token not in _IDENTITY_TOKEN_STOPWORDS}
            and set(arguments).issubset(other_arguments)
            for other_literal, other_negated, other_predicate, other_arguments in parsed_selected
            if other_literal != literal
        )
    ]
    represented_tokens: set[str] = set()
    for literal in selected_literals:
        _negated, predicate_name, _arguments = parse_symbolic_literal(literal)
        represented_tokens.update(
            token for token in predicate_name.lower().split("_") if token not in _IDENTITY_TOKEN_STOPWORDS
        )

    retained = list(selected_literals)
    for literal in universally_supported_literals:
        if is_redundant_noop_check(literal) or is_dependent_literal(literal):
            continue
        _negated, predicate_name, _arguments = parse_symbolic_literal(literal)
        identity_tokens = {
            token for token in predicate_name.lower().split("_") if token not in _IDENTITY_TOKEN_STOPWORDS
        }
        if not identity_tokens or not identity_tokens.issubset(action_tokens):
            continue
        if identity_tokens.issubset(represented_tokens):
            continue
        retained.append(literal)
        represented_tokens.update(identity_tokens)
    return sorted(dict.fromkeys(retained))


@dataclass
class LLMPreconditionSelectionModule:
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4000
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("precondition_selection_prompt.md")

    def select_preconditions(
        self,
        *,
        domain_text: str,
        action_schema: ActionSchema,
        bundle: ActionPreconditionCandidateBundle,
    ) -> tuple[list[str], str]:
        universally_supported = [
            item
            for item in bundle.candidate_stats
            if item.eligible_example_count > 0 and item.occurrence_count == item.eligible_example_count
        ]
        payload = {
            "domain_text": domain_text,
            "action_schema": action_schema.to_dict(),
            "candidate_bundle": {
                "action_name": bundle.action_name,
                "example_count": bundle.example_count,
                "candidate_stats": [item.to_dict() for item in universally_supported],
                "examples": [item.to_dict() for item in bundle.examples[:12]],
            },
        }
        reply = safe_chat(
            self._client,
            self._prompt,
            json.dumps(payload, ensure_ascii=False, indent=2),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        raw_selected = data.get("selected_precondition_literals", [])
        if not isinstance(raw_selected, list):
            raise ValueError("Precondition selection response must contain selected_precondition_literals as a list.")
        allowed_literals = {item.literal for item in universally_supported}
        selected: list[str] = []
        for item in raw_selected:
            literal = str(item).strip()
            if not literal:
                continue
            if literal not in allowed_literals:
                raise ValueError(
                    f"Precondition selector proposed literal {literal!r} for `{bundle.action_name}` "
                    "that is not present in the candidate set."
                )
            selected.append(literal)
        selected = retain_identity_anchored_preconditions(
            action_name=bundle.action_name,
            universally_supported_literals=sorted(allowed_literals),
            selected_literals=selected,
            effect_add_literals=[
                literal
                for branch in action_schema.effect_branches
                for literal in (list(branch.delta_add) + list(branch.fixed_delta_add) + list(branch.residual_delta_add))
            ],
        )
        return selected, str(data.get("selection_summary", "")).strip()
