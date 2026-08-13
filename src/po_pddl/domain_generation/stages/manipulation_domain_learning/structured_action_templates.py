from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from po_pddl.domain_generation.infrastructure.payload_utils import normalize_optional_text, validate_snake_case

_TEMPLATE_PLACEHOLDER_PATTERN = re.compile(r"{([^}]+)}")
_ACTION_NAME_STOPWORDS = frozenset({"a", "an", "the"})
_EMBEDDED_RELATION_PATTERN = re.compile(
    r"\b(?:"
    r"in\s+front\s+of|on\s+top\s+of|to\s+the\s+(?:left|right)\s+of|"
    r"in\s+the|on\s+the|at\s+the|inside|within|under(?:neath)?|below|above|behind|beside|between|"
    r"next\s+to|near|far\s+from|into|onto|from"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InducedActionTemplateArtifact:
    template_id: str
    template_text: str
    canonical_action_name: str
    action_category: str | None
    parameter_roles: list[str]
    parameter_placeholders: list[str]
    success_effect_bucket: str
    failure_effect_bucket: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedInducedTemplateAction:
    template_id: str
    template_text: str
    canonical_action_name: str
    action_category: str | None
    action_arguments: list[str]
    object_mentions: list[str]
    parameter_roles: list[str]
    parameter_placeholders: list[str]
    placeholder_values: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_template_placeholders(template_text: str) -> list[str]:
    return [
        validate_snake_case(match.group(1), field_name="template_placeholder")
        for match in _TEMPLATE_PLACEHOLDER_PATTERN.finditer(template_text)
    ]


def canonical_action_name_from_template_text(template_text: str) -> str:
    """Build a stable action name from fixed template semantics and entity slots."""
    with_entity_slots = _TEMPLATE_PLACEHOLDER_PATTERN.sub(" object ", template_text.lower())
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", with_entity_slots)
        if token not in _ACTION_NAME_STOPWORDS
    ]
    if not tokens:
        raise ValueError(f"Cannot derive an action name from template_text={template_text!r}")
    return validate_snake_case("_".join(tokens), field_name="canonical_action_name")


def induced_template_from_dict(
    data: dict[str, Any],
    *,
    require_action_category: bool = True,
) -> InducedActionTemplateArtifact:
    validate_snake_case(str(data.get("template_id") or "").strip(), field_name="template_id")
    validate_snake_case(
        str(data.get("canonical_action_name") or "").strip(),
        field_name="canonical_action_name",
    )
    action_category_raw = normalize_optional_text(data.get("action_category"))
    action_category = action_category_raw.lower() if action_category_raw else None
    if action_category is None and require_action_category:
        raise ValueError("action_category must not be empty")
    if action_category is not None and action_category not in {"manipulation", "active_observation"}:
        raise ValueError(f"Unsupported action_category in induced template: {action_category!r}")
    template_text = str(data.get("template_text") or "").strip()
    if not template_text:
        raise ValueError("template_text must not be empty")
    canonical_action_name = canonical_action_name_from_template_text(template_text)
    template_id = canonical_action_name
    inferred_parameter_placeholders = _extract_template_placeholders(template_text)
    provided_parameter_roles = [
        validate_snake_case(str(item).strip(), field_name="parameter_roles")
        for item in data.get("parameter_roles", [])
        if str(item).strip()
    ]
    parameter_placeholders = [
        validate_snake_case(str(item).strip(), field_name="parameter_placeholders")
        for item in data.get("parameter_placeholders", [])
        if str(item).strip()
    ]
    if not parameter_placeholders:
        parameter_placeholders = list(inferred_parameter_placeholders)
    if inferred_parameter_placeholders and parameter_placeholders != inferred_parameter_placeholders:
        raise ValueError(
            "parameter_placeholders must match template_text placeholder order, "
            f"got {parameter_placeholders} vs {inferred_parameter_placeholders}"
        )
    parameter_roles = list(provided_parameter_roles)
    if not parameter_roles and parameter_placeholders:
        parameter_roles = ["object"] * len(parameter_placeholders)
    if len(parameter_roles) != len(parameter_placeholders):
        raise ValueError(
            "parameter_roles and parameter_placeholders must have the same length, "
            f"got {len(parameter_roles)} and {len(parameter_placeholders)}"
        )
    return InducedActionTemplateArtifact(
        template_id=template_id,
        template_text=template_text,
        canonical_action_name=canonical_action_name,
        action_category=action_category,
        parameter_roles=parameter_roles,
        parameter_placeholders=parameter_placeholders,
        success_effect_bucket=f"{canonical_action_name}_success",
        failure_effect_bucket=f"{canonical_action_name}_failure",
    )


def compile_template_regex(template_text: str) -> re.Pattern[str]:
    pieces: list[str] = []
    cursor = 0
    for match in re.finditer(r"{([^}]+)}", template_text):
        start, end = match.span()
        placeholder = validate_snake_case(match.group(1), field_name="template_placeholder")
        pieces.append(re.escape(template_text[cursor:start]))
        pieces.append(f"(?P<{placeholder}>.+?)")
        cursor = end
    pieces.append(re.escape(template_text[cursor:]))
    return re.compile("^" + "".join(pieces) + "$", re.IGNORECASE)


def normalize_argument_value(text: str) -> str:
    entity_text = re.sub(r"^(?:a|an|the)\s+", "", text.strip(), flags=re.IGNORECASE)
    normalized = re.sub(r"[^a-z0-9]+", "_", entity_text.lower()).strip("_")
    if not normalized:
        raise ValueError(f"Failed to normalize template argument value from {text!r}")
    return normalized


def is_entity_argument_span(text: str) -> bool:
    """Return false when a capture contains a relation to another entity."""
    normalized = normalize_optional_text(text)
    if normalized is None:
        return False
    return _EMBEDDED_RELATION_PATTERN.search(normalized) is None


def parse_action_text_with_template(
    action_text: str,
    template: InducedActionTemplateArtifact,
) -> ParsedInducedTemplateAction | None:
    normalized_text = normalize_optional_text(action_text)
    if normalized_text is None:
        return None
    match = compile_template_regex(template.template_text).match(normalized_text)
    if match is None:
        return None
    placeholder_values = {
        key: str(value).strip() for key, value in match.groupdict().items() if isinstance(value, str) and value.strip()
    }
    if any(
        not is_entity_argument_span(placeholder_values.get(placeholder, ""))
        for placeholder in template.parameter_placeholders
    ):
        return None
    action_arguments = [
        normalize_argument_value(placeholder_values[placeholder]) for placeholder in template.parameter_placeholders
    ]
    return ParsedInducedTemplateAction(
        template_id=template.template_id,
        template_text=template.template_text,
        canonical_action_name=template.canonical_action_name,
        action_category=template.action_category,
        action_arguments=action_arguments,
        object_mentions=list(action_arguments),
        parameter_roles=list(template.parameter_roles),
        parameter_placeholders=list(template.parameter_placeholders),
        placeholder_values=placeholder_values,
    )


def render_action_text_from_template(
    template_text: str,
    placeholder_values: dict[str, str],
) -> str:
    rendered = template_text
    for placeholder, value in placeholder_values.items():
        rendered = rendered.replace("{" + placeholder + "}", str(value).strip())
    return normalize_optional_text(rendered) or rendered.strip()


_DIRECTION_MARKERS: tuple[str, ...] = ("left", "right", "front", "back", "near", "far")


def _extract_direction_markers(text: str) -> set[str]:
    lowered = text.lower()
    markers: set[str] = set()
    for marker in _DIRECTION_MARKERS:
        if re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", lowered):
            markers.add(marker)
    return markers


def _fixed_template_text(template_text: str) -> str:
    return re.sub(r"{[^}]+}", " ", template_text)


def _template_match_score(
    action_text: str,
    template: InducedActionTemplateArtifact,
    parsed: ParsedInducedTemplateAction,
) -> tuple[int, int, int]:
    del parsed
    action_markers = _extract_direction_markers(action_text)
    template_markers = _extract_direction_markers(" ".join([template.template_text, template.canonical_action_name]))
    direction_overlap = len(action_markers & template_markers)
    direction_mismatch = len(template_markers - action_markers)
    fixed_text = _fixed_template_text(template.template_text)
    fixed_specificity = len(re.sub(r"\s+", "", fixed_text))
    return (direction_overlap, -direction_mismatch, fixed_specificity)


def _type_signature_key(parameter_roles: list[str]) -> str:
    return "|".join(
        validate_snake_case(str(role).strip() or "object", field_name="parameter_roles") for role in parameter_roles
    )


def match_action_text_to_template(
    action_text: str,
    templates: list[InducedActionTemplateArtifact],
    *,
    object_type_map: dict[str, str] | None = None,
) -> ParsedInducedTemplateAction:
    matches: list[tuple[ParsedInducedTemplateAction, InducedActionTemplateArtifact, tuple[int, int, int]]] = []
    for template in templates:
        parsed = parse_action_text_with_template(action_text, template)
        if parsed is None:
            continue
        matches.append((parsed, template, _template_match_score(action_text, template, parsed)))
    if not matches:
        raise ValueError(f"No induced action template matched action_text={action_text!r}")
    if len(matches) == 1:
        return matches[0][0]
    matches.sort(key=lambda item: item[2], reverse=True)
    best_score = matches[0][2]
    best_matches = [item for item in matches if item[2] == best_score]
    if len(best_matches) > 1 and object_type_map:
        typed_matches = [
            item
            for item in best_matches
            if [object_type_map.get(argument, "object") for argument in item[0].action_arguments]
            == list(item[0].parameter_roles)
        ]
        if len(typed_matches) == 1:
            return typed_matches[0][0]
        if len(typed_matches) > 1:
            keyed_matches = {_type_signature_key(list(item[0].parameter_roles)): item for item in typed_matches}
            if len(keyed_matches) == 1:
                return next(iter(keyed_matches.values()))[0]
    if len(best_matches) > 1:
        raise ValueError(
            f"Multiple induced action templates matched action_text={action_text!r}: "
            f"{[item[0].canonical_action_name for item in best_matches]}"
        )
    return best_matches[0][0]


def build_action_name_map(
    templates: list[InducedActionTemplateArtifact],
) -> dict[str, dict[str, Any]]:
    actions_by_name = {
        template.canonical_action_name: {
            "action_name": template.canonical_action_name,
            "action_category": template.action_category,
            "template_id": template.template_id,
            "template_text": template.template_text,
            "parameter_roles": list(template.parameter_roles),
            "parameter_placeholders": list(template.parameter_placeholders),
            "effect_buckets": {
                "success": template.success_effect_bucket,
                "failure": template.failure_effect_bucket,
            },
            "source_action_name": None,
            "ground_truth_positive_literal": None,
        }
        for template in templates
    }
    return {
        "schema_version": 2,
        "actions_by_name": actions_by_name,
        "lookup": {
            "template_id_to_action": {template.template_id: template.canonical_action_name for template in templates},
            "template_text_to_action": {
                template.template_text: template.canonical_action_name for template in templates
            },
        },
    }


def natural_language_action_to_grounded_action(
    action_text: str,
    templates: list[InducedActionTemplateArtifact],
) -> dict[str, Any]:
    parsed = match_action_text_to_template(action_text, templates)
    grounded_action = (
        f"({parsed.canonical_action_name}"
        f"{(' ' + ' '.join(parsed.action_arguments)) if parsed.action_arguments else ''})"
    )
    return {
        "canonical_action_name": parsed.canonical_action_name,
        "action_arguments": list(parsed.action_arguments),
        "grounded_action_pddl": grounded_action,
        "template_id": parsed.template_id,
        "parameter_roles": list(parsed.parameter_roles),
        "parameter_placeholders": list(parsed.parameter_placeholders),
    }


def grounded_action_to_natural_language(
    canonical_action_name: str,
    action_arguments: list[str],
    templates: list[InducedActionTemplateArtifact],
) -> str:
    template = next((item for item in templates if item.canonical_action_name == canonical_action_name), None)
    if template is None:
        raise ValueError(f"Unknown canonical action name for NL reconstruction: {canonical_action_name!r}")
    if len(action_arguments) != len(template.parameter_placeholders):
        raise ValueError(
            f"{canonical_action_name} expects {len(template.parameter_placeholders)} arguments, "
            f"got {len(action_arguments)}"
        )
    rendered = template.template_text
    for placeholder, value in zip(template.parameter_placeholders, action_arguments):
        rendered = rendered.replace("{" + placeholder + "}", value.replace("_", " "))
    return rendered
