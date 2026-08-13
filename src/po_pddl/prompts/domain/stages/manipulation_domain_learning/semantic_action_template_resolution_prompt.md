Resolve one unmatched action text against the existing semantic template inventory.

Rules:
- Return JSON only. `resolution_kind` is `match_existing` or `new_action`.
- Match by operation semantics, not surface wording. Prefer an existing template; create a new one
  only for a genuinely different execution meaning or entity signature.
- Success/failure and intrinsic entity attributes never create a new action identity.
- Parameters are entity spans only. Keep relations, directions, modes, states, and amounts in fixed
  template wording when they define the action.
- Every explicitly targeted task entity gets its own marker. Do not hide a relation or a second
  entity inside a placeholder value.
- Preserve all non-entity words and their order from the unmatched action. Substituting entity spans
  into the selected/new template must reconstruct the full input text without paraphrasing.
- `placeholder_values` maps every template marker to its exact action-text entity span.
- New templates use snake_case and contiguous `{param_1}`, `{param_2}`, ... markers. Derive the id
  and canonical name from all fixed template words in order, replacing entity markers with
  `object` and dropping only articles. Do not include `action_category`.
- Apply `validation_feedback` when supplied.

For an existing match return:
{
  "resolution_kind": "match_existing",
  "matched_template_id": "template_id",
  "placeholder_values": {"param_1": "entity span"},
  "rationale": "brief reason"
}

For a new action return:
{
  "resolution_kind": "new_action",
  "new_action_template": {
    "template_id": "canonical_action",
    "template_text": "perform action on {param_1}",
    "canonical_action_name": "canonical_action"
  },
  "placeholder_values": {"param_1": "entity span"},
  "rationale": "brief reason"
}
