Induce reusable semantic action templates from the supplied action texts.

Rules:
- Return JSON only with top-level `action_templates`; do not classify action category here.
- Use snake_case for `template_id` and `canonical_action_name`. Derive both from the complete fixed
  template wording in order, replacing each entity marker with `object` and omitting only articles.
  Do not omit a relation, direction, mode, or entity slot represented by the template.
- Merge paraphrases with the same operation; split only genuinely different execution semantics.
- Success and failure wording share one template.
- Parameters are concrete task entities only. A targeted fixture or receptacle is still an entity
  parameter even if it never moves.
- Do not parameterize directions, regions, states, poses, amounts, or other descriptors.
- Keep action-defining descriptors in fixed wording and the action identity. Keep intrinsic object
  attributes inside the entity value, not in the action identity.
- When an action relates multiple entities, use a separate `{param_k}` for each one in grounded
  argument order; never absorb a relation and second entity into one value.
- Preserve the original non-entity words and their order in `template_text`; do not paraphrase them.
  Substituting the extracted entity spans for the markers must reconstruct every assigned input
  action text exactly, ignoring only capitalization and outer whitespace.
- `template_text` uses contiguous markers `{param_1}`, `{param_2}`, and so on.
- If `validation_feedback` is present, repair the reported structural issue.

Output shape:
{
  "action_templates": [
    {
      "template_id": "canonical_action",
      "template_text": "perform the action on {param_1}",
      "canonical_action_name": "canonical_action"
    }
  ]
}
