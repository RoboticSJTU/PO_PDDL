Parse one demonstrated step into a preliminary reusable action record.

Use the instruction and trajectory only as context for resolving the current action text.

Rules:
- Return JSON only and use snake_case.
- `action_category` is exactly `manipulation` or `active_observation`.
- A manipulation changes world state; an active observation intentionally gathers information
  without changing the task state.
- If `allowed_action_schemas` is supplied, copy the closest valid canonical name from it.
- If `allowed_object_names` is supplied, every action argument and object mention must be copied
  exactly from it. Never append a location or relation to an object identifier.
- `action_arguments` contains grounded entities in semantic argument order. Do not create arguments
  for directions, poses, amounts, states, or task roles.
- `template_text` abstracts only entity mentions into placeholders and preserves action-defining
  wording. `parameter_placeholders` aligns one-to-one with `action_arguments`.
- Merge success and failure attempts under the same action identity; outcome is learned later.
- Preserve a qualifier in the action name only when it changes how the operation is executed or its
  applicability/effect regime. Do not encode intrinsic object attributes or instance names in the
  action identity.

Output shape:
{
  "canonical_action_name": "canonical_action",
  "action_category": "manipulation",
  "action_arguments": ["entity_name"],
  "object_mentions": ["entity_name"],
  "template_text": "perform action on {entity}",
  "parameter_placeholders": ["entity"]
}
