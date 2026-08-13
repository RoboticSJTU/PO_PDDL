You are classifying already-normalized semantic action templates.

This stage happens after semantic action-template induction and action-text normalization.
Your only job is to decide whether each action template is:
- `manipulation`
- `active_observation`

You will receive:
- `instruction`
- `action_templates`

Rules:
- Return JSON only.
- The top-level object must contain `action_categories`.
- Each returned row must include:
  - `canonical_action_name`
  - `action_category`
- Do not rename actions.
- Do not add or remove action templates.
- Use only the two allowed categories:
  - `manipulation`
  - `active_observation`
- Classify from the action semantics described by the template and example texts.
- `active_observation` means the action is primarily for sensing/inspecting/checking rather than physically changing the world.
- `manipulation` means the action primarily changes the world state by moving/opening/placing/picking/stacking/pouring/etc.

Return JSON with this shape:
{
  "action_categories": [
    {
      "canonical_action_name": "pick_up_object_on_right",
      "action_category": "manipulation"
    },
    {
      "canonical_action_name": "inspect_object_contents",
      "action_category": "active_observation"
    }
  ]
}
