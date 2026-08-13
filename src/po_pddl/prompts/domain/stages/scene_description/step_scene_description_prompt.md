Describe the physical world state after the current demonstrated step.

Rules:
- Return JSON only as `{"scene_description_text": "..."}`.
- The ordered images are authoritative; the final frame is primary. Earlier frames establish what
  changed or resolve occlusion. Instruction/action text guide attention and naming only.
- Mention only entities in `allowed_object_names`, using exact stable names. Treat repeated views as
  one entity.
- Focus on action-caused changes, newly revealed facts, and the action target's final state. Preserve
  a prior fact only when no intervening evidence/action changed it and it is needed for context.
- For failed steps, describe the actual visible result rather than the intended result.
- State physical location/support/containment, open-like state, and relevant hand/gripper occupancy;
  avoid image-coordinate descriptions.
- Do not invent hidden objects or let future actions override visual evidence.
- Keep the result to one or two concise sentences.
