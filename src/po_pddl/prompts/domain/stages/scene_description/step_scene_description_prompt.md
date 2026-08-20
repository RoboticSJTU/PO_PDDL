Describe the physical world state after the current demonstrated step.

Rules:
- Return JSON only as `{"scene_description_text": "..."}`.
- The ordered images are authoritative; the final frame is primary. Earlier frames establish what
  changed or resolve occlusion. Instruction/action text guide attention and naming only.
- Mention only entities in `allowed_object_names`, using exact stable names. Treat repeated views as
  one entity.
- The allowlist is not evidence that an entity should be visible. Omit an unseen entity rather than
  stating that it is absent or not visible.
- Treat the result as a current visual observation, not a carried-forward belief state. Focus on
  action-caused changes, newly revealed facts, and the action target's final visible state.
- Do not carry a latent, content, or feature fact from prior text unless it is clearly visible in the
  current frames or the current action directly reveals it. Omit unsupported facts rather than infer
  that they persist.
- For failed steps, describe the actual visible result rather than the intended result.
- State physical location/support/containment, visible contents or features, and relevant
  hand/gripper occupancy. Explicitly report the open/closed state of each relevant articulated
  container when discernible; do not call a permanently open receptacle or ordinary cup `open`.
- Distinguish physical contents from the container's own interior color, shadows, and reflections.
  Claim contents only when a distinct object, material boundary, or liquid surface is visually
  supported; otherwise omit the claim rather than interpreting dark shading as material.
- After an inspection action, explicitly report the visually revealed property. After a grasp or
  release action, explicitly report hand/gripper occupancy when it is visible.
- When a released object visibly rests independently on its destination, state that it is no longer
  held even if the hand/gripper is only partly visible.
- Avoid image-coordinate descriptions.
- Do not invent hidden objects or let future actions override visual evidence.
- Keep the result to one or two concise sentences.
