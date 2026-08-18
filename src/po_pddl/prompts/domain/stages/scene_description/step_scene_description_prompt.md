Describe the physical world state after the current demonstrated step.

Rules:
- Return JSON only as `{"scene_description_text": "..."}`.
- The ordered images are authoritative; the final frame is primary. Earlier frames establish what
  changed or resolve occlusion. Instruction/action text guide attention and naming only.
- Mention only entities in `allowed_object_names`, using exact stable names. Treat repeated views as
  one entity.
- Treat the result as a current visual observation, not a carried-forward belief state. Focus on
  action-caused changes, newly revealed facts, and the action target's final visible state.
- Do not carry a latent, content, or feature fact from prior text unless it is clearly visible in the
  current frames or the current action directly reveals it. Omit unsupported facts rather than infer
  that they persist.
- For failed steps, describe the actual visible result rather than the intended result.
- State physical location/support/containment, open-like state, and relevant hand/gripper occupancy;
  avoid image-coordinate descriptions.
- Do not invent hidden objects or let future actions override visual evidence.
- Keep the result to one or two concise sentences.
