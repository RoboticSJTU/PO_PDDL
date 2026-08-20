Describe the initial physical world state shown in the supplied image(s).

Rules:
- Return JSON only as `{"scene_description_text": "..."}`.
- Visual evidence is authoritative. The instruction and future actions may guide attention and
  consistent naming, but they are not evidence that a hidden fact is true.
- Mention only entities in `allowed_object_names`; use each exact canonical name and do not duplicate
  one entity across views.
- The allowlist is not a checklist: omit an allowed entity when it is not visible instead of saying
  that it is absent or not visible.
- Describe task-relevant object identity, physical location/support/containment relations, visible
  contents or features, and whether a relevant hand/gripper is empty or holding something.
- For each visible articulated container that can be opened or closed, explicitly state its
  open/closed state when discernible. Do not call a permanently open receptacle or ordinary cup
  `open` merely because its interior is exposed.
- When the instruction or action sequence makes container contents relevant, inspect the visible
  interior explicitly. Report visible contents, but describe it as empty only when the relevant
  interior is sufficiently visible to support that judgment.
- Distinguish physical contents from the container's own interior color, shadows, and reflections.
  Claim contents only when a distinct object, material boundary, or liquid surface is visually
  supported; otherwise omit the claim rather than interpreting dark shading as material.
- Describe world relations, not image coordinates. Do not infer occluded contents or outcomes.
- Under genuine visual ambiguity, later actions may break a tie about a compatible prior state, but
  never override clear image evidence.
- Be concise and omit uncertain or irrelevant details.
