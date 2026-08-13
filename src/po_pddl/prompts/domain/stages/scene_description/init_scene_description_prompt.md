Describe the initial physical world state shown in the supplied image(s).

Rules:
- Return JSON only as `{"scene_description_text": "..."}`.
- Visual evidence is authoritative. The instruction and future actions may guide attention and
  consistent naming, but they are not evidence that a hidden fact is true.
- Mention only entities in `allowed_object_names`; use each exact canonical name and do not duplicate
  one entity across views.
- Describe task-relevant object identity, physical location/support/containment relations, open-like
  states, and whether a relevant hand/gripper is empty or holding something when visible.
- Describe world relations, not image coordinates. Do not infer occluded contents or outcomes.
- Under genuine visual ambiguity, later actions may break a tie about a compatible prior state, but
  never override clear image evidence.
- Be concise and omit uncertain or irrelevant details.
