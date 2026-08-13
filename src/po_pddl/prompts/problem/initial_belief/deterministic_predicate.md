Judge whether one grounded predicate is true in the supplied initial-scene image.

The payload provides the domain, image-layout note, object inventory, and `target_predicate`.

Rules:
- Return JSON only and choose exactly `true` or `false`.
- Use current visual evidence and the predicate signature/semantics only. Do not infer future actions,
  goals, or demonstration-frequency priors.
- Identify objects by category/type first and then stable intrinsic attributes. Respect predicate
  argument order.
- Prefer false when the image does not support the predicate.
- For openness, require a visible cue belonging to the object itself, such as displacement of its
  movable part, an exposed interior, or a gap created by opening. A handle, seam, image overlap,
  foreground occluder, or oblique view alone is not evidence of openness.
- For spatial relations, use physical geometry rather than 2D overlap.

Return exactly:
{
  "truth_value": "true",
  "justification": "brief image-grounded reason"
}
