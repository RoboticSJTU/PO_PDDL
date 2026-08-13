Decide whether the image resolves one object's initial location among the supplied candidates.

This is only a visibility/resolvability decision; do not select a predicate.

Rules:
- Return JSON only and follow the image-layout note.
- Return true only when both the target's identity and its relevant physical relation are clear.
- Identify category/type first, then stable intrinsic attributes. A name or matching color alone is
  insufficient when shape/category evidence conflicts.
- Treat listed objects as distinct physical instances and do not reuse one instance for two names.
- Partial visibility is sufficient only when identity and relation remain unambiguous. Return false
  for hidden, absent, heavily occluded, boundary-fragmented, or relation-ambiguous objects.
- Infer physical relations from geometry, not image overlap or demonstration frequency. Interpret
  any direction in image coordinates without mirroring.

Return exactly:
{
  "visually_resolved": true,
  "justification": "brief image-grounded reason"
}
