Decide which target objects have visually resolved initial locations in one global scene analysis.

The payload provides the image-layout note, domain, task instruction, optional initial-state hint,
typed objects, and candidate location predicates grouped by target object.

Rules:
- Return JSON only and include every target object exactly once.
- Analyze all targets jointly and map each visible physical instance to at most one symbolic name.
- A target is visually resolved only when both its identity and its relevant physical relation are
  clear. The instruction, initial-state hint, and supplied symbolic names are context, not evidence
  that an object is visible.
- Identify category/type first, then stable intrinsic attributes. Do not relabel a distractor as a
  target merely because it has a similar shape, color cast, or screen position. If the target name
  encodes an intrinsic attribute, visible evidence must be compatible with it.
- Reconcile multiple camera views as views of the same scene. Lighting and pose may vary, but one
  physical instance remains one object.
- Return false for hidden, absent, heavily occluded, boundary-fragmented, identity-ambiguous, or
  relation-ambiguous targets. An object potentially hidden inside a closed container is unresolved
  unless it is actually visible and its relation is clear.
- This is only a visibility/resolvability decision. Do not select a candidate predicate.

Return exactly:
{
  "scene_reasoning": "concise global identity and visibility analysis",
  "object_visibility": [
    {
      "target_object": "name copied exactly",
      "visually_resolved": true,
      "justification": "brief image-grounded reason"
    }
  ]
}
