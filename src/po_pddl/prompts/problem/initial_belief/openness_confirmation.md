You are independently auditing a proposed positive openness judgment for one
object in an initial-scene image.

You will receive:
- one image
- a JSON payload containing the domain summary, image description, object
  inventory, target openness predicate, and the initial justification

Task:
- Decide whether the image contains direct visual evidence that the target
  object's movable part is open.
- This is a conservative confirmation step. Confirm true only when the opening
  evidence is clear and belongs to the named target object.

Required checks:
- Locate the target object's main body before inspecting its movable part.
- Ignore handles, knobs, seams, decorations, shadows, perspective distortion,
  foreground clutter, and boundaries of overlapping objects.
- A handle, knob, seam, or closure panel may remain visible when closed. Require displacement of
  the movable closure, an exposed interior, visible articulation hardware, or a clear opening gap.
- Compare the movable part with the surrounding body and, when visible, nearby
  objects of the same category.
- The initial justification is only a claim to audit, not evidence.
- If the evidence is ambiguous or you cannot identify a concrete opening cue,
  return false.

Return one JSON object only:
{
  "confirmed_true": false,
  "opening_cues": [],
  "justification": "No displaced movable part or exposed interior is visible."
}
