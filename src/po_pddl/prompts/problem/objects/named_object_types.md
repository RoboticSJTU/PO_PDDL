You are assigning learned-domain types to a provided set of scene object names
for an online POMDPDDL planning problem.

You will receive:
- one image
- a JSON payload containing:
  - `domain_summary`
  - `instruction`
  - `image_input_note`
  - `candidate_object_names`
  - `valid_domain_types`
  - `manipulation_records`

Task:
- Return every name in `candidate_object_names` exactly once.
- Assign each object the most specific compatible type from
  `valid_domain_types`.
- Use the object name, image evidence, domain predicate/action signatures, and
  manipulation argument usage to infer its type.
- A listed object may be hidden or occluded in the current image. Do not drop it
  merely because it is not visible.

Important rules:
- Return one JSON object only.
- Copy object names exactly. Do not rename, normalize, or invent objects.
- Use only types listed in `valid_domain_types`.
- Prefer a specific subtype when the evidence supports it; use `object` only
  when no declared subtype is justified.
- Do not infer an object's type from its current left/right position.

Return JSON with this exact shape:
{
  "objects": [
    {
      "name": "example_object",
      "type_name": "object",
      "justification": "Short explanation based on the available evidence."
    }
  ]
}
