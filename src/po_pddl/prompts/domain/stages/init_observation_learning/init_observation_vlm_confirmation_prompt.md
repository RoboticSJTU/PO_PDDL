You are independently extracting initial-observation uncertainty from images.

Rules:
- Inspect every grounded assignment in `candidate_ground_truth_facts`; the `suspects` list is optional, non-exhaustive guidance and may be empty or wrong.
- `uncertain_predicate_names` was independently selected by a dataset-level image review. Return contradictions only for those predicate names.
- First decide which predicates represent properties or relations that the supplied initial images can reasonably observe. Derive this from the images, instruction, predicate comments, and predicate arguments. No feature-predicate whitelist is provided.
- `candidate_ground_truth_facts` is the complete grounded truth assignment for the relevant predicates, including both true and false literals.
- `ground_truth_value` describes the real world state. `observed_value` describes what the visual observation reports; these are deliberately allowed to disagree.
- Use the images as the authority for `observed_value`. The instruction and symbolic ground truth identify what to inspect, but they are not visual evidence that a property is present.
- For a directly inspectable visual property whose target and relevant surface/interior are visible, absence of visual evidence for the positive property means `observed_value=false`. Do not reinterpret this as unknown merely because the symbolic ground truth says true.
- When a predicate asserts that a container has some content, distinguish that content from the
  container's own interior color, shading, and reflections. A positive observation requires a
  separately visible object, material boundary, surface, meniscus, or texture discontinuity. A
  smoothly colored interior without such independent evidence is a negative observation, even if
  its color resembles the asserted content or the instruction names that content.
- If the target or the region needed to inspect the property is genuinely hidden, outside the view, or occluded, return the row with `visually_assessable=false`; it will not be counted as a contradiction.
- Return one `observation_evaluations` row for every supplied grounded assignment. Do not omit a row merely because observation and ground truth agree.
- If the images support the ground-truth value, return the matching `observed_value`; it will not be counted as a contradiction.
- Be careful not to reverse the semantics: `observed_value=false` means the literal appears visually false; `observed_value=true` means the literal appears visually true.
- Do not invent predicates or objects. Every returned positive `grounded_literal` must correspond to one assignment supplied in `candidate_ground_truth_facts`.
- Examine all supplied images before returning, even when `suspects` is empty.

Return JSON:
{
  "summary": "brief summary",
  "observation_evaluations": [
    {
      "predicate_name": "is_fragile",
      "grounded_literal": "is_fragile(tool_a)",
      "observed_value": false,
      "ground_truth_value": true,
      "visually_assessable": true,
      "rationale": "The feature is visually absent or not described in the initial scene."
    }
  ]
}
