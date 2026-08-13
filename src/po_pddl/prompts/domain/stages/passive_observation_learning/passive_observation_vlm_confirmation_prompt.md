You are confirming a small set of suspected passive-observation contradictions with images.

Rules:
- This is a confirmation step, not an open-ended world reconstruction step.
- `candidate_ground_truth_facts` is the complete grounded truth assignment for the relevant predicates, including both true and false literals.
- Only confirm contradictions that are clearly supported by the images plus the scene description.
- If the images do not clearly support the contradiction, do not confirm it.
- `observed_value` must be the actual truth value supported by the images and the scene description for the given grounded literal.
- Only include a grounded literal in `confirmed_contradictions` when the actual `observed_value` is different from `ground_truth_value`.
- If the images support the ground-truth value, do not include that grounded literal in `confirmed_contradictions`.
- Do not treat "the ground truth is visually supported" as a contradiction. In that case, omit the literal from `confirmed_contradictions`.
- Be careful not to reverse the semantics: `observed_value=false` means the literal appears visually false; `observed_value=true` means the literal appears visually true.
- If the only reason a grounded literal is not visible is that the object remains hidden inside a closed or still-occluded container/space, do not confirm a contradiction. Hidden-space partial observability by itself should not become a learned observation error.

Return JSON:
{
  "summary": "brief summary",
  "confirmed_contradictions": [
    {
      "predicate_name": "stored_in",
      "grounded_literal": "stored_in(parcel_a,receptacle_b)",
      "observed_value": false,
      "ground_truth_value": true,
      "rationale": "The expected membership is not visibly confirmed after the reveal."
    }
  ]
}
