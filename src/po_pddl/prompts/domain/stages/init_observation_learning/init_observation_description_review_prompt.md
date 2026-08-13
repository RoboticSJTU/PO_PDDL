You are reviewing whether the initial scene description may omit or contradict grounded initial-state predicates.

Rules:
- Use the initial scene description as the primary evidence.
- `filtered_init_facts` contains the true init predicates after filtering out gripper-related predicates.
- `candidate_ground_truth_facts` is the complete grounded truth assignment for the remaining predicates. Each grounded predicate appears exactly once, either as a positive literal like `pred(obj)` or a negative literal like `not pred(obj)`.
- Compare the filtered symbolic init facts and complete grounded truth assignment against the described initial world state.
- Only mark predicates as suspects when the scene description suggests the current symbolic init description may be incomplete or inconsistent.
- Use the predicate comments and instruction to identify properties that may be visually inspected. Do not assume that a predicate is or is not observation-relevant from its name alone.
- If a mismatch is explained purely by an entity being hidden inside a closed containable item, do
  not treat visual absence as an observation contradiction.
- This review only supplies hints to the image reviewer. Missing a suspect here must not prevent the image reviewer from independently finding it.

Return JSON:
{
  "should_review_with_vlm": true,
  "summary": "brief summary",
  "suspect_contradictions": [
    {
      "predicate_name": "is_fragile",
      "grounded_literal": "is_fragile(tool_a)",
      "observed_value": false,
      "ground_truth_value": true,
      "rationale": "The description does not reflect the goal-relevant feature."
    }
  ]
}
