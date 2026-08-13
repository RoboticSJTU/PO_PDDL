You are reviewing whether a manipulation-effect variant may induce passive observation uncertainty.

Rules:
- Use the scene description as the primary evidence.
- `filtered_state_before` contains the true state-before predicates after filtering out gripper-related predicates and any predicates already determined by the current action effect.
- `candidate_ground_truth_facts` is the complete grounded truth assignment for the relevant predicates. Each grounded predicate appears exactly once, either as a positive literal like `pred(obj)` or a negative literal like `not pred(obj)`.
- Compare the filtered symbolic state and complete grounded truth assignment against the described world state.
- Only mark predicates as suspects when the scene description suggests the current symbolic description may be incomplete or inconsistent.
- `feature_predicate_names` lists goal-relevant feature predicates. These predicates may or may not exist for a given task.
- When a grounded fact uses a goal-relevant feature predicate, be more sensitive to whether the scene description actually reflects that feature for the referenced object.
- Be especially sensitive to spatial or membership predicates when an action changes visibility. If an action exposes a region and a grounded fact says an entity occupies that region, but the resulting description contradicts that fact, treat it as a likely contradiction candidate.
- Always follow the provided predicate signature and argument semantics; do not assume a universal argument order for spatial relations.
- If a mismatch is explained purely by hidden-space partial observability, do not treat it as an observation contradiction. For example, if an object could simply remain hidden inside a closed or still-occluded container/space, the absence of a mention should not be learned as an observation error.
- Still be cautious: omission alone is not enough if the object is not clearly described or there is not enough evidence that the feature should have been observed.
- Be conservative. If the evidence is weak, do not escalate to VLM.

Return JSON:
{
  "should_review_with_vlm": true,
  "summary": "brief summary",
  "suspect_contradictions": [
    {
      "predicate_name": "stored_in",
      "grounded_literal": "stored_in(parcel_a,receptacle_b)",
      "observed_value": false,
      "ground_truth_value": true,
      "rationale": "The resulting description contradicts the expected visible membership."
    }
  ]
}
