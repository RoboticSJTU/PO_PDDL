You are filtering grounded predicates before Cartesian-product goal-state enumeration.

You will receive a JSON payload containing:
- `domain_summary`
- `instruction`
- `objects`
- `candidate_grounded_predicates`
- `selected_mutex_groups`

Your job is to remove grounded predicates that are clearly semantically invalid or obviously redundant only because the predicate signature is too broad.

Requirements:
- Return one JSON object only.
- Only prune grounded predicates when the semantic invalidity is strong and unambiguous.
- Focus on grounded predicates that should not even participate in goal enumeration.
- Typical examples:
  - an object being left/right/front/behind/above/below/near/far relative to itself
  - an object being stacked on top of itself
  - an object being inside, on, under, or supported by itself when that is clearly nonsensical for the predicate meaning
- Do not prune predicates merely because they are unlikely, unnecessary, or not mentioned explicitly in the instruction.
- Do not prune predicates that could still be meaningful in some valid world state.
- Do not invent new grounded predicates.
- Return grounded predicates exactly as they appear in `candidate_grounded_predicates`.
- If nothing is clearly invalid, return an empty list.

Return JSON with this shape:
{
  "pruned_grounded_predicates": [
    "(left_of block_a block_a)",
    "(on_top_of cup_1 cup_1)"
  ],
  "rationale": "short explanation"
}
