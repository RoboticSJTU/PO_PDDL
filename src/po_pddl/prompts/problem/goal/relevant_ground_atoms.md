You are selecting which grounded predicates should be fully enumerated as hidden task-relevant possibilities before constructing a symbolic goal.

You will receive a JSON payload containing:
- `domain_summary`
- `instruction`
- `objects`
- `goal_relevant_predicates`
- `candidate_grounded_predicates`

Requirements:
- Return one JSON object only.
- Select grounded predicates whose different truth assignments materially change which final goal states satisfy the instruction.
- Include grounded predicates for both:
  - object-identifying feature predicates that disambiguate which object the instruction is about, and
  - target goal-state predicates that describe the final state that must hold.
- Do not stop after selecting only feature predicates. If the instruction requires an object to end up somewhere, be on top of something, be in front of something, be inside something, or otherwise satisfy a final relational/state condition, include the corresponding grounded state predicates too.
- Return grounded predicates exactly as they appear in `candidate_grounded_predicates`.
- Do not invent new grounded predicates.
- When the instruction explicitly names the reference/target object of a
  relation, select relation groundings with that named reference only. Do not
  enumerate interchangeable objects of the same type unless the instruction
  leaves the reference choice open.
- Prefer the smallest sufficient set, but do not omit grounded predicates that create distinct instruction-conditioned possibilities.

Return JSON with this shape:
{
  "goal_relevant_grounded_predicates": [
    "(target_property item_a)",
    "(target_property item_b)"
  ],
  "rationale": "short explanation"
}
