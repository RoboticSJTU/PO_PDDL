You are selecting which predicate schemas matter for evaluating whether a final symbolic world state satisfies a task instruction.

You will receive a JSON payload containing:
- `domain_summary`
- `instruction`
- `objects`
- `candidate_predicates`

Requirements:
- Return one JSON object only.
- Select predicate schemas that materially affect whether a candidate end-state satisfies the instruction.
- When the instruction identifies an object by a property or feature, include the corresponding feature predicates needed to resolve which object the instruction refers to.
- Also include the state predicates that describe the target final condition to be achieved, not just the identifying feature predicates.
- In particular, for move/place/put/push/open/close style goals, include the predicates that encode the desired end-state location, support relation, containment relation, openness state, or other goal-state relation.
- Do not select observation-helper predicates such as `obs-*` or bookkeeping predicates such as `last_action`.
- Prefer semantic task predicates over execution-state predicates.
- It is acceptable to select hidden predicates if they determine which objects the instruction refers to.
- Return predicate names exactly as they appear in `candidate_predicates`.

Return JSON with this shape:
{
  "goal_relevant_predicates": ["target_property", "placed_in_zone"],
  "rationale": "short explanation"
}
