Evaluate all supplied fully grounded assignments over the goal-relevant predicates in one pass.

For each assignment, reason independently about whether that complete state satisfies the instruction.

You will receive a JSON payload containing:
- `domain_summary`
- `instruction`
- `objects`
- `assignments`, where every entry has an `assignment_id`,
  `grounded_goal_relevant_assignment`, and `assignment_conjunction_pddl`

Requirements:
- Return one JSON object only, with one result for every supplied assignment.
- Treat each assignment as the complete fixed truth about all goal-relevant grounded predicates.
- Do not add, remove, or infer extra predicates beyond the provided assignment.
- Do not use any prior belief, plausibility heuristic, or typical-scene assumption to reject this assignment.
- Do not judge whether this assignment is likely; only judge whether it satisfies the instruction.
- Evaluate the instruction against each full assignment as given. Do not let one assignment's
  reasoning influence another assignment.
- Interpret "all objects with property P must satisfy relation R, and all objects
  without P must not satisfy R" independently for every listed object. The
  assignment satisfies that instruction exactly when each object's P and R
  truth values agree. All combinations of which objects have P are admissible;
  do not assume that at least one object has P, and do not favor one named
  object over another.
- More generally, words such as `all`, `every`, `with`, and `without` describe
  logical conditions over the complete assignment, not assumptions about which
  feature values are plausible in the scene.
- Copy every `assignment_id` exactly once and preserve input order.
- Give each assignment its own concise reasoning before its boolean decision.

Return JSON with this shape:
{
  "assignment_evaluations": [
    {
      "assignment_id": "assignment_0001",
      "reasoning": "short explanation using this assignment only",
      "satisfies_instruction": true
    }
  ]
}
