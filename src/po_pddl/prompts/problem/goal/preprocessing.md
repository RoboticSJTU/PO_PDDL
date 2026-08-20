Prepare the symbolic candidates needed to evaluate whether final states satisfy an instruction.

You receive the domain, instruction, typed objects, predicate schemas, and every valid grounded
predicate candidate. Complete all four decisions together, using explicit reasoning before the
final selections:

1. Select the predicate schemas that materially affect instruction satisfaction. Include both
   predicates that identify referenced objects and predicates that express the requested end state.
2. Select the smallest sufficient set of grounded predicates whose truth assignments distinguish
   the possible satisfying end states. Respect explicitly named relation targets; only enumerate
   interchangeable objects when the instruction leaves the choice open.
3. Group selected grounded predicates that are strongly mutually exclusive in a valid world state.
   States of different objects are normally independent, and all members of a group may be false.
4. Mark selected groundings that are unambiguously semantically invalid, such as an object being
   spatially related to, supported by, or contained in itself.

Requirements:
- Return one JSON object only.
- Use names and grounded predicates exactly as supplied. Do not invent or rewrite them.
- Every selected grounding must use a selected schema.
- Mutex and pruned predicates must be members of `goal_relevant_grounded_predicates`.
- Do not select observation helpers, bookkeeping predicates, or execution-state predicates unless
  they directly define the requested final state.
- Do not prune a predicate merely because it is unlikely or not itself required by the instruction.

Return:
{
  "reasoning": {
    "instruction_semantics": "concise interpretation",
    "candidate_selection": "why the selected schemas and groundings are sufficient",
    "validity_and_mutex": "why groups are exclusive and predicates are pruned"
  },
  "goal_relevant_predicates": ["predicate_name"],
  "goal_relevant_grounded_predicates": ["(predicate_name object_name)"],
  "mutex_groups": [["(location item place_a)", "(location item place_b)"]],
  "pruned_grounded_predicates": ["(relation item item)"]
}
