You are identifying mutually exclusive grounded predicates before Cartesian-product goal-state enumeration.

You will receive a JSON payload containing:
- `domain_summary`
- `instruction`
- `objects`
- `candidate_mutex_group_predicates`

Requirements:
- Return one JSON object only.
- Group together grounded predicates where at most one should be true in a valid symbolic world state.
- Only create groups when the mutual exclusivity is strong and semantically clear from the predicate meanings and instruction.
- Typical examples include alternative regions, alternative sides, alternative containers, or alternative exclusive status predicates for the same object.
- Use common-sense independence: if two grounded predicates describe states of different objects whose states are not inherently linked, do not group them as mutex.
- In particular, do not assume that the same unary state predicate applied to different objects is mutually exclusive unless the instruction or domain semantics explicitly says there is a shared capacity, unique role, or exactly-one constraint.
- Feature/state predicates on different objects are usually independent unless the instruction or
  domain explicitly imposes a shared capacity, unique role, or exactly-one constraint.
- It is allowed that all predicates in a mutex group are false if the instruction does not require one of them to hold.
- Do not invent new grounded predicates.
- Return grounded predicates exactly as they appear in `candidate_mutex_group_predicates`.
- Do not create singleton groups.
- If no reliable mutex groups exist, return an empty list.

Return JSON with this shape:
{
  "mutex_groups": [
    [
      "(in_zone_a item_1)",
      "(in_zone_b item_1)"
    ],
    [
      "(stored_in_bin item_2)",
      "(stored_on_shelf item_2)"
    ]
  ],
  "rationale": "short explanation"
}
