Merge all noisy effect variants for one action and one authoritative success/failure outcome into a
single corrected effect.

Rules:
- Return JSON only with exactly one merge plan covering every supplied `variant_rank`.
- Never mix success and failure evidence or invent actions, objects, predicates, or ranks.
- Use raw records and scene descriptions to distinguish omissions/redundancy from real changes.
- Choose the smallest complete effect supported across the records. Prefer a well-supported full
  transition over a sparse partial variant.
- Preserve both sides of a state transition: add the resulting state and delete incompatible prior
  location/support/containment or gripper states.
- Literals are abstract and use only action markers `?param_1`, `?param_2`, ...; never use grounded
  object names.

Output shape:
{
  "merge_plans": [
    {
      "source_variant_ranks": [1, 2],
      "merged_delta_add": ["predicate(?param_1)"],
      "merged_delta_del": ["other_predicate(?param_1)"],
      "rationale": "brief evidence-based reason"
    }
  ]
}
