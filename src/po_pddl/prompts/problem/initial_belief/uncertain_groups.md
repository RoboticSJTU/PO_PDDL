Group all supplied uncertain grounded predicates into local initial-belief factors.

Group kinds:
- `binary`: one predicate with true/false alternatives.
- `mutex`: exhaustive, mutually exclusive predicates for one underlying state.
- `correlated`: coupled predicates that are not an exhaustive one-of-K choice.

Rules:
- Return JSON only. Copy every supplied predicate verbatim into exactly one group; never invent or
  omit predicates.
- Prefer small local groups. Use binary for a singleton unless a real dependency exists.
- Use mutex only when exactly one member must hold. If all members may be false, use correlated.
- Do not couple independent states or unrelated objects merely because they share a predicate name.

Return exactly:
{
  "groups": [
    {
      "name": "concise_factor_name",
      "group_kind": "binary",
      "predicates": ["(predicate copied exactly)"],
      "description": "brief semantic description"
    }
  ]
}
