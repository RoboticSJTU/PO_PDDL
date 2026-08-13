Consolidate all preliminary taxonomy records into a global action-schema inventory and assign every
record to exactly one schema.

Rules:
- Return one JSON object only. Use snake_case.
- Categories are exactly `manipulation` and `active_observation`.
- Merge paraphrases and actions differing only by object instances or intrinsic attributes.
- Never split success and failure attempts; they are outcomes of one action schema.
- Split actions only for a real operational distinction: different intended transition, execution
  method, parameter signature, or applicability/effect regime supported by the data.
- Preserve an action-level direction or mode only when it changes execution semantics. This does not
  require later predicate names to duplicate that qualifier.
- Parameters represent physical entity kinds, never task roles such as source or destination.
- `parameter_count` must equal the lengths of `parameter_roles` and the intended grounded argument
  list. Keep argument order consistent across assigned records.
- `precondition_literals` must use only `allowed_predicates` when supplied; otherwise leave uncertain
  preconditions empty for the dedicated learning stage.
- If `allowed_action_schemas` is supplied, use only those names and prefer exact reuse.
- Include a concise, data-grounded `schema_description`.

Output shape:
{
  "action_schemas": [
    {
      "canonical_action_name": "canonical_action",
      "action_category": "manipulation",
      "parameter_count": 1,
      "parameter_roles": ["entity_type"],
      "precondition_literals": [],
      "schema_description": "Concise action semantics."
    }
  ],
  "record_assignments": [
    {
      "episode_name": "episode_name",
      "step_index": 1,
      "canonical_action_name": "canonical_action"
    }
  ]
}
