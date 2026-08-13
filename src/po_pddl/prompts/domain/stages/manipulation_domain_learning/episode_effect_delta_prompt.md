Infer the grounded symbolic effect of one demonstrated manipulation step.

Rules:
- Return JSON only. Copy `canonical_action_name` and ordered `action_arguments` exactly from the
  current schema/record.
- Use only supplied allowed predicates and grounded objects. Candidate facts are possibilities, not
  assertions about the before-state.
- The step scene description and images are primary evidence; action text and instruction identify
  intent but do not alone prove the outcome. However, limited camera visibility is not evidence of
  failure. Mark a demonstrated step as failed only when the annotation or observations positively
  support failure. If the demonstrated transition is physically consistent (for example, a release
  is observed but its destination is just outside the final view), infer the intended transition
  from the combined action, temporal, and hand-state evidence.
- Add exactly facts that become true and delete exactly facts that cease to be true. Include every
  clearly supported action-caused change and no unchanged context.
- Apply any `review_guidance.required_effect_facts_add/del` exactly.
- Use only `gripper_empty()` and `gripper_holding(entity)` for gripper state.
- A successful acquisition normally adds holding, deletes empty, and deletes the object's previous
  incompatible location. A successful release normally deletes holding, adds empty, and adds the
  observed resulting location. Encode these transitions from evidence, not verb matching alone.
- A location transition must add the new relation and delete incompatible old relations when known.
- Failure uses the same action name with `success: false`. It is a no-op only if evidence shows no
  symbolic change; otherwise encode the actual resulting gripper and world state.
- `effect_bucket` identifies this action and outcome, not an object instance.

Output shape:
{
  "canonical_action_name": "canonical_action",
  "action_arguments": ["entity_name"],
  "delta_add": ["predicate(entity_name)"],
  "delta_del": ["other_predicate(entity_name)"],
  "effect_bucket": "canonical_action_success",
  "success": true
}
