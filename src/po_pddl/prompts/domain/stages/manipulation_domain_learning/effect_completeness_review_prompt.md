Review one grounded effect sample for an obvious action-caused state change missing from the current
runnable domain.

Be conservative:
- Return no repair unless the final image and scene description clearly show a target-state change
  omitted by both the current effect and existing predicates.
- Do not add incidental appearance, background, identity, or unchanged-state predicates.
- Reuse an existing predicate whenever it can represent the state.
- For a moved entity, check that the new location/support/containment relation is added and an
  incompatible old relation is deleted; also check consistency with gripper state.
- A new predicate must use snake_case, existing available types, and exact grounded object names.
- Never use the top type `object`.

Return JSON only:
{
  "should_apply_repair": false,
  "repair_summary": "",
  "predicate_additions": []
}

When a repair is clearly necessary, each addition has:
`predicate_name`, `parameter_types`, `comment`, `grounded_fact`, and boolean `value_after_action`.
