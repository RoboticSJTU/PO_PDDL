You are selecting which predicate schemas should be treated as location/state-of-placement predicates for initial-state image judgment.

You will receive a JSON payload containing:
- `domain_summary`
- `candidate_predicates`

Task:
- Select predicate schemas that express where a movable object is located, supported, contained, or spatially related in the scene.
- Include predicates for relations such as being in front of something, on top of something, inside something, in something, next to something, left/right of something, on a table, or similar placement/location relations.
- Do not select bookkeeping predicates such as `last_action_*`.
- Do not select gripper-state predicates such as `gripper_empty` or `gripper_holding*`.
- Do not select object feature predicates such as liquid/color/material/content features unless they directly express location.
- Be conservative, but prefer recall for genuine location predicates.

Return one JSON object only with this shape:
{
  "location_predicates": ["in_front_of", "on_top_of_box_drawer", "in"],
  "rationale": "short explanation"
}
