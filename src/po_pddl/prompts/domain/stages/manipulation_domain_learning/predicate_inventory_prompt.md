You design the smallest coherent predicate vocabulary that can explain and replay the supplied demonstrations.

Return exactly one JSON object with these keys:
`uses_movable_item_type`, `movable_item_member_types`, `uses_fixed_item_type`,
`fixed_item_member_types`, `uses_containable_item_type`,
`containable_item_member_types`, and `predicate_inventory`.

Evidence and scope:
- Use only the supplied instructions, action records, action templates, and available types.
- Design one global vocabulary for the whole dataset, not one predicate set per action or episode.
- Predicate parameter types must be drawn from `available_types` or the optional supertypes
  `movable_item`, `fixed_item`, and `containable_item`. Never use `object` or task roles such as
  source, target, or destination as a parameter type.

Type rules:
- `movable_item` contains concrete types whose instances are grasped or relocated.
- `fixed_item` contains concrete types whose instances remain spatially fixed. Changing an
  internal state such as open/closed does not make an object movable.
- `containable_item` contains types that can hold movable objects. It may overlap with either
  special supertype, but `movable_item` and `fixed_item` must not overlap.
- Set each `uses_*` flag consistently with its member list.
- If `containable_item` is used, include exactly one shared state predicate
  `in(movable_item, containable_item)`, with contained movable item first. Do not create aliases.

Vocabulary rules:
- Include `gripper_empty()` and `gripper_holding(movable_item)` as the only gripper-state
  predicates.
- Use snake_case and concise English comments.
- Prefer a small set of reusable state variables. Create one predicate per semantic relation.
- Do not copy incidental wording from action names into predicate names. In particular, when a
  binary relation already names its reference entity, an action's direction, side, source, or
  destination qualifier normally belongs to the action schema or to a separate property of that
  reference entity, not to a duplicate relation predicate.
- Preserve qualifiers that determine action applicability. If otherwise similar action families
  are distinguished by a region, side, orientation, mode, or status that cannot be recovered from
  their parameter values or parameter types, represent that distinction with a reusable predicate.
  Use a unary predicate on the qualified entity when appropriate, and mark it as a static feature
  only when the demonstrations never change it. Without such a predicate, differently qualified
  schemas would be incorrectly applicable to the same grounded objects.
- Split a relation into variants only when the variants denote independently different world
  states that cannot be recovered from the relation arguments and other predicates.
- Choose predicate arity from the state being represented: unary for an intrinsic/object state,
  binary or higher only for a genuine relation among those objects.
- For every relation, order parameters as semantic subject then reference entity, preserving the
  corresponding entity order used in action templates and descriptions. Thus containment is
  `in(contained_movable_item, container)`.
- Do not create aliases, complements, type-pair variants, episode-specific predicates, or
  predicates used only to restate an object's canonical name.
- Persistent feature predicates are optional. Add one only when an instruction or observation
  requires reasoning over a property shared by otherwise interchangeable objects. A word used as
  part of a unique object name is not by itself a reason to model that property.
- Mark changing or replay-relevant predicates as `state`; mark only non-changing selection or
  observation properties as `feature` and set `is_static_feature` accordingly.

Each `predicate_inventory` row must contain:
`predicate_name`, `parameter_types`, `comment`, `predicate_kind`, and `is_static_feature`.
`predicate_kind` is exactly `state` or `feature`; `is_static_feature` is true exactly for features.

Output shape:
{
  "uses_movable_item_type": true,
  "movable_item_member_types": ["item_type"],
  "uses_fixed_item_type": true,
  "fixed_item_member_types": ["fixture_type"],
  "uses_containable_item_type": true,
  "containable_item_member_types": ["fixture_type"],
  "predicate_inventory": [
    {
      "predicate_name": "in",
      "parameter_types": ["movable_item", "containable_item"],
      "comment": "The movable item is inside the containable item.",
      "predicate_kind": "state",
      "is_static_feature": false
    },
    {
      "predicate_name": "gripper_empty",
      "parameter_types": [],
      "comment": "The gripper is empty.",
      "predicate_kind": "state",
      "is_static_feature": false
    },
    {
      "predicate_name": "gripper_holding",
      "parameter_types": ["movable_item"],
      "comment": "The gripper is holding the movable item.",
      "predicate_kind": "state",
      "is_static_feature": false
    }
  ]
}

Return JSON only. Do not include markdown or prose outside the JSON object.
