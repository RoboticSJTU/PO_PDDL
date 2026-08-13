Extract the task-relevant visible objects from the supplied scene image.

The JSON payload provides the domain, instruction, image-layout note, learned manipulation records,
and `known_object_names`.

Rules:
- Return JSON only and follow `image_input_note` when multiple views are concatenated.
- Include every genuinely visible known object and any additional visible object that can participate
  in a domain action, predicate, or goal. `known_object_names` is a naming aid, not a closed list.
- Match category/type first, then stable intrinsic attributes. Color or a supplied name alone is not
  enough to override incompatible shape or category evidence.
- Map each physical instance to at most one symbolic object. Use the most specific matching known
  name; otherwise create a concise snake_case name from stable intrinsic attributes.
- Do not name hidden entities, background clutter, reflections, markings, spatial regions, or other
  landmarks unless the domain explicitly models them as task objects.
- Do not encode transient position such as left, right, front, or top in a new object name.
- Reuse a domain type when visually supported; otherwise use `object`.

Return exactly:
{
  "objects": [
    {
      "name": "object_name",
      "type_name": "domain_type",
      "justification": "brief image-grounded reason"
    }
  ]
}
