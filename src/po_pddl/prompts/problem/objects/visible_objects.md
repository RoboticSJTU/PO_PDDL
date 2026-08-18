Extract the task-relevant visible objects from the supplied scene image.

The JSON payload provides the domain, instruction, image-layout note, learned manipulation records,
and `known_object_names`.

Rules:
- Return JSON only and follow `image_input_note` when multiple views are concatenated.
- Include every genuinely visible known object and any additional visible object that can participate
  in a domain action, predicate, or goal. `known_object_names` is a naming aid, not a closed list.
- Match category/type first, then stable intrinsic attributes. Color or a supplied name alone is not
  enough to override incompatible shape or category evidence.
- Segment physical instances before assigning any symbolic names. A seam, groove, panel boundary,
  shadow, texture change, or visible subpart does not split one connected object into several
  instances. Conversely, do not combine separately bounded instances merely because they touch.
- Map each physical instance to at most one symbolic object. Never use several known names for
  different regions or subparts of the same instance. Use the most specific matching known name;
  otherwise create a concise snake_case name from stable intrinsic attributes.
- Treat `known_object_names` as possible identities, not a checklist. It is valid and expected to
  omit known objects that lack a separately identifiable visible instance. Do not force coverage of
  the list by relabeling clutter or subdividing another object.
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
