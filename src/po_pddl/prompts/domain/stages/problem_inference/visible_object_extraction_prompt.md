Extract objects and facts directly visible in the initial scene description.

Rules:
- Use only `step0_observation_text` and domain signatures. Return one JSON object.
- Do not infer hidden or future-revealed entities here.
- If `allowed_object_names` is supplied, use only exact names from it.
- Deduplicate repeated mentions/views unless stable evidence requires distinct simultaneous entities.
- Use declared domain types when supported; otherwise use the least specific permitted type.
- `visible_facts` contains only directly supported legal symbolic facts.
- `review_guidance` is a hint, never evidence.

Output shape:
{
  "visible_objects": [
    {
      "name": "entity_name",
      "type_name": "entity_type",
      "supporting_observation": "brief evidence",
      "visible_facts": ["predicate(entity_name)"]
    }
  ],
  "visible_facts": ["predicate(entity_name)"]
}
