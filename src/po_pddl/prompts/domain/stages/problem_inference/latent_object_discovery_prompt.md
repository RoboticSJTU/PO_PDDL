Infer the minimum additional object inventory required to explain the full episode.

Rules:
- Return one JSON object. Use visible objects plus prefix-time action/observation evidence.
- A reveal changes visibility, not existence. A successful action may imply an earlier hidden entity.
- Introduce multiple same-type entities only when they must coexist or count/history cannot be
  explained by one entity moving over time. Deduplicate ambiguous repeated descriptions.
- Explain an observation at step t using only actions up to t; never use later insertions to explain
  an earlier count.
- If `allowed_object_names` is supplied, use only those names.
- Names contain stable identity/type information only, never location, visibility, or history. Use
  neutral indices when distinct entities lack stable attributes.
- Keep type assignments consistent and do not substitute an entity of another type.
- `review_guidance` is a soft hint and requires trajectory support.

Output shape:
{
  "additional_objects": [
    {
      "name": "entity_2",
      "type_name": "entity_type",
      "evidence": ["brief prefix-time evidence"]
    }
  ]
}
