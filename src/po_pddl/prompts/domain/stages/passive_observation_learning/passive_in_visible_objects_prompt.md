You are reading a post-action scene description to identify which objects are visibly inside a specific container.

Rules:
- Only choose from `candidate_object_names`.
- Only return objects that are visibly inside the specified container in this step.
- Ignore objects that are visible somewhere else but are not clearly visible in the container.
- If no candidate object is visibly inside the container, return an empty list.
- Be conservative: if the description is ambiguous, do not include the object.

Return JSON:
{
  "visible_object_names": ["red_block", "blue_ball"],
  "summary": "brief summary"
}
