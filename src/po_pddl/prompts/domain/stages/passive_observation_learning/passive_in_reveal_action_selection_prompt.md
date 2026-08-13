You are selecting manipulation actions that can reveal the inside of a containable object for passive observation learning.

Rules:
- Only choose actions whose execution can plausibly make the interior of a `containable_item` visible.
- Select actions whose physical result exposes a previously occluded containable interior.
- Do not choose actions that merely move unrelated objects or only change external pose without revealing inside contents.
- Prefer precision over recall: only return action names that are genuinely good candidates for learning passive observations of `in(object, container)`.

Return JSON:
{
  "action_names": ["open_drawer"],
  "summary": "brief summary"
}
