You are judging one grounded observable from the initial scene image for an online POMDPDDL planning problem with an observation module.

You will receive:
- one image
- a JSON payload containing:
  - `domain_summary`
  - `instruction`
  - optional `initial_state_hint`
  - `image_input_note`
  - `objects`
  - `manipulation_records`
  - `target_observable`
  - `target_predicate`

The `image_input_note` explains whether the provided image is a single view or a vertical top-to-bottom concatenation of multiple camera views. Follow that note carefully when interpreting the image.

Task:
- Decide whether the current image supports the observable `target_observable` as true or false at planning start.
- This is an observation-space judgment, not a hidden-state judgment.
- The initial-state hint is not visual evidence for an observable.
- If the image does not clearly support the observable as true, return `false`.

Important rules:
- Return one JSON object only.
- Do not answer "unknown".
- Prefer `false` when the image is ambiguous, weak, or does not clearly support the observable.
- Do not invent unseen objects or future observations.

Return JSON with this exact shape:
{
  "observable_value": "true",
  "justification": "Short explanation grounded in the image and the observable wording."
}
