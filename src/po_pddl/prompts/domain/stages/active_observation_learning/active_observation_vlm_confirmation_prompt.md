You are confirming the values of active observation targets from the original images.

You are given:
- the instruction
- the active observation action text
- the previous and current scene descriptions
- the filtered current symbolic state
- the candidate grounded truth facts with explicit true/false values
- a list of target observables for this step
- predicate comments
- the original step images

Task:
- For every target observable, decide whether the observation value shown by the images is true or false.
- Use the images as the final authority.
- Be conservative when the visual evidence is weak.
- Do not infer new targets. Only confirm the provided targets.
- If a provided target is clearly not something this action would normally inspect according to commonsense, only confirm it as observed when the visual evidence is very explicit.

Return JSON only:
{
  "confirmed_observations": [
    {
      "predicate_name": "is_fragile",
      "grounded_literal": "is_fragile(tool_a)",
      "ground_truth_value": true,
      "observed_value": true,
      "rationale": "The inspected property is visibly present."
    }
  ],
  "summary": "short summary"
}
