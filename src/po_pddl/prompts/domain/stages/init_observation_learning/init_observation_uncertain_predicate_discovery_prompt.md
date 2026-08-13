You are identifying which state predicates have uncertain visual observations in a demonstration dataset.

You are given:
- a predicate inventory with parameter types and semantic comments
- task instructions and initial scene descriptions from multiple episodes
- representative initial-scene images

Task:
- Independently identify predicate schemas whose truth is meant to be visually perceived, but whose value may be missed or misread in the supplied initial views.
- Select from the supplied predicate inventory only.
- Focus on semantic properties that require visual inspection and can differ between the real ground-truth state and what the initial image reports.
- Do not select ordinary geometric relations merely because camera perspective could make localization difficult.
- Do not select predicates that are only internal bookkeeping or robot-control state.
- Do not assume a predicate is uncertain from a preassigned category or naming convention. Infer its observation role from the tasks, images, descriptions, and semantic comment.
- A predicate may be selected even if only some grounded instances are uncertain.

Return JSON only:
{
  "observation_uncertain_predicates": [
    {
      "predicate_name": "is_fragile",
      "rationale": "The property is task-relevant and must be visually inspected, but may not be visible in the initial view."
    }
  ],
  "summary": "short summary"
}
