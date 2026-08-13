Classify the episode's grounded predicates in the initial state.

Evidence order:
1. initial visual description and accepted visible facts;
2. symbolic consistency and object inventory;
3. backward reasoning from the earliest action involving an initially hidden entity.

Rules:
- Classify every supplied `grounded_predicate` exactly once as true or false; do not invent facts.
- Separate location/support/containment predicates into `location_predicate_classifications` and all
  others into `other_predicate_classifications`.
- Give each task-relevant movable entity one consistent initial location when candidate predicates
  and trajectory evidence support it; mutually exclusive alternatives cannot both be true.
- For hidden entities, the earliest access/acquisition action and preceding reveal actions may imply
  the prior location. This is backward evidence, not permission to infer unrelated facts.
- Apply the same consistency to stable positions/states of fixed entities when represented.
- Follow predicate signatures exactly, including argument order.
- Infer persistent selection features only when they are the minimal explanation of instruction-led
  choices across the trajectory.
- Prefer a conservative complete assignment over speculative positives. `review_guidance` is soft.

Output shape when grounded predicates are supplied:
{
  "location_predicate_classifications": [
    {"fact": "relation(entity)", "value": "true", "confidence": "high", "justification": "brief evidence"}
  ],
  "other_predicate_classifications": [
    {"fact": "state(entity)", "value": "false", "confidence": "medium", "justification": "brief evidence"}
  ]
}

If no grounded predicates are supplied, return the legacy shape:
{
  "init_facts": [
    {"fact": "relation(entity)", "confidence": "medium", "justification": "brief evidence"}
  ]
}
