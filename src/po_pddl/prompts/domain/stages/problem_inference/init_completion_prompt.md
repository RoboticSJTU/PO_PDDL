Add only initial facts omitted by the first-pass inference.

Rules:
- Return facts not already in `existing_true_init_facts`; never regenerate the whole init.
- Use initial evidence, earliest grounded actions/states, action preconditions, and the instruction.
- Add a resource/control fact when an earliest successful action requires it and no evidence
  contradicts it.
- Add a missing initial location only when the earliest interaction and preceding access sequence
  directly imply it. Follow predicate signatures and argument order exactly.
- A fact becoming true later is not evidence it was initially true.
- Persistent feature facts may be added only as the minimal explanation of demonstrated
  instruction-conditioned object selection.
- Prefer the smallest consistent completion. `review_guidance` is soft.

Return JSON only:
{
  "additional_init_facts": [
    {"fact": "predicate(entity)", "confidence": "high", "justification": "brief evidence"}
  ]
}
