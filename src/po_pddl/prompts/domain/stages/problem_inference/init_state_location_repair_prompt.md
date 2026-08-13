Repair only missing or contradictory initial location/support/containment facts.

Rules:
- Check every movable entity and represented stable position of fixed entities.
- Use the initial description first; for hidden entities, use the earliest direct interaction and
  preceding access actions as backward evidence.
- Choose one consistent location among mutually exclusive candidates.
- Copy facts only from `grounded_predicates` or `current_true_init_facts`; follow signatures and
  argument order exactly.
- Make the smallest repair. Return empty changes when none is needed.

Return JSON only:
{
  "should_repair": false,
  "repair_summary": "",
  "init_facts_add": [],
  "init_facts_remove": []
}
