Select the minimal necessary preconditions for one action schema from grounded demonstrations.

Return JSON only:
{
  "selected_precondition_literals": ["literal copied exactly from the candidates"],
  "selection_summary": "brief causal justification"
}

Rules:
- Output only a subset of the supplied universally supported candidate literals. Do not invent,
  rewrite, or ground literals, and do not introduce variables.
- A literal being true before every observed execution is evidence, not proof, that it is required.
- Keep a literal only when violating it would make the action physically impossible, violate a
  required resource/state, contradict the action's meaning, or break a demonstrated direct
  ordering/clearance dependency.
- Remove scene-specific coincidences, properties of unrelated objects, effects of earlier tasks,
  and facts that merely identify the particular demonstration instance.
- Distinguish an immediate prerequisite from history: an earlier action changing a fact only makes
  that fact a precondition when the change directly enables this action. Do not propagate every
  earlier deletion to all later actions.
- Positive relations may identify a required held object, source, support, container, or current
  state. A zero-arity resource predicate may be required when the action needs that resource free.
- Retain a static property of an action parameter when it distinguishes the schema's applicability
  from an otherwise equivalent action variant. Do not discard it merely because the property never
  changes during a trajectory. Conversely, omit static properties unrelated to physical or semantic
  applicability even when they are correlated with the demonstrations.
- The state of an action's reference entity is causal when it changes the geometry or availability
  of the source, destination, support, access path, or workspace. For example, open/closed,
  locked/unlocked, attached/detached, or powered/unpowered state should be retained when it is
  universally supported and the action's physical interaction relies on that configuration, unless
  demonstrations show the same action succeeding in the alternative state.
- For a continuous-contact transfer across multiple source/destination supports, evaluate every
  referenced support. Retain each universally supported configuration state needed to make its
  contact surface physically usable; do not drop it merely because the state word is absent from
  the action name.
- Negated candidates with dependent variables represent possible quantified clearance constraints.
  For these candidates, `eligible_example_count` counts demonstrations in which an earlier observed
  transition removed a matching grounded fact; it is causal-order evidence rather than ordinary
  state frequency. Select the candidate when it is universally supported within those eligible
  demonstrations and its relation is anchored to this action's manipulated object, destination,
  access path, or required workspace. Reject unanchored scene-wide absences.
- Do not require both a state and redundant aliases/complements of that state.
- Do not use an effect's intended result as a precondition unless repeat application is genuinely
  invalid.
- Prefer the smallest causally sufficient set. When uncertain, omit a correlation rather than
  over-constrain the domain.

The payload contains the domain, action schema, support statistics, and grounded state-before
examples. Use the action semantics and examples together; do not select by support alone.
