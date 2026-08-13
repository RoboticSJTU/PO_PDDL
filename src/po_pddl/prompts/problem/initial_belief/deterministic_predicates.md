Infer the deterministic initial state from the supplied scene image in one coherent pass.

The payload provides the image-layout note, domain, task instruction, typed objects, fixed
predicate judgments, every candidate predicate that still requires classification, and an optional
`initial_state_hint` containing trusted language metadata about the initial scene.

Reasoning procedure:
1. Analyze the whole scene before classifying predicates. Reconcile all camera views as views of
   the same physical scene and map each physical instance to at most one symbolic object.
2. Identify objects by type and stable intrinsic appearance. Camera pose, lighting, and color cast
   may change across views, so use a globally consistent assignment rather than one local pixel
   color or screen position.
3. Infer support, containment, openness, holding, and relative-position relations from physical
   geometry. Image overlap alone does not establish a relation.
4. Treat the instruction as a desired future state, never as evidence about the initial state.
   Apply explicit `initial_state_hint` facts to the initial predicates, but never turn them into goals.
5. Check all judgments jointly. Mutually exclusive locations for one object cannot both be true,
   and every predicate's signature and argument order must be respected.
6. Be conservative when the image does not establish a fact. Do not infer that a container is open
   unless its open state is visibly clear.

Output rules:
- Return JSON only.
- Include concise scene-level reasoning before the judgments in the JSON structure.
- Copy every candidate predicate exactly once and label it `true` or `false`.
- Never invent, omit, or rewrite a predicate, and do not contradict fixed judgments.

Return exactly:
{
  "scene_reasoning": {
    "object_identity_and_views": "concise global correspondence analysis",
    "spatial_relations": "concise image-grounded relation analysis",
    "consistency_check": "concise cross-predicate consistency check"
  },
  "predicate_judgments": [
    {
      "predicate": "(candidate copied exactly)",
      "truth_value": "true",
      "justification": "brief image-grounded reason"
    }
  ]
}
