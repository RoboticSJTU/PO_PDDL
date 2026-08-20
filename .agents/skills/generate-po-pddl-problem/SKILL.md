---
name: generate-po-pddl-problem
description: Generate and validate a scene-conditioned PO-PDDL problem from a learned domain, image, and instruction using a persistent pool of parallel Codex workers. Use for object extraction, initial-belief inference, goal generation, or debugging without an LLM API.
---

# Generate PO-PDDL Problem

Use the repository generator for grounding, belief construction, probability normalization, and
PDDL rendering. Codex supplies only the semantic judgments requested by its resumable protocol.

Use `po-pddl-agent` when installed. In a source checkout, prefer
`.venv/bin/po-pddl-agent`; otherwise use `PYTHONPATH=src python3 -m po_pddl.agent.cli` with the
project environment's Python. Pick one form once and use it for the whole workflow.

## Initialize

```bash
po-pddl-agent init-problem \
  <domain.pddl> \
  <scene-image-or-directory> \
  "<instruction>" \
  --output-file <problem_online.pddl> \
  --run-dir <agent-run> \
  --final-bundle-dir <final-bundle> \
  --max-workers 10 \
  --inference-strategy parallel \
  --inference-batch-size 20 \
  --location-visibility-batch-size 30 \
  --no-start
```

Use `--objects-file` for a closed task inventory that includes initially hidden objects. Use
`--initial-state-hint` only for user-provided priors and `--close-domain` only with a compatible
final bundle. `--inference-batch-size` bounds predicates or goal assignments per model task
(default 20). `--location-visibility-batch-size` sets the target capacity for grounded location
predicates judged together from the scene (default 30), while keeping every object's complete
location group in one call; `--inference-strategy parallel` exposes
independent chunks concurrently. Close-domain mode obtains object identity and type from the
grounding bundle and skips image-based object extraction.

## Persistent Worker Pool

Run the initialized workflow to completion with:

```bash
po-pddl-agent run-pool \
  --run-dir <agent-run> \
  --workers 10 \
  --tasks-per-worker 4
```

`run-pool` starts up to ten persistent `codex app-server` processes and sends prompts and local
images directly. After object extraction, initial-belief and goal inference advance concurrently;
chunks within each stage also run concurrently. Every task uses a fresh ephemeral thread,
preserving its evidence boundary while avoiding repeated Codex process startup. The command
automatically dispatches, validates, and retries tasks until the problem workflow completes.

For manual inspection or recovery, use `po-pddl-agent status`, `dispatch`, `submit`, and `reopen`.
Do not edit the generated problem manually or copy facts from a reference problem to make it pass.

## Validate

- Object declarations must represent physical task instances and valid domain types.
- Deterministic initial predicates must match visible evidence. Only learned observation-uncertain
  predicates may receive uncertainty.
- Belief factors must be coherent and normalized under the configured prior policy.
- The goal must express the instruction without accidental constraints or missing entities.
- Lint the domain/problem pair and run the terminal executor when execution validation is requested.

If semantic review identifies a bad answer, reopen it with `po-pddl-agent reopen`, provide a
domain-independent reason, and run `po-pddl-agent run-pool` again. Never copy a reference problem
into the output.
