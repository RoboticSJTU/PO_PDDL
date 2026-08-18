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
  --no-start
```

Use `--objects-file` for a closed task inventory that includes initially hidden objects. Use
`--initial-state-hint` only for user-provided priors and `--close-domain` only with a compatible
final bundle. `--inference-batch-size` bounds predicates or goal assignments per model task
(default 20); `--inference-strategy parallel` exposes independent chunks concurrently.

## Persistent Worker Pool

Create worker slots lazily, up to ten, and retain each agent ID until completion. Never spawn one
agent per task. Request balanced assignments with:

```bash
po-pddl-agent dispatch \
  --run-dir <agent-run> \
  --workers 10 \
  --tasks-per-worker 4
```

1. Spawn only missing assignment indexes. Give each worker its manifest path and forbid nested
   agents.
2. For every listed task, independently read the prompt, request, optional validation feedback,
   and all media. Preserve exact predicate names, argument order, object types, and requested
   schema. Submit directly instead of returning the response body to the parent:

```bash
po-pddl-agent submit --run-dir <agent-run> --task-id <task-id> --response '<response>'
```

3. Treat each task as a fresh evidence boundary. Do not carry visual facts, object assumptions,
   goal decisions, or answer templates from earlier tasks. Never consult a reference problem while
   answering.
4. Workers report only submitted IDs and failures. After all finish, call `dispatch` again. Resume
   the same agent for that worker index and send its next manifest; replace it only after an
   unrecoverable failure. Leave unused slots idle and close the pool only after completion.

The four-task cap amortizes turns without allowing worker context to grow unchecked. If validation
reopens a task, follow its `validation_file` and revise only that task. Do not edit the generated
problem manually to make it pass.

## Validate

- Object declarations must represent physical task instances and valid domain types.
- Deterministic initial predicates must match visible evidence. Only learned observation-uncertain
  predicates may receive uncertainty.
- Belief factors must be coherent and normalized under the configured prior policy.
- The goal must express the instruction without accidental constraints or missing entities.
- Lint the domain/problem pair and run the terminal executor when execution validation is requested.

If semantic review identifies a bad answer, reopen it with `po-pddl-agent reopen`, provide a
domain-independent reason, and let the worker pool revise it. Never copy a reference problem into
the output.
