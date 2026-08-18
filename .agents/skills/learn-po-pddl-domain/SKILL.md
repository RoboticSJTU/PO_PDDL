---
name: learn-po-pddl-domain
description: Learn or incrementally extend a PO-PDDL domain from annotated demonstration episodes using the deterministic pipeline and a persistent pool of parallel Codex workers for language and vision judgments. Use for domain generation, extension, or debugging without an LLM API.
---

# Learn PO-PDDL Domain

Run one resumable workflow. Python owns parsing, statistics, state reconstruction, probability
estimation, rendering, and validation. Codex answers only the semantic tasks it emits.

Use `po-pddl-agent` when installed. In a source checkout, prefer
`.venv/bin/po-pddl-agent`; otherwise use `PYTHONPATH=src python3 -m po_pddl.agent.cli` with the
project environment's Python. Pick one form once and use it for the whole workflow.

## Initialize

```bash
po-pddl-agent init-domain \
  --input-dir <episodes> \
  --output-dir <pipeline-output> \
  --run-dir <agent-run> \
  --max-workers 10 \
  --no-start
```

For extension, use:

```bash
po-pddl-agent init-extension \
  --bundle-dir <existing-final-bundle> \
  --input-dir <new-episodes> \
  --output-dir <pipeline-output> \
  --run-dir <agent-run> \
  --max-workers 10 \
  --no-start
```

Ten workers are recommended. The parent occupies one additional Codex thread, so configure the
session for 11 threads.

## Persistent Worker Pool

Create at most ten worker slots once and reuse their agent IDs for the entire workflow. Never spawn
a fresh subagent per task. Obtain balanced, bounded assignments with:

```bash
po-pddl-agent dispatch \
  --run-dir <agent-run> \
  --workers 10 \
  --tasks-per-worker 4
```

`dispatch` advances only when no responses are pending. Otherwise it writes one compact manifest
per active worker slot, balancing prompt and media cost to reduce stragglers.

1. Lazily spawn workers for assignment indexes without an agent ID. Workers must not spawn agents.
2. Give a worker only its assignment manifest path and this protocol. For every listed task, it
   independently reads `prompt_file`, `request_file`, optional `validation_file`, and every
   task-local media path; produces only the requested schema; then submits it directly:

```bash
po-pddl-agent submit --run-dir <agent-run> --task-id <task-id> --response '<response>'
```

3. Between tasks, reset task-specific assumptions. Session reuse must never copy entities, facts,
   visual judgments, or responses from another task. Reference outputs are prohibited. Workers may
   write submission scratch files but must not edit pipeline artifacts.
4. Workers return only submitted task IDs and failures to the parent, never response bodies. Wait
   for every assigned worker, then call `dispatch` again.
5. Reuse each slot: resume its existing agent and send the next manifest. Leave unused slots idle;
   replace a slot only if its agent is irrecoverably unavailable. Close the pool only after status
   is `complete`.

The four-task cap amortizes agent turns while limiting cross-task context and VLM memory pressure.
For unusually large videos, lower it. When validation reopens a task, its manifest includes
`validation_file`; revise only that task.

Do not bypass the learner by editing generated artifacts. If a recurring failure is caused by code
or a generic prompt, fix the implementation, test it, and resume the workflow.

## Validate

- Confirm every demonstrated semantic action class is represented without unnecessary aliases.
- Check preconditions against reconstructed pre-states, including contextual-object constraints.
- Check success/failure effects and ensure each probabilistic form sums to one.
- Check passive, initial, and active observations against visual/state disagreement.
- Parse and lint the PDDL before accepting it.

If semantic review identifies a bad answer, reopen its task and let the pool revise it:

```bash
po-pddl-agent reopen \
  --run-dir <agent-run> \
  --task-id <task-id> \
  --reason "<domain-independent semantic error>"
```

Never repair a learned domain by copying facts from a reference domain.
