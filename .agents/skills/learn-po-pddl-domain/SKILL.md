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

Ten workers are recommended.

## Persistent Worker Pool

Run the initialized workflow to completion with:

```bash
po-pddl-agent run-pool \
  --run-dir <agent-run> \
  --workers 10 \
  --tasks-per-worker 4
```

`run-pool` starts up to ten persistent `codex app-server` processes, sends prompt and media inputs
directly, and automatically advances, validates, and retries the resumable workflow. A fresh
ephemeral Codex thread is used for every task, so process reuse does not carry entities, visual
facts, or assumptions between demonstrations. `--tasks-per-worker` controls assignment balancing;
the default of four works well for mixed image and text tasks. Use `--task-timeout-seconds` for
unusually slow visual requests.

For manual inspection or recovery, use `po-pddl-agent status` and `po-pddl-agent dispatch`, then
submit a corrected task with `po-pddl-agent submit`. Normal generation should use `run-pool` rather
than spawning nested agents or one `codex exec` process per task.

Do not bypass the learner by editing generated artifacts. If a recurring failure is caused by code
or a generic prompt, fix the implementation, test it, and resume the workflow.

## Validate

- Confirm every demonstrated semantic action class is represented without unnecessary aliases.
- Check preconditions against reconstructed pre-states, including contextual-object constraints.
- Check success/failure effects and ensure each probabilistic form sums to one.
- Check passive, initial, and active observations against visual/state disagreement.
- Parse and lint the PDDL before accepting it.

If semantic review identifies a bad answer, reopen its task and rerun the pool:

```bash
po-pddl-agent reopen \
  --run-dir <agent-run> \
  --task-id <task-id> \
  --reason "<domain-independent semantic error>"
po-pddl-agent run-pool --run-dir <agent-run> --workers 10
```

Never repair a learned domain by copying facts from a reference domain.
