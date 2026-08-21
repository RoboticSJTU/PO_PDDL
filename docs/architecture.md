# Architecture

This document describes the public package boundaries and data flow of the
PO-PDDL implementation. For language syntax and demonstration input schemas,
see the [language specification](language-specification.md) and
[data-format guide](data-format.md).

## Design principles

The repository separates deterministic symbolic processing from model-backed
semantic inference:

- Python owns parsing, grounding, statistics, probability estimation,
  rendering, validation, artifact management, and runtime compilation.
- Language and vision models handle semantic interpretation tasks exposed by
  the pipelines.
- From-scratch learning and incremental extension share reusable stage
  implementations.
- Every domain-learning stage writes inspectable artifacts so interrupted runs
  can be audited or resumed.
- The API backend and Codex workflow execute the same pipeline logic and differ
  only in how model tasks are dispatched.

## Package layout

```text
src/po_pddl/
  config.py                 Typed public configuration
  core/
    models/                 Symbolic and factorized-belief data structures
    parser/                 PO-PDDL S-expression parsers
    linter/                 Cross-file syntax and semantic checks
    codegen/                Shared model-code generation helpers
  domain_generation/
    service.py              Public Python API
    pipeline/               Ordered from-scratch orchestration
    extension/              Incremental bundle extension
    stages/                 Reusable learning stages
    infrastructure/         Artifact, video, and model-provider utilities
  problem_generation/       Scene-conditioned belief and goal generation
  runtime/
    planning/               Semantic/bitwise models and DESPOT integration
    terminal/               Interactive execution session and feedback
  agent/                    Durable Codex task protocol and worker pool
  prompts/                  Prompt templates grouped by pipeline and stage
```

The top-level `example_data/` and `example_problem/` directories provide public
inputs for the README workflows. Generated artifacts belong under `outputs/`
and are not source modules.

## Public interfaces

### Command-line interfaces

The installed package exposes five commands:

| Command | Responsibility |
|---|---|
| `po-pddl-learn-domain` | Learn a domain from demonstration episodes. |
| `po-pddl-extend-domain` | Extend an existing final bundle with new demonstrations. |
| `po-pddl-generate-problem` | Generate an initial belief and goal for a scene and instruction. |
| `po-pddl-run-terminal` | Compile and test a domain/problem through interactive planning. |
| `po-pddl-agent` | Run the same generation workflows through Codex workers. |

CLI modules translate arguments into typed configuration and delegate to the
same services used by the Python API.

### Python services

Call service functions instead of constructing internal stage runners:

```python
from pathlib import Path

from po_pddl import DomainGenerationConfig, LLMSettings
from po_pddl.domain_generation import generate_domain

result = generate_domain(
    DomainGenerationConfig(
        input_dir=Path("example_data"),
        output_dir=Path("outputs/example_domain"),
        llm=LLMSettings(
            config_path=Path("large_model_config.private.json"),
            config_name="openai_config",
        ),
        max_workers=8,
    )
)
print(result.merged_domain_file)
```

Problem generation follows the same boundary:

```python
from pathlib import Path

from po_pddl import ProblemGenerationConfig
from po_pddl.problem_generation import generate_problem

config = ProblemGenerationConfig(
    domain_file=Path("outputs/example_domain/7_final_bundle/final_merged_domain.pddl"),
    image_path=Path("example_problem/camera_high.jpg"),
    instruction="Put the cup in the drawer.",
    output_file=Path("outputs/example_problem/problem_online.pddl"),
)
result = generate_problem(config)
print(config.resolved_output_file)
```

## Domain-generation flow

The from-scratch pipeline transforms demonstrations into a reusable domain in
an ordered sequence:

1. Normalize action annotations and prepare episode evidence.
2. Describe initial and per-step scenes from video or extracted frames.
3. Induce typed predicates and stochastic manipulation action schemas.
4. Ground episodes and reconstruct symbolic state trajectories.
5. Infer action preconditions from grounded pre-state evidence.
6. Learn passive, initial, and active observation models.
7. Merge, validate, annotate rewards, and write the final bundle.

The output directory stores stage-specific JSON, JSONL, Markdown, images, and
PDDL artifacts. `7_final_bundle/` is the stable input boundary for downstream
problem generation and incremental extension.

The extension pipeline reads an existing final bundle without modifying it. It
classifies new demonstrations against existing action schemas, refreshes
statistics for modeled actions, learns genuinely new schemas through the same
stage implementations, relearns affected observation components, and writes a
new bundle.

## Problem-generation flow

Problem generation consumes a domain, one initial scene, and a natural-language
instruction. When available, the final bundle supplies grounding evidence and
semantic metadata.

The generator:

1. resolves the allowed scene objects and their domain types;
2. evaluates grounded predicates visible in the initial scene;
3. groups uncertain alternatives into a factorized initial belief;
4. maps the instruction to a typed symbolic goal; and
5. renders and validates `problem_online.pddl`.

In `--close-domain` mode, object names are restricted to the supplied
`objects.txt` and known bundle types. Deterministic predicates and goal
assignments are processed in configurable batches. Location predicates are
batched without splitting the alternatives associated with one object.

## Model dispatch

### OpenAI-compatible API

API mode calls the provider configured by `LLMSettings`. Independent semantic
tasks may run concurrently up to `max_workers`; deterministic stages continue
to execute locally.

### Codex workers

Codex mode serializes model tasks into a durable run directory. A pool of
persistent `codex app-server` workers claims tasks while Python retains control
of pipeline ordering, local computation, validation, and recovery. Each task
uses an isolated Codex thread, while worker processes stay alive across tasks
to reduce startup overhead.

The run journal is an execution mechanism, not a separate learning pipeline.
Equivalent inputs and model judgments flow through the same stage code as API
mode.

## Prompt management

Prompt templates live under `src/po_pddl/prompts/`, grouped by domain or
problem pipeline and by stage. `po_pddl.prompts.catalog.PromptCatalog` indexes
the packaged Markdown templates and rejects duplicate filenames. Stages load a
canonical template by filename rather than embedding prompt copies in runner
code.

## Symbolic boundaries

The language parser and linter in `po_pddl.core` are independent of learned
task vocabulary. Generated relation schemas follow subject-reference argument
order; for example, containment is represented as `in(movable, container)`.
Problem generation and grounding use the domain declaration as the authority
for predicate arity, argument order, and object types.

Initial uncertainty is represented as disjoint factors. With the default
`prior_data_confidence=0.0`, model inference groups uncertain alternatives but
does not inject empirical frequency priors into a generated problem.

## Runtime boundary

The runtime parses and lints a domain/problem pair, grounds the symbolic model,
compiles semantic and bitwise representations, and produces DESPOT-compatible
C++ when requested.

`TerminalSession` owns planner orchestration and structured logs but does not
read directly from standard input. `ConsoleFeedback` and `ScriptedFeedback`
implement the same feedback protocol, allowing manual testing and reproducible
replay without changing planning logic.

Native DESPOT sources are located through `PO_PDDL_DESPOT_ROOT`. The build uses
the `pybind11` installation associated with the active Python interpreter so
the generated extension and interpreter ABI remain aligned.
