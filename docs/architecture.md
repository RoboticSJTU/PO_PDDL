# Architecture

The package is split by responsibility rather than by experiment history.

```text
src/po_pddl/
  config.py                 typed public configuration
  core/                     POMDPDDL models, parser, linter, codegen helpers
  domain_generation/
    service.py              public domain APIs
    pipeline/               ordered from-scratch orchestration
    extension/              incremental bundle update orchestration
    stages/                 reusable learning stages
    infrastructure/         artifact, video, and model-provider utilities
  problem_generation/       scene-to-problem generators and public service
  runtime/
    planning/                semantic, bitwise, and DESPOT compatibility core
    terminal/                feedback providers, session loop, and CLI
  prompts/
    domain/                 prompts grouped by learning stage
    problem/                goal, initial-belief, and object prompts
```

## Public API

Use the service functions instead of constructing internal runners or agents:

```python
from pathlib import Path

from po_pddl import DomainGenerationConfig, LLMSettings
from po_pddl.domain_generation import generate_domain

result = generate_domain(
    DomainGenerationConfig(
        input_dir=Path("data/demos"),
        output_dir=Path("outputs/domain"),
        llm=LLMSettings(config_path=Path("large_model_config.private.json")),
        max_workers=8,
    )
)
print(result.merged_domain_file)
```

Problem generation follows the same pattern:

```python
from po_pddl import ProblemGenerationConfig
from po_pddl.problem_generation import generate_problem

result = generate_problem(
    ProblemGenerationConfig(
        domain_file=Path("outputs/domain/7_final_bundle/final_merged_domain.pddl"),
        image_path=Path("scene.jpg"),
        instruction="Put the cup in the drawer.",
    )
)
```

The ordered learning stages and their artifact schemas remain compatible with
the research pipeline. Refactoring is limited to package boundaries, prompt
location, configuration, and public orchestration; it does not reorder stages.

## Symbolic conventions

Relations use subject-reference argument order. Containment is always
`in(movable_item, containable_item)` in predicates, effects, grounded states,
observables, and generated problems. Grounding normalization rejects the legacy
container-first representation rather than allowing both forms to coexist.

Problem generation treats a neighboring `objects.txt` as a strict object
allowlist when `--close-domain` is enabled. Initial uncertainty uses uniform
factor probabilities by default (`--prior-data-confidence 0.0`); the LLM/VLM
groups uncertain variables but does not inject historical priors. Deterministic
predicates and goal assignments are evaluated in set-level calls by default.
The optional `parallel` strategy evaluates each item independently, with
concurrency limited by `--max-workers`.

## Prompt management

`po_pddl.prompts.prompt_catalog` validates that every Markdown prompt has a
unique filename at import time. Internal stages load prompts by filename, so a
template has one canonical copy and cannot silently diverge between pipelines.

## Runtime boundaries

`TerminalSession` owns planner orchestration and structured logging but knows
nothing about `input()`. `ConsoleFeedback` and `ScriptedFeedback` implement the
same small protocol, so experiments can replay exact effect and observation
sequences without changing the executor. The migrated planning compatibility
layer remains separate because it includes model-code generation and native
DESPOT integration; the higher-level terminal code is independently testable
with a fake planner.

The native dependency is resolved through `PO_PDDL_DESPOT_ROOT`. This avoids the
absolute repository paths used by the research script and keeps deployment
configuration outside source code.
