<div align="center">

# PO-PDDL

### Learning Symbolic POMDPs from Visual Demonstrations for Robot Planning Under Uncertainty

[![Paper](https://img.shields.io/badge/arXiv-2606.15654-b31b1b.svg)](https://arxiv.org/abs/2606.15654)
[![Project Page](https://img.shields.io/badge/Project-Page-2f6f61.svg)](https://po-pddl.github.io/)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-3776ab.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Wenjing Tang, Xuanjin Jin, Yuan Liu, Renming Huang, Cewu Lu, Panpan Cai**

Official implementation accompanying our NeurIPS 2026 submission.

</div>

<p align="center">
  <img src="https://po-pddl.github.io/contexts/pipeline.png" alt="PO-PDDL learning pipeline" width="95%">
</p>

## Overview

PO-PDDL is a symbolic formulation of partially observable Markov decision
processes that retains the relational structure and model-friendly syntax of
PDDL while representing stochastic actions, observations, and belief states.
This repository implements our demonstration-driven pipeline for:

1. reconstructing latent symbolic trajectories from robot videos;
2. learning stochastic manipulation and observation models;
3. generating scene-conditioned initial beliefs and task goals; and
4. performing online belief-space planning with the learned model.

The learned domain is reusable across task instructions and initial scenes.
Intermediate artifacts are retained to support inspection, partial reruns, and
incremental domain extension.

**Links:** [Paper](https://arxiv.org/abs/2606.15654) | [Project page](https://po-pddl.github.io/) | [Prompt library](https://po-pddl.github.io/prompt-library/)

## Model Backends

All language and vision calls use the model profile selected with
`--config-name`. Two backends are supported.

### OpenAI-Compatible API (Recommended)

Create a private configuration from the provided template:

```bash
cp large_model_config.example.json large_model_config.private.json
```

Set the API key, base URL, and a text-and-image capable model under
`openai_config`. The private configuration is ignored by Git. Credentials may
also be supplied through environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Select this backend in generation commands with:

```bash
--config large_model_config.private.json --config-name openai_config
```

### Codex CLI (Supported, Not Recommended)

The Codex backend launches a local `codex exec` process for every model call.
Because the learning pipeline issues many calls, this mode is substantially
slower than direct API access and is intended primarily for compatibility and
local testing.

Install and authenticate the Codex CLI:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
codex
codex login status
```

See the [official Codex CLI documentation](https://developers.openai.com/codex/cli/)
for alternative installation and authentication methods. Select this backend
with:

```bash
--config large_model_config.private.json --config-name codex_config
```

Codex authentication is managed by the CLI; the JSON profile does not provide
an API key in this mode.

## Installation

Python 3.10 or later is required. We recommend installing the project in a
clean Conda environment:

```bash
conda create -n po_pddl python=3.10 -y
conda activate po_pddl
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Video processing requires FFmpeg. The interactive planner additionally
requires a C++ toolchain, CMake, and a DESPOT source checkout:

```bash
sudo apt-get install build-essential cmake ffmpeg
```

Verify the installation:

```bash
pytest -q
ruff check src tests
po-pddl-learn-domain --help
po-pddl-extend-domain --help
po-pddl-generate-problem --help
po-pddl-run-terminal --help
```

## Repository Structure

```text
PO_PDDL/
|-- src/po_pddl/
|   |-- domain_generation/    # From-scratch and incremental domain learning
|   |-- problem_generation/   # Initial-belief and goal generation
|   |-- runtime/              # POMDPDDL conversion and terminal execution
|   `-- prompts/              # Prompts grouped by pipeline stage
|-- example_data/             # Demonstration episodes
|-- example_problem/          # Initial scene and task specification
|-- docs/                     # Architecture and data-format documentation
|-- tests/                    # Unit and regression tests
|-- config/                   # Non-secret runtime hyperparameters
`-- large_model_config.example.json
```

## Example Inputs

`example_data/` contains 29 demonstration episodes. Each episode provides an
`annotations.json` trajectory and synchronized `video_cam_high.mp4` and
`video_right.mp4` recordings. Frame-only and single-camera episodes are also
supported; see [the data-format specification](docs/data-format.md).

`example_problem/` contains the inputs for a scene-conditioned task:

- `camera_high.jpg`: initial scene image;
- `instruction.txt`: natural-language task instruction;
- `objects.txt`: allowed object names;
- `domain.pddl`: reference domain for an isolated problem-generation test;
- `executor_task_mapping.json`: optional robot-executor mapping metadata.

The generated `problem_online.pddl` is intentionally excluded from the example
inputs.

## Domain Generation

The from-scratch pipeline executes the full ordered workflow for scene
interpretation, symbolic trajectory construction, action dynamics,
preconditions, passive/init/active observations, domain assembly, and final
bundling.

```bash
po-pddl-learn-domain \
  --input-dir example_data \
  --output-dir outputs/example_domain \
  --config large_model_config.private.json \
  --config-name openai_config \
  --annotation-video-types cam_high right \
  --annotation-fps 0.5 \
  --max-workers 8 \
  --max-iterations 3
```

The principal outputs are:

```text
outputs/example_domain/6_merged_domain/final_merged_domain.pddl
outputs/example_domain/7_final_bundle/final_merged_domain.pddl
outputs/example_domain/7_final_bundle/bundle_manifest.json
```

All intermediate artifacts remain in the run directory for audit and reuse.
Use `--run-stages` to rerun selected stages while loading their prerequisites
from an existing run.

## Incremental Domain Extension

The extension pipeline accepts an existing final bundle and additional
demonstrations. It identifies already-modeled action schemas, updates their
statistics with the new evidence, and learns previously unseen schemas through
the same components used by from-scratch generation. The source bundle is
never modified.

```bash
po-pddl-extend-domain \
  --bundle-dir /path/to/existing_run/7_final_bundle \
  --input-dir /path/to/additional_demonstrations \
  --output-dir /path/to/extended_run \
  --config large_model_config.private.json \
  --config-name openai_config \
  --annotation-fps 2.0 \
  --max-workers 8 \
  --max-iterations 3
```

The command reports the location of the extended final bundle and its manifest.

## Problem Generation

Given a learned domain, an initial image, and an instruction, the problem
pipeline validates scene objects, estimates the deterministic initial state and
factorized initial belief, infers the symbolic goal, and renders a POMDPDDL
problem.

The default `batch` strategy evaluates deterministic predicates and candidate
goal assignments in set-level model calls. The alternative `parallel` strategy
evaluates individual candidates concurrently with `--max-workers`.

```bash
BUNDLE=outputs/example_domain/7_final_bundle

po-pddl-generate-problem \
  "$BUNDLE/final_merged_domain.pddl" \
  example_problem/camera_high.jpg \
  "$(cat example_problem/instruction.txt)" \
  --final-bundle-dir "$BUNDLE" \
  --config large_model_config.private.json \
  --config-name openai_config \
  --inference-strategy batch \
  --max-workers 8 \
  --close-domain \
  --output example_problem/problem_online.pddl
```

For a standalone smoke test, use `example_problem/domain.pddl` as the first
argument and omit `--final-bundle-dir` and `--close-domain`.

## Interactive Planning

The terminal runtime converts the learned POMDPDDL model to DESPOT-compatible
C++, requests an action from the planner, and asks the operator to select the
observed outcome and observation literals. The posterior belief becomes the
initial belief for the next planning step.

Set `PO_PDDL_DESPOT_ROOT` to a directory that contains the `despot/` source
folder:

```bash
export PO_PDDL_DESPOT_ROOT=/path/to/despot-parent
```

Run the generated example problem:

```bash
BUNDLE=outputs/example_domain/7_final_bundle

po-pddl-run-terminal \
  "$BUNDLE/final_merged_domain.pddl" \
  example_problem/problem_online.pddl \
  --output-dir outputs/example_terminal_run \
  --max-steps 30 \
  --belief-update-log \
  --force-recompile-despot
```

At each step, enter the integer ID of the realized action outcome. When
observation choices are shown, enter comma-separated IDs or press Enter for no
informative observation. For reproducible scripted execution, provide a JSON
feedback file:

```bash
po-pddl-run-terminal \
  /path/to/domain.pddl \
  /path/to/problem_online.pddl \
  --feedback-script /path/to/feedback.json \
  --output-dir outputs/scripted_run
```

Use `--validate-only` to parse and compile a symbolic model without entering
the planning loop. DESPOT sources remain necessary for conversion.

## Python API

The command-line interfaces wrap typed Python services. For example:

```python
from pathlib import Path

from po_pddl import DomainGenerationConfig, LLMSettings
from po_pddl.domain_generation import generate_domain

result = generate_domain(
    DomainGenerationConfig(
        input_dir=Path("example_data"),
        output_dir=Path("outputs/example_domain"),
        llm=LLMSettings(config_path=Path("large_model_config.private.json")),
        max_workers=8,
    )
)
print(result.merged_domain_file)
```

See [the architecture guide](docs/architecture.md) for package boundaries and
[the data-format guide](docs/data-format.md) for supported demonstration
schemas.

## Citation

If this work is useful in your research, please cite:

```bibtex
@article{tang2026popddl,
  title   = {{PO-PDDL}: Learning Symbolic {POMDPs} from Visual Demonstrations for Robot Planning Under Uncertainty},
  author  = {Tang, Wenjing and Jin, Xuanjin and Liu, Yuan and Huang, Renming and Lu, Cewu and Cai, Panpan},
  journal = {arXiv preprint arXiv:2606.15654},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
