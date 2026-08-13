# Migration notes

This repository was extracted from the research workspace as a standalone core
package. The migration intentionally includes:

- the ordered from-scratch domain-learning pipeline;
- incremental final-bundle extension;
- scene-conditioned object, initial-belief, and goal generation;
- the POMDPDDL models, parser, linter, and codegen helpers required by those
  pipelines;
- all prompts used by the included code.

It intentionally excludes experiment-specific baselines, generated domains and
problems, datasets, robot/terminal executors, result-analysis scripts, and local
model credentials.

## Package mapping

| Research workspace | Open-source package |
| --- | --- |
| `offline_model_learning/learning_pipeline` | `domain_generation/pipeline` |
| `offline_model_learning/bundle_update` | `domain_generation/extension` |
| `offline_model_learning/submodules` | `domain_generation/stages` |
| `offline_model_learning/utils` | `domain_generation/infrastructure` |
| `online_planning_agent` | `problem_generation` |
| `data_structures`, `parser`, `linter`, selected `codegen` | `core` |

The numbered pipeline stage order and generated artifact formats are preserved.
The main behavior change is first-class support for frame-only episodes through
`pre_extracted_frames.manifest_path`. One invalid extension-runner argument path
and two missing imports found during packaging were also corrected.
