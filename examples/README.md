# Checked-in learned example

`final_bundle/` is the reusable bundle from the local run
`readme_codex_e2e_20260820_185524`, learned from the 29 `example_data/` episodes.
It includes schemas, statistics, observation modules, and historical grounding
needed for closed-domain problem generation and incremental extension.
The manifest uses paths relative to the bundle directory. Duplicate per-iteration
grounding debug traces are omitted; summaries may describe the original run. `original-run/` paths in historical
summaries are provenance only and are not required to load the bundle.

`../example_problem/domain.pddl` is identical to `final_merged_domain.pddl`.
`../example_problem/problem_online.pddl` is the matched generated scene problem
from that run. The image, instruction, and object allowlist are beside it.
`feedback.json` supplies synthetic successful outcomes without observations for
bounded runtime smoke testing; it is not a recorded robot trajectory.

Generated outputs belong in `outputs/`. To use a newly learned domain, generate
a matching problem as well; predicates can differ across learning runs.
