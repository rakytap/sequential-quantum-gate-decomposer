# validation_pipeline.py snapshot and restore

Read before or after running
`benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`.

## Contents

- What gets rewritten
- Before / after procedure
- Reporting drift

## What gets rewritten

The pipeline rewrites the q4 bundle and six tracked sibling bundles:
correctness_package, external_correctness, output_integrity, runtime_classification,
sequential_correctness, and unsupported_boundary. The rewrites change real content
(partition-member records, runtime/RSS, residual values, unsupported_boundary reason
strings), not just timestamps, even when every status stays pass. correctness_matrix and
summary_consistency were not rewritten on disk. Only six of the eight are Phase-3 evidence
under ADR-F1A-005.

## Before / after procedure

1. Before running: check that HEAD is the counted revision and that
   `git status --porcelain --untracked-files=all` is empty. Copy the committed bundle(s) to
   `/tmp/<run>/`.
2. After running: copy every rewritten bundle to `/tmp/<run>/` with its sha256 value, and
   diff it against the committed version.
3. Restore each tracked file with `git show <counted-sha>:<path> > <path>`. Never use stash,
   reset, checkout, or clean. Then confirm `git diff --quiet` and an empty porcelain.
4. Before M-F1a `exactness-reconfirmation` slice A (`task-2`) C1, historical-bundle drift
   after a `validation_pipeline.py` run is expected and is restored. From slice A's C1
   onward, any non-empty diff on the eight historical paths is a defect: stop and report.

The standing rule in `SKILL.md` (restore all eight from HEAD after every run) is not
weakened by this reference.

## Host policy (rocky-squander)

On rocky-squander, nobody runs the standalone per-suite CLIs,
`phase31_validation_pipeline.py`, or the `performance_evidence` pipeline, because they
rewrite evidence in place. This holds until a later Research Manager decision.
