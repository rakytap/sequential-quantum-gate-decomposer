# Phase 3 Task 8 Publication Evidence

This package implements Phase 3 Task 8 for the density-matrix project.

Its job is to turn the already-emitted Phase 3 Task 6 correctness package and
Task 7 benchmark package into one reviewer-facing, machine-checkable Paper 2
publication package.

The package is intentionally validation-first:

- it reads the authoritative Phase 3 paper surfaces in
  `docs/density_matrix_project/phases/phase-3/`,
- it reads the emitted Task 6 and Task 7 artifact bundles under
  `benchmarks/density_matrix/artifacts/correctness_evidence/` and
  `benchmarks/density_matrix/artifacts/performance_evidence/`,
- it emits Task 8 story bundles under
  `benchmarks/density_matrix/artifacts/publication_evidence/`,
- and it provides a single pipeline plus focused regression tests.

## Story Map

The package follows the Task 8 story order, but uses one shared scaffolding
layer in `common.py` so later stories can consume earlier outputs directly.

`claim_package_validation.py`
: Story 1. Freezes the Paper 2 main claim, supporting claims, and explicit
non-claims against the current full-paper surface.

`surface_alignment_validation.py`
: Story 2. Checks that abstract, technical short paper, narrative short paper,
and full paper tell the same Phase 3 story at different depths.

`claim_traceability_validation.py`
: Story 3. Maps major Paper 2 claim classes and section classes to authoritative
Phase 3 docs and emitted Task 6 / Task 7 bundles.

`evidence_closure_validation.py`
: Story 4. Enforces the Paper 2 evidence-closure rule from the emitted Task 6
and Task 7 summary bundles.

`supported_path_validation.py`
: Story 5. Verifies honest supported-path, no-fallback, bounded-planner, count,
and diagnosis wording across the paper surfaces.

`publication_manifest_validation.py`
: Story 6. Builds the top-level reviewer manifest that packages the lower Task 8
story outputs together with the required Task 6 / Task 7 bundle references.

`future_work_boundary_validation.py`
: Story 7. Keeps future-work and publication-ladder positioning explicit and
bounded.

`package_consistency_validation.py`
: Story 8. Final coherence guardrail for terminology, reviewer entry, count
stability, and diagnosis-grounded limitation summaries.

`publication_evidence_validation_pipeline.py`
: Runs Stories 1 through 8 in order and writes all emitted Task 8 bundles.

`common.py`
: Shared paths, emitted-bundle references, text-matching helpers, output
directories, and small convenience utilities used by all story validators.

## Outputs

Running the pipeline writes artifacts under:

`benchmarks/density_matrix/artifacts/publication_evidence/`

Current story output directories are:

- `claim_package/`
- `surface_alignment/`
- `claim_traceability/`
- `evidence_closure/`
- `supported_path/`
- `manifest/`
- `future_work/`
- `package_consistency/`

Each story emits one JSON bundle that later stories can reuse directly.

## Main Commands

Run the full Task 8 pipeline:

```bash
python benchmarks/density_matrix/publication_evidence/validation_pipeline.py
```

Run a single story validator:

```bash
python benchmarks/density_matrix/publication_evidence/claim_package_validation.py
python benchmarks/density_matrix/publication_evidence/package_consistency_validation.py
```

Run the focused Task 8 regression tests:

```bash
pytest tests/partitioning/test_publication_evidence.py -q
```

## What This Package Depends On

Before Task 8 can pass cleanly, the following surfaces should already be in
place:

- the Phase 3 paper docs:
  - `ABSTRACT_PHASE_3.md`
  - `SHORT_PAPER_PHASE_3.md`
  - `SHORT_PAPER_NARRATIVE.md`
  - `PAPER_PHASE_3.md`
- the Phase 3 contract docs:
  - `DETAILED_PLANNING_PHASE_3.md`
  - `ADRs_PHASE_3.md`
  - `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`
  - `TASK_1_MINI_SPEC.md` through `TASK_8_MINI_SPEC.md`
- the emitted Task 6 bundles:
  - correctness package
  - unsupported-boundary package
  - summary-consistency bundle
- the emitted Task 7 bundles:
  - benchmark package
  - diagnosis bundle
  - sensitivity matrix bundle
  - positive-threshold bundle
  - summary-consistency bundle

## Practical Notes

- This package validates the existing Paper 2 package; it does not generate the
  scientific claims itself.
- The validators are intentionally strict about claim boundary, future-work
  wording, and diagnosis-grounded performance interpretation.
- The final Task 8 implementation is incremental: the story order drives the
  behavioral milestones, while `common.py` and the emitted JSON bundles provide
  the shared implementation substrate.
