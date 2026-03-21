# Publication Evidence (Phase 3)

This package implements the **publication evidence** slice for the density-matrix
project.

Its job is to turn the already-emitted **correctness evidence** package and
**performance evidence** package into one reviewer-facing, machine-checkable Paper 2
publication package.

The package is intentionally validation-first:

- it reads the authoritative Phase 3 paper surfaces in
  `docs/density_matrix_project/phases/phase-3/`,
- it reads the emitted correctness-evidence and performance-evidence artifact bundles under
  `benchmarks/density_matrix/artifacts/correctness_evidence/` and
  `benchmarks/density_matrix/artifacts/performance_evidence/`,
- it emits publication-evidence slice bundles under
  `benchmarks/density_matrix/artifacts/publication_evidence/`,
- and it provides a single pipeline plus focused regression tests.

## Validation slice map

The package runs validators in a fixed order, using one shared scaffolding layer
in `common.py` so later slices can consume earlier outputs directly.

`claim_package_validation.py`
: Freezes the Paper 2 main claim, supporting claims, and explicit non-claims
against the current full-paper surface.

`surface_alignment_validation.py`
: **Publication surface alignment** — checks that abstract, technical short paper,
narrative short paper, and full paper tell the same Phase 3 story at different depths.

`claim_traceability_validation.py`
: **Claim traceability** — maps major Paper 2 claim classes and section classes to
authoritative Phase 3 docs and emitted correctness-evidence / performance-evidence bundles.

`evidence_closure_validation.py`
: **Evidence closure** — enforces the Paper 2 evidence-closure rule from the emitted
correctness-evidence and performance-evidence summary bundles.

`supported_path_validation.py`
: **Supported path scope** — verifies honest supported-path, no-fallback,
bounded-planner, count, and diagnosis wording across the paper surfaces.

`publication_manifest_validation.py`
: **Publication manifest** — builds the top-level reviewer manifest that packages the
lower publication-evidence slice outputs together with the required
correctness-evidence / performance-evidence bundle references.

`future_work_boundary_validation.py`
: **Future work boundary** — keeps future-work and publication-ladder positioning
explicit and bounded.

`package_consistency_validation.py`
: **Package consistency** — final coherence guardrail for terminology, reviewer entry,
count stability, and diagnosis-grounded limitation summaries.

`validation_pipeline.py`
: Runs all publication-evidence validators in order and writes every emitted slice bundle.

`common.py`
: Shared paths, emitted-bundle references, text-matching helpers, output directories,
and small convenience utilities used by all validators.

## Outputs

Running the pipeline writes artifacts under:

`benchmarks/density_matrix/artifacts/publication_evidence/`

Current slice output directories are:

- `claim_package/`
- `publication_surface_alignment/`
- `claim_traceability/`
- `evidence_closure/`
- `supported_path_scope/`
- `publication_manifest/`
- `future_work_boundary/`
- `package_consistency/`

Each slice emits one JSON bundle that later slices can reuse directly.

## Main commands

Run the full publication-evidence pipeline:

```bash
python benchmarks/density_matrix/publication_evidence/validation_pipeline.py
```

Run a single validator:

```bash
python benchmarks/density_matrix/publication_evidence/claim_package_validation.py
python benchmarks/density_matrix/publication_evidence/package_consistency_validation.py
```

Run the focused publication-evidence regression tests:

```bash
pytest tests/partitioning/test_publication_evidence.py -q
```

## What this package depends on

Before publication evidence can pass cleanly, the following surfaces should already be in
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
  - per-slice mini-specs under `docs/density_matrix_project/phases/phase-3/task-*`
    (planner surface through publication evidence)
- the emitted correctness-evidence bundles:
  - correctness package
  - unsupported-boundary package
  - summary-consistency bundle
- the emitted performance-evidence bundles:
  - benchmark package
  - diagnosis bundle
  - sensitivity matrix bundle
  - positive-threshold bundle
  - summary-consistency bundle

## Practical notes

- This package validates the existing Paper 2 package; it does not generate the
  scientific claims itself.
- The validators are intentionally strict about claim boundary, future-work
  wording, and diagnosis-grounded performance interpretation.
- The implementation is incremental: the validator order drives behavioral
  milestones, while `common.py` and the emitted JSON bundles provide the shared
  implementation substrate.
