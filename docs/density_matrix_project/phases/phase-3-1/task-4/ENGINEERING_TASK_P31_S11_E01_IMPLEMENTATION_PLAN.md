# Engineering Task P31-S11-E01 Implementation Plan

## Engineering Task Being Implemented

`Engineering Task P31-S11-E01: Expand the current hybrid pilot harness into the full frozen counted matrix`

This is a Layer 4 file-level implementation plan for the first engineering task
under Story `P31-S11` from
`../FIFTH_VERTICAL_SLICE_FULL_COUNTED_PERFORMANCE_MATRIX_STORIES_AND_ENGINEERING_TASKS.md`.

It turns the post-fourth-slice Task 4 wording into a concrete plan against the
current Phase 3.1 hybrid runtime and performance/evidence surfaces in:

- `task-4/TASK_4_MINI_SPEC.md`,
- `tests/partitioning/evidence/test_phase31_hybrid_pilot_validation.py`,
- `benchmarks/density_matrix/performance_evidence/case_selection.py`,
- `benchmarks/density_matrix/performance_evidence/records.py`,
- `benchmarks/density_matrix/performance_evidence/common.py`,
- `benchmarks/density_matrix/performance_evidence/phase31_hybrid_pilot_validation.py`,
- and, for regression-boundary only,
  `tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py`,
  `benchmarks/density_matrix/evidence_core.py`,
  `docs/density_matrix_project/phases/phase-3-1/PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md`.

## Scope

This engineering task promotes the single frozen hybrid pilot into the **full
counted 26-row performance matrix** for the frozen Phase 3.1 v1 slice:

- keep the counted families, seeds, patterns, and qubit bands exactly as frozen
  in `P31-ADR-010`,
- reuse the explicit hybrid whole-workload runtime
  `execute_partitioned_density_channel_native_hybrid(...)`,
- emit one performance row for every counted case with the required baseline
  trio:
  - sequential density reference,
  - Phase 3 partitioned+fused,
  - Phase 3.1 hybrid channel-native,
- preserve the scalar-only counted-build metadata from `P31-C-09`,
- carry forward route-coverage information on every counted row,
- preserve the current pilot row as one member of the matrix rather than a
  separate special-case artifact,
- and keep the matrix output machine-reviewable with stable case IDs and
  deterministic inventory ordering.

Out of scope for this engineering task:

- deriving the matrix-wide `break_even_table` / `justification_map`,
- publication wording or review-state upgrades,
- Task 6 host-acceleration variants,
- reopening the current default Phase 3 performance package before the Phase 3.1
  sibling matrix validates cleanly,
- and adding new workload families, seeds, or support-surface variants beyond
  the frozen contract.

## Current Performance-Evidence Gap To Close

The repo now has a stronger precondition for `P31-S11-E01` than it had at the
pilot stage:

- the bounded counted correctness package is green on the frozen six-row slice,
- the required external-reference slice is present,
- the pilot row already proves the hybrid runtime, route-coverage counters, and
  baseline trio can be emitted on one counted workload.

What is still missing is the **matrix-wide emission surface** that converts the
pilot from "illustrative row" to "frozen counted benchmark inventory."

### `benchmarks/density_matrix/performance_evidence/case_selection.py`

This file already contains the frozen Phase 3.1 matrix helpers:

- `build_phase31_performance_inventory_cases()`,
- `iter_phase31_performance_cases()`,
- and the frozen pilot ID helper surface.

The key gap is that the repo still treats the pilot path as the active dedicated
Phase 3.1 output while the full counted matrix remains an inventory helper only.

`P31-S11-E01` should therefore:

- keep the inventory helper as the source of truth,
- and add one **matrix-oriented builder / iterator** that is explicit about
  being the bounded Phase 3.1 counted matrix, not the old Phase 3 default
  package.

### `benchmarks/density_matrix/performance_evidence/records.py`

This file already contains the pilot-row measurement logic and decision-class
vocabulary. However, the dedicated emitted surface is still shaped around a
single pilot record rather than around the full counted matrix.

`P31-S11-E01` should therefore:

- reuse the pilot-row measurement style,
- but generalize it into a matrix-row builder that emits:
  - the baseline trio,
  - hybrid route counters,
  - build metadata,
  - correctness-bridge fields,
  - and stable counted-case metadata
    for all frozen rows.

### `tests/partitioning/evidence/test_phase31_hybrid_pilot_validation.py`

This file currently freezes:

- the pilot workload ID,
- the baseline trio,
- route counters,
- decision-class vocabulary,
- and the full counted inventory size (`26`) at the helper level.

That makes it a strong regression anchor for the pilot and for the frozen
inventory count, but it is **not yet** the matrix-closure validation surface.

`P31-S11-E01` should add a **sibling test module** that validates:

- the matrix case count,
- representative family coverage,
- control-family presence,
- and row-level field completeness across all counted rows.

## Dependencies And Assumptions

- The source-of-truth contract remains:
  - `TASK_4_MINI_SPEC.md`,
  - `../FIFTH_VERTICAL_SLICE_FULL_COUNTED_PERFORMANCE_MATRIX_STORIES_AND_ENGINEERING_TASKS.md`,
  - `../DETAILED_PLANNING_PHASE_3_1.md`,
  - `../ADRs_PHASE_3_1.md`
    (`P31-ADR-005`, `P31-ADR-006`, `P31-ADR-010`, `P31-ADR-013`,
    `P31-ADR-014`).
- `P31-S10` is assumed complete enough that counted performance rows are allowed
  to claim correctness on the frozen slice.
- The frozen counted matrix size is:
  - `24` primary-family rows,
  - `2` control-family rows,
  - total `26`.
- The counted-build policy remains:
  - `build_policy_id = "phase31_scalar_only_v1"`,
  - `build_flavor = "scalar"`,
  - `simd_enabled = false`,
  - `tbb_enabled = false`,
  - `thread_count = 1`,
  - `counted_claim_build = true`.
- The full matrix should prefer a Phase 3.1 **sibling** performance pipeline
  first, consistent with the staged migration logic already used for correctness.

## Target Files And Responsibilities

### Primary inventory surface: `benchmarks/density_matrix/performance_evidence/case_selection.py`

This file should provide one explicit bounded-matrix selector for the frozen
Phase 3.1 counted surface.

#### What this file should provide after `P31-S11-E01`

- one matrix builder that returns the `26` counted Phase 3.1 rows only,
- one case-context iterator aligned to the same inventory ordering,
- stable metadata for:
  - `claim_surface_id = "phase31_bounded_mixed_motif_v1"`,
  - `counted_phase31_case = True`,
  - `representation_primary = "kraus_bundle"`,
  - the scalar-only counted-build policy,
- and no silent reuse of the old Phase 3 default inventory entrypoints.

#### Recommended implementation constraint

Keep the current default Phase 3 performance builders stable in this task. Add a
Phase 3.1 sibling builder instead, for example:

- `build_phase31_counted_performance_cases()`,
- `build_phase31_counted_performance_case_contexts()`.

### Primary row-builder surface: `benchmarks/density_matrix/performance_evidence/records.py`

This file should grow one Phase 3.1 counted-matrix row builder that generalizes
the current pilot-row logic.

#### What this file should provide after `P31-S11-E01`

- a counted-row builder that emits for every frozen row:
  - sequential median runtime and peak RSS,
  - Phase 3 fused median runtime and peak RSS,
  - Phase 3.1 hybrid median runtime and peak RSS,
  - `runtime_class`,
  - `decision_class`,
  - route-coverage counters,
  - `counted_phase31_case`,
  - scalar-only counted-build metadata,
  - correctness-bridge fields reused from shared helpers where appropriate,
- and one list-builder for the full matrix rows.

#### Recommended implementation constraint

Do **not** derive the matrix-wide `break_even_table` in this task. It is fine if
rows carry `decision_class`; the aggregate decision artifact belongs to
`P31-S11-E02`.

### Dedicated matrix validation surface

Prefer a new narrow validation pair rather than mutating the old default Phase 3
package tests immediately. For example:

- `benchmarks/density_matrix/performance_evidence/phase31_counted_matrix_validation.py`
- `tests/partitioning/evidence/test_phase31_counted_matrix_validation.py`

#### What this validation surface should check

- `26` total counted rows,
- the frozen family split:
  - `24` primary rows,
  - `2` control rows,
- stable case IDs and no duplicates,
- baseline trio fields on every row,
- route-coverage fields on every row,
- scalar-build metadata on every row.

## Execution Outline

1. Add a bounded Phase 3.1 counted-matrix inventory builder and context iterator.
2. Add a counted-row record builder that generalizes the pilot row.
3. Add a sibling matrix validation script / bundle builder.
4. Add a focused pytest module that freezes row counts, family coverage, and
   required fields.
5. Keep the current pilot bundle and tests as regression anchors rather than
   deleting them in this task.

## Evidence Produced

- Regeneratable counted-matrix rows for the full frozen 26-case Phase 3.1
  inventory.
- A review-ready matrix bundle or summary payload with stable case IDs.
- Regression tests proving the counted matrix inventory and row field shape.

## Risks And Rollback

- Risk: the new matrix silently drifts from the frozen inventory.
- Mitigation: derive row creation from the frozen helper inventory and freeze the
  exact counts in tests.

- Risk: matrix work accidentally mutates the old Phase 3 default package too
  early.
- Mitigation: keep this task on a bounded Phase 3.1 sibling path; defer default
  pipeline switching until after the matrix and review surfaces are validated.

- Risk: matrix rows are emitted without enough route/build metadata to support
  later decision-artifact generation.
- Mitigation: treat route counters and scalar-build fields as required in the
  row schema from the start, even before `P31-S11-E02`.
