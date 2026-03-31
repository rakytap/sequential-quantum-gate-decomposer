# Engineering Task P31-S11-E02 Implementation Plan

## Engineering Task Being Implemented

`Engineering Task P31-S11-E02: Emit the route-aware decision artifact and machine-readable matrix summary`

This is a Layer 4 file-level implementation plan for the second engineering task
under Story `P31-S11` from
`../FIFTH_VERTICAL_SLICE_FULL_COUNTED_PERFORMANCE_MATRIX_STORIES_AND_ENGINEERING_TASKS.md`.
It turns the matrix-summary portion of Task 4 into a concrete implementation plan
against the current Phase 3.1 performance pilot and inventory surfaces in:

- `task-4/TASK_4_MINI_SPEC.md`,
- `task-4/ENGINEERING_TASK_P31_S11_E01_IMPLEMENTATION_PLAN.md`,
- `benchmarks/density_matrix/performance_evidence/records.py`,
- `benchmarks/density_matrix/performance_evidence/common.py`,
- `benchmarks/density_matrix/performance_evidence/case_selection.py`,
- `benchmarks/density_matrix/performance_evidence/benchmark_matrix_validation.py`,
- `benchmarks/density_matrix/performance_evidence/metric_surface_validation.py`,
- and, for publication-consumption constraints only,
  `../PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md`,
  `../CLOSURE_PLAN_PHASE_3_1.md`,
  `../SHORT_PAPER_PHASE_3_1.md`,
  `../PAPER_PHASE_3_1.md`.

## Scope

This engineering task converts the full counted matrix into the machine-readable
decision surface required by `P31-ADR-010` and `P31-ADR-013`:

- emit `decision_class` on every counted performance row,
- preserve the frozen vocabulary:
  - `phase3_sufficient`,
  - `phase31_justified`,
  - `phase31_not_justified_yet`,
- retain row-level `diagnosis_tag` or equivalent explanatory fields where they
  already exist or are justified by the harness,
- emit route-aware counters and route records on the counted whole-workload rows,
- build the matrix-wide `break_even_table` / `justification_map` directly from
  the measured rows,
- emit a review-ready machine-readable summary keyed by the same frozen case IDs,
- and keep the artifact neutral with respect to publication mode:
  it reports measured outcomes but does not itself declare the phase
  `positive-methods-ready` or `decision-study-ready`.

Out of scope for this engineering task:

- adding or removing counted performance rows,
- changing the frozen success threshold,
- changing correctness evidence rules,
- publication prose updates,
- Task 6 host-acceleration variants,
- or top-level program sync.

## Current Evidence Gap To Close

The repo now has:

- a frozen pilot decision-class mapping for one structured row,
- a frozen 26-row counted matrix inventory,
- and a formal pre-publication review documenting that the matrix-wide decision
  artifact is still missing.

What is still absent is the matrix-wide artifact that can answer:

- where the shipped Phase 3 fused path is already sufficient,
- where Phase 3.1 is justified on the frozen slice,
- and where Phase 3.1 is not justified yet.

That gap is exactly what `P31-S11-E02` closes.

### `benchmarks/density_matrix/performance_evidence/records.py`

This is the natural home for:

- row-level `decision_class`,
- optional `diagnosis_tag`,
- and any helper that reduces a full counted row set into one matrix-wide
  classification artifact.

However, the current implemented decision logic is still pilot-oriented. The
matrix summary therefore needs a new builder that:

- consumes all counted rows,
- preserves stable case IDs,
- and emits classification derived from the measured matrix rather than from
  narrative interpretation.

### `benchmarks/density_matrix/performance_evidence/*_validation.py`

The active validation surfaces should become able to assert:

- every counted row carries `decision_class`,
- route-coverage metadata is present where required,
- and the matrix-wide summary artifact references the same case IDs and frozen
  classification vocabulary.

## Dependencies And Assumptions

- The source-of-truth contract remains:
  - `TASK_4_MINI_SPEC.md`,
  - `ENGINEERING_TASK_P31_S11_E01_IMPLEMENTATION_PLAN.md`,
  - `../FIFTH_VERTICAL_SLICE_FULL_COUNTED_PERFORMANCE_MATRIX_STORIES_AND_ENGINEERING_TASKS.md`,
  - `../DETAILED_PLANNING_PHASE_3_1.md`,
  - `../ADRs_PHASE_3_1.md`
    (`P31-ADR-005`, `P31-ADR-006`, `P31-ADR-010`, `P31-ADR-013`,
    `P31-ADR-014`).
- `P31-S11-E01` is assumed complete enough to emit the full frozen counted
  matrix with stable IDs.
- The classification artifact must remain machine-derived from those rows; it
  must not be manually edited into the bundle after the fact.
- The matrix summary is allowed to classify the current result as mixed or
  negative. That is a first-class outcome.
- The summary should remain neutral with respect to Task 5 publication wording.

## Target Files And Responsibilities

### Primary row-shaping file: `benchmarks/density_matrix/performance_evidence/records.py`

This file should provide:

- one row-level helper ensuring every counted row gets `decision_class`,
- one matrix-level helper that builds `break_even_table` / `justification_map`,
- and optional diagnosis tagging where the harness can support it.

#### Recommended helpers to add

Examples:

```python
def build_phase31_decision_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ...

def build_phase31_break_even_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ...
```

The exact names may differ, but the artifact must be clearly Phase 3.1-specific
and clearly tied to the frozen counted matrix.

### Primary validation surface: `benchmarks/density_matrix/performance_evidence/benchmark_matrix_validation.py`

This file should validate that:

- every counted row has stable decision metadata,
- the matrix-wide artifact references the same case IDs,
- and the number of decision rows matches the frozen counted inventory.

### Machine-readable summary surface

Prefer one explicit summary bundle or one explicit summary section inside the
Phase 3.1 benchmark bundle rather than scattering matrix-level interpretation
across multiple files.

The key requirement is auditability:

- Case ID -> baselines -> route coverage -> decision

must be reconstructible without prose.

## Recommended Implementation Sequence

1. Freeze row-level `decision_class` emission on the full counted matrix.
2. Freeze matrix-wide `break_even_table` / `justification_map` emission from the
   same row set.
3. Add regression checks that:
   - every counted row is classified,
   - matrix-level summaries reference only frozen counted IDs,
   - no row is silently dropped from the decision artifact.
4. Expose one review-ready machine-readable summary for downstream
   `P31-S12-E01`.

## Definition of Done Mapping

This engineering task is complete when:

- every counted performance row includes `decision_class`,
- route-coverage counters and route records are present on the counted hybrid
  rows,
- a matrix-wide `break_even_table` / `justification_map` exists,
- and the output is machine-reviewable enough that the later pre-publication
  evidence review can consume it without reconstructing the matrix manually.

## Risks / Rollback

- **Risk:** one favorable row is mistaken for matrix closure.
  - **Mitigation:** make the matrix-wide summary mandatory and include the
    control-family rows in the same artifact.
- **Risk:** decision classes are applied inconsistently across rows.
  - **Mitigation:** centralize classification in one helper and regression-test
    the frozen vocabulary.
- **Risk:** the summary drifts into publication argument rather than measured
  classification.
  - **Mitigation:** keep the artifact machine-readable, case-indexed, and tied to
    the raw counted matrix rows.

## Out-of-Scope Reminder

This task stops at measured, machine-readable classification.

It does **not**:

- decide the publication mode,
- rewrite any paper surface,
- or perform program-level doc sync.

Those belong downstream of `P31-S12-E01` and Task 5.
