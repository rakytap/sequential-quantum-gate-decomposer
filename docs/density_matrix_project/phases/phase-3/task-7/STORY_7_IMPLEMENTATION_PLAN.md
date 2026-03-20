# Story 7 Implementation Plan

## Story Being Implemented

Story 7: One Machine-Reviewable Benchmark Package Joins Counted, Diagnosis-Only,
And Excluded Evidence

This is a Layer 4 engineering plan for implementing the seventh behavioral slice
from `TASK_7_STORIES.md`.

## Scope

This story turns Task 7 into one shared benchmark package:

- counted supported evidence, diagnosis-only evidence, and excluded or boundary
  evidence are emitted through one stable machine-reviewable package rather than
  disconnected story outputs,
- the package preserves stable provenance, planner-setting references, metric
  fields, threshold or diagnosis semantics, and exclusion status across the full
  Task 7 surface,
- later Task 8 publication packaging can consume one shared benchmark surface
  directly,
- and Story 7 closes the contract for "how Task 7 evidence is packaged for
  downstream use" without yet claiming summary-consistency or bounded-claim
  closure semantics.

Out of scope for this story:

- benchmark-matrix inventory already owned by Story 1,
- counted-supported benchmark gating already owned by Story 2,
- the explicit positive-threshold review rule already owned by Story 3,
- the sensitivity surface already owned by Story 4,
- the comparable metric surface already owned by Story 5,
- the diagnosis-path bottleneck rule already owned by Story 6,
- and summary-consistency plus bounded-claim guardrails already owned by Story
  8.

## Dependencies And Assumptions

- Stories 1 through 6 already define the case identity, counting rule,
  threshold, sensitivity, metric, and diagnosis surfaces Story 7 must package
  consistently.
- The frozen source-of-truth contract is `TASK_7_MINI_SPEC.md`,
  `TASK_7_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-006`,
  `P3-ADR-008`, and `P3-ADR-009`.
- Task 5 already emits one machine-reviewable calibration bundle Story 7 should
  remain directly compatible with through:
  - `benchmarks/density_matrix/planner_calibration/bundle.py`,
  - `benchmarks/density_matrix/planner_calibration/calibration_bundle_validation.py`,
  - and `benchmarks/density_matrix/artifacts/phase3_task5/`.
- Task 6 already emits one shared correctness package Story 7 should remain
  directly compatible with through:
  - `benchmarks/density_matrix/correctness_evidence/bundle.py`,
  - `benchmarks/density_matrix/correctness_evidence/correctness_bundle_validation.py`,
  - and `benchmarks/density_matrix/artifacts/phase3_task6/`.
- Task 4 already provides benchmark-package precedent through the Story 7
  performance bundle under
  `benchmarks/density_matrix/artifacts/phase3_task4/story7_performance/`; Story
  7 should learn from that bounded benchmark bundle without copying its fused-
  specific scope blindly.
- Story 7 should prefer additive reuse of shared Phase 3 field names over
  inventing a disconnected Task 7 packaging vocabulary.
- The current implementation learning is that Story 7 should package the shared
  metric and status record spine from Stories 1 through 6 through one
  `bundle.py` payload rather than assembling the package from unrelated
  story-local schemas.
- Later Task 8 publication packaging should be able to consume the Story 7
  benchmark package directly without manual relabeling of workload identity,
  planner-setting references, counted-status fields, or diagnosis labels.
- The natural implementation home for Task 7 package assembly is the new
  `benchmarks/density_matrix/performance_evidence/` package, with `bundle.py`
  and `benchmark_bundle_validation.py` as the shared Story 7 package surface.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 7 Benchmark-Package Provenance Tuple And Bundle Rule

**Implements story**
- `Story 7: One Machine-Reviewable Benchmark Package Joins Counted, Diagnosis-Only, And Excluded Evidence`

**Change type**
- docs | validation automation

**Definition of done**
- Story 7 defines one stable provenance tuple and one bundle rule for the Task 7
  benchmark package.
- The rule is explicit enough that later publication work can rely on it
  safely.
- Story 7 distinguishes stable package assembly from later summary-consistency
  and bounded-claim closure semantics.

**Execution checklist**
- [ ] Freeze the minimum Story 7 provenance tuple around case identity,
      planner-setting references, metric fields, counted-status, threshold
      verdicts, diagnosis labels, and exclusion semantics.
- [ ] Freeze the minimum result-surface fields every Task 7 package record must
      expose.
- [ ] Define which fields remain shared with earlier Phase 3 records unchanged
      and which are additive Task 7 package fields.
- [ ] Keep summary-consistency and bounded-claim interpretation outside the
      Story 7 bar.

**Evidence produced**
- One stable Task 7 benchmark-package provenance tuple.
- One explicit package rule for counted, diagnosis-only, and excluded Task 7
  outputs.

**Risks / rollback**
- Risk: later Task 8 consumers may assemble incompatible views of the same Task
  7 evidence if Story 7 leaves package rules loose.
- Rollback/mitigation: freeze the package rule before broadening Story 7 output
  production.

### Engineering Task 2: Reuse The Shared Phase 3 Evidence Fields And Existing Bundle Precedent As The Base

**Implements story**
- `Story 7: One Machine-Reviewable Benchmark Package Joins Counted, Diagnosis-Only, And Excluded Evidence`

**Change type**
- docs | code

**Definition of done**
- Story 7 packaging builds on the shared Phase 3 provenance and verdict surfaces
  where fields overlap.
- Story 7 learns from existing Task 4, Task 5, and Task 6 bundle precedent
  without copying scope-specific language blindly.
- Story 7 avoids creating a disconnected Task 7 packaging vocabulary.

**Execution checklist**
- [ ] Review the shared fields already emitted by Stories 1 through 6 and keep
      overlapping names stable where practical.
- [ ] Review Task 4, Task 5, and Task 6 bundle precedent and reuse only the
      parts that strengthen Task 7 consumer compatibility.
- [ ] Add only the Task 7-specific package fields needed for benchmark-package
      identity and consumer handoff.
- [ ] Document where Story 7 intentionally extends earlier field vocabularies.

**Evidence produced**
- One reviewable mapping from shared Phase 3 fields and earlier bundle patterns
  to the Task 7 benchmark package.
- One explicit boundary between reused vocabulary and Story 7-specific package
  fields.

**Risks / rollback**
- Risk: Story 7 may produce a plausible package that later consumers still need
  to translate manually.
- Rollback/mitigation: align Story 7 packaging with shared Phase 3 fields and
  proven bundle precedent wherever practical.

### Engineering Task 3: Define A Shared Task 7 Benchmark-Package Record And Bundle Surface

**Implements story**
- `Story 7: One Machine-Reviewable Benchmark Package Joins Counted, Diagnosis-Only, And Excluded Evidence`

**Change type**
- code | tests

**Definition of done**
- Story 7 defines one shared top-level benchmark-package record shape and one
  stable top-level bundle surface.
- The record separates case identity, metric fields, threshold or diagnosis
  semantics, and exclusion status cleanly.
- The shape remains stable across counted, diagnosis-only, and excluded Task 7
  evidence.

**Execution checklist**
- [ ] Define one shared top-level Task 7 benchmark-package record in
      `benchmarks/density_matrix/performance_evidence/bundle.py` or the
      smallest auditable successor.
- [ ] Record counted, diagnosis-only, and excluded evidence through one
      structurally compatible package shape.
- [ ] Keep package-level identity separate from per-case metric, verdict, and
      exclusion fields.
- [ ] Add regression checks for top-level package-schema stability.

**Evidence produced**
- One stable shared Task 7 benchmark-package record shape.
- Regression checks for package-schema stability across counted, diagnosis-only,
  and excluded evidence.

**Risks / rollback**
- Risk: Story 7 outputs may remain individually plausible but structurally hard
  to compare or package downstream.
- Rollback/mitigation: freeze one shared benchmark-package record shape before
  broadening bundle emission.

### Engineering Task 4: Cross-Check Counted, Diagnosis-Only, And Excluded Slices Against The Shared Package Surface

**Implements story**
- `Story 7: One Machine-Reviewable Benchmark Package Joins Counted, Diagnosis-Only, And Excluded Evidence`

**Change type**
- tests | validation automation

**Definition of done**
- Counted, diagnosis-only, and excluded Task 7 slices emit records through the
  same shared package surface where fields overlap.
- Schema drift across the story outputs is caught early.
- The checks stay focused on package compatibility rather than on later summary
  semantics.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_phase3_task7.py` for
      counted-versus-diagnosis-versus-excluded package compatibility.
- [ ] Compare top-level provenance, case identity, planner-setting references,
      metric-field presence, threshold or diagnosis fields, and exclusion status
      across the Task 7 slices.
- [ ] Keep the checks narrow to package compatibility rather than to Story 8
      summary-consistency.
- [ ] Fail quickly when Task 7 records diverge from the shared package surface.

**Evidence produced**
- Fast regression coverage for cross-slice Task 7 package stability.
- Reviewable compatibility checks for the shared Story 7 package surface.

**Risks / rollback**
- Risk: package drift may remain hidden until publication packaging tries to
  consume the full Task 7 surface.
- Rollback/mitigation: enforce cross-slice package checks early.

### Engineering Task 5: Preserve Direct Compatibility With Task 5 Calibration, Task 6 Correctness, And Task 8 Publication Consumers

**Implements story**
- `Story 7: One Machine-Reviewable Benchmark Package Joins Counted, Diagnosis-Only, And Excluded Evidence`

**Change type**
- code | tests | validation automation

**Definition of done**
- The Task 7 benchmark package remains directly consumable by Task 5 claim
  cross-checks, Task 6 correctness joins, and Task 8 publication packaging.
- Consumers do not need to rerun the same cases through a different interface
  just to package them.
- Story 7 preserves direct consumer compatibility where fields overlap.

**Execution checklist**
- [ ] Keep workload identity and planner-setting join keys stable enough to link
      Task 5 calibration outputs to Task 7 benchmark judgments.
- [ ] Keep Task 6 correctness references directly usable by the Task 7 package
      without manual relabeling.
- [ ] Add focused checks proving Task 8 can read one stable Task 7 package
      surface.
- [ ] Avoid per-consumer parsing rules or notebook-only reformatting.

**Evidence produced**
- One direct-consumer-compatible Task 7 benchmark-package surface.
- Focused regression coverage for consumer-facing field stability.

**Risks / rollback**
- Risk: later work may need to reconstruct Task 7 evidence manually from
  incompatible intermediate outputs.
- Rollback/mitigation: preserve direct consumer compatibility as part of Story 7
  closure.

### Engineering Task 6: Emit A Stable Story 7 Benchmark Package Or Rerunnable Checker

**Implements story**
- `Story 7: One Machine-Reviewable Benchmark Package Joins Counted, Diagnosis-Only, And Excluded Evidence`

**Change type**
- validation automation | docs

**Definition of done**
- Story 7 emits one stable machine-reviewable benchmark package or rerunnable
  checker.
- The package supports later publication and review consumers directly.
- The output is stable enough for Story 8 to add summary-consistency and
  bounded-claim guardrails without changing the underlying record shape
  materially.

**Execution checklist**
- [ ] Add a dedicated Story 7 validator under
      `benchmarks/density_matrix/performance_evidence/`, with
      `benchmark_bundle_validation.py` as the primary checker.
- [ ] Add a dedicated Story 7 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task7/story7_benchmark_package/`).
- [ ] Emit Task 7 counted, diagnosis-only, and excluded records through one
      stable schema plus a stable bundle summary.
- [ ] Record rerun commands and software metadata with the emitted package.

**Evidence produced**
- One stable Story 7 benchmark package or checker.
- One direct citation surface for the Task 7 shared evidence package.

**Risks / rollback**
- Risk: if Story 7 evidence remains scattered across story-local bundles only,
  later consumers will struggle to audit the full Task 7 surface.
- Rollback/mitigation: emit one stable shared benchmark package and cite it
  directly.

### Engineering Task 7: Document The Story 7 Consumer Handoff And Run The Shared Package Surface

**Implements story**
- `Story 7: One Machine-Reviewable Benchmark Package Joins Counted, Diagnosis-Only, And Excluded Evidence`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the Story 7 shared package rule and consumer
  expectations.
- The Story 7 package harness and bundle run successfully.
- Story 7 makes clear that summary-consistency and bounded-claim closure belong
  to Story 8.

**Execution checklist**
- [ ] Document the shared Task 7 package rule and consumer-facing field
      expectations.
- [ ] Explain how Task 8 and later publication consumers should consume the
      Story 7 package directly.
- [ ] Run focused Story 7 regression coverage and verify
      `benchmarks/density_matrix/performance_evidence/benchmark_bundle_validation.py`.
- [ ] Record stable references to the Story 7 tests, checker, and emitted
      package.

**Evidence produced**
- Passing Story 7 benchmark-package regression checks.
- One stable Story 7 benchmark-package or checker reference.

**Risks / rollback**
- Risk: later work may misread Story 7 as already closing summary semantics
  rather than as the shared evidence package it is meant to provide.
- Rollback/mitigation: document the Story 7 handoff to Story 8 explicitly.

## Exit Criteria

Story 7 is complete only when all of the following are true:

- one explicit shared Task 7 benchmark-package provenance tuple and bundle rule
  exist,
- counted, diagnosis-only, and excluded Task 7 evidence emit records through
  one stable shared package surface where fields overlap,
- later Task 5, Task 6, and Task 8 consumers can read one stable Task 7 package
  without manual relabeling,
- one stable Story 7 benchmark package or rerunnable checker exists for direct
  citation,
- and summary-consistency plus bounded-claim closure remain clearly assigned to
  Story 8.

## Implementation Notes

- Prefer additive schema evolution over renaming shared Phase 3 fields.
- In actual coding order, thin package-shape scaffolding may land earlier than
  final Story 7 closure so the earlier validators can emit structurally
  compatible records without later schema churn.
- Keep Story 7 focused on stable evidence packaging, not yet on final bounded-
  claim interpretation.
- Treat consumer compatibility as part of the benchmark-package contract, not as
  a downstream convenience.
