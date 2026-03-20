# Story 1 Implementation Plan

## Story Being Implemented

Story 1: The Benchmark Matrix Uses One Frozen Dual-Anchor Case-Identity Surface

This is a Layer 4 engineering plan for implementing the first behavioral slice
from `TASK_7_STORIES.md`.

## Scope

This story turns Task 7 into one explicit benchmark-matrix surface before any
positive-threshold or diagnosis interpretation is computed:

- continuity-anchor slices and representative structured-performance slices are
  frozen through one shared matrix rather than per-script case lists,
- stable workload IDs, deterministic seed rules, noise-pattern labels, and
  representative-review-set membership remain part of the Task 7 contract,
- the Task 7 benchmark matrix stays joinable with Task 5 claim-selection
  outputs, the Task 6 correctness package, and later Task 8 summaries,
- and Story 1 closes the contract for "which cases belong to the benchmark
  matrix and which cases belong to the representative review set" without yet
  claiming counted-status closure, positive-threshold satisfaction, sensitivity
  interpretation, or summary consistency.

Out of scope for this story:

- correctness-preserving positive counting already owned by Story 2,
- the explicit positive-threshold review rule already owned by Story 3,
- cross-knob sensitivity interpretation already owned by Story 4,
- the shared comparable metric surface already owned by Story 5,
- the diagnosis-path bottleneck rule already owned by Story 6,
- machine-reviewable benchmark-package assembly already owned by Story 7,
- and summary-consistency plus bounded-claim guardrails already owned by Story
  8.

## Dependencies And Assumptions

- The frozen source-of-truth contract is `TASK_7_MINI_SPEC.md`,
  `TASK_7_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-007`, and
  `P3-ADR-009`.
- The workload builders Story 1 should reuse already exist in the current Phase
  3 benchmark stack:
  - `iter_story2_microcase_descriptor_sets()` in
    `benchmarks/density_matrix/planner_surface/workloads.py`,
  - `iter_story2_structured_descriptor_sets()` in the same module,
  - and `build_phase3_continuity_partition_descriptor_set()` in
    `squander/partitioning/noisy_planner.py`.
- Task 1 already froze the shared provenance tuple Story 1 should extend rather
  than rename, especially `requested_mode`, `source_type`, `entry_route`,
  `workload_family`, and `workload_id`.
- Task 5 already emits claim-selection and calibration-side workload identity
  through `benchmarks/density_matrix/planner_calibration/` and
  `benchmarks/density_matrix/artifacts/phase3_task5/`; Story 1 should preserve
  joinability with that surface rather than redefining workload identity.
- Task 6 already emits counted supported evidence, excluded negative evidence,
  and summary-consistency precedent under
  `benchmarks/density_matrix/artifacts/phase3_task6/`; Story 1 should keep
  direct joinability with that package surface.
- Task 4 already demonstrates a representative structured performance slice
  through `benchmarks/density_matrix/partitioned_runtime/fused_performance_validation.py`;
  Story 1 should treat that as benchmark precedent rather than as the full Task
  7 matrix.
- The natural implementation home for new Task 7 matrix-selection helpers is a
  dedicated `benchmarks/density_matrix/performance_evidence/` package alongside
  `planner_surface`, `partitioned_runtime`, `planner_calibration`, and
  `correctness_evidence`, with `task7_case_selection.py` as the shared
  case-selection surface.
- Story 1 should prefer one explicit matrix-selection rule over one benchmark
  matrix file per later consumer.
- Story 1 should treat representative-review-set membership as a first-class
  contract field rather than as something inferred later from qubit count,
  workload family, or whether a case happened to be benchmarked in one script.

## Engineering Tasks

### Engineering Task 1: Freeze The Mandatory Task 7 Benchmark-Matrix Inventory

**Implements story**
- `Story 1: The Benchmark Matrix Uses One Frozen Dual-Anchor Case-Identity Surface`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 defines one explicit inventory for the mandatory Task 7 benchmark
  matrix.
- The inventory names the continuity-anchor slices, the representative
  structured-performance slices, and any bounded microcase carry-over needed
  for validation joinability concretely enough that later stories can interpret
  the benchmark without changing the case set.
- The inventory distinguishes general benchmark membership from representative
  review-set membership explicitly.

**Execution checklist**
- [ ] Freeze the mandatory Task 7 benchmark inventory around the required 4 / 6
      / 8 / 10 qubit continuity-anchor cases and the required representative 8
      / 10 qubit structured noisy families.
- [ ] Define which cases belong to the representative structured review set
      used for positive-threshold interpretation.
- [ ] Define how deterministic seed rules and noise-pattern labels are attached
      to each benchmark case.
- [ ] Keep counted-status closure, threshold interpretation, and summary
      semantics outside the Story 1 bar.

**Evidence produced**
- One stable Task 7 benchmark-matrix inventory.
- One explicit representative-review-set rule for later Task 7 stories.

**Risks / rollback**
- Risk: later validators may silently redefine the Task 7 benchmark matrix per
  script if Story 1 leaves the case inventory implicit.
- Rollback/mitigation: freeze one benchmark inventory before implementing
  threshold or diagnosis checkers.

### Engineering Task 2: Reuse The Shared Workload Builders And Provenance Tuple As The Base

**Implements story**
- `Story 1: The Benchmark Matrix Uses One Frozen Dual-Anchor Case-Identity Surface`

**Change type**
- docs | code

**Definition of done**
- Story 1 reuses the existing workload builders and the Task 1 provenance tuple
  where they already fit the Task 7 contract.
- Task 7 case identity remains aligned with earlier Phase 3 workload naming.
- Story 1 avoids creating a disconnected benchmark-case language.

**Execution checklist**
- [ ] Reuse `iter_story2_structured_descriptor_sets()` as the base structured
      workload builder where it matches the Task 7 review matrix.
- [ ] Reuse `build_phase3_continuity_partition_descriptor_set()` or the
      smallest auditable successor for the continuity-anchor slice.
- [ ] Reuse overlapping provenance fields from Task 1 directly where they match
      Story 1 needs.
- [ ] Document any additive Task 7 matrix fields explicitly rather than
      renaming shared provenance.

**Evidence produced**
- One reviewable mapping from existing Phase 3 workload builders to the Task 7
  benchmark matrix.
- One explicit boundary between reused provenance fields and Task 7-specific
  matrix metadata.

**Risks / rollback**
- Risk: Task 7 case identity may drift from Task 5 claim-selection outputs, Task
  6 correctness records, and Task 4 performance precedent if Story 1 invents
  new names casually.
- Rollback/mitigation: reuse the shared builders and provenance tuple directly.

### Engineering Task 3: Define A Shared Task 7 Benchmark-Case Record And Matrix Schema

**Implements story**
- `Story 1: The Benchmark Matrix Uses One Frozen Dual-Anchor Case-Identity Surface`

**Change type**
- code | tests

**Definition of done**
- Story 1 defines one shared Task 7 benchmark-case record shape.
- The record captures workload identity, benchmark-slice membership,
  representative-review-set membership, deterministic construction metadata, and
  planner-setting join keys without yet embedding later threshold or diagnosis
  verdicts.
- The schema is stable across continuity-anchor and structured-performance
  entries.

**Execution checklist**
- [ ] Define one shared Task 7 benchmark-case record in
      `benchmarks/density_matrix/performance_evidence/records.py` or the
      smallest adjacent helper.
- [ ] Record workload family, workload ID, seed or deterministic construction
      rule, noise-pattern label, requested mode, source type, and entry route.
- [ ] Record explicit benchmark-slice and representative-review-set membership
      fields.
- [ ] Keep counted-status, threshold, sensitivity, and diagnosis fields outside
      the Story 1 record bar.

**Evidence produced**
- One stable Task 7 benchmark-case record shape.
- Regression checks for schema stability across continuity and structured
  benchmark slices.

**Risks / rollback**
- Risk: later stories may need to reverse-engineer benchmark identity from mixed
  runtime or correctness outputs if Story 1 lacks a stable case record.
- Rollback/mitigation: define one shared case record before adding later Task 7
  verdict fields.

### Engineering Task 4: Build The Deterministic Task 7 Case-Selection And Matrix-Emission Helper

**Implements story**
- `Story 1: The Benchmark Matrix Uses One Frozen Dual-Anchor Case-Identity Surface`

**Change type**
- code | validation automation

**Definition of done**
- Story 1 exposes one deterministic helper for enumerating the Task 7 benchmark
  matrix.
- Enumeration is reproducible for a fixed repository state and fixed Story 1
  configuration.
- Later threshold, sensitivity, diagnosis, and summary validators can consume
  one shared matrix-emission surface.

**Execution checklist**
- [ ] Add a Task 7 case-selection helper under
      `benchmarks/density_matrix/performance_evidence/`, with
      `task7_case_selection.py` as the default implementation home.
- [ ] Emit benchmark-case records in deterministic order for the frozen Task 7
      matrix.
- [ ] Keep representative-review-set labels, workload IDs, and construction
      metadata stable across reruns.
- [ ] Avoid per-validator allowlists or consumer-specific case derivation logic.

**Evidence produced**
- One deterministic Task 7 case-selection surface.
- One shared benchmark-matrix rule for later Task 7 validators.

**Risks / rollback**
- Risk: consumer-specific case selection may create subtle benchmark drift that
  only appears in later performance summaries.
- Rollback/mitigation: centralize matrix selection and make it deterministic.

### Engineering Task 5: Add A Representative Dual-Anchor Matrix Validation Surface

**Implements story**
- `Story 1: The Benchmark Matrix Uses One Frozen Dual-Anchor Case-Identity Surface`

**Change type**
- tests | validation automation

**Definition of done**
- Story 1 proves the benchmark matrix spans the required continuity and
  structured slices.
- The validation surface is broad enough to catch missing-slice or
  missing-field drift early.
- The matrix remains representative and contract-driven rather than exhaustive
  over every optional case.

**Execution checklist**
- [ ] Add focused matrix checks covering at least one continuity-anchor case and
      one structured-family case at each required benchmark scale.
- [ ] Assert stable workload IDs, deterministic construction metadata, and
      representative-review-set labels.
- [ ] Assert joinability with Task 5 claim-selection references and Task 6
      correctness records where the shared workload identity overlaps.
- [ ] Keep the validation surface focused on matrix definition rather than later
      threshold or diagnosis verdicts.

**Evidence produced**
- One representative Story 1 benchmark-matrix validation surface.
- One review surface for dual-anchor case-identity stability.

**Risks / rollback**
- Risk: Story 1 may look coherent on the structured matrix while drifting
  silently on the continuity anchor or vice versa.
- Rollback/mitigation: freeze a small but dual-anchor validation surface early.

### Engineering Task 6: Add Fast Story 1 Regression Coverage For Benchmark-Matrix Stability

**Implements story**
- `Story 1: The Benchmark Matrix Uses One Frozen Dual-Anchor Case-Identity Surface`

**Change type**
- tests

**Definition of done**
- Story 1 has focused regression checks for case-identity and
  representative-review-set stability.
- The regression slice proves that matrix ordering and mandatory record fields
  remain auditable.
- The checks stay narrower than the later threshold, diagnosis, and package
  stories.

**Execution checklist**
- [ ] Add a dedicated Task 7 regression surface in
      `tests/partitioning/test_phase3_task7.py`.
- [ ] Assert stable case ordering or canonicalization for representative Task 7
      matrix records.
- [ ] Assert that mandatory fields such as workload ID, benchmark-slice label,
      representative-review-set flag, seed rule, and noise-pattern label are
      never omitted.
- [ ] Keep the checks at the matrix-definition layer rather than invoking full
      performance interpretation.

**Evidence produced**
- Fast regression coverage for Story 1 benchmark-matrix stability.
- One repeatable test surface for later Task 7 work to extend.

**Risks / rollback**
- Risk: matrix drift may remain hidden until later Task 7 summary or Task 8
  packaging work tries to interpret the package.
- Rollback/mitigation: add dedicated fast regression coverage early.

### Engineering Task 7: Emit A Stable Story 1 Benchmark-Matrix Bundle Or Rerunnable Checker

**Implements story**
- `Story 1: The Benchmark Matrix Uses One Frozen Dual-Anchor Case-Identity Surface`

**Change type**
- validation automation | docs

**Definition of done**
- Story 1 emits one stable machine-reviewable benchmark-matrix bundle or
  rerunnable checker.
- The bundle records the frozen case inventory and representative-review-set
  semantics directly.
- The output is stable enough for Stories 2 through 8 to extend without
  redefining benchmark identity.

**Execution checklist**
- [ ] Add a dedicated Story 1 validator under
      `benchmarks/density_matrix/performance_evidence/`, with
      `benchmark_matrix_validation.py` as the primary checker.
- [ ] Add a dedicated Story 1 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task7/story1_benchmark_matrix/`).
- [ ] Emit the shared case records, benchmark-slice labels, representative-
      review-set membership, and matrix summary through one stable schema.
- [ ] Record rerun commands and software metadata with the emitted bundle.

**Evidence produced**
- One stable Story 1 benchmark-matrix bundle or checker.
- One reusable case-identity surface for later Task 7 stories.

**Risks / rollback**
- Risk: prose-only benchmark-matrix closure will make later threshold and
  summary claims hard to audit.
- Rollback/mitigation: emit one thin machine-reviewable matrix surface early.

## Exit Criteria

Story 1 is complete only when all of the following are true:

- one explicit Task 7 benchmark-matrix inventory exists for the continuity-
  anchor and representative structured-performance slices,
- benchmark case identity is emitted through one deterministic shared surface
  with stable workload IDs, deterministic construction metadata, and
  representative-review-set labels,
- Task 7 case records remain joinable with the shared Phase 3 provenance tuple,
  Task 5 claim-selection outputs, and Task 6 correctness records where fields
  overlap,
- one stable Story 1 benchmark-matrix bundle or rerunnable checker exists for
  later reuse,
- and counted-status closure, threshold interpretation, sensitivity analysis,
  diagnosis, package assembly, and summary guardrails remain clearly assigned to
  later stories.

## Implementation Notes

- Prefer one shared case-selection rule over one benchmark list per validator.
- In actual coding order, Story 1 should emit a lightweight benchmark inventory
  first and leave full descriptor construction to the later measurement stories
  so the matrix contract stays cheap to validate.
- Treat representative-review-set membership as a first-class contract field,
  not as later script inference.
- Keep Story 1 focused on "which cases belong to Task 7," not yet on "which
  cases count, win, or diagnose."
