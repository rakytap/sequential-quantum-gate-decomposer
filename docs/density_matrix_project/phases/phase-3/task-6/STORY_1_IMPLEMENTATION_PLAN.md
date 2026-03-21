# Story 1 Implementation Plan

## Story Being Implemented

Story 1: The Correctness Matrix Uses One Frozen Case-Identity Surface

This is a Layer 4 engineering plan for implementing the first behavioral slice
from `TASK_6_STORIES.md`.

## Scope

This story turns Task 6 into one explicit correctness-matrix surface before any
threshold verdicts or summary claims are computed:

- mandatory micro-validation, continuity-anchor, and structured-family cases are
  frozen through one shared matrix rather than per-script case lists,
- stable workload IDs, deterministic seed rules, noise-pattern labels, and
  slice-membership labels remain part of the Task 6 contract,
- the Task 6 correctness matrix stays joinable with Task 5 calibration outputs
  and later Task 7 / Task 8 summaries,
- and Story 1 closes the contract for "which cases belong to the correctness
  package" without yet claiming pass thresholds, unsupported-boundary closure,
  or summary consistency.

Out of scope for this story:

- sequential-baseline exactness gating already owned by Story 2,
- the mandatory Qiskit Aer slice already owned by Story 3,
- density-validity and continuity-energy verdict closure already owned by Story
  4,
- runtime and fusion classification comparability already owned by Story 5,
- unsupported-boundary stage separation already owned by Story 6,
- machine-reviewable full-package assembly already owned by Story 7,
- and summary-consistency plus counted-status guardrails already owned by Story
  8.

## Dependencies And Assumptions

- The frozen source-of-truth contract is `TASK_6_MINI_SPEC.md`,
  `TASK_6_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-007`,
  `P3-ADR-008`, and `P3-ADR-009`.
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
- Task 5 already emits calibration-side workload and planner-setting identity
  through `benchmarks/density_matrix/planner_calibration/` and
  `benchmarks/density_matrix/artifacts/planner_calibration/`; Story 1 should preserve
  joinability with that surface rather than redefining workload identity.
- The current implementation learning is that Story 1 should freeze the Task 6
  matrix against the selected Task 5 supported candidate rather than against the
  full Task 5 comparison family, while still preserving Task 5 join keys for
  later cross-checks.
- Existing Task 3 and Task 4 validators already prove that mandatory runtime and
  fused-runtime slices can be expressed through stable case metadata under:
  - `benchmarks/density_matrix/partitioned_runtime/mandatory_workload_runtime_validation.py`,
  - `benchmarks/density_matrix/partitioned_runtime/continuity_runtime_validation.py`,
  - and `benchmarks/density_matrix/partitioned_runtime/structured_fused_runtime_validation.py`.
- The natural implementation home for new Task 6 matrix-selection helpers is a
  dedicated `benchmarks/density_matrix/correctness_evidence/` package alongside
  `planner_surface`, `partitioned_runtime`, and `planner_calibration`, with
  `task6_case_selection.py` as the shared case-selection surface.
- Story 1 should prefer one explicit case-selection rule over one matrix file
  per consumer.
- Story 1 should treat slice membership as a first-class contract field rather
  than as something inferred later from qubit count or workload family alone.

## Engineering Tasks

### Engineering Task 1: Freeze The Mandatory Task 6 Correctness-Matrix Inventory

**Implements story**
- `Story 1: The Correctness Matrix Uses One Frozen Case-Identity Surface`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 defines one explicit inventory for the mandatory Task 6 correctness
  matrix.
- The inventory names the required microcase, continuity, and structured-family
  slices concretely enough that later stories can apply thresholds without
  changing the case set.
- The inventory distinguishes internal-only versus internal-plus-external slice
  membership explicitly.

**Execution checklist**
- [ ] Freeze the mandatory Task 6 workload inventory around the required 2 to 4
      qubit microcases, the required 4 / 6 / 8 / 10 qubit continuity-anchor
      cases, and the required 8 / 10 qubit structured noisy families.
- [ ] Define which cases belong to the internal-only slice and which cases
      belong to the internal-plus-external slice.
- [ ] Define how deterministic seed rules and noise-pattern labels are attached
      to each matrix case.
- [ ] Keep threshold verdicts, unsupported-boundary closure, and summary
      semantics outside the Story 1 bar.

**Evidence produced**
- One stable Task 6 correctness-matrix inventory.
- One explicit slice-membership rule for internal-only versus
  internal-plus-external cases.

**Risks / rollback**
- Risk: later validators may silently redefine the correctness matrix per script
  if Story 1 leaves the case inventory implicit.
- Rollback/mitigation: freeze one inventory before implementing threshold
  checkers.

### Engineering Task 2: Reuse The Shared Workload Builders And Provenance Tuple As The Base

**Implements story**
- `Story 1: The Correctness Matrix Uses One Frozen Case-Identity Surface`

**Change type**
- docs | code

**Definition of done**
- Story 1 reuses the existing workload builders and the Task 1 provenance tuple
  where they already fit the Task 6 contract.
- Task 6 case identity remains aligned with earlier Phase 3 workload naming.
- Story 1 avoids creating a disconnected case-selection language.

**Execution checklist**
- [ ] Reuse `iter_story2_microcase_descriptor_sets()` and
      `iter_story2_structured_descriptor_sets()` as the base workload builders
      where they match the Task 6 matrix.
- [ ] Reuse `build_phase3_continuity_partition_descriptor_set()` or the smallest
      auditable successor for the continuity-anchor slice.
- [ ] Reuse overlapping provenance fields from Task 1 directly where they match
      Story 1 needs.
- [ ] Document any additive Task 6 matrix fields explicitly rather than renaming
      shared provenance.

**Evidence produced**
- One reviewable mapping from existing Phase 3 workload builders to the Task 6
  correctness matrix.
- One explicit boundary between reused provenance fields and Task 6-specific
  slice metadata.

**Risks / rollback**
- Risk: Task 6 case identity may drift from Task 5 calibration outputs and Task
  3 / Task 4 runtime evidence if Story 1 invents new names casually.
- Rollback/mitigation: reuse the shared builders and provenance tuple directly.

### Engineering Task 3: Define A Shared Task 6 Correctness-Case Record And Matrix Schema

**Implements story**
- `Story 1: The Correctness Matrix Uses One Frozen Case-Identity Surface`

**Change type**
- code | tests

**Definition of done**
- Story 1 defines one shared Task 6 correctness-case record shape.
- The record captures workload identity, slice membership, deterministic
  construction metadata, and planner-setting join keys without yet embedding
  later threshold verdicts.
- The schema is stable across microcase, continuity, and structured-family
  entries.

**Execution checklist**
- [ ] Define one shared Task 6 correctness-case record in
      `benchmarks/density_matrix/correctness_evidence/records.py` or the
      smallest adjacent helper.
- [ ] Record workload family, workload ID, seed or deterministic construction
      rule, noise-pattern label, requested mode, source type, and entry route.
- [ ] Record one explicit slice-membership field for internal-only versus
      internal-plus-external validation.
- [ ] Keep threshold, unsupported, and summary-consistency fields outside the
      Story 1 record bar.

**Evidence produced**
- One stable Task 6 correctness-case record shape.
- Regression checks for schema stability across all mandatory slices.

**Risks / rollback**
- Risk: later stories may need to reverse-engineer case identity from mixed
  runtime outputs if Story 1 lacks a stable record.
- Rollback/mitigation: define one shared case record before adding per-story
  verdict fields.

### Engineering Task 4: Build The Deterministic Task 6 Case-Selection And Matrix-Emission Helper

**Implements story**
- `Story 1: The Correctness Matrix Uses One Frozen Case-Identity Surface`

**Change type**
- code | validation automation

**Definition of done**
- Story 1 exposes one deterministic helper for enumerating the Task 6
  correctness matrix.
- Enumeration is reproducible for a fixed repository state and fixed story
  configuration.
- Later correctness, unsupported-boundary, and summary validators can consume
  one shared matrix-emission surface.

**Execution checklist**
- [ ] Add a Task 6 case-selection helper under
      `benchmarks/density_matrix/correctness_evidence/`, with
      `task6_case_selection.py` as the default implementation home.
- [ ] Emit correctness-case records in deterministic order for the frozen matrix.
- [ ] Keep slice-membership labels, workload IDs, and construction metadata
      stable across reruns.
- [ ] Avoid per-validator allowlists or consumer-specific case derivation logic.

**Evidence produced**
- One deterministic Task 6 case-selection surface.
- One shared matrix-emission rule for later Task 6 validators.

**Risks / rollback**
- Risk: consumer-specific case selection may create subtle matrix drift that only
  appears in later summaries.
- Rollback/mitigation: centralize matrix selection and make it deterministic.

### Engineering Task 5: Add A Representative Cross-Slice Matrix Validation Surface

**Implements story**
- `Story 1: The Correctness Matrix Uses One Frozen Case-Identity Surface`

**Change type**
- tests | validation automation

**Definition of done**
- Story 1 proves the correctness matrix spans the required workload classes.
- The validation surface is broad enough to catch missing-slice or missing-field
  drift early.
- The matrix remains representative and contract-driven rather than exhaustive
  over every optional case.

**Execution checklist**
- [ ] Add focused matrix checks covering at least one required microcase, one
      continuity-anchor case, and one structured-family case.
- [ ] Assert stable workload IDs, deterministic construction metadata, and
      slice-membership labels.
- [ ] Assert joinability with Task 5 planner-setting references where the shared
      workload identity overlaps.
- [ ] Keep the validation surface focused on matrix definition rather than later
      threshold verdicts.

**Evidence produced**
- One representative Story 1 correctness-matrix validation surface.
- One review surface for cross-slice case-identity stability.

**Risks / rollback**
- Risk: Story 1 may look coherent on one workload family while drifting silently
  on the others.
- Rollback/mitigation: freeze a small but cross-slice validation surface early.

### Engineering Task 6: Add Fast Story 1 Regression Coverage For Matrix Identity Stability

**Implements story**
- `Story 1: The Correctness Matrix Uses One Frozen Case-Identity Surface`

**Change type**
- tests

**Definition of done**
- Story 1 has focused regression checks for case-identity and
  slice-membership stability.
- The regression slice proves that matrix ordering and mandatory record fields
  remain auditable.
- The checks stay narrower than the later correctness and unsupported-boundary
  stories.

**Execution checklist**
- [ ] Add a dedicated Task 6 regression surface in
      `tests/partitioning/test_correctness_evidence.py`.
- [ ] Assert stable case ordering or canonicalization for representative Task 6
      matrix records.
- [ ] Assert that mandatory fields such as workload ID, slice membership, seed
      rule, and noise-pattern label are never omitted.
- [ ] Keep the checks at the matrix-definition layer rather than invoking full
      sequential or Aer validation.

**Evidence produced**
- Fast regression coverage for Story 1 matrix-identity stability.
- One repeatable test surface for later Task 6 work to extend.

**Risks / rollback**
- Risk: matrix drift may remain hidden until later Task 7 or Task 8 packaging.
- Rollback/mitigation: add dedicated fast regression coverage early.

### Engineering Task 7: Emit A Stable Story 1 Correctness-Matrix Bundle Or Rerunnable Checker

**Implements story**
- `Story 1: The Correctness Matrix Uses One Frozen Case-Identity Surface`

**Change type**
- validation automation | docs

**Definition of done**
- Story 1 emits one stable machine-reviewable correctness-matrix bundle or
  rerunnable checker.
- The bundle records the frozen case inventory and slice-membership semantics
  directly.
- The output is stable enough for Stories 2 through 8 to extend without
  redefining case identity.

**Execution checklist**
- [ ] Add a dedicated Story 1 validator under
      `benchmarks/density_matrix/correctness_evidence/`, with
      `correctness_matrix_validation.py` as the primary checker.
- [ ] Add a dedicated Story 1 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/correctness_evidence/correctness_matrix/`).
- [ ] Emit the shared case records, slice-membership labels, and matrix summary
      through one stable schema.
- [ ] Record rerun commands and software metadata with the emitted bundle.

**Evidence produced**
- One stable Story 1 correctness-matrix bundle or checker.
- One reusable case-identity surface for later Task 6 stories.

**Risks / rollback**
- Risk: prose-only matrix closure will make later threshold and summary claims
  hard to audit.
- Rollback/mitigation: emit one thin machine-reviewable matrix surface early.

## Exit Criteria

Story 1 is complete only when all of the following are true:

- one explicit Task 6 correctness-matrix inventory exists for the mandatory
  microcase, continuity, and structured-family slices,
- case identity is emitted through one deterministic shared surface with stable
  workload IDs, deterministic construction metadata, and slice-membership
  labels,
- Task 6 case records remain joinable with the shared Phase 3 provenance tuple
  and Task 5 calibration outputs where fields overlap,
- one stable Story 1 correctness-matrix bundle or rerunnable checker exists for
  later reuse,
- and threshold verdicts, unsupported-boundary closure, full-package assembly,
  and summary-consistency guardrails remain clearly assigned to later stories.

## Implementation Notes

- Prefer one shared case-selection rule over one case list per validator.
- Treat internal-only versus internal-plus-external slice membership as a
  first-class contract field, not as later script inference.
- Keep Story 1 focused on "which cases belong to Task 6," not yet on "which
  cases pass."
