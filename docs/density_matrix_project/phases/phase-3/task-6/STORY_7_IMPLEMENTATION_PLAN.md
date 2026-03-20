# Story 7 Implementation Plan

## Story Being Implemented

Story 7: One Machine-Reviewable Correctness Package Joins Positive And Negative
Evidence

This is a Layer 4 engineering plan for implementing the seventh behavioral slice
from `TASK_6_STORIES.md`.

## Scope

This story turns Task 6 into one shared correctness package:

- positive supported evidence and excluded negative evidence are emitted through
  one stable machine-reviewable package rather than disconnected story outputs,
- the package preserves stable provenance, join keys, verdict fields, and
  exclusion semantics across the full Task 6 surface,
- later Task 7 benchmark work and Task 8 publication packaging can consume one
  shared correctness surface directly,
- and Story 7 closes the contract for "how Task 6 evidence is packaged for
  downstream use" without yet claiming summary-consistency or final claim-
  closure semantics.

Out of scope for this story:

- correctness-matrix inventory already owned by Story 1,
- internal sequential-baseline gating already owned by Story 2,
- the bounded Qiskit Aer slice already owned by Story 3,
- output-integrity and continuity-energy emphasis already owned by Story 4,
- runtime and fusion classification comparability already owned by Story 5,
- unsupported-boundary stage separation already owned by Story 6,
- and counted-status propagation into later benchmark and publication summaries
  already owned by Story 8.

## Dependencies And Assumptions

- Stories 1 through 6 already define the case identity, exactness, integrity,
  classification, and negative-evidence surfaces Story 7 must package
  consistently.
- The frozen source-of-truth contract is `TASK_6_MINI_SPEC.md`,
  `TASK_6_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-006`,
  `P3-ADR-008`, and `P3-ADR-009`.
- Task 5 already emits one machine-reviewable calibration bundle Story 7 should
  remain directly compatible with through:
  - `benchmarks/density_matrix/planner_calibration/calibration_bundle_validation.py`,
  - `benchmarks/density_matrix/planner_calibration/bundle.py`,
  - and `benchmarks/density_matrix/artifacts/phase3_task5/`.
- Phase 2 already provides publication-oriented bundle precedent Story 7 should
  learn from where practical:
  - `benchmarks/density_matrix/workflow_evidence/workflow_publication_bundle.py`,
  - `benchmarks/density_matrix/validation_evidence/validation_evidence_publication_bundle.py`,
  - and `benchmarks/density_matrix/publication_claim_package/publication_claim_bundle.py`.
- Story 7 should prefer additive reuse of the shared Phase 3 provenance tuple
  and earlier story field names over inventing a disconnected Task 6 packaging
  language.
- The current implementation learning is that Story 7 should package the shared
  positive record spine from `records.py` together with the normalized
  unsupported-boundary records from Story 6 through one `bundle.py` payload,
  rather than assembling the package from unrelated story-local schemas.
- Later Task 7 benchmark rollups and Task 8 publication packaging should be able
  to consume the Story 7 correctness package directly without manual relabeling
  of workload identity, planner-setting references, or counted-status fields.
- Story 7 should treat consumer compatibility as part of the contract, not as a
  downstream convenience.
- The natural implementation home for Task 6 package assembly is the new
  `benchmarks/density_matrix/correctness_evidence/` package, with `bundle.py`
  and `correctness_bundle_validation.py` as the shared Story 7 package surface.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 6 Correctness-Package Provenance Tuple And Bundle Rule

**Implements story**
- `Story 7: One Machine-Reviewable Correctness Package Joins Positive And Negative Evidence`

**Change type**
- docs | validation automation

**Definition of done**
- Story 7 defines one stable provenance tuple and one bundle rule for the Task 6
  correctness package.
- The rule is explicit enough that later benchmark and publication work can rely
  on it safely.
- Story 7 distinguishes stable package assembly from later summary-consistency
  and main-claim closure semantics.

**Execution checklist**
- [ ] Freeze the minimum Story 7 provenance tuple around case identity,
      slice-membership, planner-setting references, runtime-path and fusion
      labels, threshold verdicts, and exclusion semantics.
- [ ] Freeze the minimum result-surface fields every Task 6 package record must
      expose.
- [ ] Define which fields remain shared with earlier Phase 3 records unchanged
      and which are additive Task 6 package fields.
- [ ] Keep summary-consistency and final claim-closure interpretation outside
      the Story 7 bar.

**Evidence produced**
- One stable Task 6 correctness-package provenance tuple.
- One explicit package rule for supported and excluded Task 6 outputs.

**Risks / rollback**
- Risk: later Task 7 and Task 8 consumers may assemble incompatible views of the
  same Task 6 evidence if Story 7 leaves package rules loose.
- Rollback/mitigation: freeze the package rule before broadening Story 7 output
  production.

### Engineering Task 2: Reuse The Shared Phase 3 Evidence Fields And Phase 2 Bundle Precedent As The Base

**Implements story**
- `Story 7: One Machine-Reviewable Correctness Package Joins Positive And Negative Evidence`

**Change type**
- docs | code

**Definition of done**
- Story 7 packaging builds on the shared Phase 3 provenance and verdict surfaces
  where fields overlap.
- Story 7 learns from the existing Phase 2 bundle precedent without copying its
  workflow-specific language blindly.
- Story 7 avoids creating a disconnected Task 6 packaging vocabulary.

**Execution checklist**
- [ ] Review the shared fields already emitted by Stories 1 through 6 and keep
      overlapping names stable where practical.
- [ ] Review Phase 2 publication-bundle precedent and reuse only the parts that
      strengthen Task 6 consumer compatibility.
- [ ] Add only the Task 6-specific package fields needed for correctness-package
      identity and consumer handoff.
- [ ] Document where Story 7 intentionally extends earlier field vocabularies.

**Evidence produced**
- One reviewable mapping from shared Phase 3 fields and Phase 2 bundle patterns
  to the Task 6 correctness package.
- One explicit boundary between reused vocabulary and Story 7-specific package
  fields.

**Risks / rollback**
- Risk: Story 7 may produce a plausible package that later consumers still need
  to translate manually.
- Rollback/mitigation: align Story 7 packaging with shared Phase 3 fields and
  proven bundle precedent wherever practical.

### Engineering Task 3: Define A Shared Task 6 Correctness-Package Record And Bundle Surface

**Implements story**
- `Story 7: One Machine-Reviewable Correctness Package Joins Positive And Negative Evidence`

**Change type**
- code | tests

**Definition of done**
- Story 7 defines one shared top-level correctness-package record shape and one
  stable top-level bundle surface.
- The record separates case identity, threshold verdicts, classification
  fields, and negative-evidence semantics cleanly.
- The shape remains stable across both counted supported and excluded negative
  Task 6 evidence.

**Execution checklist**
- [ ] Define one shared top-level Task 6 correctness-package record in
      `benchmarks/density_matrix/correctness_evidence/bundle.py` or the
      smallest auditable successor.
- [ ] Record positive and negative evidence through one structurally compatible
      package shape.
- [ ] Keep package-level identity separate from per-case verdict and exclusion
      fields.
- [ ] Add regression checks for top-level schema stability.

**Evidence produced**
- One stable shared Task 6 correctness-package record shape.
- Regression checks for package-schema stability across positive and negative
  evidence.

**Risks / rollback**
- Risk: Story 7 outputs may remain individually plausible but structurally hard
  to compare or package downstream.
- Rollback/mitigation: freeze one shared correctness-package record shape before
  broadening bundle emission.

### Engineering Task 4: Cross-Check Positive And Negative Task 6 Slices Against The Shared Package Surface

**Implements story**
- `Story 7: One Machine-Reviewable Correctness Package Joins Positive And Negative Evidence`

**Change type**
- tests | validation automation

**Definition of done**
- Positive and negative Task 6 slices emit records through the same shared
  package surface where fields overlap.
- Schema drift across the story outputs is caught early.
- The checks stay focused on package compatibility rather than on later summary
  semantics.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_phase3_task6.py` for
      positive-versus-negative package compatibility.
- [ ] Compare top-level provenance, case identity, planner-setting reference,
      path-label presence, verdict fields, and exclusion fields across the Task
      6 slices.
- [ ] Keep the checks narrow to package compatibility rather than to Story 8
      summary-consistency.
- [ ] Fail quickly when supported and excluded Task 6 records diverge from the
      shared package surface.

**Evidence produced**
- Fast regression coverage for cross-slice Task 6 package stability.
- Reviewable compatibility checks for the shared Story 7 package surface.

**Risks / rollback**
- Risk: package drift may remain hidden until broad benchmark or publication
  packaging tries to consume the same data.
- Rollback/mitigation: enforce cross-slice package checks early.

### Engineering Task 5: Preserve Direct Compatibility With Task 5 Calibration, Task 7 Benchmark, And Task 8 Publication Consumers

**Implements story**
- `Story 7: One Machine-Reviewable Correctness Package Joins Positive And Negative Evidence`

**Change type**
- code | tests | validation automation

**Definition of done**
- The Task 6 correctness package remains directly consumable by Task 5 claim
  cross-checks, Task 7 benchmark rollups, and Task 8 publication packaging.
- Consumers do not need to rerun the same cases through a different interface
  just to package them.
- Story 7 preserves direct consumer compatibility where fields overlap.

**Execution checklist**
- [ ] Keep workload identity and planner-setting join keys stable enough to link
      Task 5 calibration outputs to Task 6 correctness judgments.
- [ ] Keep Task 6 package fields directly usable by later Task 7 benchmark
      rollups and Task 8 publication manifests.
- [ ] Add focused checks proving later consumers can read one stable Task 6
      package surface.
- [ ] Avoid per-consumer parsing rules or notebook-only reformatting.

**Evidence produced**
- One direct-consumer-compatible Task 6 correctness-package surface.
- Focused regression coverage for consumer-facing field stability.

**Risks / rollback**
- Risk: later work may need to reconstruct Task 6 evidence manually from
  incompatible intermediate outputs.
- Rollback/mitigation: preserve direct consumer compatibility as part of Story 7
  closure.

### Engineering Task 6: Emit A Stable Story 7 Correctness Package Or Rerunnable Checker

**Implements story**
- `Story 7: One Machine-Reviewable Correctness Package Joins Positive And Negative Evidence`

**Change type**
- validation automation | docs

**Definition of done**
- Story 7 emits one stable machine-reviewable correctness package or rerunnable
  checker.
- The package supports later benchmark and publication consumers directly.
- The output is stable enough for Story 8 to add summary-consistency and
  counted-status guardrails without changing the underlying record shape
  materially.

**Execution checklist**
- [ ] Add a dedicated Story 7 validator under
      `benchmarks/density_matrix/correctness_evidence/`, with
      `correctness_bundle_validation.py` as the primary checker.
- [ ] Add a dedicated Story 7 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task6/story7_correctness_package/`).
- [ ] Emit Task 6 positive and negative records through one stable schema plus a
      stable bundle summary.
- [ ] Record rerun commands and software metadata with the emitted package.

**Evidence produced**
- One stable Story 7 correctness package or checker.
- One direct citation surface for the Task 6 shared evidence package.

**Risks / rollback**
- Risk: if Story 7 evidence remains scattered across story-local bundles only,
  later consumers will struggle to audit the full Task 6 surface.
- Rollback/mitigation: emit one stable shared correctness package and cite it
  directly.

### Engineering Task 7: Document The Story 7 Consumer Handoff And Run The Shared Package Surface

**Implements story**
- `Story 7: One Machine-Reviewable Correctness Package Joins Positive And Negative Evidence`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the Story 7 shared package rule and consumer
  expectations.
- The Story 7 package harness and bundle run successfully.
- Story 7 makes clear that summary-consistency and main-claim closure belong to
  Story 8.

**Execution checklist**
- [ ] Document the shared Task 6 package rule and consumer-facing field
      expectations.
- [ ] Explain how later benchmark and publication consumers should consume the
      Story 7 package directly.
- [ ] Run focused Story 7 regression coverage and verify
      `benchmarks/density_matrix/correctness_evidence/correctness_bundle_validation.py`.
- [ ] Record stable references to the Story 7 tests, checker, and emitted
      package.

**Evidence produced**
- Passing Story 7 correctness-package regression checks.
- One stable Story 7 correctness-package or checker reference.

**Risks / rollback**
- Risk: later work may misread Story 7 as already closing summary semantics
  rather than as the shared evidence package it is meant to provide.
- Rollback/mitigation: document the Story 7 handoff to Story 8 explicitly.

## Exit Criteria

Story 7 is complete only when all of the following are true:

- one explicit shared Task 6 correctness-package provenance tuple and bundle
  rule exist,
- positive and negative Task 6 evidence emit records through one stable shared
  package surface where fields overlap,
- later Task 5, Task 7, and Task 8 consumers can read one stable Task 6 package
  without manual relabeling,
- one stable Story 7 correctness package or rerunnable checker exists for direct
  citation,
- and summary-consistency plus main-claim closure remain clearly assigned to
  Story 8.

## Implementation Notes

- Prefer additive schema evolution over renaming shared Phase 3 fields.
- Keep Story 7 focused on stable evidence packaging, not yet on final
  counted-status interpretation.
- Treat consumer compatibility as part of the correctness-package contract, not
  as a downstream convenience.
