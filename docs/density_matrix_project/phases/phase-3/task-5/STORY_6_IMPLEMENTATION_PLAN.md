# Story 6 Implementation Plan

## Story Being Implemented

Story 6: Calibration Results And Provenance Are Emitted In One
Machine-Reviewable Shared Surface

This is a Layer 4 engineering plan for implementing the sixth behavioral slice
from `TASK_5_STORIES.md`.

## Scope

This story turns Task 5 calibration into one stable result and audit surface:

- calibration records, selected-plan summaries, provenance, correctness verdicts,
  and metric fields are emitted through one shared machine-reviewable bundle,
- the bundle extends the shared Phase 3 provenance vocabulary rather than
  replacing it,
- continuity, microcase, and structured-workload calibration slices remain
  structurally comparable where fields overlap,
- and Story 6 closes stable calibration-bundle packaging without yet claiming
  the final approximation or deferred-boundary summary.

Out of scope for this story:

- planner-candidate identity already owned by Story 1,
- workload-matrix anchoring already owned by Story 2,
- density-aware signal differentiation already owned by Story 3,
- correctness-gated evidence admissibility already owned by Story 4,
- supported-claim selection already owned by Story 5,
- and explicit approximation and deferred-boundary handling already owned by
  Story 7.

## Dependencies And Assumptions

- Stories 1 through 5 already define the candidate surface, workload matrix,
  density-aware signal surface, correctness gate, and supported-claim selection
  Story 6 must package consistently.
- The frozen source-of-truth contract is `TASK_5_MINI_SPEC.md`,
  `TASK_5_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-006`,
  `P3-ADR-008`, and `P3-ADR-009`.
- Task 1 through Task 4 already provide the core provenance and metric fields
  Story 6 should extend rather than rename, especially through:
  - the Task 1 and Task 2 provenance tuple,
  - `NoisyRuntimeExecutionResult`,
  - `build_runtime_audit_record()`,
  - and the existing artifact bundles under
    `benchmarks/density_matrix/artifacts/partitioned_runtime/` and
    `benchmarks/density_matrix/artifacts/partitioned_runtime/`.
- Story 6 should prefer additive schema evolution over inventing a disconnected
  Task 5 result language.
- The most conservative implementation path is to keep overlapping field names
  stable where practical and add Task 5 calibration fields explicitly.
- Later Task 6, Task 7, and publication packaging work should be able to consume
  the Task 5 bundle directly without per-workload parsing rules.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 5 Calibration Provenance Tuple And Bundle Rule

**Implements story**
- `Story 6: Calibration Results And Provenance Are Emitted In One Machine-Reviewable Shared Surface`

**Change type**
- docs | validation automation

**Definition of done**
- Story 6 defines one stable calibration provenance tuple and one bundle rule for
  supported Task 5 outputs.
- The rule is explicit enough that later validation, benchmark, and paper work
  can rely on it safely.
- The story distinguishes stable packaging from final approximation-boundary
  interpretation.

**Execution checklist**
- [ ] Freeze the minimum Task 5 provenance tuple around candidate identity,
      workload identity, planner settings, runtime-path classification,
      correctness verdicts, and calibration outputs.
- [ ] Freeze the minimum result-surface fields every supported Task 5 record must
      expose.
- [ ] Define which fields remain shared with earlier Phase 3 bundles unchanged
      and which are additive Task 5 extensions.
- [ ] Keep final approximation and deferred-boundary interpretation outside the
      Story 6 bar.

**Evidence produced**
- One stable Task 5 calibration provenance tuple.
- One explicit calibration-bundle rule for supported outputs.

**Risks / rollback**
- Risk: later Task 5 results may be individually plausible but structurally hard
  to compare.
- Rollback/mitigation: freeze the calibration bundle rule before broadening
  output production.

### Engineering Task 2: Reuse The Shared Phase 3 Provenance And Metric Surfaces As The Base

**Implements story**
- `Story 6: Calibration Results And Provenance Are Emitted In One Machine-Reviewable Shared Surface`

**Change type**
- docs | code

**Definition of done**
- Task 5 bundle packaging builds on the existing Phase 3 provenance and metric
  surfaces where fields overlap.
- Task 5-specific extensions are explicit and reviewable.
- Story 6 avoids creating a disconnected fifth result language.

**Execution checklist**
- [ ] Review the shared provenance fields already emitted by earlier Phase 3
      planner, descriptor, and runtime validators.
- [ ] Reuse overlapping field names directly where they already match the Task 5
      contract.
- [ ] Add only the Task 5-specific fields needed for candidate identity,
      calibration outputs, and claim-selection review.
- [ ] Document where Task 5 intentionally extends the shared vocabulary.

**Evidence produced**
- One reviewable mapping from shared Phase 3 fields to Task 5 calibration
  fields.
- One explicit boundary between reused vocabulary and Task 5-specific
  extensions.

**Risks / rollback**
- Risk: Task 5 may create a disconnected result language that later reviewers
  must translate mentally against earlier bundles.
- Rollback/mitigation: align Task 5 output packaging with earlier surfaces
  wherever practical.

### Engineering Task 3: Define A Shared Calibration Record And Summary Surface

**Implements story**
- `Story 6: Calibration Results And Provenance Are Emitted In One Machine-Reviewable Shared Surface`

**Change type**
- code | tests

**Definition of done**
- Supported Task 5 outputs emit one shared calibration record shape.
- The record separates case-level provenance, candidate identity, metric
  summaries, correctness verdicts, and claim-selection summaries cleanly.
- The shape is stable across supported Task 5 cases.

**Execution checklist**
- [ ] Define one shared top-level Task 5 calibration record shape.
- [ ] Record case-level provenance separately from candidate settings, metric
      fields, correctness verdicts, and claim-selection fields.
- [ ] Keep supported Task 5 records machine-readable and structurally stable.
- [ ] Add regression checks for top-level schema stability.

**Evidence produced**
- One stable shared Task 5 calibration record shape.
- Regression checks for schema stability across supported cases.

**Risks / rollback**
- Risk: later Task 5 bundles may remain plausible case by case but structurally
  incomparable.
- Rollback/mitigation: freeze one shared calibration record shape before
  broadening bundle emission.

### Engineering Task 4: Cross-Check Continuity, Microcase, And Structured Slices Against The Shared Calibration Surface

**Implements story**
- `Story 6: Calibration Results And Provenance Are Emitted In One Machine-Reviewable Shared Surface`

**Change type**
- tests

**Definition of done**
- Continuity, microcase, and structured-workload Task 5 slices emit records
  through the same shared calibration surface where they overlap.
- Schema drift across supported cases is caught early.
- The checks stay focused on record stability rather than final approximation
  interpretation.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_planner_calibration.py` for
      calibration-record stability across representative supported cases.
- [ ] Compare top-level provenance, candidate identity, metric-field presence,
      correctness-verdict fields, and claim-selection fields across the
      supported cases.
- [ ] Keep the checks narrow to output and audit structure rather than final
      Story 7 boundary language.
- [ ] Fail quickly when supported Task 5 cases diverge from the shared record
      surface.

**Evidence produced**
- Fast regression coverage for cross-case Task 5 calibration output stability.
- Reviewable comparison checks for the shared calibration surface.

**Risks / rollback**
- Risk: bundle drift may remain hidden until later benchmark rollups or paper
  packaging.
- Rollback/mitigation: enforce cross-case output checks early.

### Engineering Task 5: Preserve Direct Compatibility With Later Validation, Benchmark, And Paper Consumers

**Implements story**
- `Story 6: Calibration Results And Provenance Are Emitted In One Machine-Reviewable Shared Surface`

**Change type**
- code | tests | validation automation

**Definition of done**
- Supported Task 5 outputs remain directly consumable by later validation and
  benchmark work.
- The calibration bundle does not require re-running the same cases through a
  different interface just to package them.
- Story 6 preserves direct compatibility with later Task 6, Task 7, and paper
  evidence assembly where fields overlap.

**Execution checklist**
- [ ] Keep candidate, workload, metric, and correctness fields in a stable
      machine-readable form suitable for later consumers.
- [ ] Keep supported-claim selection outputs directly attached to the same record
      shape rather than emitting a disconnected second summary file only.
- [ ] Add focused checks proving later consumers can read one stable Task 5
      bundle surface.
- [ ] Avoid per-workload parsing rules or notebook-only summaries.

**Evidence produced**
- One direct-consumer-compatible Task 5 bundle surface.
- Focused regression coverage for consumer-facing field stability.

**Risks / rollback**
- Risk: later review work may need to reconstruct Task 5 evidence manually from
  incompatible intermediate outputs.
- Rollback/mitigation: preserve direct consumer compatibility as part of Story 6
  closure.

### Engineering Task 6: Emit A Stable Story 6 Calibration Bundle Or Rerunnable Checker

**Implements story**
- `Story 6: Calibration Results And Provenance Are Emitted In One Machine-Reviewable Shared Surface`

**Change type**
- validation automation | docs

**Definition of done**
- Story 6 emits one stable machine-reviewable calibration bundle or rerunnable
  checker.
- The bundle supports later validation, benchmark, and publication consumers
  directly.
- The output is stable enough for Story 7 to add explicit boundary summaries
  without changing the underlying record shape materially.

**Execution checklist**
- [ ] Add a dedicated Story 6 validator under
      `benchmarks/density_matrix/planner_calibration/`, with
      `calibration_bundle_validation.py` as the primary checker.
- [ ] Add a dedicated Story 6 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/planner_calibration/calibration_bundle/`).
- [ ] Emit supported Task 5 records through one stable schema plus a stable
      bundle summary.
- [ ] Record rerun commands and software metadata with the emitted bundle.

**Evidence produced**
- One stable Story 6 calibration bundle or checker.
- One direct citation surface for the Task 5 supported calibration package.

**Risks / rollback**
- Risk: if Task 5 evidence remains scattered across logs or notebooks, later
  reviewers will struggle to audit the claim surface.
- Rollback/mitigation: emit one stable calibration bundle and cite it directly.

### Engineering Task 7: Document The Story 6 Consumer Handoff And Run The Shared Surface

**Implements story**
- `Story 6: Calibration Results And Provenance Are Emitted In One Machine-Reviewable Shared Surface`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the Task 5 shared bundle rule and consumer
  expectations.
- The Story 6 calibration harness and bundle run successfully.
- Story 6 makes clear that explicit approximation and deferred-boundary language
  belongs to Story 7.

**Execution checklist**
- [ ] Document the shared Task 5 calibration record rule and consumer-facing
      field expectations.
- [ ] Explain how later validators and paper packaging should consume the Story 6
      bundle directly.
- [ ] Run focused Story 6 regression coverage and verify
      `benchmarks/density_matrix/planner_calibration/calibration_bundle_validation.py`.
- [ ] Record stable references to the Story 6 tests, checker, and emitted
      bundle.

**Evidence produced**
- Passing Story 6 calibration-bundle regression checks.
- One stable Story 6 calibration-bundle or checker reference.

**Risks / rollback**
- Risk: later work may misread Story 6 as the final Story 7 boundary summary
  instead of the shared evidence surface it is meant to provide.
- Rollback/mitigation: document the handoff to Story 7 explicitly.

## Exit Criteria

Story 6 is complete only when all of the following are true:

- one explicit shared Task 5 calibration provenance tuple and bundle rule exist,
- continuity, microcase, and structured-workload slices emit records through one
  stable shared Task 5 calibration surface where fields overlap,
- later validation, benchmark, and paper consumers can read one stable Task 5
  bundle without workload-specific parsing rules,
- one stable Story 6 calibration bundle or rerunnable checker exists for direct
  citation,
- and explicit approximation and deferred-boundary handling remain clearly
  assigned to Story 7.

## Implementation Notes

- Prefer additive schema evolution over renaming shared Phase 3 fields.
- Keep Story 6 focused on stable evidence packaging, not yet on the final
  boundary language for what remains approximate or deferred.
- Treat consumer compatibility as part of the contract, not as a downstream
  convenience.
