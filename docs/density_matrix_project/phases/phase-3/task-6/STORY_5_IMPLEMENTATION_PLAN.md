# Story 5 Implementation Plan

## Story Being Implemented

Story 5: Runtime And Fusion Classifications Stay Correctness-Comparable

This is a Layer 4 engineering plan for implementing the fifth behavioral slice
from `TASK_6_STORIES.md`.

## Scope

This story turns Task 6 into one runtime-path-aware correctness surface:

- plain partitioned, real fused, supported-but-unfused, and deferred runtime
  situations remain explicitly distinguishable in Task 6 evidence,
- the same correctness thresholds remain comparable across the supported runtime
  paths that can legitimately count as positive evidence,
- no supported path is silently relabeled as a different kind of success,
- and Story 5 closes the contract for "how runtime and fusion classifications
  participate in Task 6 correctness" without yet claiming unsupported-boundary
  closure, full-package assembly, or performance interpretation.

Out of scope for this story:

- correctness-matrix inventory already owned by Story 1,
- internal sequential-baseline gating already owned by Story 2,
- the bounded Qiskit Aer slice already owned by Story 3,
- output-integrity and continuity-energy emphasis already owned by Story 4,
- unsupported-boundary stage separation already owned by Story 6,
- full correctness-package assembly already owned by Story 7,
- counted-status propagation into later summaries already owned by Story 8,
- and any performance-benefit interpretation that belongs to Task 7.

## Dependencies And Assumptions

- Stories 1 through 4 already define the shared Task 6 case identity plus the
  internal, external, and output-integrity verdict surfaces Story 5 must carry
  across runtime-path labels honestly.
- The frozen source-of-truth contract is `TASK_6_MINI_SPEC.md`,
  `TASK_6_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-005`,
  `P3-ADR-008`, and `P3-ADR-010`.
- Task 3 already defines the baseline runtime-path and audit substrate Story 5
  should reuse through:
  - `benchmarks/density_matrix/partitioned_runtime/runtime_audit_validation.py`,
  - `runtime_output_validation.py`,
  - and `mandatory_workload_runtime_validation.py`.
- Task 4 already defines the fused classification and fused audit substrate Story
  5 should reuse through:
  - `benchmarks/density_matrix/partitioned_runtime/fused_classification_validation.py`,
  - `fused_runtime_audit_validation.py`,
  - `fused_semantics_validation.py`,
  - and `benchmarks/density_matrix/artifacts/phase3_task4/`.
- Story 5 should prefer additive extension of the shared Task 3 / Task 4 audit
  vocabulary over inventing a disconnected Task 6 classification language.
- The current implementation learning is that Story 5 should derive its runtime
  and fusion classification from the fused-capable shared positive records
  already built for Stories 2 through 4, rather than by reintroducing a second
  per-path execution matrix.
- Real fused-path cases can count positively only when they preserve the same
  correctness thresholds required by Stories 2 through 4.
- Supported-but-unfused and deferred situations must remain visible, but they do
  not automatically count as fused success.
- The natural implementation home for Task 6 runtime-classification validators
  is the new `benchmarks/density_matrix/correctness_evidence/` package, with
  `runtime_classification_validation.py` operating on the shared positive record
  surface from `records.py`.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 6 Runtime-Classification And Comparability Rule

**Implements story**
- `Story 5: Runtime And Fusion Classifications Stay Correctness-Comparable`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 defines one explicit Task 6 rule for how runtime-path and fusion
  classifications interact with correctness.
- The rule distinguishes which classifications can count positively and which
  remain boundary-only or non-counted.
- The rule is explicit enough that later Task 7 and Task 8 summaries can reuse
  it safely.

**Execution checklist**
- [ ] Freeze the supported runtime-path vocabulary for Task 6, including plain
      partitioned, real fused, supported-but-unfused, and deferred path
      semantics where they are relevant.
- [ ] Define which classifications can count as positive supported evidence and
      which remain visible but non-counted.
- [ ] Define the no-silent-relabeling rule for runtime-path and fusion status.
- [ ] Keep unsupported-boundary closure, package assembly, and performance
      interpretation outside the Story 5 bar.

**Evidence produced**
- One stable Task 6 runtime-classification rule.
- One explicit boundary between counted path classes and visible non-counted
  path classes.

**Risks / rollback**
- Risk: later summaries may overstate fused success or hide supported-but-
  unfused outcomes if Story 5 leaves classification semantics implicit.
- Rollback/mitigation: freeze the classification rule before packaging Story 5
  evidence.

### Engineering Task 2: Reuse The Shared Task 3 And Task 4 Audit Surfaces As The Base

**Implements story**
- `Story 5: Runtime And Fusion Classifications Stay Correctness-Comparable`

**Change type**
- docs | code

**Definition of done**
- Story 5 reuses the existing Task 3 runtime-audit and Task 4 fused-audit
  surfaces where they already match the contract.
- Task 6 path semantics remain auditable back to supported Task 3 / Task 4
  outputs.
- Story 5 avoids inventing a detached Task 6-only path language.

**Execution checklist**
- [ ] Reuse the shared Task 3 runtime-path labels and Task 4 fused-classification
      fields where they already fit Story 5 needs.
- [ ] Reuse overlapping audit fields from Task 3 and Task 4 directly where they
      match the Task 6 contract.
- [ ] Add only the Task 6-specific counted-status or comparability fields needed
      for Story 5.
- [ ] Document where Story 5 intentionally extends the shared audit vocabulary.

**Evidence produced**
- One reviewable mapping from Task 3 / Task 4 audit fields to the Task 6
  runtime-classification surface.
- One explicit boundary between reused fields and Story 5-specific extensions.

**Risks / rollback**
- Risk: Task 6 may create a disconnected classification language that later
  reviewers must translate mentally against the real runtime audit surfaces.
- Rollback/mitigation: align Story 5 with Task 3 / Task 4 audit vocabularies
  wherever practical.

### Engineering Task 3: Define A Shared Classification-Aware Task 6 Correctness Record

**Implements story**
- `Story 5: Runtime And Fusion Classifications Stay Correctness-Comparable`

**Change type**
- code | tests

**Definition of done**
- Story 5 defines one shared Task 6 record shape for runtime-path and fusion
  classification.
- The record carries runtime-path labels, fusion classification, and the
  threshold verdicts already defined by Stories 2 through 4 cleanly together.
- The shape remains stable across supported runtime situations.

**Execution checklist**
- [ ] Extend the shared Task 6 record with runtime-path and fusion classification
      fields in `benchmarks/density_matrix/correctness_evidence/records.py` or
      the smallest auditable successor.
- [ ] Keep Story 2 through Story 4 verdict fields structurally compatible across
      plain partitioned and fused-capable cases.
- [ ] Record whether a case is counted positively, visible but non-counted, or
      boundary-only due to its path classification.
- [ ] Add regression checks for classification-field presence and stability.

**Evidence produced**
- One stable Story 5 classification-aware Task 6 record shape.
- Regression checks for required classification-field stability.

**Risks / rollback**
- Risk: path semantics may remain implicit in separate artifact files and become
  hard to compare against correctness verdicts.
- Rollback/mitigation: attach path and classification semantics directly to the
  shared Task 6 record.

### Engineering Task 4: Build The Task 6 Runtime-Classification Validation Harness

**Implements story**
- `Story 5: Runtime And Fusion Classifications Stay Correctness-Comparable`

**Change type**
- code | validation automation

**Definition of done**
- Story 5 has one reusable harness for evaluating correctness-comparable
  runtime-path classifications.
- The harness records path labels, counted-status fields, and reused threshold
  verdicts beside the shared Story 1 case identity.
- The harness is reusable by Story 7 package assembly and Story 8 summary
  guardrails.

**Execution checklist**
- [ ] Add a dedicated Story 5 validation driver under
      `benchmarks/density_matrix/correctness_evidence/`, with
      `runtime_classification_validation.py` as the primary checker.
- [ ] Evaluate representative plain partitioned, fused, and supported-but-
      unfused cases through the shared Task 6 record surface.
- [ ] Record path labels, classification summaries, and counted-status semantics
      together with reused correctness verdicts.
- [ ] Keep the harness rooted in supported runtime outputs rather than in
      narrative-only classification notes.

**Evidence produced**
- One reusable Story 5 runtime-classification harness.
- One comparable record surface across supported runtime situations.

**Risks / rollback**
- Risk: runtime classification may remain a loose narrative overlay instead of a
  machine-reviewable part of Task 6 evidence.
- Rollback/mitigation: centralize Story 5 in one stable validation entry point.

### Engineering Task 5: Add A Representative Runtime-Classification Matrix Across Supported Paths

**Implements story**
- `Story 5: Runtime And Fusion Classifications Stay Correctness-Comparable`

**Change type**
- tests | validation automation

**Definition of done**
- Story 5 covers representative supported cases across the relevant runtime-path
  classifications.
- The matrix is broad enough to prove that the same correctness rules remain
  comparable across supported paths.
- The matrix remains representative and contract-driven rather than exhaustive
  over every possible fusion scenario.

**Execution checklist**
- [ ] Include at least one plain partitioned supported case.
- [ ] Include at least one real fused-path supported case where the current Task
      4 boundary makes that comparison real.
- [ ] Include at least one supported-but-unfused or deferred visible case proving
      it remains explicit and not mislabeled as fused success.
- [ ] Keep unsupported-boundary stage separation and summary-consistency
      semantics outside the Story 5 matrix.

**Evidence produced**
- One representative Story 5 runtime-classification matrix.
- One review surface for counted versus visible non-counted path semantics.

**Risks / rollback**
- Risk: Story 5 may look coherent on one runtime path while mislabeling another.
- Rollback/mitigation: freeze a small but path-spanning classification matrix
  early.

### Engineering Task 6: Emit A Stable Story 5 Runtime-Classification Bundle Or Rerunnable Checker

**Implements story**
- `Story 5: Runtime And Fusion Classifications Stay Correctness-Comparable`

**Change type**
- validation automation | docs

**Definition of done**
- Story 5 emits one stable machine-reviewable runtime-classification bundle or
  rerunnable checker.
- The bundle records path labels, fusion classifications, counted-status fields,
  and reused correctness verdicts through one stable schema.
- The output is stable enough for Story 7 full-package assembly and Story 8
  summary-consistency checks.

**Execution checklist**
- [ ] Add a dedicated Story 5 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task6/story5_runtime_classification/`).
- [ ] Emit shared case identity, runtime-path labels, fusion classification,
      counted-status fields, and reused verdict fields through one stable schema.
- [ ] Record rerun commands and software metadata with the emitted bundle.
- [ ] Keep path-specific counted-versus-visible semantics explicit in the bundle
      summary.

**Evidence produced**
- One stable Story 5 runtime-classification bundle or checker.
- One reusable classification surface for later Task 6 stories.

**Risks / rollback**
- Risk: prose-only Story 5 closure will make later reviewers unable to tell
  which runtime paths actually counted and which only remained visible.
- Rollback/mitigation: emit one machine-reviewable classification bundle
  directly.

### Engineering Task 7: Document The Story 5 Classification Rule And Run The Gate

**Implements story**
- `Story 5: Runtime And Fusion Classifications Stay Correctness-Comparable`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain how runtime-path and fusion classifications
  participate in Task 6 correctness.
- The Story 5 classification harness and bundle run successfully.
- Story 5 makes clear that unsupported-boundary closure, full-package assembly,
  and summary-consistency remain assigned to later stories.

**Execution checklist**
- [ ] Document the Story 5 counted-versus-visible path rule for Task 6.
- [ ] Explain how fused-path correctness remains comparable to plain partitioned
      correctness without turning supported-but-unfused cases into fused
      successes.
- [ ] Run focused Story 5 regression coverage and verify
      `benchmarks/density_matrix/correctness_evidence/runtime_classification_validation.py`.
- [ ] Record stable references to the Story 5 tests, checker, and emitted
      bundle.

**Evidence produced**
- Passing Story 5 runtime-classification regression checks.
- One stable Story 5 runtime-classification bundle or checker reference.

**Risks / rollback**
- Risk: later reviewers may mistake Story 5 path semantics for unsupported-
  boundary closure or performance closure.
- Rollback/mitigation: document the Story 5 handoff to Stories 6 through 8
  explicitly.

## Exit Criteria

Story 5 is complete only when all of the following are true:

- one explicit Task 6 rule exists for how runtime-path and fusion
  classifications interact with counted correctness evidence,
- path and fusion labels are attached directly to the shared Task 6 records they
  govern,
- representative supported path classes remain correctness-comparable without
  silent relabeling,
- one stable Story 5 runtime-classification bundle or rerunnable checker exists
  for later reuse,
- and unsupported-boundary closure, full-package assembly, summary-consistency
  guardrails, and performance interpretation remain clearly assigned to later
  stories.

## Implementation Notes

- Prefer additive extension of Task 3 / Task 4 audit fields over inventing a
  detached Task 6 path language.
- Keep Story 5 focused on correctness-comparable path semantics, not on whether
  one path is faster.
- Treat supported-but-unfused and deferred path situations as required visible
  evidence, not as embarrassing cases to hide.
