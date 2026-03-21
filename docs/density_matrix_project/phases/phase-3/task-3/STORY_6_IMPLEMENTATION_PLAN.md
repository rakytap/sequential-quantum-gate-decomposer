# Story 6 Implementation Plan

## Story Being Implemented

Story 6: Runtime Provenance And Path Labels Stay Stable Across Supported Cases

This is a Layer 4 engineering plan for implementing the sixth behavioral slice
from `TASK_3_STORIES.md`.

## Scope

This story turns Task 3 runtime reviewability into one stable cross-workload
evidence surface:

- supported runtime outputs reuse one stable case-level provenance vocabulary,
- runtime-path labels remain consistent across continuity, microcase, and
  structured-family workloads,
- one reusable runtime-audit bundle or rerunnable checker becomes the review
  surface for later correctness, benchmark, and publication work,
- and Story 6 closes cross-workload audit stability without claiming unsupported
  runtime taxonomy closure or final numerical verdict packaging.

Out of scope for this story:

- positive continuity and shared mandatory-workload coverage already owned by
  Stories 1 and 2,
- direct descriptor-to-runtime handoff owned by Story 3,
- partition-local semantic stress owned by Story 4,
- stable comparison-ready output packaging owned by Story 5,
- runtime-stage unsupported-boundary closure owned by Story 7,
- and real fused execution, full correctness-threshold packaging, or
  performance summary analysis owned by later Phase 3 tasks.

## Dependencies And Assumptions

- Stories 1 through 5 already define the supported positive runtime slices that
  Story 6 must package into one stable audit surface.
- The frozen source-of-truth contract is `TASK_3_MINI_SPEC.md`,
  `TASK_3_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-005`,
  `P3-ADR-008`, and `P3-ADR-009`.
- Task 1 already froze a planner-entry provenance tuple and Task 2 already froze
  a descriptor-audit provenance tuple in `squander/partitioning/noisy_planner.py`.
  Story 6 should extend that vocabulary rather than replace it.
- Story 5 of Task 2 already emitted a shared descriptor-audit bundle under
  `benchmarks/density_matrix/artifacts/planner_surface/descriptor_audit/`; Story 6
  should keep overlapping provenance and schema identity fields aligned with
  that positive descriptor surface.
- Story 5 of Task 3 now also defines the positive runtime result surface whose
  exact-output and observable fields Story 6 must package into one stable audit
  record.
- The likely shared implementation substrate therefore remains:
  - the Task 2 descriptor module in `squander/partitioning/noisy_planner.py`,
  - the Task 3 runtime layer in `squander/partitioning/noisy_runtime.py`,
  - and artifact-emission helpers under
    `benchmarks/density_matrix/partitioned_runtime/`.
- Story 6 should prefer one shared case-level provenance vocabulary and one
  stable runtime-path classification over workload-local labeling conventions.
- The runtime-audit surface should remain review-oriented and machine-readable.
  It is not intended to become a second execution API.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 3 Case-Level Provenance Tuple And Runtime-Path Vocabulary

**Implements story**
- `Story 6: Runtime Provenance And Path Labels Stay Stable Across Supported Cases`

**Change type**
- docs | validation automation

**Definition of done**
- Story 6 defines one stable case-level provenance tuple for supported runtime
  cases.
- The runtime-path classification rule is explicit enough that later outputs can
  be compared safely.
- The story distinguishes audit stability from unsupported or final-verdict
  behavior.

**Execution checklist**
- [ ] Freeze the Task 3 case-level provenance tuple around planner schema
      version, descriptor schema version, requested mode, source type, entry
      route, workload family, workload ID, and qubit count.
- [ ] Freeze one runtime-path vocabulary covering the plain partitioned baseline
      and any later clearly labeled fused extension.
- [ ] Freeze the minimum runtime-summary fields every supported case must
      expose, such as partition count and partition-span summary.
- [ ] Keep unsupported-diagnostic stability for Story 7 rather than overloading
      Story 6.

**Evidence produced**
- One stable Task 3 runtime provenance tuple.
- One explicit runtime-path classification rule for supported runtime artifacts.

**Risks / rollback**
- Risk: if provenance and runtime-path labels remain loose, later bundles will
  be hard to compare and easy to misread.
- Rollback/mitigation: freeze the case-level audit vocabulary before broadening
  artifact production.

### Engineering Task 2: Reuse Task 1 And Task 2 Audit Vocabulary As The Runtime-Audit Base

**Implements story**
- `Story 6: Runtime Provenance And Path Labels Stay Stable Across Supported Cases`

**Change type**
- docs | code

**Definition of done**
- Task 3 runtime audit builds on the existing Task 1 and Task 2 provenance
  vocabulary where the fields overlap.
- Runtime-specific extensions are explicit and reviewable.
- Story 6 avoids creating a disconnected third audit language.

**Execution checklist**
- [ ] Review the Task 1 planner-entry and Task 2 descriptor-audit provenance
      fields already used in supported artifacts.
- [ ] Reuse overlapping labels directly where they already match the Task 3
      runtime contract.
- [ ] Add only the runtime-specific fields needed for runtime-path
      classification, partition execution summaries, and exact-output/result
      references.
- [ ] Document where Task 3 intentionally extends the earlier audit
      vocabulary.

**Evidence produced**
- One reviewable mapping from Task 1 and Task 2 audit fields to Task 3 runtime
  audit fields.
- One explicit boundary between reused vocabulary and Task 3-specific
  extensions.

**Risks / rollback**
- Risk: Task 3 may create a third audit language that later reviewers must
  translate mentally against Task 1 and Task 2.
- Rollback/mitigation: align the runtime-audit base with Task 1 and Task 2
  wherever practical.

### Engineering Task 3: Define A Shared Runtime-Audit Record And Summary Surface

**Implements story**
- `Story 6: Runtime Provenance And Path Labels Stay Stable Across Supported Cases`

**Change type**
- code | tests

**Definition of done**
- Supported runtime cases emit one shared runtime-audit record shape.
- The record separates case-level provenance, runtime-summary metadata, and
  detailed result references cleanly.
- The record shape is stable across workload classes.

**Execution checklist**
- [ ] Define one shared top-level record shape for Task 3 runtime-audit output.
- [ ] Record case-level provenance separately from runtime-summary fields and
      detailed result references.
- [ ] Add summary fields that make later review efficient without replacing the
      exact-output records themselves.
- [ ] Keep the audit shape machine-readable and stable across supported
      workloads.

**Evidence produced**
- One stable shared runtime-audit record shape.
- Regression checks for top-level schema stability across supported cases.

**Risks / rollback**
- Risk: later outputs may remain individually reasonable but structurally
  incomparable.
- Rollback/mitigation: freeze one shared record shape before broadening bundle
  emission.

### Engineering Task 4: Cross-Check Supported Workload Classes Against The Shared Runtime-Audit Surface

**Implements story**
- `Story 6: Runtime Provenance And Path Labels Stay Stable Across Supported Cases`

**Change type**
- tests

**Definition of done**
- Continuity, microcase, and structured-family slices emit audit records through
  the same shared runtime surface.
- Audit-shape drift across workload classes is caught early.
- The checks stay focused on audit stability rather than final numerical
  correctness.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_partitioned_runtime.py` for
      workload-class runtime-audit stability.
- [ ] Compare top-level provenance, runtime-path classification, schema
      identity, and summary-field presence across supported workload classes.
- [ ] Keep the checks narrow to audit structure rather than to final numerical
      threshold outcomes.
- [ ] Fail quickly when supported workload classes diverge from the shared
      runtime-audit contract.

**Evidence produced**
- Fast regression coverage for cross-workload runtime-audit stability.
- Reviewable workload-class comparison checks for the audit surface.

**Risks / rollback**
- Risk: runtime-audit drift may remain hidden until paper packaging or broad
  benchmark work.
- Rollback/mitigation: enforce cross-workload audit checks early.

### Engineering Task 5: Emit A Stable Story 6 Runtime-Audit Bundle

**Implements story**
- `Story 6: Runtime Provenance And Path Labels Stay Stable Across Supported Cases`

**Change type**
- validation automation | docs

**Definition of done**
- Story 6 emits one stable machine-reviewable runtime-audit bundle or rerunnable
  checker.
- The bundle records supported cases across the mandatory Task 3 workload
  surface.
- The bundle is reusable by later correctness, benchmark, and publication work.

**Execution checklist**
- [ ] Add a dedicated Story 6 artifact location
      (for example `benchmarks/density_matrix/artifacts/partitioned_runtime/runtime_audit/`).
- [ ] Emit at least one continuity case and one methods-oriented case through
      the shared Task 3 runtime-audit surface.
- [ ] Include schema identity, case-level provenance, runtime-path labels,
      summary metadata, and result references in the emitted output.
- [ ] Record rerun commands and software metadata with the bundle.

**Evidence produced**
- One stable Task 3 runtime-audit bundle or checker.
- One reusable output surface for later benchmark and paper packaging.

**Risks / rollback**
- Risk: if Story 6 emits only ad hoc local artifacts, later tasks will still
  lack one canonical runtime evidence surface.
- Rollback/mitigation: emit one stable shared bundle and treat it as canonical.

### Engineering Task 6: Document And Run The Story 6 Runtime-Auditability Gate

**Implements story**
- `Story 6: Runtime Provenance And Path Labels Stay Stable Across Supported Cases`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain how to inspect supported Task 3 runtime
  evidence consistently.
- Fast regression coverage and the Story 6 bundle run successfully.
- Story 6 closes with a stable review path for runtime auditability.

**Execution checklist**
- [ ] Document the shared Task 3 provenance tuple, runtime-path vocabulary, and
      runtime-audit record structure.
- [ ] Explain how Story 6 extends Task 1 and Task 2 auditability without
      replacing the earlier contracts.
- [ ] Run focused audit-stability checks and verify
      `benchmarks/density_matrix/partitioned_runtime/runtime_audit_validation.py`.
- [ ] Record stable test and artifact references for Story 7 and later Phase 3
      tasks.

**Evidence produced**
- Passing Story 6 runtime-auditability checks.
- One stable Story 6 runtime-audit bundle or checker reference.

**Risks / rollback**
- Risk: Story 6 may appear complete while still leaving contributors unsure how
  to compare supported runtime outputs consistently.
- Rollback/mitigation: document the audit surface and require a rerunnable
  bundle.

## Exit Criteria

Story 6 is complete only when all of the following are true:

- supported runtime outputs reuse one stable Task 3 case-level provenance tuple,
- one shared runtime-audit record shape is used across supported workload
  classes,
- fast regression coverage detects audit-shape drift across supported cases,
- one stable Story 6 runtime-audit bundle or checker exists for later reuse,
- and unsupported runtime-boundary closure remains clearly assigned to Story 7.

## Implementation Notes

- Prefer one shared case-level provenance vocabulary over workload-local audit
  naming conventions.
- Keep the runtime-audit surface machine-reviewable and review-oriented, not
  execution-oriented.
- Treat Story 6 as the place where Task 3 evidence becomes stable enough for
  later publication and benchmark packaging.
