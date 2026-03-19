# Story 5 Implementation Plan

## Story Being Implemented

Story 5: Descriptor Provenance And Audit Evidence Stay Stable Across Supported
Cases

This is a Layer 4 engineering plan for implementing the fifth behavioral slice
from `TASK_2_STORIES.md`.

## Scope

This story turns Task 2 auditability into one stable cross-workload descriptor
evidence surface:

- supported descriptor outputs reuse one stable case-level provenance
  vocabulary,
- descriptor schema identity and audit-summary fields remain consistent across
  continuity, microcase, structured-family, and supported exact-lowering paths,
- one reusable descriptor-audit bundle or rerunnable checker becomes the review
  surface for later runtime, validation, and benchmark work,
- and Story 5 closes cross-workload audit stability without claiming unsupported
  taxonomy closure or runtime numerical results.

Out of scope for this story:

- positive workload coverage already owned by Stories 1 and 2,
- exact order/noise semantics already owned by Story 3,
- reconstructible remapping and parameter-routing semantics already owned by
  Story 4,
- unsupported or lossy descriptor-boundary closure owned by Story 6,
- and runtime exactness, fused execution, or performance claims owned by later
  Phase 3 tasks.

## Dependencies And Assumptions

- Stories 1 through 4 already define the positive supported descriptor slices
  that Story 5 must package into one stable audit surface.
- Task 1 already froze a concrete planner-entry provenance tuple and a
  schema-versioned audit surface that Story 5 should extend rather than replace.
- The shared Task 2 descriptor substrate now lives in
  `squander/partitioning/noisy_planner.py`, and Story 1 already emits the first
  positive artifact bundle under
  `benchmarks/density_matrix/artifacts/phase3_task2/story1_continuity_descriptors/`.
- Story 2 now also emits the first multi-workload descriptor bundle under
  `benchmarks/density_matrix/artifacts/phase3_task2/story2_workloads/`, which
  should become a direct input to Story 5 audit-shape stabilization.
- Story 4 now also froze the positive reconstruction field set around
  `local_to_global_qbits`, `global_to_local_qbits`, `requires_remap`, and
  `parameter_routing`, plus the shared validation helper
  `validate_partition_descriptor_set()` in
  `squander/partitioning/noisy_planner.py`.
- The likely shared implementation substrate therefore remains the Task 2
  descriptor module under `squander/partitioning/` plus the existing
  artifact-emission surfaces under `benchmarks/density_matrix/artifacts/`.
- Story 5 should prefer one shared case-level provenance vocabulary over
  workload-local naming conventions.
- Story 5 should keep operation-level provenance lightweight and preserve most
  provenance at the case and artifact level unless a supported descriptor case
  truly needs more.
- The descriptor-audit surface should remain review-oriented and
  machine-readable. It is not intended to become a second runtime API.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 2 Case-Level Provenance Tuple And Schema Identity Rule

**Implements story**
- `Story 5: Descriptor Provenance And Audit Evidence Stay Stable Across Supported Cases`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 defines one stable case-level provenance tuple for supported
  descriptors.
- The schema identity rule is explicit enough that later outputs can be
  compared safely.
- The story distinguishes provenance stability from runtime or unsupported-case
  behavior.

**Execution checklist**
- [ ] Freeze the Task 2 case-level provenance tuple around descriptor schema
      version, requested mode, source type, entry route, workload family, and
      workload ID.
- [ ] Define how descriptor-specific audit fields extend the Task 1 provenance
      vocabulary without replacing it.
- [ ] Freeze the minimum descriptor-audit summary fields that every supported
      case must expose.
- [ ] Keep unsupported-diagnostic stability for Story 6 rather than overloading
      Story 5.

**Evidence produced**
- One stable Task 2 provenance tuple.
- One explicit schema-identity rule for supported descriptor artifacts.

**Risks / rollback**
- Risk: if provenance and schema identity remain loose, later bundles will be
  hard to compare and easy to misread.
- Rollback/mitigation: freeze the case-level audit vocabulary before expanding
  artifact production.

### Engineering Task 2: Reuse The Task 1 Planner-Entry Vocabulary As The Descriptor-Audit Base

**Implements story**
- `Story 5: Descriptor Provenance And Audit Evidence Stay Stable Across Supported Cases`

**Change type**
- docs | code

**Definition of done**
- Task 2 descriptor audit builds on the existing Task 1 planner-entry
  vocabulary where the fields overlap.
- Descriptor-specific extensions are explicit and reviewable.
- Story 5 avoids creating a disconnected second audit language.

**Execution checklist**
- [ ] Review the Task 1 planner-entry provenance and audit fields already used
      in supported artifacts.
- [ ] Reuse overlapping labels directly where they already match the Task 2
      contract.
- [ ] Add only the descriptor-specific fields needed for partition ordering,
      partition span, and reconstructibility summaries.
- [ ] Document where Task 2 intentionally extends the Task 1 vocabulary.

**Evidence produced**
- One reviewable mapping from Task 1 audit fields to Task 2 descriptor audit
  fields.
- One explicit boundary between reused vocabulary and Task 2-specific
  extensions.

**Risks / rollback**
- Risk: Task 2 may create a second audit language that later reviewers must
  translate mentally against Task 1.
- Rollback/mitigation: align the descriptor-audit base with Task 1 wherever
  practical.

### Engineering Task 3: Define A Shared Descriptor-Audit Record And Summary Surface

**Implements story**
- `Story 5: Descriptor Provenance And Audit Evidence Stay Stable Across Supported Cases`

**Change type**
- code | tests

**Definition of done**
- Supported cases emit one shared descriptor-audit record shape.
- The record separates case-level provenance, descriptor-summary metadata, and
  detailed descriptor payloads cleanly.
- The record shape is stable across workload classes.

**Execution checklist**
- [ ] Define one shared top-level record shape for descriptor-audit output.
- [ ] Record case-level provenance separately from descriptor-summary fields and
      detailed descriptor payloads.
- [ ] Add summary fields that make later review efficient without replacing the
      detailed descriptor content.
- [ ] Keep the audit shape machine-readable and stable across supported
      workloads.

**Evidence produced**
- One stable shared descriptor-audit record shape.
- Regression checks for top-level schema stability across supported cases.

**Risks / rollback**
- Risk: later outputs may remain individually reasonable but structurally
  incomparable.
- Rollback/mitigation: freeze one shared record shape before broadening bundle
  emission.

### Engineering Task 4: Cross-Check Supported Workload Classes Against The Shared Audit Surface

**Implements story**
- `Story 5: Descriptor Provenance And Audit Evidence Stay Stable Across Supported Cases`

**Change type**
- tests

**Definition of done**
- Continuity, microcase, structured-family, and any supported exact-lowering
  slices emit audit records through the same shared surface.
- Audit shape drift across workload classes is caught early.
- The checks stay focused on audit stability rather than runtime exactness.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_phase3_task2.py` for
      workload-class audit stability.
- [ ] Compare top-level provenance, schema identity, and summary-field presence
      across supported workload classes.
- [ ] Keep the checks narrow to audit structure rather than numerical behavior.
- [ ] Fail quickly when supported workload classes diverge from the shared
      descriptor-audit contract.

**Evidence produced**
- Fast regression coverage for cross-workload audit stability.
- Reviewable workload-class comparison checks for the audit surface.

**Risks / rollback**
- Risk: audit drift may remain hidden until paper packaging or broad benchmark
  work.
- Rollback/mitigation: enforce cross-workload audit checks early.

### Engineering Task 5: Emit A Stable Task 2 Descriptor-Audit Bundle

**Implements story**
- `Story 5: Descriptor Provenance And Audit Evidence Stay Stable Across Supported Cases`

**Change type**
- validation automation | docs

**Definition of done**
- Story 5 emits one stable machine-reviewable descriptor-audit bundle or
  rerunnable checker.
- The bundle records supported cases across the mandatory workload surface.
- The bundle is reusable by later runtime, validation, and publication work.

**Execution checklist**
- [ ] Add a dedicated Story 5 artifact location
      (for example `benchmarks/density_matrix/artifacts/phase3_task2/story5_audit/`).
- [ ] Emit at least one continuity case and one methods-oriented case through
      the shared Task 2 audit surface.
- [ ] Include descriptor schema identity, case-level provenance, summary
      metadata, and detailed descriptor payloads in the emitted output.
- [ ] Record rerun commands and software metadata with the bundle.

**Evidence produced**
- One stable Task 2 descriptor-audit bundle or checker.
- One reusable output surface for later benchmark and paper packaging.

**Risks / rollback**
- Risk: if Story 5 emits only ad hoc local artifacts, later tasks will still
  lack one canonical descriptor evidence surface.
- Rollback/mitigation: emit one stable shared bundle and treat it as canonical.

### Engineering Task 6: Document And Run The Story 5 Auditability Gate

**Implements story**
- `Story 5: Descriptor Provenance And Audit Evidence Stay Stable Across Supported Cases`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain how to inspect supported Task 2 descriptor
  evidence consistently.
- Fast regression coverage and the Story 5 bundle run successfully.
- Story 5 closes with a stable review path for descriptor auditability.

**Execution checklist**
- [ ] Document the shared Task 2 provenance tuple and audit-record structure.
- [ ] Explain how Story 5 extends Task 1 auditability without replacing the Task
      1 planner-entry contract.
- [ ] Run focused audit-stability checks and verify the emitted bundle.
- [ ] Record stable test and artifact references for Story 6 and later Phase 3
      tasks.

**Evidence produced**
- Passing Story 5 audit-stability checks.
- One stable Story 5 audit bundle or checker reference.

**Risks / rollback**
- Risk: Story 5 may appear complete while still leaving contributors unsure how
  to compare supported descriptor outputs consistently.
- Rollback/mitigation: document the audit surface and require a rerunnable
  bundle.

## Exit Criteria

Story 5 is complete only when all of the following are true:

- supported descriptor outputs reuse one stable Task 2 case-level provenance
  tuple,
- one shared descriptor-audit record shape is used across supported workload
  classes,
- fast regression coverage detects audit-shape drift across supported cases,
- one stable Story 5 descriptor-audit bundle or checker exists for later reuse,
- and unsupported-boundary closure remains clearly assigned to Story 6.

## Implementation Notes

- Prefer one shared case-level provenance vocabulary over workload-local audit
  naming conventions.
- Keep the descriptor-audit surface machine-reviewable and review-oriented, not
  runtime-oriented.
- Treat Story 5 as the place where Task 2 evidence becomes stable enough for
  later publication packaging.
