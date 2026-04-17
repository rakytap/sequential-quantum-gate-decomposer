# Story 3 Implementation Plan

## Story Being Implemented

Story 3: Descriptor-To-Runtime Handoff Is Direct, Auditable, And Shared

This is a Layer 4 engineering plan for implementing the third behavioral slice
from `TASK_3_STORIES.md`.

## Scope

This story turns Task 3 runtime audibility into one explicit handoff contract:

- the supported runtime consumes the validated Task 2 descriptor contract
  directly rather than a second private runtime-only schema,
- runtime preparation exposes one reviewable handoff surface, such as an
  execution-plan summary or per-partition execution record,
- supported continuity, microcase, and structured-family cases all use that same
  handoff contract,
- and Story 3 closes direct handoff auditability without claiming partition-local
  semantic stress closure, final result-shape closure, or unsupported runtime
  taxonomy closure.

Out of scope for this story:

- positive continuity closure already owned by Story 1,
- shared mandatory-workload coverage already owned by Story 2,
- partition-local semantic stress closure owned by Story 4,
- stable comparison-ready output packaging owned by Story 5,
- cross-workload runtime provenance stability owned by Story 6,
- runtime-stage unsupported-boundary closure owned by Story 7,
- and real fused execution, full correctness-threshold analysis, or performance
  conclusions owned by later Phase 3 tasks.

## Dependencies And Assumptions

- Stories 1 and 2 already define the supported positive runtime slices whose
  handoff Story 3 must stabilize.
- The frozen source-of-truth contract is `TASK_3_MINI_SPEC.md`,
  `TASK_3_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-003`, and
  `P3-ADR-005`.
- The shared Task 2 descriptor substrate already lives in
  `squander/partitioning/noisy_planner.py`, where:
  - `NoisyPartitionDescriptorSet`, `NoisyPartitionDescriptor`, and
    `NoisyPartitionDescriptorMember` already define the schema-versioned
    partition handoff contract,
  - `validate_partition_descriptor_set()` and
    `validate_partition_descriptor_set_against_surface()` already validate that
    contract,
  - and `build_descriptor_audit_record()` already emits a positive
    machine-reviewable Task 2 audit surface.
- Story 5 of Task 2 already emitted one shared supported descriptor-audit bundle
  under `benchmarks/density_matrix/artifacts/planner_surface/descriptor_audit/`; Story
  3 should extend that evidence line into runtime handoff records rather than
  inventing a disconnected runtime bookkeeping language.
- The likely shared implementation substrate therefore remains:
  - the Task 2 descriptor module in `squander/partitioning/noisy_planner.py`,
  - the shared Task 3 runtime layer in `squander/partitioning/noisy_runtime.py`,
  - and validation or artifact-emission helpers under
    `benchmarks/density_matrix/partitioned_runtime/`.
- Story 3 should prefer one direct descriptor-to-runtime handoff contract over
  runtime-local translation layers that later reviewers cannot audit.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 3 Descriptor-To-Runtime Handoff Rule

**Implements story**
- `Story 3: Descriptor-To-Runtime Handoff Is Direct, Auditable, And Shared`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 defines one explicit handoff rule for supported runtime execution.
- The rule distinguishes direct descriptor consumption from private runtime-only
  reinterpretation.
- The story keeps semantic-stress, output, and unsupported-boundary closure out
  of the Story 3 bar.

**Execution checklist**
- [ ] Freeze the supported meaning of direct runtime consumption of
      `NoisyPartitionDescriptorSet`.
- [ ] Define what counts as a disallowed second private runtime-only schema or
      hidden planner-state dependency.
- [ ] Define the minimal positive Story 3 handoff evidence needed for review.
- [ ] Keep partition-local semantics, final result-shape closure, and runtime
      unsupported-taxonomy closure outside the Story 3 bar.

**Evidence produced**
- One stable Story 3 descriptor-to-runtime handoff contract.
- One clear boundary between direct handoff auditability and later runtime
  concerns.

**Risks / rollback**
- Risk: if Story 3 defines runtime handoff too loosely, later Task 3 evidence
  will not prove that the runtime actually consumed the accepted Task 2
  contract.
- Rollback/mitigation: freeze the direct handoff rule before widening runtime
  packaging.

### Engineering Task 2: Define A Shared Runtime Request Or Execution-Plan Record Shape

**Implements story**
- `Story 3: Descriptor-To-Runtime Handoff Is Direct, Auditable, And Shared`

**Change type**
- code | tests

**Definition of done**
- Supported runtime preparation emits one shared request or execution-plan record
  shape.
- The record roots runtime preparation in planner and descriptor schema identity.
- The record stays reviewable across supported workload classes.

**Execution checklist**
- [ ] Define one shared top-level handoff record shape for Task 3 runtime
      preparation.
- [ ] Include planner schema version, descriptor schema version, workload
      identity, partition count, and partition-level execution summaries or
      equivalent direct handoff fields.
- [ ] Separate descriptor-derived runtime preparation metadata from later result
      packaging fields.
- [ ] Keep the handoff record machine-reviewable and workload-agnostic.

**Evidence produced**
- One stable Story 3 handoff record shape.
- Regression checks for top-level handoff-shape stability.

**Risks / rollback**
- Risk: later runtime outputs may remain individually reasonable but still be
  structurally incomparable at the handoff boundary.
- Rollback/mitigation: freeze one shared handoff record shape before broadening
  runtime packaging.

### Engineering Task 3: Reuse Task 2 Descriptor Fields Directly In Runtime Preparation

**Implements story**
- `Story 3: Descriptor-To-Runtime Handoff Is Direct, Auditable, And Shared`

**Change type**
- code | tests

**Definition of done**
- Supported runtime preparation reuses Task 2 descriptor fields directly.
- Runtime preparation does not require a hidden second schema to recover
  partition membership, canonical references, or partition spans.
- The implementation remains visibly rooted in the validated descriptor
  contract.

**Execution checklist**
- [ ] Reuse `canonical_operation_indices`, `members`, `local_to_global_qbits`,
      and `parameter_routing` or the smallest auditable successors directly in
      runtime preparation.
- [ ] Preserve partition ordering and partition-level identity explicitly in the
      runtime handoff record.
- [ ] Avoid runtime-local opaque tables that duplicate the descriptor contract
      without exposing their mapping back to the descriptor set.
- [ ] Add focused tests proving supported runtime preparation can be rebuilt
      from the descriptor contract alone.

**Evidence produced**
- Direct descriptor-derived runtime preparation records.
- Focused regression coverage for direct descriptor reuse.

**Risks / rollback**
- Risk: the runtime may appear contract-driven while still depending on hidden
  planner or workload-local state.
- Rollback/mitigation: derive handoff records directly from the emitted
  descriptor set and audit the mapping explicitly.

### Engineering Task 4: Cross-Check Handoff Semantics Across Supported Workload Classes

**Implements story**
- `Story 3: Descriptor-To-Runtime Handoff Is Direct, Auditable, And Shared`

**Change type**
- tests

**Definition of done**
- Story 3 handoff semantics apply equally to continuity, microcase, and
  structured-family workloads.
- Shared runtime handoff rules do not become workload-specific conventions.
- Handoff-shape drift is caught early.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_partitioned_runtime.py` for
      workload-class handoff stability.
- [ ] Compare schema identity, partition-level handoff fields, and direct
      descriptor references across supported workload classes.
- [ ] Keep the checks narrow to handoff structure rather than later numerical
      correctness or unsupported-boundary behavior.
- [ ] Fail quickly when supported workload classes diverge from the shared
      direct handoff contract.

**Evidence produced**
- Fast regression coverage for cross-workload handoff stability.
- Reviewable workload-class comparison checks for the Story 3 handoff surface.

**Risks / rollback**
- Risk: handoff semantics may drift across workload classes while still looking
  locally acceptable.
- Rollback/mitigation: cross-check the same positive handoff rules on multiple
  required workload types.

### Engineering Task 5: Add A Focused Story 3 Handoff Validation Gate

**Implements story**
- `Story 3: Descriptor-To-Runtime Handoff Is Direct, Auditable, And Shared`

**Change type**
- tests | validation automation

**Definition of done**
- Story 3 has a rerunnable validation layer dedicated to descriptor-to-runtime
  handoff auditability.
- The validation layer proves that supported runtime preparation uses the Task 2
  descriptor contract directly.
- The gate remains narrower than later semantic-stress or output-shape work.

**Execution checklist**
- [ ] Add a dedicated handoff validation entry point under
      `benchmarks/density_matrix/partitioned_runtime/`, with
      `runtime_handoff_validation.py` as the primary Story 3 validator.
- [ ] Check at least one continuity case and one methods-oriented case through
      the Story 3 handoff surface.
- [ ] Assert that runtime preparation remains auditable through the validated
      descriptor contract rather than hidden planner state.
- [ ] Keep the Story 3 validation gate separate from later result-shape and
      unsupported-runtime checks.

**Evidence produced**
- One rerunnable Story 3 handoff validation surface.
- Fast regression coverage for direct descriptor-to-runtime auditability.

**Risks / rollback**
- Risk: Story 3 may close with only local inspection and no repeatable proof
  that runtime preparation uses the accepted descriptor contract directly.
- Rollback/mitigation: require one dedicated handoff validation gate before
  closure.

### Engineering Task 6: Emit A Stable Story 3 Runtime-Handoff Bundle

**Implements story**
- `Story 3: Descriptor-To-Runtime Handoff Is Direct, Auditable, And Shared`

**Change type**
- validation automation | docs

**Definition of done**
- Story 3 emits one stable machine-reviewable handoff bundle or rerunnable
  checker.
- The bundle records direct descriptor-to-runtime preparation on supported
  cases.
- The bundle is reusable by later semantic, validation, and benchmark work.

**Execution checklist**
- [ ] Add a dedicated Story 3 artifact location
      (for example `benchmarks/density_matrix/artifacts/partitioned_runtime/runtime_handoff/`).
- [ ] Emit at least one continuity case and one methods-oriented case through
      the Story 3 handoff surface.
- [ ] Include planner schema identity, descriptor schema identity, partition
      summaries, and per-partition execution-plan metadata in the emitted
      output.
- [ ] Record rerun commands and software metadata with the bundle.

**Evidence produced**
- One stable Story 3 runtime-handoff bundle or checker.
- One reusable handoff output surface for later Task 3 work.

**Risks / rollback**
- Risk: if Story 3 emits only ephemeral debug output, later review and paper
  preparation will not have a stable handoff artifact to cite.
- Rollback/mitigation: emit one machine-reviewable bundle and keep it narrow.

### Engineering Task 7: Document And Run The Story 3 Handoff Gate

**Implements story**
- `Story 3: Descriptor-To-Runtime Handoff Is Direct, Auditable, And Shared`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the supported Story 3 direct-handoff rule.
- Fast regression checks and the Story 3 bundle run successfully.
- Story 3 closes with a stable review path for descriptor-to-runtime
  auditability.

**Execution checklist**
- [ ] Document the direct descriptor-consumption rule and the ban on private
      runtime-only reinterpretation.
- [ ] Explain how Story 3 differs from the Task 2 descriptor-audit surface and
      from later runtime semantic or result-shape work.
- [ ] Run focused Story 3 handoff regression coverage and verify the emitted
      bundle.
- [ ] Record stable test and artifact references for Stories 4 through 7 and
      later Phase 3 tasks.

**Evidence produced**
- Passing Story 3 handoff regression checks.
- One stable Story 3 handoff-bundle or checker reference.

**Risks / rollback**
- Risk: Story 3 may appear complete while still leaving implementers unsure how
  the direct-handoff rule is reviewed consistently.
- Rollback/mitigation: document the rule and require a rerunnable handoff
  bundle.

## Exit Criteria

Story 3 is complete only when all of the following are true:

- supported runtime preparation reuses the validated Task 2 descriptor contract
  directly,
- one shared handoff record shape is used across supported workload classes,
- fast regression coverage detects handoff-shape drift across supported cases,
- one stable Story 3 handoff bundle or checker exists for later reuse,
- and partition-local semantic stress, result-shape stabilization,
  cross-workload runtime audit stability, and unsupported-boundary closure
  remain clearly assigned to later stories.

## Implementation Notes

- Prefer one explicit descriptor-to-runtime handoff surface over multiple
  runtime-local bookkeeping layers.
- Keep the handoff surface machine-reviewable and review-oriented, not
  performance-oriented.
- Treat Story 3 as the point where Task 3 proves it consumes Task 2 honestly,
  not as the point where full correctness or benchmark closure is reached.
