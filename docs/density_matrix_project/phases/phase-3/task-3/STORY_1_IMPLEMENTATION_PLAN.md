# Story 1 Implementation Plan

## Story Being Implemented

Story 1: The Frozen Phase 2 Continuity Workflow Executes End To End Through The
Partitioned Runtime

This is a Layer 4 engineering plan for implementing the first behavioral slice
from `TASK_3_STORIES.md`.

## Scope

This story turns the frozen Phase 2 continuity anchor into the first positive
Task 3 runtime slice:

- the noisy XXZ `HEA` continuity workflow already accepted by Task 1 and Task 2
  now executes through a real partitioned density runtime,
- the supported path begins from the validated Task 2 continuity descriptor set
  rather than from hidden planner state or workload-specific ad hoc adapters,
- the runtime result is reviewable as partitioned execution rather than as a
  silent sequential fallback,
- and the continuity path stays tied to the existing exact density workflow
  rather than to new Phase 4 workflow growth.

Out of scope for this story:

- shared runtime coverage for the mandatory methods workloads owned by Story 2,
- direct descriptor-to-runtime audit stabilization owned by Story 3,
- partition-local semantic stress closure owned by Story 4,
- stable comparison-ready output packaging owned by Story 5,
- cross-workload runtime provenance stability owned by Story 6,
- runtime-stage unsupported-boundary closure owned by Story 7,
- and real fused execution, full correctness-threshold packaging, or
  performance claims owned by later Phase 3 tasks.

## Dependencies And Assumptions

- The frozen source-of-truth contract is `TASK_3_MINI_SPEC.md`,
  `TASK_3_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-005`,
  `P3-ADR-008`, and `P3-ADR-009`.
- Task 2 already established the continuity descriptor path through:
  - `build_phase3_story1_continuity_vqe()` in
    `benchmarks/density_matrix/planner_surface/common.py`,
  - `build_phase3_continuity_partition_descriptor_set()` in
    `squander/partitioning/noisy_planner.py`,
  - `validate_partition_descriptor_set()` and
    `validate_partition_descriptor_set_against_surface()` in the same module,
  - and the emitted Task 2 continuity artifacts under
    `benchmarks/density_matrix/artifacts/phase3_task2/story1_continuity_descriptors/`.
- The sequential exact continuity workflow already has real evidence surfaces in
  `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py`;
  Story 1 should reuse those continuity-case identities and observable
  expectations rather than inventing a second continuity inventory.
- The existing density backend already exposes the exact execution primitive
  needed for a first partitioned runtime slice through `NoisyCircuit`,
  `DensityMatrix`, and `NoisyCircuit.apply_to()`.
- The likely shared Task 3 implementation substrate therefore remains:
  - the Task 2 descriptor layer in `squander/partitioning/noisy_planner.py`,
  - the shared Task 3 runtime layer in `squander/partitioning/noisy_runtime.py`,
  - and benchmark or validation helpers under
    `benchmarks/density_matrix/partitioned_runtime/`.
- Story 1 should establish one explicit positive continuity-runtime slice
  without prematurely freezing the shared multi-workload runtime surface owned
  by Story 2 or the richer output, audit, and unsupported surfaces owned by
  Stories 3 through 7.

## Engineering Tasks

### Engineering Task 1: Freeze The Continuity Runtime Slice And Review Boundary

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Executes End To End Through The Partitioned Runtime`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 names the exact continuity cases it owns.
- Story 1 defines successful runtime execution as explicit partitioned
  execution, not as fused execution or final performance closure.
- The handoff from Story 1 to Stories 2 through 7 is explicit.

**Execution checklist**
- [ ] Freeze the required continuity runtime case inventory around the noisy XXZ
      `HEA` workflow at the mandated 4, 6, 8, and 10 qubit sizes.
- [ ] Define what counts as successful Story 1 runtime execution: supported
      `partitioned_density` execution from validated Task 2 descriptors with no
      silent sequential fallback.
- [ ] Define the minimum positive Story 1 runtime metadata needed for review,
      such as workload ID, requested mode, partition count, runtime-path label,
      and exact-output presence.
- [ ] Keep shared methods-workload runtime coverage, richer audit packaging, and
      unsupported-boundary closure explicitly outside the Story 1 bar.

**Evidence produced**
- One stable Story 1 continuity-runtime contract description.
- One reviewable continuity case inventory for the required anchor sizes.

**Risks / rollback**
- Risk: if Story 1 mixes continuity runtime delivery with later fused or
  benchmark claims, the first positive Task 3 slice will become hard to review.
- Rollback/mitigation: keep Story 1 focused on explicit continuity runtime
  execution only.

### Engineering Task 2: Reuse The Task 2 Continuity Descriptor Path As The Runtime Input Contract

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Executes End To End Through The Partitioned Runtime`

**Change type**
- code | tests

**Definition of done**
- The continuity anchor reaches Task 3 runtime execution from one explicit Task
  2 descriptor path.
- The implementation does not introduce a second competing interpretation of
  the same continuity workload.
- The runtime adapter stays visibly rooted in the already accepted Task 2
  continuity descriptor surface.

**Execution checklist**
- [ ] Reuse `build_phase3_continuity_partition_descriptor_set()` or the smallest
      shared successor as the input contract for Story 1 runtime execution.
- [ ] Add only the minimal runtime adapter layer needed to turn the validated
      continuity descriptor set into executable partition-local work.
- [ ] Keep the continuity runtime path deterministic on the existing supported
      continuity fixtures.
- [ ] Avoid moving the supported continuity runtime contract into
      benchmark-only scripts when the partitioning module already exposes the
      accepted descriptor surface.

**Evidence produced**
- One reviewable continuity-descriptor-to-runtime adapter rooted in the Task 2
  contract.
- Focused regression coverage proving the continuity anchor reaches that path.

**Risks / rollback**
- Risk: a second continuity runtime interpretation path can drift from the
  accepted Task 2 descriptor contract.
- Rollback/mitigation: make the Task 2 continuity descriptor set the single
  supported input contract for Story 1 runtime execution.

### Engineering Task 3: Define The Minimal Story 1 Continuity Runtime Record Shape

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Executes End To End Through The Partitioned Runtime`

**Change type**
- docs | code | tests

**Definition of done**
- Story 1 defines one narrow positive runtime record shape for the continuity
  anchor.
- The record shape is sufficient to review partitioned execution without
  requiring the later shared Task 3 audit surface.
- The record shape remains small enough that later stories can extend it
  cleanly.

**Execution checklist**
- [ ] Freeze one runtime record shape for Story 1 continuity cases.
- [ ] Include workload identity, requested mode, partition count, runtime-path
      classification, and exact-output presence or equivalent result metadata.
- [ ] Include only the minimum observable-comparison fields needed to keep the
      continuity anchor reviewable.
- [ ] Keep shared cross-workload provenance closure and richer result packaging
      for Stories 5 and 6 rather than overloading Story 1.

**Evidence produced**
- One stable Story 1 continuity runtime record shape.
- One clear boundary between Story 1 review fields and later runtime audit
  fields.

**Risks / rollback**
- Risk: an oversized Story 1 record shape can blur the boundary between the
  first positive runtime slice and later shared Task 3 work.
- Rollback/mitigation: freeze only the minimum fields needed for continuity
  runtime review.

### Engineering Task 4: Execute Supported Continuity Cases Through The Partitioned Runtime

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Executes End To End Through The Partitioned Runtime`

**Change type**
- code | tests | validation automation

**Definition of done**
- Supported 4, 6, 8, and 10 qubit continuity cases execute through the shared
  Story 1 partitioned runtime path.
- Runtime execution stays deterministic across reruns for fixed descriptors and
  parameter vectors.
- The emitted records remain auditable without consulting hidden planner state.

**Execution checklist**
- [ ] Add runtime execution for the required supported continuity sizes.
- [ ] Preserve explicit `partitioned_density` mode selection on the supported
      continuity path.
- [ ] Preserve descriptor identity and partition metadata explicitly in the
      emitted runtime records.
- [ ] Keep the Story 1 runtime slice separate from later fused execution or full
      correctness-threshold packaging.

**Evidence produced**
- Continuity runtime outputs for the required anchor sizes.
- Regression coverage proving deterministic runtime execution on the supported
  continuity slice.

**Risks / rollback**
- Risk: continuity cases may appear supported while still relying on hidden
  sequential execution or benchmark-only wiring.
- Rollback/mitigation: record the positive continuity slice entirely through the
  emitted partitioned runtime contract.

### Engineering Task 5: Add A Focused Story 1 Continuity-Runtime Validation Gate

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Executes End To End Through The Partitioned Runtime`

**Change type**
- tests | validation automation

**Definition of done**
- Story 1 has a rerunnable validation layer dedicated to the continuity runtime
  slice.
- The validation layer covers the required anchor sizes and runtime labels.
- The gate remains narrower than later full correctness-threshold and
  unsupported-boundary work.

**Execution checklist**
- [ ] Add a focused Task 3 regression slice in
      `tests/partitioning/test_phase3_task3.py` or a tightly related successor.
- [ ] Add a continuity-runtime validation entry point under
      `benchmarks/density_matrix/partitioned_runtime/`, with
      `continuity_runtime_validation.py` as the primary Story 1 validator.
- [ ] Check supported 4, 6, 8, and 10 qubit continuity cases.
- [ ] Assert explicit partitioned mode selection, supported runtime-path
      labeling, and no silent sequential fallback on the supported slice.

**Evidence produced**
- One rerunnable Story 1 continuity-runtime validation surface.
- Fast regression coverage for the first positive Task 3 slice.

**Risks / rollback**
- Risk: Story 1 may close with only informal inspection and no repeatable
  continuity-runtime gate.
- Rollback/mitigation: require one dedicated validation layer before closure.

### Engineering Task 6: Emit A Stable Story 1 Continuity-Runtime Artifact Bundle

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Executes End To End Through The Partitioned Runtime`

**Change type**
- validation automation | docs

**Definition of done**
- Story 1 emits one stable machine-reviewable artifact bundle or rerunnable
  checker for the continuity-runtime slice.
- The output records enough metadata to prove supported partitioned execution
  on the continuity anchor.
- The artifact shape is stable enough for later Task 3 stories to extend.

**Execution checklist**
- [ ] Add a dedicated Story 1 artifact location
      (for example `benchmarks/density_matrix/artifacts/phase3_task3/story1_continuity_runtime/`).
- [ ] Record workload ID, requested mode, partition count, runtime-path label,
      exact-output presence, and continuity observable metadata for supported
      cases.
- [ ] Record the rerun command and software metadata with the artifact.
- [ ] Keep the artifact narrow to continuity runtime execution rather than
      rolled-up performance interpretation.

**Evidence produced**
- One stable Story 1 continuity-runtime artifact bundle.
- One reusable output shape for later shared runtime work.

**Risks / rollback**
- Risk: prose-only closure makes the first positive Task 3 slice hard to cite
  and easy to misstate later.
- Rollback/mitigation: emit one thin machine-reviewable bundle early.

### Engineering Task 7: Document The Story 1 Handoff To Later Runtime Stories

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Executes End To End Through The Partitioned Runtime`

**Change type**
- docs

**Definition of done**
- Story 1 notes explain exactly what the continuity runtime slice closes.
- The continuity anchor is documented as the first positive Task 3 runtime path,
  not as the whole Task 3 contract.
- Developer-facing notes point to the Story 1 validation and artifact path.

**Execution checklist**
- [ ] Document the supported continuity runtime slice and its evidence surface.
- [ ] Explain that shared methods-workload runtime coverage belongs to Story 2.
- [ ] Explain that handoff auditability, semantic stress, result-shape
      stabilization, audit stability, and unsupported-boundary closure belong to
      Stories 3 through 7.
- [ ] Record stable references to the Story 1 tests and artifact bundle.

**Evidence produced**
- Updated developer-facing notes for the Story 1 continuity-runtime gate.
- One stable handoff reference for later Task 3 work.

**Risks / rollback**
- Risk: later Task 3 work may over-assume Story 1 closed the full runtime
  contract.
- Rollback/mitigation: document the handoff boundaries explicitly.

## Exit Criteria

Story 1 is complete only when all of the following are true:

- the frozen Phase 2 noisy XXZ `HEA` continuity workflow at 4, 6, 8, and 10
  qubits executes through one auditable partitioned runtime path,
- the continuity runtime path is rooted in one explicit Task 2 descriptor route
  rather than multiple competing interpretations,
- explicit partitioned-mode labeling and no silent fallback are reviewable on
  the emitted runtime records,
- one stable Story 1 validation command or artifact bundle proves runtime
  execution for the continuity anchor,
- and shared methods-workload coverage, richer runtime audit stability,
  unsupported-boundary closure, and fused execution remain clearly assigned to
  later stories or tasks.

## Implementation Notes

- Prefer reusing the existing Task 2 continuity descriptor path over rebuilding
  continuity runtime execution from raw VQE internals.
- Keep Story 1 focused on the first positive runtime slice, not on full
  correctness-threshold closure.
- Treat the continuity anchor as a required Task 3 workload, not as evidence
  that every supported workload already shares the final runtime contract.
