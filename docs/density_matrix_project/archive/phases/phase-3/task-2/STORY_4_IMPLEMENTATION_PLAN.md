# Story 4 Implementation Plan

## Story Being Implemented

Story 4: Descriptor Metadata Reconstructs Qubit Support, Remapping, And
Parameter Routing

This is a Layer 4 engineering plan for implementing the fourth behavioral slice
from `TASK_2_STORIES.md`.

## Scope

This story turns Task 2 reconstructibility into one explicit descriptor
contract:

- supported descriptor records expose operation-level qubit support and
  partition-level qubit-span metadata,
- when partition-local indexing differs from the canonical planner surface, the
  descriptor carries invertible remapping metadata,
- parameter-routing metadata is explicit enough that later runtime and audit
  tooling can reconstruct the intended global-to-local parameter semantics,
- and Story 4 closes reconstructibility without claiming runtime agreement
  thresholds, fused execution behavior, or performance improvements.

Out of scope for this story:

- positive workload coverage already owned by Stories 1 and 2,
- exact within-partition gate/noise-order closure already owned by Story 3,
- cross-workload provenance and descriptor-audit stability owned by Story 5,
- unsupported or lossy descriptor-boundary closure owned by Story 6,
- and later runtime, fused execution, or density-aware planning behavior owned
  by later Phase 3 tasks.

## Dependencies And Assumptions

- Stories 1 through 3 already define the positive descriptor slices and the
  exact ordering semantics that Story 4 must make reconstructible.
- Story 2 now also froze the real methods-workload descriptor fixture surface in
  `benchmarks/density_matrix/planner_surface/workloads.py`, which Story 4
  should reuse for nontrivial remap and parameter-routing cases instead of
  creating new ad hoc fixtures.
- The canonical noisy planner surface already records target/control qubits,
  qubit support, parameter counts, and parameter starts at the operation level.
- The current Task 2 descriptor substrate in
  `squander/partitioning/noisy_planner.py` already emits
  `local_to_global_qbits`, `local_qubit_support`, `local_param_start`, and
  partition-level `parameter_routing` fields. Story 4 should strengthen and
  validate those concrete fields rather than renaming them casually.
- Existing state-vector partitioning machinery may provide useful scaffolding
  for local qubit grouping and parameter reordering, especially:
  - `translate_param_order()` in `squander/partitioning/tools.py`,
  - partition grouping outputs from `kahn.py`, `tdag.py`, or related planner
    helpers,
  - and existing partition metadata in `partition.py`.
- Story 4 should reuse those surfaces only as implementation scaffolding. The
  Task 2 contract is the explicit descriptor metadata, not any implicit
  state-vector convention.
- Story 4 should support cases where remapping is unnecessary by allowing a
  descriptor to state the identity mapping cleanly instead of forcing artificial
  remap complexity.
- Story 4 should prefer explicit, reviewable reconstruction metadata over
  inference from hidden planner state.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 2 Reconstruction Contract

**Implements story**
- `Story 4: Descriptor Metadata Reconstructs Qubit Support, Remapping, And Parameter Routing`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 defines the minimum reconstruction obligations for supported
  descriptors.
- The contract distinguishes qubit support, partition span, remapping, and
  parameter routing clearly.
- The contract keeps runtime numerical validation and unsupported taxonomy out
  of the Story 4 bar.

**Execution checklist**
- [ ] Freeze the minimum supported reconstruction metadata for Task 2.
- [ ] Define when a descriptor may omit remapping because the mapping is
      identity and how that identity case is represented explicitly.
- [ ] Define the positive Story 4 review fields for qubit support, partition
      span, remapping, and parameter-routing semantics.
- [ ] Keep unsupported-boundary closure and runtime agreement thresholds outside
      Story 4.

**Evidence produced**
- One stable Story 4 reconstruction contract.
- One reviewable boundary between reconstruction semantics and later tasks.

**Risks / rollback**
- Risk: if reconstruction obligations remain vague, later runtime code may rely
  on hidden planner conventions.
- Rollback/mitigation: freeze the reconstruction contract before expanding the
  runtime path.

### Engineering Task 2: Reuse Existing Partitioning Scaffolding Without Inheriting Hidden State-Vector Assumptions

**Implements story**
- `Story 4: Descriptor Metadata Reconstructs Qubit Support, Remapping, And Parameter Routing`

**Change type**
- code | docs

**Definition of done**
- Story 4 identifies the existing partitioning helpers it can safely reuse.
- The reused helpers are wrapped or documented in a way that makes Task 2
  descriptor semantics explicit.
- Hidden state-vector-only assumptions do not become part of the supported
  Task 2 contract.

**Execution checklist**
- [ ] Review existing partitioning helpers for local qubit grouping and
      parameter-order translation.
- [ ] Reuse those helpers only where the resulting descriptor metadata remains
      explicit and auditable.
- [ ] Add the smallest adapter layer needed to expose reconstruction metadata in
      descriptor records.
- [ ] Document where reused scaffolding is implementation-only and not part of
      the scientific Task 2 claim.

**Evidence produced**
- One reviewable mapping from reused scaffolding to explicit descriptor fields.
- One documented boundary between reusable implementation helpers and supported
  descriptor semantics.

**Risks / rollback**
- Risk: raw reuse of legacy helpers can smuggle in hidden assumptions about
  local indexing or parameter meaning.
- Rollback/mitigation: wrap reused helpers behind explicit descriptor fields and
  document the boundary.

### Engineering Task 3: Add Operation-Level Qubit Support And Partition-Span Metadata

**Implements story**
- `Story 4: Descriptor Metadata Reconstructs Qubit Support, Remapping, And Parameter Routing`

**Change type**
- code | tests

**Definition of done**
- Supported descriptors expose operation-level qubit support directly.
- Each partition also exposes the qubit-span metadata needed for later runtime
  and benchmark consumers.
- The metadata is auditable from descriptor records alone.

**Execution checklist**
- [ ] Record operation-level qubit support on descriptor members using canonical
      planner-surface semantics.
- [ ] Record partition-level qubit-span or equivalent local-support summary
      metadata.
- [ ] Add focused checks on continuity and methods-oriented fixtures that stress
      nontrivial partition span.
- [ ] Keep the descriptor fields explicit rather than reconstructing span from
      hidden planner tables.

**Evidence produced**
- Descriptor records with explicit qubit support and partition-span metadata.
- Focused regression coverage for qubit-support reconstruction.

**Risks / rollback**
- Risk: later runtime work may need to reverse-engineer local support from
  partial descriptor records.
- Rollback/mitigation: encode qubit support and span explicitly now.

### Engineering Task 4: Add Invertible Remapping Metadata Where Partition-Local Indexing Differs

**Implements story**
- `Story 4: Descriptor Metadata Reconstructs Qubit Support, Remapping, And Parameter Routing`

**Change type**
- code | tests

**Definition of done**
- Supported descriptors can record partition-local remapping when needed.
- The remapping metadata is invertible and reviewable.
- Identity-mapping cases remain simple and explicit.

**Execution checklist**
- [ ] Define one explicit remapping representation for supported descriptor
      records.
- [ ] Record identity mapping clearly when no local remap is needed.
- [ ] Add focused fixtures where partition-local indexing genuinely differs from
      canonical global indexing.
- [ ] Add checks proving the remapping metadata can reconstruct the original
      global support exactly.

**Evidence produced**
- Descriptor records with explicit remapping metadata where needed.
- Focused tests proving invertible remapping on supported cases.

**Risks / rollback**
- Risk: incomplete remapping metadata can survive long enough to cause subtle
  runtime bugs later.
- Rollback/mitigation: require invertibility checks before Story 4 closes.

### Engineering Task 5: Add Explicit Parameter-Routing Metadata And Round-Trip Checks

**Implements story**
- `Story 4: Descriptor Metadata Reconstructs Qubit Support, Remapping, And Parameter Routing`

**Change type**
- code | tests

**Definition of done**
- Supported descriptors expose unambiguous parameter-routing metadata.
- The metadata is sufficient to map the global parameter vector into
  partition-local semantics.
- Multi-parameter fixtures can round-trip through the parameter-routing model.

**Execution checklist**
- [ ] Define one explicit parameter-routing representation for Task 2
      descriptors.
- [ ] Reuse existing parameter-order scaffolding where practical, but emit the
      result as descriptor metadata rather than hidden implementation state.
- [ ] Add fixtures with multi-parameter `U3` content that stress local parameter
      routing.
- [ ] Add round-trip checks proving parameter-routing metadata can reconstruct
      the intended global-to-local parameter meaning.

**Evidence produced**
- Descriptor records with explicit parameter-routing metadata.
- Round-trip regression coverage for supported parameter-routing cases.

**Risks / rollback**
- Risk: parameter routing may appear to work on simple cases while remaining
  ambiguous on real partition-local fixtures.
- Rollback/mitigation: require explicit round-trip checks on multi-parameter
  cases before closure.

### Engineering Task 6: Add A Story 4 Reconstruction Audit Matrix

**Implements story**
- `Story 4: Descriptor Metadata Reconstructs Qubit Support, Remapping, And Parameter Routing`

**Change type**
- tests | validation automation

**Definition of done**
- Story 4 includes a small audit matrix that exercises qubit support,
  partition-span, remapping, and parameter-routing reconstruction.
- The matrix is strong enough to expose incomplete remapping or ambiguous
  parameter routing early.
- The audit surface remains narrower than runtime numerical validation.

**Execution checklist**
- [ ] Add focused reconstruction checks in
      `tests/partitioning/test_planner_surface_descriptors.py`.
- [ ] Cover at least one continuity case and at least one methods-oriented case.
- [ ] Include fixtures with nontrivial local indexing and multi-parameter gate
      routing.
- [ ] Fail quickly when reconstruction metadata becomes incomplete or
      inconsistent.

**Evidence produced**
- One Story 4 reconstruction audit matrix.
- Fast regression coverage for reconstructibility.

**Risks / rollback**
- Risk: incomplete reconstruction metadata may not fail until much later runtime
  integration.
- Rollback/mitigation: add a descriptor-level audit matrix before runtime work.

### Engineering Task 7: Emit And Document A Stable Story 4 Reconstruction Bundle

**Implements story**
- `Story 4: Descriptor Metadata Reconstructs Qubit Support, Remapping, And Parameter Routing`

**Change type**
- validation automation | docs

**Definition of done**
- Story 4 emits one stable machine-reviewable reconstruction bundle or
  rerunnable checker.
- The output records qubit support, partition span, remapping, and
  parameter-routing metadata for supported cases.
- Developer-facing notes explain how to review the Story 4 reconstruction gate.

**Execution checklist**
- [ ] Add a dedicated Story 4 artifact location
      (for example `benchmarks/density_matrix/artifacts/planner_surface/descriptor_reconstruction/`).
- [ ] Emit supported cases through the Story 4 reconstruction bundle.
- [ ] Record rerun commands, software metadata, and the reconstruction summary
      with the artifact.
- [ ] Document how the Story 4 audit surface differs from later runtime
      exactness validation.

**Evidence produced**
- One stable Story 4 reconstruction bundle or checker.
- One stable developer-facing reference for Story 4 review.

**Risks / rollback**
- Risk: Story 4 may look complete while still leaving implementers unsure how to
  inspect remapping and parameter-routing semantics consistently.
- Rollback/mitigation: emit a rerunnable reconstruction bundle and document it.

## Exit Criteria

Story 4 is complete only when all of the following are true:

- supported descriptors expose explicit operation-level qubit support and
  partition-level qubit-span metadata,
- descriptors can record invertible remapping when partition-local indexing
  differs and identity mapping when it does not,
- supported descriptors expose unambiguous parameter-routing metadata strong
  enough for round-trip reconstruction checks,
- one stable Story 4 reconstruction bundle or checker exists for review and
  later reuse,
- and cross-workload audit stability and unsupported-boundary closure remain
  clearly assigned to Stories 5 and 6.

## Implementation Notes

- Prefer explicit remapping and parameter-routing fields over inference from
  hidden planner state.
- Reuse existing partitioning helpers only when the resulting descriptor meaning
  remains explicit and auditable.
- Treat Story 4 as the reconstructibility gate for Task 2, not as runtime
  correctness or acceleration work.
