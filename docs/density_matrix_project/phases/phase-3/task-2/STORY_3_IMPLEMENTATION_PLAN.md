# Story 3 Implementation Plan

## Story Being Implemented

Story 3: Exact Gate And Noise Order Remains Explicit Inside Each Partition
Descriptor

This is a Layer 4 engineering plan for implementing the third behavioral slice
from `TASK_2_STORIES.md`.

## Scope

This story turns Task 2 semantic preservation into explicit descriptor-ordering
behavior:

- each supported partition descriptor records exact within-partition gate/noise
  order rather than only partition membership,
- noise placement remains first-class descriptor content instead of
  partition-boundary-only metadata,
- boundary-sensitive cases around noise operations stay auditable enough for
  later runtime exactness work to consume directly,
- and Story 3 closes the positive ordering semantics without claiming
  reconstructible remapping, cross-workload audit stability, or unsupported
  taxonomy closure.

Out of scope for this story:

- positive continuity and mandatory methods-workload coverage already owned by
  Stories 1 and 2,
- reconstructible remapping and parameter-routing closure owned by Story 4,
- cross-workload provenance and audit-bundle stability owned by Story 5,
- unsupported or lossy descriptor-boundary closure owned by Story 6,
- and runtime exactness thresholds, fused execution, or performance behavior
  owned by later Phase 3 tasks.

## Dependencies And Assumptions

- Stories 1 and 2 already define the supported positive workload slices whose
  descriptor ordering Story 3 must preserve.
- Story 2 now widened the shared positive coverage through
  `iter_story2_microcase_descriptor_sets()` and
  `iter_story2_structured_descriptor_sets()` in
  `benchmarks/density_matrix/planner_surface/workloads.py`, plus the emitted
  bundle under
  `benchmarks/density_matrix/artifacts/planner_surface/mandatory_workload/`.
- The canonical noisy planner surface from Task 1 already provides ordered gate
  and noise operations plus canonical operation indices that Story 3 should
  reuse rather than reconstruct.
- The shared Task 2 descriptor substrate now lives in
  `squander/partitioning/noisy_planner.py`, where the current positive slice
  already emits `NoisyPartitionDescriptorSet`, `NoisyPartitionDescriptor`, and
  `NoisyPartitionDescriptorMember` records.
- The current descriptor payload already freezes `canonical_operation_indices`
  and ordered `members`; Story 3 should refine the semantic guarantees around
  those existing fields rather than inventing a second ordered-membership
  schema.
- Story 1 also established the first Task 2 regression and validation entry
  points in `tests/partitioning/test_planner_surface_descriptors.py` and
  `benchmarks/density_matrix/planner_surface/continuity_descriptor_validation.py`.
- Story 3 should keep ordering and noise-placement semantics explicit in the
  descriptor records rather than relying on hidden runtime conventions.
- Existing artifact and audit surfaces from Task 1 can provide starting schema
  patterns, but Story 3 now moves the semantic focus from planner-entry records
  to partition-descriptor records.
- Story 3 should prefer auditable positive ordering semantics and leave the
  structured unsupported-taxonomy rules to Story 6.

## Engineering Tasks

### Engineering Task 1: Freeze The Exact Within-Partition Ordering Contract

**Implements story**
- `Story 3: Exact Gate And Noise Order Remains Explicit Inside Each Partition Descriptor`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 defines one explicit ordering contract for supported partition
  descriptors.
- The contract distinguishes exact within-partition ordering from mere
  partition membership.
- The contract stays narrow enough that later stories can add remapping,
  parameter-routing, and audit-stability details cleanly.

**Execution checklist**
- [ ] Freeze the supported meaning of exact within-partition gate/noise order for
      Task 2 descriptor records.
- [ ] Define what counts as explicit noise placement inside a descriptor rather
      than a boundary-only annotation.
- [ ] Define the minimal positive Story 3 audit fields needed to review order
      and noise placement.
- [ ] Keep reconstructibility and unsupported-taxonomy closure outside the Story
      3 bar.

**Evidence produced**
- One stable Story 3 ordering contract for supported descriptors.
- One clear boundary between ordering semantics and later descriptor concerns.

**Risks / rollback**
- Risk: if Story 3 defines ordering too loosely, later semantic-preservation
  claims will rest on ambiguous descriptor meaning.
- Rollback/mitigation: freeze exact within-partition ordering before extending
  other descriptor metadata.

### Engineering Task 2: Add Ordered Operation References To The Descriptor Schema

**Implements story**
- `Story 3: Exact Gate And Noise Order Remains Explicit Inside Each Partition Descriptor`

**Change type**
- code | tests

**Definition of done**
- Supported descriptors record ordered references back to the canonical noisy
  operation sequence.
- The descriptor schema can distinguish the relative position of gates and noise
  operations inside each partition.
- Descriptor interpretation does not rely on hidden planner state.

**Execution checklist**
- [ ] Reuse canonical operation indices or an equivalent ordered canonical
      reference scheme in the descriptor records.
- [ ] Preserve within-partition order explicitly rather than inferring it from
      unordered membership or side tables.
- [ ] Keep the descriptor record shape reviewable on continuity and
      methods-oriented fixtures.
- [ ] Add focused tests proving ordered canonical references survive descriptor
      emission.

**Evidence produced**
- Ordered descriptor records with explicit canonical-operation references.
- Regression coverage for explicit ordered membership.

**Risks / rollback**
- Risk: descriptors may appear complete while still requiring hidden internal
  state to reconstruct order.
- Rollback/mitigation: store ordering explicitly in the emitted record.

### Engineering Task 3: Make Noise Placement First-Class Inside Descriptor Records

**Implements story**
- `Story 3: Exact Gate And Noise Order Remains Explicit Inside Each Partition Descriptor`

**Change type**
- code | tests

**Definition of done**
- Noise operations remain first-class members inside supported partition
  descriptors.
- Descriptor records do not reduce noise placement to boundary markers or other
  side metadata.
- Mixed gate/noise partitions remain auditable on supported cases.

**Execution checklist**
- [ ] Represent noise operations inside the same ordered descriptor structure
      used for gate operations.
- [ ] Preserve the relative position of noise with respect to neighboring gates
      inside each partition.
- [ ] Keep descriptor emission honest on cases with sparse, periodic, and dense
      local-noise placement.
- [ ] Add focused checks showing that noise placement is visible from the
      descriptor record itself.

**Evidence produced**
- Descriptor records with explicit gate and noise membership.
- Focused tests proving noise placement remains first-class.

**Risks / rollback**
- Risk: a descriptor may look partitioned while silently degrading noise
  semantics to partition-boundary markers.
- Rollback/mitigation: encode noise directly in the descriptor membership model.

### Engineering Task 4: Add Boundary-Sensitive Story 3 Audit Cases

**Implements story**
- `Story 3: Exact Gate And Noise Order Remains Explicit Inside Each Partition Descriptor`

**Change type**
- tests | validation automation

**Definition of done**
- Story 3 includes cases that stress partitions beginning or ending near noise
  operations.
- The audit cases are strong enough to expose undocumented reordering across
  noise boundaries.
- The audit surface remains narrower than full runtime exactness validation.

**Execution checklist**
- [ ] Select continuity and methods-oriented fixtures where partitions touch or
      straddle noise-adjacent regions.
- [ ] Add focused audit checks for exact within-partition ordering on those
      fixtures.
- [ ] Keep the checks at the descriptor level rather than at runtime numerical
      comparison.
- [ ] Record a small set of boundary-focused fixtures that later stories can
      reuse.

**Evidence produced**
- Boundary-sensitive descriptor audit cases.
- Reviewable evidence that unsupported reordering is not silently accepted.

**Risks / rollback**
- Risk: order bugs near noise boundaries may remain hidden if Story 3 tests only
  simple partitions.
- Rollback/mitigation: include explicit boundary-stressing fixtures in the Story
  3 gate.

### Engineering Task 5: Cross-Check Ordering Semantics Across Continuity And Methods Workloads

**Implements story**
- `Story 3: Exact Gate And Noise Order Remains Explicit Inside Each Partition Descriptor`

**Change type**
- tests

**Definition of done**
- Story 3 ordering semantics apply equally to the continuity anchor and to the
  mandatory methods workloads.
- Shared descriptor ordering rules do not become workload-specific conventions.
- Schema drift in ordering semantics is caught early.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_planner_surface_descriptors.py` for
      continuity and methods-oriented fixtures.
- [ ] Compare ordered descriptor membership across supported workload classes.
- [ ] Keep the checks narrow to ordering and explicit noise placement rather
      than to later remapping or runtime correctness.
- [ ] Fail quickly when descriptor ordering semantics differ across workload
      classes.

**Evidence produced**
- Fast regression coverage for shared ordering semantics.
- Reviewable cross-workload ordering checks.

**Risks / rollback**
- Risk: ordering semantics may drift subtly across workload classes while still
  looking locally reasonable.
- Rollback/mitigation: cross-check the same positive ordering rules on multiple
  supported workload types.

### Engineering Task 6: Emit A Stable Story 3 Order-And-Noise Audit Bundle

**Implements story**
- `Story 3: Exact Gate And Noise Order Remains Explicit Inside Each Partition Descriptor`

**Change type**
- validation automation | docs

**Definition of done**
- Story 3 emits one stable audit bundle or rerunnable checker for descriptor
  ordering and noise placement.
- The output records ordered descriptor membership and boundary-sensitive cases
  in a machine-reviewable way.
- The bundle is reusable by later runtime and validation tasks.

**Execution checklist**
- [ ] Add a dedicated Story 3 artifact location
      (for example `benchmarks/density_matrix/artifacts/planner_surface/descriptor_ordering/`).
- [ ] Emit at least one continuity case and one methods-oriented case through
      the Story 3 audit bundle.
- [ ] Record the ordered descriptor membership and explicit noise-placement
      information in the emitted artifact.
- [ ] Keep the bundle focused on order and noise semantics rather than later
      runtime outcomes.

**Evidence produced**
- One stable Story 3 order-and-noise audit bundle.
- One reusable output shape for later Task 2 and Task 3 work.

**Risks / rollback**
- Risk: if Story 3 emits only ephemeral debug output, later review and paper
  preparation will not have a stable ordering artifact to cite.
- Rollback/mitigation: emit one machine-reviewable bundle and keep it narrow.

### Engineering Task 7: Document And Run The Story 3 Ordering Gate

**Implements story**
- `Story 3: Exact Gate And Noise Order Remains Explicit Inside Each Partition Descriptor`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the supported Story 3 ordering semantics.
- Fast regression checks and the Story 3 bundle run successfully.
- Story 3 closes with a stable review path for descriptor order and noise
  placement.

**Execution checklist**
- [ ] Document the exact within-partition ordering rule and first-class
      noise-placement requirement.
- [ ] Explain how Story 3 differs from the Task 1 planner-entry audit surface
      and from later remapping/runtime work.
- [ ] Run focused Story 3 regression coverage and verify the emitted audit
      bundle.
- [ ] Record stable test and artifact references for Stories 4 through 6 and
      later Phase 3 tasks.

**Evidence produced**
- Passing Story 3 ordering regression checks.
- One stable Story 3 audit-bundle or checker reference.

**Risks / rollback**
- Risk: Story 3 may appear complete while still leaving implementers unsure how
  descriptor order semantics are reviewed consistently.
- Rollback/mitigation: document the rule and require a rerunnable audit bundle.

## Exit Criteria

Story 3 is complete only when all of the following are true:

- supported partition descriptors expose exact within-partition gate/noise order
  explicitly,
- noise placement remains first-class descriptor content rather than
  boundary-only metadata,
- continuity and mandatory methods workloads share the same positive ordering
  semantics,
- one stable Story 3 audit bundle or checker exists for order and noise
  placement,
- and reconstructibility, cross-workload audit stability, and unsupported
  boundary closure remain clearly assigned to later stories.

## Implementation Notes

- Prefer explicit ordered canonical-operation references over inferred order from
  unordered membership tables.
- Keep Story 3 focused on positive order/noise semantics, not on the full
  unsupported-boundary taxonomy.
- Treat exact order and first-class noise placement as descriptor semantics,
  not as runtime conventions to be explained later.
