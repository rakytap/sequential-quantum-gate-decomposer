# Story 3 Implementation Plan

## Story Being Implemented

Story 3: Gate And Noise Operations Are First-Class And Auditable At Planner
Entry

This is a Layer 4 engineering plan for implementing the third behavioral slice
from `TASK_1_STORIES.md`.

## Scope

This story turns the canonical planner surface into an auditable mixed-state
representation rather than a black-box handoff:

- gate operations and noise operations are exposed as first-class ordered
  planner objects,
- planner-entry records capture the minimum metadata needed to review supported
  mixed-state semantics,
- entry-route provenance is explicit enough to distinguish continuity lowering,
  structured-family generation, and later optional exact lowerings,
- and the audit surface remains stable enough that later partition-descriptor
  and runtime stories can build on it directly.

Out of scope for this story:

- establishing workload coverage for continuity and methods families already
  owned by Stories 1 and 2,
- optional exact lowering from legacy source surfaces owned by Story 4,
- unsupported planner-boundary closure owned by Story 5,
- partition-descriptor metadata beyond planner entry owned by Task 2,
- and partition execution or fused runtime work owned by later Phase 3 tasks.

## Dependencies And Assumptions

- Story 3 depends on Stories 1 and 2 for supported inputs that already reach the
  canonical planner surface.
- The shared Task 1 planner substrate is now the Python-side module
  `squander/partitioning/noisy_planner.py`, which already emits stable ordered
  gate/noise records for the continuity path.
- Stories 1 and 2 have now also established real emitted bundle schemas under
  `benchmarks/density_matrix/artifacts/planner_surface/`; Story 3 should extend
  that frozen planner-surface schema rather than redesign it.
- `NoisyCircuit::OperationInfo` in
  `squander/src-cpp/density_matrix/include/noisy_circuit.h` is currently a
  useful but minimal inspection surface: it records only operation name,
  unitarity, and parameter-start metadata.
- The existing VQE bridge inspection path already exposes richer information
  through `DensityBridgeOperationInfo` and the Python method
  `describe_density_bridge()`.
- Story 3 should prefer extending the planner-side audit vocabulary first and
  only extend raw `NoisyCircuit` inspection if a concrete supported Task 1 case
  truly needs it.
- The audit surface should be narrow and review-oriented. It is not intended to
  become a second execution API or a full partition-descriptor format.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Planner-Operation Schema

**Implements story**
- `Story 3: Gate And Noise Operations Are First-Class And Auditable At Planner Entry`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 extends one already-frozen planner-operation schema for supported
  cases rather than inventing a replacement.
- The schema captures the minimum first-class semantics needed at planner entry.
- The schema separates planner-entry audit metadata from later partition or
  runtime metadata.

**Execution checklist**
- [ ] Review the passing Story 1 and Story 2 planner-surface bundles and treat
      their shared payload shape as the starting schema.
- [ ] Include the minimum fields needed for ordered auditability:
      operation kind, name, unitary flag, qubit support or target/control
      metadata, parameter metadata, and fixed-value metadata where relevant.
- [ ] Freeze one provenance vocabulary for how a workload reached planner entry.
- [ ] Keep partition-descriptor metadata, remapping metadata, and runtime-only
      counters out of the Story 3 schema unless they are strictly needed at
      planner entry.

**Evidence produced**
- One stable Story 3 planner-operation schema.
- One stable provenance vocabulary for supported planner-entry routes.

**Risks / rollback**
- Risk: if audit fields are defined ad hoc, later partition and benchmark
  artifacts will become inconsistent.
- Rollback/mitigation: freeze the planner-operation schema before broadening the
  inspection surfaces that emit it.

### Engineering Task 2: Expose Rich Planner-Entry Inspection From The Density Backend Or A Planner Adapter

**Implements story**
- `Story 3: Gate And Noise Operations Are First-Class And Auditable At Planner Entry`

**Change type**
- code | tests

**Definition of done**
- Supported planner-entry cases can emit rich ordered operation records.
- The emitted records make gate and noise operations equally visible as
  first-class planner objects.
- The implementation avoids maintaining two incompatible inspection contracts
  for the same canonical planner surface.

**Execution checklist**
- [ ] Reuse and extend the planner-side record shape in
      `squander/partitioning/noisy_planner.py` as the primary Story 3 audit
      surface.
- [ ] Reuse the richer VQE bridge inspection fields where practical rather than
      re-inventing them.
- [ ] Treat `NoisyCircuit::get_operation_info()` as a secondary cross-check
      surface unless Story 3 evidence shows that raw density inspection must be
      enriched.
- [ ] Ensure supported noise operations can carry the fixed-value metadata
      needed for auditability.
- [ ] Keep the inspection surface stable in ordering and field naming across all
      supported workload classes.

**Evidence produced**
- One rich planner-entry inspection surface for supported cases.
- Regression coverage proving gate and noise operations are both visible.

**Risks / rollback**
- Risk: extending the wrong inspection surface can either break existing users
  or leave Phase 3 with two competing metadata formats.
- Rollback/mitigation: prefer one canonical planner-inspection format and keep
  existing wrappers as thin adapters where necessary.

### Engineering Task 3: Add Entry-Route Provenance And Source Labels To Planner Records

**Implements story**
- `Story 3: Gate And Noise Operations Are First-Class And Auditable At Planner Entry`

**Change type**
- code | validation automation

**Definition of done**
- Planner-entry records make clear how a case reached the canonical surface.
- Provenance is machine-reviewable and stable across supported workload classes.
- Story 3 provenance works for continuity lowering, structured-family
  generation, and future optional exact lowerings.

**Execution checklist**
- [ ] Add source-type labels or equivalent provenance metadata to planner-entry
      records.
- [ ] Distinguish at minimum:
      continuity lowering, structured-family generation, direct microcase
      generation, and optional exact legacy-source lowering.
- [ ] Record provenance at the case level and keep operation-level provenance as
      lightweight as possible.
- [ ] Reuse existing bridge source labels where they already match the frozen
      contract.

**Evidence produced**
- Reviewable provenance metadata for supported planner-entry cases.
- Stable source labels that later bundles can reuse.

**Risks / rollback**
- Risk: later artifacts may prove representability but still fail to show what
  was actually normalized into the planner.
- Rollback/mitigation: make source-route provenance part of the Story 3 schema,
  not an optional side note.

### Engineering Task 4: Cross-Check Planner-Entry Inspection Against Existing Bridge And Density Surfaces

**Implements story**
- `Story 3: Gate And Noise Operations Are First-Class And Auditable At Planner Entry`

**Change type**
- tests

**Definition of done**
- Supported planner-entry records agree with the existing Phase 2 bridge
  metadata where their overlap should match.
- Deterministic `NoisyCircuit` fixtures can be inspected without VQE-specific
  context and still produce the expected ordered semantics.
- Story 3 catches field drift between density inspection and planner-entry
  inspection early.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_planner_surface_entry.py` or a
      tightly related successor for planner-surface audit semantics.
- [ ] Add focused checks in `tests/VQE/test_VQE.py` or a tightly related
      successor for overlap with Phase 2 bridge metadata.
- [ ] Compare ordered operation count, operation names, unitary flags, and
      planner-entry provenance on deterministic fixtures.
- [ ] Use raw `NoisyCircuit` inspection only as an auxiliary consistency check
      where it materially strengthens Story 3 coverage.
- [ ] Keep the checks narrow to auditability rather than full runtime
      correctness.

**Evidence produced**
- Fast regression coverage for planner-entry inspection stability.
- Reviewable overlap checks between raw density inspection and VQE bridge
  inspection.

**Risks / rollback**
- Risk: the planner-entry schema may silently diverge from the already deployed
  bridge metadata even while each surface looks reasonable in isolation.
- Rollback/mitigation: cross-check the overlapping fields explicitly on stable
  fixtures.

### Engineering Task 5: Emit A Story 3 Planner-Entry Audit Bundle

**Implements story**
- `Story 3: Gate And Noise Operations Are First-Class And Auditable At Planner Entry`

**Change type**
- validation automation | docs

**Definition of done**
- Story 3 emits one stable audit bundle or rerunnable checker for planner-entry
  inspection.
- The output records planner-operation fields and source-route provenance for
  supported cases.
- The bundle is stable enough for later partition-descriptor and benchmark work
  to reference directly.

**Execution checklist**
- [ ] Add a dedicated Story 3 artifact location
      (for example `benchmarks/density_matrix/artifacts/planner_surface/`).
- [ ] Emit at least one continuity case and one methods-oriented case through
      the planner-entry inspection bundle.
- [ ] Record schema version or equivalent output-shape identity with the bundle.
- [ ] Keep the bundle focused on entry auditability rather than performance or
      execution outcomes.

**Evidence produced**
- One stable Story 3 planner-entry audit bundle.
- One reusable output schema for later Task 2 and Task 3 work.

**Risks / rollback**
- Risk: if Story 3 emits only ephemeral console output, later review and paper
  preparation will not have a stable inspection artifact to cite.
- Rollback/mitigation: emit one small machine-reviewable bundle and version its
  schema explicitly.

### Engineering Task 6: Document And Run The Story 3 Auditability Gate

**Implements story**
- `Story 3: Gate And Noise Operations Are First-Class And Auditable At Planner Entry`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain how to inspect supported planner-entry cases.
- Fast regression coverage and the Story 3 audit bundle run successfully.
- Story 3 closes with a stable review path for planner-entry semantics.

**Execution checklist**
- [ ] Document the canonical planner-operation schema and provenance vocabulary.
- [ ] Explain how the Story 3 audit surface differs from `NoisyCircuit` runtime
      execution and from later partition descriptors.
- [ ] Run focused Story 3 regression checks and verify the emitted audit bundle.
- [ ] Record stable test and artifact references for Story 4 and later Phase 3
      work.

**Evidence produced**
- Passing Story 3 regression checks.
- One stable Story 3 audit bundle or checker reference.

**Risks / rollback**
- Risk: Story 3 may appear complete while still leaving future implementers
  unsure how to inspect planner-entry semantics consistently.
- Rollback/mitigation: document the audit surface and require a rerunnable
  bundle as part of closure.

## Exit Criteria

Story 3 is complete only when all of the following are true:

- supported planner-entry cases expose gate and noise operations as first-class
  ordered planner objects,
- one stable planner-operation schema and provenance vocabulary are frozen for
  supported cases,
- raw density inspection and VQE bridge inspection overlap cleanly where they
  describe the same semantics,
- one stable Story 3 audit bundle or checker exists for review and later reuse,
- and legacy-source lowering, unsupported-boundary closure, partition
  descriptors, and runtime execution remain clearly assigned to later stories or
  later tasks.

## Implementation Notes

- Prefer a planner-specific inspection record over mutating the minimal
  `NoisyCircuit::OperationInfo` only if backward-compatibility or API clarity
  would otherwise suffer.
- Keep the audit schema small but semantically meaningful. More fields are not
  automatically better if they confuse planner entry with later runtime state.
- Treat provenance as part of scientific auditability, not as optional metadata.
