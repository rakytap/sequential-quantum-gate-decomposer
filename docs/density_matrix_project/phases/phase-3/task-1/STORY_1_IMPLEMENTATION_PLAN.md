# Story 1 Implementation Plan

## Story Being Implemented

Story 1: The Frozen Phase 2 Continuity Workflow Reaches The Canonical Planner
Surface

This is a Layer 4 engineering plan for implementing the first behavioral slice
from `TASK_1_STORIES.md`.

## Scope

This story turns the frozen Phase 2 continuity anchor into explicit Phase 3
planner-entry behavior:

- the noisy XXZ `HEA` workflow already validated in Phase 2 is accepted as a
  required Phase 3 source workload,
- that workload reaches one canonical ordered noisy mixed-state planner surface
  before any partition heuristic is invoked,
- the canonical planner entry remains auditable enough that later stories can
  add methods workloads, richer provenance, optional source lowering, and
  unsupported-case coverage without replacing the continuity path,
- and the continuity-anchor route stays tied to the existing exact density
  workflow rather than to new Phase 4 workflow growth.

Out of scope for this story:

- structured noisy `U3` / `CNOT` workload-family coverage owned by Story 2,
- richer planner-entry audit schema owned by Story 3,
- optional exact lowering from `qgd_Circuit` or `Gates_block` owned by Story 4,
- unsupported planner-boundary closure owned by Story 5,
- and partition execution, fused runtime behavior, or cost-model work owned by
  later Phase 3 tasks.

## Dependencies And Assumptions

- The frozen source-of-truth contract is `TASK_1_MINI_SPEC.md`,
  `TASK_1_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-003`,
  `P3-ADR-007`, and `P3-ADR-009`.
- The existing Phase 2 density workflow already exposes a reviewable bridge
  surface through:
  - `Variational_Quantum_Eigensolver_Base::inspect_density_bridge()` in
    `squander/src-cpp/variational_quantum_eigensolver/Variational_Quantum_Eigensolver_Base.cpp`,
  - the Python wrapper `describe_density_bridge()` in
    `squander/VQA/qgd_Variational_Quantum_Eigensolver_Base.py`,
  - and continuity evidence under
    `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py`.
- `NoisyCircuit` remains the sequential exact mixed-state oracle and already
  exposes ordered execution semantics through
  `squander/src-cpp/density_matrix/noisy_circuit.cpp`.
- The current state-vector partitioning entry points in
  `squander/partitioning/partition.py`, `kahn.py`, `tdag.py`, and `tools.py`
  are relevant reuse targets, but they currently assume `qgd_Circuit`-style
  gate surfaces rather than a noisy mixed-state planner contract.
- Story 1 should establish the canonical continuity entry path without deciding
  yet whether the eventual Phase 3 planner uses a pure ordered list, a DAG view,
  or both internally. The contract is the ordered noisy surface.

## Engineering Tasks

### Engineering Task 1: Freeze The Story 1 Continuity-Entry Contract

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Reaches The Canonical Planner Surface`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 names the exact continuity workload slice it owns.
- The continuity slice is expressed as planner-entry behavior, not as runtime or
  benchmark-performance behavior.
- The handoff boundary from Story 1 to Stories 2 through 5 is explicit.

**Execution checklist**
- [ ] Freeze the required continuity case inventory around the noisy XXZ `HEA`
      workflow at the mandated 4, 6, 8, and 10 qubit sizes.
- [ ] Define what counts as successful planner entry for those cases:
      canonical ordered noisy mixed-state operations before partition planning.
- [ ] Define the minimal continuity-entry metadata needed for Story 1 review:
      source label, qubit count, operation count, and ordered-operation audit
      trace.
- [ ] Keep partition execution, partition descriptors, and fused runtime
      semantics explicitly outside the Story 1 closure bar.

**Evidence produced**
- One stable Story 1 continuity-entry contract description.
- One reviewable continuity case inventory for the required anchor sizes.

**Risks / rollback**
- Risk: if Story 1 mixes planner-entry closure with runtime work, later Phase 3
  implementation will blur representational success with acceleration claims.
- Rollback/mitigation: keep Story 1 focused only on canonical planner entry for
  the continuity anchor.

### Engineering Task 2: Reuse The Existing Phase 2 Bridge Surface As The Canonicalization Seed

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Reaches The Canonical Planner Surface`

**Change type**
- code | tests

**Definition of done**
- The continuity anchor reaches the Phase 3 planner contract from one explicit
  VQE-side exact lowering path.
- The code does not introduce a second competing interpretation of the same
  generated-`HEA` bridge.
- The canonical continuity adapter is visibly rooted in the existing density
  bridge surfaces rather than in ad hoc benchmark-only logic.

**Execution checklist**
- [ ] Review and reuse `inspect_density_bridge()` and
      `describe_density_bridge()` as the primary continuity-entry substrate.
- [ ] Add the smallest adapter layer needed to materialize a Phase 3
      planner-entry object from that bridge metadata.
- [ ] Keep the canonicalization step auditable and deterministic on the existing
      supported continuity workflow.
- [ ] Avoid duplicating bridge traversal logic in benchmark scripts when the VQE
      runtime already exposes the needed ordered operation metadata.

**Evidence produced**
- One reviewable continuity-to-planner adapter rooted in the existing bridge
  path.
- Focused regression coverage proving the continuity anchor reaches that adapter.

**Risks / rollback**
- Risk: a second interpretation path can drift from the VQE bridge and create
  contradictory continuity semantics.
- Rollback/mitigation: make the VQE bridge inspection surface the single source
  for continuity canonicalization unless a stricter planner-specific adapter is
  required.

### Engineering Task 3: Preserve Ordered Gate, Noise, And Parameter Metadata Across Continuity Canonicalization

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Reaches The Canonical Planner Surface`

**Change type**
- code | tests

**Definition of done**
- The continuity adapter preserves the ordered gate/noise stream needed for the
  canonical planner surface.
- Parameter-count and parameter-start information remain stable enough for later
  partition and validation tasks to reuse directly.
- The resulting planner-entry representation is reviewable without inspecting
  hidden runtime state.

**Execution checklist**
- [ ] Preserve operation order from the existing density bridge metadata.
- [ ] Preserve parameter metadata needed to map continuity workloads into later
      planner and runtime stages.
- [ ] Keep continuity canonicalization tied to explicit gate/noise operations
      instead of reconstructing order indirectly.
- [ ] Add focused assertions that continuity canonicalization is stable across 4,
      6, 8, and 10 qubit supported fixtures.

**Evidence produced**
- Ordered continuity planner-entry traces.
- Regression coverage proving parameter and operation-order stability.

**Risks / rollback**
- Risk: continuity cases may look representable while hidden order or parameter
  drift breaks later semantic-preservation work.
- Rollback/mitigation: validate canonicalization metadata directly before moving
  on to partition descriptors or runtime work.

### Engineering Task 4: Add A Focused Story 1 Continuity Validation Surface

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Reaches The Canonical Planner Surface`

**Change type**
- tests | validation automation

**Definition of done**
- Story 1 has a rerunnable validation layer dedicated to planner entry for the
  continuity anchor.
- The validation surface covers the required anchor sizes and emits stable
  reviewable metadata.
- Story 1 validation reuses the existing continuity workflow tooling where
  practical.

**Execution checklist**
- [ ] Extend `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py`
      or add a tightly related Phase 3 successor focused on planner-entry
      evidence.
- [ ] Record canonical-entry metadata for supported 4, 6, 8, and 10 qubit
      continuity cases.
- [ ] Keep the Story 1 validation layer separate from later structured-family
      or unsupported-case bundles.
- [ ] Add a fast regression slice in `tests/VQE/test_VQE.py` or a tightly
      related successor that checks continuity-entry availability without
      requiring the full benchmark bundle.

**Evidence produced**
- One rerunnable Story 1 continuity-entry validation command or script.
- Focused fast regression coverage for the continuity canonicalization slice.

**Risks / rollback**
- Risk: Story 1 may end with only informal spot checks and no reusable evidence
  surface for later review.
- Rollback/mitigation: require one explicit validation entry point and one fast
  regression layer before closing Story 1.

### Engineering Task 5: Emit A Stable Continuity Planner-Entry Artifact Bundle

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Reaches The Canonical Planner Surface`

**Change type**
- validation automation | docs

**Definition of done**
- Story 1 emits at least one stable artifact bundle or rerunnable command
  reference for the continuity anchor.
- The artifact records enough metadata to prove canonical planner entry
  happened on supported continuity cases.
- The output shape is stable enough for Stories 2 and 3 to extend instead of
  replace.

**Execution checklist**
- [ ] Add a dedicated Phase 3 Task 1 artifact location
      (for example under `benchmarks/density_matrix/artifacts/phase3_task1/story1_continuity/`).
- [ ] Record source label, qubit count, operation counts, and ordered operation
      metadata for the supported continuity cases.
- [ ] Record the rerun command and software metadata with the artifact.
- [ ] Keep the artifact narrow to canonical planner entry rather than runtime or
      performance interpretation.

**Evidence produced**
- One stable Story 1 continuity artifact bundle.
- One reusable output schema for later planner-surface stories.

**Risks / rollback**
- Risk: prose-only closure makes continuity-entry review hard to repeat and easy
  to misstate in later Task 1 outputs.
- Rollback/mitigation: emit one thin, machine-reviewable artifact bundle early.

### Engineering Task 6: Document The Story 1 Handoff To Later Phase 3 Planner Work

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Reaches The Canonical Planner Surface`

**Change type**
- docs

**Definition of done**
- Story 1 notes explain what was closed and what remains for later stories.
- The continuity anchor is documented as a required source workload, not as the
  entire Phase 3 planner surface.
- Developer-facing notes point to the continuity validation and artifact bundle.

**Execution checklist**
- [ ] Document the supported continuity-entry slice and its evidence surface.
- [ ] Explain that structured-family coverage belongs to Story 2.
- [ ] Explain that richer planner-operation audit fields belong to Story 3.
- [ ] Explain that optional legacy-source lowering and unsupported-case closure
      belong to Stories 4 and 5.

**Evidence produced**
- Updated developer-facing notes for the Story 1 continuity-entry gate.
- One stable place where later Task 1 work can find the Story 1 evidence path.

**Risks / rollback**
- Risk: later Task 1 work may over-assume Story 1 closed the entire canonical
  planner surface.
- Rollback/mitigation: document the exact handoff boundaries explicitly.

### Engineering Task 7: Run Story 1 Validation And Confirm Continuity Planner Entry

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Reaches The Canonical Planner Surface`

**Change type**
- tests | validation automation

**Definition of done**
- Fast Story 1 regression checks pass.
- The continuity-entry validation command or bundle runs successfully.
- Story 1 finishes with stored evidence rather than with code changes alone.

**Execution checklist**
- [ ] Run focused Story 1 regression coverage for continuity canonicalization.
- [ ] Run the Story 1 continuity-entry validation script or artifact-emission
      path.
- [ ] Confirm all required continuity sizes appear in the output.
- [ ] Record stable test and artifact references for handoff into Story 2 and
      later Phase 3 tasks.

**Evidence produced**
- Passing Story 1 regression checks.
- One stable continuity planner-entry artifact or command reference.

**Risks / rollback**
- Risk: Story 1 can appear complete while still lacking reviewable proof that
  the continuity anchor actually reached the canonical planner surface.
- Rollback/mitigation: make rerunnable evidence part of the Story 1 exit gate.

## Exit Criteria

Story 1 is complete only when all of the following are true:

- the frozen Phase 2 noisy XXZ `HEA` continuity workflow at 4, 6, 8, and 10
  qubits reaches one canonical ordered noisy mixed-state planner surface,
- the continuity canonicalization path is rooted in one explicit bridge or
  adapter surface rather than multiple competing interpretations,
- ordered operation and parameter metadata remain stable enough for later
  semantic-preservation and runtime tasks to reuse,
- one stable validation command or artifact bundle proves canonical planner
  entry for the continuity anchor,
- and structured-family coverage, richer planner-entry audit fields, optional
  legacy-source lowering, and unsupported-case closure remain clearly assigned
  to Stories 2 through 5.

## Implementation Notes

- Prefer reusing the existing Phase 2 density bridge inspection path over
  rebuilding continuity lowering from raw VQE internals.
- Keep the contract surface ordered and auditable even if an internal DAG view
  is added later for heuristics.
- Treat the continuity anchor as a required Phase 3 workload, not as evidence
  that the whole planner surface is already complete.
