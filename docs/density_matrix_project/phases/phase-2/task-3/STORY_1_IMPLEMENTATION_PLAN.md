# Story 1 Implementation Plan

## Story Being Implemented

Story 1: Supported Generated `HEA` Circuits Lower Into `NoisyCircuit` With
Explicit Ordered Local Noise

This is a Layer 4 engineering plan for implementing the first behavioral slice
from `TASK_3_STORIES.md`.

## Scope

This story turns the frozen positive bridge contract into explicit, auditable
runtime behavior on the existing VQE path:

- explicit `density_matrix` selection on supported anchor workloads reaches the
  VQE-side bridge rather than a standalone-only density helper,
- the default generated `HEA` circuit lowers into `NoisyCircuit` through the
  supported `U3` / `CNOT` bridge surface,
- ordered local-noise insertion is explicit, reviewable, and attached to the
  same bridged operation stream the workflow executes,
- and the resulting bridge path can be inspected directly enough that later
  stories can add support-matrix breadth, unsupported-case coverage, and
  provenance packaging without replacing the bridge itself.

Out of scope for this story:

- the 1 to 3 qubit support-matrix micro-validation gate owned by Story 2,
- unsupported-case hard-error closure owned by Story 3,
- the 4/6/8/10 workflow-scale bridge-completion package owned by Story 4,
- the full reproducibility and publication bundle owned by Story 5,
- broad source-surface expansion beyond the guaranteed generated `HEA` path,
- and any attempt to claim full `qgd_Circuit` or `Gates_block` parity.

## Dependencies And Assumptions

- Task 1 already provides explicit backend selection and the no-fallback
  contract for `density_matrix`.
- The current VQE bridge substrate already exists in
  `validate_density_anchor_support()`,
  `lower_anchor_circuit_to_noisy_circuit()`,
  `append_density_noise_for_gate_index()`, and
  `evaluate_density_matrix_backend()` inside
  `squander/src-cpp/variational_quantum_eigensolver/Variational_Quantum_Eigensolver_Base.cpp`.
- The Python-facing `NoisyCircuit` binding already exposes
  `get_operation_info()`, which can serve as the natural inspection surface for
  ordered bridge output.
- Existing positive anchor tests in `tests/VQE/test_VQE.py` and workflow-level
  validation in `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py`
  are the right starting points for Story 1 evidence.
- Story 1 should harden the supported positive bridge slice; it should not
  reopen the frozen bridge scope in `P2-ADR-011`, the support matrix in
  `P2-ADR-012`, or the workload anchor in `P2-ADR-013`.

## Engineering Tasks

### Engineering Task 1: Stabilize Density-Backed Bridge Dispatch For Supported Generated-`HEA` Calls

**Implements story**
- `Story 1: Supported Generated HEA Circuits Lower Into NoisyCircuit With Explicit Ordered Local Noise`

**Change type**
- code | tests

**Definition of done**
- Supported fixed-parameter VQE calls with explicit `density_matrix` selection
  reach the bridge path through the standard VQE-facing entry point.
- The bridge entry point is visibly rooted in the existing VQE runtime rather
  than in a parallel standalone-only execution path.
- Legacy `state_vector` behavior remains unaffected for callers outside the
  Story 1 slice.

**Execution checklist**
- [ ] Review the current `Optimization_Problem()` and closely related
      density-backed entry points to confirm where the bridge is activated.
- [ ] Keep bridge dispatch anchored on the existing VQE path that already calls
      `validate_density_anchor_support()` and
      `evaluate_density_matrix_backend()`.
- [ ] Add or tighten focused tests proving supported explicit
      `density_matrix` requests really traverse the bridge-enabled runtime path.

**Evidence produced**
- Reviewable VQE-side runtime dispatch for the supported bridge slice.
- Focused regression coverage proving the bridge path is active on supported
  generated-`HEA` requests.

**Risks / rollback**
- Risk: dispatch edits can accidentally perturb the legacy `state_vector` path
  or create a second inconsistent bridge entry point.
- Rollback/mitigation: keep the Story 1 branch narrow, rooted in one runtime
  path, and validated against the existing backend-selection contract.

### Engineering Task 2: Freeze Generated-`HEA` Lowering Into The Required `U3` / `CNOT` `NoisyCircuit` Sequence

**Implements story**
- `Story 1: Supported Generated HEA Circuits Lower Into NoisyCircuit With Explicit Ordered Local Noise`

**Change type**
- code | tests

**Definition of done**
- The generated `HEA` source is lowered into `NoisyCircuit` through one
  reviewable lowering path.
- The positive Story 1 bridge slice uses only the frozen required gate families
  on the density target side.
- Parameter order and gate order remain stable enough for later micro-validation
  and workflow-scale stories to audit directly.

**Execution checklist**
- [ ] Review `lower_anchor_circuit_to_noisy_circuit()` against the frozen
      generated-`HEA` bridge contract.
- [ ] Keep the supported lowering path limited to the required `U3` / `CNOT`
      surface and avoid broadening source or gate compatibility here.
- [ ] Add focused checks for gate-order and parameter-order stability on a
      deterministic supported fixture.

**Evidence produced**
- One reviewable lowering path from generated `HEA` circuitry into
  `NoisyCircuit`.
- Focused gate-order and parameter-order evidence for the supported bridge path.

**Risks / rollback**
- Risk: the positive path can appear to work while hidden gate-order or
  parameter-order drift breaks later exactness and provenance work.
- Rollback/mitigation: inspect and test the lowered gate stream directly before
  relying on larger workflow evidence.

### Engineering Task 3: Keep Ordered Local-Noise Insertion Explicit And Stable On The Bridge Path

**Implements story**
- `Story 1: Supported Generated HEA Circuits Lower Into NoisyCircuit With Explicit Ordered Local Noise`

**Change type**
- code

**Definition of done**
- Ordered local-noise insertion is attached explicitly to the bridged gate
  sequence rather than inferred heuristically at execution time.
- The supported noise insertion order is stable enough to inspect and serialize
  later.
- Story 1 uses only the frozen local-noise contract: local depolarizing,
  amplitude damping, and phase damping / dephasing.

**Execution checklist**
- [ ] Review `append_density_noise_for_gate_index()` and its call site inside
      `lower_anchor_circuit_to_noisy_circuit()`.
- [ ] Keep noise insertion ordered by the actual bridged gate index so later
      stories can reuse the same ordering vocabulary.
- [ ] Avoid mixing optional or deferred noise families into the Story 1 positive
      slice.

**Evidence produced**
- Reviewable ordered-noise insertion logic on the supported bridge path.
- One stable noise-ordering convention for reuse in later Task 3 stories.

**Risks / rollback**
- Risk: bridge logic may lower the right gates but attach noise in the wrong
  order, weakening both scientific validity and later artifact auditability.
- Rollback/mitigation: keep ordered-noise insertion explicit and tied to the
  same gate index vocabulary across code and tests.

### Engineering Task 4: Expose A Reviewable Bridge-Inspection Surface For Supported Cases

**Implements story**
- `Story 1: Supported Generated HEA Circuits Lower Into NoisyCircuit With Explicit Ordered Local Noise`

**Change type**
- code | tests | validation automation

**Definition of done**
- Supported bridge outputs can be inspected without reverse-engineering internal
  state from unrelated code paths.
- The inspection surface makes `GateOperation` and `NoiseOperation` ordering
  auditable on deterministic Story 1 fixtures.
- The inspection path remains narrow and bridge-focused rather than becoming a
  generic new public API for arbitrary circuit introspection.

**Execution checklist**
- [ ] Reuse `NoisyCircuit.get_operation_info()` or add the smallest equivalent
      bridge-facing inspection helper needed for Story 1.
- [ ] Make sure the inspection output distinguishes unitary and noise operations
      clearly enough for story-level review.
- [ ] Add focused tests that assert the supported bridged circuit exposes a
      reviewable ordered operation list.

**Evidence produced**
- One reviewable inspection surface for supported bridged circuits.
- Focused tests proving Story 1 can audit ordered gate and noise content.

**Risks / rollback**
- Risk: if inspection requires ad hoc debugger work, later bridge stories will
  drift into unverifiable claims.
- Rollback/mitigation: keep one narrow, stable inspection surface and reuse it
  across validation and artifact generation.

### Engineering Task 5: Add Positive Regression Coverage For The Supported Bridge Slice

**Implements story**
- `Story 1: Supported Generated HEA Circuits Lower Into NoisyCircuit With Explicit Ordered Local Noise`

**Change type**
- tests

**Definition of done**
- Fast automated tests cover the supported generated-`HEA` bridge slice on the
  VQE path.
- The tests demonstrate that the workflow traverses the documented bridge and
  yields inspectable ordered operations.
- Regression coverage stays small and deterministic enough to run regularly
  during development.

**Execution checklist**
- [ ] Extend or tighten `tests/VQE/test_VQE.py` for supported bridge-positive
      cases.
- [ ] Add assertions that the bridged path can expose ordered operation metadata
      on deterministic fixtures.
- [ ] Keep Story 1 regression tests focused on the positive supported slice and
      leave wider coverage to later stories.

**Evidence produced**
- Focused pytest coverage for the supported bridge-positive path.
- Reviewable failures that localize Story 1 bridge regressions.

**Risks / rollback**
- Risk: weak smoke tests can prove only that execution finished, not that the
  documented bridge behavior actually occurred.
- Rollback/mitigation: assert bridge-specific inspection signals, not just final
  scalar results.

### Engineering Task 6: Emit A Stable Story 1 Bridge Artifact Or Rerunnable Command

**Implements story**
- `Story 1: Supported Generated HEA Circuits Lower Into NoisyCircuit With Explicit Ordered Local Noise`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Story 1 can emit at least one stable artifact or rerunnable command that shows
  the supported bridge output on a deterministic anchor fixture.
- The artifact records the circuit source, backend, and enough ordered-operation
  metadata to audit the positive bridge slice.
- The output format is stable enough that later stories can extend it rather
  than replace it.

**Execution checklist**
- [ ] Extend `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` or a
      tightly related successor with Story 1 bridge-inspection output.
- [ ] Record source type, ansatz, backend, and ordered operation metadata for at
      least one deterministic supported case.
- [ ] Keep the artifact narrow to Story 1 positive-bridge evidence rather than
      the full Task 3 provenance bundle.

**Evidence produced**
- One stable Story 1 bridge artifact or rerunnable command.
- A bridge-output schema that later Task 3 stories can build on.

**Risks / rollback**
- Risk: ad hoc console-only inspection output will be hard to audit and harder
  to reuse later.
- Rollback/mitigation: define one small machine-readable output now and extend it
  incrementally.

### Engineering Task 7: Update Developer-Facing Notes For The Supported Bridge Entry Point

**Implements story**
- `Story 1: Supported Generated HEA Circuits Lower Into NoisyCircuit With Explicit Ordered Local Noise`

**Change type**
- docs

**Definition of done**
- Developer-facing notes identify the supported generated-`HEA` bridge entry
  point and the required `NoisyCircuit` target representation.
- The notes make clear that Story 1 closes only the positive bridge slice, not
  the full support matrix, unsupported-case matrix, or provenance bundle.
- The documented support boundary matches the frozen Task 3 contract exactly.

**Execution checklist**
- [ ] Update the most relevant bridge-facing docstrings, examples, or developer
      notes near the VQE density path.
- [ ] Keep wording aligned with `P2-ADR-011` and `TASK_3_MINI_SPEC.md`.
- [ ] Refer later support-matrix, unsupported-case, and provenance work to
      Stories 2 to 5 rather than overclaiming closure here.

**Evidence produced**
- Updated developer-facing notes for the Story 1 bridge-positive slice.
- One stable place where the supported bridge entry point is documented.

**Risks / rollback**
- Risk: documentation can easily outrun the actual supported bridge slice and
  create false parity expectations.
- Rollback/mitigation: tie the notes directly to the same generated-`HEA`
  fixtures used by Story 1 tests and validation output.

### Engineering Task 8: Run Story 1 Validation And Confirm The Positive Bridge Slice

**Implements story**
- `Story 1: Supported Generated HEA Circuits Lower Into NoisyCircuit With Explicit Ordered Local Noise`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 1 bridge-positive tests pass.
- At least one rerunnable validation command or artifact demonstrates the
  supported positive bridge slice outside the fast test layer.
- Story 1 exits with reviewable bridge evidence rather than with code changes
  alone.

**Execution checklist**
- [ ] Run the focused Story 1 pytest coverage for supported bridge-positive
      cases.
- [ ] Run the Story 1 bridge-inspection validation command or artifact emission
      path.
- [ ] Confirm the resulting evidence is named or stored stably enough for later
      Task 3 stories to reuse.

**Evidence produced**
- Passing focused Story 1 pytest coverage.
- One stable artifact or command reference for the supported positive bridge
  slice.

**Risks / rollback**
- Risk: Story 1 can look complete without any stored evidence that the bridge
  output itself was reviewed.
- Rollback/mitigation: treat bridge inspection evidence as part of the exit
  gate, not optional follow-up.

## Exit Criteria

Story 1 is complete only when all of the following are true:

- supported generated `HEA` anchor requests with explicit `density_matrix`
  selection traverse the documented VQE-side bridge path,
- the bridged output reaches `NoisyCircuit` through a stable, reviewable
  lowering path using the required `U3` / `CNOT` surface,
- ordered local-noise insertion is explicit and auditable on deterministic
  supported fixtures,
- at least one stable inspection artifact or rerunnable command proves the
  supported bridge slice outside the unit-test layer,
- and support-matrix breadth, unsupported-case closure, workflow-scale bridge
  completion, and final provenance packaging remain clearly assigned to later
  stories.

## Implementation Notes

- `validate_density_anchor_support()`,
  `lower_anchor_circuit_to_noisy_circuit()`, and
  `append_density_noise_for_gate_index()` already provide the natural backbone
  for Story 1. Implementation should harden and expose this path rather than
  invent a second bridge framework.
- `NoisyCircuit.get_operation_info()` is the most natural existing inspection
  surface for auditing ordered gate and noise output. Story 1 should reuse it or
  extend it minimally.
- `tests/VQE/test_VQE.py` already contains density-backend anchor smoke tests,
  and `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` already
  provides the most natural workflow-facing validation entry point.
- Story 1 should stay tightly bounded to the guaranteed generated-`HEA` bridge
  path and required local-noise contract even though the underlying
  `NoisyCircuit` module can express more operations.
