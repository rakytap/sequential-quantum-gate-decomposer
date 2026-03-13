# Story 2 Implementation Plan

## Story Being Implemented

Story 2: Required Gate And Local-Noise Support Lowers Cleanly On
Micro-Validation Cases

This is a Layer 4 engineering plan for implementing the second behavioral slice
from `TASK_3_STORIES.md`.

## Scope

This story turns the frozen bridge support surface into a reproducible
micro-validation gate that can catch local lowering and ordering regressions
before the phase depends on larger workflow-scale evidence:

- the mandatory bridge-validation matrix covers the required `U3` and `CNOT`
  gate surface plus each required local-noise model,
- each required microcase lowers through the documented bridge vocabulary
  without manual rewriting or bridge-surface exceptions,
- ordered bridge output is inspectable in a machine-readable way for every
  mandatory case,
- and the micro-validation layer stays small and fast enough to serve as the
  local bridge-correctness gate below the 4 to 10 qubit workflow package.

Out of scope for this story:

- broad source-surface expansion beyond the guaranteed generated-`HEA` path,
- unsupported-case hard-error closure owned by Story 3,
- the 4/6/8/10 workflow-scale bridge package owned by Story 4,
- the final provenance and publication bundle owned by Story 5,
- and numeric exactness or observable-threshold claims already owned by Task 2.

## Dependencies And Assumptions

- Story 1 is already in place: the supported positive bridge slice exists on the
  VQE path and exposes a reviewable inspection surface through
  `describe_density_bridge()` backed by the C++ `inspect_density_bridge()`
  implementation.
- The low-level density-matrix substrate already exists in
  `squander/src-cpp/density_matrix/`, `squander/density_matrix/bindings.cpp`,
  and `tests/density_matrix/test_density_matrix.py`.
- Story 1 already established a canonical bridge-artifact vocabulary in
  `benchmarks/density_matrix/story2_vqe_density_validation.py`:
  `bridge_source_type`, `bridge_parameter_count`, `bridge_operation_count`,
  `bridge_gate_count`, `bridge_noise_count`, and `bridge_operations`.
- Existing density-matrix benchmark scaffolding in
  `benchmarks/density_matrix/circuits.py` and
  `benchmarks/density_matrix/validate_squander_vs_qiskit.py` can be reused where
  that helps, but Story 2 should not widen the user-facing bridge contract.
- Current implementation-backed constraint from Story 1: any validation path that
  calls `get_Qiskit_Circuit()` should first set a deterministic optimized
  parameter vector on the VQE instance rather than assuming export is safe before
  parameters are initialized.
- Story 2 should validate the required bridge support matrix locally without
  reopening `P2-ADR-011` or `P2-ADR-012`.

## Engineering Tasks

### Engineering Task 1: Freeze The Mandatory 1 To 3 Qubit Bridge-Validation Matrix

**Implements story**
- `Story 2: Required Gate And Local-Noise Support Lowers Cleanly On Micro-Validation Cases`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- The Story 2 micro-validation matrix explicitly covers the required bridge
  surface at 1 to 3 qubits.
- Each mandatory case has a stable identifier, a declared lowering purpose, and
  an expected gate/noise pattern it is meant to exercise.
- Mandatory cases remain clearly separated from optional exploratory or future
  parity cases.

**Execution checklist**
- [ ] Enumerate mandatory 1 to 3 qubit bridge-validation cases covering `U3`,
      `CNOT`, local depolarizing, amplitude damping, and phase damping /
      dephasing individually and in mixed sequences.
- [ ] Give each microcase a stable case ID and a short purpose statement.
- [ ] Keep optional gate families, optional source representations, and stress
      cases outside the mandatory Story 2 gate.
- [ ] Record the matrix in one stable implementation-facing location that later
      stories can reuse.

**Evidence produced**
- A named mandatory bridge-validation matrix for the required 1 to 3 qubit
  support surface.
- Stable case IDs for reuse in tests, artifacts, and later docs.

**Risks / rollback**
- Risk: an underspecified matrix can leave required gate or noise coverage
  ambiguous while still claiming Story 2 completion.
- Rollback/mitigation: keep the mandatory matrix small but explicitly traceable
  to the frozen required gate and noise contract only.

### Engineering Task 2: Add Reusable Story 2 Bridge Case Builders Without Widening The User Contract

**Implements story**
- `Story 2: Required Gate And Local-Noise Support Lowers Cleanly On Micro-Validation Cases`

**Change type**
- code | benchmark harness

**Definition of done**
- Story 2 can construct deterministic microcases that exercise the required
  bridge surface cleanly.
- The case builders stay narrow to validation needs and do not imply new
  user-facing source surfaces.
- Generated case metadata captures the intended gate and noise coverage for each
  microcase.

**Execution checklist**
- [ ] Add or refine compact microcase builders for the mandatory Story 2 matrix.
- [ ] Reuse generated `HEA` fragments or bridge-helper fixtures where that keeps
      the Story 2 cases honest to the real bridge contract.
- [ ] Set deterministic optimized parameters before any Story 2 helper depends
      on `get_Qiskit_Circuit()` or other parameter-sensitive export paths.
- [ ] Avoid turning Story 2 builders into an implied full `qgd_Circuit` parity
      layer.
- [ ] Record intended gate/noise coverage as part of the case metadata.

**Evidence produced**
- Reusable builders for the mandatory Story 2 bridge microcases.
- Stable per-case metadata that says what each microcase is exercising.

**Risks / rollback**
- Risk: overly synthetic microcases can drift away from the actual bridge while
  still producing attractive local results.
- Rollback/mitigation: keep builders tied to the documented bridge vocabulary and
  source assumptions.

### Engineering Task 3: Add Operation-Sequence Inspection And Expected-Shape Assertions

**Implements story**
- `Story 2: Required Gate And Local-Noise Support Lowers Cleanly On Micro-Validation Cases`

**Change type**
- code | tests | validation automation

**Definition of done**
- Every mandatory Story 2 microcase can expose ordered operation metadata from
  the VQE-side supported bridge.
- The validation layer can assert the expected shape of the lowered gate and
  noise sequence for each microcase.
- Inspection output is machine-readable and stable enough for later provenance
  packaging.

**Execution checklist**
- [ ] Reuse `describe_density_bridge()` and the canonical Story 1
      `bridge_operations` schema first; only drop to lower-level
      `NoisyCircuit.get_operation_info()` helpers when that materially improves
      defect localization.
- [ ] Add expected-shape assertions for required gate presence, required noise
      presence, and ordering of mixed sequences.
- [ ] Keep inspection vocabulary stable across tests and artifacts.
- [ ] Ensure Story 2 distinguishes gate operations from noise operations clearly.

**Evidence produced**
- Machine-readable ordered-operation output for mandatory bridge microcases.
- Focused checks that assert expected lowered bridge shapes case by case.

**Risks / rollback**
- Risk: without ordered-operation assertions, Story 2 can prove only that
  something lowered, not that the right bridged sequence was produced.
- Rollback/mitigation: treat expected-shape checks as part of the mandatory
  microcase contract.

### Engineering Task 4: Confirm Mandatory Microcases Produce Execution-Ready `NoisyCircuit` Outputs

**Implements story**
- `Story 2: Required Gate And Local-Noise Support Lowers Cleanly On Micro-Validation Cases`

**Change type**
- code | tests

**Definition of done**
- Mandatory Story 2 bridge outputs are not only inspectable, but executable by
  the density backend without bridge-surface exceptions.
- Required gate/noise combinations do not require hidden manual corrections or
  special-case workarounds.
- Story 2 remains focused on bridge readiness, not on broader observable or
  performance claims.

**Execution checklist**
- [ ] Add focused checks that the bridged `NoisyCircuit` outputs can be consumed
      cleanly by the density execution layer on mandatory microcases.
- [ ] Keep these checks narrow to execution readiness and avoid duplicating the
      full Task 2 exactness matrix.
- [ ] Localize failures so they can be distinguished from pure inspection
      mismatches.

**Evidence produced**
- Focused execution-readiness checks for the mandatory bridge microcases.
- Reviewable failures that distinguish lowering defects from later execution
  issues.

**Risks / rollback**
- Risk: a bridge output can look structurally plausible while still failing when
  handed to the actual density execution layer.
- Rollback/mitigation: require every mandatory Story 2 microcase to be
  execution-ready, not only inspectable.

### Engineering Task 5: Add Focused Regression Tests For The Mandatory Bridge Microcases

**Implements story**
- `Story 2: Required Gate And Local-Noise Support Lowers Cleanly On Micro-Validation Cases`

**Change type**
- tests

**Definition of done**
- Fast automated tests cover representative mandatory Story 2 bridge microcases.
- The regression layer catches local gate-lowering, noise-ordering, and
  execution-readiness regressions without requiring the larger workflow package
  to run.
- Test failures are specific enough to localize the bridge defect category.

**Execution checklist**
- [ ] Extend `tests/VQE/test_VQE.py` or related focused bridge tests with
      representative mandatory microcases.
- [ ] Reuse existing low-level `tests/density_matrix/test_density_matrix.py`
      coverage where it helps avoid duplication.
- [ ] Keep exhaustive matrix execution in dedicated validation commands rather
      than the fast default suite.
- [ ] Make test failures specific enough to distinguish source, gate, noise, and
      ordering defects.

**Evidence produced**
- Focused pytest coverage for the Story 2 bridge micro-validation slice.
- Reviewable failures that localize local bridge regressions.

**Risks / rollback**
- Risk: putting the whole matrix into the default suite can make Story 2 slow
  and brittle.
- Rollback/mitigation: keep one representative fast regression layer and put the
  full mandatory matrix in a dedicated validation command.

### Engineering Task 6: Emit A Stable Machine-Readable Story 2 Bridge Artifact Bundle

**Implements story**
- `Story 2: Required Gate And Local-Noise Support Lowers Cleanly On Micro-Validation Cases`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Story 2 produces a stable machine-readable output bundle for the mandatory
  bridge microcases.
- Each record includes case identity, source type, bridge metadata, ordered
  operation summary, and pass/fail status.
- The artifact shape is stable enough that Story 5 can later reuse it directly.

**Execution checklist**
- [ ] Define the per-case fields required for mandatory Story 2 bridge artifacts.
- [ ] Start from the Story 1 canonical bridge fields:
      `bridge_source_type`, `bridge_parameter_count`,
      `bridge_operation_count`, `bridge_gate_count`, `bridge_noise_count`, and
      `bridge_operations`.
- [ ] Record source type, ansatz, gate/noise coverage intent, ordered-operation
      metadata, and pass/fail status in that same vocabulary.
- [ ] Keep artifact naming stable and easy to reference in later docs and
      bundles.
- [ ] Avoid mixing workflow-scale or publication-only fields into the Story 2
      bundle.

**Evidence produced**
- A stable machine-readable Story 2 bridge artifact bundle.
- A reproducible command that regenerates the mandatory Story 2 outputs.

**Risks / rollback**
- Risk: console-only or ad hoc validation output will not be auditable enough
  for later bridge provenance work.
- Rollback/mitigation: define a minimal structured schema now and extend it
  incrementally.

### Engineering Task 7: Document The Story 2 Validation Entry Point And Scope Boundary

**Implements story**
- `Story 2: Required Gate And Local-Noise Support Lowers Cleanly On Micro-Validation Cases`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Story 2 validates, how to rerun it, and
  which cases are mandatory.
- The notes make clear that Story 2 closes the local bridge support-surface gate
  below the workflow-scale package.
- The documentation stays aligned with the frozen required gate and noise
  contract and does not imply broader parity.

**Execution checklist**
- [ ] Document the Story 2 validation entry point and mandatory case inventory.
- [ ] Make the required bridge surface explicit: `U3`, `CNOT`, and the required
      local-noise models.
- [ ] Explain how Story 2 depends on Story 1 and hands off to Stories 3 to 5.
- [ ] Keep optional gates, optional sources, and later provenance packaging
      clearly out of Story 2 scope.

**Evidence produced**
- Updated developer-facing guidance for the Story 2 bridge-validation gate.
- One stable location where Story 2 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 2 is undocumented, later contributors may skip it or confuse it
  with broader parity testing.
- Rollback/mitigation: tie the notes directly to the same command and artifact
  bundle used by Story 2 validation.

### Engineering Task 8: Run Story 2 Micro-Validation And Confirm The Local Bridge Gate

**Implements story**
- `Story 2: Required Gate And Local-Noise Support Lowers Cleanly On Micro-Validation Cases`

**Change type**
- tests | validation automation

**Definition of done**
- The mandatory Story 2 bridge-validation matrix runs successfully end to end.
- Every required microcase lowers cleanly and produces auditable ordered bridge
  output.
- Story 2 completion is backed by reviewable artifacts rather than by code
  changes alone.

**Execution checklist**
- [ ] Run the focused Story 2 regression tests.
- [ ] Run the dedicated Story 2 validation command that emits the mandatory
      bridge artifact bundle.
- [ ] Verify `100%` pass rate on the mandatory Story 2 bridge matrix.
- [ ] Record the stable test run and artifact references for later Task 3 docs
      and provenance work.

**Evidence produced**
- Passing focused pytest coverage for Story 2.
- A machine-readable Story 2 bridge artifact bundle with a `100%` pass rate on
  mandatory cases.

**Risks / rollback**
- Risk: Story 2 can appear complete while still lacking a reproducible proof of
  the local bridge support surface.
- Rollback/mitigation: treat the emitted artifact bundle and full pass rate as
  part of the exit criteria, not optional follow-up.

## Exit Criteria

Story 2 is complete only when all of the following are true:

- the mandatory 1 to 3 qubit bridge-validation matrix covers the required
  `U3` / `CNOT` and local-noise contract,
- every mandatory microcase lowers cleanly through the documented bridge without
  manual rewriting or unsupported-operation workarounds,
- ordered operation metadata is available and auditable for every mandatory
  microcase,
- every mandatory microcase produces an execution-ready bridged `NoisyCircuit`
  output,
- and the results are available through one stable validation command and one
  stable machine-readable artifact bundle.

## Implementation Notes

- Story 2 should reuse the Story 1 bridge-inspection surface,
  `describe_density_bridge()`, and the already implemented
  `bridge_operations` schema rather than inventing a second incompatible
  inspection path.
- `tests/density_matrix/test_density_matrix.py` already exercises many low-level
  density operations and is the right place to borrow lightweight regression
  patterns where helpful.
- `benchmarks/density_matrix/circuits.py` and
  `benchmarks/density_matrix/validate_squander_vs_qiskit.py` already provide a
  useful low-level validation scaffold; Story 2 should extend that style where
  it helps without widening the bridge contract.
- Story 1 exposed a practical wrapper constraint: parameter-sensitive export
  paths such as `get_Qiskit_Circuit()` should be treated as requiring an
  explicit deterministic parameter vector first. Story 2 validation helpers
  should freeze parameters before using those paths.
- Story 2 should stay tightly bounded to the required `U3` / `CNOT` plus local
  depolarizing / amplitude damping / phase damping surface even though the
  underlying density module can express more operations.
