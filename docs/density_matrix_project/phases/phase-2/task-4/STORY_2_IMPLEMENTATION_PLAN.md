# Story 2 Implementation Plan

## Story Being Implemented

Story 2: Required Local Noise Models And Mixed Schedules Validate Cleanly On 1
To 3 Qubit Exact Microcases

This is a Layer 4 engineering plan for implementing the second behavioral slice
from `TASK_4_STORIES.md`.

## Scope

This story turns the required local-noise baseline from Story 1 into a
reproducible exactness gate on small deterministic cases before the phase
depends on larger workflow-scale evidence:

- the mandatory 1 to 3 qubit micro-validation matrix covers each required local
  noise model individually and at least one mixed schedule composed only from
  required local models,
- every mandatory microcase compares SQUANDER against Qiskit Aer on the frozen
  exact observable contract `Re Tr(H*rho)`,
- every mandatory microcase records density-validity, trace-preservation, and
  Hermitian-observable consistency metrics at the frozen tolerances,
- and the mixed-sequence layer preserves auditable local-noise order rather than
  validating only aggregate scalar outputs.

Out of scope for this story:

- the positive VQE-path visibility slice already owned by Story 1,
- optional-versus-required classification closure owned by Story 3,
- unsupported and deferred noise hard-error closure owned by Story 4,
- the 4/6/8/10 workflow-scale sufficiency package owned by Story 5,
- the full reproducibility and publication bundle owned by Story 6,
- whole-register depolarizing or other optional extensions as part of the
  mandatory pass/fail set,
- and broad API or workflow expansion beyond the exact micro-validation gate.

## Dependencies And Assumptions

- Story 1 is already in place: the required local-noise models execute on the
  supported VQE-facing density path and expose reviewable bridge metadata through
  `required_local_noise_validation_validation.py`.
- The current low-level exactness scaffold already exists in
  `benchmarks/density_matrix/circuits.py`,
  `benchmarks/density_matrix/validate_squander_vs_qiskit.py`, and
  `tests/density_matrix/test_density_matrix.py`.
- `MANDATORY_MICROCASES` in
  `benchmarks/density_matrix/circuits.py` already provides a near-complete
  mandatory matrix for this story:
  1-qubit individual-noise cases, 2-qubit `U3` / `CNOT` individual-noise cases,
  and a 3-qubit mixed required-noise sequence.
- `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` already imports the micro-validation bundle
  into the broader workflow package, so Story 2 output shape should stay stable
  enough for later reuse rather than forking into a wholly separate artifact
  vocabulary.
- The frozen Story 2 numeric gate is already defined by `P2-ADR-014` and
  `P2-ADR-015`:
  - maximum absolute energy error `<= 1e-10` on mandatory 1 to 3 qubit
    microcases,
  - `rho.is_valid(tol=1e-10)`,
  - `|Tr(rho) - 1| <= 1e-10`,
  - `|Im Tr(H*rho)| <= 1e-10`,
  - and `100%` pass rate on the mandatory micro-validation matrix.
- Story 2 should harden and reuse the current exactness kernel rather than
  reopen the phase-level support split, workflow anchor, or larger benchmark
  package.

## Engineering Tasks

### Engineering Task 1: Freeze The Mandatory 1 To 3 Qubit Required-Noise Micro-Validation Matrix

**Implements story**
- `Story 2: Required Local Noise Models And Mixed Schedules Validate Cleanly On 1 To 3 Qubit Exact Microcases`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- The Story 2 mandatory micro-validation matrix explicitly covers the frozen
  required local-noise baseline at 1 to 3 qubits.
- Each mandatory case has a stable identifier, declared purpose, and clear
  mapping to the required noise coverage it is meant to prove.
- The matrix separates mandatory cases from optional exploratory or later-phase
  cases so the Story 2 gate remains unambiguous.

**Execution checklist**
- [ ] Review and freeze the mandatory Story 2 case inventory in
      `MANDATORY_MICROCASES`.
- [ ] Confirm that the matrix covers `local_depolarizing`,
      `amplitude_damping`, and `phase_damping` individually, plus at least one
      mixed required-noise schedule.
- [ ] Keep whole-register depolarizing and other optional or deferred models out
      of the mandatory pass/fail set.
- [ ] Preserve stable case IDs and purpose strings for reuse in tests,
      validation bundles, and later docs.

**Evidence produced**
- A named mandatory Story 2 micro-validation matrix for the required local-noise
  contract.
- Stable case identifiers and coverage metadata reusable across tests and
  artifacts.

**Risks / rollback**
- Risk: an underspecified matrix can leave required local-noise coverage
  ambiguous while still claiming Story 2 completion.
- Rollback/mitigation: keep the matrix small but explicitly traceable to the
  frozen required-model set only.

### Engineering Task 2: Harden Deterministic Microcase Builders And Hamiltonian Fixtures

**Implements story**
- `Story 2: Required Local Noise Models And Mixed Schedules Validate Cleanly On 1 To 3 Qubit Exact Microcases`

**Change type**
- code | benchmark harness

**Definition of done**
- Story 2 can build deterministic microcases that exercise each required local
  model and the mixed-sequence case repeatably.
- Each microcase carries a concrete Hermitian Hamiltonian suitable for exact
  `Re Tr(H*rho)` comparison.
- Builder metadata stays narrow to validation needs and does not imply a broader
  user-facing circuit-construction contract.

**Execution checklist**
- [ ] Reuse and tighten the existing `DualBuilder`-based builders in
      `benchmarks/density_matrix/circuits.py`.
- [ ] Keep per-case operations, Hamiltonian metadata, and parameter vectors
      deterministic and easy to audit.
- [ ] Preserve the mixed-sequence case as a first-class mandatory microcase
      rather than reducing Story 2 to only individual-noise examples.
- [ ] Avoid creating a second overlapping microcase vocabulary unless a concrete
      validation gap requires it.

**Evidence produced**
- Reusable deterministic builders for the mandatory Story 2 exact microcases.
- Stable Hamiltonian and operation metadata for each mandatory case.

**Risks / rollback**
- Risk: overly synthetic builders can drift away from the actual required local
  noise baseline while still producing attractive exactness numbers.
- Rollback/mitigation: keep the builders tied to the frozen required model names
  and explicit operation order.

### Engineering Task 3: Reuse The Exact Aer Comparison Path And Freeze The Numeric Pass/Fail Gate

**Implements story**
- `Story 2: Required Local Noise Models And Mixed Schedules Validate Cleanly On 1 To 3 Qubit Exact Microcases`

**Change type**
- code | benchmark harness | validation automation

**Definition of done**
- Every mandatory Story 2 microcase compares SQUANDER and Qiskit Aer on the same
  Hermitian Hamiltonian and density matrix observable.
- The validation path records exact-energy error, density validity, trace
  deviation, and Hermitian-observable imaginary-component metrics.
- Story 2 uses the frozen numeric thresholds exactly rather than inventing a
  second local tolerance scheme.

**Execution checklist**
- [ ] Reuse `density_energy()` and the current metric flow in
      `benchmarks/density_matrix/validate_squander_vs_qiskit.py`.
- [ ] Keep the comparison contract explicitly tied to `Re Tr(H*rho)` against
      Qiskit Aer density-matrix simulation.
- [ ] Ensure `rho.is_valid(tol=1e-10)`, `|Tr(rho) - 1| <= 1e-10`, and
      `|Im Tr(H*rho)| <= 1e-10` are part of the mandatory case result schema.
- [ ] Verify the Story 2 pass/fail gate is exactly `100%` pass rate on the
      mandatory matrix.

**Evidence produced**
- A reusable exact-comparison path for mandatory Story 2 microcases.
- Machine-readable numeric pass/fail metrics aligned with `P2-ADR-014` and
  `P2-ADR-015`.

**Risks / rollback**
- Risk: local validation can drift from the frozen Phase 2 thresholds and
  validate the wrong quantity or wrong tolerance.
- Rollback/mitigation: keep the metric names and thresholds explicit and frozen
  in one validation path.

### Engineering Task 4: Preserve Required-Noise Coverage And Mixed-Schedule Order In Machine-Readable Outputs

**Implements story**
- `Story 2: Required Local Noise Models And Mixed Schedules Validate Cleanly On 1 To 3 Qubit Exact Microcases`

**Change type**
- code | tests | validation automation

**Definition of done**
- Mandatory Story 2 outputs make required local-noise coverage visible for each
  case, especially the mixed-sequence case.
- Reviewers can inspect operation order and noise composition without reverse
  engineering the builder implementation.
- Story 2 proves not only numeric agreement, but that the mixed schedule
  remained ordered and auditable.

**Execution checklist**
- [ ] Reuse the existing per-case `operations` output or add the smallest
      equivalent ordered-operation summary needed for Story 2 review.
- [ ] Ensure individual-noise cases clearly identify which required model they
      exercise.
- [ ] Ensure the mixed-sequence case records the ordered combination of
      `local_depolarizing`, `amplitude_damping`, and `phase_damping`.
- [ ] Keep the output vocabulary stable enough for later provenance and workflow
      bundles to reuse directly.

**Evidence produced**
- Machine-readable ordered operation and required-noise coverage output for the
  mandatory Story 2 microcases.
- Reviewable mixed-sequence evidence showing order and composition explicitly.

**Risks / rollback**
- Risk: Story 2 can prove aggregate exactness while still leaving mixed-schedule
  order implicit and unauditable.
- Rollback/mitigation: treat ordered-operation visibility as part of the
  mandatory Story 2 schema, not optional debug output.

### Engineering Task 5: Add Focused Regression Tests For Representative Mandatory Microcases

**Implements story**
- `Story 2: Required Local Noise Models And Mixed Schedules Validate Cleanly On 1 To 3 Qubit Exact Microcases`

**Change type**
- tests

**Definition of done**
- Fast automated tests cover representative mandatory Story 2 microcases and the
  core exact-comparison helper behavior.
- Regression coverage is specific enough to localize failures in noise
  application, exact observable comparison, density validity, or mixed-sequence
  handling.
- The fast layer stays small enough to run regularly while the full matrix
  remains in a dedicated validation command.

**Execution checklist**
- [ ] Extend `tests/density_matrix/test_density_matrix.py` with representative
      mandatory-case and schema-level assertions as needed.
- [ ] Reuse the existing helper and representative microcase tests rather than
      duplicating the entire matrix in pytest.
- [ ] Add focused assertions for mixed-sequence coverage if the current fast
      tests only cover individual-noise cases.
- [ ] Keep exhaustive matrix execution in the validation harness rather than the
      default fast test path.

**Evidence produced**
- Focused pytest coverage for the mandatory Story 2 local exactness slice.
- Reviewable failures that localize Story 2 regressions cleanly.

**Risks / rollback**
- Risk: moving the whole matrix into default pytest makes Story 2 slow and
  brittle.
- Rollback/mitigation: keep representative fast tests in pytest and the full
  mandatory matrix in one dedicated validation command.

### Engineering Task 6: Emit One Stable Task 4 Story 2 Artifact Bundle Without Forking The Exactness Harness

**Implements story**
- `Story 2: Required Local Noise Models And Mixed Schedules Validate Cleanly On 1 To 3 Qubit Exact Microcases`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Story 2 produces one stable machine-readable artifact bundle for the mandatory
  local-noise micro-validation matrix.
- The bundle includes case identity, required-noise coverage, Hamiltonian
  metadata, exactness metrics, density-validity metrics, and pass/fail status.
- The artifact path is stable enough for later Task 4 stories and publication
  bundles to reference without redesigning the schema.

**Execution checklist**
- [ ] Start from `validate_squander_vs_qiskit.py` and its current
      `micro_validation_bundle` output instead of creating a parallel
      exactness harness.
- [ ] Add Task 4-specific requirement metadata only where it materially improves
      traceability for required local-noise coverage.
- [ ] Keep artifact naming stable and easy to reference from later Task 4 docs
      and bundles.
- [ ] Avoid mixing workflow-scale or publication-only fields into the Story 2
      local exactness artifact.

**Evidence produced**
- One stable machine-readable Task 4 Story 2 artifact bundle.
- A reproducible command that regenerates the full mandatory Story 2 outputs.

**Risks / rollback**
- Risk: forking the exactness harness into multiple near-duplicate scripts will
  create schema drift and harder auditability later.
- Rollback/mitigation: reuse one canonical exactness bundle and extend it only
  where traceability clearly benefits.

### Engineering Task 7: Document The Story 2 Validation Entry Point And Its Hand-Offs

**Implements story**
- `Story 2: Required Local Noise Models And Mixed Schedules Validate Cleanly On 1 To 3 Qubit Exact Microcases`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Story 2 validates, how to rerun it, and
  which cases are mandatory.
- The notes make clear that Story 2 closes the small-case exactness gate below
  the workflow package.
- The documentation stays aligned with the frozen required local-noise baseline
  and does not imply optional, deferred, or workflow-scale closure.

**Execution checklist**
- [ ] Document the Story 2 validation command and the mandatory case inventory.
- [ ] Make the required local-noise coverage explicit:
      `local_depolarizing`, `amplitude_damping`, `phase_damping`, and the mixed
      required-noise schedule.
- [ ] Explain how Story 2 builds on Story 1 and hands off optional,
      unsupported-case, workflow-scale, and publication work to Stories 3 to 6.
- [ ] Keep whole-register depolarizing and other optional models clearly out of
      the mandatory Story 2 definition of done.

**Evidence produced**
- Updated developer-facing guidance for the Story 2 micro-validation gate.
- One stable location documenting rerun instructions and scope boundaries.

**Risks / rollback**
- Risk: if Story 2 is poorly documented, later contributors may confuse it with
  broader workflow validation or optional-noise exploration.
- Rollback/mitigation: tie the notes directly to the same command and artifact
  bundle used for Story 2 completion.

### Engineering Task 8: Run Story 2 Micro-Validation And Confirm The Local Exactness Gate

**Implements story**
- `Story 2: Required Local Noise Models And Mixed Schedules Validate Cleanly On 1 To 3 Qubit Exact Microcases`

**Change type**
- tests | validation automation

**Definition of done**
- The full mandatory Story 2 micro-validation matrix runs successfully end to
  end.
- Every mandatory case satisfies the frozen numeric thresholds and the mixed case
  remains auditable in machine-readable output.
- Story 2 completion is backed by stable artifacts and rerunnable commands rather
  than by code changes alone.

**Execution checklist**
- [ ] Run the focused Story 2 pytest coverage.
- [ ] Run the dedicated Story 2 validation command that emits the mandatory
      artifact bundle.
- [ ] Verify `100%` pass rate on the mandatory Story 2 matrix.
- [ ] Record the stable test run and artifact references for later Task 4 docs
      and publication work.

**Evidence produced**
- Passing focused pytest coverage for Story 2.
- A machine-readable Story 2 artifact bundle with a `100%` pass rate on the
  mandatory matrix.

**Risks / rollback**
- Risk: Story 2 can look complete while still lacking a reproducible proof of
  the required local-noise exactness gate.
- Rollback/mitigation: treat the emitted bundle and the full pass rate as part
  of the exit criteria, not optional follow-up.

## Exit Criteria

Story 2 is complete only when all of the following are true:

- the mandatory 1 to 3 qubit matrix covers each required local noise model
  individually and at least one mixed required-noise schedule,
- every mandatory microcase satisfies maximum absolute energy error
  `<= 1e-10`,
- every mandatory microcase satisfies `rho.is_valid(tol=1e-10)`,
  `|Tr(rho) - 1| <= 1e-10`, and `|Im Tr(H*rho)| <= 1e-10`,
- required local-noise coverage and mixed-schedule order are auditable in
  machine-readable output,
- one stable validation command and one stable artifact bundle produce a
  `100%` pass rate on the mandatory matrix,
- and optional classification, unsupported-case closure, workflow-scale
  sufficiency, and publication packaging remain clearly assigned to later Task 4
  stories.

## Implementation Notes

- `MANDATORY_MICROCASES` in
  `benchmarks/density_matrix/circuits.py` already matches the shape of the
  Story 2 contract closely. Implementation should harden and reuse it rather
  than invent another overlapping mandatory matrix.
- `benchmarks/density_matrix/validate_squander_vs_qiskit.py` already contains
  the right exactness kernel: Qiskit Aer comparison, `density_energy()`,
  density-validity checks, trace deviation, and observable-imaginary-component
  metrics. Story 2 should extend or reuse this path instead of duplicating it.
- `tests/density_matrix/test_density_matrix.py` already provides footholds for
  Story 2 through `test_story2_density_energy_helper()` and
  `test_story2_representative_microcase_passes()`. Those are the natural
  starting points for fast regression coverage.
- Story 1 already established canonical required local-noise naming and
  positive-path evidence in
  `benchmarks/density_matrix/noise_support/required_local_noise_validation.py`.
  Story 2 should stay aligned with that vocabulary and avoid introducing
  alternate labels for the same required models.
- `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` already consumes the micro-validation
  bundle inside the broader workflow package. Story 2 should preserve bundle
  compatibility where possible so later Task 4 stories can reuse the same local
  exactness evidence directly.
