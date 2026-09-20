# Story 2 Implementation Plan

## Story Being Implemented

Story 2: Micro-Validation Circuits Confirm Local Exactness And Density Validity

This is a Layer 4 engineering plan for implementing the second behavioral slice
from `TASK_2_STORIES.md`.

## Scope

This story turns the frozen 1 to 3 qubit micro-validation contract into a
reproducible evidence layer that catches local correctness regressions before
the project depends on larger workflow benchmarks:

- the required micro-validation matrix covers the Phase 2 required gate family
  and required local noise models,
- each required microcase compares exact noisy energy against Qiskit Aer
  density-matrix simulation,
- recorded outputs include density-validity and Hermitian-observable consistency
  metrics,
- and the results are small and fast enough to serve as the local correctness
  gate below the 4 to 10 qubit workflow package.

Out of scope for this story:

- the 4, 6, 8, and 10 qubit workflow parameter-sweep package owned by Story 4,
- unsupported-case hard-error coverage owned by Story 3,
- the optimization-trace and publication-ready artifact package owned by Story
  5,
- broad gate-family expansion beyond the required `U3` and `CNOT` contract,
- and VQE public-API expansion to return the density matrix directly.

## Dependencies And Assumptions

- Story 1 is already in place: the exact noisy positive path exists for the
  anchor VQE workflow and provides the first workflow-level observable proof.
- The current low-level density-matrix validation scaffold already exists in
  `benchmarks/density_matrix/circuits.py`,
  `benchmarks/density_matrix/validate_squander_vs_qiskit.py`, and
  `tests/density_matrix/test_density_matrix.py`.
- `DualBuilder`, `NoisyCircuit`, and `DensityMatrix` already expose the right
  substrate for microcases that directly inspect density validity and compare
  with Qiskit Aer.
- Story 2 should validate the exact observable contract locally without
  reopening the phase-level observable scope or requiring a new generic
  observable API.
- Phase 2 thresholds are frozen already:
  - maximum absolute energy error `<= 1e-10` on required 1 to 3 qubit
    microcases,
  - `rho.is_valid(tol=1e-10)` and `|Tr(rho) - 1| <= 1e-10`,
  - and `|Im Tr(H*rho)| <= 1e-10` for the exact observable path.

## Engineering Tasks

### Engineering Task 1: Freeze The Mandatory 1 To 3 Qubit Micro-Validation Matrix

**Implements story**
- `Story 2: Micro-Validation Circuits Confirm Local Exactness And Density Validity`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- The Story 2 micro-validation matrix explicitly covers the required Phase 2
  gate and noise contract at 1 to 3 qubits.
- Each microcase has a stable identifier, a declared purpose, and a matching
  Hermitian Hamiltonian used for `Re Tr(H*rho)` comparison.
- The matrix separates mandatory cases from optional exploratory cases so the
  Story 2 pass/fail gate stays unambiguous.

**Execution checklist**
- [ ] Enumerate mandatory 1 to 3 qubit microcases for `U3`, `CNOT`, and the
      required local noise models individually and in mixed sequences.
- [ ] Attach each microcase to a concrete Hermitian Hamiltonian and expected
      validation metrics.
- [ ] Keep optional gate families or stress cases outside the mandatory Story 2
      pass/fail set.
- [ ] Record the matrix in one stable implementation-facing location so later
      benchmark work can reuse the same case IDs.

**Evidence produced**
- A named mandatory micro-validation matrix for the required 1 to 3 qubit cases.
- Stable case identifiers that can be reused in test output and artifacts.

**Risks / rollback**
- Risk: an underspecified matrix can let important gate or noise regressions
  slip through while still claiming Story 2 completion.
- Rollback/mitigation: keep the mandatory matrix small but explicitly traceable
  to the frozen required gate and noise contract only.

### Engineering Task 2: Add Exact Observable Comparison Helpers For Microcases

**Implements story**
- `Story 2: Micro-Validation Circuits Confirm Local Exactness And Density Validity`

**Change type**
- code | benchmark harness

**Definition of done**
- The micro-validation harness can evaluate `Re Tr(H*rho)` for SQUANDER and
  Qiskit Aer on the same 1 to 3 qubit microcase.
- The comparison path also records the imaginary component of `Tr(H*rho)` so the
  Hermitian-observable consistency rule is measurable.
- Story 2 reuses or mirrors the existing exact observable contract rather than
  introducing a second user-facing observable surface.

**Execution checklist**
- [ ] Add a reusable helper that computes exact energy and imaginary-component
      diagnostics from the microcase density matrix and Hermitian Hamiltonian.
- [ ] Reuse the same Hamiltonian representation consistently between SQUANDER and
      Qiskit Aer comparisons.
- [ ] Keep the helper narrow to Story 2 validation needs and avoid turning it
      into a generic measurement framework.
- [ ] Add focused tests for the helper on deterministic tiny fixtures where the
      expected behavior is easy to inspect.

**Evidence produced**
- A reusable micro-validation observable-comparison helper.
- Focused tests proving exact-energy and imaginary-component extraction works on
  tiny deterministic cases.

**Risks / rollback**
- Risk: observable-comparison code can drift from the Phase 2 contract and
  silently validate the wrong quantity.
- Rollback/mitigation: keep the helper explicitly tied to `Re Tr(H*rho)` and
  Hermitian Hamiltonians only.

### Engineering Task 3: Record Density Validity And Hermitian Consistency Metrics

**Implements story**
- `Story 2: Micro-Validation Circuits Confirm Local Exactness And Density Validity`

**Change type**
- code | benchmark harness | validation automation

**Definition of done**
- Every mandatory microcase records whether the resulting density matrix is
  valid at the frozen tolerance.
- Every mandatory microcase records trace-preservation and Hermitian-observable
  consistency metrics needed by the Story 2 gate.
- The micro-validation outputs can distinguish an energy-match success from a
  physically invalid density-state result.

**Execution checklist**
- [ ] Extend the validation output to record `rho.is_valid(tol=1e-10)`.
- [ ] Record `|Tr(rho) - 1|` for each mandatory microcase.
- [ ] Record `|Im Tr(H*rho)|` alongside exact-energy comparison results.
- [ ] Ensure these metrics are available in machine-readable output, not only in
      console text.

**Evidence produced**
- Machine-readable density-validity metrics for mandatory microcases.
- Reviewable Hermitian-consistency and trace-preservation output.

**Risks / rollback**
- Risk: validating only energy error can miss physically invalid density states
  that still yield plausible energies.
- Rollback/mitigation: make density-validity and trace-preservation metrics part
  of the mandatory Story 2 result schema.

### Engineering Task 4: Add Focused Regression Tests For The Mandatory Microcases

**Implements story**
- `Story 2: Micro-Validation Circuits Confirm Local Exactness And Density Validity`

**Change type**
- tests

**Definition of done**
- Focused automated tests cover the Story 2 mandatory micro-validation slice.
- Fast regression coverage catches local gate, noise, and observable regressions
  without requiring the full benchmark package to run on every edit.
- The test layer stays small enough to be practical while still exercising every
  mandatory gate and noise contract.

**Execution checklist**
- [ ] Add focused tests for representative mandatory 1 to 3 qubit microcases.
- [ ] Reuse existing low-level density-matrix tests where possible rather than
      duplicating the same gate or noise checks in multiple places.
- [ ] Keep heavyweight matrix-wide validation in dedicated validation commands or
      benchmark harnesses rather than the default fast suite.
- [ ] Make the failing signal specific enough to tell whether a regression is in
      gate lowering, noise application, observable evaluation, or density-state
      validity.

**Evidence produced**
- Focused pytest coverage for the mandatory Story 2 local-correctness slice.
- Reviewable failures that localize micro-validation regressions.

**Risks / rollback**
- Risk: adding the full matrix to the default test suite can make Story 2
  brittle and slow.
- Rollback/mitigation: keep smoke-level mandatory coverage in pytest and push
  exhaustive matrix execution into a dedicated validation command.

### Engineering Task 5: Emit A Stable Machine-Readable Micro-Validation Artifact Bundle

**Implements story**
- `Story 2: Micro-Validation Circuits Confirm Local Exactness And Density Validity`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Story 2 produces a stable machine-readable output bundle for the mandatory
  micro-validation matrix.
- Each record includes enough metadata to audit case identity, backend,
  Hamiltonian, observable result, exactness error, and density-validity status.
- The artifact shape is stable enough that later stories can extend it without
  replacing the Story 2 evidence format.

**Execution checklist**
- [ ] Define the per-case artifact fields for mandatory Story 2 microcases.
- [ ] Include backend, reference backend, case ID, Hamiltonian metadata, energy
      metrics, validity metrics, and pass/fail status.
- [ ] Keep the artifact naming scheme stable and easy to reference in later
      docs, papers, and benchmark scripts.
- [ ] Avoid bundling later-story workflow or optimization-trace data into the
      Story 2 micro-validation output.

**Evidence produced**
- A stable machine-readable Story 2 micro-validation artifact bundle.
- A reproducible command that regenerates the mandatory Story 2 outputs.

**Risks / rollback**
- Risk: ad hoc or human-only validation output will not be auditable enough for
  later paper claims.
- Rollback/mitigation: define a minimal structured schema now and extend it
  incrementally in later stories.

### Engineering Task 6: Wire Story 2 Validation Into The Existing Benchmark And Test Surfaces

**Implements story**
- `Story 2: Micro-Validation Circuits Confirm Local Exactness And Density Validity`

**Change type**
- tests | benchmark harness | validation automation

**Definition of done**
- Story 2 has one clear rerunnable validation entry point for the mandatory
  micro-validation matrix.
- The validation path is anchored to the current density-matrix comparison
  scaffold rather than a one-off script.
- Story 2 outputs can be consumed later by Story 4 and Story 5 work without
  redesigning the local validation layer.

**Execution checklist**
- [ ] Extend `benchmarks/density_matrix/validate_squander_vs_qiskit.py` or a
      tightly related successor to cover Story 2 observable metrics.
- [ ] Reuse `benchmarks/density_matrix/circuits.py` for shared 1 to 3 qubit case
      construction where possible.
- [ ] Keep the validation entry point explicit and documented so Story 2 can be
      rerun independently of the larger workflow package.
- [ ] Ensure the Story 2 path remains distinct from the workflow-level VQE
      validation in `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py`.

**Evidence produced**
- One stable validation command for the Story 2 micro-validation matrix.
- Reviewable benchmark/test integration that reuses existing density validation
  scaffolding.

**Risks / rollback**
- Risk: scattering Story 2 validation across unrelated scripts makes the local
  exactness gate hard to rerun and maintain.
- Rollback/mitigation: anchor Story 2 on the existing low-level comparison
  scaffold and keep the entry point singular.

### Engineering Task 7: Update Developer-Facing Notes For The Micro-Validation Gate

**Implements story**
- `Story 2: Micro-Validation Circuits Confirm Local Exactness And Density Validity`

**Change type**
- docs

**Definition of done**
- Developer-facing guidance explains what Story 2 validates, how to rerun it,
  and which metrics are mandatory.
- The notes make clear that Story 2 is the local exactness and density-validity
  gate below the workflow-level benchmark package.
- The doc surface stays aligned with the frozen Phase 2 thresholds and does not
  overclaim later-story closure.

**Execution checklist**
- [ ] Document the Story 2 validation entry point and mandatory metrics.
- [ ] Make the required thresholds explicit: `<= 1e-10` energy error,
      density-validity pass, trace-preservation, and `|Im Tr(H*rho)| <= 1e-10`.
- [ ] Explain how Story 2 relates to Story 1 and how it hands off to Story 4 and
      Story 5.
- [ ] Keep unsupported-case coverage and full publication packaging clearly
      assigned to later stories.

**Evidence produced**
- Updated developer-facing instructions for the Story 2 micro-validation gate.
- One stable place where Story 2 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if the micro-validation gate is undocumented, later contributors may
  skip it or misread it as optional exploratory work.
- Rollback/mitigation: tie the notes directly to the same command and artifact
  outputs used in Story 2 validation.

### Engineering Task 8: Run Story 2 Micro-Validation And Confirm The Local Gate

**Implements story**
- `Story 2: Micro-Validation Circuits Confirm Local Exactness And Density Validity`

**Change type**
- tests | validation automation

**Definition of done**
- The mandatory Story 2 micro-validation matrix runs successfully end-to-end.
- Every required 1 to 3 qubit microcase passes the frozen exactness and
  density-validity thresholds.
- Story 2 completion is backed by reviewable outputs rather than by code changes
  alone.

**Execution checklist**
- [ ] Run the focused Story 2 regression tests.
- [ ] Run the dedicated Story 2 validation command that emits the mandatory
      micro-validation artifact bundle.
- [ ] Verify `100%` pass rate on the mandatory micro-validation matrix.
- [ ] Record the stable test run and artifact references for later Task 2 docs
      and paper evidence.

**Evidence produced**
- Passing focused pytest coverage for Story 2.
- A machine-readable Story 2 micro-validation artifact bundle with a `100%`
  pass rate on mandatory cases.

**Risks / rollback**
- Risk: Story 2 can appear complete while still lacking a reproducible proof of
  the local exactness gate.
- Rollback/mitigation: treat the emitted artifact bundle and full pass rate as
  part of the exit criteria, not optional follow-up.

## Exit Criteria

Story 2 is complete only when all of the following are true:

- the mandatory 1 to 3 qubit micro-validation matrix covers the required gate
  and noise contract,
- every mandatory microcase compares `Re Tr(H*rho)` against Qiskit Aer with
  maximum absolute energy error `<= 1e-10`,
- every mandatory microcase records `rho.is_valid(tol=1e-10)`,
  `|Tr(rho) - 1| <= 1e-10`, and `|Im Tr(H*rho)| <= 1e-10`,
- the Story 2 local-correctness gate achieves a `100%` pass rate on the
  mandatory matrix,
- and the results are available through one stable validation command and one
  stable machine-readable artifact bundle.

## Implementation Notes

- `benchmarks/density_matrix/circuits.py` and
  `benchmarks/density_matrix/validate_squander_vs_qiskit.py` already provide the
  natural backbone for Story 2. Implementation should extend that scaffold for
  observable metrics and validity checks rather than create a second low-level
  validation framework.
- `tests/density_matrix/test_density_matrix.py` already exercises many density
  operations and validity checks. Story 2 should reuse that coverage style for
  fast regression rather than duplicate every low-level behavior in a new test
  module.
- Story 2 should stay tightly bounded to the required `U3` / `CNOT` plus local
  depolarizing / amplitude damping / phase damping contract even though the
  low-level density module can express more operations.
- Story 2 should not broaden the VQE public API to expose density matrices.
  Local density-validity metrics can be recorded at the validation-harness layer
  where `DensityMatrix` is already directly accessible.
