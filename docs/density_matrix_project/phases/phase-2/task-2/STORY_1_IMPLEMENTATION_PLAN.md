# Story 1 Implementation Plan

## Story Being Implemented

Story 1: Supported Anchor VQE Calls Return Exact Real `Re Tr(H*rho)` Energy

This is a Layer 4 engineering plan for implementing the first behavioral slice
from `TASK_2_STORIES.md`.

## Scope

This story stabilizes the first positive exact-observable slice on the existing
VQE surface:

- explicit `density_matrix` selection on supported anchor workloads reaches the
  density-backed observable path,
- the supported Hermitian sparse-Hamiltonian interface remains the
  user-visible observable input surface,
- fixed-parameter anchor evaluations return finite real energy from mixed-state
  evolution,
- and the slice stays narrow enough that later stories can add threshold
  closure, unsupported-case coverage, and publication-grade artifact packaging.

Out of scope for this story:

- the full Aer threshold matrix at 1 to 10 qubits,
- the unsupported-case hard-error matrix owned by Story 3,
- the optimization-trace and full reproducibility package owned by Story 5,
- gradients, shot-noise estimators, and broader observable families,
- and runtime or scaling claims beyond proving the supported positive path.

## Dependencies And Assumptions

- Task 1 already provides explicit backend selection and the no-fallback
  contract for `density_matrix`.
- The current VQE core already contains the partial density-backend substrate in
  `validate_density_anchor_support()`,
  `lower_anchor_circuit_to_noisy_circuit()`,
  `evaluate_density_matrix_backend()`, and
  `expectation_value_of_density_energy_real()`.
- The current positive anchor fixtures in `tests/VQE/test_VQE.py` and
  `benchmarks/density_matrix/story2_vqe_density_validation.py` are the correct
  starting point for Story 1 evidence.
- Story 1 should harden and expose the supported positive observable slice; it
  should not reopen the phase-level observable scope, bridge scope, or threshold
  decisions.

## Engineering Tasks

### Engineering Task 1: Stabilize Density-Backed Runtime Dispatch For Fixed-Parameter Energy Calls

**Implements story**
- `Story 1: Supported Anchor VQE Calls Return Exact Real Re Tr(H*rho) Energy`

**Change type**
- code | tests

**Definition of done**
- A supported fixed-parameter VQE energy call with explicit `density_matrix`
  selection reaches the density-backed observable path through the standard
  VQE-facing entry point.
- The positive path is demonstrably distinct from the legacy state-vector path
  on a deterministic noisy anchor fixture.
- The same sparse-Hamiltonian input surface remains the user-visible observable
  contract.

**Execution checklist**
- [ ] Review the current `Optimization_Problem()` dispatch path and freeze the
      supported density-backed branch for the Task 2 Story 1 slice.
- [ ] Keep the density-backed path behind dedicated helper boundaries so later
      stories can add threshold and unsupported-case logic without reworking
      the runtime entry point.
- [ ] Add or tighten focused tests that prove supported fixed-parameter calls
      actually execute the density-backed branch.

**Evidence produced**
- Focused pytest coverage proving the supported fixed-parameter call reaches the
  density-backed branch.
- Reviewable runtime path rooted in the VQE core rather than in a
  standalone-only observable helper.

**Risks / rollback**
- Risk: dispatch changes can accidentally perturb the legacy state-vector path.
- Rollback/mitigation: keep the branch narrow, default all legacy callers to the
  existing state-vector path, and validate omitted-backend behavior separately.

### Engineering Task 2: Harden The Density-Energy Helper Around Exact `Re Tr(H*rho)`

**Implements story**
- `Story 1: Supported Anchor VQE Calls Return Exact Real Re Tr(H*rho) Energy`

**Change type**
- code | tests

**Definition of done**
- The density-backed helper computes the exact real Hermitian energy from the
  evolved density state using the existing sparse Hamiltonian.
- Dimension mismatches or obvious contract violations fail clearly rather than
  producing misleading values.
- The observable path exposes enough reviewable behavior that later stories can
  validate Hermitian-imaginary consistency without replacing this helper.

**Execution checklist**
- [ ] Review `expectation_value_of_density_energy_real()` against the frozen
      `Re Tr(H*rho)` contract and tighten any missing guardrails.
- [ ] Keep the helper anchored to the existing sparse-Hamiltonian surface rather
      than inventing a second observable API.
- [ ] Add or refine unit-level tests that cover dimension checks and deterministic
      exact-energy behavior on small supported fixtures.

**Evidence produced**
- Reviewable exact-energy helper implementation tied to the VQE Hamiltonian
  surface.
- Focused tests covering correct-value and dimension-mismatch behavior.

**Risks / rollback**
- Risk: small indexing or conjugation mistakes can produce believable but wrong
  energies.
- Rollback/mitigation: validate the helper first on tiny deterministic fixtures
  before depending on larger anchor cases.

### Engineering Task 3: Keep The Supported `HEA` Plus Local-Noise Slice Aligned With Observable Evaluation

**Implements story**
- `Story 1: Supported Anchor VQE Calls Return Exact Real Re Tr(H*rho) Energy`

**Change type**
- code

**Definition of done**
- The supported `HEA`-generated gate stream, parameter ordering, and ordered
  local-noise insertions remain aligned with the density-backed observable path.
- The Story 1 positive slice uses only the frozen required gate and noise
  surface.
- The observable result is produced from the same mixed-state evolution that the
  supported anchor workflow actually executes.

**Execution checklist**
- [ ] Review `lower_anchor_circuit_to_noisy_circuit()` and
      `append_density_noise_for_gate_index()` for parameter-order and
      gate-order stability on the required `U3` and `CNOT` path.
- [ ] Keep Story 1 pinned to the supported local-noise contract: local
      depolarizing, amplitude damping, and phase damping / dephasing.
- [ ] Avoid broadening circuit or noise support inside Story 1; keep unsupported
      expansion for later stories.

**Evidence produced**
- Reviewable bridge and noise-ordering path used by the supported observable
  slice.
- Deterministic anchor fixture configuration suitable for reuse in later stories.

**Risks / rollback**
- Risk: observable logic can look correct while the bridged circuit or noise
  ordering is subtly wrong.
- Rollback/mitigation: use a small deterministic anchor fixture and inspect the
  lowered path before extending validation breadth.

### Engineering Task 4: Add Positive Fixed-Parameter Anchor Tests On The VQE Surface

**Implements story**
- `Story 1: Supported Anchor VQE Calls Return Exact Real Re Tr(H*rho) Energy`

**Change type**
- tests

**Definition of done**
- Supported 4- and 6-qubit anchor cases show that explicit `density_matrix`
  selection returns finite real energies through the VQE-facing API.
- The test evidence is strong enough to show the path depends on mixed-state
  evolution rather than accidentally reusing the legacy state-vector evaluator.
- Story 1 test coverage remains small and fast enough to serve as regression
  coverage.

**Execution checklist**
- [ ] Extend or harden `tests/VQE/test_VQE.py` for deterministic 4- and 6-qubit
      fixed-parameter anchor cases.
- [ ] Reuse a nontrivial local-noise fixture so the density-backed result is
      observably different from the state-vector baseline on the same parameter
      vector.
- [ ] Keep full Aer threshold checks out of the fast regression layer; that work
      belongs to later stories and benchmark harnesses.

**Evidence produced**
- Focused regression tests for supported fixed-parameter anchor evaluation.
- Reviewable positive-path evidence that the density-backed observable path is
  real and distinct.

**Risks / rollback**
- Risk: overly weak fixtures can let the density and state-vector results look
  accidentally similar and weaken the story signal.
- Rollback/mitigation: use deterministic nontrivial noise settings and assert
  only the behavioral differences needed for Story 1.

### Engineering Task 5: Extend The Validation Harness To Emit Story 1 Observable Evidence

**Implements story**
- `Story 1: Supported Anchor VQE Calls Return Exact Real Re Tr(H*rho) Energy`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- The validation harness can emit stable fixed-parameter artifacts for the Story
  1 positive slice.
- Artifact fields already needed by later Task 2 stories are captured without
  forcing Story 1 to close the full threshold package.
- Story 1 produces at least one stable command or artifact that reviewers can
  rerun to verify the supported positive path.

**Execution checklist**
- [ ] Extend `benchmarks/density_matrix/story2_vqe_density_validation.py` or an
      equivalent successor so Story 1 fixed-parameter cases emit stable metadata.
- [ ] Record backend label, Hamiltonian metadata, noise specification, parameter
      vector, and raw energy result for the supported positive cases.
- [ ] Keep the artifact schema stable enough that later stories can add Aer
      error, density-validity, and optimization-trace fields without replacing
      the Story 1 output format.

**Evidence produced**
- One stable fixed-parameter artifact or rerunnable command for Story 1.
- A reviewable artifact schema that can be extended by later Task 2 stories.

**Risks / rollback**
- Risk: ad hoc artifact formats will drift and make later exactness evidence
  hard to audit.
- Rollback/mitigation: adopt one stable metadata shape early and extend it
  incrementally.

### Engineering Task 6: Update Developer-Facing Notes For The Exact Observable Entry Point

**Implements story**
- `Story 1: Supported Anchor VQE Calls Return Exact Real Re Tr(H*rho) Energy`

**Change type**
- docs

**Definition of done**
- Developer-facing notes describe the supported exact noisy energy entry point on
  the VQE surface.
- The notes make clear that the observable contract is the existing Hermitian
  sparse-Hamiltonian path plus explicit `density_matrix` selection on the
  supported anchor workflow.
- The notes do not overclaim threshold closure, broader observables, or later
  story responsibilities.

**Execution checklist**
- [ ] Update the most relevant developer-facing docstrings, examples, or notes
      near the VQE and density-backend entry points.
- [ ] Keep wording aligned with the frozen Task 2 scope: exact Hermitian energy,
      supported `HEA` anchor slice, and required local noise only.
- [ ] Refer readers to later validation artifacts for full threshold and
      publication-grade evidence.

**Evidence produced**
- Updated developer-facing notes for the exact noisy observable entry point.
- One stable place where the Story 1 support boundary is documented.

**Risks / rollback**
- Risk: documentation can overtake the actual supported slice and create false
  expectations.
- Rollback/mitigation: keep examples tied to the same anchor fixtures used by
  Story 1 tests and validation commands.

### Engineering Task 7: Run Story 1 Validation

**Implements story**
- `Story 1: Supported Anchor VQE Calls Return Exact Real Re Tr(H*rho) Energy`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 1 regression tests pass on the supported positive slice.
- At least one fixed-parameter validation command or artifact can be reproduced
  outside the unit-test layer.
- Story 1 exits with reviewable evidence rather than with code changes alone.

**Execution checklist**
- [ ] Run the focused positive fixed-parameter VQE tests for the density-backed
      anchor slice.
- [ ] Run the Story 1 validation harness command that emits the fixed-parameter
      observable artifact.
- [ ] Confirm that Story 1 evidence is archived or named in a stable way for
      later Task 2 stories to reuse.

**Evidence produced**
- Passing focused pytest coverage for Story 1.
- One rerunnable validation artifact or command reference for the supported
  positive path.

**Risks / rollback**
- Risk: Story 1 can appear complete without any stored evidence that reviewers
  can inspect later.
- Rollback/mitigation: treat the emitted artifact or rerunnable command as part
  of the exit gate, not as optional follow-up.

## Exit Criteria

Story 1 is complete only when all of the following are true:

- supported fixed-parameter 4- and 6-qubit anchor cases return finite real
  energies through the explicit `density_matrix` backend path,
- the observable contract remains the existing Hermitian sparse-Hamiltonian
  input surface,
- the supported positive path is visibly density-backed and behaviorally distinct
  from the legacy state-vector baseline on the chosen nontrivial noise fixture,
- at least one stable validation artifact or command proves the observable path
  outside the unit-test layer,
- and full threshold closure, unsupported-case coverage, and optimization-trace
  packaging remain clearly assigned to later stories.

## Implementation Notes

- The current `evaluate_density_matrix_backend()` and
  `expectation_value_of_density_energy_real()` helpers already provide the
  natural backbone for Story 1. Implementation should harden and expose this
  path rather than introduce a second observable layer.
- `validate_density_anchor_support()` should continue to gate Story 1 to the
  supported `HEA`-generated circuit source and required gate/noise surface so
  the story does not silently expand into broader observable support.
- The existing smoke cases in `tests/VQE/test_VQE.py` and the fixed-parameter
  validation flow in `benchmarks/density_matrix/story2_vqe_density_validation.py`
  are the right starting points for deterministic Story 1 evidence.
