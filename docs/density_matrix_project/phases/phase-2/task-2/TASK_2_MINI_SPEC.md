# Task 2: Exact Noisy Expectation-Value Path

This mini-spec turns Phase 2 Task 2 into an implementation-ready contract. It
inherits the frozen phase decisions from `DETAILED_PLANNING_PHASE_2.md`,
`P2-ADR-010`, `P2-ADR-011`, `P2-ADR-012`, `P2-ADR-013`, `P2-ADR-014`, and
`P2-ADR-015`, plus the closed observable-contract and numeric-threshold items
in `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. It does not reopen observable
scope, bridge scope, workload anchor, benchmark minimum, or numeric-threshold
decisions.

## Given / When / Then
- Given explicit `density_matrix` backend selection, a supported anchor noisy
  VQE case, and a Hermitian sparse Hamiltonian within the frozen Phase 2
  support surface.
- When the workflow evaluates the cost function on the noisy density state
  produced by the supported bridge and noise path.
- Then the workflow returns the exact real-valued energy `Re Tr(H*rho)`,
  preserves density validity within the frozen tolerances, and agrees with
  Qiskit Aer within the frozen thresholds or fails explicitly if the request is
  unsupported.

## Assumptions and dependencies
- Task 1 provides explicit backend selection and no-fallback behavior for
  `density_matrix`.
- The guaranteed circuit bridge is the default `HEA` circuit from
  `qgd_Variational_Quantum_Eigensolver_Base.Generate_Circuit()` lowered into
  `NoisyCircuit`; broader `qgd_Circuit` parity is not assumed.
- The required noise surface is the Phase 2 local-noise support matrix: local
  single-qubit depolarizing, local amplitude damping, and local phase damping /
  dephasing. Whole-register depolarizing may appear only as a regression or
  stress baseline and does not widen the core Task 2 contract.
- Acceptance evidence uses the Aer-centered benchmark package and the numeric
  exactness thresholds already frozen at the phase level.
- This task defines the observable contract needed for noisy variational
  training; it does not define gradients, sampling-based estimators, or
  acceleration behavior.

## Required behavior
- The Task 2 observable contract is exact real-valued Hermitian energy
  evaluation `E(theta) = Re Tr(H*rho(theta))`.
- The required user-visible input path is the existing Hermitian
  sparse-Hamiltonian interface used by
  `qgd_Variational_Quantum_Eigensolver_Base`.
- Required positive-path validation is anchored on XXZ Hamiltonians with
  optional local `Z` field, evaluated inside the Phase 2 noisy VQE workflow
  with the default `HEA` ansatz and explicit `density_matrix` selection.
- For supported requests, energy evaluation must operate on the density state
  produced by the supported bridge and noise path rather than by a separate
  standalone-only execution surface.
- The returned Phase 2 observable is the real energy. Any imaginary component
  larger than the frozen tolerance is a validation failure, not an acceptable
  scientific result.
- For required micro-validation cases at 1 to 3 qubits, maximum absolute energy
  error versus Qiskit Aer must be `<= 1e-10`.
- For required workflow parameter-sweep cases at 4, 6, 8, and 10 qubits,
  maximum absolute energy error versus Qiskit Aer must be `<= 1e-8`.
- Recorded validation outputs for supported cases must satisfy
  `rho.is_valid(tol=1e-10)` and `|Tr(rho) - 1| <= 1e-10`.
- The exact observable path must satisfy `|Im Tr(H*rho)| <= 1e-10` on recorded
  validation outputs.
- The observable path must be usable by at least one reproducible 4- or 6-qubit
  anchor optimization trace so the exact noisy energy contract is demonstrated
  inside a training-relevant loop.
- When the requested observable, Hamiltonian family, circuit lowering, gate
  family, or noise model falls outside the frozen Phase 2 support matrix, the
  request must fail explicitly rather than degrade into approximation,
  sampling, or silent fallback.

## Unsupported behavior
- Broad generic observable APIs beyond exact Hermitian energy evaluation.
- Arbitrary non-Hermitian observables, general POVMs, or batched
  multi-observable APIs as part of the Task 2 minimum.
- Sampled, shot-noise, or readout-noise-based estimation as the primary
  acceptance path for Task 2.
- Claiming arbitrary Hamiltonian families beyond the XXZ-anchored acceptance
  workload as required Phase 2 evidence.
- Silent fallback to `state_vector`, silent omission of unsupported terms, or
  partial best-effort evaluation when the request is outside the documented
  support surface.
- Treating a large imaginary component in `Tr(H*rho)` as acceptable by simply
  discarding it without meeting the documented tolerance.
- Using Task 2 to imply gradient support, optimizer support, or general
  measurement infrastructure beyond what the phase contract already freezes.

## Acceptance evidence
- Micro-validation comparisons against Qiskit Aer cover 1 to 3 qubit circuits
  that exercise each required gate family and each required noise model
  individually and in mixed sequences, with maximum absolute energy error
  `<= 1e-10`.
- Workflow-level comparisons against Qiskit Aer cover anchor `HEA` noisy VQE XXZ
  cases at 4, 6, 8, and 10 qubits with at least 10 fixed parameter vectors per
  mandatory workflow size, with maximum absolute energy error `<= 1e-8`.
- Recorded validation artifacts show `rho.is_valid(tol=1e-10)`,
  `|Tr(rho) - 1| <= 1e-10`, and `|Im Tr(H*rho)| <= 1e-10` for the exact
  observable path.
- At least one reproducible 4- or 6-qubit optimization trace completes
  end-to-end using the Task 2 exact noisy energy path inside the anchor
  workflow.
- Negative tests or validation cases show that out-of-scope observables,
  unsupported lowering paths, or unsupported gate/noise combinations fail
  explicitly and do not silently switch to another evaluation model.
- The reproducibility bundle for Task 2 evidence records the Hamiltonian,
  ansatz, backend, noise schedule, parameter vectors or seeds, versions or
  commit, and raw observable results.
- Traceability target: satisfy the Phase 2 Task 2 evidence requirements in
  `DETAILED_PLANNING_PHASE_2.md`.
- Traceability target: satisfy the full-phase acceptance criterion requiring
  validated `Re Tr(H*rho)` for the supported Hermitian Hamiltonian class.
- Traceability target: satisfy the Phase 2 benchmark-minimum and
  numeric-threshold decisions frozen in `P2-ADR-014` and `P2-ADR-015`.

## Affected interfaces
- The Hermitian sparse-Hamiltonian input surface on
  `qgd_Variational_Quantum_Eigensolver_Base`.
- The VQE-facing energy-evaluation path that consumes the density state and
  returns the exact real energy value.
- The bridge boundary between supported circuit lowering / noise insertion and
  the observable evaluator.
- Validation and benchmark harnesses that compare SQUANDER results against
  Qiskit Aer and record observable, density-validity, and workflow-completion
  metrics.
- Reproducibility and benchmark metadata that must make the Hamiltonian family,
  backend, noise model, thresholds, and observable results auditable.
- Change classification: additive for the supported density-matrix VQE path, but
  stricter for ambiguous or out-of-scope observable requests, which become
  explicit hard failures rather than undocumented behavior.

## Publication relevance
- Supports Paper 1's central claim that SQUANDER can evaluate exact noisy
  mixed-state energies through `Re Tr(H*rho)` inside the anchor VQE workflow.
- Provides the scientifically defensible observable-evaluation evidence needed
  for the Phase 2 abstract, short paper, and full paper.
- Keeps publication claims bounded to exact Hermitian energy evaluation, which
  aligns the paper narrative with the frozen Phase 2 scope and validation
  thresholds.
- Supplies the observable-agreement results that make noisy VQA comparisons and
  optimization traces publishable rather than anecdotal.
