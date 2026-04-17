# Task 2 Stories

This document decomposes Phase 2 Task 2 into Layer 3 behavioral stories.
These stories inherit the frozen contract from `TASK_2_MINI_SPEC.md`,
`DETAILED_PLANNING_PHASE_2.md`, `P2-ADR-010`, `P2-ADR-011`, `P2-ADR-012`,
`P2-ADR-013`, `P2-ADR-014`, and `P2-ADR-015`. They describe behavioral slices,
not implementation chores.

Story ordering is intentional:

1. make the supported exact noisy energy path real on the existing VQE
   interface,
2. prove local exactness and physical validity on micro-validation cases,
3. lock unsupported observable boundaries before broader claims are made,
4. scale validation to the mandatory 4 to 10 qubit anchor workload,
5. capture training-relevant and publication-ready evidence in reproducible
   artifacts.

## Story 1: Supported Anchor VQE Calls Return Exact Real `Re Tr(H*rho)` Energy

**User/Research value**
- This story turns the density backend from a routing concept into a
  scientifically useful observable path on the existing VQE interface.

**Given / When / Then**
- Given explicit `density_matrix` selection, a generated `HEA` circuit,
  supported local noise, and a Hermitian sparse Hamiltonian in the XXZ anchor
  family.
- When a user evaluates a supported fixed parameter vector through the
  VQE-facing energy call.
- Then the workflow returns a finite real energy computed from mixed-state
  evolution as `Re Tr(H*rho)` rather than from the legacy state-vector path.

**Scope**
- In: the positive fixed-parameter path for supported XXZ plus `HEA` plus local
  noise anchor cases, using the existing Hermitian sparse-Hamiltonian input
  surface.
- Out: the full Aer threshold matrix, unsupported-case hard-error coverage, the
  optimization-trace package, and the full reproducibility bundle.

**Acceptance signals**
- Supported 4- and 6-qubit anchor smoke cases run through the density path and
  return finite real energies.
- The exercised runtime path is rooted in the density-backed VQE evaluator
  rather than the legacy state-vector evaluator.
- For a deterministic nontrivial noise fixture, the density-backed energy is
  observably different from the noise-free state-vector baseline on the same
  parameter vector.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Task 2; Section 12
  full-phase acceptance criteria; `TASK_2_MINI_SPEC.md` required behavior.
- ADR decision(s): `P2-ADR-010`, `P2-ADR-011`, `P2-ADR-013`.

## Story 2: Micro-Validation Circuits Confirm Local Exactness And Density Validity

**User/Research value**
- This story catches local algebra, ordering, and noise-application regressions
  before the project depends on larger workflow benchmarks.

**Given / When / Then**
- Given supported 1 to 3 qubit micro-validation circuits that cover the required
  `U3` and `CNOT` gate contract and each required local noise model
  individually and in mixed sequences.
- When the exact noisy energy is compared against Qiskit Aer
  density-matrix simulation.
- Then the observable error stays within the micro-validation threshold, the
  density state remains physically valid, and the imaginary component remains
  within the frozen tolerance.

**Scope**
- In: the required 1 to 3 qubit micro-validation matrix, external Aer
  comparison, density-validity checks, and Hermitian-observable consistency
  checks.
- Out: 4 to 10 qubit workflow sweeps, optimization traces, and optional gate
  families beyond the required contract.

**Acceptance signals**
- Required 1 to 3 qubit microcases pass with maximum absolute energy error
  `<= 1e-10` versus Qiskit Aer.
- Recorded outputs satisfy `rho.is_valid(tol=1e-10)` and
  `|Tr(rho) - 1| <= 1e-10`.
- Recorded outputs satisfy `|Im Tr(H*rho)| <= 1e-10` on the exact observable
  path.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Section 10.1 numeric
  thresholds; Section 13 validation and benchmark matrix; `TASK_2_MINI_SPEC.md`
  acceptance evidence.
- ADR decision(s): `P2-ADR-010`, `P2-ADR-012`, `P2-ADR-014`, `P2-ADR-015`.

## Story 3: Out-Of-Scope Observable Requests Fail Explicitly Instead Of Degrading Silently

**User/Research value**
- This story protects scientific trust by making unsupported observable-surface
  requests visible to the caller instead of silently producing misleading
  results.

**Given / When / Then**
- Given a request that falls outside the frozen Task 2 support surface, such as
  an unsupported observable family, unsupported circuit lowering, unsupported
  gate family, or unsupported noise configuration.
- When the user requests exact noisy energy evaluation through the
  `density_matrix` backend.
- Then the workflow fails explicitly, identifies the first unsupported
  condition, and does not fall back to `state_vector`, shot-noise, or partial
  best-effort evaluation.

**Scope**
- In: explicit hard-error behavior for observable, bridge, gate, and noise
  boundary violations inside the Task 2 surface.
- Out: broad generic observable design, future phase extensions, and optional
  support-matrix expansion beyond the frozen Phase 2 scope.

**Acceptance signals**
- Negative tests cover at least one unsupported request in each major boundary
  class owned by Task 2.
- Unsupported requests do not auto-switch to another backend or evaluation
  model.
- Error handling makes the first unsupported condition visible in test output or
  validation artifacts.

**Traceability**
- Phase requirement(s): `TASK_2_MINI_SPEC.md` unsupported behavior;
  `DETAILED_PLANNING_PHASE_2.md` Section 12 full-phase acceptance criteria.
- ADR decision(s): `P2-ADR-010`, `P2-ADR-011`, `P2-ADR-012`, `P2-ADR-015`.

## Story 4: Workflow-Scale Anchor Sweeps Meet Exactness Thresholds Across The Mandatory Exact Regime

**User/Research value**
- This story proves that the exact noisy observable path is not just locally
  correct, but dependable across the mandatory Phase 2 anchor workload sizes.

**Given / When / Then**
- Given supported XXZ anchor cases at 4, 6, 8, and 10 qubits with at least 10
  fixed parameter vectors per required workflow size.
- When those cases are evaluated through the exact noisy density-matrix path and
  compared against Qiskit Aer.
- Then the workflow-level observable error remains within the frozen threshold,
  the cases complete without unsupported-operation workarounds, and the results
  are attributable to the density backend.

**Scope**
- In: the mandatory 4, 6, 8, and 10 qubit parameter-sweep package, workflow
  completion checks, and workflow-level exactness thresholds.
- Out: runtime pass or fail thresholds, optional simulator bake-offs, and
  broader algorithm families beyond the anchor VQE workload.

**Acceptance signals**
- At least 10 fixed parameter vectors per mandatory workflow size are evaluated
  through the density backend.
- Maximum absolute energy error is `<= 1e-8` versus Qiskit Aer on the mandatory
  4, 6, 8, and 10 qubit workflow cases.
- All mandatory workflow cases complete without unsupported-operation
  workarounds and remain attributable to the selected backend.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Section 10.1 numeric
  thresholds; Section 12 full-phase acceptance criteria; Section 13 workload
  classes.
- ADR decision(s): `P2-ADR-013`, `P2-ADR-014`, `P2-ADR-015`.

## Story 5: Optimization Traces And Artifacts Make The Exact Observable Path Reproducible And Publishable

**User/Research value**
- This story supplies the training-relevant and publication-ready evidence that
  lets reviewers audit exact noisy observable claims rather than trust informal
  anecdotes.

**Given / When / Then**
- Given supported fixed-parameter benchmark cases and at least one supported 4-
  or 6-qubit anchor optimization run.
- When validation and reproducibility artifacts are recorded for the exact noisy
  observable path.
- Then one reproducible optimization trace demonstrates training-loop use, and
  the artifact bundle preserves the metadata and raw results needed to audit
  every Task 2 claim.

**Scope**
- In: one reproducible 4- or 6-qubit optimization trace, benchmark metadata,
  artifact completeness, and publication-ready evidence packaging for Task 2.
- Out: broad optimizer comparisons, acceleration claims, and paper-writing work
  beyond the evidence package itself.

**Acceptance signals**
- One reproducible 4- or 6-qubit optimization trace completes end-to-end
  through the exact noisy energy path.
- Stored artifacts let a reviewer identify the backend, Hamiltonian, ansatz,
  noise schedule, parameter vectors or seeds, versions or commit, and raw
  observable results for each Task 2 case.
- Task 2 evidence can be cited directly in the Phase 2 abstract, short paper,
  and full paper without reinterpreting hidden assumptions.

**Traceability**
- Phase requirement(s): `TASK_2_MINI_SPEC.md` acceptance evidence;
  `DETAILED_PLANNING_PHASE_2.md` Section 12 full-phase acceptance criteria;
  Section 13 benchmark package.
- ADR decision(s): `P2-ADR-013`, `P2-ADR-014`, `P2-ADR-015`.
