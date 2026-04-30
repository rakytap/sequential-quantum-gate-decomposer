# Task 6: Noisy Workflow Demonstration Goal

This mini-spec turns Phase 2 Task 6 into an implementation-ready contract. It
inherits the frozen phase decisions from `DETAILED_PLANNING_PHASE_2.md`,
`P2-ADR-007`, `P2-ADR-009`, `P2-ADR-010`, `P2-ADR-011`, `P2-ADR-012`,
`P2-ADR-013`, `P2-ADR-014`, and `P2-ADR-015`, plus the closed
workflow-anchor, benchmark-minimum, and numeric-threshold items in
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. It does not reopen workflow
family, backend-mode semantics, bridge scope, support-matrix scope, benchmark
minimum, or numeric-threshold decisions.

## Given / When / Then
- Given explicit `density_matrix` backend selection, the frozen XXZ + `HEA`
  anchor workflow, and requests inside the documented bridge, support-matrix,
  noise, and observable contract.
- When the canonical Phase 2 workflow demonstration executes with reproducible
  inputs and emits the mandatory end-to-end and benchmark artifacts.
- Then at least one publishable noisy workflow is documented and demonstrably
  supported with explicit input/output behavior, stable pass/fail
  interpretation, and deterministic unsupported-case boundaries.

## Assumptions and dependencies
- Task 1 provides explicit backend selection and no-fallback behavior for
  `density_matrix`.
- Task 2 provides the exact noisy observable path `Re Tr(H*rho)` and the
  frozen exactness thresholds used to interpret workflow outcomes.
- Task 3 provides the HEA-first bridge into `NoisyCircuit` with explicit,
  ordered, auditable noise insertion.
- Task 4 provides the required local-noise baseline and the required /
  optional / deferred support split.
- Task 5 provides the minimum validation baseline and reproducibility
  requirements needed for scientifically defensible workflow claims.
- The mandatory external reference remains Qiskit Aer density-matrix
  simulation; optional secondary baselines may supplement but do not replace
  the Aer-centered interpretation.
- This task defines the workflow-level demonstration contract and evidence
  boundary for Phase 2; it does not introduce new algorithm families, broad
  generic APIs, acceleration claims, or Phase 3 / Phase 4 goals.

## Required behavior
- Task 6 freezes one canonical representative noisy workflow for Phase 2:
  noisy VQE ground-state estimation of a 1D XXZ spin chain with local `Z`
  field using the default `HEA` ansatz and explicit `density_matrix`
  selection.
- The canonical workflow must be specified as a complete behavioral contract,
  not as informal benchmark notes.
- The workflow definition must include explicit input contract fields:
  Hamiltonian family and parameters, qubit count, ansatz/depth, backend mode,
  noise schedule, execution mode (fixed-parameter sweep or optimization trace),
  seed policy, and tolerances or stopping settings where applicable.
- The workflow definition must include explicit output contract fields:
  real-valued energy result, workflow completion status, density-validity
  checks, runtime, peak memory, case identifiers, and unsupported-case
  diagnostics when relevant.
- The required positive path uses the frozen Phase 2 support surface:
  `U3` / `CNOT` bridge compatibility, required local-noise classes, and exact
  Hermitian-energy evaluation `Re Tr(H*rho)`.
- End-to-end workflow execution is mandatory at 4 and 6 qubits, including at
  least one reproducible optimization trace at one of those sizes.
- Benchmark-ready fixed-parameter evaluation is mandatory at 8 and 10 qubits
  inside the same canonical workflow contract.
- The mandatory workflow matrix includes at least 10 fixed parameter vectors
  per required size for 4, 6, 8, and 10 qubits.
- A documented 10-qubit anchor evaluation case is mandatory and is treated as
  the exact-regime acceptance anchor.
- Mandatory workflow outcomes must use the frozen pass/fail interpretation from
  the Phase 2 threshold contract, including explicit case status rather than
  narrative-only claims.
- Demonstration artifacts must preserve stable workflow and case identifiers so
  reruns and publication references remain auditable.
- Supported, optional, deferred, and unsupported behavior must remain
  explicitly distinguished in workflow artifacts and reporting.
- Unsupported requests must fail before execution with deterministic errors and
  must not silently reroute to another backend, observable path, or support
  surface.
- Task 6 completion means one workflow is fully specified and evidenced
  end-to-end; it does not require broad multi-workflow parity.

## Unsupported behavior
- Treating Task 6 as complete with only disconnected micro-validation results
  and no explicit end-to-end workflow contract.
- Claiming workflow support from favorable ad hoc runs without a stable
  workflow identifier and explicit input/output contract.
- Counting 4- or 6-qubit optimization evidence while omitting the required
  8- and 10-qubit fixed-parameter anchor evaluations.
- Claiming completion from a `state_vector` path or any run that silently falls
  back from `density_matrix`.
- Using unsupported gates, unsupported noise models, unsupported bridge inputs,
  or unsupported observables as positive Task 6 evidence.
- Collapsing required local-noise requests into whole-register depolarizing or
  other optional baselines while still reporting canonical workflow success.
- Treating undocumented workflow retries, partial reruns, or hand-selected
  favorable subsets as sufficient publication evidence.
- Using Task 6 to imply broad readiness for additional noisy algorithm
  families, broad Hamiltonian classes, or acceleration-scale claims.

## Acceptance evidence
- A canonical workflow specification artifact defines the Task 6 workflow ID,
  required input contract, required output contract, and supported/unsupported
  boundaries.
- Positive workflow artifacts show explicit `density_matrix` selection through
  `qgd_Variational_Quantum_Eigensolver_Base` and the documented bridge/noise
  path for required cases.
- At least one reproducible 4- or 6-qubit optimization trace completes
  end-to-end under the canonical workflow contract.
- Fixed-parameter workflow artifacts cover 4, 6, 8, and 10 qubits with at
  least 10 parameter vectors per mandatory size.
- Mandatory workflow cases satisfy the frozen workflow exactness threshold
  (`<= 1e-8` maximum absolute energy error versus Qiskit Aer) and preserve the
  frozen 100% pass-rate interpretation for required workflow cases.
- At least one documented 10-qubit anchor evaluation case is present in the
  Task 6 evidence package.
- Recorded workflow artifacts include completion status, runtime, and peak
  memory alongside correctness and validity metadata.
- Negative evidence shows unsupported workflow variations fail explicitly and
  are excluded from positive completion claims.
- Reproducibility artifacts record Hamiltonian definition, ansatz, backend,
  bridge route or equivalent lowering metadata, noise schedule, parameter
  vectors or seeds, software version or commit, raw outputs, and case-level
  status interpretation.
- A machine-readable manifest or equivalent completeness checker confirms the
  required workflow evidence set is present and auditable.
- Traceability target: satisfy the Phase 2 Task 6 evidence requirements in
  `DETAILED_PLANNING_PHASE_2.md`.
- Traceability target: satisfy the full-phase acceptance criterion requiring at
  least one end-to-end noisy workflow supported at 4 to 6 qubits with 8 and 10
  qubit evaluation cases recorded.
- Traceability target: satisfy workflow-anchor and evidence-threshold decisions
  frozen in `P2-ADR-007`, `P2-ADR-013`, `P2-ADR-014`, and `P2-ADR-015`.

## Affected interfaces
- Workflow-definition and configuration surfaces on
  `qgd_Variational_Quantum_Eigensolver_Base` that must expose backend identity,
  Hamiltonian inputs, ansatz settings, and noise-path configuration.
- Bridge and noise-insertion boundaries that map canonical workflow requests
  into the supported `NoisyCircuit` execution path.
- Observable-evaluation interfaces that return exact real-valued energy and
  enforce deterministic unsupported-case behavior.
- Validation, benchmark, and reproducibility harnesses that execute the
  canonical workflow matrix and produce auditable artifacts.
- Artifact schemas and manifests that encode workflow IDs, case IDs, status
  fields, and completeness checks.
- Documentation and publication-facing reporting surfaces that describe the
  canonical workflow boundary without overstating deferred scope.
- Change classification: additive for codifying and automating the canonical
  workflow contract, but stricter for ambiguous or unsupported workflow
  requests, which become explicit hard failures.

## Publication relevance
- Supports Paper 1's workflow-level claim that the density-matrix backend is
  usable in a complete noisy VQE research workflow, not only in isolated
  backend checks.
- Provides the concrete workflow contract and reproducibility evidence needed
  for defensible abstract, short-paper, and full-paper statements.
- Keeps publication claims honest by anchoring them to the documented exact
  regime and a single fully specified workflow boundary.
- Supplies a reusable canonical workflow evidence package that later phases can
  extend when adding acceleration, broader workflows, or richer optimization
  studies.
