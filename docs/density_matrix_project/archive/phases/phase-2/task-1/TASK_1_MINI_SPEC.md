# Task 1: Backend Selection Contract

This mini-spec turns Phase 2 Task 1 into an implementation-ready contract.
It inherits the frozen phase decisions from `DETAILED_PLANNING_PHASE_2.md`,
`P2-ADR-009`, and the closed backend-selection item in
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. It does not reopen backend-mode,
fallback, or workflow-anchor scope.

## Required behavior
- Backend choice is a user-facing workflow-level setting on
  `qgd_Variational_Quantum_Eigensolver_Base` or an equivalent VQE-facing
  configuration surface.
- Phase 2 accepts exactly two backend modes: `state_vector` and
  `density_matrix`.
- If the user does not specify a backend, execution remains on
  `state_vector` so existing VQE workflows preserve current behavior.
- `density_matrix` must be selected explicitly for workflows that claim exact
  noisy mixed-state execution.
- Backend selection is interpreted before execution begins, and the selected
  backend is unambiguous in documentation, tests, and benchmark artifacts.
- When `density_matrix` is selected and the requested circuit source, gate
  family, noise model, or observable falls outside the frozen Phase 2 support
  matrix, the workflow must fail with a hard pre-execution error.
- When `state_vector` is selected but the requested workflow depends on Phase
  2-only mixed-state behavior, the workflow must also fail early rather than
  swapping backends automatically.
- The required positive path for this task is the anchor noisy VQE workflow:
  XXZ Hamiltonian with local `Z` field, default `HEA` ansatz, and backend
  selection performed through the VQE-facing entry point.

## Unsupported behavior
- Implicit `auto`, heuristic, or noise-triggered backend selection.
- Silent fallback from `density_matrix` to `state_vector`.
- Making `density_matrix` the default in Phase 2.
- Treating unsupported density requests as warnings or best-effort retries.
- Claiming non-VQE workflows or decomposition flows as required
  backend-selection coverage for this task.
- Introducing additional backend modes, aliases, or mixed fallback semantics
  not frozen at the phase level.

## Acceptance evidence
- Interface-level tests show that omitted backend selection preserves the
  current `state_vector` behavior.
- Positive integration tests show that explicit `density_matrix` selection
  reaches the Phase 2 anchor noisy VQE path through
  `qgd_Variational_Quantum_Eigensolver_Base`.
- Negative tests show that unsupported density requests fail before execution
  and do not fall back silently.
- Reproducible validation or benchmark artifacts record the selected backend
  for the mandatory XXZ workload set at 4, 6, 8, and 10 qubits.
- At least one reproducible 4- or 6-qubit optimization trace exercises
  explicit `density_matrix` selection end-to-end.
- Traceability target: satisfy the Phase 2 Task 1 evidence requirements in
  `DETAILED_PLANNING_PHASE_2.md`.
- Traceability target: satisfy the full-phase acceptance criterion requiring
  explicit backend selection with no implicit fallback.
- Traceability target: satisfy the Section 10.1 workflow-completeness
  threshold that explicit `density_matrix` selection works without fallback.

## Affected interfaces
- User-facing backend selection surface on
  `qgd_Variational_Quantum_Eigensolver_Base`, whether exposed as a constructor
  config entry or another equivalent VQE-facing configuration mechanism.
- Python/C++ wrapper plumbing that carries backend selection into
  `Variational_Quantum_Eigensolver_Base`.
- Pre-execution validation that checks backend mode against the frozen Phase 2
  support matrix.
- Tests, benchmarks, and reproducibility metadata that must state which
  backend executed a case.
- Change classification: additive for existing callers that rely on the
  default `state_vector` path, but stricter for ambiguous or unsupported
  mixed-state requests, which become explicit hard errors.

## Publication relevance
- Supports Paper 1's core claim that SQUANDER exposes an explicitly selectable
  exact noisy density-matrix backend inside the anchor VQE workflow rather than
  only as a standalone backend.
- Makes benchmark attribution scientifically defensible by tying each result to
  a named backend.
- Provides contract evidence for the backend-integration and
  workflow-completeness statements in the Phase 2 abstract, short paper, and
  full paper.
