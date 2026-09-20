# Task 1 Stories

This document decomposes Phase 2 Task 1 into Layer 3 behavioral stories.
These stories inherit the frozen contract from `TASK_1_MINI_SPEC.md`,
`DETAILED_PLANNING_PHASE_2.md`, and `P2-ADR-009`. They describe behavioral
slices, not implementation chores.

Story ordering is intentional:

1. preserve backward compatibility first,
2. enable the supported explicit density path second,
3. make unsupported behavior fail clearly rather than fall back,
4. expose backend provenance in validation artifacts.

## Story 1: Legacy VQE Calls Remain Valid When Backend Selection Is Omitted

### Behavioral slice

When a user creates `qgd_Variational_Quantum_Eigensolver_Base` without any
backend-selection setting, the workflow continues to behave as the existing
`state_vector` VQE path.

### Why this story exists

Phase 2 introduces explicit backend selection, but the frozen contract requires
backward compatibility for existing callers. This story protects current VQE
usage while creating the entry point for later density-matrix work.

### Done when

- omitted backend selection remains a supported call pattern,
- the effective default is `state_vector`,
- existing supported VQE entry points continue to work without caller changes,
- and the new backend-selection surface does not force immediate migration.

### Acceptance evidence

- regression tests for current VQE usage pass without adding a backend setting,
- a paired test shows omitted backend behavior is equivalent to explicit
  `state_vector` on the same supported case,
- and Story 1 can be demonstrated without requiring density-matrix execution.

## Story 2: Explicit `density_matrix` Selection Activates The Supported Anchor Noisy VQE Path

### Behavioral slice

When a user explicitly selects `density_matrix` for the supported Phase 2
anchor workflow, the workflow reaches the exact noisy mixed-state execution path
instead of the legacy state-vector path.

### Why this story exists

Task 1 is not complete unless backend choice becomes a real user-facing
behavior rather than only a documented future intention.

### Done when

- explicit `density_matrix` selection is accepted on the VQE-facing entry point,
- the supported XXZ plus `HEA` anchor workflow reaches the density path,
- the selected backend is unambiguous in the executed workflow,
- and the behavior is usable for later Task 2 observable validation.

### Acceptance evidence

- positive integration tests show explicit `density_matrix` selection reaches
  the supported anchor workflow path,
- supported 4- and 6-qubit anchor cases execute through the selected backend,
- and benchmark setup records that the density path, not the legacy path, was
  exercised.

## Story 3: Unsupported Backend Requests Fail Early And Never Fall Back Silently

### Behavioral slice

When backend selection is incompatible with the requested workflow, the system
fails before execution with a hard error instead of silently choosing another
backend.

### Why this story exists

The phase contract explicitly rejects fallback behavior. Scientific claims and
benchmarks become ambiguous if unsupported density requests run on
`state_vector` without the user realizing it.

### Done when

- unsupported `density_matrix` combinations are rejected pre-execution,
- `state_vector` requests that attempt Phase 2-only mixed-state behavior are
  rejected pre-execution,
- no implicit `auto` or best-effort backend switching exists,
- and error handling makes the scope boundary visible to the caller.

### Acceptance evidence

- negative tests show unsupported density requests fail before execution,
- negative tests show mixed-state-only requests do not auto-switch from
  `state_vector` to `density_matrix`,
- and benchmark cases outside the support matrix are marked unsupported rather
  than run on the wrong backend.

## Story 4: Validation And Reproducibility Artifacts Name The Backend Used

### Behavioral slice

When Phase 2 validation, benchmark, and reproducibility artifacts are produced,
they name which backend executed each case so the results are scientifically
attributable.

### Why this story exists

Task 1 requires backend choice to be unambiguous not only at runtime, but also
in the evidence used for Paper 1 and Phase 2 acceptance.

### Done when

- tests or validation logs identify the selected backend,
- benchmark artifacts distinguish `state_vector` from `density_matrix`,
- and reproducibility bundles preserve backend provenance for the mandatory
  workload set.

### Acceptance evidence

- mandatory workflow benchmark outputs include backend provenance,
- reproducibility notes or machine-readable artifacts capture the selected
  backend for each case,
- and reviewers can determine from stored evidence whether a result came from
  the legacy path or the Phase 2 density path.
