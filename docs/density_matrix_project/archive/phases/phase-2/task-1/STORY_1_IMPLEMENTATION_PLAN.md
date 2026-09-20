# Story 1 Implementation Plan

## Story Being Implemented

`Story 1: Legacy VQE Calls Remain Valid When Backend Selection Is Omitted`

This is a Layer 4 engineering plan for implementing the first behavioral slice
from `TASK_1_STORIES.md`.

## Scope

This story introduces the user-facing backend-selection surface while freezing
the legacy default behavior:

- omitted backend selection still means `state_vector`,
- current VQE callers do not need to change their code,
- explicit backend handling is introduced in a way that later stories can build
  on,
- and no density-matrix routing is required for Story 1 completion.

Out of scope for this story:

- routing supported workflows into the density-matrix path,
- validating density support-matrix coverage against gates, noise, or
  observables,
- benchmark provenance reporting beyond what is needed for Story 1 tests,
- and any fallback-versus-hard-error cases tied specifically to unsupported
  density execution.

## Engineering Tasks

### 1. Freeze the user-facing backend-selection API

- Choose the Story 1 public entry point on
  `qgd_Variational_Quantum_Eigensolver_Base`.
- Accept the contract values `state_vector` and `density_matrix` at the
  user-facing layer, while keeping omission valid and equivalent to
  `state_vector`.
- Document the chosen parameter name and input semantics so later stories reuse
  the same surface instead of introducing parallel APIs.

### 2. Add Python-side normalization for backend mode

- Update `squander/VQA/qgd_Variational_Quantum_Eigensolver_Base.py` to
  normalize omitted backend selection and explicit backend values before they
  cross into the C++ wrapper.
- Keep the user-facing contract string-based even if the internal
  representation is numeric or enum-based.
- Reject malformed backend-mode input early at the Python boundary if it cannot
  map to the frozen contract values.

### 3. Extend the wrapper/config path to carry backend mode safely

- Update `squander/VQA/qgd_VQE_Base_Wrapper.cpp` so backend selection survives
  object construction.
- Because the current wrapper and `Config_Element` path do not carry string
  values, choose one internal transport strategy and use it consistently:
  numeric enum key in `config`, dedicated setter, or equivalent explicit
  backend field.
- Preserve behavior for all existing numeric config entries already used by VQE
  callers and tests.

### 4. Store the effective backend mode in the VQE core

- Extend `Variational_Quantum_Eigensolver_Base` so each instance has an
  effective backend mode with default `state_vector`.
- Initialize that mode during construction from the normalized backend
  selection.
- Keep the legacy state-vector execution path as the only active execution path
  for omitted backend and explicit `state_vector` in Story 1.

### 5. Preserve legacy execution behavior

- Audit the VQE setup and execution path used by
  `Generate_Circuit()`, `Optimization_Problem()`, and related entry points to
  ensure Story 1 changes do not alter existing state-vector behavior.
- Avoid any implicit density routing, mixed execution, or silent backend
  switching in this story.
- If internal branching is introduced, make `state_vector` the default branch
  when no backend is specified.

### 6. Add regression tests for backward compatibility

- Add a focused test showing that constructing
  `qgd_Variational_Quantum_Eigensolver_Base` without backend selection still
  supports the current VQE flow.
- Add a paired test showing explicit `state_vector` behaves equivalently to the
  omitted-backend case on the same deterministic workload.
- Reuse the style and workload patterns from `tests/VQE/test_VQE.py` so Story 1
  evidence stays close to existing caller behavior.

### 7. Add focused API validation tests

- Add tests that accepted user-facing backend values are normalized correctly at
  object-construction time.
- Add a negative test for malformed backend names at the user-facing API
  boundary.
- Keep unsupported-but-well-formed density workflow combinations for Story 3,
  not this story.

### 8. Update developer-facing documentation

- Update constructor or wrapper docstrings so the default behavior is explicit:
  omitted backend means `state_vector`.
- Add short implementation notes where useful so Story 2 can reuse the same
  backend-selection surface without re-deciding semantics.

### 9. Run Story 1 validation

- Run the relevant VQE regression coverage from `tests/VQE/test_VQE.py`.
- Run the new Story 1 backend-selection tests.
- Confirm that the added backend-selection surface does not regress existing
  state-vector VQE behavior.

## Exit Criteria

Story 1 is complete only when all of the following are true:

- existing VQE callers can omit backend selection and still use the current
  state-vector behavior,
- explicit `state_vector` is behaviorally equivalent to omitted backend on the
  chosen regression case,
- malformed backend names are rejected at the user-facing boundary,
- and no density-matrix routing is required to demonstrate Story 1 completion.

## Implementation Notes

- The current Python wrapper already accepts a `config` dictionary, so the
  implementation should reuse that integration surface unless there is a strong
  reason not to.
- The current wrapper converts integer and float config values, but not string
  values, so raw backend strings cannot be forwarded unchanged through the
  existing `Config_Element` path.
- Story 1 should create the stable API and default behavior needed by later
  stories, but it should not spill into Story 2 density-path routing or Story 3
  support-matrix validation.
