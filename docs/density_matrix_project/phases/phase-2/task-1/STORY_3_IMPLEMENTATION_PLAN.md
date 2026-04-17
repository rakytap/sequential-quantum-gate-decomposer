# Story 3 Implementation Plan

## Story Being Implemented

`Story 3: Unsupported Backend Requests Fail Early And Never Fall Back
Silently`

This is a Layer 4 engineering plan for implementing the third behavioral slice
from `TASK_1_STORIES.md`.

## Scope

This story turns the frozen Phase 2 unsupported-case contract into explicit,
deterministic runtime behavior:

- explicit `density_matrix` requests are validated against the frozen Phase 2
  support matrix before execution starts,
- `state_vector` requests that depend on Phase 2-only mixed-state behavior fail
  early rather than switching backends,
- the first unsupported gate, noise channel, optimizer, or workflow condition
  is reported clearly,
- and tests and benchmark harnesses mark unsupported cases as unsupported rather
  than running them on the wrong backend.

Out of scope for this story:

- expanding the frozen support matrix,
- making optional Phase 2 extensions mandatory,
- adding new supported gate or noise families,
- the full backend-provenance package owned by Story 4,
- and broader observable or benchmark-threshold work owned by later tasks.

## Dependencies And Assumptions

- Story 1 is already in place: explicit backend selection exists, default
  `state_vector` behavior is preserved, and no implicit fallback exists at the
  API level.
- Story 2 is already in place: a supported positive `density_matrix` anchor path
  exists for default `HEA`, ordered local noise, and narrow exact density
  energy evaluation.
- The current implementation already contains the first partial support hooks in
  `validate_density_anchor_support()`, `density_optimizer_supported()`, and the
  VQE-facing density-noise surface, but Story 3 must make the unsupported-case
  contract complete and consistent.
- The frozen source-of-truth decisions remain:
  `P2-ADR-009`, `P2-ADR-011`, `P2-ADR-012`, the backend-selection decision in
  `DETAILED_PLANNING_PHASE_2.md`, and the closure language in
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.

## Engineering Tasks

### Engineering Task 1: Centralize Phase 2 Support-Matrix Preflight Validation

**Implements story**
- `Story 3: Unsupported Backend Requests Fail Early And Never Fall Back Silently`

**Change type**
- code

**Definition of done**
- A single shared preflight validator owns the Phase 2 support-matrix decision
  for VQE backend selection.
- The same validation logic is used by fixed-parameter evaluation, optimization
  entry, and any helper path that can trigger the density backend.
- Unsupported requests are rejected before state evolution or optimizer work
  begins.

**Execution checklist**
- [ ] Refactor the current density validation hooks into one shared
      support-matrix validator rooted in
      `Variational_Quantum_Eigensolver_Base`.
- [ ] Ensure `Optimization_Problem(...)`, `Optimization_Problem_Batch(...)`,
      `Start_Optimization()`, and any density-specific helper entry points all
      pass through the same validation gate.
- [ ] Keep the validator phase-contract driven: backend mode, ansatz, circuit
      source, gate family, noise surface, optimizer choice, and any
      mixed-state-only workflow request.

**Evidence produced**
- One reviewable validation path rather than scattered ad hoc checks.
- Negative regression tests proving failures happen before execution.

**Risks / rollback**
- Risk: split validation logic can drift and produce inconsistent behavior
  across entry points.
- Rollback/mitigation: keep one validator as the only support-matrix authority
  and call it from all runtime entry points.

### Engineering Task 2: Detect And Report The First Unsupported Gate Or Noise Condition

**Implements story**
- `Story 3: Unsupported Backend Requests Fail Early And Never Fall Back Silently`

**Change type**
- code | tests

**Definition of done**
- Unsupported bridge cases fail deterministically before execution and name the
  first unsupported gate, fused block, or noise insertion that violates the
  frozen support matrix.
- Error reporting distinguishes supported required behavior from optional or
  deferred behavior.
- Silent omission, silent rewriting, or partial lowering of unsupported
  operations is not possible on the mandatory workflow path.

**Execution checklist**
- [ ] Extend the current `HEA`-first bridge validation so the first unsupported
      operation is named and surfaced predictably.
- [ ] Cover unsupported gate families, invalid or out-of-range ordered-noise
      insertions, unsupported density-noise channels, and future unsupported
      optional circuit-source requests.
- [ ] Preserve the distinction between required support, optional Phase 2
      extension points, and explicitly deferred behavior.

**Evidence produced**
- Focused negative tests that assert the first unsupported item appears in the
  error message.
- Reviewable bridge-validation output tied to the frozen support matrix.

**Risks / rollback**
- Risk: vague or aggregate failure messages make debugging and auditability
  weak.
- Rollback/mitigation: report the first unsupported item and its category
  consistently so tests can lock the behavior down.

### Engineering Task 3: Reject `state_vector` Requests That Depend On Mixed-State-Only Features

**Implements story**
- `Story 3: Unsupported Backend Requests Fail Early And Never Fall Back Silently`

**Change type**
- code | tests

**Definition of done**
- Selecting `state_vector` while also requesting Phase 2-only mixed-state
  features results in a hard pre-execution error.
- No automatic backend swap to `density_matrix` occurs.
- The guardrail is applied at the VQE-facing workflow boundary, not left to
  downstream implementation accidents.

**Execution checklist**
- [ ] Define the minimal set of VQE-facing mixed-state-only features that imply
      the density backend for Phase 2.
- [ ] Reject `state_vector` requests that include density-noise configuration or
      other Story 2-only mixed-state workflow surfaces.
- [ ] Add focused negative tests showing the request fails before execution and
      does not silently route into the density backend.

**Evidence produced**
- Negative regression tests for `state_vector` plus mixed-state-only request
  combinations.
- Clear user-facing error wording that explains the backend mismatch.

**Risks / rollback**
- Risk: mixed-state-only requests may still sneak through helper APIs and run on
  the wrong backend.
- Rollback/mitigation: keep the guardrail at the top-level VQE workflow surface
  and reuse it across all entry points.

### Engineering Task 4: Gate Unsupported Optimizers, Ansatz Variants, And Circuit Sources Explicitly

**Implements story**
- `Story 3: Unsupported Backend Requests Fail Early And Never Fall Back Silently`

**Change type**
- code | tests

**Definition of done**
- Unsupported optimizer/backend combinations fail before optimization starts.
- Unsupported ansatz variants and unsupported circuit-source modes fail before
  lowering or execution starts.
- The positive Story 2 path remains narrow and explicit: default `HEA`,
  supported local noise, and the allowed value-only optimization surface.

**Execution checklist**
- [ ] Validate optimizer compatibility for `density_matrix` mode and reject
      unsupported gradient-based or otherwise unsupported combinations.
- [ ] Reject unsupported ansatz variants such as non-`HEA` density requests
      until they are explicitly added to the support matrix.
- [ ] Reject unsupported manual `qgd_Circuit` or `Gates_block` density paths
      unless they are explicitly recognized as cleanly lowerable optional
      extensions.

**Evidence produced**
- Negative tests for unsupported optimizer, ansatz, and circuit-source
  combinations.
- Stable preflight behavior before any heavy optimization or lowering work.

**Risks / rollback**
- Risk: leaving optimizer or ansatz rejection implicit will create hidden
  backend-dependent failures deep in execution.
- Rollback/mitigation: perform these checks in the shared support-matrix
  validator before dispatch.

### Engineering Task 5: Add Unsupported-Case Regression And Benchmark Coverage

**Implements story**
- `Story 3: Unsupported Backend Requests Fail Early And Never Fall Back Silently`

**Change type**
- tests | benchmark harness | validation automation

**Definition of done**
- The standard regression surface contains explicit negative cases for the Story
  3 contract.
- The benchmark or validation harness records unsupported status for out-of-scope
  cases rather than running them on `state_vector` or partially lowering them.
- Negative evidence is reproducible and reviewable alongside the positive Story 2
  path.

**Execution checklist**
- [ ] Extend `tests/VQE/test_VQE.py` with negative backend-mismatch, ansatz,
      optimizer, gate, and noise-path cases.
- [ ] Extend density-backend tests where useful to lock down ordered-noise and
      unsupported-operation behavior.
- [ ] Update benchmark or validation scripts so unsupported cases are explicitly
      marked unsupported instead of silently executed or skipped without reason.
- [ ] Keep fast negative checks in pytest and heavier unsupported-matrix sweeps
      in dedicated validation commands.

**Evidence produced**
- Negative pytest coverage for unsupported combinations.
- Benchmark or validation outputs that include explicit unsupported-case status.

**Risks / rollback**
- Risk: unsupported cases may disappear into silent skips or unrelated
  exceptions, weakening the contract.
- Rollback/mitigation: make unsupported status an intentional and testable
  outcome in both regression and benchmark tooling.

### Engineering Task 6: Stabilize Error Surface And Developer-Facing Guidance

**Implements story**
- `Story 3: Unsupported Backend Requests Fail Early And Never Fall Back Silently`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing guidance explains which backend combinations are guaranteed,
  optional, deferred, or unsupported in Phase 2.
- Error messages are stable enough to serve as audit evidence and test targets.
- Validation notes clearly distinguish unsupported outcomes from execution
  failures or numeric mismatches.

**Execution checklist**
- [ ] Update relevant VQE and density-backend docstrings, comments, or
      developer notes so unsupported cases are documented explicitly.
- [ ] Record the expected unsupported-case categories and example error wording
      in one stable location near the Story 3 workflow.
- [ ] Keep Story 3 wording consistent with the frozen support matrix and avoid
      implying any broader support than Phase 2 actually guarantees.
- [ ] Ensure validation outputs use clear unsupported labels rather than generic
      error buckets.

**Evidence produced**
- Updated developer-facing unsupported-case guidance.
- Stable negative-evidence wording usable by tests and review.

**Risks / rollback**
- Risk: unclear or drifting error text can break reproducibility and make review
  ambiguous.
- Rollback/mitigation: keep a small, explicit error taxonomy tied to the frozen
  Phase 2 support matrix.

## Exit Criteria

Story 3 is complete only when all of the following are true:

- explicit `density_matrix` requests outside the documented support matrix fail
  before execution,
- `state_vector` requests that depend on mixed-state-only behavior fail before
  execution,
- no implicit backend fallback or silent partial lowering exists on the
  mandatory workflow path,
- the first unsupported gate, noise insertion, optimizer, or workflow condition
  is reported clearly,
- and regression plus validation tooling can distinguish unsupported outcomes
  from executed workflows.

## Implementation Notes

- Story 2 already introduced the first runtime validation hooks, so Story 3
  should evolve those hooks into the full support-matrix authority instead of
  adding a second unsupported-case framework.
- The frozen contract already names the major unsupported-case categories:
  unsupported density combinations, unsupported bridge operations, unsupported
  gate families, unsupported noise models, and `state_vector` requests that
  depend on mixed-state-only behavior.
- Optional extensions should stay clearly separate from guaranteed support so
  Story 3 does not accidentally widen Phase 2 scope while improving error
  handling.
- Story 4 remains responsible for the broader provenance and reporting package;
  Story 3 only needs enough validation surface to make unsupported outcomes
  explicit and reproducible.
