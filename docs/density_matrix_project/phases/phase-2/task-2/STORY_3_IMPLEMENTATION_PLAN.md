# Story 3 Implementation Plan

## Story Being Implemented

Story 3: Out-Of-Scope Observable Requests Fail Explicitly Instead Of Degrading
Silently

This is a Layer 4 engineering plan for implementing the third behavioral slice
from `TASK_2_STORIES.md`.

## Scope

This story turns the frozen unsupported-case contract for Task 2 into explicit,
auditable runtime and validation behavior:

- exact noisy observable requests outside the documented Phase 2 support surface
  fail before execution,
- unsupported observable, Hamiltonian, bridge, gate, and noise conditions are
  classified and reported deterministically,
- validation and benchmark tooling record unsupported cases as structured
  outcomes instead of silently skipping, degrading, or rerouting them,
- and no implicit fallback to `state_vector`, shot-noise, or partial best-effort
  evaluation remains in the mandatory Task 2 path.

Out of scope for this story:

- widening the frozen observable contract beyond exact Hermitian energy
  evaluation,
- adding new supported gate or noise families,
- 4 to 10 qubit workflow-sweep exactness work owned by Story 4,
- publication-bundle and optimization-trace packaging owned by Story 5,
- and VQE public-API expansion to expose generic measurement or density-matrix
  objects directly.

## Dependencies And Assumptions

- Story 1 is already in place: the supported exact noisy positive path exists in
  the anchor VQE workflow.
- Story 2 is already in place: the mandatory 1 to 3 qubit micro-validation gate
  exists and proves the supported local exactness surface.
- The current implementation already contains partial unsupported-case hooks in
  `validate_density_anchor_support()` and related VQE tests, plus structured
  unsupported artifact capture in
  `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py`.
- The frozen source-of-truth decisions remain:
  `P2-ADR-009`, `P2-ADR-010`, `P2-ADR-011`, `P2-ADR-012`, and the backend,
  observable, bridge, and support-matrix decisions in
  `DETAILED_PLANNING_PHASE_2.md`.
- Story 3 should complete unsupported-case behavior without reopening phase-level
  scope decisions or broadening the supported exact observable surface.

## Engineering Tasks

### Engineering Task 1: Centralize Task 2 Support-Surface Preflight Validation

**Implements story**
- `Story 3: Out-Of-Scope Observable Requests Fail Explicitly Instead Of Degrading Silently`

**Change type**
- code

**Definition of done**
- One shared preflight validator owns the Task 2 support-surface decision for
  exact noisy observable evaluation.
- The same support checks are applied consistently to fixed-parameter VQE
  execution, optimization entry points, and relevant validation helper paths.
- Unsupported requests fail before state evolution, optimizer work, or external
  comparison begins.

**Execution checklist**
- [ ] Refactor or extend the current VQE-side validation hooks so one shared
      support-surface validator covers backend mode, observable form,
      Hamiltonian contract, circuit source, gate family, and noise surface.
- [ ] Ensure all exact noisy energy entry points that can reach the density path
      use the same validator rather than duplicating partial checks.
- [ ] Keep the validator phase-contract driven and avoid embedding ad hoc
      support decisions in downstream helpers.

**Evidence produced**
- One reviewable support-surface authority for exact noisy observable requests.
- Clear pre-execution failures instead of downstream accidental exceptions.

**Risks / rollback**
- Risk: split or duplicated validation logic can drift and produce inconsistent
  unsupported behavior across entry points.
- Rollback/mitigation: keep one validator as the only source of Task 2
  support-matrix truth and call it everywhere.

### Engineering Task 2: Reject Unsupported Observable And Hamiltonian Forms Explicitly

**Implements story**
- `Story 3: Out-Of-Scope Observable Requests Fail Explicitly Instead Of Degrading Silently`

**Change type**
- code | tests

**Definition of done**
- Requests outside the frozen observable contract fail explicitly before
  execution.
- The only supported Task 2 observable path remains exact real-valued Hermitian
  energy evaluation through the existing sparse-Hamiltonian VQE surface.
- Unsupported observable families do not degrade into a nearby but scientifically
  different evaluation mode.

**Execution checklist**
- [ ] Validate the Hermitian sparse-Hamiltonian contract explicitly where needed
      for the exact noisy path.
- [ ] Reject arbitrary non-Hermitian observable requests, generic measurement
      surfaces, batched multi-observable behavior, and shot-noise/readout-based
      paths as unsupported in the Task 2 minimum.
- [ ] Add focused negative tests that show unsupported observable or Hamiltonian
      forms fail before density evolution or reference comparison begins.

**Evidence produced**
- Focused negative tests for unsupported observable and Hamiltonian forms.
- Stable error wording tied to the frozen exact Hermitian energy contract.

**Risks / rollback**
- Risk: unsupported observable requests may silently look valid if they degrade
- into a nearby energy-only path.
- Rollback/mitigation: validate the exact contract shape up front and hard-error
  anything outside it.

### Engineering Task 3: Detect And Report The First Unsupported Bridge, Gate, Or Noise Condition

**Implements story**
- `Story 3: Out-Of-Scope Observable Requests Fail Explicitly Instead Of Degrading Silently`

**Change type**
- code | tests

**Definition of done**
- Unsupported circuit-source, gate-family, or ordered-noise conditions fail
  deterministically before execution.
- Error reporting identifies the first unsupported condition and its category.
- Silent omission, silent rewriting, or partial lowering is not possible on the
  mandatory Task 2 exact observable path.

**Execution checklist**
- [ ] Extend the current bridge and support-matrix validation so unsupported
      circuit sources, first unsupported gates, invalid ordered-noise insertions,
      and unsupported density-noise channels are surfaced predictably.
- [ ] Preserve the distinction between required support, optional documented
      extensions, and explicitly deferred behavior.
- [ ] Add focused negative tests that assert the first unsupported item appears
      in the error message and that execution does not proceed.

**Evidence produced**
- Reviewable first-failure reporting for bridge, gate, and noise violations.
- Negative regression tests that lock down deterministic unsupported behavior.

**Risks / rollback**
- Risk: vague aggregate failures make debugging and scientific auditability weak.
- Rollback/mitigation: report the first unsupported condition and its category
  consistently so tests can pin the behavior down.

### Engineering Task 4: Reject Backend And Evaluation-Mode Degradation Paths

**Implements story**
- `Story 3: Out-Of-Scope Observable Requests Fail Explicitly Instead Of Degrading Silently`

**Change type**
- code | tests

**Definition of done**
- `state_vector` requests that depend on mixed-state-only observable behavior
  fail before execution.
- Unsupported density requests do not silently fall back to `state_vector`,
  shot-noise, or any partial best-effort route.
- The frozen Task 2 exact observable path remains a hard yes/no contract rather
  than a heuristic degradation ladder.

**Execution checklist**
- [ ] Reuse Task 1 backend-selection guardrails for Task 2 observable requests
      that implicitly require mixed-state execution.
- [ ] Reject degradation into shot-noise or readout-style paths when exact noisy
      density evaluation is unsupported.
- [ ] Add focused negative regression coverage for backend mismatch and
      evaluation-mode degradation cases.

**Evidence produced**
- Negative tests for backend mismatch and forbidden degradation behavior.
- Clear, reproducible errors showing no fallback occurred.

**Risks / rollback**
- Risk: unsupported requests may still appear to work if a nearby evaluation
  path is silently invoked.
- Rollback/mitigation: perform explicit backend/evaluation-mode checks before
  dispatch and treat degradation as a contract violation.

### Engineering Task 5: Add Structured Unsupported-Case Artifact Coverage

**Implements story**
- `Story 3: Out-Of-Scope Observable Requests Fail Explicitly Instead Of Degrading Silently`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Unsupported observable-surface cases are emitted as structured, reviewable
  artifact outcomes.
- Validation tooling distinguishes unsupported cases from numeric mismatches and
  from completed supported cases.
- Structured unsupported output is stable enough to be cited as negative
  evidence in later docs and papers.

**Execution checklist**
- [ ] Extend the existing unsupported artifact pattern in
      `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` where useful
      for workflow-level negative cases.
- [ ] Decide whether low-level Story 2 validators should stay supported-only or
      gain a separate unsupported bundle, but keep the mandatory supported gate
      separate from unsupported-case reporting.
- [ ] Use stable fields for unsupported category, reason, backend, case identity,
      and support-surface layer.
- [ ] Keep unsupported outcomes explicit rather than disappearing into generic
      script failures or silent skips.

**Evidence produced**
- Stable structured unsupported-case artifacts for relevant Task 2 negative
  cases.
- Reviewable negative evidence distinct from supported validation bundles.

**Risks / rollback**
- Risk: unsupported cases may become untraceable if they are mixed into supported
  bundles or emitted as generic script crashes.
- Rollback/mitigation: keep unsupported status intentional, typed, and
  separately auditable.

### Engineering Task 6: Add Focused Regression Coverage Across VQE And Validation Layers

**Implements story**
- `Story 3: Out-Of-Scope Observable Requests Fail Explicitly Instead Of Degrading Silently`

**Change type**
- tests | benchmark harness | validation automation

**Definition of done**
- The standard regression surface contains explicit negative cases for the major
  Task 2 unsupported categories.
- Validation tooling also exercises representative unsupported cases in a
  reproducible way.
- The combined test surface is strong enough to keep Story 3 closed while still
  remaining practical to run during development.

**Execution checklist**
- [ ] Extend `tests/VQE/test_VQE.py` with negative observable-surface cases for
      unsupported Hamiltonian forms, unsupported gate/noise requests, backend
      mismatch, and unsupported circuit sources where appropriate.
- [ ] Add or extend focused density-validation tests only where they improve
      unsupported-case localization.
- [ ] Keep fast negative checks in pytest and heavier unsupported-matrix runs in
      dedicated validation commands.
- [ ] Ensure failure signals clearly differentiate unsupported scope outcomes
      from numeric exactness failures.

**Evidence produced**
- Negative pytest coverage across the main unsupported Task 2 categories.
- Dedicated validation outputs that reproduce representative unsupported cases.

**Risks / rollback**
- Risk: without explicit regression coverage, unsupported behavior can regress
  into silent execution or ambiguous failures.
- Rollback/mitigation: keep one representative test per major unsupported
  category and one stable validation command for broader review.

### Engineering Task 7: Stabilize Developer-Facing Guidance And Error Taxonomy

**Implements story**
- `Story 3: Out-Of-Scope Observable Requests Fail Explicitly Instead Of Degrading Silently`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing guidance explains which exact noisy observable requests are
  guaranteed, optional, deferred, or unsupported in Phase 2.
- Error messages and unsupported categories are stable enough to serve as test
  targets and review evidence.
- Validation notes clearly distinguish unsupported outcomes from executed but
  numerically failing workflows.

**Execution checklist**
- [ ] Update relevant VQE and validation docstrings, comments, or developer notes
      so the unsupported Task 2 surface is documented explicitly.
- [ ] Record a small supported/unsupported error taxonomy in one stable location
      near the Story 3 workflow.
- [ ] Keep Story 3 wording aligned with the frozen observable and support-matrix
      decisions and avoid implying broader support.
- [ ] Ensure validation outputs use consistent unsupported labels rather than
      generic error buckets.

**Evidence produced**
- Updated developer-facing unsupported-case guidance for Task 2.
- Stable unsupported-case taxonomy usable by tests and review.

**Risks / rollback**
- Risk: drifting or ambiguous error text weakens reproducibility and makes later
  review harder.
- Rollback/mitigation: keep a small explicit taxonomy tied directly to the
  frozen support matrix and reuse it across code, tests, and artifacts.

### Engineering Task 8: Run Story 3 Validation And Confirm Unsupported Outcomes Are Explicit

**Implements story**
- `Story 3: Out-Of-Scope Observable Requests Fail Explicitly Instead Of Degrading Silently`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 3 negative regression tests pass.
- Representative unsupported cases are emitted through stable validation or
  benchmark paths as structured unsupported outcomes.
- Story 3 completion is backed by reviewable negative evidence rather than by
  code changes alone.

**Execution checklist**
- [ ] Run the focused Story 3 regression tests.
- [ ] Run the relevant validation or benchmark commands that emit structured
      unsupported-case results.
- [ ] Verify that unsupported cases fail before execution and are not counted as
      supported-pass numeric validations.
- [ ] Record the stable test run and artifact references for later Task 2 docs
      and paper evidence.

**Evidence produced**
- Passing focused Story 3 negative pytest coverage.
- Stable unsupported-case artifact references from validation or benchmark runs.

**Risks / rollback**
- Risk: Story 3 can appear complete while still lacking auditable proof that
  unsupported outcomes are explicit and reproducible.
- Rollback/mitigation: treat the negative artifact references and focused test
  runs as part of the exit gate, not optional follow-up.

## Exit Criteria

Story 3 is complete only when all of the following are true:

- exact noisy `density_matrix` requests outside the frozen Task 2 support matrix
  fail before execution,
- unsupported observable or Hamiltonian forms fail explicitly rather than
  degrading into another measurement path,
- `state_vector` requests that depend on mixed-state-only behavior fail before
  execution,
- the first unsupported circuit-source, gate, or noise condition is reported
  clearly,
- and regression plus validation tooling can distinguish unsupported outcomes
  from supported executed workflows and from numeric exactness failures.

## Implementation Notes

- The current `validate_density_anchor_support()` path already covers several
  Story 3 categories, so Story 3 should evolve that path into the full
  support-surface authority rather than introduce a second unsupported-case
  framework.
- `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` already emits
  structured unsupported artifacts for at least one backend-mismatch case.
  Story 3 should reuse and extend that pattern where it adds negative evidence.
- The Story 2 low-level micro-validation bundle is a supported-only local
  correctness gate. Story 3 should avoid diluting that bundle with unsupported
  cases unless it does so through a clearly separate negative-evidence surface.
- Optional extensions should stay clearly separated from guaranteed behavior so
  Story 3 strengthens unsupported-case clarity without accidentally widening
  Phase 2 scope.
