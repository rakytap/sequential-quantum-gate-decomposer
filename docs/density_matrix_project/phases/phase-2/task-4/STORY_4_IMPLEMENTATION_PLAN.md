# Story 4 Implementation Plan

## Story Being Implemented

Story 4: Deferred Or Unsupported Noise Requests Fail Before Execution With The
First Unsupported Noise Condition

This is a Layer 4 engineering plan for implementing the fourth behavioral slice
from `TASK_4_STORIES.md`.

## Scope

This story turns the frozen unsupported and deferred Task 4 noise boundary into
explicit, auditable runtime and validation behavior:

- unsupported or deferred noise classes fail before density execution begins,
- the first unsupported noise condition is reported deterministically and in a
  machine-reviewable way,
- unsupported requests do not silently substitute another noise model, silently
  collapse into a whole-register baseline, or silently reroute to another
  backend,
- and validation tooling records negative Task 4 outcomes as structured
  unsupported evidence instead of generic crashes or silent disappearance.

Out of scope for this story:

- promoting optional whole-register depolarizing or any justified extension into
  the required baseline,
- the required local-noise positive-path evidence already owned by Story 1,
- the exact micro-validation gate already owned by Story 2,
- the optional-versus-required classification layer already owned by Story 3,
- the 4/6/8/10 workflow-scale sufficiency package owned by Story 5,
- and the final provenance and publication bundle owned by Story 6.

## Dependencies And Assumptions

- Story 1 is already in place: the required local-noise positive path exists and
  exposes reviewable bridge metadata through
  `required_local_noise_validation_validation.py`.
- Story 2 is already in place: the mandatory 1 to 3 qubit exact micro-validation
  gate passes and produces machine-readable exactness artifacts through
  `validate_squander_vs_qiskit.py` and
  `required_local_noise_micro_validation.py`.
- Story 3 is already in place: optional whole-register depolarizing evidence is
  explicitly classified as optional through
  `optional_noise_classification_validation.py`, so Story 4 must
  not reclassify that optional baseline as unsupported.
- The current Python-side boundary already normalizes and rejects unsupported
  channels in `_normalize_density_noise_spec()` inside
  `squander/VQA/qgd_Variational_Quantum_Eigensolver_Base.py`.
- The current C++-side boundary already validates `density_noise` structure and
  insertion semantics in `set_density_noise_specs()` and
  `validate_density_anchor_support()` inside
  `squander/src-cpp/variational_quantum_eigensolver/Variational_Quantum_Eigensolver_Base.cpp`.
- Existing unsupported evidence patterns already exist in
  `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` and
  `benchmarks/density_matrix/bridge_scope/unsupported_bridge_validation.py`.
- The frozen support surface remains:
  - required: `local_depolarizing`, `amplitude_damping`, `phase_damping`,
  - optional: whole-register depolarizing as regression or stress baseline, and
    generalized amplitude damping or coherent over-rotation only if a justified
    benchmark extension requires them,
  - deferred: correlated multi-qubit noise, readout noise as a density-backend
    feature, calibration-aware noise, and non-Markovian noise.
- Story 4 should harden negative boundary behavior without reopening
  `P2-ADR-012` or weakening Task 1's no-fallback rules.

## Engineering Tasks

### Engineering Task 1: Centralize Task 4 Noise-Boundary Preflight Validation

**Implements story**
- `Story 4: Deferred Or Unsupported Noise Requests Fail Before Execution With The First Unsupported Noise Condition`

**Change type**
- code

**Definition of done**
- One shared validation path owns the Task 4 noise boundary for deferred and
  unsupported requests.
- The same checks apply consistently to fixed-parameter execution, optimization
  entry points, and validation helpers that can reach the density path.
- Unsupported requests fail before lowering, density evolution, or external
  comparison begins.

**Execution checklist**
- [ ] Keep `_normalize_density_noise_spec()`, `set_density_noise_specs()`, and
      `validate_density_anchor_support()` as the primary Task 4 noise-boundary
      authorities.
- [ ] Ensure all density-noise entry points rely on the same validation logic
      rather than duplicating partial checks.
- [ ] Preserve the distinction between required, optional, deferred, and
      unsupported noise cases across Python and C++ boundaries.
- [ ] Avoid embedding ad hoc support decisions in downstream execution helpers.

**Evidence produced**
- One reviewable Task 4 noise-boundary validation authority.
- Clear pre-execution failures instead of downstream accidental exceptions.

**Risks / rollback**
- Risk: split or duplicated validation logic can drift and produce inconsistent
  unsupported behavior across density entry points.
- Rollback/mitigation: keep one authoritative validation path and call it
  everywhere.

### Engineering Task 2: Reject Deferred Noise Families Explicitly And Deterministically

**Implements story**
- `Story 4: Deferred Or Unsupported Noise Requests Fail Before Execution With The First Unsupported Noise Condition`

**Change type**
- code | tests

**Definition of done**
- Deferred noise families fail explicitly before density execution.
- Error reporting distinguishes deferred classes from required and optional
  classes.
- Deferred requests do not silently degrade into supported local or optional
  whole-register baselines.

**Execution checklist**
- [ ] Add focused negative coverage for correlated multi-qubit noise, readout
      noise as a density-backend feature, calibration-aware noise, and
      non-Markovian noise.
- [ ] Keep the first-failure semantics explicit even when the unsupported class
      is rejected on the Python normalization boundary.
- [ ] Make sure deferred classes are represented as unsupported for current
      execution, but remain documented as deferred scope rather than accidental
      omissions.
- [ ] Preserve the already passing required and optional cases as regression
      guards.

**Evidence produced**
- Deterministic negative tests for deferred Task 4 noise families.
- Stable failure wording tied to the frozen deferred support-matrix entries.

**Risks / rollback**
- Risk: deferred classes can be misread as optional if the failure taxonomy is
  vague.
- Rollback/mitigation: keep deferred classes explicitly named in tests and
  artifact metadata.

### Engineering Task 3: Detect And Report Invalid Ordered-Noise Insertions Before Execution

**Implements story**
- `Story 4: Deferred Or Unsupported Noise Requests Fail Before Execution With The First Unsupported Noise Condition`

**Change type**
- code | tests

**Definition of done**
- Invalid ordered-noise insertions fail deterministically before density
  execution.
- The first unsupported condition reports whether the defect is insertion order,
  invalid position, or structural metadata mismatch.
- Supported ordered local-noise behavior remains distinct from insertion errors.

**Execution checklist**
- [ ] Reuse and tighten `after_gate_index` validation across Python and C++.
- [ ] Add focused negative tests for invalid `after_gate_index`, negative or
      structurally invalid insertion metadata where applicable, and other ordered
      insertion defects owned by Task 4.
- [ ] Keep failure timing before bridge lowering or density evolution whenever
      possible.
- [ ] Preserve the current supported ordered local-noise cases as regression
      guards against overblocking valid requests.

**Evidence produced**
- Reviewable first-failure reporting for invalid ordered-noise insertions.
- Negative regression tests locking down insertion-boundary behavior.

**Risks / rollback**
- Risk: insertion defects may surface later as generic execution failures,
  weakening auditability.
- Rollback/mitigation: validate ordered insertion semantics before any density
  execution helper is called.

### Engineering Task 4: Reject Unsupported Noise Types Without Silent Substitution

**Implements story**
- `Story 4: Deferred Or Unsupported Noise Requests Fail Before Execution With The First Unsupported Noise Condition`

**Change type**
- code | tests

**Definition of done**
- Unsupported noise types fail explicitly and do not silently normalize into a
  different supported or optional class.
- Optional whole-register depolarizing remains optional by explicit request only,
  not as a fallback destination for unsupported local-noise requests.
- Error reporting identifies the first unsupported noise type clearly enough for
  tests and artifacts to pin down.

**Execution checklist**
- [ ] Tighten unsupported-channel handling in `_normalize_density_noise_spec()`
      and the wrapper/C++ boundary where needed.
- [ ] Add focused negative tests for unsupported channels beyond the current
      required and optional sets.
- [ ] Assert that unsupported requests do not reappear as `local_depolarizing`,
      `phase_damping`, `amplitude_damping`, or `depolarizing` after normalization.
- [ ] Keep alias support (`dephasing` to `phase_damping`, `depolarizing` to
      `local_depolarizing` where intentionally allowed) explicit and documented.

**Evidence produced**
- Deterministic unsupported-channel failure coverage.
- Reviewable proof that unsupported noise types are not silently substituted.

**Risks / rollback**
- Risk: overbroad aliasing or normalization may accidentally turn unsupported
  requests into apparently valid ones.
- Rollback/mitigation: keep the accepted canonical and alias set small,
  explicit, and directly tested.

### Engineering Task 5: Ban Silent Backend Or Baseline Reroute For Unsupported Noise Requests

**Implements story**
- `Story 4: Deferred Or Unsupported Noise Requests Fail Before Execution With The First Unsupported Noise Condition`

**Change type**
- code | tests

**Definition of done**
- Unsupported Task 4 noise requests cannot silently fall back to `state_vector`.
- Unsupported requests cannot silently degrade into a noiseless path or an
  optional whole-register depolarizing baseline while still appearing
  successful.
- Failure remains a hard yes/no contract for the mandatory Phase 2 density-noise
  path.

**Execution checklist**
- [ ] Audit the density-noise call path to ensure unsupported requests cannot
      continue through hidden fallback or alternate helpers.
- [ ] Add focused negative regression coverage for reroute, rewrite, or
      substitution risks where practical.
- [ ] Keep Story 4 aligned with Task 1 no-fallback rules without turning it into
      a second backend-selection story.
- [ ] Confirm optional whole-register cases remain available only when explicitly
      requested and explicitly classified.

**Evidence produced**
- Negative tests proving unsupported noise requests do not reroute or degrade.
- Clear reproducible errors showing the request failed before execution.

**Risks / rollback**
- Risk: unsupported requests may still appear to work if a nearby supported path
  is silently invoked.
- Rollback/mitigation: perform explicit support checks before any execution
  helper is called.

### Engineering Task 6: Emit Structured Task 4 Unsupported-Noise Artifacts

**Implements story**
- `Story 4: Deferred Or Unsupported Noise Requests Fail Before Execution With The First Unsupported Noise Condition`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Unsupported Task 4 noise cases are emitted as structured, reviewable
  artifacts.
- Validation tooling distinguishes unsupported Task 4 outcomes from required
  passes, optional cases, and numeric exactness failures owned elsewhere.
- Negative artifact output is stable enough to be cited in later docs and
  publication bundles.

**Execution checklist**
- [ ] Extend the current unsupported artifact pattern from
      `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` and/or
      `unsupported_bridge_validation.py` for Task 4 noise boundary
      cases.
- [ ] Use stable fields for unsupported category, first unsupported condition,
      support tier, backend, and case identity.
- [ ] Keep the negative artifact vocabulary compatible with the Task 4
      classification and required-baseline fields where practical.
- [ ] Ensure unsupported outcomes are explicit artifact records rather than
      silent script exits or generic crashes.

**Evidence produced**
- Stable structured unsupported-noise artifacts for Task 4 negative cases.
- Reviewable negative evidence distinct from required and optional bundles.

**Risks / rollback**
- Risk: unsupported Task 4 noise cases may become untraceable if they are mixed
  into supported bundles or emitted only as generic exceptions.
- Rollback/mitigation: keep unsupported status typed, intentional, and
  separately auditable.

### Engineering Task 7: Add Focused Regression Coverage And A Small Task 4 Unsupported Taxonomy

**Implements story**
- `Story 4: Deferred Or Unsupported Noise Requests Fail Before Execution With The First Unsupported Noise Condition`

**Change type**
- tests | docs | validation automation

**Definition of done**
- The standard regression surface contains representative negative cases for the
  major Task 4 unsupported categories.
- Developer-facing guidance explains which Task 4 noise requests are required,
  optional, deferred, or unsupported in current Phase 2 execution.
- Error categories are stable enough to serve as test targets and review
  evidence.

**Execution checklist**
- [ ] Extend `tests/VQE/test_VQE.py` and/or focused density-matrix tests with
      negative Task 4 cases for deferred classes, unsupported channels, and
      invalid ordered insertion.
- [ ] Record a small required/optional/deferred/unsupported taxonomy in one
      stable Task 4-facing location near the validation workflow.
- [ ] Keep validation outputs and test expectations aligned with the same
      taxonomy rather than ad hoc strings.
- [ ] Reuse Task 4 support-tier fields and unsupported artifact categories where
      possible to avoid schema translation between stories.
- [ ] Avoid implying that optional breadth has already been promoted to required
      support.

**Evidence produced**
- Negative pytest coverage across the main unsupported Task 4 categories.
- Stable Task 4 unsupported taxonomy usable by tests and review.

**Risks / rollback**
- Risk: drifting or inconsistent error text weakens reproducibility and makes
  later provenance work harder.
- Rollback/mitigation: keep a small explicit taxonomy tied directly to the
  frozen Task 4 support surface and reuse it everywhere.

### Engineering Task 8: Run Story 4 Validation And Confirm Unsupported Noise Outcomes Are Explicit

**Implements story**
- `Story 4: Deferred Or Unsupported Noise Requests Fail Before Execution With The First Unsupported Noise Condition`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 4 negative regression tests pass.
- Representative unsupported Task 4 noise cases are emitted through stable
  validation paths as structured unsupported outcomes.
- Story 4 completion is backed by reviewable negative evidence rather than code
  changes alone.

**Execution checklist**
- [ ] Run the focused Story 4 regression tests.
- [ ] Run the relevant validation commands that emit structured unsupported Task
      4 results.
- [ ] Verify unsupported noise cases fail before execution and are not counted as
      required or optional passes.
- [ ] Record stable test and artifact references for later Task 4 docs and
      publication bundles.

**Evidence produced**
- Passing focused Story 4 negative pytest coverage.
- Stable unsupported-noise artifact references from validation runs.

**Risks / rollback**
- Risk: Story 4 can appear complete while still lacking auditable proof that
  unsupported Task 4 outcomes are explicit and reproducible.
- Rollback/mitigation: treat negative artifact references and focused test runs
  as part of the exit gate, not optional cleanup.

## Exit Criteria

Story 4 is complete only when all of the following are true:

- deferred or unsupported Task 4 noise requests fail before execution,
- invalid ordered-noise insertions fail deterministically and report the first
  unsupported condition,
- unsupported requests do not silently substitute another required or optional
  noise model, do not fall back to `state_vector`, and do not degrade into a
  noiseless path,
- required and optional Task 4 cases remain distinct from unsupported outcomes
  in machine-readable artifacts,
- and regression plus validation tooling can distinguish unsupported Task 4
  outcomes from required passes, optional cases, and unrelated numeric
  exactness failures.

## Implementation Notes

- `_normalize_density_noise_spec()`, `set_density_noise_specs()`, and
  `validate_density_anchor_support()` already contain most of the natural Task 4
  boundary logic and should evolve into the full unsupported-noise authority
  rather than being replaced by a second framework.
- `append_density_noise_for_gate_index()` is the natural place to preserve
  first-unsupported reporting for post-normalization noise-type and insertion
  failures.
- Story 3 already established a support-tier classification vocabulary. Story 4
  should extend that vocabulary with unsupported outcomes rather than inventing a
  separate incompatible schema.
- Existing unsupported artifact patterns in
  `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` and
  `unsupported_bridge_validation.py` are the right starting points
  for Task 4 negative evidence.
- Optional whole-register depolarizing must remain optional by explicit request;
  Story 4 should treat any silent substitution toward it as a bug, not as a
  convenience behavior.
