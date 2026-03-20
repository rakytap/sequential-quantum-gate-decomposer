# Story 4 Implementation Plan

## Story Being Implemented

Story 4: Positive Calibration Evidence Is Counted Only On Correctness-Preserving
Supported Runs

This is a Layer 4 engineering plan for implementing the fourth behavioral slice
from `TASK_5_STORIES.md`.

## Scope

This story turns the Task 5 calibration claim into one correctness-gated evidence
surface:

- only supported runs that preserve the frozen correctness thresholds count as
  positive calibration evidence,
- the sequential density baseline and required Aer baseline remain visible where
  the phase contract requires them,
- the positive-evidence rule stays aligned with the supported runtime surface and
  no-silent-fallback behavior,
- and Story 4 closes the contract for "which calibration runs count positively"
  without yet claiming final supported-planner selection or final bundle
  packaging.

Out of scope for this story:

- planner-candidate identity already owned by Story 1,
- workload-matrix anchoring already owned by Story 2,
- density-aware signal differentiation already owned by Story 3,
- final supported-claim selection already owned by Story 5,
- stable calibration-bundle packaging already owned by Story 6,
- and explicit approximation and deferred-boundary handling already owned by
  Story 7.

## Dependencies And Assumptions

- Stories 1 through 3 already define the candidate surface, workload matrix, and
  density-aware signal surface Story 4 must count honestly.
- The frozen source-of-truth contract is `TASK_5_MINI_SPEC.md`,
  `TASK_5_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-005`,
  `P3-ADR-006`, and `P3-ADR-008`.
- Task 3 already provides the exact internal reference and the core metric helper
  surface Story 4 should reuse through:
  - `execute_sequential_density_reference()` in
    `squander/partitioning/noisy_runtime.py`,
  - `execute_partitioned_with_reference()` in
    `benchmarks/density_matrix/partitioned_runtime/common.py`,
  - `PHASE3_RUNTIME_DENSITY_TOL`,
  - and `PHASE3_RUNTIME_ENERGY_TOL`.
- Task 4 already provides the fused-capable reference path Story 4 should reuse
  when fused-capable calibration runs are counted through:
  - `execute_fused_with_reference()` in
    `benchmarks/density_matrix/partitioned_runtime/common.py`,
  - and the fused validation bundles under
    `benchmarks/density_matrix/artifacts/phase3_task4/`.
- Current Task 5 implementation learning: span-budget candidates above the
  present 2-qubit fused-kernel boundary can still be valid baseline
  partitioned-runtime comparison cases, but they are not automatically valid
  fused-capable positive cases.
- The frozen correctness thresholds remain:
  - maximum Frobenius-norm density difference `<= 1e-10`,
  - `|Tr(rho) - 1| <= 1e-10`,
  - `rho.is_valid(tol=1e-10)`,
  - and continuity-anchor energy error `<= 1e-8`.
- Qiskit Aer remains the required external reference on the frozen 2 to 4 qubit
  microcases and any representative small continuity subset explicitly counted
  in the calibration package.
- Story 4 should prefer reusing the supported runtime and reference surfaces over
  inventing a disconnected calibration-only correctness check.
- Story 4 should therefore gate positive evidence first on the supported
  baseline partitioned-runtime surface and treat fused-capable comparisons as
  optional only where the current fused runtime boundary actually supports the
  candidate.

## Engineering Tasks

### Engineering Task 1: Freeze The Positive Calibration Counting Rule

**Implements story**
- `Story 4: Positive Calibration Evidence Is Counted Only On Correctness-Preserving Supported Runs`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 defines one explicit rule for which calibration runs count as positive
  evidence.
- The rule ties positive counting directly to frozen correctness thresholds and
  the supported runtime surface.
- The rule is explicit enough that later claim-selection and publication work
  can rely on it safely.

**Execution checklist**
- [ ] Freeze the rule that only correctness-preserving supported runs count as
      positive calibration evidence.
- [ ] Define how the frozen sequential-density thresholds gate positive counting.
- [ ] Define where the required Aer baseline applies inside the Task 5 package.
- [ ] Keep final claim selection and final bundle packaging outside the Story 4
      bar.

**Evidence produced**
- One stable Task 5 positive-counting rule.
- One explicit boundary between counted evidence and non-counted exploratory or
  failing runs.

**Risks / rollback**
- Risk: favorable planner results may be counted even when correctness is
  degraded or unclear.
- Rollback/mitigation: freeze the positive-counting rule before broadening
  calibration summaries.

### Engineering Task 2: Reuse The Shared Sequential And Aer Reference Surfaces

**Implements story**
- `Story 4: Positive Calibration Evidence Is Counted Only On Correctness-Preserving Supported Runs`

**Change type**
- docs | code

**Definition of done**
- Story 4 reuses the existing sequential and external reference surfaces where
  the phase contract already requires them.
- The correctness gate stays auditable back to the supported runtime result
  shapes.
- Story 4 avoids inventing a second calibration-only reference language.

**Execution checklist**
- [ ] Reuse `execute_partitioned_with_reference()` and
      `execute_fused_with_reference()` for internal reference comparisons where
      they fit the Story 4 surface.
- [ ] Reuse the existing Task 3 and Task 4 exact-output record shapes for
      correctness checks.
- [ ] Keep Aer comparisons aligned with the frozen microcase and small
      continuity slices.
- [ ] Document which existing reference helpers Story 4 uses directly.

**Evidence produced**
- One reviewable mapping from supported runtime surfaces to the Task 5
  correctness gate.
- One explicit no-second-reference-language rule for Story 4.

**Risks / rollback**
- Risk: calibration correctness may drift into ad hoc comparison logic that
  later reviewers cannot align with earlier tasks.
- Rollback/mitigation: reuse the shared reference surfaces directly.

### Engineering Task 3: Build The Calibration Correctness-Gate Harness

**Implements story**
- `Story 4: Positive Calibration Evidence Is Counted Only On Correctness-Preserving Supported Runs`

**Change type**
- code | validation automation

**Definition of done**
- Story 4 has one reusable harness for evaluating correctness-gated calibration
  runs.
- The harness records pass or fail status for the frozen thresholds together
  with the candidate and workload identity.
- The harness is reusable by later claim-selection and bundle-packaging work.

**Execution checklist**
- [ ] Add a dedicated Story 4 validation driver under
      `benchmarks/density_matrix/planner_calibration/`, with
      `calibration_correctness_validation.py` as the primary checker.
- [ ] Evaluate candidate-workload runs through the supported runtime and
      reference surfaces.
- [ ] Record threshold verdicts together with shared workload and candidate
      provenance.
- [ ] Keep the harness rooted in supported runtime outputs rather than in
      synthetic or manually edited summaries.

**Evidence produced**
- One reusable Story 4 correctness-gate harness.
- One comparable verdict schema for counted and non-counted calibration runs.

**Risks / rollback**
- Risk: correctness gating may remain scattered across scripts and hard to audit
  later.
- Rollback/mitigation: centralize the correctness gate in one stable validation
  entry point.

### Engineering Task 4: Tie Correctness Verdicts Directly To Calibration Records

**Implements story**
- `Story 4: Positive Calibration Evidence Is Counted Only On Correctness-Preserving Supported Runs`

**Change type**
- code | tests

**Definition of done**
- Story 4 calibration records expose correctness verdicts beside the candidate,
  workload, and metric fields they govern.
- Positive counting cannot occur without the required threshold fields being
  present.
- The story avoids post hoc or external interpretation for basic correctness
  status.

**Execution checklist**
- [ ] Add correctness-verdict fields to the shared Task 5 calibration record
      shape or the smallest auditable successor.
- [ ] Record Frobenius-norm difference, trace-validity checks, and continuity
      energy error where applicable.
- [ ] Ensure counted runs cannot omit threshold fields silently.
- [ ] Add focused regression checks for correctness-field presence and pass or
      fail stability.

**Evidence produced**
- One explicit correctness-gated calibration record shape.
- Regression coverage for required correctness-field stability.

**Risks / rollback**
- Risk: positive calibration evidence may look stronger than it is if threshold
  verdicts are not attached directly to the records being counted.
- Rollback/mitigation: record correctness verdicts alongside every counted run.

### Engineering Task 5: Add A Representative Correctness-Gated Calibration Matrix

**Implements story**
- `Story 4: Positive Calibration Evidence Is Counted Only On Correctness-Preserving Supported Runs`

**Change type**
- tests | validation automation

**Definition of done**
- Story 4 covers representative counted and non-counted calibration cases across
  the mandatory workload classes.
- The matrix is broad enough to show the correctness gate is shared across the
  calibration surface.
- The matrix remains representative and contract-driven rather than exhaustive.

**Execution checklist**
- [ ] Add at least one continuity-anchor case to the correctness-gated matrix.
- [ ] Add at least one required microcase with Aer-comparable output.
- [ ] Add at least one structured workload case that exercises the supported
      calibration surface.
- [ ] Include at least one non-counted case or intentionally failing path to
      prove the gate does not count everything positively.

**Evidence produced**
- One representative Story 4 correctness-gated calibration matrix.
- One review surface for counted versus non-counted Task 5 runs.

**Risks / rollback**
- Risk: correctness gating may look coherent on a narrow slice while drifting on
  the broader supported matrix.
- Rollback/mitigation: freeze a small but workload-spanning correctness matrix.

### Engineering Task 6: Emit A Stable Story 4 Correctness-Gate Bundle Or Rerunnable Checker

**Implements story**
- `Story 4: Positive Calibration Evidence Is Counted Only On Correctness-Preserving Supported Runs`

**Change type**
- validation automation | docs

**Definition of done**
- Story 4 emits one stable machine-reviewable correctness-gate bundle or
  rerunnable checker.
- The bundle records counted and non-counted runs with explicit threshold
  verdicts and shared provenance.
- The output shape is stable enough for Stories 5 through 7 to extend.

**Execution checklist**
- [ ] Add a dedicated Story 4 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task5/story4_correctness_gate/`).
- [ ] Emit candidate identity, workload provenance, threshold metrics, pass or
      fail verdicts, and counted-status fields through one stable schema.
- [ ] Record rerun commands and software metadata with the emitted bundle.
- [ ] Keep counted-status derivation explicit in the bundle summary.

**Evidence produced**
- One stable Story 4 correctness-gate bundle or checker.
- One reusable evidence surface for later supported-claim selection.

**Risks / rollback**
- Risk: prose-only correctness gating will make later Task 5 claim selection
  hard to defend and easy to misread.
- Rollback/mitigation: emit one thin machine-reviewable correctness-gate surface
  early.

### Engineering Task 7: Document The Story 4 Counting Rule And Run The Correctness Surface

**Implements story**
- `Story 4: Positive Calibration Evidence Is Counted Only On Correctness-Preserving Supported Runs`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the Story 4 positive-counting rule and its
  threshold dependencies.
- The Story 4 correctness harness and bundle run successfully.
- Story 4 makes clear that supported-claim selection belongs to Story 5.

**Execution checklist**
- [ ] Document the counted-versus-non-counted rule for Task 5 calibration runs.
- [ ] Explain where the sequential-density and Aer baselines apply.
- [ ] Run focused Story 4 regression coverage and verify
      `benchmarks/density_matrix/planner_calibration/calibration_correctness_validation.py`.
- [ ] Record stable references to the Story 4 tests, checker, and emitted
      bundle.

**Evidence produced**
- Passing Story 4 correctness-gate regression checks.
- One stable Story 4 correctness-bundle or checker reference.

**Risks / rollback**
- Risk: later reviewers may mistake correctness-gated counting for a hidden
  claim-selection rule rather than one prerequisite evidence gate.
- Rollback/mitigation: document the handoff to Story 5 explicitly.

## Exit Criteria

Story 4 is complete only when all of the following are true:

- one explicit Task 5 positive-counting rule exists for correctness-preserving
  supported runs,
- counted calibration runs are evaluated through the shared sequential-density
  and Aer reference surfaces where the phase contract requires them,
- correctness verdicts are attached directly to the calibration records they
  govern,
- one stable Story 4 correctness-gate bundle or rerunnable checker exists for
  later reuse,
- and final supported-claim selection, stable final packaging, and explicit
  approximation-boundary handling remain clearly assigned to later stories.

## Implementation Notes

- Prefer explicit counted-status fields over verbal interpretation of whether a
  run "basically passed."
- Keep Story 4 focused on evidence admissibility, not yet on which planner claim
  ultimately closes Task 5.
- Treat failing or non-counted cases as required scientific evidence, not as
  noise to hide.
