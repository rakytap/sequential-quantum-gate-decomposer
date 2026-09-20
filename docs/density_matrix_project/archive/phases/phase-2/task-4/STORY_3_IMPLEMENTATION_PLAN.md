# Story 3 Implementation Plan

## Story Being Implemented

Story 3: Optional Noise Extensions Remain Clearly Optional And Do Not Substitute
For The Mandatory Baseline

This is a Layer 4 engineering plan for implementing the third behavioral slice
from `TASK_4_STORIES.md`.

## Scope

This story turns the frozen required / optional / deferred noise split into an
explicit classification layer across validation, artifacts, and developer-facing
evidence:

- optional whole-register depolarizing and any justified extension cases are
  labeled as optional rather than milestone-defining,
- mandatory required-local-noise results remain identifiable and reviewable
  independently of any optional bundle,
- optional artifacts can exist as regression, stress, or comparison evidence
  without being mistaken for the main scientific Task 4 baseline,
- and the current broader low-level density-noise surface is prevented from
  inflating the guaranteed VQE-facing Phase 2 contract by implication.

Out of scope for this story:

- promoting any optional model into the required baseline,
- implementing generalized amplitude damping or coherent over-rotation unless a
  separately justified benchmark extension requires them,
- unsupported and deferred hard-error closure owned by Story 4,
- the 4/6/8/10 workflow-scale sufficiency package owned by Story 5,
- the final publication bundle owned by Story 6,
- and reopening the frozen support-matrix decision in `P2-ADR-012`.

## Dependencies And Assumptions

- Story 1 is already in place: the required local-noise positive path exists and
  is documented through
  `benchmarks/density_matrix/noise_support/required_local_noise_validation.py`.
- Story 2 is already in place: the mandatory 1 to 3 qubit exact micro-validation
  gate closes through the canonical
  `benchmarks/density_matrix/validate_squander_vs_qiskit.py` path and the Task 4
  Story 2 wrapper
  `benchmarks/density_matrix/noise_support/required_local_noise_micro_validation.py`.
- The frozen required baseline remains
  `local_depolarizing`, `amplitude_damping`, and `phase_damping`; Story 3 must
  not dilute that baseline by merging optional cases into the same completion
  signal.
- The low-level density module already exposes broader optional surfaces such as
  whole-register depolarizing through `NoisyCircuit.add_depolarizing()` and
  `DualBuilder.depolarizing()` in `benchmarks/density_matrix/circuits.py`.
- The frozen support-matrix decision already defines:
  - optional: whole-register depolarizing as a regression or stress-test
    baseline, and generalized amplitude damping or coherent over-rotation only
    when a justified benchmark extension requires them,
  - deferred: correlated multi-qubit noise, readout noise as a density-backend
    feature, calibration-aware noise, and non-Markovian noise.
- Story 3 should harden the classification and evidence boundary without
  requiring that every optional model be implemented immediately.

## Engineering Tasks

### Engineering Task 1: Freeze One Support-Tier Vocabulary For Required, Optional, And Deferred Noise Cases

**Implements story**
- `Story 3: Optional Noise Extensions Remain Clearly Optional And Do Not Substitute For The Mandatory Baseline`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 uses one stable vocabulary for support-tier classification across docs
  and artifacts.
- Required, optional, and deferred are distinguishable in a machine-readable
  way.
- The classification vocabulary is narrow enough to align directly with
  `P2-ADR-004` and `P2-ADR-012`.

**Execution checklist**
- [ ] Define stable support-tier labels for Task 4 artifact and validation use.
- [ ] Distinguish required baseline cases from optional regression or stress
      cases and from deferred or unsupported classes.
- [ ] Keep the vocabulary aligned with the frozen support matrix rather than ad
      hoc benchmark labels.
- [ ] Record the vocabulary in one stable implementation-facing location that
      later Task 4 stories can reuse.

**Evidence produced**
- A stable Task 4 support-tier vocabulary.
- One reviewable mapping from frozen support-matrix decisions to artifact labels.

**Risks / rollback**
- Risk: drifting or inconsistent labels will make optional cases look more
  important than the frozen contract allows.
- Rollback/mitigation: keep one explicit vocabulary tied directly to the phase
  ADRs and reuse it everywhere.

### Engineering Task 2: Add Optional Whole-Register Depolarizing Cases As Explicit Regression Or Stress Evidence

**Implements story**
- `Story 3: Optional Noise Extensions Remain Clearly Optional And Do Not Substitute For The Mandatory Baseline`

**Change type**
- code | benchmark harness

**Definition of done**
- At least one optional whole-register depolarizing case exists as explicit
  optional evidence.
- The case is clearly classified as regression or stress evidence rather than as
  the mandatory scientific baseline.
- The optional case reuses existing low-level density-noise surfaces without
  widening the guaranteed VQE-facing support contract.

**Execution checklist**
- [ ] Reuse `DualBuilder.depolarizing()` or a similarly narrow low-level path to
      define explicit optional whole-register depolarizing cases.
- [ ] Give each optional case a stable case ID and short purpose statement.
- [ ] Keep optional whole-register cases separate from the mandatory required
      local-noise matrix.
- [ ] Avoid implying that the optional case is needed to close Story 2 or Story
      5.

**Evidence produced**
- Stable optional whole-register depolarizing regression or stress cases.
- Clear case metadata showing that the optional cases do not define Task 4
  completion.

**Risks / rollback**
- Risk: if optional whole-register cases are inserted into the wrong bundle or
  summary, they can quietly substitute for the required local baseline.
- Rollback/mitigation: keep optional cases in separately typed outputs and label
  them clearly at definition time.

### Engineering Task 3: Preserve Mandatory-Baseline Independence In Validation Summaries And Bundles

**Implements story**
- `Story 3: Optional Noise Extensions Remain Clearly Optional And Do Not Substitute For The Mandatory Baseline`

**Change type**
- code | validation automation

**Definition of done**
- Validation summaries make it impossible to confuse optional evidence with
  required-baseline completion.
- Required-bundle pass/fail status is computed independently of optional cases.
- Optional results can be present without changing the mandatory Story 1 or
  Story 2 completion signal.

**Execution checklist**
- [ ] Add summary fields that separately count required and optional cases.
- [ ] Ensure required completion stays tied only to the mandatory local-noise
      bundles and their pass rates.
- [ ] Keep optional bundle status reviewable without letting it satisfy the
      required baseline gate.
- [ ] Reuse the current Task 4 Story 1 and Story 2 bundle patterns rather than
      inventing a wholly separate summary scheme.

**Evidence produced**
- Bundle summaries that distinguish required completion from optional evidence.
- Stable pass/fail semantics that cannot be satisfied by optional-only results.

**Risks / rollback**
- Risk: a single aggregate pass rate can blur the difference between mandatory
  baseline closure and optional comparison coverage.
- Rollback/mitigation: keep required and optional counts explicit and compute the
  completion gate from required cases only.

### Engineering Task 4: Add Task 4 Artifact Fields That Explain Why A Case Exists

**Implements story**
- `Story 3: Optional Noise Extensions Remain Clearly Optional And Do Not Substitute For The Mandatory Baseline`

**Change type**
- code | validation automation

**Definition of done**
- Task 4 artifacts explain whether a case is required, optional regression,
  optional stress, optional comparison, deferred, or unsupported.
- Reviewers can tell why an optional case was run and whether it counts toward
  milestone completion.
- Artifact fields remain simple enough to reuse in later Task 4 provenance work.

**Execution checklist**
- [ ] Extend Task 4 bundle schemas with support-tier and case-purpose fields.
- [ ] Record whether each case counts toward the mandatory Task 4 completion
      gate.
- [ ] Keep field names stable and auditable across Story 1, Story 2, and Story 3
      outputs where practical.
- [ ] Avoid overloading numeric exactness fields with classification semantics.

**Evidence produced**
- Machine-readable Task 4 classification fields on required and optional cases.
- Stable artifact semantics describing whether a case is milestone-defining.

**Risks / rollback**
- Risk: reviewers may misread optional cases if purpose and support tier are
  implicit rather than encoded.
- Rollback/mitigation: include simple explicit case-purpose fields instead of
  relying on filenames or comments alone.

### Engineering Task 5: Add Focused Regression Tests For Classification Semantics

**Implements story**
- `Story 3: Optional Noise Extensions Remain Clearly Optional And Do Not Substitute For The Mandatory Baseline`

**Change type**
- tests

**Definition of done**
- Fast automated tests prove that optional cases are labeled optional and do not
  satisfy the mandatory Task 4 baseline gate.
- Regression coverage is specific enough to catch classification drift without
  running the full publication bundle.
- Tests make clear that whole-register depolarizing is optional evidence, not a
  replacement for required local-noise coverage.

**Execution checklist**
- [ ] Add focused tests for support-tier and counts-toward-completion semantics.
- [ ] Assert that optional whole-register depolarizing cases remain outside the
      mandatory pass/fail gate.
- [ ] Reuse current Task 4 bundle builders where possible rather than duplicating
      artifact assembly logic.
- [ ] Keep the test layer narrow to classification semantics and not the full
      workflow package.

**Evidence produced**
- Focused pytest coverage for Task 4 optional-versus-required classification.
- Reviewable failures that localize bundle or summary drift cleanly.

**Risks / rollback**
- Risk: artifact labels can drift silently if there is no regression coverage for
  the classification layer itself.
- Rollback/mitigation: add direct tests for summary semantics and per-case
  support-tier fields.

### Engineering Task 6: Represent Unimplemented Optional Extensions As Explicitly Non-Mandatory Slots

**Implements story**
- `Story 3: Optional Noise Extensions Remain Clearly Optional And Do Not Substitute For The Mandatory Baseline`

**Change type**
- docs | validation automation

**Definition of done**
- Optional extensions such as generalized amplitude damping or coherent
  over-rotation do not appear as accidental gaps in required coverage.
- When they are not implemented, their absence is documented as optional and
  non-blocking rather than as a hidden failure to complete the required
  baseline.
- If later introduced, they have a reserved classification path that keeps them
  optional by default.

**Execution checklist**
- [ ] Record in one stable location that optional extension absence is not a Task
      4 completion failure.
- [ ] Keep any placeholder or reserved artifact language clearly separated from
      required-case results.
- [ ] Avoid turning “not yet exercised” optional extensions into implied
      unsupported-case failures owned by Story 4.
- [ ] Make the promotion path explicit: optional becomes required only through a
      new phase-level decision.

**Evidence produced**
- Clear documentation and artifact semantics for unimplemented optional
  extensions.
- A stable path for later optional extension adoption without scope confusion.

**Risks / rollback**
- Risk: optional but unimplemented cases can be misread as missing mandatory
  coverage.
- Rollback/mitigation: document them explicitly as optional, non-blocking slots
  with no impact on required completion.

### Engineering Task 7: Update Developer- And Paper-Facing Notes For Optional Baselines

**Implements story**
- `Story 3: Optional Noise Extensions Remain Clearly Optional And Do Not Substitute For The Mandatory Baseline`

**Change type**
- docs

**Definition of done**
- Developer-facing and paper-facing notes make the required-versus-optional split
  explicit.
- The notes explain that whole-register depolarizing is allowed as regression or
  stress evidence, but not as the main scientific Phase 2 baseline.
- The documentation stays aligned with the frozen support matrix and does not
  imply that optional breadth has already been promoted into required support.

**Execution checklist**
- [ ] Update the most relevant Task 4-facing notes or bundle metadata docs.
- [ ] Make the mandatory local-noise baseline explicit beside the optional
      whole-register baseline rule.
- [ ] Explain how Story 3 builds on the already passing Story 1 and Story 2
      evidence layers.
- [ ] Keep deferred and unsupported families clearly distinct from optional
      baseline cases.

**Evidence produced**
- Updated Task 4 notes for optional-versus-required classification.
- One stable location where reviewers can verify the optional baseline rule.

**Risks / rollback**
- Risk: if docs lag the artifact semantics, reviewers may still overread optional
  evidence as milestone-defining.
- Rollback/mitigation: tie the notes directly to the same support-tier fields
  used in machine-readable bundles.

### Engineering Task 8: Run Story 3 Validation And Confirm Optional Cases Cannot Close The Mandatory Baseline

**Implements story**
- `Story 3: Optional Noise Extensions Remain Clearly Optional And Do Not Substitute For The Mandatory Baseline`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 3 classification tests pass.
- Required and optional bundles or summaries are generated in a way that keeps
  the mandatory baseline independently reviewable.
- Story 3 completion is backed by explicit artifact semantics rather than code
  changes alone.

**Execution checklist**
- [ ] Run the focused Story 3 classification tests.
- [ ] Run the relevant required and optional Task 4 validation commands that emit
      classified bundles.
- [ ] Verify that optional cases never satisfy the required Task 4 completion
      gate.
- [ ] Record stable test and artifact references for later Task 4 docs and
      publication bundles.

**Evidence produced**
- Passing focused Story 3 regression coverage.
- Stable required and optional Task 4 artifact references proving the mandatory
  baseline is independent.

**Risks / rollback**
- Risk: Story 3 can appear complete while still lacking auditable proof that
  optional evidence cannot substitute for the mandatory baseline.
- Rollback/mitigation: treat classified bundle outputs and focused summary tests
  as part of the exit gate, not optional cleanup.

## Exit Criteria

Story 3 is complete only when all of the following are true:

- required, optional, and deferred noise cases are classified with one stable
  Task 4 vocabulary,
- optional whole-register depolarizing cases, if present, are explicitly marked
  optional and do not count toward mandatory Task 4 completion,
- required local-noise baseline results remain reviewable independently of any
  optional cases,
- optional extension absence does not appear as a required-baseline failure,
- and validation summaries and artifacts make it impossible to mistake optional
  evidence for milestone-defining evidence.

## Implementation Notes

- `NoisyCircuit.add_depolarizing()` and `DualBuilder.depolarizing()` already show
  that the low-level density module can express a broader optional
  whole-register baseline than the frozen required VQE-facing contract. Story 3
  should classify that breadth rather than suppress or overclaim it.
- `required_local_noise_validation_validation.py` and
  `required_local_noise_micro_validation.py` already provide the
  required-baseline evidence layers. Story 3 should preserve their semantics and
  add classification on top rather than redesigning them.
- `validate_squander_vs_qiskit.py` and `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` are the
  natural places to keep bundle and summary semantics aligned with the broader
  workflow.
- Optional generalized amplitude damping or coherent over-rotation should remain
  clearly optional by default even if later benchmark extensions add them.
- Story 4 should handle unsupported and deferred hard-error behavior; Story 3
  should not conflate “optional and not milestone-defining” with “unsupported
  and must hard-fail.”
