# Story 3 Implementation Plan

## Story Being Implemented

Story 3: The Mandatory Qiskit Aer Slice Remains Explicit And Bounded

This is a Layer 4 engineering plan for implementing the third behavioral slice
from `TASK_6_STORIES.md`.

## Scope

This story turns Task 6 into one bounded external-exactness surface:

- the required 2 to 4 qubit microcases and representative small continuity
  subset are compared against Qiskit Aer through one explicit external slice,
- external exactness remains structurally aligned with the shared Task 6 case
  identity and internal-correctness surfaces,
- optional broader simulator comparisons do not redefine the Task 6 closure bar,
- and Story 3 closes the contract for "which cases require Qiskit Aer and how
  they are compared" without yet claiming full-package assembly or downstream
  summary semantics.

Out of scope for this story:

- correctness-matrix inventory already owned by Story 1,
- internal sequential-baseline gating already owned by Story 2,
- density-validity and continuity-energy emphasis already owned by Story 4,
- runtime and fusion classification comparability already owned by Story 5,
- unsupported-boundary stage separation already owned by Story 6,
- full correctness-package assembly already owned by Story 7,
- and counted-status carry-forward into benchmark or paper summaries already
  owned by Story 8.

## Dependencies And Assumptions

- Stories 1 and 2 already define the mandatory case matrix and the internal
  exactness gate Story 3 must extend rather than replace.
- The frozen source-of-truth contract is `TASK_6_MINI_SPEC.md`,
  `TASK_6_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-008`, and
  `P3-ADR-009`.
- The required external slice remains:
  - all mandatory 2 to 4 qubit microcases,
  - plus the representative small continuity subset frozen by the Phase 3
    contract.
- The current benchmark stack already contains Qiskit comparison precedent Story
  3 should reuse where practical:
  - `benchmarks/density_matrix/validate_squander_vs_qiskit.py`,
  - the external-metric handling in
    `benchmarks/density_matrix/planner_calibration/calibration_records.py`,
  - and the Phase 2 micro-validation patterns under
    `benchmarks/density_matrix/noise_support/required_local_noise_micro_validation.py`.
- Story 3 should keep external exactness rooted in the same exact-output shapes
  used by Story 2 rather than inventing a separate external-only result surface.
- The current implementation learning is that Story 3 should reuse the shared
  Task 6 positive records from `records.py` and expose the bounded Aer slice as
  a filtered validator view rather than as a second standalone case-selection
  path.
- External comparison remains a required bounded slice, not a blanket
  requirement on every mandatory Task 6 case.
- Optional secondary baselines may exist later, but they are explicitly out of
  scope for Story 3 closure.
- The natural implementation home for Task 6 external-slice validators is the
  new `benchmarks/density_matrix/correctness_evidence/` package, with
  `external_correctness_validation.py` operating on the shared positive record
  surface.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 6 External-Slice Inventory And Rule

**Implements story**
- `Story 3: The Mandatory Qiskit Aer Slice Remains Explicit And Bounded`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 defines one explicit external-slice inventory for Task 6.
- The inventory is explicit enough that later validators and paper packaging can
  tell which cases require Qiskit Aer and which do not.
- The story distinguishes the required external slice from optional broader
  simulator comparisons.

**Execution checklist**
- [ ] Freeze the required external slice around all mandatory 2 to 4 qubit
      microcases and the representative small continuity subset.
- [ ] Define how external-slice membership is recorded on shared Task 6 case
      records.
- [ ] Define the rule that optional secondary comparisons do not enlarge the
      required Task 6 closure surface.
- [ ] Keep full-package assembly and summary-consistency semantics outside the
      Story 3 bar.

**Evidence produced**
- One stable Task 6 external-slice inventory.
- One explicit rule separating required external evidence from optional
  secondary comparisons.

**Risks / rollback**
- Risk: later consumers may treat ad hoc Aer comparisons as the required Task 6
  slice if Story 3 leaves membership implicit.
- Rollback/mitigation: freeze one external-slice rule before broadening external
  runs.

### Engineering Task 2: Reuse The Shared Aer Comparison Helpers And Exact-Output Surfaces

**Implements story**
- `Story 3: The Mandatory Qiskit Aer Slice Remains Explicit And Bounded`

**Change type**
- docs | code

**Definition of done**
- Story 3 reuses the existing Qiskit comparison helpers and Task 6 exact-output
  shapes where they already match the contract.
- External exactness remains auditable back to the same supported runtime
  outputs used by Story 2.
- Story 3 avoids inventing a second external-only result language.

**Execution checklist**
- [ ] Reuse existing Aer comparison helpers or the smallest auditable successor
      rooted in `benchmarks/density_matrix/validate_squander_vs_qiskit.py`.
- [ ] Reuse the exact-output records and case identity emitted by Stories 1 and
      2 where they overlap with Story 3 needs.
- [ ] Keep external comparison metrics aligned with the shared Phase 3 tolerance
      vocabulary.
- [ ] Document which existing Qiskit comparison surfaces Story 3 consumes
      directly.

**Evidence produced**
- One reviewable mapping from shared Task 6 outputs to the Story 3 Aer gate.
- One explicit no-second-external-result-language rule for Story 3.

**Risks / rollback**
- Risk: Story 3 may drift into ad hoc Aer comparison logic that later reviewers
  cannot reconcile with the counted supported outputs.
- Rollback/mitigation: reuse the existing comparison helpers and exact-output
  surfaces directly.

### Engineering Task 3: Build The Task 6 External Correctness Validation Harness

**Implements story**
- `Story 3: The Mandatory Qiskit Aer Slice Remains Explicit And Bounded`

**Change type**
- code | validation automation

**Definition of done**
- Story 3 has one reusable harness for evaluating Qiskit Aer agreement on the
  required external slice.
- The harness records external metrics and pass/fail verdicts beside the shared
  Task 6 case identity.
- The harness is reusable by Story 7 and later publication packaging.

**Execution checklist**
- [ ] Add a dedicated Story 3 validation driver under
      `benchmarks/density_matrix/correctness_evidence/`, with
      `external_correctness_validation.py` as the primary checker.
- [ ] Evaluate the required external slice through Qiskit Aer and the supported
      partitioned or fused runtime outputs.
- [ ] Record Frobenius and maximum-absolute-difference metrics plus external
      pass/fail verdicts for required cases.
- [ ] Keep the harness bounded to the frozen external slice rather than growing
      into a broad multi-framework comparison tool.

**Evidence produced**
- One reusable Story 3 external-correctness harness.
- One comparable verdict schema for required external Task 6 cases.

**Risks / rollback**
- Risk: external exactness logic may remain scattered across experiments and
  become hard to audit later.
- Rollback/mitigation: centralize the Story 3 gate in one stable validation
  entry point.

### Engineering Task 4: Tie External Exactness Verdicts Directly To Shared Task 6 Records

**Implements story**
- `Story 3: The Mandatory Qiskit Aer Slice Remains Explicit And Bounded`

**Change type**
- code | tests

**Definition of done**
- External exactness verdicts are attached directly to the Task 6 case records
  they govern.
- Required external cases cannot count as externally validated without the
  necessary external metric fields.
- Story 3 avoids post hoc interpretation for basic external pass/fail status.

**Execution checklist**
- [ ] Add external-slice membership, external metrics, and external verdict fields
      to the shared Task 6 correctness record or the smallest auditable
      successor.
- [ ] Ensure required external cases cannot omit external fields silently.
- [ ] Keep non-external cases clearly labeled as out of the required Aer slice
      rather than as failed Aer cases.
- [ ] Add focused regression checks for external-field presence and pass/fail
      stability.

**Evidence produced**
- One explicit external-exactness extension of the Task 6 record shape.
- Regression coverage for required external-field stability.

**Risks / rollback**
- Risk: later summaries may overstate external validation if Story 3 does not
  distinguish required external cases from internal-only cases explicitly.
- Rollback/mitigation: attach external verdicts directly to the shared record.

### Engineering Task 5: Add A Representative Bounded External-Validation Matrix

**Implements story**
- `Story 3: The Mandatory Qiskit Aer Slice Remains Explicit And Bounded`

**Change type**
- tests | validation automation

**Definition of done**
- Story 3 covers representative external-validation cases across the required
  microcase and small continuity slices.
- The matrix is broad enough to show the bounded external slice is real and
  shared.
- The matrix remains representative and contract-driven rather than exhaustive
  over every possible Aer-comparable workload.

**Execution checklist**
- [ ] Include at least one required microcase stressing partition boundaries.
- [ ] Include at least one required microcase stressing explicit noise
      placement.
- [ ] Include at least one representative small continuity case in the external
      slice.
- [ ] Keep broader optional external comparisons outside the required Story 3
      matrix.

**Evidence produced**
- One representative Story 3 bounded external-validation matrix.
- One review surface for required external-slice stability.

**Risks / rollback**
- Risk: Story 3 may appear bounded in prose but drift toward ad hoc external
  comparisons in practice.
- Rollback/mitigation: freeze a small but representative external matrix early.

### Engineering Task 6: Emit A Stable Story 3 External-Slice Bundle Or Rerunnable Checker

**Implements story**
- `Story 3: The Mandatory Qiskit Aer Slice Remains Explicit And Bounded`

**Change type**
- validation automation | docs

**Definition of done**
- Story 3 emits one stable machine-reviewable external-slice bundle or
  rerunnable checker.
- The bundle records required external cases, external metrics, and pass/fail
  semantics through one stable schema.
- The output is stable enough for Story 7 package assembly and later paper
  packaging.

**Execution checklist**
- [ ] Add a dedicated Story 3 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task6/story3_external_slice/`).
- [ ] Emit required external cases, case identity, exact-output references,
      external metrics, and verdicts through one stable schema.
- [ ] Record rerun commands and software metadata with the emitted bundle.
- [ ] Keep the required-slice summary explicit in the bundle output.

**Evidence produced**
- One stable Story 3 external-slice bundle or checker.
- One reusable external-validation surface for later Task 6 stories.

**Risks / rollback**
- Risk: prose-only Story 3 closure will make the bounded external slice hard to
  verify later.
- Rollback/mitigation: emit one machine-reviewable external bundle and cite it
  directly.

### Engineering Task 7: Document The Story 3 Bounded External Rule And Run The Gate

**Implements story**
- `Story 3: The Mandatory Qiskit Aer Slice Remains Explicit And Bounded`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain which Task 6 cases require Qiskit Aer and why.
- The Story 3 external-correctness harness and bundle run successfully.
- Story 3 makes clear that optional secondary baselines remain outside the
  closure bar.

**Execution checklist**
- [ ] Document the required Task 6 external slice and its bounded scope.
- [ ] Explain how internal-only cases differ from required external cases.
- [ ] Run focused Story 3 regression coverage and verify
      `benchmarks/density_matrix/correctness_evidence/external_correctness_validation.py`.
- [ ] Record stable references to the Story 3 tests, checker, and emitted
      bundle.

**Evidence produced**
- Passing Story 3 external-correctness regression checks.
- One stable Story 3 external-slice bundle or checker reference.

**Risks / rollback**
- Risk: later reviewers may mistake optional Aer experiments for the required
  Task 6 closure bar.
- Rollback/mitigation: document the bounded external rule explicitly and require
  a rerunnable checker.

## Exit Criteria

Story 3 is complete only when all of the following are true:

- one explicit Task 6 external-slice inventory exists for the required
  microcases and representative small continuity cases,
- required external cases are evaluated through one bounded Qiskit Aer surface
  aligned with the shared Task 6 case identity,
- external verdicts are attached directly to the records they govern,
- one stable Story 3 external-slice bundle or rerunnable checker exists for
  later reuse,
- and optional broader simulator comparisons, full-package assembly, and
  summary-consistency guardrails remain clearly assigned to later stories.

## Implementation Notes

- Prefer one bounded external slice over a broad but weakly interpretable
  simulator bake-off.
- Keep Story 3 focused on required external exactness, not on optional secondary
  baselines.
- Treat external-slice membership as a first-class field, not as later script
  inference from qubit count alone.
