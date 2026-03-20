# Story 4 Implementation Plan

## Story Being Implemented

Story 4: Density Validity And Continuity-Anchor Energy Agreement Stay
First-Class

This is a Layer 4 engineering plan for implementing the fourth behavioral slice
from `TASK_6_STORIES.md`.

## Scope

This story turns Task 6 into one output-integrity and continuity-anchor verdict
surface:

- trace preservation and density validity remain visible parts of the Task 6
  correctness verdict rather than hidden derived checks,
- the required 4 / 6 / 8 / 10 qubit continuity-anchor cases carry explicit
  energy-agreement verdicts,
- output-integrity fields remain attached directly to the same case records used
  by the internal and external exactness stories,
- and Story 4 closes the contract for "which validity and continuity-anchor
  checks are first-class" without yet claiming runtime classification,
  unsupported-boundary closure, or summary-consistency.

Out of scope for this story:

- correctness-matrix inventory already owned by Story 1,
- internal sequential-baseline gating already owned by Story 2,
- the bounded Qiskit Aer slice already owned by Story 3,
- runtime and fusion classification comparability already owned by Story 5,
- unsupported-boundary stage separation already owned by Story 6,
- full correctness-package assembly already owned by Story 7,
- and counted-status propagation into later summaries already owned by Story 8.

## Dependencies And Assumptions

- Stories 1 through 3 already define the mandatory case matrix and the two-
  baseline exactness surfaces Story 4 must enrich rather than replace.
- The frozen source-of-truth contract is `TASK_6_MINI_SPEC.md`,
  `TASK_6_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, and `P3-ADR-008`.
- The frozen output-integrity and continuity thresholds remain:
  - `|Tr(rho) - 1| <= 1e-10`,
  - `rho.is_valid(tol=1e-10)`,
  - and maximum absolute energy error `<= 1e-8` on the required 4 / 6 / 8 /
    10 qubit continuity-anchor cases.
- Existing runtime validators already provide precedent Story 4 should reuse
  where practical:
  - `benchmarks/density_matrix/partitioned_runtime/runtime_output_validation.py`,
  - `benchmarks/density_matrix/partitioned_runtime/continuity_runtime_validation.py`,
  - `PHASE3_RUNTIME_ENERGY_TOL` in
    `benchmarks/density_matrix/partitioned_runtime/common.py`,
  - and the continuity metric handling in
    `benchmarks/density_matrix/planner_calibration/calibration_records.py`.
- Story 4 should attach validity and energy verdicts to the same shared Task 6
  records used by Stories 1 through 3 rather than creating a detached
  continuity-only reporting path.
- The current implementation learning is that Story 2 already populates the
  underlying trace-validity and density-validity fields on the shared positive
  records, so Story 4 should elevate and summarize those fields while adding the
  continuity-energy verdict rather than duplicating a second internal metric
  layer.
- Story 4 should treat trace-validity and density-validity as required semantic
  evidence, not as debug-only derived checks.
- The natural implementation home for Task 6 validity and continuity validators
  is the new `benchmarks/density_matrix/correctness_evidence/` package, with
  `output_integrity_validation.py` reading the same shared positive records used
  by Stories 2 and 3.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 6 Output-Integrity And Continuity-Verdict Rule

**Implements story**
- `Story 4: Density Validity And Continuity-Anchor Energy Agreement Stay First-Class`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 defines one explicit rule for how trace-validity, density-validity,
  and continuity-anchor energy agreement participate in Task 6 closure.
- The rule is explicit enough that later benchmark and publication work can
  reuse it safely.
- Story 4 distinguishes first-class output-integrity closure from later summary
  semantics.

**Execution checklist**
- [ ] Freeze the rule that trace-validity and density-validity checks are
      required parts of the counted correctness verdict.
- [ ] Freeze the rule that required 4 / 6 / 8 / 10 qubit continuity-anchor
      cases carry explicit energy-agreement verdicts.
- [ ] Define how these verdicts attach to the shared Task 6 record surface.
- [ ] Keep runtime classification, unsupported-boundary closure, and summary
      semantics outside the Story 4 bar.

**Evidence produced**
- One stable Task 6 output-integrity and continuity-verdict rule.
- One explicit boundary between first-class output-integrity checks and later
  summary interpretation.

**Risks / rollback**
- Risk: later summaries may reduce correctness to one density-difference metric
  and hide required validity or continuity semantics.
- Rollback/mitigation: freeze the Story 4 verdict rule before broadening package
  assembly.

### Engineering Task 2: Reuse The Shared Runtime Output And Continuity Validation Surfaces

**Implements story**
- `Story 4: Density Validity And Continuity-Anchor Energy Agreement Stay First-Class`

**Change type**
- docs | code

**Definition of done**
- Story 4 reuses the existing runtime-output and continuity validation surfaces
  wherever they already fit the contract.
- Output-integrity checks remain auditable back to the same supported runtime
  outputs used by Stories 2 and 3.
- Story 4 avoids inventing a detached continuity-only result language.

**Execution checklist**
- [ ] Reuse `runtime_output_validation.py` and
      `continuity_runtime_validation.py` as the base precedent for required
      output-integrity and continuity-energy checks.
- [ ] Reuse the shared exact-output records from Stories 2 and 3 where they
      already match Story 4 needs.
- [ ] Reuse `PHASE3_RUNTIME_ENERGY_TOL` and the shared density-validity
      vocabulary directly where it fits.
- [ ] Document which existing validation surfaces Story 4 consumes directly.

**Evidence produced**
- One reviewable mapping from existing runtime-output surfaces to the Story 4
  integrity gate.
- One explicit no-detached-continuity-language rule for Story 4.

**Risks / rollback**
- Risk: Story 4 may drift into one-off continuity scripts that later reviewers
  cannot align with the counted Task 6 records.
- Rollback/mitigation: reuse the shared runtime-output and continuity helpers
  directly.

### Engineering Task 3: Build The Task 6 Output-Integrity And Continuity Harness

**Implements story**
- `Story 4: Density Validity And Continuity-Anchor Energy Agreement Stay First-Class`

**Change type**
- code | validation automation

**Definition of done**
- Story 4 has one reusable harness for evaluating trace-validity, density-
  validity, and continuity-anchor energy agreement.
- The harness records those metrics and verdicts beside the shared Task 6 case
  identity.
- The harness is reusable by Story 7 package assembly and later publication
  packaging.

**Execution checklist**
- [ ] Add a dedicated Story 4 validation driver under
      `benchmarks/density_matrix/correctness_evidence/`, with
      `output_integrity_validation.py` as the primary checker.
- [ ] Evaluate required continuity-anchor cases for explicit energy agreement
      and all counted supported outputs for trace-validity and density-validity.
- [ ] Record the relevant metrics and pass/fail verdicts beside the shared Story
      1 case identity.
- [ ] Keep the harness rooted in supported runtime outputs rather than in
      summary-only tables.

**Evidence produced**
- One reusable Story 4 output-integrity harness.
- One comparable verdict schema for trace-validity, density-validity, and
  continuity-energy checks.

**Risks / rollback**
- Risk: validity and continuity checks may remain scattered or become optional in
  practice.
- Rollback/mitigation: centralize Story 4 in one stable validation entry point.

### Engineering Task 4: Attach Validity And Continuity Verdicts Directly To Shared Task 6 Records

**Implements story**
- `Story 4: Density Validity And Continuity-Anchor Energy Agreement Stay First-Class`

**Change type**
- code | tests

**Definition of done**
- Story 4 verdicts attach directly to the Task 6 records they govern.
- Counted supported outputs cannot omit trace-validity or density-validity
  fields silently.
- Required continuity-anchor cases cannot omit explicit energy-agreement fields.

**Execution checklist**
- [ ] Add trace-deviation, density-validity status, and continuity-energy fields
      to the shared Task 6 record or the smallest auditable successor.
- [ ] Ensure non-continuity cases remain clearly labeled as out of the
      continuity-energy slice rather than as failed energy cases.
- [ ] Ensure counted supported records cannot omit trace-validity or
      density-validity fields silently.
- [ ] Add focused regression checks for field presence and verdict stability.

**Evidence produced**
- One explicit Story 4 record extension for output-integrity and continuity
  verdicts.
- Regression coverage for required field stability.

**Risks / rollback**
- Risk: later package consumers may treat validity and energy semantics as
  optional if Story 4 does not attach them directly to shared case records.
- Rollback/mitigation: record validity and continuity verdicts beside every
  relevant case.

### Engineering Task 5: Add A Representative Output-Integrity And Continuity Matrix

**Implements story**
- `Story 4: Density Validity And Continuity-Anchor Energy Agreement Stay First-Class`

**Change type**
- tests | validation automation

**Definition of done**
- Story 4 covers representative counted supported outputs across required
  continuity-anchor and non-continuity slices.
- The matrix is broad enough to show that output-integrity closure is not
  limited to one workload family.
- The matrix remains representative and contract-driven rather than exhaustive.

**Execution checklist**
- [ ] Include at least one required microcase proving trace-validity and
      density-validity are attached outside the continuity slice.
- [ ] Include required 4 / 6 / 8 / 10 qubit continuity-anchor cases or a
      representative subset sufficient to prove explicit energy-verdict wiring.
- [ ] Include at least one structured-family case proving Story 4 fields remain
      present outside the continuity anchor.
- [ ] Keep runtime classification and unsupported-boundary closure outside the
      Story 4 matrix.

**Evidence produced**
- One representative Story 4 output-integrity and continuity matrix.
- One review surface for field presence across the mandatory slices.

**Risks / rollback**
- Risk: Story 4 may appear complete on continuity cases while drifting on the
  rest of the counted supported matrix.
- Rollback/mitigation: freeze a small but slice-spanning integrity matrix.

### Engineering Task 6: Emit A Stable Story 4 Output-Integrity Bundle Or Rerunnable Checker

**Implements story**
- `Story 4: Density Validity And Continuity-Anchor Energy Agreement Stay First-Class`

**Change type**
- validation automation | docs

**Definition of done**
- Story 4 emits one stable machine-reviewable output-integrity bundle or
  rerunnable checker.
- The bundle records trace-validity, density-validity, and continuity-energy
  verdicts through one stable schema.
- The output is stable enough for Story 7 package assembly and Story 8 summary
  guardrails.

**Execution checklist**
- [ ] Add a dedicated Story 4 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task6/story4_output_integrity/`).
- [ ] Emit shared case identity, validity metrics, energy metrics where
      required, and verdict fields through one stable schema.
- [ ] Record rerun commands and software metadata with the emitted bundle.
- [ ] Keep required-versus-not-applicable semantics explicit in the bundle
      summary.

**Evidence produced**
- One stable Story 4 output-integrity bundle or checker.
- One reusable validity-and-continuity evidence surface for later Task 6
  stories.

**Risks / rollback**
- Risk: prose-only Story 4 closure will make later reviewers unable to tell
  whether validity and continuity semantics were actually enforced.
- Rollback/mitigation: emit one machine-reviewable integrity bundle directly.

### Engineering Task 7: Document The Story 4 Integrity Rule And Run The Gate

**Implements story**
- `Story 4: Density Validity And Continuity-Anchor Energy Agreement Stay First-Class`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain how trace-validity, density-validity, and
  continuity-energy verdicts participate in Task 6 closure.
- The Story 4 integrity harness and bundle run successfully.
- Story 4 makes clear that runtime classification, unsupported-boundary
  semantics, and summary-consistency remain assigned to later stories.

**Execution checklist**
- [ ] Document the Story 4 output-integrity rule and its required threshold
      fields.
- [ ] Explain which Task 6 cases require explicit continuity-energy verdicts.
- [ ] Run focused Story 4 regression coverage and verify
      `benchmarks/density_matrix/correctness_evidence/output_integrity_validation.py`.
- [ ] Record stable references to the Story 4 tests, checker, and emitted
      bundle.

**Evidence produced**
- Passing Story 4 output-integrity regression checks.
- One stable Story 4 output-integrity bundle or checker reference.

**Risks / rollback**
- Risk: later reviewers may treat continuity energy agreement as a paper-only
  claim instead of a machine-checked Task 6 requirement.
- Rollback/mitigation: document and rerun the Story 4 integrity gate explicitly.

## Exit Criteria

Story 4 is complete only when all of the following are true:

- one explicit Task 6 rule exists for trace-validity, density-validity, and
  continuity-anchor energy agreement,
- those verdicts are attached directly to the shared Task 6 records they govern,
- required continuity-anchor cases carry explicit machine-reviewable energy-
  agreement verdicts,
- one stable Story 4 output-integrity bundle or rerunnable checker exists for
  later reuse,
- and runtime classification, unsupported-boundary closure, full-package
  assembly, and summary-consistency guardrails remain clearly assigned to later
  stories.

## Implementation Notes

- Prefer explicit "not applicable" semantics for non-continuity cases over
  leaving energy fields absent without explanation.
- Keep Story 4 focused on first-class output integrity, not yet on runtime-path
  interpretation.
- Treat trace-validity and density-validity as required semantic evidence, not
  as debug-only derived checks.
