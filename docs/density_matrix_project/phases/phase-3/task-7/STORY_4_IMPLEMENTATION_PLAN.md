# Story 4 Implementation Plan

## Story Being Implemented

Story 4: Sensitivity Over Planner Settings, Noise Placement, And Workload
Identity Stays First-Class

This is a Layer 4 engineering plan for implementing the fourth behavioral slice
from `TASK_7_STORIES.md`.

## Scope

This story turns Task 7 into one explicit sensitivity-analysis surface:

- benchmark behavior remains visible across the bounded planner-setting surface,
  sparse / periodic / dense noise placement, and workload-identity knobs rather
  than being averaged into one opaque result,
- the required sensitivity dimensions are recorded through one stable
  machine-reviewable surface,
- close or rerun-sensitive outcomes inside the bounded Task 5 planner-setting
  family remain explicit,
- and Story 4 closes the contract for "how Task 7 records sensitivity" without
  yet taking ownership of the shared metric schema, diagnosis-path bottleneck
  reporting, or final benchmark-package assembly.

Out of scope for this story:

- benchmark-matrix inventory already owned by Story 1,
- counted-supported benchmark eligibility already owned by Story 2,
- explicit positive-threshold interpretation already owned by Story 3,
- the shared comparable metric surface already owned by Story 5,
- diagnosis-path bottleneck reporting already owned by Story 6,
- shared benchmark-package assembly already owned by Story 7,
- and summary-consistency plus bounded-claim guardrails already owned by Story
  8.

## Dependencies And Assumptions

- Stories 1 through 3 already define the benchmark matrix, counted-supported
  gate, and positive-threshold review set Story 4 must interpret consistently
  rather than replace.
- The frozen source-of-truth contract is `TASK_7_MINI_SPEC.md`,
  `TASK_7_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-006`, and
  `P3-ADR-009`.
- Task 5 already emits the bounded planner-setting surface Story 4 should reuse
  directly through:
  - `benchmarks/density_matrix/planner_calibration/claim_selection.py`,
  - `benchmarks/density_matrix/planner_calibration/calibrated_claim_selection_validation.py`,
  - and `benchmarks/density_matrix/artifacts/phase3_task5/`.
- The structured workload builders and noise-pattern vocabulary Story 4 should
  reuse already exist in
  `benchmarks/density_matrix/planner_surface/workloads.py`, especially
  `MANDATORY_NOISE_PATTERNS`, `STRUCTURED_FAMILY_NAMES`, and
  `STRUCTURED_QUBITS`.
- Story 4 should not widen the Task 5 planner-claim surface. It should interpret
  sensitivity only inside the bounded supported planner-setting family.
- The current implementation learning is that a rerun-sensitive winner inside
  the bounded Task 5 family should remain explicit benchmark evidence rather
  than being hidden behind one permanently frozen universal setting.
- Story 4 should treat sensitivity over noise placement as central scientific
  evidence rather than as appendix-only detail.
- The natural implementation home for Task 7 sensitivity validation is the new
  `benchmarks/density_matrix/performance_evidence/` package, with
  `sensitivity_matrix_validation.py` and `signals.py` as the shared Story 4
  surface.
- Story 4 should prefer one explicit sensitivity rule over one interpretation
  convention per benchmark figure or paper table.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 7 Sensitivity Dimensions And Interpretation Rule

**Implements story**
- `Story 4: Sensitivity Over Planner Settings, Noise Placement, And Workload Identity Stays First-Class`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 defines one explicit rule for which Task 7 dimensions count as
  required sensitivity evidence.
- The rule covers planner-setting, noise-placement, and workload-identity
  variation concretely enough for later benchmark and paper work to rely on it.
- Story 4 distinguishes required sensitivity evidence from optional exploratory
  comparisons.

**Execution checklist**
- [ ] Freeze the required Story 4 sensitivity dimensions around bounded
      planner-setting variation, sparse / periodic / dense noise placement, and
      workload family / qubit count / seed-fixed identity.
- [ ] Define which sensitivity differences must remain explicit in the emitted
      benchmark package.
- [ ] Define how rerun-sensitive or close planner-setting outcomes remain part
      of the visible sensitivity surface.
- [ ] Keep diagnosis, final metric packaging, and summary semantics outside the
      Story 4 bar.

**Evidence produced**
- One stable Task 7 sensitivity-dimension rule.
- One explicit interpretation rule for close or rerun-sensitive outcomes inside
  the bounded Task 5 family.

**Risks / rollback**
- Risk: later benchmark summaries may collapse meaningful planner-setting or
  noise-placement variation into one average if Story 4 leaves the rule
  implicit.
- Rollback/mitigation: freeze one explicit sensitivity rule before broad
  benchmark packaging.

### Engineering Task 2: Reuse The Bounded Task 5 Planner Surface And Shared Noise-Pattern Vocabulary As The Base

**Implements story**
- `Story 4: Sensitivity Over Planner Settings, Noise Placement, And Workload Identity Stays First-Class`

**Change type**
- docs | code

**Definition of done**
- Story 4 reuses the existing Task 5 planner-setting surface and shared
  workload/noise vocabulary where they already fit the Task 7 sensitivity
  contract.
- Story 4 keeps sensitivity interpretation aligned with the supported planner
  claim rather than inventing a new benchmark-only setting vocabulary.
- Story 4 avoids redefining noise-placement labels or workload identity.

**Execution checklist**
- [ ] Reuse Task 5 claim-selection identifiers and supported planner-setting
      fields directly where they match Story 4 needs.
- [ ] Reuse the existing sparse / periodic / dense noise-pattern labels and
      structured workload-family names without renaming them.
- [ ] Keep workload identity aligned with the shared Phase 3 benchmark matrix.
- [ ] Document any additive Story 4 sensitivity fields explicitly rather than
      renaming Task 5 or planner-surface vocabulary.

**Evidence produced**
- One reviewable mapping from the Task 5 claim surface and shared workload
  vocabulary to the Task 7 sensitivity surface.
- One explicit boundary between reused phase-wide vocabulary and Story 4-
  specific sensitivity fields.

**Risks / rollback**
- Risk: Story 4 may produce plausible sensitivity plots that no longer map
  cleanly to the supported Task 5 claim boundary.
- Rollback/mitigation: anchor Story 4 directly on the bounded Task 5 planner
  surface and shared noise vocabulary.

### Engineering Task 3: Build The Task 7 Sensitivity Validation Harness

**Implements story**
- `Story 4: Sensitivity Over Planner Settings, Noise Placement, And Workload Identity Stays First-Class`

**Change type**
- code | validation automation

**Definition of done**
- Story 4 has one reusable harness for validating sensitivity across the
  required Task 7 dimensions.
- The harness records cross-knob comparisons in a machine-reviewable way.
- The harness is reusable by later package and summary consumers.

**Execution checklist**
- [ ] Add a dedicated Story 4 validation driver under
      `benchmarks/density_matrix/performance_evidence/`, with
      `sensitivity_matrix_validation.py` as the primary checker.
- [ ] Read the Story 1 matrix, Story 2 counted-supported output, and bounded
      Task 5 planner-setting references directly.
- [ ] Record cross-knob comparisons for planner-setting, noise-placement, and
      workload-identity variation.
- [ ] Keep the harness rooted in explicit sensitivity interpretation rather than
      prose-only observations.

**Evidence produced**
- One reusable Story 4 sensitivity-validation harness.
- One machine-reviewable sensitivity schema for later Task 7 consumers.

**Risks / rollback**
- Risk: sensitivity logic may remain scattered across ad hoc scripts and drift
  from the supported claim surface.
- Rollback/mitigation: centralize Story 4 in one stable validation entry point.

### Engineering Task 4: Attach Sensitivity Fields Directly To Shared Task 7 Records

**Implements story**
- `Story 4: Sensitivity Over Planner Settings, Noise Placement, And Workload Identity Stays First-Class`

**Change type**
- code | tests

**Definition of done**
- Story 4 defines explicit fields for the required sensitivity dimensions and
  their benchmark interpretations on the shared Task 7 records.
- Sensitivity signals remain attached to stable workload and planner-setting
  identity.
- Story 4 avoids post hoc interpretation for basic cross-knob semantics.

**Execution checklist**
- [ ] Add explicit planner-setting, noise-pattern, and workload-identity
      sensitivity fields to the Task 7 record surface or the smallest auditable
      successor.
- [ ] Define how close or rerun-sensitive planner-setting outcomes remain
      visible in the emitted records.
- [ ] Ensure sensitivity fields remain attached to stable workload and
      claim-selection references.
- [ ] Add focused regression checks for sensitivity-field presence and semantic
      stability.

**Evidence produced**
- One explicit Story 4 sensitivity rule on shared Task 7 records.
- Regression coverage for required sensitivity-field stability.

**Risks / rollback**
- Risk: later summaries may cite sensitivity results without preserving which
  planner settings or noise patterns were actually compared.
- Rollback/mitigation: attach sensitivity fields directly to the benchmark
  records.

### Engineering Task 5: Add A Representative Cross-Knob Sensitivity Matrix

**Implements story**
- `Story 4: Sensitivity Over Planner Settings, Noise Placement, And Workload Identity Stays First-Class`

**Change type**
- tests | validation automation

**Definition of done**
- Story 4 covers representative cross-knob comparisons for the required
  sensitivity dimensions.
- The matrix is broad enough to show that sensitivity remains visible across the
  main Task 7 benchmark knobs.
- The matrix remains representative and contract-driven rather than exhaustive
  over every optional planner-setting or workload combination.

**Execution checklist**
- [ ] Include at least one comparison across bounded planner-setting variants
      inside the supported Task 5 surface.
- [ ] Include at least one sparse / periodic / dense noise-placement comparison.
- [ ] Include at least one workload-identity comparison across family, qubit
      count, or seed-fixed instance.
- [ ] Keep diagnosis closure, final metric packaging, and summary rollups
      outside the Story 4 matrix.

**Evidence produced**
- One representative Story 4 cross-knob sensitivity matrix.
- One review surface for explicit sensitivity visibility across required
  dimensions.

**Risks / rollback**
- Risk: Story 4 may appear correct for one knob while drifting on the others.
- Rollback/mitigation: freeze a small but cross-knob sensitivity matrix early.

### Engineering Task 6: Emit A Stable Story 4 Sensitivity Bundle Or Rerunnable Checker

**Implements story**
- `Story 4: Sensitivity Over Planner Settings, Noise Placement, And Workload Identity Stays First-Class`

**Change type**
- validation automation | docs

**Definition of done**
- Story 4 emits one stable machine-reviewable sensitivity bundle or rerunnable
  checker.
- The bundle records cross-knob benchmark differences through one stable schema.
- The output is stable enough for later Task 7 stories and publication review
  to cite directly.

**Execution checklist**
- [ ] Add a dedicated Story 4 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task7/story4_sensitivity_matrix/`).
- [ ] Emit sensitivity comparisons through one stable schema plus a stable
      bundle summary.
- [ ] Record rerun commands and software metadata with the emitted bundle.
- [ ] Keep the relationship to the bounded Task 5 claim surface explicit in the
      bundle summary.

**Evidence produced**
- One stable Story 4 sensitivity bundle or checker.
- One direct citation surface for the Task 7 sensitivity matrix.

**Risks / rollback**
- Risk: prose-only Story 4 closure will make later reviewers unable to tell how
  sensitivity was actually measured.
- Rollback/mitigation: emit one machine-reviewable sensitivity bundle directly.

### Engineering Task 7: Document The Sensitivity Rule And Run The Story 4 Surface

**Implements story**
- `Story 4: Sensitivity Over Planner Settings, Noise Placement, And Workload Identity Stays First-Class`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the Story 4 sensitivity rule and required
  cross-knob dimensions concretely.
- The Story 4 sensitivity harness and bundle run successfully.
- Story 4 keeps diagnosis semantics and final summary interpretation clearly
  outside the sensitivity rule itself.

**Execution checklist**
- [ ] Document the Story 4 required sensitivity dimensions and how close or
      rerun-sensitive outcomes should be interpreted.
- [ ] Explain how later Task 7 stories should consume the Story 4 sensitivity
      surface.
- [ ] Run focused Story 4 regression coverage and verify
      `benchmarks/density_matrix/performance_evidence/sensitivity_matrix_validation.py`.
- [ ] Record stable references to the Story 4 tests, checker, and emitted
      bundle.

**Evidence produced**
- Passing Story 4 sensitivity regression checks.
- One stable Story 4 sensitivity bundle or checker reference.

**Risks / rollback**
- Risk: later reviewers may mistake Story 4 for broad support expansion rather
  than for bounded benchmark sensitivity over the frozen Task 7 matrix.
- Rollback/mitigation: document the Story 4 boundary to the supported claim
  surface explicitly.

## Exit Criteria

Story 4 is complete only when all of the following are true:

- one explicit Task 7 sensitivity rule exists across bounded planner-setting,
  noise-placement, and workload-identity dimensions,
- sensitivity signals remain attached to stable workload identity and bounded
  Task 5 claim-selection references,
- close or rerun-sensitive outcomes remain visible rather than being hidden
  behind one permanently frozen setting identity,
- one stable Story 4 sensitivity bundle or rerunnable checker exists for direct
  citation,
- and metric-surface packaging, diagnosis closure, package assembly, and summary
  guardrails remain clearly assigned to later stories.

## Implementation Notes

- Prefer one explicit sensitivity rule over one interpretation convention per
  benchmark figure.
- In actual coding order, Story 4 should build on the shared metric and record
  surface once Story 5 lands those primitives instead of introducing a second
  sensitivity-specific measurement schema.
- Keep Story 4 focused on required cross-knob visibility, not yet on diagnosis
  or publication rollups.
- Treat noise-placement sensitivity as core Phase 3 evidence, not as optional
  appendix-only material.
