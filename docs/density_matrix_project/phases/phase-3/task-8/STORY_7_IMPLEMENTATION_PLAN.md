# Story 7 Implementation Plan

## Story Being Implemented

Story 7: Future Work And Publication-Ladder Positioning Stay Explicit In The
Paper Package

This is a Layer 4 engineering plan for implementing the seventh behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story turns Task 8 into one explicit future-work and publication-ladder
boundary surface for Paper 2:

- the Paper 2 package positions Phase 3 as the noise-aware partitioning and
  limited-fusion methods milestone in the density-matrix publication ladder,
- deferred follow-on branches stay visible as future work without being treated
  as current results or hidden incompleteness,
- broader noisy VQE/VQA growth, optimizer studies, and approximate-scaling paths
  remain clearly outside the current Paper 2 result,
- and Story 7 closes the contract for "how Paper 2 situates itself in the
  broader roadmap" without taking ownership of the claim package, evidence
  closure, or final package-consistency guardrails.

Out of scope for this story:

- freezing the main claim and non-claims, which is owned by Story 1,
- keeping publication surfaces aligned, which is owned by Story 2,
- claim-to-source traceability owned by Story 3,
- evidence-floor and threshold-or-diagnosis closure owned by Story 4,
- supported-path and benchmark-honesty wording owned by Story 5,
- manifest-driven reviewer packaging owned by Story 6,
- and package-level terminology, reviewer-entry, and summary-consistency
  guardrails owned by Story 8.

## Dependencies And Assumptions

- Stories 1 through 6 are already expected to freeze the claim package, align
  surfaces, define traceability, define evidence closure, freeze honest
  supported-path wording, and provide one manifest-driven reviewer package. Story
  7 should frame future work against those stable surfaces rather than reopen
  them.
- The publication ladder is already frozen in:
  - `docs/density_matrix_project/planning/PUBLICATIONS.md`,
  - `docs/density_matrix_project/planning/PLANNING.md`,
  - `DETAILED_PLANNING_PHASE_3.md`,
  - and `ADRs_PHASE_3.md`.
- The phase-order argument is already explicit in the Phase 3 planning set,
  especially the "Why Phase 3 Must Precede Phase 4" section and the Phase 3
  decision-gate wording.
- Story 7 should preserve the current benchmark-driven follow-on architecture
  framing:
  - fully channel-native fused noisy blocks remain a follow-on branch,
  - broader noisy VQE/VQA growth remains Phase 4+ work,
  - and approximate scaling remains a later branch beyond the current Paper 2
    claim.
- The natural implementation home for Task 8 future-work boundary validation is
  the same `benchmarks/density_matrix/publication_evidence/` package, with
  `future_work_boundary_validation.py` as the Story 7 validation surface and
  emitted artifacts rooted in `benchmarks/density_matrix/artifacts/phase3_task8/`.
- Story 7 should validate future-work wording and publication-ladder positioning
  only. It should not replace the roadmap docs or produce a new planning
  document.

## Engineering Tasks

### Engineering Task 1: Freeze The Paper 2 Future-Work Boundary And Publication-Ladder Inventory

**Implements story**
- `Story 7: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- docs | validation automation

**Definition of done**
- Story 7 defines one explicit future-work boundary inventory for Paper 2.
- Story 7 defines one explicit publication-ladder positioning inventory for
  Phase 3.
- The inventory is stable enough that later Task 8 stories can package it
  directly.

**Execution checklist**
- [ ] Freeze one explicit future-work inventory covering channel-native fused
      noisy blocks, broader noisy VQE/VQA growth, density-backend gradients,
      optimizer studies, and approximate scaling.
- [ ] Freeze one explicit publication-ladder positioning statement for Phase 3 as
      the methods milestone between Phase 2 integration and Phase 4 broader
      workflow science.
- [ ] Distinguish future-work boundary statements from optional aspirational
      narrative or venue-specific motivation.
- [ ] Treat missing future-work boundary items as a Story 7 failure condition.

**Evidence produced**
- One stable Story 7 future-work boundary inventory.
- One stable Story 7 publication-ladder positioning inventory.

**Risks / rollback**
- Risk: without a frozen future-work inventory, later publication edits can
  quietly imply current support for later-phase results.
- Rollback/mitigation: freeze the future-work boundary before final package
  consistency work.

### Engineering Task 2: Reuse The Frozen Roadmap Order And Benchmark-Driven Follow-On Decision Logic

**Implements story**
- `Story 7: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- docs | code

**Definition of done**
- Story 7 derives its future-work and positioning rules directly from the frozen
  roadmap and decision-gate docs.
- Story 7 preserves benchmark-driven follow-on logic rather than taste-driven
  future-work phrasing.
- Story 7 avoids creating a second roadmap vocabulary for the same phase order.

**Execution checklist**
- [ ] Reuse the Phase 3 versus Phase 4 ordering already frozen in
      `DETAILED_PLANNING_PHASE_3.md` and `PLANNING.md`.
- [ ] Reuse the publication-ladder positioning already frozen in
      `PUBLICATIONS.md`.
- [ ] Reuse the decision-gate framing that later architecture branches must be
      benchmark-driven.
- [ ] Avoid renaming or weakening the accepted roadmap sequence in Story 7.

**Evidence produced**
- One reviewable mapping from the frozen roadmap and decision-gate docs to Story
  7 publication wording.
- One explicit boundary between reused roadmap logic and Story 7-specific
  publication fields.

**Risks / rollback**
- Risk: Story 7 may sound forward-looking while still drifting away from the
  accepted roadmap order.
- Rollback/mitigation: anchor Story 7 directly on the frozen program and phase
  docs.

### Engineering Task 3: Define The Story 7 Future-Work Boundary Record Schema And Checker

**Implements story**
- `Story 7: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- code | validation automation

**Definition of done**
- Story 7 has one reusable future-work boundary checker.
- The checker records deferred branches, allowed future-work phrasing, and
  publication-ladder positioning through one stable schema.
- The checker stays focused on roadmap boundary integrity rather than on the full
  package manifest or terminology inventory.

**Execution checklist**
- [ ] Add a Story 7 checker under
      `benchmarks/density_matrix/publication_evidence/`, with
      `future_work_boundary_validation.py` as the primary validation surface.
- [ ] Define one stable Story 7 future-work boundary schema.
- [ ] Record explicit deferred-branch categories and Phase 3 positioning fields.
- [ ] Keep final package-consistency and reviewer-entry semantics outside the
      Story 7 checker.

**Evidence produced**
- One reusable Story 7 future-work boundary checker.
- One stable Story 7 record schema for later Task 8 reuse.

**Risks / rollback**
- Risk: Story 7 can become vague motivational prose if it lacks one structured
  validation surface.
- Rollback/mitigation: validate one machine-reviewable future-work boundary
  record set directly.

### Engineering Task 4: Add Explicit Checks For Deferred-Branch Visibility And Non-Current-Result Language

**Implements story**
- `Story 7: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- code | tests

**Definition of done**
- Story 7 checks the highest-risk future-work boundary statements directly.
- Deferred branches remain visible and explicitly labeled as future work.
- Current Paper 2 wording cannot slide into later-phase commitment language.

**Execution checklist**
- [ ] Add explicit Story 7 record fields or checks for channel-native fusion
      follow-on status.
- [ ] Add explicit checks for broader noisy VQE/VQA growth, density-backend
      gradients, optimizer studies, and approximate scaling being labeled as
      future work.
- [ ] Add explicit checks for Phase 3 being positioned as the methods milestone
      in the publication ladder.
- [ ] Add focused regression checks for missing or inflated future-work boundary
      wording.

**Evidence produced**
- One explicit Story 7 future-work visibility and positioning rule.
- Regression coverage for required future-work boundary fields.

**Risks / rollback**
- Risk: later publication edits may preserve broad honesty while still
  accidentally reframing future work as current implied scope.
- Rollback/mitigation: attach the key future-work fields directly to the Story 7
  validation surface.

### Engineering Task 5: Add A Representative Future-Work And Publication-Ladder Matrix Across Publication Surfaces

**Implements story**
- `Story 7: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- tests | validation automation

**Definition of done**
- Story 7 covers representative future-work and publication-positioning
  statements across the Paper 2 package.
- The matrix is broad enough to show that one boundary rule spans abstract,
  technical short-paper, narrative short-paper, full-paper, and reviewer-entry
  packaging.
- The matrix remains representative and contract-driven rather than exhaustive
  over every sentence.

**Execution checklist**
- [ ] Include at least one future-work statement in a compact surface.
- [ ] Include at least one future-work statement in a long-form surface.
- [ ] Include at least one publication-ladder positioning statement.
- [ ] Include at least one benchmark-driven follow-on architecture statement.

**Evidence produced**
- One representative Story 7 future-work and publication-ladder matrix.
- One review surface for cross-surface roadmap-boundary coverage.

**Risks / rollback**
- Risk: Story 7 may appear correct in one surface while another surface quietly
  broadens or blurs the roadmap order.
- Rollback/mitigation: freeze a small but cross-surface future-work matrix
  early.

### Engineering Task 6: Add Focused Regression Checks For Future-Work Inflation Or Roadmap Drift

**Implements story**
- `Story 7: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- tests

**Definition of done**
- Fast checks catch future-work statements being rewritten as current results or
  roadmap drift being introduced into Paper 2 wording.
- Negative cases prove Story 7 fails when deferred branches are treated as
  baseline Phase 3 commitments.
- Regression coverage remains narrow and publication-package focused.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_phase3_task8.py` or a
      tightly related successor for Story 7 future-work validation.
- [ ] Add negative checks for channel-native fusion being framed as baseline
      delivered behavior.
- [ ] Add negative checks for broader noisy workflow growth or approximate
      scaling being framed as current Paper 2 scope.
- [ ] Add negative checks for Phase 3 being mispositioned in the publication
      ladder.

**Evidence produced**
- Focused regression coverage for Story 7 future-work inflation failures.
- Reviewable failures for roadmap drift or overclaiming of later-phase results.

**Risks / rollback**
- Risk: roadmap drift can survive manual review because the paper still sounds
  ambitious and coherent.
- Rollback/mitigation: add targeted checks for the highest-risk future-work
  regressions.

### Engineering Task 7: Emit A Stable Story 7 Future-Work Boundary Bundle

**Implements story**
- `Story 7: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- validation automation | docs

**Definition of done**
- Story 7 emits one stable machine-reviewable future-work boundary bundle or one
  stable rerunnable checker output.
- The output records deferred branches, allowed future-work phrasing, and
  publication-ladder positioning through one stable schema.
- The output is stable enough for later Story 8 consistency checks to consume
  directly.

**Execution checklist**
- [ ] Add one stable Story 7 output location under
      `benchmarks/density_matrix/artifacts/phase3_task8/story7_future_work_boundary/`.
- [ ] Emit one artifact such as `future_work_boundary_bundle.json`.
- [ ] Record generation command, software metadata, and roadmap-boundary summary
      in the output.
- [ ] Keep the output focused on future-work and publication-ladder semantics
      rather than on final package consistency.

**Evidence produced**
- One stable Story 7 future-work boundary bundle or rerunnable checker output.
- One reusable Story 7 output schema for later Task 8 handoffs.

**Risks / rollback**
- Risk: prose-only Story 7 closure will make later reviewers unable to tell
  whether the package actually preserved the roadmap boundary.
- Rollback/mitigation: emit one machine-reviewable future-work boundary surface
  directly.

### Engineering Task 8: Document Story 7 Roadmap Rules And Run The Story 7 Gate

**Implements story**
- `Story 7: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain what Story 7 validates, how to rerun it, and
  how it hands off to Story 8.
- The Story 7 checker and emitted artifact run successfully.
- Story 7 completion is backed by rerunnable roadmap-boundary validation rather
  than by editorial confidence alone.

**Execution checklist**
- [ ] Document the Story 7 future-work inventory and publication-ladder rule.
- [ ] Make the Story 7 rule explicit:
      Paper 2 must position Phase 3 honestly in the publication ladder and keep
      later branches labeled as future work.
- [ ] Explain how Story 7 hands off final terminology, reviewer-entry stability,
      and summary-consistency enforcement to Story 8.
- [ ] Run focused Story 7 regression checks and verify the emitted Story 7
      bundle or checker output.

**Evidence produced**
- Passing focused checks for Story 7 roadmap-boundary integrity.
- One stable Story 7 output proving honest Paper 2 future-work framing.

**Risks / rollback**
- Risk: Story 7 can look complete while still allowing subtle roadmap inflation
  in later publication edits.
- Rollback/mitigation: require passing checks plus one stable emitted output
  before closing Story 7.

## Exit Criteria

Story 7 is complete only when all of the following are true:

- one explicit future-work boundary inventory defines which branches stay outside
  the current Paper 2 result,
- one explicit publication-ladder positioning inventory keeps Phase 3 aligned as
  the methods milestone between Phase 2 and later workflow science,
- roadmap drift or future-work inflation fails focused Story 7 checks,
- one stable Story 7 bundle or rerunnable checker captures the future-work and
  positioning surface,
- and package-level terminology, reviewer-entry stability, and summary-
  consistency remain clearly assigned to Story 8.

## Implementation Notes

- Story 7 is about honest scientific positioning, not about making the paper
  sound less ambitious. Clear phase positioning strengthens Paper 2.
- Keep benchmark-driven follow-on logic explicit. Later architecture branches
  should remain justified by evidence, not by narrative convenience.
- Treat future-work wording as contract-sensitive language. It defines what the
  current paper does not claim.
- Story 7 should stay close to the frozen roadmap docs and decision-gate logic.
  That closeness is what makes the publication ladder coherent.
