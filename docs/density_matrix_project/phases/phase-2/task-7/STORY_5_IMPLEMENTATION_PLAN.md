# Story 5 Implementation Plan

## Story Being Implemented

Story 5: Future Work And Non-Goals Stay Visibly Separated From Current Phase 2
Commitments

This is a Layer 4 engineering plan for implementing the fifth behavioral slice
from `TASK_7_STORIES.md`.

## Scope

This story turns the frozen roadmap boundary into one explicit documentation gate
for what Phase 2 is not:

- later-phase work remains visible and intellectually connected to Phase 2
  without being allowed to inflate current commitments,
- non-goals and deferrals remain explicit across developer-facing, roadmap-
  facing, and publication-facing documentation,
- roadmap order is preserved so Phase 2 remains the exact noisy backend
  integration milestone rather than an accidental umbrella for later
  acceleration or trainability phases,
- and the resulting current-versus-future boundary stays narrow enough that
  Story 6 can close terminology and cross-reference consistency without reopening
  roadmap decisions.

Out of scope for this story:

- changing the roadmap or phase boundaries themselves,
- redefining the current Phase 2 evidence floor owned by Story 4,
- broad terminology and cross-document consistency closure owned by Story 6,
- and implementing any deferred feature such as partitioning, fusion, gradients,
  optimizer comparisons, or trainability analysis.

## Dependencies And Assumptions

- Story 4 already closes how the current Phase 2 claim is supported. Story 5
  should preserve that claim boundary while clarifying what remains outside it.
- Phase focus and future-work boundaries are already frozen in:
  - `P2-ADR-002`,
  - `P2-ADR-008`,
  - `DETAILED_PLANNING_PHASE_2.md` sections on in-scope, out-of-scope, why
    Phase 2 must precede Phase 3, and non-goals,
  - `RESEARCH_ALIGNMENT.md`,
  - and `CHANGELOG.md`.
- Paper-facing documents already mention later-phase work. Story 5 should align
  that wording with the frozen roadmap order instead of treating the paper as a
  separate roadmap source.
- Story 1 and later Task 7 stories should reuse one shared reader-facing entry
  surface in
  `docs/density_matrix_project/phases/phase-2/PHASE_2_DOCUMENTATION_INDEX.md`.
  Story 5 should extend that same surface with current-versus-future boundary
  guidance instead of adding a disconnected roadmap note.
- Task 7 Story 5 is a documentation-boundary layer only. It does not promote any
  deferred item into the current contract.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Future-Work And Non-Goal Taxonomy For Phase 2

**Implements story**
- `Story 5: Future Work And Non-Goals Stay Visibly Separated From Current Phase 2 Commitments`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 names one canonical taxonomy for current Phase 2 scope, explicit
  non-goals, and later-phase work.
- The taxonomy is rooted in frozen roadmap and ADR decisions.
- The taxonomy stays small enough to be reused consistently across docs.

**Execution checklist**
- [ ] Freeze one canonical Story 5 taxonomy covering in-scope, non-goal,
      deferred, and later-phase categories.
- [ ] Include density-aware partitioning, fusion, gradients, approximate
      scaling, broader optimizer studies, and trainability analysis in the
      correct later-phase buckets.
- [ ] Record which items are explicit Phase 2 non-goals versus legitimate
      later-phase milestones.
- [ ] Keep the taxonomy rooted in the existing planning, ADR, and roadmap docs.

**Evidence produced**
- One stable Story 5 future-work and non-goal taxonomy.
- One reviewable mapping from deferred item to roadmap location.

**Risks / rollback**
- Risk: if the future-work vocabulary remains implicit, later-phase topics can
  leak into Phase 2 claims through "helpful context."
- Rollback/mitigation: freeze one small taxonomy and require Story 5 outputs to
  use it directly.

### Engineering Task 2: Reuse Existing Roadmap Phase Labels And Later-Phase Descriptions Without Renaming Them

**Implements story**
- `Story 5: Future Work And Non-Goals Stay Visibly Separated From Current Phase 2 Commitments`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 reuses the existing phase names and later-phase descriptions wherever
  practical.
- Readers can trace future-work statements directly to roadmap docs without a
  translation layer.
- Story 5 remains a clarification gate, not a second roadmap narrative.

**Execution checklist**
- [ ] Reuse existing phase labels `Phase 2`, `Phase 3`, `Phase 4`, and
      `Phase 5` exactly.
- [ ] Reuse current later-phase descriptions for partitioning, fusion,
      acceleration, broader noisy-VQA integration, optimizer studies, and
      trainability analysis where they already match the frozen roadmap.
- [ ] Reuse existing "exact noisy backend integration" wording for Phase 2
      whenever practical.
- [ ] Avoid Story 5-only phase nicknames or alternate milestone labels.

**Evidence produced**
- One Story 5 boundary layer rooted in the canonical roadmap vocabulary.
- Reviewable traceability from future-work wording to the roadmap docs.

**Risks / rollback**
- Risk: renamed phase labels or custom milestone phrases can make roadmap order
  look negotiable or fuzzy.
- Rollback/mitigation: preserve existing phase names and milestone wording unless
  there is a narrow clarity need backed by explicit mapping.

### Engineering Task 3: Add Explicit Current-Versus-Future Labeling Rules Across Phase 2 Docs

**Implements story**
- `Story 5: Future Work And Non-Goals Stay Visibly Separated From Current Phase 2 Commitments`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 defines which future-work and non-goal labels must appear explicitly in
  the documentation bundle.
- Missing or ambiguous current-versus-future labeling fails Story 5 review.
- The labeling rules are stable enough for fast checks and later bundle review.

**Execution checklist**
- [ ] Mark later-phase topics as mandatory explicit labels wherever they are
      mentioned in Phase 2-facing docs.
- [ ] Require visible labels such as `future work`, `deferred`, or `non-goal`
      where deferred capabilities are discussed.
- [ ] Require wording that preserves roadmap order when Phase 2 is compared to
      Phases 3 to 5.
- [ ] Keep the labeling rules machine-checkable where practical.

**Evidence produced**
- One explicit Story 5 rule set for current-versus-future labeling.
- One reviewable list of later-phase topics that must never appear unlabeled in
  Phase 2 docs.

**Risks / rollback**
- Risk: documentation can mention later work accurately but still let readers
  assume it is part of the current milestone.
- Rollback/mitigation: require explicit labels whenever later-phase work appears
  inside Phase 2-facing text.

### Engineering Task 4: Preserve Roadmap Order And Current-Scope Boundaries In Story 5 Outputs

**Implements story**
- `Story 5: Future Work And Non-Goals Stay Visibly Separated From Current Phase 2 Commitments`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 outputs preserve the ordering that Phase 2 establishes the exact noisy
  backend, while later phases extend, optimize, and study it.
- Current-scope boundaries remain visible across developer-facing and
  publication-facing outputs.
- Later-phase topics remain informative context, not hidden commitments.

**Execution checklist**
- [ ] Keep explicit wording that Phase 2 is the exact noisy backend integration
      milestone.
- [ ] Keep explicit wording that Phase 3 is the first legitimate phase for
      acceleration and broader density-aware optimization claims.
- [ ] Keep explicit wording that broader noisy-VQA integration and optimizer
      comparisons belong later than Phase 2.
- [ ] Keep explicit wording that trainability analysis belongs later than the
      current backend-integration milestone.

**Evidence produced**
- Story 5 outputs with preserved roadmap order and current-scope boundaries.
- Reviewable wording showing how Phase 2 hands off to later phases.

**Risks / rollback**
- Risk: if roadmap order is blurred, reviewers may evaluate Phase 2 against
  claims that properly belong to later milestones.
- Rollback/mitigation: keep the phase handoff structure explicit in every Story 5
  output.

### Engineering Task 5: Add Focused Regression Checks For Scope Drift And Phase Leakage

**Implements story**
- `Story 5: Future Work And Non-Goals Stay Visibly Separated From Current Phase 2 Commitments`

**Change type**
- tests

**Definition of done**
- Fast checks catch missing future-work labels, missing non-goal wording, or
  Phase 2 text that overstates later-phase capability.
- Negative cases show that Story 5 fails if deferred work is described as
  current support.
- Regression coverage remains documentation-focused and lightweight.

**Execution checklist**
- [ ] Add focused Story 5 checks in `tests/density_matrix/test_phase2_docs.py`
      or a tightly related successor.
- [ ] Add negative checks for unlabeled mentions of partitioning, fusion,
      gradients, optimizer studies, or trainability analysis inside Phase 2
      claims.
- [ ] Add at least one check that fails if later-phase work is described as a
      delivered Phase 2 commitment.
- [ ] Keep roadmap-wide planning validation outside this fast documentation check
      layer.

**Evidence produced**
- Focused regression coverage for Story 5 future-work and non-goal semantics.
- Reviewable failure messages for phase leakage and scope drift.

**Risks / rollback**
- Risk: scope drift can happen through seemingly harmless roadmap prose even when
  the underlying implementation scope is still frozen.
- Rollback/mitigation: lock the current-versus-future boundary down with focused
  checks.

### Engineering Task 6: Emit One Stable Story 5 Future-Work Boundary Audit Or Rerunnable Checker

**Implements story**
- `Story 5: Future Work And Non-Goals Stay Visibly Separated From Current Phase 2 Commitments`

**Change type**
- validation automation | docs

**Definition of done**
- Story 5 can emit one stable machine-readable future-work audit or one stable
  rerunnable checker.
- The output records later-phase topics, their labels, and the docs that mention
  them.
- The output is stable enough for Story 6 to consume directly.

**Execution checklist**
- [ ] Add one Story 5 command, script, or checker
      (for example under `benchmarks/density_matrix/`) for future-work audit
      emission.
- [ ] Emit one stable artifact in a Task 7 artifact directory
      (for example `benchmarks/density_matrix/artifacts/documentation_contract/`).
- [ ] Record source references, generation command, and current-versus-future
      metadata with the output.
- [ ] Keep the output narrow to future-work and non-goal auditing rather than
      broad terminology consistency.

**Evidence produced**
- One stable Task 7 Story 5 future-work boundary audit or rerunnable checker.
- One reusable Story 5 output schema for later Task 7 handoffs.

**Risks / rollback**
- Risk: future-work separation can stay prose-only and become hard to review
  systematically.
- Rollback/mitigation: emit one thin structured Story 5 surface that makes the
  roadmap boundary explicit and auditable.

### Engineering Task 7: Document Story 5 Boundary Semantics And Handoff To Story 6

**Implements story**
- `Story 5: Future Work And Non-Goals Stay Visibly Separated From Current Phase 2 Commitments`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Story 5 validates, how to rerun it, and
  why it is the canonical current-versus-future boundary gate for Task 7.
- The notes make clear that Story 5 closes scope separation but not the final
  terminology and cross-reference bundle owned by Story 6.
- The documentation stays aligned with the frozen roadmap order.

**Execution checklist**
- [ ] Document the Story 5 audit or checker and its relationship to the roadmap,
      planning, and paper-facing docs.
- [ ] Make the main Story 5 rule explicit:
      later-phase work may be mentioned, but it must remain visibly separate
      from current Phase 2 commitments.
- [ ] Explain how Story 5 hands off terminology and cross-reference closure to
      Story 6.
- [ ] Keep the notes explicit that Story 5 does not promote any deferred item
      into the current milestone.

**Evidence produced**
- Updated developer-facing guidance for the Task 7 Story 5 boundary gate.
- One stable place where Story 5 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 5 is poorly documented, Story 6 may be forced to repair scope
  confusion while trying to do terminology cleanup.
- Rollback/mitigation: close the scope boundary explicitly before bundle-level
  consistency work begins.

### Engineering Task 8: Run Story 5 Validation And Confirm Current-Versus-Future Separation

**Implements story**
- `Story 5: Future Work And Non-Goals Stay Visibly Separated From Current Phase 2 Commitments`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 5 scope-boundary checks pass.
- The Story 5 audit or checker runs successfully and emits stable output.
- Story 5 completion is backed by rerunnable boundary evidence rather than by
  narrative-only roadmap prose.

**Execution checklist**
- [ ] Run focused Story 5 regression checks for future-work and non-goal
      labeling.
- [ ] Run the Story 5 audit or checker command and verify emitted output.
- [ ] Confirm later-phase topics are labeled explicitly and do not appear as
      current Phase 2 commitments.
- [ ] Record stable test and artifact references for Story 6 and later Task 7
      work.

**Evidence produced**
- Passing focused checks for Story 5 scope-boundary semantics.
- One stable Story 5 output proving that current and future work remain visibly
  separated.

**Risks / rollback**
- Risk: Story 5 can appear complete while still leaving reviewers unsure what is
  current versus aspirational.
- Rollback/mitigation: require both passing checks and one stable structured
  Story 5 output before closure.

## Exit Criteria

Story 5 is complete only when all of the following are true:

- one stable taxonomy defines current scope, non-goals, and later-phase work for
  Phase 2,
- later-phase topics remain explicitly labeled whenever they appear in Phase
  2-facing documentation,
- roadmap order is preserved so Phase 2 remains the exact noisy backend
  integration milestone,
- deferred items such as partitioning, fusion, gradients, broader optimizer
  studies, and trainability analysis do not appear as current Phase 2
  commitments,
- missing or ambiguous future-work labeling fails Story 5 completeness checks,
- one stable Story 5 output or rerunnable checker captures the future-work
  boundary in structured form,
- and final terminology and cross-reference bundle closure remains clearly
  assigned to Story 6.

## Implementation Notes

- Keep Story 5 grounded in the current roadmap docs. It should clarify the
  frozen order of phases, not renegotiate it.
- Reuse `tests/density_matrix/test_phase2_docs.py` as the default fast
  documentation-regression surface unless Story 5 reveals a compelling need for
  a separate roadmap-only checker.
- Treat future-work visibility as a positive quality signal only when it stays
  visibly separate from current commitments.
- Be especially careful with paper-facing prose. Readers often overread future
  potential as delivered scope unless the labels are explicit.
- If Story 5 is weak, Story 6 consistency work will only make the wrong scope
  more polished. Close the current-versus-future boundary first.
