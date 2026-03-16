# Story 6 Implementation Plan

## Story Being Implemented

Story 6: Future Work And Publication-Ladder Positioning Stay Explicit In The
Paper Package

This is a Layer 4 engineering plan for implementing the sixth behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story turns the Task 8 roadmap-honesty requirement into one explicit
publication-positioning gate for Paper 1:

- later-phase work remains visible as motivation and future direction without
  being read as current Paper 1 evidence,
- Paper 1 is positioned clearly as the Phase 2 exact noisy backend integration
  milestone within the broader PhD publication ladder,
- publication-facing significance language stays aligned with the roadmap order
  already frozen by the planning set,
- and the resulting future-work layer stays narrow enough that Story 7 can own
  final terminology, reviewer entry paths, and publication-package bundle
  closure.

Out of scope for this story:

- redefining the Paper 1 claim package already frozen by Story 1,
- cross-surface alignment already owned by Story 2,
- claim-to-source traceability already owned by Story 3,
- evidence-floor and claim-closure interpretation already owned by Story 4,
- supported-path and exact-regime wording already owned by Story 5,
- and final bundle-level terminology and reviewer-navigation closure owned by
  Story 7.

## Dependencies And Assumptions

- Stories 1 to 5 are already in place and provide the Paper 1 claim package,
  surface alignment, traceability, evidence closure, and supported-path wording.
- Story 5 is expected to emit
  `benchmarks/density_matrix/artifacts/publication_claim_package/supported_path_scope_bundle.json`
  or an equivalent rerunnable checker. Story 6 should assume supported-path
  honesty is already frozen and focus on roadmap framing.
- The broader publication ladder is already frozen by:
  - `docs/density_matrix_project/planning/PUBLICATIONS.md`,
  - `docs/density_matrix_project/RESEARCH_ALIGNMENT.md`,
  - `docs/density_matrix_project/CHANGELOG.md`,
  - `DETAILED_PLANNING_PHASE_2.md`,
  - and `P2-ADR-002` plus `P2-ADR-008`.
- Current paper-facing docs already mention later phases as motivation. Story 6
  should align those mentions with the frozen roadmap rather than removing them
  entirely.
- Story 6 should define publication positioning and future-work boundary rules
  only. It should not reopen the roadmap order or add new Phase 2 commitments.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Future-Work And Publication-Ladder Inventory

**Implements story**
- `Story 6: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- docs | validation automation

**Definition of done**
- Story 6 names one stable inventory of later-phase topics relevant to Paper 1
  framing.
- Story 6 names one stable inventory of publication-ladder positioning
  statements for Phase 2.
- The inventory stays aligned with the frozen roadmap docs.

**Execution checklist**
- [ ] Freeze one canonical list of later-phase topics that may appear as future
      work in Paper 1.
- [ ] Freeze one canonical list of positioning statements that identify Paper 1
      as the Phase 2 integration paper.
- [ ] Keep the inventory rooted in the current roadmap docs rather than in ad
      hoc aspirational wording from manuscript drafts.
- [ ] Record the inventory in one stable Story 6 planning surface that later
      Task 8 stories can reference directly.

**Evidence produced**
- One stable Story 6 future-work and publication-ladder inventory.
- One reviewable mapping from later-phase topic to allowed Phase 2 positioning
  statement.

**Risks / rollback**
- Risk: if future-work and positioning language remain implicit, manuscript
  drafts can overstate current results while still sounding forward-looking.
- Rollback/mitigation: freeze the allowed roadmap-facing language explicitly and
  validate it directly.

### Engineering Task 2: Reuse Existing Roadmap Docs Without Reframing The Publication Ladder

**Implements story**
- `Story 6: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- docs | validation automation

**Definition of done**
- Story 6 reuses the existing publication ladder and roadmap order wherever
  practical.
- Readers can trace future-work framing directly to the planning set rather than
  to ad hoc paper rhetoric.
- Story 6 remains a publication-positioning layer, not a second roadmap.

**Execution checklist**
- [ ] Reuse the Paper 1 positioning language already present in `PUBLICATIONS.md`
      where it matches the frozen Phase 2 contract.
- [ ] Reuse roadmap order from `RESEARCH_ALIGNMENT.md` and `CHANGELOG.md` where
      those docs already describe later phases.
- [ ] Avoid introducing Story 6-only phase ordering or alternative publication
      ladder terminology.
- [ ] Keep any necessary paraphrase explicitly mapped back to the canonical
      roadmap source.

**Evidence produced**
- One Story 6 positioning layer rooted in canonical roadmap vocabulary.
- Reviewable traceability from future-work wording to the authoritative planning
  docs.

**Risks / rollback**
- Risk: a new publication-ladder vocabulary can silently make Phase 2 sound like
  it already includes Phase 3 or Phase 4 outcomes.
- Rollback/mitigation: preserve the frozen roadmap order and cite it directly.

### Engineering Task 3: Encode Explicit Allowed Future-Work And Forbidden Current-Claim Wording

**Implements story**
- `Story 6: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- docs | validation automation

**Definition of done**
- Story 6 records one explicit inventory of allowed future-work framing for
  later-phase topics.
- Story 6 records one explicit inventory of forbidden wording that would turn
  later-phase topics into current Paper 1 claims.
- Missing or contradictory framing rules block Story 6 completion.

**Execution checklist**
- [ ] Define allowed future-work wording for density-aware partitioning, fusion,
      gradient-path completion, approximate scaling, broader optimizer studies,
      and trainability analysis.
- [ ] Define forbidden wording that makes those later-phase topics sound like
      delivered Paper 1 results.
- [ ] Distinguish future-work motivation from current-evidence statements.
- [ ] Keep the rules machine-checkable where practical.

**Evidence produced**
- One explicit Story 6 future-work framing rule set.
- One reviewable list of allowed and forbidden roadmap-facing statements.

**Risks / rollback**
- Risk: without explicit rules, future-work sections often become a back door for
  overclaiming.
- Rollback/mitigation: define the allowed and forbidden framing directly and
  check it as part of Story 6 completion.

### Engineering Task 4: Preserve Phase 2 Positioning As The Exact Noisy Integration Milestone

**Implements story**
- `Story 6: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- docs | validation automation

**Definition of done**
- Story 6 preserves the positioning of Paper 1 as the Phase 2 integration paper.
- Phase 2 is described as enabling later phases rather than replacing them.
- The positioning remains stable across all publication-facing surfaces.

**Execution checklist**
- [ ] Keep explicit wording that Phase 2 is the exact noisy backend integration
      milestone.
- [ ] Keep explicit wording that later phases own acceleration, broader
      optimizer studies, and trainability science.
- [ ] Preserve the significance of Phase 2 as an enabling research instrument
      without turning enabling value into completed future results.
- [ ] Treat missing or contradictory milestone-positioning statements as Story 6
      failures.

**Evidence produced**
- Story 6 outputs that position Paper 1 as the Phase 2 integration step within
  the larger publication ladder.
- Reviewable wording showing how Phase 2 enables but does not replace later
  phases.

**Risks / rollback**
- Risk: significance framing can drift into “already achieved later science”
  wording even when claim and evidence boundaries elsewhere stay strong.
- Rollback/mitigation: make milestone-positioning language explicit and validate
  it directly.

### Engineering Task 5: Add Focused Regression Checks For Scope Drift In Future-Work Wording

**Implements story**
- `Story 6: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- tests

**Definition of done**
- Fast checks catch future-work sections that blur current results with
  later-phase goals.
- Negative cases show that Story 6 fails if later-phase work is described as
  delivered Paper 1 output.
- Regression coverage remains small and publication-package focused.

**Execution checklist**
- [ ] Add focused checks in `tests/density_matrix/test_phase2_publication_docs.py`
      or a tightly related successor for Story 6 roadmap framing.
- [ ] Add negative checks for later-phase acceleration, optimizer, or
      trainability work being described as current Paper 1 result.
- [ ] Add at least one check for missing Phase 2 milestone-positioning language.
- [ ] Keep full narrative-quality review outside this fast regression layer.

**Evidence produced**
- Focused regression coverage for Story 6 future-work and positioning integrity.
- Reviewable failures when roadmap framing drifts into overclaim territory.

**Risks / rollback**
- Risk: scope drift often appears as harmless motivation text until it changes
  how the paper is interpreted.
- Rollback/mitigation: lock the roadmap-facing language down with targeted
  checks.

### Engineering Task 6: Emit One Stable Story 6 Future-Work Boundary Audit Or Checker

**Implements story**
- `Story 6: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- validation automation | docs

**Definition of done**
- Story 6 can emit one stable machine-readable future-work boundary audit or one
  stable rerunnable checker.
- The output records allowed future-work topics, milestone-positioning rules,
  and validation status.
- The output is stable enough for Story 7 and publication review to consume.

**Execution checklist**
- [ ] Add one Story 6 command, script, or checker
      (for example under `benchmarks/density_matrix/`) for future-work boundary
      audit emission.
- [ ] Emit one stable artifact in a Task 8 artifact directory such as
      `benchmarks/density_matrix/artifacts/publication_claim_package/future_work_boundary_bundle.json`.
- [ ] Record source references, generation command, and scope notes in the
      output.
- [ ] Keep the output narrow to roadmap framing and milestone positioning rather
      than terminology bundle closure.

**Evidence produced**
- One stable Task 8 Story 6 future-work boundary audit or rerunnable checker.
- One reusable Story 6 output schema for later Task 8 handoffs.

**Risks / rollback**
- Risk: ad hoc future-work prose is hard to compare and easy to overinterpret
  across evolving paper surfaces.
- Rollback/mitigation: define one thin structured Story 6 output that makes the
  boundary and positioning rules explicit.

### Engineering Task 7: Document Story 6 Roadmap Rules And Handoff To Story 7

**Implements story**
- `Story 6: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Story 6 validates, how to rerun it, and
  why it is the canonical future-work framing gate for Task 8.
- The notes make clear that Story 6 closes roadmap honesty but not final
  publication-package terminology and reviewer navigation.
- The documentation stays aligned with the frozen Phase 2 publication ladder.

**Execution checklist**
- [ ] Document the Story 6 audit or checker and how it relates to the publication
      strategy and broader roadmap docs.
- [ ] Make the main Story 6 rule explicit:
      Paper 1 must present Phase 2 as the exact noisy integration milestone and
      later phases as future work.
- [ ] Explain how Story 6 hands off top-level terminology and reviewer-path
      packaging to Story 7.
- [ ] Keep future-work value statements clearly separate from current evidence.

**Evidence produced**
- Updated developer-facing guidance for the Task 8 Story 6 roadmap gate.
- One stable location where Story 6 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 6 is poorly documented, later editorial passes can reintroduce
  roadmap drift under the label of significance.
- Rollback/mitigation: document Story 6 as the explicit future-work and
  publication-ladder gate and keep its handoff boundary visible.

### Engineering Task 8: Run Story 6 Validation And Confirm Explicit Phase Positioning

**Implements story**
- `Story 6: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 6 roadmap checks pass.
- The Story 6 audit or checker runs successfully and emits stable output.
- Story 6 completion is backed by rerunnable positioning evidence rather than by
  editorial confidence alone.

**Execution checklist**
- [ ] Run focused Story 6 regression checks for future-work and
      milestone-positioning coverage.
- [ ] Run the Story 6 audit or checker command and verify emitted output.
- [ ] Confirm that later-phase topics remain explicitly future work and that
      Phase 2 remains positioned as the exact noisy backend integration
      milestone.
- [ ] Record stable test and artifact references for Story 7 and final Task 8
      closure.

**Evidence produced**
- Passing focused checks for Story 6 roadmap-framing completeness.
- One stable Story 6 output proving the Paper 1 package keeps later work
  explicit and Phase 2 positioned correctly.

**Risks / rollback**
- Risk: Story 6 can appear complete while still leaving subtle “already solved”
  implications in publication-facing motivation text.
- Rollback/mitigation: require passing checks plus one stable emitted output
  before closing Story 6.

## Exit Criteria

Story 6 is complete only when all of the following are true:

- one stable future-work and publication-ladder inventory defines the
  roadmap-facing surface for Paper 1,
- later-phase topics remain explicitly marked as future work rather than current
  Paper 1 results,
- Phase 2 is positioned consistently as the exact noisy backend integration
  milestone that enables later phases,
- scope-drift wording fails Story 6 roadmap checks,
- one stable Story 6 output or rerunnable checker captures the future-work and
  publication-positioning surface,
- and final terminology, reviewer entry paths, and publication-package bundle
  closure remain clearly assigned to Story 7.

## Implementation Notes

- Treat future work as a boundary statement, not just as inspirational
  paragraph text. For Paper 1, later-phase visibility should strengthen
  honesty, not weaken it.
- Reuse the publication ladder from `PUBLICATIONS.md` directly whenever
  practical. Story 6 should not become a second strategy document.
- Keep “enables later work” separate from “already contains later work.” That
  distinction is the heart of this story.
- Significance language is still allowed and useful. The failure mode is scope
  inflation, not ambition.
- If Story 6 is strong, Paper 1 can sound forward-looking without creating
  confusion about what Phase 2 already delivered.
