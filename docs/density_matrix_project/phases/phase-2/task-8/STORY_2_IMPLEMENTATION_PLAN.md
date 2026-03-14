# Story 2 Implementation Plan

## Story Being Implemented

Story 2: Abstract, Short-Paper, And Full-Paper Surfaces Tell The Same Phase 2
Story At Different Depths

This is a Layer 4 engineering plan for implementing the second behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story turns the multi-surface publication requirement into one explicit
alignment gate for Task 8:

- the abstract, compact short-paper surface, narrative short-paper surface, and
  full-paper draft all inherit the same Paper 1 claim package,
- each surface is allowed to compress or expand explanation according to review
  depth without changing the scientific boundary,
- the narrative short-paper surface can prioritize problem, result, evidence,
  and significance for a general PhD-conference audience without detaching from
  the implementation-backed paper contract,
- and the resulting alignment layer stays narrow enough that Story 3 can focus
  on traceability and Story 4 can focus on evidence closure rather than on basic
  cross-surface consistency.

Out of scope for this story:

- freezing the Paper 1 main claim and non-claim inventory already owned by
  Story 1,
- section-level claim-to-source traceability owned by Story 3,
- mandatory evidence-floor interpretation owned by Story 4,
- supported-path and exact-regime scope honesty owned by Story 5,
- future-work and publication-ladder framing owned by Story 6,
- and final terminology plus reviewer-navigation bundle closure owned by Story
  7.

## Dependencies And Assumptions

- Story 1 already provides the canonical Paper 1 claim package. Story 2 should
  reuse that package rather than discover claim scope independently.
- Story 1 is expected to emit
  `benchmarks/density_matrix/artifacts/phase2_task8/story1_claim_package.json`
  or an equivalent rerunnable checker. Story 2 should treat that output as the
  claim-boundary authority for cross-surface alignment.
- Publication-facing surfaces already exist and should be aligned rather than
  recreated:
  - `ABSTRACT_PHASE_2.md`,
  - `SHORT_PAPER_PHASE_2.md`,
  - `SHORT_PAPER_NARRATIVE.md`,
  - and `PAPER_PHASE_2.md`.
- `SHORT_PAPER_NARRATIVE.md` is the compact narrative short-paper surface for a
  broader PhD-conference audience. Story 2 should preserve that audience focus
  without letting the narrative surface drift from the frozen Phase 2 claim.
- `SHORT_PAPER_PHASE_2.md` and `PAPER_PHASE_2.md` already contain more
  implementation-backed detail. Story 2 should allow that extra depth while
  requiring the same core contribution boundary.
- The canonical machine-readable evidence surface remains the Task 6 publication
  bundle under `benchmarks/density_matrix/artifacts/phase2_task6/`. Story 2
  should align paper surfaces to that evidence rather than let any surface
  overstate it.
- Story 2 defines cross-surface alignment and depth-specific narrative
  compression. It should not invent new evidence, new claims, or a new review
  taxonomy.

## Engineering Tasks

### Engineering Task 1: Freeze The Publication-Surface Inventory And Role Taxonomy

**Implements story**
- `Story 2: Abstract, Short-Paper, And Full-Paper Surfaces Tell The Same Phase 2 Story At Different Depths`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 names one canonical inventory of publication-facing surfaces for Task
  8.
- Each surface has one explicit role in the publication package.
- The role taxonomy stays aligned with the current document set and review
  audiences.

**Execution checklist**
- [ ] Freeze one canonical inventory covering the abstract, implementation-backed
      short paper, narrative short paper, and full-paper draft.
- [ ] Define one explicit role for each surface:
      conference abstract, implementation-backed compact paper, narrative
      general-conference short paper, or full-paper research draft.
- [ ] Keep the inventory rooted in the existing file structure and active Phase 2
      paper docs rather than in ad hoc manuscript variants.
- [ ] Record the inventory in one stable Story 2 planning surface that later
      Task 8 stories can reference directly.

**Evidence produced**
- One stable Story 2 publication-surface inventory and role taxonomy.
- One reviewable mapping from paper surface to review-depth role.

**Risks / rollback**
- Risk: if surface roles remain implicit, conflicting prose may be mistaken for
  acceptable depth variation instead of genuine claim drift.
- Rollback/mitigation: define the surface inventory and role taxonomy explicitly
  before aligning content.

### Engineering Task 2: Reuse The Story 1 Claim Package Across All Publication Surfaces

**Implements story**
- `Story 2: Abstract, Short-Paper, And Full-Paper Surfaces Tell The Same Phase 2 Story At Different Depths`

**Change type**
- docs | validation automation

**Definition of done**
- All publication surfaces inherit the same Story 1 main claim and non-claim
  package.
- Surface-specific wording can differ in detail but not in claim boundary.
- Story 2 remains an alignment layer, not a second claim-definition layer.

**Execution checklist**
- [ ] Reuse the Story 1 main claim and supporting-claim inventory in every
      publication surface.
- [ ] Reuse the Story 1 non-claim inventory in compressed or expanded form as
      appropriate to surface depth.
- [ ] Avoid introducing surface-specific claim exceptions unless they are
      explicitly mapped to the same Story 1 boundary.
- [ ] Keep any audience-specific compression subordinate to the frozen claim
      package rather than vice versa.

**Evidence produced**
- One Story 2 alignment layer rooted in the Story 1 claim package.
- Reviewable traceability from each publication surface to the same claim
  boundary.

**Risks / rollback**
- Risk: without claim-package reuse, different surfaces can drift into different
  papers while still looking individually coherent.
- Rollback/mitigation: make Story 1 the required upstream source for all surface
  alignment work.

### Engineering Task 3: Define Required Section And Statement Coverage For Each Publication Surface

**Implements story**
- `Story 2: Abstract, Short-Paper, And Full-Paper Surfaces Tell The Same Phase 2 Story At Different Depths`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 makes explicit which sections or statement classes each publication
  surface must contain.
- Missing mandatory sections or mandatory claim statements block Story 2
  closure.
- Coverage rules remain stable enough for fast checks and later Task 8 review.

**Execution checklist**
- [ ] Define required statement coverage for the abstract:
      problem, contribution, evidence bar, and bounded significance.
- [ ] Define required section coverage for both short-paper surfaces:
      problem, contribution, evidence, limitations, and future-work boundary.
- [ ] Define required section coverage for the full-paper draft:
      research gap, contribution, evidence, limitations, related work, and
      future directions.
- [ ] Keep optional contextual or venue-specific sections separate from the
      mandatory alignment inventory.

**Evidence produced**
- One explicit coverage rule set for Story 2 publication surfaces.
- One reviewable list of mandatory sections and statement classes per surface.

**Risks / rollback**
- Risk: alignment review can stay subjective if it does not define what each
  surface must actually cover.
- Rollback/mitigation: freeze the section and statement inventory explicitly and
  validate surfaces against it.

### Engineering Task 4: Preserve Audience-Appropriate Compression Without Losing Scientific Boundary

**Implements story**
- `Story 2: Abstract, Short-Paper, And Full-Paper Surfaces Tell The Same Phase 2 Story At Different Depths`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 defines how each surface may compress or expand details without
  changing the scientific story.
- The narrative short-paper surface can prioritize significance and results for
  general conference review while remaining contract-aligned.
- The full-paper surface may add implementation detail and context without
  broadening scope.

**Execution checklist**
- [ ] Preserve a compact, problem-first framing for `ABSTRACT_PHASE_2.md`.
- [ ] Preserve a results-and-significance-first framing for
      `SHORT_PAPER_NARRATIVE.md` while keeping its claims aligned with Story 1.
- [ ] Preserve a more implementation-backed compact paper role for
      `SHORT_PAPER_PHASE_2.md` without letting it claim a different result.
- [ ] Preserve the right of `PAPER_PHASE_2.md` to expand method and context while
      keeping the same contribution boundary and evidence bar.

**Evidence produced**
- One explicit depth-compression policy for the Task 8 publication surfaces.
- Reviewable wording showing how audience adaptation stays inside the same claim
  boundary.

**Risks / rollback**
- Risk: depth-specific editing can become a pretext for adding or dropping scope
  constraints.
- Rollback/mitigation: keep audience adaptation explicit and subordinate to the
  shared claim package.

### Engineering Task 5: Add Focused Regression Checks For Cross-Surface Claim Drift

**Implements story**
- `Story 2: Abstract, Short-Paper, And Full-Paper Surfaces Tell The Same Phase 2 Story At Different Depths`

**Change type**
- tests

**Definition of done**
- Fast checks catch mismatched claims, mismatched thresholds, or mismatched
  scope boundaries across the publication surfaces.
- Negative cases show that Story 2 fails if one surface becomes broader or
  materially different from the others.
- Regression coverage remains focused and publication-package oriented.

**Execution checklist**
- [ ] Add focused checks in `tests/density_matrix/test_phase2_publication_docs.py`
      or a tightly related successor for Story 2 surface alignment.
- [ ] Add negative checks for conflicting main claims, conflicting workflow
      identity, or conflicting exact-regime wording.
- [ ] Add at least one check for the narrative short-paper surface dropping the
      mandatory limitation or evidence-bar boundary.
- [ ] Keep full editorial review and venue formatting outside this fast
      regression layer.

**Evidence produced**
- Focused regression coverage for Story 2 cross-surface alignment.
- Reviewable failures when publication surfaces drift apart semantically.

**Risks / rollback**
- Risk: surface drift can survive unnoticed because each individual draft still
  reads fluently.
- Rollback/mitigation: compare the surfaces through targeted checks before later
  Task 8 bundling.

### Engineering Task 6: Emit One Stable Story 2 Surface-Alignment Manifest Or Checker

**Implements story**
- `Story 2: Abstract, Short-Paper, And Full-Paper Surfaces Tell The Same Phase 2 Story At Different Depths`

**Change type**
- validation automation | docs

**Definition of done**
- Story 2 can emit one stable machine-readable surface-alignment summary or one
  stable rerunnable checker.
- The output records surface roles, required coverage classes, and alignment
  status against the Story 1 claim package.
- The output is stable enough for Story 7 and publication review to consume.

**Execution checklist**
- [ ] Add one Story 2 command, script, or checker
      (for example under `benchmarks/density_matrix/`) for publication-surface
      alignment summary emission.
- [ ] Emit one stable artifact in a Task 8 artifact directory such as
      `benchmarks/density_matrix/artifacts/phase2_task8/story2_publication_surface_alignment.json`.
- [ ] Record surface roles, source references, generation command, and scope
      notes in the output.
- [ ] Keep the output narrow to cross-surface alignment rather than full
      claim-to-evidence traceability.

**Evidence produced**
- One stable Task 8 Story 2 surface-alignment summary or rerunnable checker.
- One reusable Story 2 output schema for later Task 8 handoffs.

**Risks / rollback**
- Risk: ad hoc side-by-side review is slow and easy to skip when drafts evolve.
- Rollback/mitigation: define one thin structured Story 2 output that makes
  surface alignment explicit and rerunnable.

### Engineering Task 7: Document Story 2 Surface Roles And Handoff To Story 3 And Story 4

**Implements story**
- `Story 2: Abstract, Short-Paper, And Full-Paper Surfaces Tell The Same Phase 2 Story At Different Depths`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Story 2 validates, how to rerun it, and
  why it is the canonical cross-surface alignment gate.
- The notes make clear that Story 2 closes surface consistency but not
  claim-to-evidence traceability or evidence-floor interpretation.
- The documentation stays aligned with the frozen Phase 2 publication boundary.

**Execution checklist**
- [ ] Document the Story 2 summary or checker and how it relates to Story 1 plus
      the current paper-facing files.
- [ ] Make the main Story 2 rule explicit:
      different surface depths must still tell the same Phase 2 story.
- [ ] Explain how Story 2 hands off claim-to-source traceability to Story 3 and
      evidence-closure semantics to Story 4.
- [ ] Keep surface-specific style and venue adaptation clearly outside the claim
      boundary itself.

**Evidence produced**
- Updated developer-facing guidance for the Task 8 Story 2 alignment gate.
- One stable location where Story 2 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 2 is poorly documented, later draft edits can silently reopen
  the same cross-surface drift problem.
- Rollback/mitigation: document Story 2 as the explicit alignment gate and keep
  its handoff boundaries visible.

### Engineering Task 8: Run Story 2 Validation And Confirm Aligned Multi-Depth Paper Surfaces

**Implements story**
- `Story 2: Abstract, Short-Paper, And Full-Paper Surfaces Tell The Same Phase 2 Story At Different Depths`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 2 alignment checks pass.
- The Story 2 summary or checker runs successfully and emits stable output.
- Story 2 completion is backed by rerunnable alignment evidence rather than by
  subjective editorial confidence.

**Execution checklist**
- [ ] Run focused Story 2 regression checks for publication-surface alignment.
- [ ] Run the Story 2 summary or checker command and verify emitted output.
- [ ] Confirm that the abstract, short-paper, narrative short-paper, and
      full-paper surfaces share the same claim boundary, evidence bar, and
      limitation structure.
- [ ] Record stable test and artifact references for Story 3 and later Task 8
      work.

**Evidence produced**
- Passing focused checks for Story 2 multi-surface alignment.
- One stable Story 2 output proving the publication surfaces tell the same Phase
  2 story at different depths.

**Risks / rollback**
- Risk: Story 2 can look complete while still leaving one surface semantically
  incompatible with the others.
- Rollback/mitigation: require passing checks plus one stable emitted output
  before closing Story 2.

## Exit Criteria

Story 2 is complete only when all of the following are true:

- one stable publication-surface inventory defines the Task 8 paper surfaces and
  their roles,
- all publication surfaces inherit the same Story 1 claim package,
- required sections or statement classes are present for each surface,
- audience-appropriate compression does not change the scientific boundary,
- mismatched claims, mismatched scope boundaries, or dropped limitation
  statements fail Story 2 alignment checks,
- one stable Story 2 output or rerunnable checker captures cross-surface
  alignment status,
- and claim-to-source traceability, evidence closure, supported-path wording,
  future-work framing, and publication-package bundle closure remain clearly
  assigned to Stories 3 to 7.

## Implementation Notes

- Treat `SHORT_PAPER_NARRATIVE.md` as a narrative-depth variant, not as a
  license to tell a different scientific story.
- Keep the abstract compact, but do not let compactness erase the evidence bar
  or the limitation boundary.
- Allow the full paper to be richer in method and context. Story 2 only fails
  when that richer surface changes the claim boundary or scope honesty.
- Prefer one surface inventory with explicit roles over a pile of “latest”
  manuscript variants. Reviewers need clarity about what each surface is for.
- If Story 2 is strong, later Task 8 work can treat the paper package as one
  coherent multi-depth publication set rather than as unrelated drafts.
