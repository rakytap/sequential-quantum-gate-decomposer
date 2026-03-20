# Story 2 Implementation Plan

## Story Being Implemented

Story 2: Abstract, Technical Short-Paper, Narrative Short-Paper, And Full-Paper
Surfaces Tell The Same Phase 3 Story At Different Depths

This is a Layer 4 engineering plan for implementing the second behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story turns Task 8 into one explicit cross-surface alignment layer for
Paper 2:

- the abstract, technical short-paper, narrative short-paper, and full-paper
  surfaces all inherit one frozen claim boundary instead of drifting into
  parallel paper narratives,
- each surface preserves its audience-specific role while staying aligned on the
  same technical result, evidence bar, and limitation structure,
- surface differences are treated as controlled depth changes rather than as
  license to change the scientific claim,
- and Story 2 closes the contract for "how Paper 2 surfaces stay aligned" before
  later stories package traceability, evidence closure, manifest packaging, and
  summary integrity.

Out of scope for this story:

- freezing the main claim and non-claims, which is owned by Story 1,
- section-level claim-to-source traceability owned by Story 3,
- evidence-floor and threshold-or-diagnosis closure semantics owned by Story 4,
- supported-path, no-fallback, bounded planner-claim, and benchmark-honesty
  wording owned by Story 5,
- manifest-driven reviewer packaging owned by Story 6,
- future-work and publication-ladder positioning owned by Story 7,
- and terminology, reviewer-entry, and summary-consistency guardrails owned by
  Story 8.

## Dependencies And Assumptions

- Story 1 is already expected to freeze the stable Paper 2 main claim, the
  bounded supporting-claim set, and the explicit non-claims Story 2 must align
  across surfaces rather than rediscover.
- The publication-facing files already exist:
  - `ABSTRACT_PHASE_3.md`,
  - `SHORT_PAPER_PHASE_3.md`,
  - `SHORT_PAPER_NARRATIVE.md`,
  - and `PAPER_PHASE_3.md`.
- The phase workflow in the spec-driven skill already defines distinct roles for
  the technical short paper and the narrative short paper. Story 2 should
  preserve those role differences rather than flatten them into near-duplicate
  documents.
- Story 2 should treat surface roles as:
  - abstract: compact claim and evidence summary,
  - technical short paper: compact methods and validation surface,
  - narrative short paper: motivation, scientific arc, and positioning surface,
  - full paper: longest-form methods and systems surface.
- The natural implementation home for Task 8 surface-alignment validation is the
  same `benchmarks/density_matrix/publication_evidence/` package, with
  `surface_alignment_validation.py` as the Story 2 validation surface and
  emitted artifacts rooted in `benchmarks/density_matrix/artifacts/phase3_task8/`.
- Story 2 should validate alignment and controlled compression only. It should
  not become a general prose linter or a venue-formatting tool.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 8 Publication-Surface Role Matrix

**Implements story**
- `Story 2: Abstract, Technical Short-Paper, Narrative Short-Paper, And Full-Paper Surfaces Tell The Same Phase 3 Story At Different Depths`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 defines one explicit role matrix for the four Paper 2 surfaces.
- The matrix distinguishes allowed depth differences from prohibited claim
  differences.
- The role matrix is stable enough that later Task 8 stories can consume it
  directly.

**Execution checklist**
- [ ] Freeze one role definition for the abstract, technical short paper,
      narrative short paper, and full paper.
- [ ] Define which information classes must remain aligned across all surfaces:
      main claim, non-claims, evidence-closure rule, supported-path boundary,
      and limitation structure.
- [ ] Define which information classes may legitimately vary by depth:
      literature context, implementation detail, motivation detail, and section
      length.
- [ ] Keep venue-specific formatting and stylistic polish outside the Story 2
      role matrix.

**Evidence produced**
- One stable Task 8 publication-surface role matrix.
- One explicit boundary between aligned content and controlled depth variation.

**Risks / rollback**
- Risk: without a role matrix, later surface edits can drift while still looking
  plausible in isolation.
- Rollback/mitigation: freeze the cross-surface role rule before validating the
  individual files.

### Engineering Task 2: Build A Cross-Surface Claim And Boundary Mapping Table

**Implements story**
- `Story 2: Abstract, Technical Short-Paper, Narrative Short-Paper, And Full-Paper Surfaces Tell The Same Phase 3 Story At Different Depths`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 defines one mapping table showing how the same Paper 2 claim boundary
  appears across all four publication surfaces.
- The table covers the major claim, non-claims, limitation path, and future-work
  boundary without forcing the same paragraph structure everywhere.
- Story 2 makes gaps or conflicts explicit rather than leaving them to manual
  review.

**Execution checklist**
- [ ] Map the main claim across all four publication surfaces.
- [ ] Map explicit non-claims and deferred branches across all four surfaces.
- [ ] Map current benchmark interpretation wording, including diagnosis-grounded
      closure when present.
- [ ] Keep detailed section-to-source traceability outside the Story 2 table so
      Story 3 can own it cleanly.

**Evidence produced**
- One cross-surface claim and boundary mapping table.
- One explicit gap list for missing or conflicting publication-surface wording.

**Risks / rollback**
- Risk: the surfaces may share broad themes while still differing on specific
  boundaries that matter to reviewers.
- Rollback/mitigation: map the key claim and boundary statements explicitly
  before broad prose cleanup.

### Engineering Task 3: Define The Story 2 Surface-Alignment Record Schema And Checker

**Implements story**
- `Story 2: Abstract, Technical Short-Paper, Narrative Short-Paper, And Full-Paper Surfaces Tell The Same Phase 3 Story At Different Depths`

**Change type**
- code | validation automation

**Definition of done**
- Story 2 has one reusable surface-alignment checker.
- The checker records which aligned content classes are present per surface and
  which controlled depth differences are allowed.
- The checker stays focused on claim and boundary alignment rather than on prose
  quality or grammar.

**Execution checklist**
- [ ] Add a Story 2 checker under
      `benchmarks/density_matrix/publication_evidence/`, with
      `surface_alignment_validation.py` as the primary validation surface.
- [ ] Define one stable Story 2 alignment-record schema.
- [ ] Record per-surface presence of required aligned items and allowed role-
      specific items.
- [ ] Keep section-level traceability and bundle-manifest packaging outside the
      Story 2 checker.

**Evidence produced**
- One reusable Story 2 surface-alignment checker.
- One stable Story 2 alignment schema for later Task 8 reuse.

**Risks / rollback**
- Risk: Story 2 can turn into vague editorial review if it lacks one concrete
  alignment surface.
- Rollback/mitigation: validate one structured set of alignment expectations
  rather than relying on manual memory.

### Engineering Task 4: Add A Representative Four-Surface Alignment Matrix

**Implements story**
- `Story 2: Abstract, Technical Short-Paper, Narrative Short-Paper, And Full-Paper Surfaces Tell The Same Phase 3 Story At Different Depths`

**Change type**
- tests | validation automation

**Definition of done**
- Story 2 covers all four mandatory publication surfaces directly.
- The validation matrix is broad enough to catch missing-boundary or
  over-claiming drift early.
- The matrix remains representative and contract-driven rather than exhaustive
  over every sentence in the manuscripts.

**Execution checklist**
- [ ] Include the abstract as the compact claim surface.
- [ ] Include the technical short paper as the compact methods surface.
- [ ] Include the narrative short paper as the positioning surface.
- [ ] Include the full paper as the long-form methods and systems surface.

**Evidence produced**
- One representative Story 2 four-surface alignment matrix.
- One review surface for cross-document claim and limitation alignment.

**Risks / rollback**
- Risk: Story 2 may look correct for one or two files while drift persists in the
  others.
- Rollback/mitigation: validate all four mandatory surfaces as part of Story 2
  closure.

### Engineering Task 5: Add Focused Regression Checks For Cross-Surface Conflict Conditions

**Implements story**
- `Story 2: Abstract, Technical Short-Paper, Narrative Short-Paper, And Full-Paper Surfaces Tell The Same Phase 3 Story At Different Depths`

**Change type**
- tests

**Definition of done**
- Fast checks catch the highest-risk cross-surface conflicts.
- Negative cases prove Story 2 fails when one surface broadens the claim,
  weakens the limitation wording, or omits a mandatory non-claim.
- Regression coverage remains narrow and publication-package focused.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_phase3_task8.py` or a
      tightly related successor for Story 2 surface alignment.
- [ ] Add negative checks for one surface making a stronger technical claim than
      the others.
- [ ] Add negative checks for one surface omitting diagnosis-grounded limitation
      wording when it is required.
- [ ] Keep bibliography completeness and venue-format review outside the fast
      regression layer.

**Evidence produced**
- Focused regression coverage for cross-surface alignment failures.
- Reviewable failures for conflicting or missing publication-surface boundaries.

**Risks / rollback**
- Risk: cross-surface drift is easy to miss in manual editing because each file
  still looks coherent on its own.
- Rollback/mitigation: add targeted tests for the highest-risk conflict classes.

### Engineering Task 6: Emit A Stable Story 2 Surface-Alignment Bundle

**Implements story**
- `Story 2: Abstract, Technical Short-Paper, Narrative Short-Paper, And Full-Paper Surfaces Tell The Same Phase 3 Story At Different Depths`

**Change type**
- validation automation | docs

**Definition of done**
- Story 2 emits one stable machine-reviewable surface-alignment bundle or one
  stable rerunnable checker output.
- The output records which surfaces were checked, which aligned items were
  validated, and which controlled depth differences were accepted.
- The output is stable enough for later Task 8 stories to consume directly.

**Execution checklist**
- [ ] Add one stable Story 2 output location under
      `benchmarks/density_matrix/artifacts/phase3_task8/story2_surface_alignment/`.
- [ ] Emit one artifact such as `publication_surface_alignment_bundle.json`.
- [ ] Record generation command, software metadata, checked file set, and
      alignment summary in the output.
- [ ] Keep the output focused on alignment and controlled depth variation rather
      than on full traceability or evidence closure.

**Evidence produced**
- One stable Story 2 surface-alignment bundle or rerunnable checker output.
- One reusable Story 2 output schema for later Task 8 handoffs.

**Risks / rollback**
- Risk: prose-only Story 2 closure will make later reviewers unable to tell
  whether the four surfaces were actually checked together.
- Rollback/mitigation: emit one machine-reviewable alignment surface directly.

### Engineering Task 7: Document Surface Roles And Run The Story 2 Alignment Gate

**Implements story**
- `Story 2: Abstract, Technical Short-Paper, Narrative Short-Paper, And Full-Paper Surfaces Tell The Same Phase 3 Story At Different Depths`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain what Story 2 aligns, how to rerun it, and how
  it hands off to later Task 8 stories.
- The Story 2 checker and emitted artifact run successfully.
- Story 2 completion is backed by rerunnable alignment evidence rather than by
  editorial confidence alone.

**Execution checklist**
- [ ] Document the Story 2 role matrix and alignment checker.
- [ ] Make the Story 2 rule explicit:
      the four publication surfaces may differ in depth, but not in the frozen
      claim boundary.
- [ ] Explain how Story 2 hands off source traceability to Story 3 and
      evidence-closure interpretation to Story 4.
- [ ] Run focused Story 2 regression checks and verify the emitted Story 2
      bundle or checker output.

**Evidence produced**
- Passing focused checks for Story 2 surface alignment.
- One stable Story 2 output proving aligned Paper 2 surface roles.

**Risks / rollback**
- Risk: Story 2 can look complete while still allowing one surface to drift
  later because the role rule was never documented clearly.
- Rollback/mitigation: document Story 2 as the explicit cross-surface alignment
  gate and rerun it as part of Task 8 review.

## Exit Criteria

Story 2 is complete only when all of the following are true:

- one explicit role matrix defines what each Paper 2 surface is allowed to
  compress or expand,
- the abstract, technical short-paper, narrative short-paper, and full-paper
  surfaces all align on the frozen claim boundary, non-claims, and limitation
  structure,
- cross-surface conflicts fail focused Story 2 checks,
- one stable Story 2 bundle or rerunnable checker captures the alignment
  surface,
- and section-level traceability, evidence closure, supported-path wording,
  manifest packaging, future-work framing, and package-level consistency remain
  clearly assigned to Stories 3 through 8.

## Implementation Notes

- Story 2 is about controlled compression, not document uniformity. The four
  surfaces should not become copies of each other.
- The narrative short paper should stay narratively distinct while remaining
  technically bounded by the same claim package as the technical short paper.
- Treat missing limitation wording as seriously as missing positive-claim
  wording. For Paper 2, honest alignment includes aligned limitations.
- Keep the Story 2 output thin and structural. Later stories need one stable
  alignment surface, not another long manuscript summary.
