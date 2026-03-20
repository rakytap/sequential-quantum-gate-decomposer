# Story 1 Implementation Plan

## Story Being Implemented

Story 1: Paper 2 Has One Stable Main Claim, Supporting Claim Set, And Explicit
Non-Claims

This is a Layer 4 engineering plan for implementing the first behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story turns the Task 8 claim-boundary requirement into one explicit Paper 2
claim package for Phase 3:

- the main Paper 2 claim is frozen as one honest methods result rather than as a
  moving summary of the whole density-matrix roadmap,
- the supporting claims that can appear beneath that main claim are made
  explicit and kept subordinate to the frozen Phase 3 contribution,
- explicit non-claims prevent later-phase optimizer, workflow-growth, and
  approximate-scaling work from being read as already delivered,
- and the resulting claim package stays narrow enough that later stories can
  align surfaces, traceability, evidence closure, manifest packaging, and
  reviewer navigation without reopening what Paper 2 actually claims.

Out of scope for this story:

- cross-surface alignment across abstract, technical short-paper, narrative
  short-paper, and full-paper surfaces, which is owned by Story 2,
- section-level claim-to-source traceability owned by Story 3,
- evidence-floor and threshold-or-diagnosis interpretation owned by Story 4,
- supported-path, no-fallback, bounded planner-claim, and benchmark-honesty
  wording owned by Story 5,
- manifest-driven reviewer packaging owned by Story 6,
- future-work and publication-ladder framing owned by Story 7,
- and package-level terminology, reviewer-entry, and summary-consistency
  integrity owned by Story 8.

## Dependencies And Assumptions

- Task 8 mini-spec and stories are already frozen in `TASK_8_MINI_SPEC.md` and
  `TASK_8_STORIES.md`; Story 1 must not reopen their contract boundaries.
- `docs/density_matrix_project/planning/PUBLICATIONS.md` already defines Paper 2
  as the major Phase 3 methods and systems paper and gives the authoritative
  contribution boundary for the second major publication.
- The Phase 3 contract already freezes the scientific boundary through:
  - `DETAILED_PLANNING_PHASE_3.md`,
  - `ADRs_PHASE_3.md`,
  - `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`,
  - `TASK_1_MINI_SPEC.md` through `TASK_8_MINI_SPEC.md`,
  - the emitted Task 6 correctness and unsupported-boundary bundles,
  - and the emitted Task 7 benchmark and diagnosis bundles.
- Existing publication-facing docs `ABSTRACT_PHASE_3.md`,
  `SHORT_PAPER_PHASE_3.md`, `SHORT_PAPER_NARRATIVE.md`, and
  `PAPER_PHASE_3.md` already contain candidate Paper 2 wording. Story 1 should
  freeze the claim package they must share rather than let each surface invent
  its own claim set.
- Story 5, not Story 1, owns the detailed supported-path, no-fallback, bounded
  planner-claim, and current-count honesty checks. Story 1 may name those items
  as supporting claims only at the boundary level.
- The natural implementation home for Task 8 claim-boundary validation is a
  dedicated `benchmarks/density_matrix/publication_evidence/` package, with
  `claim_package_validation.py` as the Story 1 validation surface and emitted
  artifacts rooted in `benchmarks/density_matrix/artifacts/phase3_task8/`.
- Story 1 should produce a claim package and validation gate only. It should not
  become a second planning document, a second publication strategy, or a shadow
  ADR set.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Paper 2 Main-Claim And Supporting-Claim Inventory

**Implements story**
- `Story 1: Paper 2 Has One Stable Main Claim, Supporting Claim Set, And Explicit Non-Claims`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 names one stable main claim for Paper 2.
- Story 1 names one explicit inventory of supporting claims that can appear
  beneath that main claim.
- The inventory stays aligned with the frozen Phase 3 contribution boundary.

**Execution checklist**
- [ ] Freeze one one-sentence Paper 2 main claim rooted in the delivered Phase 3
      noise-aware partitioning and limited-fusion slice.
- [ ] Freeze one explicit set of allowed supporting claims for canonical noisy
      planner input, semantic preservation, executable partitioned runtime, at
      least one real fused path, bounded benchmark-calibrated Task 5 planning,
      and the machine-reviewable Task 6 plus Task 7 evidence package.
- [ ] Keep the claim inventory rooted in the existing Phase 3 contract and Paper
      2 positioning rather than in ad hoc wording from individual draft files.
- [ ] Record the claim inventory in one stable Story 1 planning surface that
      later Task 8 stories can reuse directly.

**Evidence produced**
- One stable Story 1 Paper 2 main-claim inventory.
- One reviewable mapping from allowed supporting claim to Phase 3 source.

**Risks / rollback**
- Risk: if the main claim remains implicit, different paper surfaces can appear
  reasonable while still making different scientific promises.
- Rollback/mitigation: freeze one minimal claim inventory first and require later
  Task 8 work to inherit it unchanged.

### Engineering Task 2: Freeze Explicit Non-Claims For The Paper 2 Boundary

**Implements story**
- `Story 1: Paper 2 Has One Stable Main Claim, Supporting Claim Set, And Explicit Non-Claims`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 records one explicit non-claim inventory for Paper 2.
- The non-claim inventory makes later-phase work visible without turning it into
  current publication scope.
- Missing or contradictory non-claims are treated as Story 1 incompleteness.

**Execution checklist**
- [ ] Define explicit non-claims for fully channel-native fused noisy blocks,
      broader noisy VQE/VQA workflow growth, density-backend gradients,
      optimizer studies, approximate scaling, and full direct `qgd_Circuit`
      parity.
- [ ] Distinguish non-claims from future-work motivation so later-phase value can
      be stated without weakening the current boundary.
- [ ] Keep the non-claim list aligned with the roadmap order already frozen in
      the planning and ADR set.
- [ ] Treat omitted high-risk non-claims as a real Story 1 failure condition.

**Evidence produced**
- One stable Story 1 non-claim inventory for Paper 2.
- One reviewable list of high-risk future topics that must stay outside the main
  claim.

**Risks / rollback**
- Risk: without explicit non-claims, readers can import later-phase work into
  Paper 2 even when the positive claim text stays narrow.
- Rollback/mitigation: define the exclusions directly and validate their presence
  wherever Paper 2 scope is summarized.

### Engineering Task 3: Reuse Existing Phase 3 Contract Wording Instead Of Inventing A New Claim Language

**Implements story**
- `Story 1: Paper 2 Has One Stable Main Claim, Supporting Claim Set, And Explicit Non-Claims`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 reuses frozen Phase 3 contract wording wherever practical.
- Readers can trace claim language directly to planning, ADR, mini-spec, and
  emitted evidence surfaces without semantic translation.
- Story 1 remains a claim-freezing layer, not a second vocabulary.

**Execution checklist**
- [ ] Reuse the frozen wording for exact noisy mixed-state circuits, canonical
      noisy planner surface, partitioned density runtime, real fused path,
      counted supported, diagnosis-grounded closure, and required / optional /
      deferred / unsupported.
- [ ] Reuse the Paper 2 positioning language already present in
      `PUBLICATIONS.md` where it matches the frozen Phase 3 contract.
- [ ] Avoid Story 1-only synonyms for the main contribution when existing Phase 3
      wording is already adequate.
- [ ] Keep any necessary paraphrase explicitly mapped back to the canonical
      wording source.

**Evidence produced**
- One Story 1 claim package rooted in canonical Phase 3 vocabulary.
- Reviewable traceability from claim wording to the authoritative contract docs.

**Risks / rollback**
- Risk: a new claim vocabulary can quietly broaden or soften the meaning of the
  Phase 3 result even without adding new sentences.
- Rollback/mitigation: preserve the frozen contract wording unless a very small
  clarity improvement is explicitly mapped and justified.

### Engineering Task 4: Encode Coverage Rules For Mandatory Positive Claims And Mandatory Non-Claims

**Implements story**
- `Story 1: Paper 2 Has One Stable Main Claim, Supporting Claim Set, And Explicit Non-Claims`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 makes explicit which positive claims and non-claims must be present in
  the Paper 2 claim package.
- Missing mandatory claim elements or missing mandatory non-claims block Story 1
  closure.
- Coverage rules are stable enough for fast checks and later Task 8 bundle
  review.

**Execution checklist**
- [ ] Mark the main Paper 2 claim and the allowed supporting-claim inventory as
      mandatory covered items.
- [ ] Mark the later-phase exclusions with highest overclaim risk as mandatory
      non-claim items.
- [ ] Distinguish mandatory claim-package items from optional context, venue
      shaping, or literature framing.
- [ ] Keep the coverage rules machine-checkable where practical.

**Evidence produced**
- One explicit completeness rule set for Story 1 claim coverage.
- One reviewable list of mandatory claim and non-claim elements.

**Risks / rollback**
- Risk: the paper package can sound coherent while omitting one crucial scope
  boundary that reviewers would assume is implied.
- Rollback/mitigation: treat missing mandatory claim-package elements as a real
  Story 1 failure instead of as editorial polish work.

### Engineering Task 5: Build The Story 1 Claim-Package Checker And Record Schema

**Implements story**
- `Story 1: Paper 2 Has One Stable Main Claim, Supporting Claim Set, And Explicit Non-Claims`

**Change type**
- code | validation automation

**Definition of done**
- Story 1 has one reusable checker for claim-package completeness.
- The checker records the main claim, allowed supporting claims, mandatory
  non-claims, and source references through one stable schema.
- The checker is thin enough to stay focused on claim boundary rather than on
  cross-surface editorial alignment.

**Execution checklist**
- [ ] Add a Story 1 checker under
      `benchmarks/density_matrix/publication_evidence/`, with
      `claim_package_validation.py` as the primary validation surface.
- [ ] Define one stable claim-package record schema in the same package or the
      smallest auditable successor.
- [ ] Record source references back to the Phase 3 contract docs and relevant
      draft-paper surfaces.
- [ ] Keep detailed benchmark-honesty and manifest packaging semantics outside
      the Story 1 checker.

**Evidence produced**
- One reusable Story 1 claim-package checker.
- One stable Story 1 claim-package schema for later Task 8 reuse.

**Risks / rollback**
- Risk: ad hoc prose edits are hard to compare and easy to overread.
- Rollback/mitigation: define one thin structured Story 1 output that makes the
  allowed Paper 2 claim boundary explicit.

### Engineering Task 6: Add Focused Regression Checks For Missing Or Inflated Claim Wording

**Implements story**
- `Story 1: Paper 2 Has One Stable Main Claim, Supporting Claim Set, And Explicit Non-Claims`

**Change type**
- tests

**Definition of done**
- Fast checks catch missing main-claim elements, missing non-claims, or obvious
  overbroad Paper 2 wording.
- Negative cases show that Story 1 fails if later-phase results are described as
  current Paper 2 output.
- Regression coverage remains small and publication-package focused.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_phase3_task8.py` or a
      tightly related successor for Story 1 claim-package coverage.
- [ ] Add negative checks for omitted main-claim elements or omitted mandatory
      non-claims.
- [ ] Add at least one check for future-work language being written as current
      Paper 2 result.
- [ ] Keep full manuscript review and venue polish outside this fast regression
      layer.

**Evidence produced**
- Focused regression coverage for Story 1 claim-boundary completeness.
- Reviewable failures when mandatory claim or non-claim statements are missing.

**Risks / rollback**
- Risk: claim-boundary regressions do not break code and therefore can survive
  until external review.
- Rollback/mitigation: lock the critical claim statements down with targeted
  tests before broader publication editing continues.

### Engineering Task 7: Emit A Stable Story 1 Claim-Package Bundle

**Implements story**
- `Story 1: Paper 2 Has One Stable Main Claim, Supporting Claim Set, And Explicit Non-Claims`

**Change type**
- validation automation | docs

**Definition of done**
- Story 1 emits one stable machine-reviewable claim package or one stable
  rerunnable checker output for the Paper 2 main claim.
- The output records the main claim, supporting-claim inventory, mandatory
  non-claims, and source links.
- The output is stable enough for later Task 8 stories and reviewer-facing
  packaging to consume directly.

**Execution checklist**
- [ ] Add one stable Story 1 output location under
      `benchmarks/density_matrix/artifacts/phase3_task8/story1_claim_package/`.
- [ ] Emit one artifact such as `claim_package_bundle.json`.
- [ ] Record generation command, software metadata, and scope notes in the
      output.
- [ ] Keep the output narrow to claim-boundary and non-claim semantics rather
      than full paper-surface alignment.

**Evidence produced**
- One stable Story 1 claim-package bundle or rerunnable checker output.
- One reusable Story 1 output schema for later Task 8 handoffs.

**Risks / rollback**
- Risk: prose-only Story 1 closure will make later validation and publication
  work hard to audit.
- Rollback/mitigation: emit one thin machine-reviewable claim-boundary surface
  directly.

### Engineering Task 8: Document Story 1 Claim-Boundary Rules And Run The Story 1 Gate

**Implements story**
- `Story 1: Paper 2 Has One Stable Main Claim, Supporting Claim Set, And Explicit Non-Claims`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain what Story 1 freezes, how to rerun it, and why
  it is the canonical claim-boundary gate for Task 8.
- The Story 1 checker and artifact run successfully.
- Story 1 completion is backed by rerunnable publication evidence rather than by
  editorial confidence alone.

**Execution checklist**
- [ ] Document the Story 1 bundle or checker and how it relates to
      `PUBLICATIONS.md`, the Phase 3 contract, and later Task 8 stories.
- [ ] Make the main Story 1 rule explicit:
      one stable main claim plus explicit non-claims must be present.
- [ ] Explain how Story 1 hands off surface alignment to Story 2 and evidence-
      closure interpretation to Story 4.
- [ ] Run focused Story 1 regression checks and verify the emitted Story 1
      bundle or checker output.

**Evidence produced**
- Passing focused checks for Story 1 claim-boundary completeness.
- One stable Story 1 output proving the honest Paper 2 claim package.

**Risks / rollback**
- Risk: Story 1 can look complete while still letting inflated or incomplete
  Paper 2 claims survive in draft surfaces.
- Rollback/mitigation: require passing checks plus one stable emitted output
  before closing Story 1.

## Exit Criteria

Story 1 is complete only when all of the following are true:

- one stable main-claim inventory defines the Paper 2 contribution boundary,
- one stable non-claim inventory keeps later-phase work outside the current
  Paper 2 result,
- allowed supporting claims remain subordinate to the frozen Phase 3 methods
  milestone,
- missing mandatory main-claim or non-claim elements fail Story 1 completeness
  checks,
- one stable Story 1 bundle or rerunnable checker captures the claim package,
- and cross-surface alignment, section-level traceability, evidence closure,
  supported-path wording, manifest packaging, future-work framing, and package-
  consistency integrity remain clearly assigned to Stories 2 through 8.

## Implementation Notes

- Start from the smallest honest claim, not from the broadest exciting phrasing.
  Story 1 should protect Paper 2 from roadmap inflation.
- Prefer reuse of Phase 3 contract wording over fresh slogans. Story 1 is a
  claim-freezing layer, not a branding exercise.
- Treat explicit non-claims as first-class requirements. For Paper 2, what is
  excluded is as important as what is included.
- Keep the Story 1 output thin and structured. Later stories need a stable claim
  package they can consume, not another long narrative draft.
- If Story 1 is strong, later editing of abstract, short-paper, narrative, and
  full-paper surfaces becomes an alignment problem rather than a claim-discovery
  problem.
