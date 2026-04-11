# Story 1 Implementation Plan

## Story Being Implemented

Story 1: Paper 1 Has One Stable Main Claim And Explicit Non-Claims

This is a Layer 4 engineering plan for implementing the first behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story turns the Task 8 claim-boundary requirement into one explicit Paper 1
claim package for Phase 2:

- the main Paper 1 claim is frozen as one honest integration result rather than
  as a moving summary of the whole density-matrix roadmap,
- the supporting claims that can appear beneath that main claim are made
  explicit and kept subordinate to the frozen Phase 2 contribution,
- explicit non-claims prevent later-phase acceleration, optimizer, and
  trainability work from being read as already delivered,
- and the resulting claim package stays narrow enough that later stories can
  align paper surfaces, traceability, evidence closure, and reviewer navigation
  without reopening what Paper 1 actually claims.

Out of scope for this story:

- aligning abstract, short-paper, narrative, and full-paper surfaces at
  different review depths, which is owned by Story 2,
- section-level claim-to-source traceability owned by Story 3,
- evidence-floor and claim-closure interpretation owned by Story 4,
- supported-path and exact-regime wording owned by Story 5,
- future-work and publication-ladder framing owned by Story 6,
- and bundle-level terminology and reviewer-navigation closure owned by Story 7.

## Dependencies And Assumptions

- Task 8 mini-spec and stories are already frozen in `TASK_8_MINI_SPEC.md` and
  `TASK_8_STORIES.md`; Story 1 must not reopen their contract boundaries.
- `docs/density_matrix_project/planning/PUBLICATIONS.md` already defines Paper 1
  as the Phase 2 integration paper and gives the authoritative contribution
  boundary for the first major publication.
- The Phase 2 contract already freezes the scientific boundary through:
  - `DETAILED_PLANNING_PHASE_2.md`,
  - `ADRs_PHASE_2.md`,
  - `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`,
  - `TASK_1_MINI_SPEC.md` through `TASK_8_MINI_SPEC.md`,
  - and the Task 6 workflow-facing publication bundle.
- Existing publication-facing docs
  `ABSTRACT_PHASE_2.md`, `SHORT_PAPER_PHASE_2.md`, `SHORT_PAPER_NARRATIVE.md`,
  and `PAPER_PHASE_2.md` already contain candidate paper wording. Story 1 should
  freeze the claim package they must share rather than let each surface invent
  its own claim set.
- The canonical machine-readable evidence surface already exists at
  `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_publication_bundle.json`.
  Story 1 should treat that surface as the workflow-backed evidence anchor for
  positive claims rather than replace it.
- Task 7 already provides a documentation entry path and consistent support
  vocabulary. Story 1 should reuse that vocabulary and not introduce a parallel
  Phase 2 claim language.
- Story 1 should produce a claim package and validation gate only. It should not
  become a second planning document, a second publication strategy, or a shadow
  ADR set.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Paper 1 Main-Claim And Supporting-Claim Inventory

**Implements story**
- `Story 1: Paper 1 Has One Stable Main Claim And Explicit Non-Claims`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 names one stable main claim for Paper 1.
- Story 1 names one explicit inventory of supporting claims that can appear
  under that main claim.
- The inventory stays aligned with the frozen Phase 2 contribution boundary.

**Execution checklist**
- [ ] Freeze one one-sentence Paper 1 main claim rooted in the delivered Phase 2
      exact noisy integration slice.
- [ ] Freeze one explicit set of allowed supporting claims for backend
      selection, exact `Re Tr(H*rho)` evaluation, generated-`HEA` bridge,
      realistic local-noise support, and publication-grade validation.
- [ ] Keep the claim inventory rooted in the existing Phase 2 contract and Paper
      1 positioning rather than in ad hoc wording from individual draft files.
- [ ] Record the claim inventory in one stable Story 1 planning surface that
      later Task 8 stories can reuse directly.

**Evidence produced**
- One stable Story 1 Paper 1 main-claim inventory.
- One reviewable mapping from allowed supporting claim to Phase 2 source.

**Risks / rollback**
- Risk: if the main claim remains implicit, different paper surfaces can appear
  reasonable while still making different scientific promises.
- Rollback/mitigation: freeze one minimal claim inventory first and require later
  Task 8 work to inherit it unchanged.

### Engineering Task 2: Freeze Explicit Non-Claims For The Paper 1 Boundary

**Implements story**
- `Story 1: Paper 1 Has One Stable Main Claim And Explicit Non-Claims`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 records one explicit non-claim inventory for Paper 1.
- The non-claim inventory makes later-phase work visible without turning it into
  current publication scope.
- Missing or contradictory non-claims are treated as Story 1 incompleteness.

**Execution checklist**
- [ ] Define explicit non-claims for density-aware partitioning, fusion,
      gradient-path completion, approximate scaling, broader optimizer studies,
      and trainability analysis.
- [ ] Distinguish non-claims from future-work motivation so later-phase value can
      be stated without weakening the current boundary.
- [ ] Keep the non-claim list aligned with the roadmap order already frozen in
      the planning and ADR set.
- [ ] Treat omitted high-risk non-claims as a real Story 1 failure condition.

**Evidence produced**
- One stable Story 1 non-claim inventory for Paper 1.
- One reviewable list of high-risk future topics that must stay outside the main
  claim.

**Risks / rollback**
- Risk: without explicit non-claims, readers can import later-phase work into
  Paper 1 even when the positive claim text stays narrow.
- Rollback/mitigation: define the exclusions directly and validate their presence
  wherever Paper 1 scope is summarized.

### Engineering Task 3: Reuse Existing Phase 2 Contract Wording Instead Of Inventing A New Claim Language

**Implements story**
- `Story 1: Paper 1 Has One Stable Main Claim And Explicit Non-Claims`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 reuses frozen Phase 2 contract wording wherever practical.
- Readers can trace claim language directly to planning, ADR, mini-spec, and
  evidence surfaces without semantic translation.
- Story 1 remains a claim-freezing layer, not a second vocabulary.

**Execution checklist**
- [ ] Reuse the frozen wording for `density_matrix`, exact `Re Tr(H*rho)`,
      generated-`HEA`, canonical workflow, exact regime, acceptance anchor, and
      required / optional / deferred / unsupported.
- [ ] Reuse the Paper 1 positioning language already present in
      `PUBLICATIONS.md` where it matches the frozen Phase 2 contract.
- [ ] Avoid Story 1-only synonyms for the main contribution when existing Phase 2
      wording is already adequate.
- [ ] Keep any necessary paraphrase explicitly mapped back to the canonical
      wording source.

**Evidence produced**
- One Story 1 claim package rooted in canonical Phase 2 vocabulary.
- Reviewable traceability from claim wording to the authoritative contract docs.

**Risks / rollback**
- Risk: a new claim vocabulary can quietly broaden or soften the meaning of the
  Phase 2 result even without adding new sentences.
- Rollback/mitigation: preserve the frozen contract wording unless a very small
  clarity improvement is explicitly mapped and justified.

### Engineering Task 4: Encode Coverage Rules For Mandatory Positive Claims And Mandatory Non-Claims

**Implements story**
- `Story 1: Paper 1 Has One Stable Main Claim And Explicit Non-Claims`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 makes explicit which positive claims and non-claims must be present in
  the Paper 1 claim package.
- Missing mandatory claim elements or missing mandatory non-claims block Story 1
  closure.
- Coverage rules are stable enough for fast checks and later Task 8 bundle
  review.

**Execution checklist**
- [ ] Mark the main Paper 1 claim and the allowed supporting-claim inventory as
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

### Engineering Task 5: Add Focused Regression Checks For Missing Or Inflated Claim Wording

**Implements story**
- `Story 1: Paper 1 Has One Stable Main Claim And Explicit Non-Claims`

**Change type**
- tests

**Definition of done**
- Fast checks catch missing main-claim elements, missing non-claims, or obvious
  overbroad Paper 1 wording.
- Negative cases show that Story 1 fails if later-phase results are described as
  current Paper 1 output.
- Regression coverage remains small and publication-package focused.

**Execution checklist**
- [ ] Add focused checks in `tests/density_matrix/test_phase2_publication_docs.py`
      or a tightly related successor for Story 1 claim-package coverage.
- [ ] Add negative checks for omitted main-claim elements or omitted mandatory
      non-claims.
- [ ] Add at least one check for future-work language being written as current
      Paper 1 result.
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

### Engineering Task 6: Emit One Stable Story 1 Claim-Package Summary Or Checker

**Implements story**
- `Story 1: Paper 1 Has One Stable Main Claim And Explicit Non-Claims`

**Change type**
- validation automation | docs

**Definition of done**
- Story 1 can emit one stable machine-readable claim package or one stable
  rerunnable checker for the Paper 1 main claim.
- The output records the main claim, allowed supporting claims, mandatory
  non-claims, and source links.
- The output is stable enough for later Task 8 stories and paper-facing review
  to consume directly.

**Execution checklist**
- [ ] Add one Story 1 command, script, or checker
      (for example under `benchmarks/density_matrix/`) for claim-package
      summary emission.
- [ ] Emit one stable artifact in a Task 8 artifact directory such as
      `benchmarks/density_matrix/artifacts/publication_claim_package/claim_package.json`.
- [ ] Record source references, generation command, and scope notes in the
      output.
- [ ] Keep the output narrow to claim-boundary and non-claim semantics rather
      than full paper-surface alignment.

**Evidence produced**
- One stable Task 8 Story 1 claim-package summary or rerunnable checker.
- One reusable Story 1 output schema for later Task 8 handoffs.

**Risks / rollback**
- Risk: ad hoc prose edits are hard to compare and easy to overread.
- Rollback/mitigation: define one thin structured Story 1 output that makes the
  allowed Paper 1 claim boundary explicit.

### Engineering Task 7: Document Story 1 Claim-Boundary Rules And Handoff To Story 2 And Story 4

**Implements story**
- `Story 1: Paper 1 Has One Stable Main Claim And Explicit Non-Claims`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Story 1 freezes, how to rerun it, and why
  it is the canonical claim-boundary gate for Task 8.
- The notes make clear that Story 1 closes claim scope but not cross-surface
  alignment or evidence-closure interpretation.
- The documentation stays aligned with the frozen Phase 2 publication boundary.

**Execution checklist**
- [ ] Document the Story 1 summary or checker and how it relates to
      `PUBLICATIONS.md`, Task 6 evidence, and Task 7 documentation outputs.
- [ ] Make the main Story 1 rule explicit:
      one stable main claim plus explicit non-claims must be present.
- [ ] Explain how Story 1 hands off cross-surface alignment to Story 2 and
      evidence-closure interpretation to Story 4.
- [ ] Keep later-phase opportunity statements clearly outside Story 1 closure.

**Evidence produced**
- Updated developer-facing guidance for the Task 8 Story 1 claim-boundary gate.
- One stable location where Story 1 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 1 is poorly documented, later editing can silently treat claim
  expansion as harmless drafting rather than as a contract failure.
- Rollback/mitigation: document Story 1 as the explicit Paper 1 claim gate and
  keep its handoff boundaries visible.

### Engineering Task 8: Run Story 1 Validation And Confirm Honest Paper 1 Claim Closure

**Implements story**
- `Story 1: Paper 1 Has One Stable Main Claim And Explicit Non-Claims`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 1 claim-boundary checks pass.
- The Story 1 summary or checker runs successfully and emits stable output.
- Story 1 completion is backed by rerunnable publication evidence rather than by
  editorial confidence alone.

**Execution checklist**
- [ ] Run focused Story 1 regression checks for main-claim and non-claim
      coverage.
- [ ] Run the Story 1 summary or checker command and verify emitted output.
- [ ] Confirm that the main claim, allowed supporting claims, and mandatory
      non-claims are all present and aligned with the frozen Phase 2 contract.
- [ ] Record stable test and artifact references for Story 2 and later Task 8
      work.

**Evidence produced**
- Passing focused checks for Story 1 claim-boundary completeness.
- One stable Story 1 output proving the honest Paper 1 claim package.

**Risks / rollback**
- Risk: Story 1 can look complete while still letting inflated or incomplete
  Paper 1 claims survive in draft surfaces.
- Rollback/mitigation: require passing checks plus one stable emitted output
  before closing Story 1.

## Exit Criteria

Story 1 is complete only when all of the following are true:

- one stable main-claim inventory defines the Paper 1 contribution boundary,
- one stable non-claim inventory keeps later-phase work outside the current
  Paper 1 result,
- allowed supporting claims remain subordinate to the frozen Phase 2
  integration milestone,
- missing mandatory main-claim or non-claim elements fail Story 1 completeness
  checks,
- one stable Story 1 output or rerunnable checker captures the claim package,
- and cross-surface alignment, section-level traceability, evidence closure,
  supported-path wording, future-work framing, and publication-package
  consistency remain clearly assigned to Stories 2 to 7.

## Implementation Notes

- Start from the smallest honest claim, not from the biggest exciting phrasing.
  Story 1 should protect Paper 1 from roadmap inflation.
- Prefer reuse of Phase 2 contract wording over fresh slogans. Story 1 is a
  claim-freezing layer, not a branding exercise.
- Treat explicit non-claims as first-class requirements. For Paper 1, what is
  excluded is as important as what is included.
- Keep the Story 1 output thin and structured. Later stories need a stable claim
  package they can consume, not another long narrative draft.
- If Story 1 is strong, later editing of abstract, short-paper, narrative, and
  full-paper surfaces becomes an alignment problem rather than a claim-discovery
  problem.
