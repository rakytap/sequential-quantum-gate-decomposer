# Story 1 Implementation Plan

## Story Being Implemented

Story 1: The Phase 2 Documentation Contract Is Discoverable Through One
Authoritative, Citable Source-Of-Truth Path

This is a Layer 4 engineering plan for implementing the first behavioral slice
from `TASK_7_STORIES.md`.

## Scope

This story turns the Task 7 discoverability goal into one explicit contract map
for Phase 2 documentation:

- one authoritative source-of-truth path is exposed for the frozen Phase 2
  contract rather than leaving readers to infer authority from scattered files,
- the distinct roles of planning, ADR, checklist, mini-spec, story, validation,
  and publication-facing surfaces are made explicit,
- stable reference entry points are defined so reviewers and paper drafts can
  cite the same contract boundary consistently,
- and the resulting discoverability layer stays thin enough that later stories
  can clarify supported entry semantics, support-surface boundaries, evidence
  rules, future-work separation, and terminology consistency without reopening
  what counts as the authoritative document path.

Out of scope for this story:

- rewriting or broadening the frozen Phase 2 contract itself,
- supported backend-selection wording and canonical workflow explanation owned
  by Story 2,
- required versus optional / deferred / unsupported support-surface semantics
  owned by Story 3,
- evidence-bar interpretation owned by Story 4,
- roadmap and future-work separation owned by Story 5,
- and final terminology and cross-bundle consistency closure owned by Story 6.

## Dependencies And Assumptions

- Task 7 mini-spec and stories are already frozen in `TASK_7_MINI_SPEC.md` and
  `TASK_7_STORIES.md`; Story 1 must not reopen their contract boundaries.
- The Phase 2 contract already exists in a layered form and should be indexed,
  not reauthored, through:
  - `DETAILED_PLANNING_PHASE_2.md`,
  - `ADRs_PHASE_2.md`,
  - `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`,
  - `TASK_1_MINI_SPEC.md` through `TASK_7_MINI_SPEC.md`,
  - and story / implementation-plan outputs where they are the authoritative
    explanation for a narrower behavioral slice.
- Task 1 through Task 6 already define the concrete backend, observable, bridge,
  support, validation, and workflow contracts that Task 7 must help readers
  find rather than duplicate.
- The Phase 2 paper-facing documents
  `ABSTRACT_PHASE_2.md`, `SHORT_PAPER_PHASE_2.md`, and `PAPER_PHASE_2.md`
  already need a stable citation path for Phase 2 claims and should reuse Story
  1 outputs instead of inventing separate reference wording.
- The existing Task 6 publication bundle under
  `benchmarks/density_matrix/artifacts/phase2_task6/` already provides the
  canonical machine-readable evidence surface for workflow-backed claims. Story
  1 should point to that surface where appropriate instead of replacing it.
- Story 1 should produce a discoverability and reference map only. It should not
  become a second planning document or a shadow ADR layer.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Phase 2 Documentation Inventory And Role Taxonomy

**Implements story**
- `Story 1: The Phase 2 Documentation Contract Is Discoverable Through One Authoritative, Citable Source-Of-Truth Path`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 names one canonical inventory of document classes that make up the
  Phase 2 contract.
- Each class has one explicit role in the source-of-truth hierarchy.
- The inventory stays aligned with existing Phase 2 documents rather than
  inventing a second taxonomy.

**Execution checklist**
- [ ] Freeze one canonical inventory covering planning, ADR, checklist,
      mini-spec, stories, implementation plans, validation artifacts, and
      publication-facing references.
- [ ] Define one explicit role for each inventory class:
      phase contract, task contract, behavioral slice, engineering plan,
      evidence surface, or publication-facing citation surface.
- [ ] Keep the inventory rooted in existing file structure and frozen Phase 2
      documentation rather than ad hoc reviewer notes.
- [ ] Record the inventory in one stable implementation-facing location that
      later Task 7 stories can reference directly.

**Evidence produced**
- One stable Task 7 Story 1 documentation inventory and role taxonomy.
- One reviewable mapping from document class to authority role.

**Risks / rollback**
- Risk: if document roles remain implicit, reviewers can cite different files as
  authoritative for the same claim.
- Rollback/mitigation: freeze one explicit inventory and role map early and
  require later Task 7 outputs to reference it.

### Engineering Task 2: Reuse Existing Phase 2 Contract Surfaces Without Duplicating Their Semantics

**Implements story**
- `Story 1: The Phase 2 Documentation Contract Is Discoverable Through One Authoritative, Citable Source-Of-Truth Path`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 points readers to existing authoritative files instead of copying
  contract text into a new pseudo-source.
- Reused surfaces preserve current wording wherever practical so reviewers can
  trace claims directly to the canonical files.
- Story 1 remains a navigation and discoverability layer, not a second contract.

**Execution checklist**
- [ ] Reuse the current Phase 2 planning, ADR, checklist, mini-spec, and paper
      surfaces as authoritative references rather than restating their content
      wholesale.
- [ ] Reuse the Task 6 publication bundle as the authoritative workflow-evidence
      surface where workflow-backed claims need a machine-readable reference.
- [ ] Avoid introducing Story 1-only synonyms when an existing document already
      defines the needed concept.
- [ ] Keep copied summary text minimal and explicitly subordinate to the
      referenced authoritative document.

**Evidence produced**
- One Story 1 discoverability layer rooted in existing Phase 2 references.
- Reviewable traceability from summary-level guidance to authoritative files.

**Risks / rollback**
- Risk: duplicated contract prose can drift from the planning and ADR set even if
  the underlying decisions stay frozen.
- Rollback/mitigation: treat Story 1 as a thin reference map and let the
  canonical documents remain the only decision-bearing surfaces.

### Engineering Task 3: Encode Explicit Topic-To-Source Mapping For The Frozen Phase 2 Contract

**Implements story**
- `Story 1: The Phase 2 Documentation Contract Is Discoverable Through One Authoritative, Citable Source-Of-Truth Path`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 records one explicit mapping from major Phase 2 contract topics to the
  files that authoritatively define them.
- The mapping covers the major Task 7 areas:
  backend selection, observable scope, bridge scope, support matrix, workflow
  anchor, benchmark minimum, numeric thresholds, and non-goals.
- Missing topic coverage is treated as Story 1 incompleteness rather than as an
  acceptable documentation gap.

**Execution checklist**
- [ ] Define a stable topic list for major Phase 2 contract areas.
- [ ] Map each topic to one primary authoritative reference and any necessary
      supporting references.
- [ ] Mark whether the topic is phase-level, task-level, evidence-level, or
      paper-facing for review purposes.
- [ ] Add completeness checks so uncovered mandatory topics fail Story 1 review.

**Evidence produced**
- One explicit topic-to-source map for the frozen Phase 2 contract.
- One completeness rule stating which mandatory topics must always be present.

**Risks / rollback**
- Risk: without topic-level mapping, readers can find the documents but still
  miss where a specific claim is actually defined.
- Rollback/mitigation: make topic coverage explicit and validate it as part of
  Story 1 completion.

### Engineering Task 4: Add One Stable Citable Entry Point And Cross-Reference Navigation Surface

**Implements story**
- `Story 1: The Phase 2 Documentation Contract Is Discoverable Through One Authoritative, Citable Source-Of-Truth Path`

**Change type**
- docs

**Definition of done**
- Story 1 exposes one stable entry point or equivalent reference surface that
  reviewers can cite for the Phase 2 documentation contract.
- The entry point links cleanly to lower-level authoritative files without
  claiming authority it does not own.
- Cross-document navigation is explicit enough that readers can move from the
  entry point to the exact contract layer they need.

**Execution checklist**
- [ ] Add one stable entry-point document or equivalent summary surface for the
      Phase 2 contract map.
- [ ] Link that surface to the planning, ADR, checklist, task mini-spec, and
      evidence surfaces needed for Phase 2 claims.
- [ ] Make explicit when the entry point is summarizing versus when it is the
      canonical place to start.
- [ ] Keep the entry point narrow to discoverability and navigation rather than
      to broad contract duplication.

**Evidence produced**
- One stable Phase 2 contract entry point for Task 7 Story 1.
- Explicit cross-reference paths from summary-level guidance to canonical files.

**Risks / rollback**
- Risk: an uncitable or unstable entry point forces paper and review work back to
  ad hoc file-by-file explanation.
- Rollback/mitigation: freeze one stable starting surface and keep its role
  tightly limited to navigation and discoverability.

### Engineering Task 5: Add Focused Regression Checks For Missing Or Conflicting Source-Of-Truth Coverage

**Implements story**
- `Story 1: The Phase 2 Documentation Contract Is Discoverable Through One Authoritative, Citable Source-Of-Truth Path`

**Change type**
- tests

**Definition of done**
- Fast checks catch missing mandatory topic coverage, missing reference targets,
  or conflicting authority assignments in the Story 1 map.
- Negative cases show that incomplete discoverability surfaces fail clearly.
- Regression coverage remains lightweight and documentation-focused.

**Execution checklist**
- [ ] Add focused checks in a lightweight documentation-consistency test surface
      such as `tests/density_matrix/test_phase2_docs.py` or a tightly related
      successor.
- [ ] Add negative checks for missing mandatory topics, duplicate primary
      authorities, or broken reference targets.
- [ ] Add at least one check showing that Story 1 cannot pass if a major Phase 2
      contract area has no authoritative mapping.
- [ ] Keep full document-bundle review outside this fast regression layer.

**Evidence produced**
- Focused regression coverage for Story 1 discoverability completeness.
- Reviewable failure messages for missing or conflicting source-of-truth mapping.

**Risks / rollback**
- Risk: discoverability regressions can remain invisible until paper drafting or
  external review.
- Rollback/mitigation: add small deterministic checks that fail as soon as the
  mapping becomes incomplete or contradictory.

### Engineering Task 6: Emit One Stable Story 1 Contract-Reference Map Artifact Or Rerunnable Checker

**Implements story**
- `Story 1: The Phase 2 Documentation Contract Is Discoverable Through One Authoritative, Citable Source-Of-Truth Path`

**Change type**
- validation automation | docs

**Definition of done**
- Story 1 can emit one stable machine-readable reference map or one stable
  rerunnable checker that summarizes the authoritative documentation path.
- The output records document roles, topic mappings, and entry-point references.
- The output shape is stable enough for Story 6 and publication-facing surfaces
  to reference directly.

**Execution checklist**
- [ ] Add one Story 1 entry command, script, or checker
      (for example under `benchmarks/density_matrix/`) for reference-map
      emission.
- [ ] Emit one stable artifact in a Task 7 artifact directory
      (for example `benchmarks/density_matrix/artifacts/phase2_task7/`).
- [ ] Record generation command, suite identity, and reference metadata with the
      emitted output.
- [ ] Keep the emitted output narrow to discoverability and authority mapping
      rather than broader contract interpretation.

**Evidence produced**
- One stable Task 7 Story 1 contract-reference map artifact or rerunnable
  checker.
- One reusable output schema for later Task 7 handoffs.

**Risks / rollback**
- Risk: prose-only discoverability updates are hard to reuse and easy to drift.
- Rollback/mitigation: emit one thin structured output that later documentation
  and publication tooling can consume directly.

### Engineering Task 7: Document The Story 1 Discoverability Gate And Its Hand-Off To Story 2

**Implements story**
- `Story 1: The Phase 2 Documentation Contract Is Discoverable Through One Authoritative, Citable Source-Of-Truth Path`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Story 1 validates, how to rerun it, and
  why it is the canonical discoverability gate for Task 7.
- The notes make clear that Story 1 sits above raw document sprawl and below the
  supported-entry and workflow-clarity work owned by Story 2.
- Documentation stays aligned with the frozen Phase 2 hierarchy and does not
  overstate Story 1 as a new contract layer.

**Execution checklist**
- [ ] Document the Story 1 reference map or checker and its relationship to the
      planning, ADR, checklist, and mini-spec layers.
- [ ] Explain the main Story 1 rule:
      every mandatory Phase 2 contract topic must have a stable authoritative
      reference path.
- [ ] Explain how Story 1 hands off supported-entry semantics to Story 2.
- [ ] Keep the notes explicit that Story 1 improves discoverability but does not
      itself redefine backend or workflow behavior.

**Evidence produced**
- Updated developer-facing guidance for the Task 7 Story 1 discoverability gate.
- One stable place where Story 1 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 1 is poorly documented, reviewers may still treat it as a loose
  helper rather than as the canonical navigation surface.
- Rollback/mitigation: tie the notes directly to the same structured output or
  checker used for Story 1 validation.

### Engineering Task 8: Run Story 1 Validation And Confirm Source-Of-Truth Discoverability

**Implements story**
- `Story 1: The Phase 2 Documentation Contract Is Discoverable Through One Authoritative, Citable Source-Of-Truth Path`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 1 checks pass for topic coverage, reference integrity, and entry
  point stability.
- The Story 1 artifact emission path or checker runs successfully and produces
  stable output.
- Story 1 completion is backed by rerunnable evidence rather than by prose-only
  updates.

**Execution checklist**
- [ ] Run focused Story 1 regression checks for topic coverage and authority
      mapping.
- [ ] Run the Story 1 reference-map emission or checker command and verify output.
- [ ] Confirm mandatory contract topics, document roles, and entry-point
      references are present and complete.
- [ ] Record stable test and artifact references for Story 2 and later Task 7
      work.

**Evidence produced**
- Passing focused checks for Story 1 discoverability completeness.
- One stable Story 1 artifact or checker reference proving authoritative
  source-of-truth discoverability.

**Risks / rollback**
- Risk: Story 1 can look complete while still leaving reviewers without a stable
  path to the real contract.
- Rollback/mitigation: require both passing checks and one stable emitted
  discoverability surface before closing Story 1.

## Exit Criteria

Story 1 is complete only when all of the following are true:

- one stable documentation inventory defines the authoritative Phase 2 document
  classes and their roles,
- every mandatory Phase 2 contract topic has an explicit authoritative
  reference path,
- one stable entry point exists for citing and navigating the Phase 2 contract,
- missing or conflicting authority mappings fail Story 1 completeness checks,
- one stable Story 1 artifact or rerunnable checker captures the discoverability
  layer in structured form,
- and supported-entry wording, support-surface boundaries, evidence-bar rules,
  future-work separation, and terminology closure remain clearly assigned to
  Stories 2 to 6.

## Implementation Notes

- Prefer a thin reference map over a new narrative document. The authoritative
  contract already lives in the Phase 2 planning, ADR, checklist, mini-spec, and
  evidence layers.
- Keep Story 1 aligned with the publication-facing documents by giving them a
  citable entry path rather than a second claim-bearing summary.
- Reuse existing Task 6 evidence surfaces where workflow-backed claims need a
  machine-readable anchor; Story 1 should index them, not replace them.
- Treat broken discoverability as a real phase-risk issue. If reviewers cannot
  tell where a claim lives, the documentation contract is not actually closed.
