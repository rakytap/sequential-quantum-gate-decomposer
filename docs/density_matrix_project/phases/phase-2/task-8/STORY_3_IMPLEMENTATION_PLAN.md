# Story 3 Implementation Plan

## Story Being Implemented

Story 3: Major Paper Claims And Sections Are Traceable To Authoritative Phase 2
Sources

This is a Layer 4 engineering plan for implementing the third behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story turns the Task 8 traceability requirement into one explicit
claim-to-source mapping layer for Paper 1:

- major paper claims and major paper sections are mapped to authoritative Phase
  2 contract docs and evidence surfaces,
- the machine-readable Task 6 publication bundle is used as the workflow-backed
  evidence anchor rather than being replaced by narrative-only summaries,
- reviewers can move from a paper-level statement to the underlying contract or
  evidence without source-code inspection,
- and the resulting traceability layer stays narrow enough that Story 4 can own
  claim-closure semantics while Story 7 can own bundle-level terminology and
  navigation closure.

Out of scope for this story:

- redefining the Paper 1 claim package already frozen by Story 1,
- aligning surface depth and audience compression already owned by Story 2,
- deciding which evidence closes the main claim, which is owned by Story 4,
- supported-path and exact-regime scope honesty owned by Story 5,
- future-work and publication-ladder framing owned by Story 6,
- and final bundle-level terminology and navigation closure owned by Story 7.

## Dependencies And Assumptions

- Stories 1 and 2 are already in place and provide a stable claim package plus a
  stable publication-surface inventory.
- Story 1 is expected to emit
  `benchmarks/density_matrix/artifacts/phase2_task8/story1_claim_package.json`
  or an equivalent rerunnable checker. Story 3 should reuse that output for
  claim identity rather than reconstructing it.
- Story 2 is expected to emit
  `benchmarks/density_matrix/artifacts/phase2_task8/story2_publication_surface_alignment.json`
  or an equivalent rerunnable checker. Story 3 should assume publication
  surfaces already share one common claim boundary.
- The authoritative documentation entry path already exists in
  `PHASE_2_DOCUMENTATION_INDEX.md` and should be reused as the reader-facing
  starting point for traceability.
- The canonical machine-readable evidence surface already exists at
  `benchmarks/density_matrix/artifacts/phase2_task6/task6_story6_publication_bundle.json`
  and preserves traceability to Task 5 validation layers. Story 3 should build
  on that surface, not replace it.
- The relevant contract docs already exist and should serve as traceability
  targets:
  - `DETAILED_PLANNING_PHASE_2.md`,
  - `ADRs_PHASE_2.md`,
  - `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`,
  - `TASK_1_MINI_SPEC.md` through `TASK_8_MINI_SPEC.md`,
  - and the Task 6 publication bundle plus Task 7 documentation bundle.
- Story 3 should produce a claim-to-source mapping only. It should not become a
  second publication narrative or a new source of scientific decisions.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Paper 1 Claim-And-Section Traceability Inventory

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Authoritative Phase 2 Sources`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 names one stable inventory of paper claims and paper sections that
  require traceability.
- The inventory is aligned with the Story 1 claim package and the current paper
  surfaces.
- The inventory stays narrow enough for reproducible checking.

**Execution checklist**
- [ ] Freeze one canonical list of major Paper 1 claims that require
      traceability.
- [ ] Freeze one canonical list of major paper sections or section classes such
      as contribution, evidence, limitations, and future directions.
- [ ] Keep the inventory rooted in the existing Phase 2 paper surfaces rather
      than in ad hoc reviewer notes.
- [ ] Record the inventory in one stable Story 3 planning surface that later
      Task 8 stories can reference directly.

**Evidence produced**
- One stable Story 3 claim-and-section traceability inventory.
- One reviewable mapping from publication item to traceability requirement.

**Risks / rollback**
- Risk: without a frozen inventory, traceability can devolve into a best-effort
  editorial pass that misses important claims.
- Rollback/mitigation: define the traceability target set explicitly before
  building mappings.

### Engineering Task 2: Reuse The Phase 2 Documentation Index And Task 6 Publication Bundle As Primary Traceability Anchors

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Authoritative Phase 2 Sources`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 uses the existing Phase 2 documentation index as the stable entry
  point for contract traceability.
- Story 3 uses the Task 6 publication bundle as the stable entry point for
  workflow-backed evidence traceability.
- Story 3 remains a mapping layer rather than a replacement for those surfaces.

**Execution checklist**
- [ ] Reuse `PHASE_2_DOCUMENTATION_INDEX.md` as the reader-facing contract entry
      surface for Paper 1 traceability.
- [ ] Reuse `task6_story6_publication_bundle.json` as the canonical
      machine-readable workflow-evidence surface.
- [ ] Keep paper claims mapped to existing contract docs and bundle outputs
      rather than copying those sources wholesale.
- [ ] Avoid introducing Story 3-only summary surfaces that compete with the
      existing index or publication bundle.

**Evidence produced**
- One Story 3 traceability layer rooted in the Task 7 entry point and Task 6
  publication bundle.
- Reviewable linkage from paper statements to already authoritative surfaces.

**Risks / rollback**
- Risk: duplicated traceability summaries can drift from the very sources they
  are supposed to reference.
- Rollback/mitigation: treat the index and publication bundle as the authority
  surfaces and keep Story 3 thin.

### Engineering Task 3: Encode Explicit Claim-To-Source And Section-To-Source Mapping Rules

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Authoritative Phase 2 Sources`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 records one explicit mapping from each major paper claim or section to
  one primary authoritative source plus any necessary supporting sources.
- The mapping covers both contract-bearing docs and machine-readable evidence.
- Missing mapping coverage is treated as Story 3 incompleteness.

**Execution checklist**
- [ ] Map each major Paper 1 claim to one primary authoritative contract source.
- [ ] Map each evidence-heavy section to one primary machine-readable evidence
      source and any needed supporting contract docs.
- [ ] Mark whether each source is phase-level, task-level, evidence-level, or
      publication-level for reviewer interpretation.
- [ ] Add completeness rules so uncovered mandatory claims or sections fail Story
      3 review.

**Evidence produced**
- One explicit claim-to-source and section-to-source map for Paper 1.
- One completeness rule stating which publication items must always have source
  coverage.

**Risks / rollback**
- Risk: readers may be able to find the documents but still have no clear path
  from a specific paper statement to the governing source.
- Rollback/mitigation: make the mapping explicit and validate it as part of
  Story 3 completion.

### Engineering Task 4: Preserve Traceability Through The Task 6 Bundle Down To Underlying Task 5 Validation Layers

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Authoritative Phase 2 Sources`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 preserves the existing Task 6-to-Task 5 traceability path for
  workflow-backed evidence claims.
- Reviewers can tell when a Paper 1 claim is supported by top-level workflow
  evidence and when it relies on lower-level validation layers.
- The mapping does not sever bundle-level evidence from its underlying
  validation provenance.

**Execution checklist**
- [ ] Reuse the Task 6 publication bundle as the top-level workflow-evidence
      layer for Paper 1 workflow claims.
- [ ] Preserve references from the Task 6 bundle to the relevant Task 5
      validation layers instead of flattening them into prose-only claims.
- [ ] Distinguish direct contract sources from evidence-bundle sources in the
      Story 3 mapping.
- [ ] Keep claim traceability auditable from paper surface to lower-level
      validation evidence where applicable.

**Evidence produced**
- Story 3 traceability rules that preserve Task 6 to Task 5 provenance.
- Reviewable mapping showing how workflow-backed publication claims remain linked
  to lower validation layers.

**Risks / rollback**
- Risk: if Story 3 cites only top-level prose, reviewers lose visibility into
  what validation layers actually support a claim.
- Rollback/mitigation: preserve the existing layered evidence chain rather than
  collapsing it into a single summary reference.

### Engineering Task 5: Add Focused Regression Checks For Missing, Broken, Or Ambiguous Traceability

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Authoritative Phase 2 Sources`

**Change type**
- tests

**Definition of done**
- Fast checks catch missing source mappings, broken reference targets, or
  ambiguous primary-source assignments for major paper items.
- Negative cases show that Story 3 fails if major claims or sections do not have
  a reviewable source path.
- Regression coverage remains focused and publication-package oriented.

**Execution checklist**
- [ ] Add focused checks in `tests/density_matrix/test_phase2_publication_docs.py`
      or a tightly related successor for Story 3 traceability coverage.
- [ ] Add negative checks for missing mandatory claim mappings or missing section
      mappings.
- [ ] Add negative checks for broken reference targets or conflicting primary
      authorities.
- [ ] Keep full manuscript editorial review outside this fast regression layer.

**Evidence produced**
- Focused regression coverage for Story 3 traceability completeness.
- Reviewable failures for missing or broken claim-to-source mapping.

**Risks / rollback**
- Risk: traceability regressions can remain invisible because the paper text
  itself still reads smoothly.
- Rollback/mitigation: require explicit source mapping and validate it with
  targeted tests.

### Engineering Task 6: Emit One Stable Story 3 Claim-Traceability Manifest Or Checker

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Authoritative Phase 2 Sources`

**Change type**
- validation automation | docs

**Definition of done**
- Story 3 can emit one stable machine-readable traceability manifest or one
  stable rerunnable checker.
- The output records publication items, source roles, reference targets, and
  validation status.
- The output is stable enough for Story 7 and review-facing reuse.

**Execution checklist**
- [ ] Add one Story 3 command, script, or checker
      (for example under `benchmarks/density_matrix/`) for claim-traceability
      emission.
- [ ] Emit one stable artifact in a Task 8 artifact directory such as
      `benchmarks/density_matrix/artifacts/phase2_task8/story3_claim_traceability_bundle.json`.
- [ ] Record source references, generation command, and scope notes in the
      output.
- [ ] Keep the output narrow to publication traceability rather than
      evidence-closure semantics.

**Evidence produced**
- One stable Task 8 Story 3 traceability manifest or rerunnable checker.
- One reusable Story 3 output schema for later Task 8 handoffs.

**Risks / rollback**
- Risk: without a stable traceability artifact, paper review falls back to
  manual file hunting.
- Rollback/mitigation: define one thin structured Story 3 output that makes the
  source path explicit and rerunnable.

### Engineering Task 7: Document Story 3 Reviewer Navigation And Handoff To Story 4

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Authoritative Phase 2 Sources`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing and reviewer-facing notes explain what Story 3 validates, how
  to rerun it, and why it is the canonical claim-to-source mapping gate.
- The notes make clear that Story 3 proves traceability but does not decide
  claim-closure semantics.
- The documentation stays aligned with the frozen Phase 2 publication boundary.

**Execution checklist**
- [ ] Document the Story 3 manifest or checker and how it relates to the Task 7
      documentation index plus the Task 6 publication bundle.
- [ ] Make the main Story 3 rule explicit:
      every major Paper 1 claim and section needs a stable source path.
- [ ] Explain how Story 3 hands off evidence-closure interpretation to Story 4.
- [ ] Keep reviewer navigation focused on authoritative docs and evidence
      surfaces rather than on source-code exploration.

**Evidence produced**
- Updated guidance for the Task 8 Story 3 traceability gate.
- One stable location where Story 3 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 3 is poorly documented, reviewers may still experience the
  bundle as a collection of disconnected files.
- Rollback/mitigation: document Story 3 as the explicit reviewer navigation and
  source-trace gate.

### Engineering Task 8: Run Story 3 Validation And Confirm Claim-To-Source Traceability

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Authoritative Phase 2 Sources`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 3 traceability checks pass.
- The Story 3 manifest or checker runs successfully and emits stable output.
- Story 3 completion is backed by rerunnable source-trace evidence rather than
  by editorial confidence alone.

**Execution checklist**
- [ ] Run focused Story 3 regression checks for claim and section traceability.
- [ ] Run the Story 3 manifest or checker command and verify emitted output.
- [ ] Confirm that major Paper 1 claims and sections all map to authoritative
      contract docs or evidence surfaces with stable references.
- [ ] Record stable test and artifact references for Story 4 and later Task 8
      work.

**Evidence produced**
- Passing focused checks for Story 3 traceability completeness.
- One stable Story 3 output proving Paper 1 claims and sections are traceable to
  authoritative Phase 2 sources.

**Risks / rollback**
- Risk: Story 3 can appear complete while still leaving key paper statements
  effectively untraceable.
- Rollback/mitigation: require passing checks plus one stable emitted output
  before closing Story 3.

## Exit Criteria

Story 3 is complete only when all of the following are true:

- one stable traceability inventory defines which paper claims and sections must
  have source coverage,
- the Task 7 documentation index and Task 6 publication bundle are reused as the
  primary contract and workflow-evidence entry surfaces,
- every mandatory paper claim or section maps to an authoritative source path,
- missing mappings, broken references, or ambiguous primary authorities fail
  Story 3 checks,
- one stable Story 3 output or rerunnable checker captures the publication
  traceability surface,
- and evidence-closure semantics, supported-path wording, future-work framing,
  and final publication-package consistency remain clearly assigned to Stories 4
  to 7.

## Implementation Notes

- Prefer thin traceability manifests over long prose justifications. Reviewers
  need stable source paths more than additional narrative.
- Reuse the Task 7 documentation index and Task 6 publication bundle as entry
  anchors. Story 3 should connect them, not compete with them.
- Treat the Task 6 to Task 5 evidence chain as an asset. Paper 1 credibility is
  stronger when workflow claims can still be traced down to lower validation
  layers.
- Keep traceability explicit even for limitations and future-work statements.
  Reviewers often scrutinize boundary claims as closely as positive claims.
- If Story 3 is strong, later Task 8 bundle review can focus on whether claims
  are acceptable, not on whether the supporting sources can even be found.
