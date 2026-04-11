# Story 3 Implementation Plan

## Story Being Implemented

Story 3: Major Paper Claims And Sections Are Traceable To Phase 3 Contract Docs
And Emitted Task 6 / Task 7 Bundles

This is a Layer 4 engineering plan for implementing the third behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story turns Task 8 into one explicit traceability layer for Paper 2:

- major Paper 2 claims, sections, and limitation statements are mapped back to
  authoritative Phase 3 contract docs and emitted Task 6 / Task 7 bundle
  surfaces,
- the publication package records stable reviewer-readable paths from paper
  wording to machine-reviewable evidence rather than relying on prose alone,
- traceability stays section- and claim-oriented rather than degenerating into a
  repository-wide file index,
- and Story 3 closes the contract for "where Paper 2 claims come from" before
  later stories interpret evidence closure, package reviewer-entry, and enforce
  cross-package consistency.

Out of scope for this story:

- freezing the main claim and non-claims, which is owned by Story 1,
- keeping publication surfaces aligned at different depths, which is owned by
  Story 2,
- deciding which evidence can close the main Paper 2 claim, which is owned by
  Story 4,
- supported-path, no-fallback, bounded planner-claim, and benchmark-honesty
  wording owned by Story 5,
- manifest-driven reviewer packaging owned by Story 6,
- future-work positioning owned by Story 7,
- and package-level terminology, reviewer-entry, and summary-consistency
  guardrails owned by Story 8.

## Dependencies And Assumptions

- Stories 1 and 2 are already expected to freeze the claim package and
  publication-surface alignment. Story 3 should trace those surfaces rather than
  redefine them.
- The authoritative Phase 3 contract sources already exist:
  - `DETAILED_PLANNING_PHASE_3.md`,
  - `ADRs_PHASE_3.md`,
  - `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`,
  - and `TASK_1_MINI_SPEC.md` through `TASK_8_MINI_SPEC.md`.
- The emitted Task 6 bundle family already provides stable machine-reviewable
  evidence entry points under
  `benchmarks/density_matrix/artifacts/correctness_evidence/`, including:
  - `correctness_package/correctness_package_bundle.json`,
  - `unsupported_boundary/unsupported_boundary_bundle.json`,
  - and `summary_consistency/summary_consistency_bundle.json`.
- The emitted Task 7 bundle family already provides stable machine-reviewable
  evidence entry points under
  `benchmarks/density_matrix/artifacts/performance_evidence/`, including:
  - `benchmark_package/benchmark_package_bundle.json`,
  - `diagnosis/diagnosis_bundle.json`,
  - `sensitivity_matrix/sensitivity_matrix_bundle.json`,
  - and `summary_consistency/summary_consistency_bundle.json`.
- Story 3 should prefer claim-level and section-level traceability over raw
  file-level indexing. It is a reviewer-audit surface, not a general document
  browser.
- The natural implementation home for Task 8 traceability validation is the same
  `benchmarks/density_matrix/publication_evidence/` package, with
  `claim_traceability_validation.py` as the Story 3 validation surface and
  emitted artifacts rooted in `benchmarks/density_matrix/artifacts/publication_evidence/`.

## Engineering Tasks

### Engineering Task 1: Freeze The Mandatory Paper 2 Claim-And-Section Traceability Inventory

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Phase 3 Contract Docs And Emitted Task 6 / Task 7 Bundles`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 defines which Paper 2 claims and sections require explicit traceability
  records.
- The inventory is specific enough to catch missing evidence links early.
- The inventory remains bounded to the paper-facing claim surface rather than to
  every repository file.

**Execution checklist**
- [ ] Freeze the mandatory traceability inventory for main claim, supporting
      claims, limitation statements, current benchmark interpretation, and
      future-work boundary sections.
- [ ] Distinguish claim-level traceability from section-organization metadata.
- [ ] Keep optional contextual literature discussion outside the mandatory
      traceability inventory unless it carries a claim-boundary role.
- [ ] Treat missing traceability for a mandatory claim or section as a real
      Story 3 failure condition.

**Evidence produced**
- One stable Story 3 mandatory claim-and-section traceability inventory.
- One explicit boundary between mandatory traceability items and optional
  contextual material.

**Risks / rollback**
- Risk: Paper 2 may look polished while still leaving reviewers unable to tell
  what evidence actually supports its central claims.
- Rollback/mitigation: freeze the mandatory traceability inventory before broad
  publication packaging proceeds.

### Engineering Task 2: Build The Canonical Source Map Across Phase 3 Contract Docs And Emitted Bundles

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Phase 3 Contract Docs And Emitted Task 6 / Task 7 Bundles`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 defines one canonical source map from paper-facing claim classes to
  the authoritative Phase 3 docs and emitted bundle entry points.
- The source map keeps Phase 3 docs and machine-reviewable artifacts both
  visible.
- The source map is stable enough for later manifest packaging.

**Execution checklist**
- [ ] Map claim-boundary and scope sources to the planning, ADR, checklist, and
      task mini-spec docs.
- [ ] Map correctness and unsupported-boundary claims to the emitted Task 6
      bundles.
- [ ] Map benchmark, sensitivity, and diagnosis claims to the emitted Task 7
      bundles.
- [ ] Avoid replacing the source map with prose-only explanations or human
      memory.

**Evidence produced**
- One canonical Story 3 source map from Paper 2 claim classes to contract docs
  and emitted evidence.
- One reviewable list of mandatory bundle entry points for later Task 8 reuse.

**Risks / rollback**
- Risk: without a source map, later reviewers may need to reverse-engineer claim
  support manually from multiple directories.
- Rollback/mitigation: define the paper-facing source map explicitly and keep it
  stable.

### Engineering Task 3: Define The Story 3 Claim-Traceability Record Schema And Checker

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Phase 3 Contract Docs And Emitted Task 6 / Task 7 Bundles`

**Change type**
- code | validation automation

**Definition of done**
- Story 3 has one reusable claim-traceability checker.
- The checker records claim IDs or section IDs, source categories, canonical
  references, and emitted bundle references through one stable schema.
- The checker stays focused on traceability rather than on evidence-closure
  interpretation or prose quality.

**Execution checklist**
- [ ] Add a Story 3 checker under
      `benchmarks/density_matrix/publication_evidence/`, with
      `claim_traceability_validation.py` as the primary validation surface.
- [ ] Define one stable Story 3 traceability-record schema.
- [ ] Record direct references to planning, ADR, checklist, mini-spec, Task 6,
      and Task 7 sources.
- [ ] Keep evidence-closure verdicts and manifest-level reviewer packaging
      outside the Story 3 checker.

**Evidence produced**
- One reusable Story 3 claim-traceability checker.
- One stable Story 3 traceability schema for later Task 8 reuse.

**Risks / rollback**
- Risk: Story 3 can turn into manual citation review if it lacks one structured
  traceability surface.
- Rollback/mitigation: validate one machine-reviewable claim-traceability record
  set directly.

### Engineering Task 4: Preserve Direct References To The Current Task 6 And Task 7 Evidence Entry Points

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Phase 3 Contract Docs And Emitted Task 6 / Task 7 Bundles`

**Change type**
- docs | code

**Definition of done**
- Story 3 traces Paper 2 claims directly to the emitted Task 6 and Task 7 bundle
  entry points rather than to vague directory references alone.
- The reference surface is explicit enough for reviewers and later manifest
  consumers.
- Story 3 avoids inventing a second evidence vocabulary for the same bundles.

**Execution checklist**
- [ ] Record direct references to the current Task 6 correctness-package,
      unsupported-boundary, and summary-consistency bundles.
- [ ] Record direct references to the current Task 7 benchmark-package,
      diagnosis, sensitivity-matrix, and summary-consistency bundles.
- [ ] Keep bundle naming consistent with the emitted artifact tree.
- [ ] Treat broken or ambiguous bundle references as a Story 3 failure.

**Evidence produced**
- One reviewable list of explicit Task 6 and Task 7 bundle references used by
  Paper 2 traceability.
- One stable direct-reference surface for later Task 8 manifest packaging.

**Risks / rollback**
- Risk: directory-only traceability can remain technically true while still being
  too vague for paper review.
- Rollback/mitigation: point Story 3 to the emitted bundle entry points
  directly.

### Engineering Task 5: Add A Representative Claim-To-Source Traceability Matrix

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Phase 3 Contract Docs And Emitted Task 6 / Task 7 Bundles`

**Change type**
- tests | validation automation

**Definition of done**
- Story 3 covers representative claim classes across the full Paper 2 package.
- The matrix is broad enough to show that one traceability rule spans claim,
  correctness, benchmark, limitation, and future-work sections.
- The matrix remains representative and contract-driven rather than exhaustive
  over every sentence.

**Execution checklist**
- [ ] Include at least one claim-boundary or scope section.
- [ ] Include at least one correctness or unsupported-boundary section backed by
      Task 6 bundles.
- [ ] Include at least one benchmark or diagnosis section backed by Task 7
      bundles.
- [ ] Include at least one limitation or future-work section mapped to the
      relevant phase-contract sources.

**Evidence produced**
- One representative Story 3 claim-to-source traceability matrix.
- One review surface for cross-category traceability coverage.

**Risks / rollback**
- Risk: Story 3 may appear correct for one class of statement while drifting for
  others.
- Rollback/mitigation: freeze a small but cross-category traceability matrix
  early.

### Engineering Task 6: Add Focused Regression Checks For Broken, Missing, Or Ambiguous Traceability

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Phase 3 Contract Docs And Emitted Task 6 / Task 7 Bundles`

**Change type**
- tests

**Definition of done**
- Fast checks catch broken source references, missing claim mappings, or
  ambiguous evidence links.
- Negative cases prove Story 3 fails when important Paper 2 claims cannot be
  traced to one clear authoritative source or emitted bundle.
- Regression coverage remains narrow and publication-package focused.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_publication_evidence.py` or a
      tightly related successor for Story 3 claim traceability.
- [ ] Add negative checks for missing traceability on mandatory claims or
      sections.
- [ ] Add negative checks for broken Task 6 or Task 7 bundle references.
- [ ] Keep broader manuscript review and bibliography formatting outside the fast
      regression layer.

**Evidence produced**
- Focused regression coverage for Story 3 traceability failures.
- Reviewable failures for missing, broken, or ambiguous claim-to-source
  mappings.

**Risks / rollback**
- Risk: traceability regressions can survive manual review because the prose still
  looks persuasive.
- Rollback/mitigation: lock the highest-risk traceability surfaces down with
  targeted tests.

### Engineering Task 7: Emit A Stable Story 3 Claim-Traceability Bundle

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Phase 3 Contract Docs And Emitted Task 6 / Task 7 Bundles`

**Change type**
- validation automation | docs

**Definition of done**
- Story 3 emits one stable machine-reviewable claim-traceability bundle or one
  stable rerunnable checker output.
- The output records claim or section identifiers, source categories, and direct
  evidence references through one stable schema.
- The output is stable enough for later Task 8 manifest and consistency stories
  to consume directly.

**Execution checklist**
- [ ] Add one stable Story 3 output location under
      `benchmarks/density_matrix/artifacts/publication_evidence/claim_traceability/`.
- [ ] Emit one artifact such as `claim_traceability_bundle.json`.
- [ ] Record generation command, software metadata, and source coverage summary
      in the output.
- [ ] Keep the output focused on traceability, not on evidence-closure verdicts
      or final reviewer navigation.

**Evidence produced**
- One stable Story 3 claim-traceability bundle or rerunnable checker output.
- One reusable Story 3 output schema for later Task 8 handoffs.

**Risks / rollback**
- Risk: prose-only Story 3 closure will make later reviewers unable to tell
  whether the publication package is actually auditable.
- Rollback/mitigation: emit one machine-reviewable traceability surface
  directly.

### Engineering Task 8: Document Story 3 Traceability Rules And Run The Story 3 Gate

**Implements story**
- `Story 3: Major Paper Claims And Sections Are Traceable To Phase 3 Contract Docs And Emitted Task 6 / Task 7 Bundles`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain what Story 3 traces, how to rerun it, and how
  it hands off to later Task 8 stories.
- The Story 3 checker and emitted artifact run successfully.
- Story 3 completion is backed by rerunnable traceability evidence rather than
  by editorial confidence alone.

**Execution checklist**
- [ ] Document the Story 3 source map and claim-traceability checker.
- [ ] Make the Story 3 rule explicit:
      major Paper 2 claims must remain traceable to authoritative contract docs
      and emitted Task 6 / Task 7 bundles.
- [ ] Explain how Story 3 hands off evidence-closure interpretation to Story 4
      and manifest packaging to Story 6.
- [ ] Run focused Story 3 regression checks and verify the emitted Story 3
      bundle or checker output.

**Evidence produced**
- Passing focused checks for Story 3 traceability completeness.
- One stable Story 3 output proving that Paper 2 claims are auditable.

**Risks / rollback**
- Risk: Story 3 can look complete while still leaving one major paper section
  unsupported or ambiguously supported.
- Rollback/mitigation: require passing checks plus one stable emitted output
  before closing Story 3.

## Exit Criteria

Story 3 is complete only when all of the following are true:

- one stable mandatory traceability inventory defines which Paper 2 claims and
  sections must be auditable,
- one canonical source map links those claims and sections to authoritative
  Phase 3 docs and emitted Task 6 / Task 7 bundles,
- broken, missing, or ambiguous claim-to-source mappings fail focused Story 3
  checks,
- one stable Story 3 bundle or rerunnable checker captures the traceability
  surface,
- and evidence closure, supported-path wording, manifest packaging, future-work
  framing, and package-level consistency remain clearly assigned to Stories 4
  through 8.

## Implementation Notes

- Story 3 should make Paper 2 reviewable, not merely cite-heavy. Traceability
  should help a reviewer move from claim to evidence quickly.
- Prefer direct bundle-entry references over directory-only references wherever
  the emitted evidence surface is already stable.
- Treat limitation statements as traceable claims too. Honest negatives need
  evidence paths just as much as positives do.
- Keep the Story 3 output thin and structural. Later stories need a stable
  auditable source map, not another narrative review document.
