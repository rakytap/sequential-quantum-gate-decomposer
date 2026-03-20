# Story 6 Implementation Plan

## Story Being Implemented

Story 6: One Manifest-Driven Review Package Keeps Positive Evidence, Negative
Boundary Evidence, And Diagnosis Visible Together

This is a Layer 4 engineering plan for implementing the sixth behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story turns Task 8 into one explicit manifest-driven reviewer package for
Paper 2:

- positive supported evidence, negative boundary evidence, and diagnosis-
  grounded limitation reporting are packaged together on one stable reviewer-
  entry surface rather than scattered across paper prose and story-local
  artifacts,
- the package makes it possible for later Task 8 and publication consumers to
  navigate the current Phase 3 evidence surface directly,
- the manifest stays machine-reviewable and points to emitted Task 6 and Task 7
  bundles plus the lower Task 8 story outputs,
- and Story 6 closes the contract for "how reviewers enter the Paper 2 evidence
  package" without taking ownership of future-work framing or final package-
  consistency guardrails.

Out of scope for this story:

- freezing the claim package, which is owned by Story 1,
- keeping publication surfaces aligned, which is owned by Story 2,
- claim-to-source traceability owned by Story 3,
- evidence-closure interpretation owned by Story 4,
- supported-path and benchmark-honesty wording owned by Story 5,
- future-work and publication-ladder positioning owned by Story 7,
- and terminology, reviewer-entry stability, and summary-consistency guardrails
  across the whole package owned by Story 8.

## Dependencies And Assumptions

- Stories 1 through 5 are already expected to freeze the claim package, align
  surfaces, define traceability, define evidence closure, and freeze honest
  supported-path wording. Story 6 should package those surfaces rather than
  redefine them.
- The current emitted Phase 3 bundle surfaces already exist under:
  - `benchmarks/density_matrix/artifacts/phase3_task6/`,
  - and `benchmarks/density_matrix/artifacts/phase3_task7/`.
- Story 6 should treat those emitted bundles as first-class inputs, especially:
  - Task 6 correctness-package, unsupported-boundary, and summary-consistency,
  - Task 7 benchmark-package, diagnosis, sensitivity-matrix, and
    summary-consistency.
- Story 6 should also expect lower Task 8 story outputs from:
  - Story 1 claim package,
  - Story 2 surface alignment,
  - Story 3 claim traceability,
  - Story 4 evidence closure,
  - and Story 5 supported-path scope.
- The natural implementation home for Task 8 top-level reviewer packaging is the
  same `benchmarks/density_matrix/publication_evidence/` package, with
  `publication_manifest_validation.py` as the Story 6 validation surface and
  emitted artifacts rooted in `benchmarks/density_matrix/artifacts/phase3_task8/`.
- Story 6 should produce one manifest-driven review package and completeness
  checker only. It should not become a second evidence source or a replacement
  for the lower-story outputs it packages.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 8 Top-Level Publication-Manifest Rule And Schema

**Implements story**
- `Story 6: One Manifest-Driven Review Package Keeps Positive Evidence, Negative Boundary Evidence, And Diagnosis Visible Together`

**Change type**
- docs | validation automation

**Definition of done**
- Story 6 defines one stable top-level publication-manifest rule for Paper 2.
- One manifest schema is frozen for reviewer-entry packaging.
- Mandatory inputs, references, and reviewer-navigation fields are explicit.

**Execution checklist**
- [ ] Freeze one top-level Story 6 manifest rule for packaging positive evidence,
      negative boundary evidence, and diagnosis together.
- [ ] Freeze one manifest schema with mandatory fields for lower-story outputs,
      Task 6 and Task 7 bundle references, reviewer-entry fields, and validation
      status.
- [ ] Distinguish mandatory manifest inputs from optional contextual references.
- [ ] Keep package-consistency and future-work framing outside the Story 6 schema
      bar.

**Evidence produced**
- One stable Story 6 publication-manifest rule.
- One stable Story 6 top-level manifest schema.

**Risks / rollback**
- Risk: without an explicit manifest rule, reviewer entry remains scattered even
  if lower-story surfaces are individually strong.
- Rollback/mitigation: freeze one top-level manifest before broader packaging.

### Engineering Task 2: Reuse Story 1 Through Story 5 Outputs And Emitted Task 6 / Task 7 Bundles As Direct Inputs

**Implements story**
- `Story 6: One Manifest-Driven Review Package Keeps Positive Evidence, Negative Boundary Evidence, And Diagnosis Visible Together`

**Change type**
- docs | code

**Definition of done**
- Story 6 packages the lower Task 8 story outputs and emitted Phase 3 bundles
  without renaming their core identities.
- Positive evidence, negative boundary evidence, and diagnosis reporting remain
  distinct while being grouped together.
- Story 6 avoids creating a disconnected publication-only package vocabulary.

**Execution checklist**
- [ ] Reuse the Story 1 through Story 5 output identities directly where fields
      overlap.
- [ ] Reuse the current Task 6 and Task 7 emitted bundle references directly
      where they match Story 6 needs.
- [ ] Keep positive supported evidence, boundary evidence, and diagnosis entries
      distinct in the top-level manifest.
- [ ] Document where Story 6 intentionally adds top-level packaging fields beyond
      the lower-story outputs.

**Evidence produced**
- One reviewable mapping from lower-story outputs and emitted Task 6 / Task 7
  bundles to the Story 6 manifest.
- One explicit boundary between reused lower-level fields and Story 6-specific
  manifest fields.

**Risks / rollback**
- Risk: Story 6 may appear coherent while still forcing later consumers to
  translate between incompatible package vocabularies.
- Rollback/mitigation: anchor the manifest directly on the existing lower-level
  and emitted surfaces.

### Engineering Task 3: Build The Story 6 Manifest Builder And Completeness Checker

**Implements story**
- `Story 6: One Manifest-Driven Review Package Keeps Positive Evidence, Negative Boundary Evidence, And Diagnosis Visible Together`

**Change type**
- code | validation automation

**Definition of done**
- Story 6 has one reusable manifest builder and completeness checker.
- The builder records lower-story outputs, emitted Task 6 / Task 7 bundle
  references, reviewer-entry fields, and validation status through one stable
  schema.
- The checker stays focused on package completeness and visibility, not on final
  package-consistency semantics.

**Execution checklist**
- [ ] Add a Story 6 checker under
      `benchmarks/density_matrix/publication_evidence/`, with
      `publication_manifest_validation.py` as the primary validation surface.
- [ ] Define one stable Story 6 manifest-record schema.
- [ ] Record explicit references to lower Task 8 story outputs and emitted Task 6
      / Task 7 bundle entry points.
- [ ] Keep future-work framing and package-wide summary-consistency outside the
      Story 6 checker.

**Evidence produced**
- One reusable Story 6 manifest builder and completeness checker.
- One stable Story 6 manifest schema for later Task 8 reuse.

**Risks / rollback**
- Risk: Story 6 can turn into prose-only reviewer guidance if it lacks one
  machine-reviewable manifest surface.
- Rollback/mitigation: define one explicit manifest and checker rather than a
  narrative package summary.

### Engineering Task 4: Preserve Stable Reviewer Entry Fields And Explicit Bundle Visibility

**Implements story**
- `Story 6: One Manifest-Driven Review Package Keeps Positive Evidence, Negative Boundary Evidence, And Diagnosis Visible Together`

**Change type**
- docs | code

**Definition of done**
- Story 6 exposes one stable reviewer-entry path into the Paper 2 package.
- The manifest makes positive evidence, explicit boundary evidence, and the
  diagnosis path all visible together.
- Reviewer entry does not depend on source-code inspection or manual directory
  exploration.

**Execution checklist**
- [ ] Add reviewer-entry fields that point to the main Paper 2 package, the
      lower-story outputs, and the current emitted Task 6 / Task 7 bundles.
- [ ] Add explicit manifest fields for positive supported evidence, negative
      boundary evidence, and diagnosis-grounded limitation evidence.
- [ ] Ensure the manifest can distinguish these evidence classes without
      collapsing them into one optimistic summary.
- [ ] Treat missing reviewer-entry or missing evidence-class visibility as a
      Story 6 failure.

**Evidence produced**
- One stable Story 6 reviewer-entry surface.
- One explicit manifest-level visibility rule for positive, boundary, and
  diagnosis evidence.

**Risks / rollback**
- Risk: later consumers may only see the positive side of the package if Story 6
  leaves visibility rules implicit.
- Rollback/mitigation: make reviewer-entry and evidence-class visibility first-
  class manifest fields.

### Engineering Task 5: Add A Representative Story 6 Completeness Matrix Across Lower-Story And Emitted Inputs

**Implements story**
- `Story 6: One Manifest-Driven Review Package Keeps Positive Evidence, Negative Boundary Evidence, And Diagnosis Visible Together`

**Change type**
- tests | validation automation

**Definition of done**
- Story 6 covers representative lower-story outputs and emitted Task 6 / Task 7
  inputs directly.
- The matrix is broad enough to show that the top-level manifest can package the
  mandatory Paper 2 evidence surface coherently.
- The matrix remains representative and contract-driven rather than exhaustive
  over every optional file.

**Execution checklist**
- [ ] Include at least one lower-story Task 8 output from Stories 1 through 5.
- [ ] Include at least one Task 6 positive-evidence or boundary-evidence bundle.
- [ ] Include at least one Task 7 benchmark or diagnosis bundle.
- [ ] Keep package-consistency and terminology enforcement outside the Story 6
      matrix so Story 8 can own them cleanly.

**Evidence produced**
- One representative Story 6 manifest-completeness matrix.
- One review surface for cross-input packaging coverage.

**Risks / rollback**
- Risk: Story 6 may seem complete for one subset of inputs while failing to
  package the full mandatory review surface.
- Rollback/mitigation: freeze a small but cross-input completeness matrix early.

### Engineering Task 6: Add Focused Regression Checks For Missing Inputs, Hidden Boundary Evidence, Or Broken Reviewer Entry

**Implements story**
- `Story 6: One Manifest-Driven Review Package Keeps Positive Evidence, Negative Boundary Evidence, And Diagnosis Visible Together`

**Change type**
- tests

**Definition of done**
- Fast checks catch missing lower-story outputs, missing emitted bundle
  references, hidden boundary evidence, or broken reviewer-entry fields.
- Negative cases prove Story 6 fails when the top-level package hides explicit
  limitation surfaces or cannot be navigated coherently.
- Regression coverage remains narrow and publication-package focused.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_phase3_task8.py` or a
      tightly related successor for Story 6 manifest completeness.
- [ ] Add negative checks for missing lower-story outputs.
- [ ] Add negative checks for missing or hidden Task 6 boundary evidence.
- [ ] Add negative checks for broken reviewer-entry fields or missing diagnosis
      references.

**Evidence produced**
- Focused regression coverage for Story 6 manifest completeness failures.
- Reviewable failures for missing inputs or hidden evidence classes.

**Risks / rollback**
- Risk: package incompleteness can remain hidden because the paper narrative
  still looks coherent.
- Rollback/mitigation: add targeted checks for the highest-risk packaging
  failures.

### Engineering Task 7: Emit A Stable Story 6 Publication-Manifest Bundle

**Implements story**
- `Story 6: One Manifest-Driven Review Package Keeps Positive Evidence, Negative Boundary Evidence, And Diagnosis Visible Together`

**Change type**
- validation automation | docs

**Definition of done**
- Story 6 emits one stable machine-reviewable publication-manifest bundle or one
  stable rerunnable checker output.
- The output records lower-story outputs, emitted Task 6 / Task 7 bundle
  references, reviewer-entry fields, and package completeness status through one
  stable schema.
- The output is stable enough for later Story 8 consistency checks to consume
  directly.

**Execution checklist**
- [ ] Add one stable Story 6 output location under
      `benchmarks/density_matrix/artifacts/phase3_task8/story6_publication_manifest/`.
- [ ] Emit one artifact such as `publication_manifest_bundle.json`.
- [ ] Record generation command, software metadata, and manifest completeness
      summary in the output.
- [ ] Keep the output focused on reviewer-entry packaging rather than on future-
      work framing or package-wide consistency.

**Evidence produced**
- One stable Story 6 publication-manifest bundle or rerunnable checker output.
- One reusable Story 6 output schema for later Task 8 handoffs.

**Risks / rollback**
- Risk: prose-only Story 6 closure will make later reviewers unable to tell
  whether the Paper 2 package is actually navigable and complete.
- Rollback/mitigation: emit one machine-reviewable top-level manifest directly.

### Engineering Task 8: Document The Story 6 Reviewer Package And Run The Story 6 Gate

**Implements story**
- `Story 6: One Manifest-Driven Review Package Keeps Positive Evidence, Negative Boundary Evidence, And Diagnosis Visible Together`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain what Story 6 packages, how to rerun it, and how
  it hands off to Stories 7 and 8.
- The Story 6 checker and emitted artifact run successfully.
- Story 6 completion is backed by rerunnable manifest and completeness evidence
  rather than by editorial confidence alone.

**Execution checklist**
- [ ] Document the Story 6 manifest rule and reviewer-entry surface.
- [ ] Make the Story 6 rule explicit:
      Paper 2 reviewer packaging must keep positive evidence, negative boundary
      evidence, and diagnosis visible together.
- [ ] Explain how Story 6 hands off future-work framing to Story 7 and package-
      consistency guardrails to Story 8.
- [ ] Run focused Story 6 regression checks and verify the emitted Story 6
      bundle or checker output.

**Evidence produced**
- Passing focused checks for Story 6 manifest completeness.
- One stable Story 6 output proving coherent reviewer-entry packaging.

**Risks / rollback**
- Risk: Story 6 can look complete while still leaving reviewers to reconstruct
  the package manually.
- Rollback/mitigation: require passing checks plus one stable emitted output
  before closing Story 6.

## Exit Criteria

Story 6 is complete only when all of the following are true:

- one explicit top-level publication-manifest rule defines how Paper 2 reviewer
  entry works,
- lower-story Task 8 outputs and emitted Task 6 / Task 7 bundle references are
  packaged together on one stable manifest surface,
- positive evidence, negative boundary evidence, and diagnosis-grounded
  limitation evidence remain visible together,
- missing inputs or broken reviewer-entry fields fail focused Story 6 checks,
- one stable Story 6 bundle or rerunnable checker captures the manifest-driven
  review package,
- and future-work framing plus package-level terminology, reviewer-entry
  stability, and summary consistency remain clearly assigned to Stories 7 and 8.

## Implementation Notes

- Story 6 is where the Paper 2 package becomes reviewable as one package rather
  than as a pile of nearby files.
- Keep the manifest explicit and thin. Reviewers need clear entry and evidence
  visibility more than more narrative prose.
- Make negative evidence visible by design. That visibility is part of the
  scientific honesty of the Paper 2 package.
- Story 6 should not replace lower-story outputs. It should package them cleanly
  and make them easier to audit.
