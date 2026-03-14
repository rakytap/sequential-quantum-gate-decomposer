# Story 7 Implementation Plan

## Story Being Implemented

Story 7: Terminology And Reviewer Navigation Stay Consistent Across The Paper
Package

This is a Layer 4 engineering plan for implementing the seventh behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story assembles Task 8 outputs from Stories 1 to 6 into one coherent
publication-package consistency surface for Paper 1:

- the Paper 1 claim package, surface alignment, traceability, evidence-closure
  rules, supported-path wording, and future-work framing are packaged under one
  consistent terminology and reviewer-navigation layer,
- stable term definitions, canonical references, and reviewer entry paths are
  preserved across abstract, short-paper, narrative, full-paper, documentation,
  and evidence-facing surfaces,
- bundle completeness and semantic integrity are validated explicitly rather
  than inferred from file presence alone,
- and the resulting Task 8 bundle is reviewable and citable without hidden
  publication-package drift.

Out of scope for this story:

- changing the underlying claim, evidence, supported-path, or roadmap decisions
  already frozen by Stories 1 to 6,
- generating new validation or workflow evidence,
- rewriting the Phase 2 contract itself,
- and writing venue-specific final manuscript polish beyond delivering a stable
  publication package that reviewers can navigate and cite.

## Dependencies And Assumptions

- Stories 1 to 6 are already in place and provide stable surfaces for:
  claim package, multi-surface alignment, claim traceability, evidence closure,
  supported-path honesty, and future-work framing.
- Story 1 is expected to emit
  `benchmarks/density_matrix/artifacts/phase2_task8/story1_claim_package.json`.
- Story 2 is expected to emit
  `benchmarks/density_matrix/artifacts/phase2_task8/story2_publication_surface_alignment.json`.
- Story 3 is expected to emit
  `benchmarks/density_matrix/artifacts/phase2_task8/story3_claim_traceability_bundle.json`.
- Story 4 is expected to emit
  `benchmarks/density_matrix/artifacts/phase2_task8/story4_evidence_closure_bundle.json`.
- Story 5 is expected to emit
  `benchmarks/density_matrix/artifacts/phase2_task8/story5_supported_path_scope_bundle.json`.
- Story 6 is expected to emit
  `benchmarks/density_matrix/artifacts/phase2_task8/story6_future_work_boundary_bundle.json`.
- The Phase 2 paper package already spans multiple document classes:
  - `ABSTRACT_PHASE_2.md`,
  - `SHORT_PAPER_PHASE_2.md`,
  - `SHORT_PAPER_NARRATIVE.md`,
  - `PAPER_PHASE_2.md`,
  - `PHASE_2_DOCUMENTATION_INDEX.md`,
  - and `task6_story6_publication_bundle.json`.
- Story 7 should unify these surfaces without replacing their roles. It is the
  Task 8 package and integrity layer, not a new source of Phase 2 decisions.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 8 Terminology Inventory And Top-Level Publication-Package Manifest Schema

**Implements story**
- `Story 7: Terminology And Reviewer Navigation Stay Consistent Across The Paper Package`

**Change type**
- docs | validation automation

**Definition of done**
- Story 7 names one stable terminology inventory for the Task 8 publication
  package.
- One top-level publication-package manifest schema is frozen for Task 8.
- Bundle completeness is judged against explicit term and reference inventory
  rather than ad hoc review.

**Execution checklist**
- [ ] Define one canonical terminology inventory covering backend names,
      canonical workflow labels, exact-regime language, acceptance-anchor
      wording, evidence-closure vocabulary, future-work labels, and
      reviewer-navigation terms.
- [ ] Freeze one top-level publication-package manifest schema with mandatory
      fields for surfaces, lower-story outputs, canonical references, reviewer
      entry points, and validation status.
- [ ] Preserve support for story-level outputs from Stories 1 to 6 as
      first-class inputs into the Story 7 bundle.
- [ ] Keep optional contextual terminology explicitly separate from the mandatory
      Task 8 inventory.

**Evidence produced**
- One stable Task 8 terminology inventory.
- One stable Task 8 top-level publication-package manifest schema.

**Risks / rollback**
- Risk: without a frozen inventory and schema, Story 7 turns into a vague
  editorial pass that cannot be validated or reproduced.
- Rollback/mitigation: define the terminology and manifest schema explicitly and
  validate every bundle against them.

### Engineering Task 2: Unify Story 1 To Story 6 Outputs Into One Coherent Task 8 Publication Surface

**Implements story**
- `Story 7: Terminology And Reviewer Navigation Stay Consistent Across The Paper Package`

**Change type**
- validation automation | docs

**Definition of done**
- Story 7 assembles Story 1 to Story 6 outputs into one coherent Task 8 bundle.
- Cross-story linkage is stable and traceable from the top-level manifest to
  lower-level outputs.
- Claim package, surface alignment, traceability, evidence closure,
  supported-path honesty, and roadmap framing remain distinct while packaged
  together.

**Execution checklist**
- [ ] Reuse lower-level Story 1 to Story 6 output schemas and avoid incompatible
      remapping.
- [ ] Add top-level manifest entries linking each mandatory Story 1 to Story 6
      output.
- [ ] Preserve class distinctions:
      claim package, surface alignment, traceability, evidence closure,
      supported path, future work.
- [ ] Ensure path and ID references remain stable enough for reviewer-facing and
      developer-facing citation.

**Evidence produced**
- One coherent Task 8 Story 7 bundle referencing all mandatory story outputs.
- Stable cross-artifact linkage from top-level publication summary to lower-level
  evidence.

**Risks / rollback**
- Risk: disconnected lower-story outputs make the paper package brittle even if
  each lower-level story is individually strong.
- Rollback/mitigation: unify them through one manifest and explicit
  cross-references.

### Engineering Task 3: Preserve Stable Reviewer Entry Paths Through The Documentation Index And Publication Bundle

**Implements story**
- `Story 7: Terminology And Reviewer Navigation Stay Consistent Across The Paper Package`

**Change type**
- docs | validation automation

**Definition of done**
- Story 7 preserves at least one stable reviewer path from paper-facing surfaces
  to the authoritative Phase 2 contract and workflow evidence.
- The documentation index and Task 6 publication bundle are packaged as explicit
  navigation anchors.
- Reviewer navigation remains stable even as paper surfaces evolve.

**Execution checklist**
- [ ] Preserve `PHASE_2_DOCUMENTATION_INDEX.md` as the stable contract entry
      point for reviewer navigation.
- [ ] Preserve `task6_story6_publication_bundle.json` as the stable
      machine-readable workflow-evidence entry point.
- [ ] Record how paper-facing surfaces should point reviewers to those anchors.
- [ ] Add integrity checks that fail if the reviewer path becomes ambiguous or
      broken.

**Evidence produced**
- Stable reviewer-entry and reviewer-navigation fields in the Task 8 bundle.
- Reviewable linkage from paper-facing surfaces to contract and evidence anchors.

**Risks / rollback**
- Risk: even with strong lower-level docs, reviewers can still feel lost if the
  package does not preserve one clear navigation path.
- Rollback/mitigation: treat reviewer entry paths as first-class bundle fields
  rather than as optional notes.

### Engineering Task 4: Record File-Level Coverage And Cross-Reference Provenance For Task 8 Bundle Reproducibility

**Implements story**
- `Story 7: Terminology And Reviewer Navigation Stay Consistent Across The Paper Package`

**Change type**
- validation automation | docs

**Definition of done**
- Task 8 bundle records which mandatory files were checked, which terms or
  references they contributed, and how lower-story outputs were incorporated.
- Provenance is sufficient for rerunnable publication-package review and audit.
- Coverage metadata is validated as part of bundle integrity.

**Execution checklist**
- [ ] Record the mandatory file set covered by Story 7 bundle validation.
- [ ] Record which files supply canonical wording for key terms, claim surfaces,
      reviewer entry points, and bundle references.
- [ ] Record generation command, git revision or document revision context where
      practical, and any file-level coverage summaries used by the checker.
- [ ] Add provenance-field presence checks to Story 7 validation.

**Evidence produced**
- Machine-readable file-coverage and cross-reference provenance for Task 8 Story
  7 bundle.
- Reproducibility metadata aligned with publication review needs.

**Risks / rollback**
- Risk: a publication bundle without provenance is hard to rerun and hard to
  trust during review.
- Rollback/mitigation: enforce file coverage and provenance capture as mandatory
  Story 7 schema fields.

### Engineering Task 5: Capture Raw Publication Surfaces And Add A Minimal Task 8 Bundle Completeness Checker

**Implements story**
- `Story 7: Terminology And Reviewer Navigation Stay Consistent Across The Paper Package`

**Change type**
- validation automation

**Definition of done**
- Story 7 includes explicit references to mandatory lower-story outputs and key
  publication-facing files.
- A lightweight completeness checker verifies mandatory file presence, story
  output presence, reviewer-entry paths, and required metadata fields.
- Bundle validation fails clearly on missing or semantically incomplete
  publication evidence.

**Execution checklist**
- [ ] Ensure mandatory Story 1 to Story 6 outputs are referenced and resolvable
      from the Story 7 manifest.
- [ ] Ensure the mandatory Task 8 paper-facing files and reviewer-entry anchors
      are referenced and resolvable from the bundle.
- [ ] Add one lightweight completeness-check routine for required files,
      required outputs, required metadata fields, and reviewer-path fields.
- [ ] Keep checker scope aligned with the frozen Task 8 inventory instead of
      expanding into a general project-wide paper linter.

**Evidence produced**
- One Task 8 Story 7 completeness checker with machine-checkable output.
- Explicit missing or mismatched diagnostics for publication-bundle failures.

**Risks / rollback**
- Risk: file-presence-only review can miss broken reviewer navigation or missing
  lower-story semantics.
- Rollback/mitigation: validate both presence and expected field/reference
  semantics.

### Engineering Task 6: Add Focused Regression Checks For Terminology And Reviewer-Navigation Integrity

**Implements story**
- `Story 7: Terminology And Reviewer Navigation Stay Consistent Across The Paper Package`

**Change type**
- tests

**Definition of done**
- Fast checks validate Story 7 manifest schema, required term inventory,
  reviewer-entry paths, and lower-story output presence.
- Negative cases cover missing mandatory story output, missing reviewer path,
  conflicting terminology, and broken reference targets.
- Regression coverage remains focused and faster than full publication-bundle
  reassembly.

**Execution checklist**
- [ ] Add focused Story 7 integrity tests in
      `tests/density_matrix/test_phase2_publication_docs.py` or a tightly
      related successor.
- [ ] Add negative checks for missing mandatory Story 1 to Story 6 outputs.
- [ ] Add negative checks for missing reviewer-entry paths, conflicting
      terminology, or broken canonical reference targets.
- [ ] Keep full manual editorial review outside the fast regression layer.

**Evidence produced**
- Focused regression coverage for Task 8 Story 7 terminology and navigation
  integrity.
- Reviewable failures for publication-package integrity regressions.

**Risks / rollback**
- Risk: without focused tests, Story 7 can drift into a one-time cleanup rather
  than a stable publication-package gate.
- Rollback/mitigation: add targeted checks that fail immediately on terminology
  or reviewer-navigation regressions.

### Engineering Task 7: Align Developer-Facing And Reviewer-Facing Notes With The Delivered Task 8 Bundle

**Implements story**
- `Story 7: Terminology And Reviewer Navigation Stay Consistent Across The Paper Package`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing and reviewer-facing notes can cite the Task 8 bundle as the
  canonical publication-package consistency surface.
- The notes make clear that Story 7 packages and validates Stories 1 to 6
  rather than redefining them.
- The documentation stays aligned with the frozen Phase 2 claim boundary.

**Execution checklist**
- [ ] Document the Story 7 bundle and its relationship to Stories 1 to 6.
- [ ] Explain how developers, reviewers, and paper-facing docs should cite or
      reuse the bundle for terminology and reviewer-navigation consistency.
- [ ] Keep the notes explicit that Story 7 does not widen Phase 2 scope or
      soften lower-level claim boundaries.
- [ ] Ensure the notes point to the stable Task 8 bundle entry surface rather
      than to ad hoc file collections.

**Evidence produced**
- Updated developer-facing and reviewer-facing guidance for the Task 8 Story 7
  bundle.
- One stable place where Story 7 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 7 is poorly documented, reviewers may still rely on scattered
  files rather than on the validated publication package.
- Rollback/mitigation: tie the notes directly to the same manifest and checker
  used for Story 7 validation.

### Engineering Task 8: Run Story 7 Bundle Validation And Confirm Coherent Task 8 Publication Package

**Implements story**
- `Story 7: Terminology And Reviewer Navigation Stay Consistent Across The Paper Package`

**Change type**
- tests | validation automation

**Definition of done**
- The Task 8 Story 7 bundle runs successfully end to end.
- Mandatory story outputs, canonical references, terminology inventory, and
  reviewer-entry paths all pass integrity checks.
- Story 7 completion is backed by stable outputs and rerunnable checks rather
  than by editorial confidence alone.

**Execution checklist**
- [ ] Run focused Story 7 regression tests for publication-package integrity.
- [ ] Run the dedicated Story 7 bundle-emission or checker path.
- [ ] Verify that mandatory Story 1 to Story 6 outputs, canonical references,
      terminology inventory, and reviewer-entry paths are all present and
      consistent.
- [ ] Record stable test and artifact references for final Task 8 closure and
      paper-facing reuse.

**Evidence produced**
- Passing focused pytest coverage for Task 8 Story 7.
- A machine-readable Task 8 Story 7 bundle or rerunnable checker proving
  publication-package coherence and reviewer-navigation integrity.

**Risks / rollback**
- Risk: Story 7 can look complete while still leaving subtle term, reference, or
  reviewer-path conflicts unresolved.
- Rollback/mitigation: treat emitted bundle validation plus focused regression
  coverage as mandatory exit evidence.

## Exit Criteria

Story 7 is complete only when all of the following are true:

- one stable terminology inventory defines the required Task 8
  publication-package consistency surface,
- one stable Task 8 bundle manifest references all mandatory Story 1 to Story 6
  outputs and mandatory publication-facing file surfaces,
- key terms, canonical references, and reviewer-entry paths are consistent
  across the mandatory bundle files,
- missing lower-level outputs, missing reviewer-entry paths, conflicting
  terminology, or broken canonical references fail Story 7 integrity checks,
- one stable Task 8 Story 7 bundle or rerunnable checker records coverage and
  provenance for the publication package,
- and developer-facing and reviewer-facing notes can cite the delivered Task 8
  bundle without reopening lower-level scope decisions.

## Implementation Notes

- Treat Story 7 as a publication-package bundle and integrity layer, not as a
  new source of Phase 2 decisions. The decisions are already frozen by Stories 1
  to 6 and the underlying Phase 2 contract docs.
- Prefer a thin manifest plus checker over a large narrative summary. Reviewers
  need consistent terms and stable entry paths more than more prose.
- Reuse `PHASE_2_DOCUMENTATION_INDEX.md` and
  `task6_story6_publication_bundle.json` as the default reviewer anchors unless
  a later story reveals a strong reason to add another entry surface.
- Keep the mandatory file set explicit. Story 7 should remain reproducible and
  auditable even when the broader repository keeps evolving.
- If Story 7 is strong, paper-facing and reviewer-facing work can point to one
  validated Task 8 surface instead of to scattered manuscript and evidence files.
