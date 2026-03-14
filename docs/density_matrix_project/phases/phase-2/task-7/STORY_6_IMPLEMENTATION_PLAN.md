# Story 6 Implementation Plan

## Story Being Implemented

Story 6: Terminology, Support Labels, And Cross-References Stay Consistent
Across The Phase 2 Bundle

This is a Layer 4 engineering plan for implementing the sixth behavioral slice
from `TASK_7_STORIES.md`.

## Scope

This story assembles Task 7 outputs from Stories 1 to 5 into one coherent
documentation-consistency surface for Phase 2:

- the source-of-truth map, supported-entry guide, support-surface reference,
  evidence-bar gate, and future-work boundary are packaged under one consistent
  terminology and cross-reference layer,
- stable term definitions, support labels, and canonical references are
  preserved across planning, ADR, mini-spec, task, roadmap, and publication-
  facing docs,
- bundle completeness and semantic integrity are validated explicitly rather
  than inferred from file presence alone,
- and the resulting Task 7 bundle is reviewable and citable without hidden
  contract drift.

Out of scope for this story:

- changing the underlying contract decisions already frozen by Stories 1 to 5,
- widening the Phase 2 scope or revising roadmap order,
- rewriting the validation or workflow artifacts themselves,
- and writing final manuscript prose beyond delivering the documentation bundle
  that paper-facing surfaces can cite.

## Dependencies And Assumptions

- Stories 1 to 5 are already in place and provide stable surfaces for:
  source-of-truth mapping, supported entry and workflow wording,
  support-surface classification, evidence-bar interpretation, and future-work
  separation.
- Story 1 is expected to emit a stable authoritative contract-reference map.
- Story 1 now establishes
  `docs/density_matrix_project/phases/phase-2/PHASE_2_DOCUMENTATION_INDEX.md`
  as the stable reader-facing entry point for the full Task 7 bundle. Story 6
  should preserve and package that entry point rather than replace it.
- Story 2 is expected to emit a stable supported-entry and canonical-workflow
  summary or checker.
- Story 3 is expected to emit a stable support-surface reference bundle or
  checker.
- Story 4 is expected to emit a stable evidence-bar summary or checker.
- Story 5 is expected to emit a stable future-work boundary audit or checker.
- The Phase 2 bundle already spans multiple document classes:
  planning, ADRs, checklist, mini-specs, story docs, implementation plans,
  roadmap docs, and paper-facing summaries. Story 6 should unify these surfaces
  without replacing their roles.
- Existing publication-facing docs
  `ABSTRACT_PHASE_2.md`, `SHORT_PAPER_PHASE_2.md`, and `PAPER_PHASE_2.md`
  already need a consistent contract vocabulary. Story 6 should package and
  validate that vocabulary rather than author a second claim set.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 7 Terminology Inventory And Top-Level Consistency Manifest Schema

**Implements story**
- `Story 6: Terminology, Support Labels, And Cross-References Stay Consistent Across The Phase 2 Bundle`

**Change type**
- docs | validation automation

**Definition of done**
- Story 6 names one stable terminology inventory for the Phase 2 documentation
  bundle.
- One top-level consistency-manifest schema is frozen for Task 7.
- Bundle completeness is judged against explicit term and reference inventory
  rather than ad hoc review.

**Execution checklist**
- [ ] Define one canonical terminology inventory covering backend names, exact-
      regime language, acceptance-anchor wording, canonical workflow labels,
      support labels, reproducibility bundle wording, and future-work labels.
- [ ] Freeze one top-level consistency-manifest schema with mandatory fields for
      files, terms, canonical references, and validation status.
- [ ] Preserve support for story-level outputs from Stories 1 to 5 as first-class
      inputs into the Story 6 bundle.
- [ ] Keep optional contextual terms explicitly separate from the mandatory Task 7
      terminology inventory.

**Evidence produced**
- One stable Task 7 terminology inventory.
- One stable Task 7 top-level consistency-manifest schema.

**Risks / rollback**
- Risk: without a frozen inventory, Story 6 turns into a vague editorial pass
  that cannot be validated or reproduced.
- Rollback/mitigation: define the terminology and manifest schema explicitly and
  validate every bundle against them.

### Engineering Task 2: Unify Story 1 To Story 5 Outputs Into One Coherent Task 7 Documentation Surface

**Implements story**
- `Story 6: Terminology, Support Labels, And Cross-References Stay Consistent Across The Phase 2 Bundle`

**Change type**
- validation automation | docs

**Definition of done**
- Story 6 assembles Story 1 to Story 5 outputs into one coherent Task 7 bundle.
- Cross-story linkage is stable and traceable from the top-level manifest to
  lower-level outputs.
- Discoverability, entry semantics, support boundaries, evidence-bar rules, and
  future-work separation remain distinct while packaged together.

**Execution checklist**
- [ ] Reuse lower-level Story 1 to Story 5 output schemas and avoid incompatible
      remapping.
- [ ] Add top-level manifest entries linking each mandatory Story 1 to Story 5
      output.
- [ ] Preserve class distinctions:
      discoverability, supported-entry, support-surface, evidence-bar,
      future-work.
- [ ] Ensure path and ID references remain stable enough for developer-facing and
      paper-facing citation.

**Evidence produced**
- One coherent Task 7 Story 6 bundle referencing all mandatory story outputs.
- Stable cross-artifact linkage from top-level summary to lower-level evidence.

**Risks / rollback**
- Risk: disconnected Story 1 to Story 5 outputs make the documentation review
  brittle even if each lower-level story is individually strong.
- Rollback/mitigation: unify them through one manifest and explicit
  cross-references.

### Engineering Task 3: Preserve Stable Term Definitions, Support Labels, And Canonical References Across Mandatory Files

**Implements story**
- `Story 6: Terminology, Support Labels, And Cross-References Stay Consistent Across The Phase 2 Bundle`

**Change type**
- docs | validation automation

**Definition of done**
- Story 6 preserves stable terminology and reference semantics across the
  mandatory Phase 2 file set.
- Missing term definitions, conflicting support labels, or broken canonical
  references fail Story 6 integrity checks.
- Reviewers can trace a term or claim from one doc surface to another without
  reinterpretation.

**Execution checklist**
- [ ] Verify required terms are defined and used consistently across planning,
      ADR, mini-spec, story, roadmap, and paper-facing docs.
- [ ] Verify support labels such as required, optional, deferred, unsupported,
      non-goal, and future work are used consistently with Story 3 to Story 5
      outputs.
- [ ] Verify canonical references for backend selection, workflow anchor,
      thresholds, and roadmap boundary point to the correct source files.
- [ ] Add integrity checks that fail bundle validation on conflicting terms or
      references.

**Evidence produced**
- Stable term-definition and reference fields across the Task 7 bundle.
- Integrity-check output proving terminology and reference completeness.

**Risks / rollback**
- Risk: even small term drift can make the same Phase 2 claim mean different
  things in different documents.
- Rollback/mitigation: treat terminology and canonical references as mandatory
  bundle-integrity criteria.

### Engineering Task 4: Record File-Level Coverage And Cross-Reference Provenance For Task 7 Bundle Reproducibility

**Implements story**
- `Story 6: Terminology, Support Labels, And Cross-References Stay Consistent Across The Phase 2 Bundle`

**Change type**
- validation automation | docs

**Definition of done**
- Task 7 bundle records which mandatory files were checked, which terms or
  references they contributed, and how conflicts were resolved or flagged.
- Provenance is sufficient for rerunnable bundle review and audit.
- Coverage metadata is validated as part of bundle integrity.

**Execution checklist**
- [ ] Record the mandatory file set covered by Story 6 bundle validation.
- [ ] Record which files supply canonical wording for key terms and references.
- [ ] Record generation command, git revision or document revision context where
      practical, and any file-level coverage summaries used by the checker.
- [ ] Add provenance-field presence checks to Story 6 validation.

**Evidence produced**
- Machine-readable file-coverage and cross-reference provenance for Task 7 Story
  6 bundle.
- Reproducibility metadata aligned with documentation review needs.

**Risks / rollback**
- Risk: a bundle without coverage provenance is hard to rerun and hard to trust
  during review.
- Rollback/mitigation: enforce file coverage and provenance capture as mandatory
  Story 6 schema fields.

### Engineering Task 5: Capture Raw Reference Surfaces And Add A Minimal Task 7 Bundle Completeness Checker

**Implements story**
- `Story 6: Terminology, Support Labels, And Cross-References Stay Consistent Across The Phase 2 Bundle`

**Change type**
- validation automation

**Definition of done**
- Story 6 includes explicit references to mandatory lower-level outputs and key
  document files.
- A lightweight completeness checker verifies mandatory file presence, story
  output presence, and required metadata fields.
- Bundle validation fails clearly on missing or semantically incomplete
  documentation evidence.

**Execution checklist**
- [ ] Ensure mandatory Story 1 to Story 5 outputs are referenced and resolvable
      from the Story 6 manifest.
- [ ] Ensure the mandatory Phase 2 file set is referenced and resolvable from the
      bundle.
- [ ] Add one lightweight completeness-check routine for required files,
      required outputs, and required metadata fields.
- [ ] Keep checker scope aligned with the frozen Task 7 inventory instead of
      expanding into a general project-wide doc linter.

**Evidence produced**
- One Task 7 Story 6 completeness checker with machine-checkable output.
- Explicit missing or mismatched diagnostics for bundle failures.

**Risks / rollback**
- Risk: file-presence-only review can miss broken cross-references or semantic
  drift.
- Rollback/mitigation: validate both presence and expected field / reference
  semantics.

### Engineering Task 6: Add Focused Regression Checks For Terminology And Cross-Reference Integrity

**Implements story**
- `Story 6: Terminology, Support Labels, And Cross-References Stay Consistent Across The Phase 2 Bundle`

**Change type**
- tests

**Definition of done**
- Fast checks validate Story 6 manifest schema, required term inventory,
  canonical references, and lower-level story-output presence.
- Negative cases cover missing mandatory story output, missing file coverage,
  conflicting terminology, and broken references.
- Regression coverage remains focused and faster than full document-bundle
  reassembly.

**Execution checklist**
- [ ] Add focused Story 6 integrity tests in
      `tests/density_matrix/test_phase2_docs.py` or a tightly related successor.
- [ ] Add negative checks for missing mandatory Story 1 to Story 5 outputs.
- [ ] Add negative checks for conflicting terminology or broken canonical
      reference targets.
- [ ] Keep full manual editorial review outside the fast regression layer.

**Evidence produced**
- Focused regression coverage for Task 7 Story 6 terminology and reference
  integrity.
- Reviewable failures for bundle-integrity regressions.

**Risks / rollback**
- Risk: without focused tests, Story 6 can drift into a one-time cleanup rather
  than a stable bundle gate.
- Rollback/mitigation: add targeted checks that fail immediately on terminology
  or reference regressions.

### Engineering Task 7: Align Developer-Facing And Publication-Facing Notes With The Delivered Task 7 Bundle

**Implements story**
- `Story 6: Terminology, Support Labels, And Cross-References Stay Consistent Across The Phase 2 Bundle`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing and publication-facing notes can cite the Task 7 bundle as
  the canonical consistency surface.
- The notes make clear that Story 6 packages and validates Stories 1 to 5 rather
  than redefining them.
- The documentation stays aligned with the frozen Phase 2 claim boundary.

**Execution checklist**
- [ ] Document the Story 6 bundle and its relationship to Stories 1 to 5.
- [ ] Explain how developers and paper-facing docs should cite or reuse the
      bundle for terminology and reference consistency.
- [ ] Keep the notes explicit that Story 6 does not widen Phase 2 scope or
      soften lower-level claim boundaries.
- [ ] Ensure the notes point to the stable Task 7 bundle entry surface rather
      than to ad hoc file collections.

**Evidence produced**
- Updated developer-facing and publication-facing guidance for the Task 7 Story
  6 bundle.
- One stable place where Story 6 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 6 is poorly documented, reviewers may still rely on scattered
  files rather than on the validated bundle.
- Rollback/mitigation: tie the notes directly to the same manifest and checker
  used for Story 6 validation.

### Engineering Task 8: Run Story 6 Bundle Validation And Confirm Coherent Task 7 Documentation

**Implements story**
- `Story 6: Terminology, Support Labels, And Cross-References Stay Consistent Across The Phase 2 Bundle`

**Change type**
- tests | validation automation

**Definition of done**
- The Task 7 Story 6 bundle runs successfully end to end.
- Mandatory story outputs, canonical references, and terminology inventory all
  pass integrity checks.
- Story 6 completion is backed by stable outputs and rerunnable checks rather
  than by editorial confidence alone.

**Execution checklist**
- [ ] Run focused Story 6 regression tests for bundle integrity.
- [ ] Run the dedicated Story 6 bundle-emission or checker path.
- [ ] Verify that mandatory Story 1 to Story 5 outputs, canonical references,
      and terminology inventory are all present and consistent.
- [ ] Record stable test and artifact references for final Task 7 closure and
      paper-facing reuse.

**Evidence produced**
- Passing focused pytest coverage for Task 7 Story 6.
- A machine-readable Task 7 Story 6 bundle or rerunnable checker proving bundle
  coherence and consistency.

**Risks / rollback**
- Risk: Story 6 can look complete while still leaving subtle term or
  cross-reference conflicts unresolved.
- Rollback/mitigation: treat emitted bundle validation plus focused regression
  coverage as mandatory exit evidence.

## Exit Criteria

Story 6 is complete only when all of the following are true:

- one stable terminology inventory defines the required Task 7 consistency
  surface,
- one stable Task 7 bundle manifest references all mandatory Story 1 to Story 5
  outputs and mandatory Phase 2 file surfaces,
- key terms, support labels, and canonical references are consistent across the
  mandatory bundle files,
- missing lower-level outputs, missing file coverage, conflicting terminology,
  or broken canonical references fail Story 6 integrity checks,
- one stable Task 7 Story 6 bundle or rerunnable checker records coverage and
  provenance for the documentation-consistency surface,
- and developer-facing and publication-facing notes can cite the delivered Task
  7 bundle without reopening lower-level scope decisions.

## Implementation Notes

- Treat Story 6 as a bundle and integrity layer, not as a new source of Phase 2
  decisions. The decisions are already frozen by Stories 1 to 5 and the
  underlying Phase 2 contract docs.
- Prefer a thin manifest plus checker over a large narrative summary. Reviewers
  need consistency and traceability more than more prose.
- Reuse `tests/density_matrix/test_phase2_docs.py` as the default fast
  regression surface for lower-story and bundle-level documentation checks
  unless a later story reveals a strong need to split the suite.
- Keep the mandatory file set explicit. Story 6 should be reproducible and
  auditable even when the broader repository keeps evolving.
- If Story 6 is strong, paper-facing and developer-facing docs can point to one
  validated Task 7 surface instead of to scattered consistency claims.
