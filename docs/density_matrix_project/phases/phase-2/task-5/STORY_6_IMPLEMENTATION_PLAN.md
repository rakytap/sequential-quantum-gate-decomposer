# Story 6 Implementation Plan

## Story Being Implemented

Story 6: Validation Artifacts Preserve Stable Case Identity, Thresholds, And
Pass/Fail Provenance For Publication Evidence

This is a Layer 4 engineering plan for implementing the sixth behavioral slice
from `TASK_5_STORIES.md`.

## Scope

This story turns the delivered Task 5 evidence layers from Stories 1 to 5 into a
single reproducible, publication-ready validation surface:

- the local correctness gate, workflow-scale pass/fail baseline, trace-and-anchor
  package, metric-completeness gate, and interpretation guardrails are
  assembled into one coherent Task 5 bundle,
- stable case identifiers, thresholds, status fields, provenance, and raw-result
  references are preserved across artifact classes,
- bundle completeness and semantic integrity are validated explicitly rather than
  inferred from loose file presence,
- and the resulting Task 5 bundle can be cited directly by the Phase 2 abstract,
  short paper, and full paper without reinterpretation of hidden assumptions.

Out of scope for this story:

- changing the frozen Task 5 correctness, workflow, trace, metric, or
  interpretation contracts already closed by Stories 1 to 5,
- widening the mandatory simulator baseline beyond the Aer-centered package,
- adding new optimizer, workflow, or support-surface science beyond the frozen
  Phase 2 scope,
- and writing the paper text itself rather than delivering the evidence surface
  it cites.

## Dependencies And Assumptions

- Stories 1 to 5 are already in place: the local correctness gate, workflow
  matrix, trace-and-anchor package, metric-completeness gate, and interpretation
  guardrails already define the mandatory Task 5 evidence layers that Story 6
  must package.
- The current project already has two strong top-level manifest patterns:
  `build_exact_density_validation_bundle()` in
  `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` and
  `benchmarks/density_matrix/noise_support/noise_support_publication_bundle.py`.
- Those existing manifest patterns already stabilize useful top-level fields such
  as `suite_name`, `status`, `backend`, `reference_backend`, `software`,
  `provenance`, `summary`, and `artifacts`, plus per-artifact fields such as
  `artifact_id`, `artifact_class`, `mandatory`, `path`, `status`,
  `expected_statuses`, `purpose`, `generation_command`, and `summary`.
- The lower-level Task 5 story outputs are expected to preserve stable case IDs,
  thresholds, required-case accounting, and explicit status semantics; Story 6
  packages those semantics rather than redefining them.
- The frozen Task 5 publication contract is already implied by `P2-ADR-006`,
  `P2-ADR-014`, `P2-ADR-015`, `PUBLICATIONS.md`, `DETAILED_PLANNING_PHASE_2.md`,
  and the publication-facing expectations already reflected in
  `PAPER_PHASE_2.md`.
- Story 6 should assemble and validate Task 5 evidence; it should not reopen the
  support matrix, the workflow anchor, the threshold package, or the Story 5
  interpretation rules.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 5 Evidence Inventory And Top-Level Manifest

**Implements story**
- `Story 6: Validation Artifacts Preserve Stable Case Identity, Thresholds, And Pass/Fail Provenance For Publication Evidence`

**Change type**
- docs | validation automation

**Definition of done**
- The Task 5 evidence inventory explicitly names the required artifact classes:
  local correctness baseline, workflow-scale exact-regime baseline,
  trace-and-anchor package, metric-completeness layer, and interpretation
  summary or checker.
- Those inventory items map to the Task 5 story outputs directly rather than to
  older lower-level workflow or publication bundles alone.
- A stable top-level manifest format exists for the full Task 5 reproducibility
  bundle.
- Bundle completeness is judged against one explicit inventory rather than ad hoc
  file presence.

**Execution checklist**
- [ ] Define the required Task 5 artifact inventory for publication-facing
      evidence.
- [ ] Treat the Task 5 Story 1 and Story 2 bundles as first-class top-level
      inventory items, with Story 3 adding the dedicated trace-and-anchor
      artifact rather than collapsing those layers back into older lower-level
      bundles.
- [ ] Treat the Task 5 Story 4 metric-completeness bundle and the Task 5 Story 5
      interpretation bundle as first-class top-level inventory items, not merely
      transient checker outputs.
- [ ] Preserve the Story 3 bundle and its linked raw trace artifact as distinct
      pieces of evidence where that separation is needed for auditability,
      rather than flattening the trace into summary-only metadata.
- [ ] Create a stable manifest shape that records artifact identity, purpose,
      generation command, expected status, and summary semantics.
- [ ] Distinguish mandatory bundle items from optional supporting artifacts.
- [ ] Keep the manifest narrow enough for Task 5 while still audit-friendly for
      Phase 2 Paper 1 use.
- [ ] Treat the existing lower-level local, workflow, trace, metric, and
      interpretation outputs as distinct inventory items rather than
      re-deriving them from ad hoc logs.

**Evidence produced**
- A stable Task 5 evidence manifest for the full validation-baseline bundle.
- Explicit bundle-completeness criteria for publishable Task 5 evidence.

**Risks / rollback**
- Risk: without an explicit inventory, later paper assembly may rely on partial
  or inconsistent Task 5 artifacts.
- Rollback/mitigation: freeze the evidence inventory first and make bundle
  generation validate against it.

### Engineering Task 2: Unify Task 5 Story Outputs In One Coherent Validation Surface

**Implements story**
- `Story 6: Validation Artifacts Preserve Stable Case Identity, Thresholds, And Pass/Fail Provenance For Publication Evidence`

**Change type**
- code | validation automation

**Definition of done**
- Story 6 assembles the Task 5 Story 1 to Story 5 outputs into one coherent
  validation bundle.
- Cross-artifact metadata is consistent enough that a reviewer can trace a
  validation claim from the top-level manifest to the supporting structured
  outputs without schema translation.
- The bundle preserves the distinction between positive mandatory evidence,
  negative or excluded evidence, and interpretation-layer outputs while
  packaging them together.

**Execution checklist**
- [ ] Reuse the current artifact schemas from the lower-level Task 5 story
      outputs rather than inventing another incompatible layer.
- [ ] Add manifest entries or references linking the local correctness,
      workflow-scale, trace-and-anchor, metric-completeness, and interpretation
      artifacts.
- [ ] Keep positive mandatory evidence, excluded evidence, and interpretation
      outputs clearly separated while still packaging them into one auditable
      Task 5 surface.
- [ ] Ensure artifact references remain stable and path-independent enough for
      later doc and paper citations.

**Evidence produced**
- One coherent Task 5 bundle that references all mandatory evidence classes.
- Stable cross-artifact linkage between top-level summary and lower-level raw
  results.

**Risks / rollback**
- Risk: disconnected artifacts can make review and publication assembly brittle
  even when the underlying evidence exists.
- Rollback/mitigation: unify artifacts through one manifest rather than by loose
  naming conventions alone.

### Engineering Task 3: Preserve Stable Case Identity, Thresholds, And Pass/Fail Provenance Across Artifact Classes

**Implements story**
- `Story 6: Validation Artifacts Preserve Stable Case Identity, Thresholds, And Pass/Fail Provenance For Publication Evidence`

**Change type**
- code | validation automation

**Definition of done**
- Every relevant Task 5 artifact class preserves stable case identity, threshold
  values, backend identity, and pass/fail provenance.
- Reviewers can determine from the Task 5 bundle why a case or artifact passed,
  failed, or was excluded from the main claim.
- Missing or degraded provenance becomes a bundle-integrity failure rather than a
  silent omission.

**Execution checklist**
- [ ] Audit the lower-level Task 5 outputs for the minimum identity, threshold,
      and provenance fields needed by publication review.
- [ ] Reuse existing stable fields such as case IDs, support-tier or closure
      semantics, threshold metadata, status fields, and summary counters where
      they are already present.
- [ ] Fill any remaining provenance gaps in the least invasive way possible
      while preserving earlier bundle compatibility.
- [ ] Keep the required field set stable enough that Story 6 can validate it
      explicitly.
- [ ] Preserve why an artifact was excluded from the main claim when Story 5
      interpretation semantics apply.

**Evidence produced**
- Stable case-identity, threshold, and pass/fail provenance fields across Task 5
  artifact classes.
- Reviewable proof that every stored Task 5 artifact preserves its validation
  interpretation.

**Risks / rollback**
- Risk: if different Task 5 story outputs drift into incompatible field names or
  threshold semantics, later reviewers may misread or fail to compare evidence
  across artifact classes.
- Rollback/mitigation: preserve one stable Task 5 provenance vocabulary and
  extend it incrementally only when clearly justified.

### Engineering Task 4: Record Software, Command, And Environment Provenance For Task 5 Evidence

**Implements story**
- `Story 6: Validation Artifacts Preserve Stable Case Identity, Thresholds, And Pass/Fail Provenance For Publication Evidence`

**Change type**
- validation automation | docs

**Definition of done**
- The Task 5 bundle records the minimum provenance needed to rerun the validation
  evidence: software versions, commit or revision identity, commands, and
  relevant environment assumptions.
- Provenance is captured in machine-readable form, not only in prose notes.
- The bundle makes clear which environment assumptions are required for reruns.

**Execution checklist**
- [ ] Record Python, NumPy, Qiskit, and other relevant software versions or
      environment identifiers.
- [ ] Include git revision identity where practical.
- [ ] Record the commands or scripts used to generate each mandatory Task 5
      artifact.
- [ ] Keep provenance capture lightweight but sufficient for scientific audit and
      reproducibility.

**Evidence produced**
- Machine-readable software and command provenance for the Task 5 bundle.
- Rerun metadata aligned with the Phase 2 publication strategy.

**Risks / rollback**
- Risk: artifacts without software and command provenance are hard to reproduce
  or defend in review.
- Rollback/mitigation: treat provenance metadata as a mandatory bundle field, not
  an optional afterthought.

### Engineering Task 5: Capture Raw Results And Add A Minimal Task 5 Bundle Completeness Checker

**Implements story**
- `Story 6: Validation Artifacts Preserve Stable Case Identity, Thresholds, And Pass/Fail Provenance For Publication Evidence`

**Change type**
- code | validation automation

**Definition of done**
- Raw benchmark and validation outputs remain linked to the mandatory Task 5
  evidence items.
- A lightweight completeness checker can assert that the Task 5 bundle contains
  every mandatory artifact and metadata field.
- Bundle validation fails clearly when required evidence is missing, malformed,
  or semantically incomplete.

**Execution checklist**
- [ ] Ensure raw outputs for mandatory Task 5 artifacts are stored alongside their
      higher-level summaries or manifest entries.
- [ ] Add a small bundle-validation routine that checks required files and
      required metadata fields for each Task 5 artifact class.
- [ ] Keep completeness checks aligned with the frozen Task 5 evidence package
      rather than expanding into later-phase expectations.
- [ ] Make completeness-check output human-readable and machine-checkable.
- [ ] Validate that the top-level bundle preserves both the lower-level closure
      semantics and the Story 5 interpretation semantics without contradiction.

**Evidence produced**
- Raw result files referenced by the Task 5 manifest.
- A minimal completeness checker for the Task 5 publication-ready bundle.

**Risks / rollback**
- Risk: summary-only artifacts can obscure the underlying Task 5 evidence and
  make later audit impossible.
- Rollback/mitigation: keep raw outputs and summary outputs linked through the
  same validated manifest.

### Engineering Task 6: Add Focused Regression Checks For Task 5 Bundle Integrity

**Implements story**
- `Story 6: Validation Artifacts Preserve Stable Case Identity, Thresholds, And Pass/Fail Provenance For Publication Evidence`

**Change type**
- tests | validation automation

**Definition of done**
- Focused automated checks verify the Task 5 manifest, required metadata fields,
  and bundle completeness logic.
- Integrity checks catch missing Task 5 bundle components before paper-facing
  notes rely on them.
- The regression surface stays light enough to run regularly while still
  protecting the Task 5 evidence contract.

**Execution checklist**
- [ ] Add focused tests for the Task 5 manifest structure and completeness
      checker.
- [ ] Add focused checks for required artifact references and stable identity /
      threshold / provenance fields.
- [ ] Keep full bundle generation in dedicated validation commands rather than
      the default fast unit-test path.
- [ ] Ensure the tests fail specifically on Task 5 bundle-integrity problems
      rather than generic parsing errors.

**Evidence produced**
- Focused regression coverage for Task 5 bundle integrity.
- Reviewable failures when mandatory Task 5 evidence fields or files are missing.

**Risks / rollback**
- Risk: without integrity checks, the Task 5 bundle can drift silently and break
  later publication assembly.
- Rollback/mitigation: keep a small test surface that locks the Task 5 evidence
  contract down.

### Engineering Task 7: Align Developer-Facing And Publication-Facing Notes With The Delivered Task 5 Bundle

**Implements story**
- `Story 6: Validation Artifacts Preserve Stable Case Identity, Thresholds, And Pass/Fail Provenance For Publication Evidence`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing and publication-facing notes explain what the Task 5 bundle
  contains, how to rerun it, and how the delivered artifact classes should be
  interpreted.
- The notes align with the actual delivered Task 5 bundle instead of describing
  a broader or partially hypothetical evidence surface.
- The documentation makes it easy to cite the Task 5 bundle directly from Phase
  2 paper artifacts.

**Execution checklist**
- [ ] Update the most relevant Task 5-facing notes or bundle metadata guidance.
- [ ] Explain how the local correctness gate, workflow baseline, trace-and-anchor
      package, metric-completeness layer, and interpretation guardrails fit
      together.
- [ ] Keep optional or excluded evidence visibly distinct in the notes.
- [ ] Tie rerun instructions directly to the same scripts, checkers, and manifest
      entries used by Story 6 bundle generation.
- [ ] Keep wording aligned with `PUBLICATIONS.md`, the frozen Phase 2 acceptance
      rules, and the current Task 5 interpretation semantics.

**Evidence produced**
- Updated Task 5 notes aligned with the delivered bundle.
- One stable location where reviewers can verify Task 5 bundle composition and
  rerun instructions.

**Risks / rollback**
- Risk: if docs lag the actual Task 5 artifact semantics, reviewers may still
  misread the validation claim surface.
- Rollback/mitigation: tie the notes directly to the same manifest and
  machine-readable fields used in bundle validation.

### Engineering Task 8: Run Story 6 Bundle Validation And Confirm Task 5 Evidence Is Publication-Ready

**Implements story**
- `Story 6: Validation Artifacts Preserve Stable Case Identity, Thresholds, And Pass/Fail Provenance For Publication Evidence`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 6 regression checks pass.
- The top-level Task 5 bundle, manifest, and completeness checker run
  successfully.
- Story 6 completion is backed by reviewable artifact references rather than by
  code changes alone.

**Execution checklist**
- [ ] Run the focused Story 6 regression checks.
- [ ] Run the Task 5 bundle-generation command that emits the top-level manifest
      and linked artifacts.
- [ ] Verify that every mandatory Task 5 artifact is present, correctly typed,
      and semantically complete.
- [ ] Record stable test and artifact references for later Phase 2 paper and doc
      citations.

**Evidence produced**
- Passing focused Story 6 regression coverage.
- A stable top-level Task 5 bundle reference proving the validation-baseline
  evidence surface is publication-ready.

**Risks / rollback**
- Risk: Story 6 can appear complete while still lacking auditable proof that the
  full Task 5 evidence package is complete and reproducible.
- Rollback/mitigation: treat manifest validation, raw output references, and
  stable bundle artifacts as part of the exit gate, not optional follow-up.

## Exit Criteria

Story 6 is complete only when all of the following are true:

- one top-level Task 5 manifest references the local correctness, workflow,
  trace-and-anchor, metric-completeness, and interpretation artifact classes,
- every mandatory Task 5 artifact preserves stable case identity, thresholds,
  status semantics, and pass/fail provenance,
- software, command, and environment provenance are recorded in machine-readable
  form,
- raw results remain linked to the summary-level Task 5 artifacts,
- a minimal completeness checker can prove the Task 5 bundle is intact,
- and the resulting bundle can be cited directly for Phase 2 publication
  evidence without reinterpretation of hidden assumptions.

## Implementation Notes

- `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` already contains
  the project’s strongest initial publication-bundle pattern through
  `build_exact_density_validation_bundle()`. Story 6 should adapt that shape for Task 5 rather
  than invent another manifest style.
- `benchmarks/density_matrix/noise_support/noise_support_publication_bundle.py` already shows
  how to validate a richer top-level manifest with required summary fields and
  linked raw artifacts. Story 6 should reuse that validation style where
  practical.
- Stories 1 to 5 already establish the required Task 5 evidence layers. Story 6
  should package them, not redefine them.
- Task 5 Story 4 and Story 5 now produce phase-level checker bundles in their
  own right. Story 6 should preserve those layers explicitly so reviewers can
  distinguish evidence generation from evidence interpretation.
- Task 5 Story 3 now preserves both a phase-level trace-and-anchor bundle and a
  linked raw canonical trace artifact. Story 6 should keep that two-level shape
  when it materially improves auditability rather than collapsing everything
  into summary-only fields.
- The final Task 5 bundle should preserve the lower-level stable field names
  wherever practical so reviewers can trace a claim from the top-level manifest
  down to the supporting raw artifact without translation.
- Story 6 is the Task 5 publication-facing packaging layer, not a new benchmark
  surface and not a replacement for the earlier story-level validation gates.
