# Story 6 Implementation Plan

## Story Being Implemented

Story 6: Noise Artifacts Preserve Model Identity, Placement, And Classification
For Reproducible Publication Evidence

This is a Layer 4 engineering plan for implementing the sixth behavioral slice
from `TASK_4_STORIES.md`.

## Scope

This story turns the delivered Task 4 noise-surface evidence from Stories 1 to
5 into a reproducible, publication-ready artifact surface:

- required, optional, and unsupported noise evidence is assembled into one
  coherent reproducibility bundle,
- every stored case preserves noise model identity, insertion order, target
  placement, parameters or probabilities, and support-tier classification,
- raw results, summaries, commands, software metadata, and bundle completeness
  checks are preserved in one auditable package,
- and the resulting Task 4 bundle can be cited directly by the Phase 2 abstract,
  short paper, and full paper without reinterpretation of hidden assumptions.

Out of scope for this story:

- widening the required noise support surface,
- changing the support-tier rules already fixed in Stories 3 and 4,
- adding new simulator families beyond the mandatory Aer-centered package,
- and later-phase trainability, acceleration, or optimizer-science claims
  outside the Phase 2 Paper 1 evidence bundle.

## Dependencies And Assumptions

- Story 1 is already in place: required local-noise positive-path evidence exists
  through `required_local_noise_validation_validation.py`.
- Story 2 is already in place: the mandatory 1 to 3 qubit exact micro-validation
  gate and artifact bundle exist through
  `validate_squander_vs_qiskit.py` and
  `required_local_noise_micro_validation.py`.
- Story 3 is already in place: optional whole-register depolarizing evidence is
  explicitly classified through
  `optional_noise_classification_validation.py`.
- Story 4 is already in place: unsupported and deferred Task 4 noise requests
  emit structured negative evidence rather than silently degrading through
  `benchmarks/density_matrix/noise_support/unsupported_noise_validation.py` and
  `unsupported_noise_bundle.json`. The stabilized negative schema now
  includes `support_tier`, `unsupported_category`,
  `first_unsupported_condition`, `noise_boundary_class`, `failure_stage`, and
  `unsupported_status_cases`.
- Story 5 is already in place: the required local-noise workflow-scale exact
  regime bundle and bounded optimization trace exist through
  `required_local_noise_workflow_validation.py` as
  `required_local_noise_workflow_bundle.json` and
  `required_local_noise_trace_4q.json`. The current workflow bundle now
  fixes the required-baseline summary fields
  `required_cases`, `required_passed_cases`, `required_pass_rate`,
  `mandatory_baseline_completed`, `unsupported_status_cases`,
  `required_trace_case_name`, `required_trace_present`,
  `required_trace_completed`, and `required_trace_bridge_supported`.
- The frozen publication and benchmark requirements remain:
  `P2-ADR-013`, `P2-ADR-014`, `P2-ADR-015`,
  `DETAILED_PLANNING_PHASE_2.md`, `PUBLICATIONS.md`, and the publication-facing
  expectations already reflected in `PAPER_PHASE_2.md`.
- The current project already has a strong publication-bundle pattern in
  `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py`, especially
  `build_exact_density_validation_bundle()` and the `exact_density_validation_bundle.json` manifest
  shape.
- Story 6 should assemble and stabilize Task 4 evidence; it should not reopen
  the required/optional/deferred contract, the workflow anchor, or numeric
  thresholds.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 4 Evidence Inventory And Top-Level Manifest

**Implements story**
- `Story 6: Noise Artifacts Preserve Model Identity, Placement, And Classification For Reproducible Publication Evidence`

**Change type**
- docs | validation automation

**Definition of done**
- The Task 4 evidence inventory explicitly names the required artifact classes:
  required positive-path bundle, exact micro-validation bundle, optional
  classification bundle, unsupported-noise bundle, workflow-scale bundle, and
  optimization trace.
- A stable top-level manifest format exists for the full Task 4 reproducibility
  bundle.
- Bundle completeness is judged against one explicit checklist rather than ad hoc
  file presence.

**Execution checklist**
- [ ] Define the required Task 4 artifact inventory for publication-facing
      evidence.
- [ ] Create a stable manifest shape that records artifact identity, purpose,
      support tier, generation command, and status.
- [ ] Distinguish mandatory bundle items from optional supporting artifacts.
- [ ] Keep the manifest narrow enough for Task 4 while still audit-friendly for
      Paper 1 use.
- [ ] Treat `unsupported_noise_bundle.json` as the current negative bundle
      authority rather than reconstructing unsupported cases from ad hoc logs.
- [ ] Treat `required_local_noise_workflow_bundle.json` and
      `required_local_noise_trace_4q.json` as distinct mandatory Story 5
      inventory items rather than re-deriving them from the older Task 2
      workflow outputs.

**Evidence produced**
- A stable Task 4 evidence manifest for the full noise-baseline bundle.
- Explicit bundle-completeness criteria for publishable Task 4 evidence.

**Risks / rollback**
- Risk: without an explicit inventory, later paper assembly may rely on partial
  or inconsistent Task 4 artifacts.
- Rollback/mitigation: freeze the evidence inventory first and make bundle
  generation validate against it.

### Engineering Task 2: Unify Required, Optional, Unsupported, And Workflow-Scale Outputs In One Task 4 Surface

**Implements story**
- `Story 6: Noise Artifacts Preserve Model Identity, Placement, And Classification For Reproducible Publication Evidence`

**Change type**
- code | validation automation

**Definition of done**
- Story 6 assembles the required baseline, optional baseline, unsupported
  boundary, and workflow-scale results into one coherent Task 4 bundle.
- Cross-artifact metadata is consistent enough that a reviewer can trace a noise
  claim from manifest entry to raw result without schema translation.
- The bundle preserves the distinction between required, optional, and
  unsupported noise evidence while packaging them together.

**Execution checklist**
- [ ] Reuse the current Task 4 artifact schemas from Stories 1 to 5 rather than
      inventing another incompatible layer.
- [ ] Add manifest entries or references linking the required, optional,
      unsupported, and workflow-scale bundles plus the optimization trace.
- [ ] Keep required, optional, and unsupported evidence clearly separated while
      still packaging them into one auditable surface.
- [ ] Ensure artifact references remain stable and path-independent enough for
      later doc and paper citations.
- [ ] Preserve Story 4’s support-tier split between deferred and unsupported
      negative cases so the top-level manifest does not collapse them into one
      ambiguous failure bucket.
- [ ] Preserve Story 5’s required-workflow summary fields
      `required_cases`, `required_passed_cases`, `required_pass_rate`,
      `mandatory_baseline_completed`, and `unsupported_status_cases` verbatim so
      Story 6 does not reinterpret workflow sufficiency.

**Evidence produced**
- One coherent Task 4 bundle that references all mandatory positive, optional,
  and negative evidence classes.
- Stable cross-artifact linkage between summary and raw results.

**Risks / rollback**
- Risk: disconnected artifacts can make review and publication assembly brittle
  even when the underlying evidence exists.
- Rollback/mitigation: unify artifacts through one manifest rather than by loose
  naming conventions alone.

### Engineering Task 3: Preserve Noise Model Identity, Placement, And Parameter Metadata In Every Artifact Class

**Implements story**
- `Story 6: Noise Artifacts Preserve Model Identity, Placement, And Classification For Reproducible Publication Evidence`

**Change type**
- code | validation automation

**Definition of done**
- Every relevant Task 4 artifact class preserves noise model identity, insertion
  order, target placement, and parameter or probability metadata.
- Required, optional, and unsupported outputs use stable field names wherever
  practical so later review does not need schema translation.
- Missing or degraded noise metadata becomes a bundle-integrity failure rather
  than a silent omission.

**Execution checklist**
- [ ] Audit the existing Task 4 bundles for the minimum noise metadata needed by
      publication review.
- [ ] Reuse existing fields such as noise sequences, targets, fixed values,
      operation summaries, support tiers, and unsupported reasons where they are
      already present.
- [ ] Fill any remaining metadata gaps in the least invasive way possible while
      preserving earlier bundle compatibility.
- [ ] Keep the required field set stable enough that Story 6 can validate it.
- [ ] Keep Story 4 negative metadata fields
      `requested_noise_channel`, `unsupported_category`,
      `first_unsupported_condition`, `noise_boundary_class`, and
      `failure_stage` available without schema translation.
- [ ] Keep Story 5 trace metadata fields `support_tier`, `case_purpose`,
      `counts_toward_mandatory_baseline`, and `required_validation_trace` visible in
      the top-level bundle without repackaging them under different names.

**Evidence produced**
- Stable noise metadata fields across Task 4 artifact classes.
- Reviewable proof that every stored Task 4 case preserves its noise semantics.

**Risks / rollback**
- Risk: if different Task 4 stories drift into incompatible field names, later
  reviewers may misread or fail to compare evidence across artifact classes.
- Rollback/mitigation: preserve one stable Task 4 noise metadata vocabulary and
  extend it incrementally only when clearly justified.

### Engineering Task 4: Record Software, Command, And Environment Provenance For Task 4 Evidence

**Implements story**
- `Story 6: Noise Artifacts Preserve Model Identity, Placement, And Classification For Reproducible Publication Evidence`

**Change type**
- validation automation | docs

**Definition of done**
- The Task 4 bundle records the minimum provenance needed to rerun the noise
  evidence: software versions, commit or revision identity, commands, and
  relevant environment assumptions.
- Provenance is captured in machine-readable form, not only in prose notes.
- The bundle makes clear which environment assumptions are required for reruns.

**Execution checklist**
- [ ] Record Python, NumPy, Qiskit, and other relevant software versions or
      environment identifiers.
- [ ] Include git revision identity where practical.
- [ ] Record the commands or scripts used to generate each mandatory Task 4
      artifact.
- [ ] Keep provenance capture lightweight but sufficient for scientific audit and
      reproducibility.

**Evidence produced**
- Machine-readable software and command provenance for the Task 4 bundle.
- Rerun metadata aligned with the Phase 2 publication strategy.

**Risks / rollback**
- Risk: artifacts without software and command provenance are hard to reproduce
  or defend in review.
- Rollback/mitigation: treat provenance metadata as a mandatory bundle field, not
  an optional afterthought.

### Engineering Task 5: Capture Raw Results And Add A Minimal Task 4 Bundle Completeness Checker

**Implements story**
- `Story 6: Noise Artifacts Preserve Model Identity, Placement, And Classification For Reproducible Publication Evidence`

**Change type**
- code | validation automation

**Definition of done**
- Raw benchmark outputs are retained for the mandatory Task 4 evidence items.
- A lightweight completeness checker can assert that the Task 4 bundle contains
  every mandatory artifact and metadata field.
- Bundle validation fails clearly when required evidence is missing, malformed,
  or underclassified.

**Execution checklist**
- [ ] Ensure raw outputs for mandatory Task 4 traces, workflow cases, optional
      bundles, and unsupported bundles are stored alongside their higher-level
      summaries or manifest entries.
- [ ] Add a small bundle-validation routine that checks required files and
      required metadata fields for each artifact class.
- [ ] Keep completeness checks aligned with the frozen Task 4 evidence package
      rather than expanding into later-phase expectations.
- [ ] Make completeness-check output human-readable and machine-checkable.
- [ ] Validate the Story 4 negative bundle against both support-tier counts and
      `unsupported_status_cases` so deferred and unsupported noise families stay
      distinguishable in the final package.
- [ ] Validate the Story 5 workflow bundle against its required-workflow summary
      fields and confirm the separate Story 5 trace file matches
      `required_trace_case_name`.

**Evidence produced**
- Raw result files referenced by the Task 4 manifest.
- A minimal completeness checker for the Task 4 publication-ready bundle.

**Risks / rollback**
- Risk: summary-only artifacts can obscure the underlying Task 4 evidence and
  make later audit impossible.
- Rollback/mitigation: keep raw outputs and summary outputs linked through the
  same validated manifest.

### Engineering Task 6: Add Focused Regression Checks For Task 4 Bundle Integrity

**Implements story**
- `Story 6: Noise Artifacts Preserve Model Identity, Placement, And Classification For Reproducible Publication Evidence`

**Change type**
- tests | validation automation

**Definition of done**
- Focused automated checks verify the Task 4 manifest, required metadata fields,
  and bundle completeness logic.
- Integrity checks catch missing Task 4 bundle components before paper-facing
  notes rely on them.
- The regression surface stays light enough to run regularly while still
  protecting the Task 4 evidence contract.

**Execution checklist**
- [ ] Add focused tests for the Task 4 manifest structure and completeness
      checker.
- [ ] Add focused checks for required artifact references and support-tier /
      provenance / noise-metadata fields.
- [ ] Keep full bundle generation in dedicated validation commands rather than
      the default fast unit-test path.
- [ ] Ensure the tests fail specifically on Task 4 bundle-integrity problems
      rather than generic parsing errors.

**Evidence produced**
- Focused regression coverage for Task 4 bundle integrity.
- Reviewable failures when mandatory Task 4 evidence fields or files are missing.

**Risks / rollback**
- Risk: without integrity checks, the Task 4 bundle can drift silently and break
  later publication assembly.
- Rollback/mitigation: keep a small test surface that locks the Task 4 evidence
  contract down.

### Engineering Task 7: Align Developer-Facing And Publication-Facing Notes With The Delivered Task 4 Bundle

**Implements story**
- `Story 6: Noise Artifacts Preserve Model Identity, Placement, And Classification For Reproducible Publication Evidence`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing and publication-facing notes explain what the Task 4 bundle
  contains, how to rerun it, and how required, optional, and unsupported noise
  evidence should be interpreted.
- The notes align with the actual delivered Task 4 bundle instead of describing a
  broader or partially hypothetical evidence surface.
- The documentation makes it easy to cite the Task 4 noise bundle directly from
  Phase 2 paper artifacts.

**Execution checklist**
- [ ] Update the most relevant Task 4-facing notes or bundle metadata guidance.
- [ ] Explain how the required baseline, optional baseline, unsupported noise
      boundary, and workflow-scale evidence fit together.
- [ ] Keep whole-register optional baselines and unsupported deferred classes
      visibly distinct in the notes.
- [ ] Tie rerun instructions directly to the same scripts and manifest entries
      used by Story 6 bundle generation.
- [ ] Cite `benchmarks/density_matrix/noise_support/unsupported_noise_validation.py` and its emitted
      `unsupported_noise_bundle.json` artifact explicitly in the final
      rerun guidance.
- [ ] Cite `required_local_noise_workflow_validation.py` and the
      emitted `required_local_noise_workflow_bundle.json` /
      `required_local_noise_trace_4q.json` pair explicitly in the final
      rerun guidance.

**Evidence produced**
- Updated Task 4 notes aligned with the delivered bundle.
- One stable location where reviewers can verify Task 4 bundle composition and
  rerun instructions.

**Risks / rollback**
- Risk: if docs lag the actual Task 4 artifact semantics, reviewers may still
  misread optional or unsupported evidence.
- Rollback/mitigation: tie the notes directly to the same manifest and
  machine-readable fields used in bundle validation.

### Engineering Task 8: Run Story 6 Bundle Validation And Confirm Task 4 Evidence Is Publication-Ready

**Implements story**
- `Story 6: Noise Artifacts Preserve Model Identity, Placement, And Classification For Reproducible Publication Evidence`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 6 regression checks pass.
- The top-level Task 4 bundle, manifest, and completeness checker run
  successfully.
- Story 6 completion is backed by reviewable artifact references rather than by
  code changes alone.

**Execution checklist**
- [ ] Run the focused Story 6 regression checks.
- [ ] Run the Task 4 bundle-generation command that emits the top-level manifest
      and linked artifacts.
- [ ] Verify that every mandatory Task 4 artifact is present, correctly typed,
      and semantically complete.
- [ ] Record stable test and artifact references for later Phase 2 paper and doc
      citations.

**Evidence produced**
- Passing focused Story 6 regression coverage.
- A stable top-level Task 4 bundle reference proving the noise-baseline evidence
  surface is publication-ready.

**Risks / rollback**
- Risk: Story 6 can appear complete while still lacking auditable proof that the
  full Task 4 noise evidence package is complete and reproducible.
- Rollback/mitigation: treat manifest validation, raw output references, and
  stable bundle artifacts as part of the exit gate, not optional follow-up.

## Exit Criteria

Story 6 is complete only when all of the following are true:

- one top-level Task 4 manifest references the required, optional, unsupported,
  workflow-scale, and trace artifact classes,
- every mandatory Task 4 artifact preserves noise model identity, placement,
  parameters or probabilities, and support-tier classification,
- software, command, and environment provenance are recorded in machine-readable
  form,
- raw results remain linked to the summary-level Task 4 artifacts,
- a minimal completeness checker can prove the Task 4 bundle is intact,
- and the resulting bundle can be cited directly for Phase 2 publication
  evidence without reinterpretation of hidden assumptions.

## Implementation Notes

- `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` already contains the project’s strongest
  publication-bundle pattern through `build_exact_density_validation_bundle()`. Story 6 should
  adapt that shape for Task 4 rather than invent another manifest style.
- Stories 1 to 5 already established the required positive-path, exact
  micro-validation, optional classification, unsupported boundary, and
  workflow-scale sufficiency evidence layers. Story 6 should package them, not
  redefine them.
- The Task 4 support-tier vocabulary already exists and should remain the common
  language across the top-level bundle. Story 4 now fixes the negative schema
  around `support_tier`, `unsupported_category`,
  `first_unsupported_condition`, `noise_boundary_class`, `failure_stage`, and
  `unsupported_status_cases`; Story 6 should consume those fields directly.
- Story 5 now fixes the workflow artifact names
  `required_local_noise_workflow_bundle.json` and
  `required_local_noise_trace_4q.json` plus the required-workflow summary
  fields `required_cases`, `required_passed_cases`, `required_pass_rate`,
  `mandatory_baseline_completed`, `unsupported_status_cases`,
  `required_trace_case_name`, `required_trace_present`,
  `required_trace_completed`, and `required_trace_bridge_supported`. Story 6
  should package those fields directly rather than recomputing them from the raw
  workflow cases.
- The publication-facing docs already expect a stable top-level bundle and clear
  unsupported-case evidence. Story 6 should make those claims directly
  re-runnable from the delivered artifact surface.
