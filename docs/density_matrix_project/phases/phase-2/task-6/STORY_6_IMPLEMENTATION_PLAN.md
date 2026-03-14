# Story 6 Implementation Plan

## Story Being Implemented

Story 6: Workflow Artifacts Preserve Stable Workflow Identity And Provenance For
Reproducible Publication Evidence

This is a Layer 4 engineering plan for implementing the sixth behavioral slice
from `TASK_6_STORIES.md`.

## Scope

This story assembles Task 6 outputs from Stories 1 to 5 into one reproducible
publication-facing evidence surface:

- the canonical workflow contract, end-to-end plus trace evidence, fixed-
  parameter matrix baseline, unsupported-boundary bundle, and interpretation
  guardrail bundle are packaged under one top-level Task 6 manifest,
- stable workflow ID, stable case IDs, threshold semantics, status semantics,
  and provenance are preserved across artifact classes,
- bundle completeness and semantic integrity are validated explicitly rather than
  inferred from file presence alone,
- and the resulting Task 6 bundle is citable by Phase 2 publication outputs
  without hidden assumptions.

Out of scope for this story:

- changing closure rules or semantics already frozen by Stories 1 to 5,
- widening simulator baseline expectations beyond frozen Aer-centered decisions,
- adding new workflow science beyond frozen Phase 2 scope,
- and writing paper text itself rather than delivering the evidence package it
  cites.

## Dependencies And Assumptions

- Stories 1 to 5 are already in place and provide stable artifact surfaces for:
  canonical contract, 4/6 end-to-end plus trace, 4/6/8/10 matrix, unsupported
  boundaries, and interpretation guardrails.
- Story 1's emitted artifact
  `benchmarks/density_matrix/artifacts/phase2_task6/story1_canonical_workflow_contract.json`
  is the canonical source of Task 6 workflow identity, contract-version,
  threshold metadata, and bundle-level contract semantics. Story 6 should
  package and validate that artifact as the root identity surface for the full
  Task 6 bundle.
- Story 2's emitted artifact
  `benchmarks/density_matrix/artifacts/phase2_task6/story2_end_to_end_trace_bundle.json`
  is now the canonical source of Task 6 4q/6q end-to-end plus required-trace
  evidence. Story 6 should package and validate that artifact as a first-class
  mandatory bundle item rather than reconstructing Story 2 semantics from lower
  Task 5 artifacts.
- Story 3's emitted artifact
  `benchmarks/density_matrix/artifacts/phase2_task6/story3_matrix_baseline_bundle.json`
  is now the canonical source of Task 6 4/6/8/10 matrix evidence and explicit
  10-qubit anchor presence. Story 6 should package and validate that artifact as
  a first-class mandatory bundle item rather than reconstructing matrix
  semantics from lower Task 5 artifacts alone.
- Story 4's emitted artifact
  `benchmarks/density_matrix/artifacts/phase2_task6/story4_unsupported_workflow_bundle.json`
  is the canonical unsupported/deferred negative-evidence layer for Task 6.
  Story 6 should package and validate that artifact as a first-class mandatory
  bundle item.
- Story 5's emitted artifact
  `benchmarks/density_matrix/artifacts/phase2_task6/story5_interpretation_bundle.json`
  is the canonical interpretation-guardrail layer for Task 6 completion.
  Story 6 should package and validate that artifact as a first-class mandatory
  bundle item rather than recomputing Story 5 semantics inside the publication
  bundle.
- Task 5 Story 6 implementation in
  `benchmarks/density_matrix/task5_story6_publication_bundle.py` provides the
  canonical top-level manifest and integrity-check pattern to reuse.
- Existing manifest field patterns should be preserved where practical:
  `suite_name`, `status`, `backend`, `reference_backend`, `software`,
  `provenance`, `summary`, `artifacts`, and per-artifact fields such as
  `artifact_id`, `artifact_class`, `mandatory`, `path`, `status`,
  `expected_statuses`, `purpose`, `generation_command`, and `summary`.
- Task 6 Story 6 must preserve and expose canonical workflow identity fields from
  Story 1, including threshold and contract-semantics metadata, plus case
  identity fields from Stories 2 and 3.
- Frozen publication alignment remains unchanged:
  `P2-ADR-007`, `P2-ADR-013`, `P2-ADR-014`, `P2-ADR-015`,
  `DETAILED_PLANNING_PHASE_2.md`, and `PAPER_PHASE_2.md`.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 6 Evidence Inventory And Top-Level Manifest Schema

**Implements story**
- `Story 6: Workflow Artifacts Preserve Stable Workflow Identity And Provenance For Reproducible Publication Evidence`

**Change type**
- docs | validation automation

**Definition of done**
- Task 6 Story 6 explicitly names mandatory artifact classes from Stories 1 to
  5.
- One stable top-level manifest schema is frozen for Task 6 publication evidence.
- Bundle completeness is judged against explicit inventory and statuses.

**Execution checklist**
- [ ] Define mandatory Task 6 artifact inventory:
      Story 1 contract bundle, Story 2 end-to-end plus trace bundle, Story 3
      matrix baseline bundle, Story 4 unsupported bundle, Story 5
      interpretation bundle.
- [ ] Preserve optional support for raw trace artifact references where required
      for auditability.
- [ ] Freeze one top-level manifest schema with mandatory artifact metadata
      fields.
- [ ] Keep optional artifacts explicitly marked non-mandatory.

**Evidence produced**
- One stable Task 6 evidence inventory.
- One stable Task 6 top-level manifest schema.

**Risks / rollback**
- Risk: implicit inventory allows inconsistent publication bundles and missing
  mandatory evidence.
- Rollback/mitigation: freeze explicit inventory and validate every bundle
  against it.

### Engineering Task 2: Unify Story 1 To Story 5 Outputs Into One Coherent Task 6 Validation Surface

**Implements story**
- `Story 6: Workflow Artifacts Preserve Stable Workflow Identity And Provenance For Reproducible Publication Evidence`

**Change type**
- code | validation automation

**Definition of done**
- Story 6 assembles Story 1 to Story 5 outputs into one coherent Task 6 bundle.
- Cross-artifact linkage is stable and traceable from manifest to raw artifacts.
- Positive evidence, negative evidence, and interpretation outputs remain
  explicitly distinct while packaged together.

**Execution checklist**
- [ ] Reuse lower-level Story 1 to Story 5 artifact schemas and avoid incompatible
      remapping.
- [ ] Add manifest entries linking each mandatory Story 1 to Story 5 artifact.
- [ ] Preserve class distinctions (contract, positive execution, matrix,
      unsupported boundary, interpretation).
- [ ] Ensure path and ID references remain stable enough for docs and paper
      citations.

**Evidence produced**
- One coherent Task 6 Story 6 bundle referencing all mandatory artifact classes.
- Stable cross-artifact linkage from top-level summary to lower-level evidence.

**Risks / rollback**
- Risk: disconnected artifact surfaces make publication review brittle even if
  evidence exists.
- Rollback/mitigation: unify through one manifest and explicit cross-references.

### Engineering Task 3: Preserve Stable Workflow And Case Identity With Threshold And Pass/Fail Provenance

**Implements story**
- `Story 6: Workflow Artifacts Preserve Stable Workflow Identity And Provenance For Reproducible Publication Evidence`

**Change type**
- code | validation automation

**Definition of done**
- Story 6 preserves canonical workflow ID/version and case-level identity across
  all relevant artifact classes.
- Threshold and pass/fail provenance is explicit and reviewable.
- Missing identity/provenance fields fail Story 6 integrity checks.

**Execution checklist**
- [ ] Verify each mandatory artifact preserves required identity fields:
      workflow ID/version and case IDs where applicable.
- [ ] Verify each mandatory artifact preserves threshold/status provenance fields
      required for interpretation.
- [ ] Reuse stable field names where available and add minimal bridging fields
      only when necessary.
- [ ] Add integrity checks that fail bundle validation on identity/provenance
      gaps.

**Evidence produced**
- Stable identity/provenance fields across Task 6 artifact classes.
- Integrity-check output proving identity/provenance completeness.

**Risks / rollback**
- Risk: identity drift across artifacts makes reproducibility and citation
  ambiguous.
- Rollback/mitigation: treat identity/provenance fields as mandatory
  bundle-integrity criteria.

### Engineering Task 4: Record Software, Command, And Environment Provenance For Task 6 Bundle Reproducibility

**Implements story**
- `Story 6: Workflow Artifacts Preserve Stable Workflow Identity And Provenance For Reproducible Publication Evidence`

**Change type**
- validation automation | docs

**Definition of done**
- Task 6 bundle records software versions, generation commands, git revision, and
  working-directory context in machine-readable form.
- Provenance is sufficient for reproducible reruns and audit.
- Provenance fields are validated as part of bundle integrity.

**Execution checklist**
- [ ] Record software metadata (Python, NumPy, Qiskit, Qiskit Aer, and relevant
      project versions where available).
- [ ] Record generation commands per mandatory artifact and top-level bundle.
- [ ] Record git revision and working-directory provenance where practical.
- [ ] Add provenance-field presence checks to Story 6 validation.

**Evidence produced**
- Machine-readable software and command provenance for Task 6 Story 6 bundle.
- Reproducibility metadata aligned with publication needs.

**Risks / rollback**
- Risk: bundles without provenance are difficult to defend or rerun.
- Rollback/mitigation: enforce provenance capture as mandatory Story 6 schema.

### Engineering Task 5: Capture Raw References And Add A Minimal Task 6 Bundle Completeness Checker

**Implements story**
- `Story 6: Workflow Artifacts Preserve Stable Workflow Identity And Provenance For Reproducible Publication Evidence`

**Change type**
- code | validation automation

**Definition of done**
- Story 6 includes explicit references to mandatory raw or lower-level artifacts.
- A lightweight completeness checker verifies mandatory artifact presence and
  required metadata fields.
- Bundle validation fails clearly on missing or semantically incomplete evidence.

**Execution checklist**
- [ ] Ensure mandatory artifact files are referenced and resolvable in manifest.
- [ ] Add one lightweight completeness-check routine for required files and
      required metadata fields.
- [ ] Validate mandatory artifact statuses against expected status sets.
- [ ] Keep checker scope aligned with frozen Task 6 inventory (no extra phase
      expansion).

**Evidence produced**
- One Task 6 Story 6 completeness checker with machine-checkable output.
- Explicit missing/mismatched diagnostics for bundle failures.

**Risks / rollback**
- Risk: file-presence-only checks can miss semantic drift or status mismatches.
- Rollback/mitigation: validate both presence and expected status/field
  semantics.

### Engineering Task 6: Add Focused Regression Checks For Task 6 Bundle Integrity

**Implements story**
- `Story 6: Workflow Artifacts Preserve Stable Workflow Identity And Provenance For Reproducible Publication Evidence`

**Change type**
- tests

**Definition of done**
- Focused tests validate Story 6 manifest schema, required artifact IDs, expected
  statuses, and provenance completeness.
- Tests include representative negative cases for missing mandatory artifact,
  mismatched status, and missing identity/provenance field.
- Regression coverage remains lightweight versus full bundle generation.

**Execution checklist**
- [ ] Add focused Story 6 bundle integrity tests in
      `tests/density_matrix/test_density_matrix.py` or a related successor.
- [ ] Add negative test for missing mandatory artifact entry.
- [ ] Add negative test for artifact status outside expected set.
- [ ] Add negative test for missing workflow ID/version or missing top-level
      provenance field.

**Evidence produced**
- Focused Story 6 regression coverage for bundle integrity semantics.
- Reviewable failures for schema and semantic integrity drift.

**Risks / rollback**
- Risk: Story 6 bundle integrity can regress silently without targeted tests.
- Rollback/mitigation: enforce manifest and provenance invariants in fast
  regression coverage.

### Engineering Task 7: Align Developer-Facing And Publication-Facing Notes With Delivered Task 6 Bundle

**Implements story**
- `Story 6: Workflow Artifacts Preserve Stable Workflow Identity And Provenance For Reproducible Publication Evidence`

**Change type**
- docs | validation automation

**Definition of done**
- Documentation identifies Task 6 Story 6 artifact inventory, rerun command,
  and interpretation of top-level status fields.
- Notes map each Story 1 to Story 5 artifact class to publication claim usage.
- Documentation avoids overclaiming beyond frozen Task 6 boundaries.

**Execution checklist**
- [ ] Document Story 6 rerun command, artifact path, and top-level manifest
      fields.
- [ ] Document mapping from artifact class to Task 6 story and publication
      relevance.
- [ ] Document expected status semantics for mandatory artifacts.
- [ ] Keep notes aligned with `TASK_6_MINI_SPEC.md`, `TASK_6_STORIES.md`, and
      Story 1 to Story 5 implementation plans.

**Evidence produced**
- Updated Task 6 Story 6 documentation for developers and publication review.
- One stable reference explaining Task 6 bundle interpretation.

**Risks / rollback**
- Risk: without clear docs, bundle consumers may misinterpret mandatory versus
  supplemental evidence.
- Rollback/mitigation: tie docs directly to the same manifest schema validated in
  Story 6 checks.

### Engineering Task 8: Run Story 6 Bundle Validation And Confirm Publication-Ready Task 6 Evidence Surface

**Implements story**
- `Story 6: Workflow Artifacts Preserve Stable Workflow Identity And Provenance For Reproducible Publication Evidence`

**Change type**
- tests | validation automation

**Definition of done**
- Story 6 focused tests pass.
- Story 6 bundle generation command emits stable manifest and passes integrity
  checks.
- Task 6 evidence surface is rerunnable and machine-auditable.

**Execution checklist**
- [ ] Run focused Story 6 integrity regression tests.
- [ ] Run Story 6 top-level bundle generation command.
- [ ] Verify mandatory artifact inventory presence and expected statuses.
- [ ] Verify workflow/case identity and provenance fields are present and stable.
- [ ] Record test and bundle references for publication handoff.

**Evidence produced**
- Passing Story 6 focused tests.
- Stable Story 6 top-level bundle artifact with pass/fail integrity verdict.

**Risks / rollback**
- Risk: Story 6 can appear complete without executable proof that all evidence is
  assembled and consistent.
- Rollback/mitigation: require successful bundle generation plus integrity check
  pass as exit evidence.

## Exit Criteria

Story 6 is complete only when all of the following are true:

- a stable top-level Task 6 manifest schema and mandatory artifact inventory are
  frozen,
- Story 1 to Story 5 artifacts are assembled into one coherent Task 6 bundle,
- workflow identity, case identity, threshold semantics, and status provenance
  are preserved and validated across artifact classes,
- software/command/environment provenance is present in machine-readable form,
- bundle completeness checker passes for mandatory artifact presence and expected
  status semantics,
- one stable Story 6 artifact or rerunnable command produces publication-ready
  Task 6 evidence output.

## Implementation Notes

- Use `benchmarks/density_matrix/task5_story6_publication_bundle.py` as the
  canonical implementation pattern for top-level manifest assembly and integrity
  checks.
- Treat the emitted Story 1 contract artifact as the root identity surface for
  the whole Task 6 bundle and require later artifact classes to preserve or
  reference its `workflow_id` and `contract_version`.
- Keep Story 6 field vocabulary aligned with existing bundle conventions to
  minimize translation overhead.
- Treat Story 6 as packaging and integrity validation only; do not re-open Story
  1 to Story 5 closure semantics here.
- Prefer one thin top-level bundle that references lower-level artifact files and
  summaries over duplicating all raw payloads inside a monolithic JSON.
