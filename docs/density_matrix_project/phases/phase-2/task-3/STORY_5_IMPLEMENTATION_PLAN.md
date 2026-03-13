# Story 5 Implementation Plan

## Story Being Implemented

Story 5: Bridge Artifacts Preserve Circuit Source And Lowering Provenance For
Reproducible Publication Evidence

This is a Layer 4 engineering plan for implementing the fifth behavioral slice
from `TASK_3_STORIES.md`.

## Scope

This story turns the Task 3 bridge outputs from Stories 1 to 4 into a
reproducible, publication-ready provenance surface:

- supported bridge-positive, micro-validation, workflow-scale, optimization
  trace, and unsupported-case outputs are assembled into one coherent bridge
  evidence package,
- stored artifacts preserve the circuit source, bridge route, lowered
  gate/noise metadata, backend, and unsupported diagnostics needed for audit,
- the bundle records enough command, software, and schema provenance for a
  reviewer to rerun or inspect the bridge evidence honestly,
- and the resulting package can support the bridge-related claims in the Phase 2
  abstract, short paper, and full paper without relying on hidden assumptions
  about lowering behavior.

Out of scope for this story:

- widening the supported bridge surface,
- adding new workload families or optimizer-comparison science,
- broad simulator bake-offs beyond what existing workflow bundles already carry,
- and paper writing itself beyond stabilizing the evidence surface it will cite.

## Dependencies And Assumptions

- Stories 1 to 4 are already in place: positive bridge behavior, local support
  validation, unsupported-case evidence, and workflow-scale bridge artifacts all
  exist.
- `benchmarks/density_matrix/story2_vqe_density_validation.py` already contains
  several useful artifact helpers, including `build_case_metadata()`,
  `validate_artifact_payload()`, `build_story1_bridge_metadata()`, and Story 5
  bundle-style manifest entries.
- Story 1 already established the canonical supported-bridge schema for
  fixed-parameter artifacts:
  `bridge_source_type`, `bridge_parameter_count`, `bridge_operation_count`,
  `bridge_gate_count`, `bridge_noise_count`, and `bridge_operations`.
- The frozen publication expectations remain `PUBLICATIONS.md`,
  `DETAILED_PLANNING_PHASE_2.md`, `TASK_3_MINI_SPEC.md`, and the Paper 1 claims
  bounded by `P2-ADR-011`, `P2-ADR-012`, and `P2-ADR-013`.
- Story 5 should assemble and validate bridge evidence; it should not reopen the
  supported source, gate, noise, or workflow surface.

## Engineering Tasks

### Engineering Task 1: Freeze The Story 5 Bridge Evidence Inventory And Bundle Manifest

**Implements story**
- `Story 5: Bridge Artifacts Preserve Circuit Source And Lowering Provenance For Reproducible Publication Evidence`

**Change type**
- docs | validation automation

**Definition of done**
- The Story 5 evidence inventory explicitly names the required Task 3 artifact
  classes: supported positive bridge artifact, micro-validation bundle,
  workflow-scale bundle, supported optimization trace, and representative
  unsupported outputs.
- A stable manifest format exists for the full Story 5 bridge bundle.
- Bundle completeness is judged against a named checklist rather than ad hoc
  file presence.

**Execution checklist**
- [ ] Define the required artifact inventory for the Task 3 bridge evidence
      package.
- [ ] Treat the existing Story 1 fixed-parameter artifacts as the canonical
      supported positive bridge artifact class rather than inventing a second
      “bridge positive” format.
- [ ] Create a stable manifest shape that records artifact identity, purpose,
      generation command, and status.
- [ ] Distinguish mandatory bundle items from optional supporting artifacts.
- [ ] Keep the manifest narrow enough for Phase 2 while still audit-friendly for
      publication use.

**Evidence produced**
- A stable Story 5 bridge evidence manifest for the full Task 3 bundle.
- Explicit bundle-completeness criteria for publication-ready bridge evidence.

**Risks / rollback**
- Risk: without an explicit inventory, later bridge claims may rely on partial
  or inconsistent artifacts.
- Rollback/mitigation: freeze the evidence inventory first and make bundle
  generation validate against it.

### Engineering Task 2: Extend Artifact Schemas With Bridge-Provenance Fields

**Implements story**
- `Story 5: Bridge Artifacts Preserve Circuit Source And Lowering Provenance For Reproducible Publication Evidence`

**Change type**
- code | validation automation

**Definition of done**
- Story 5 artifact schemas record the bridge-specific metadata needed for audit.
- Required fields include source type, ansatz, backend, bridge route, and a
  stable way to identify the lowered gate/noise content or summary.
- Bridge-provenance fields remain consistent across supported and unsupported
  artifact classes where they apply.

**Execution checklist**
- [ ] Extend the current case-metadata and bundle-entry schema with Task 3
      bridge-provenance fields.
- [ ] Keep the Story 1 bridge fields
      (`bridge_source_type`, `bridge_parameter_count`,
      `bridge_operation_count`, `bridge_gate_count`, `bridge_noise_count`, and
      `bridge_operations`) as the canonical base schema across supported fixed-
      parameter, micro, workflow, and optimization-trace artifacts where
      possible.
- [ ] Avoid introducing bundle fields that imply broader bridge support than the
      frozen contract.
- [ ] Validate schema changes with one narrow compatibility check before
      expanding usage across all Story 5 artifacts.

**Evidence produced**
- One stable Task 3 bridge-provenance schema reused across artifact classes.
- Reviewable machine-readable fields for circuit source and bridge route.

**Risks / rollback**
- Risk: inconsistent bridge field naming across artifacts will make bundle
  assembly and review brittle.
- Rollback/mitigation: freeze a small common schema and extend it only when a
  concrete bridge evidence need requires it.

### Engineering Task 3: Capture Lowered Gate And Noise Provenance In A Reviewable Form

**Implements story**
- `Story 5: Bridge Artifacts Preserve Circuit Source And Lowering Provenance For Reproducible Publication Evidence`

**Change type**
- code | validation automation

**Definition of done**
- Supported Task 3 artifacts preserve reviewable lowered gate/noise provenance,
  either as a raw ordered operation listing or as a stable summarized form.
- Reviewers can determine how a circuit reached `NoisyCircuit` without relying
  on hidden implementation assumptions.
- Large-case provenance capture stays practical while still auditable.

**Execution checklist**
- [ ] Reuse the Story 1 and Story 4 bridge-inspection outputs for provenance
      capture where possible.
- [ ] Decide which artifact classes retain raw ordered-operation data and which
      use stable summarized bridge metadata.
- [ ] Keep raw `bridge_operations` mandatory for the small Story 1 fixed-
      parameter artifacts; allow summarized bridge metadata only where workflow
      scale makes the raw form impractical.
- [ ] Keep provenance capture consistent enough that a reviewer can trace from a
      workflow result back to the bridge output that produced it.
- [ ] Avoid storing provenance only in prose notes or console logs.

**Evidence produced**
- Reviewable lowered gate/noise provenance for mandatory supported Task 3 cases.
- Stable linkage from case-level artifacts to the bridged operation surface.

**Risks / rollback**
- Risk: provenance that is too thin makes the bridge claim hard to audit;
  provenance that is too large can become impractical and unstable.
- Rollback/mitigation: keep one explicit policy for raw versus summarized bridge
  metadata and apply it consistently.

### Engineering Task 4: Preserve Unsupported Diagnostics In The Same Bundle Vocabulary

**Implements story**
- `Story 5: Bridge Artifacts Preserve Circuit Source And Lowering Provenance For Reproducible Publication Evidence`

**Change type**
- validation automation | docs

**Definition of done**
- Representative unsupported Task 3 cases are included in the Story 5 bundle
  using the same manifest vocabulary as supported artifacts.
- Unsupported artifacts preserve the first failing bridge condition and its
  category.
- Positive bridge evidence and negative unsupported evidence remain clearly
  separated while still being auditable together.

**Execution checklist**
- [ ] Reuse the structured unsupported artifact pattern from Story 3 in the
      Story 5 bundle manifest.
- [ ] Record unsupported category, first failing condition, backend, and source
      context for each included negative artifact.
- [ ] Keep unsupported artifacts visibly unsupported rather than mixing them into
      supported pass summaries.
- [ ] Make clear which unsupported artifacts are mandatory representatives versus
      optional extras.

**Evidence produced**
- Story 5 manifest entries linking representative unsupported bridge artifacts.
- Stable negative-evidence records aligned with the same bundle vocabulary as
  supported outputs.

**Risks / rollback**
- Risk: unsupported evidence can disappear from publication-facing bundles if it
  is not given a first-class manifest presence.
- Rollback/mitigation: keep at least one representative unsupported artifact as a
  named mandatory bundle item.

### Engineering Task 5: Add A Minimal Bridge-Bundle Completeness Checker

**Implements story**
- `Story 5: Bridge Artifacts Preserve Circuit Source And Lowering Provenance For Reproducible Publication Evidence`

**Change type**
- code | validation automation

**Definition of done**
- A lightweight completeness checker can assert that the Story 5 bundle contains
  every mandatory bridge artifact and metadata field.
- Bundle validation fails clearly when required bridge provenance is missing or
  malformed.
- Completeness checks remain aligned with the frozen Task 3 evidence inventory
  rather than expanding into later-phase expectations.

**Execution checklist**
- [ ] Add a small bundle-validation routine that checks required files and
      bridge-provenance fields.
- [ ] Reuse existing manifest-validation patterns where possible rather than
      inventing a second bundle checker.
- [ ] Require the canonical Story 1 bridge fields on mandatory supported
      positive artifacts and the corresponding summary fields on larger derived
      artifacts.
- [ ] Keep completeness output human-readable and machine-checkable.
- [ ] Make missing mandatory provenance fields fail specifically rather than as
      generic parse errors.

**Evidence produced**
- A minimal completeness checker for the Story 5 bridge bundle.
- Clear validation output for missing artifacts or missing bridge metadata.

**Risks / rollback**
- Risk: summary-only bundles can drift silently and leave bridge claims
  unsupported even when some artifact files exist.
- Rollback/mitigation: validate required files and required bridge-provenance
  fields together through one checker.

### Engineering Task 6: Add Focused Integrity Tests For Bridge Bundle Schema And Manifest

**Implements story**
- `Story 5: Bridge Artifacts Preserve Circuit Source And Lowering Provenance For Reproducible Publication Evidence`

**Change type**
- tests | validation automation

**Definition of done**
- Focused automated checks verify the Story 5 manifest, bridge-provenance
  schema, and completeness checker.
- Integrity checks catch missing mandatory fields before later docs or papers
  rely on them.
- The regression surface stays light enough to run regularly while still
  protecting the Task 3 evidence contract.

**Execution checklist**
- [ ] Add focused tests for the Story 5 manifest structure and completeness
      checker.
- [ ] Add focused checks for mandatory bridge-provenance fields on supported and
      unsupported artifact classes.
- [ ] Keep full bundle generation in dedicated validation commands rather than
      the default fast suite.
- [ ] Ensure failures localize missing bridge metadata rather than collapsing
      into generic parsing errors.

**Evidence produced**
- Focused regression coverage for Story 5 bridge-bundle integrity.
- Reviewable failures when mandatory evidence fields or files are missing.

**Risks / rollback**
- Risk: without integrity checks, the bundle can drift silently and break later
  publication assembly.
- Rollback/mitigation: keep a small test surface that locks the bridge evidence
  contract down.

### Engineering Task 7: Align Developer-Facing And Publication-Facing Notes With The Delivered Bridge Bundle

**Implements story**
- `Story 5: Bridge Artifacts Preserve Circuit Source And Lowering Provenance For Reproducible Publication Evidence`

**Change type**
- docs

**Definition of done**
- Developer-facing notes explain how to regenerate and validate the Story 5
  bridge bundle.
- Publication-facing notes accurately describe which bridge evidence is already
  bundled and what claims it supports.
- The notes remain explicit about the bundle being a bounded Phase 2 bridge
  evidence package rather than a generic future-parity artifact dump.

**Execution checklist**
- [ ] Document the Story 5 bridge bundle inventory, manifest, and rerun
      commands.
- [ ] Explain how the bridge-positive, micro-validation, workflow, optimization,
      and representative unsupported artifacts support the abstract, short
      paper, and full paper claims.
- [ ] Keep wording aligned with `PUBLICATIONS.md` and the frozen Task 3
      acceptance rules.
- [ ] Avoid overstating the bundle as proof of full circuit parity or later
      acceleration work.

**Evidence produced**
- Updated developer-facing instructions for generating and validating the Story
  5 bridge bundle.
- Publication-facing notes that correctly map bundled bridge evidence to Paper 1
  claim surfaces.

**Risks / rollback**
- Risk: if notes drift from the actual delivered bundle, paper claims can outrun
  the evidence.
- Rollback/mitigation: tie the notes directly to the same manifest and bundle
  checker used in Story 5 validation.

### Engineering Task 8: Run Story 5 Bundle Generation And Confirm Publication-Ready Bridge Evidence

**Implements story**
- `Story 5: Bridge Artifacts Preserve Circuit Source And Lowering Provenance For Reproducible Publication Evidence`

**Change type**
- tests | validation automation

**Definition of done**
- The Story 5 bridge bundle generates successfully end to end.
- The completeness checker passes for all mandatory bridge evidence items.
- Story 5 completion is backed by reviewable bundle references that can be cited
  directly in later Task 3 docs and Phase 2 papers.

**Execution checklist**
- [ ] Run the Story 5 bridge bundle-generation command or workflow.
- [ ] Run the bundle completeness checker and focused integrity tests.
- [ ] Confirm the mandatory supported and unsupported artifact classes are all
      present and linked.
- [ ] Record stable bundle references for later Task 3 docs and publication
      drafting.

**Evidence produced**
- A complete Story 5 reproducibility and bridge-provenance bundle.
- Passing bundle-integrity checks and stable artifact references.

**Risks / rollback**
- Risk: Story 5 can appear complete while still lacking a coherent, validated
  bridge evidence package.
- Rollback/mitigation: treat successful bundle generation and completeness
  validation as part of the exit gate, not optional cleanup.

## Exit Criteria

Story 5 is complete only when all of the following are true:

- the Story 1 positive bridge artifact, Story 2 micro-validation bundle, Story
  4 workflow bundle, one supported optimization trace, and representative Story
  3 unsupported artifacts are all referenced in one stable manifest,
- software, command, revision, and bridge-provenance metadata is recorded for
  the mandatory evidence items,
- stored artifacts let a reviewer determine circuit source, bridge route,
  lowered gate/noise provenance or summary, backend, and unsupported diagnostics
  for each mandatory case,
- a bundle completeness checker validates that all mandatory Story 5 bridge
  evidence is present,
- and the resulting bundle can support the bridge-related claims in the Phase 2
  abstract, short paper, and full paper without relying on implicit context.

## Implementation Notes

- `benchmarks/density_matrix/story2_vqe_density_validation.py` already contains
  the natural starting point for Story 5 manifest and bundle logic and should be
  extended rather than replaced.
- Story 5 should reuse artifact vocabulary from Stories 1 to 4, with the
  Story 1 fixed-parameter bridge fields and `build_story1_bridge_metadata()` as
  the canonical starting point for supported-bridge provenance.
- `PUBLICATIONS.md` makes reproducible benchmark definitions and configuration
  logging central to Paper 1 credibility, so Story 5 should prioritize audit
  clarity over feature breadth.
- Story 1 established a workable raw-versus-summary provenance split:
  small fixed-parameter artifacts can retain full `bridge_operations`, while
  larger workflow artifacts may need summarized bridge metadata so long as the
  linkage remains auditable.
- Story 5 is about evidence integrity and publication readiness for the bridge
  contract, not about widening the supported backend surface or making new
  scientific claims.
