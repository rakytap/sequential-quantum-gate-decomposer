# Story 6 Implementation Plan

## Story Being Implemented

Story 6: Lossy Or Unsupported Descriptor Generation Fails Before Runtime With
Structured Diagnostics

This is a Layer 4 engineering plan for implementing the sixth behavioral slice
from `TASK_2_STORIES.md`.

## Scope

This story turns the Task 2 negative boundary into an enforceable
descriptor-generation gate:

- descriptor generation fails before runtime when it would drop operations,
  obscure noise placement, produce ambiguous parameter routing, or require
  incomplete remapping,
- failure reporting records stable unsupported category, first unsupported
  condition, and failure-stage metadata in a machine-reviewable way,
- validation and benchmark surfaces cannot quietly relabel failed
  descriptor-generation cases as supported Task 2 behavior,
- and the unsupported-descriptor evidence is explicit enough to support honest
  Paper 2 scope language.

Out of scope for this story:

- positive workload coverage already owned by Stories 1 and 2,
- positive order/noise semantics already owned by Story 3,
- positive reconstructibility closure already owned by Story 4,
- cross-workload audit stability already owned by Story 5,
- and runtime numerical failures, fused execution issues, or performance
  diagnosis owned by later Phase 3 tasks.

## Dependencies And Assumptions

- Stories 1 through 5 already define the supported positive descriptor slices
  that Story 6 must protect.
- Task 1 already established a structured unsupported vocabulary at
  planner-entry time, including category, first unsupported condition, and
  failure-stage concepts that Story 6 should align with where practical.
- The shared Task 2 descriptor substrate now lives in
  `squander/partitioning/noisy_planner.py`, where
  `NoisyDescriptorValidationError` already exists as the natural home for the
  Task 2 structured unsupported vocabulary.
- Story 2 now also established the first supported multi-workload descriptor
  fixture matrix in
  `benchmarks/density_matrix/planner_surface/workloads.py` and the emitted
  bundle under
  `benchmarks/density_matrix/artifacts/planner_surface/mandatory_workload/`; Story
  6 should use those supported cases as the comparison baseline for negative
  evidence.
- Story 4 now also established the shared positive reconstruction checker
  `validate_partition_descriptor_set()` and froze the concrete reconstruction
  field names `local_to_global_qbits`, `global_to_local_qbits`,
  `requires_remap`, and `parameter_routing`. Story 6 should use those exact
  names when defining ambiguous-routing or incomplete-remap failures.
- Story 5 now also established the shared positive audit helper
  `build_descriptor_audit_record()` and the emitted bundle under
  `benchmarks/density_matrix/artifacts/planner_surface/descriptor_audit/`; Story 6
  should keep overlapping provenance and metadata fields aligned with that
  positive audit surface.
- The likely shared implementation substrate therefore remains the Task 2
  descriptor generation layer under `squander/partitioning/` plus the existing
  audit and artifact-emission surfaces under `benchmarks/density_matrix/artifacts/`.
- Story 6 should introduce one explicit unsupported-descriptor vocabulary rather
  than allowing every test, script, or wrapper to invent its own failure
  taxonomy.
- The current codebase does not yet expose the full Task 3 partitioned density
  runtime, so Story 6 must enforce the negative boundary before runtime exists.
- Story 6 should treat negative evidence as a required output, not as an
  optional appendix to positive descriptor cases.

## Engineering Tasks

### Engineering Task 1: Freeze The Unsupported-Descriptor Taxonomy For Task 2

**Implements story**
- `Story 6: Lossy Or Unsupported Descriptor Generation Fails Before Runtime With Structured Diagnostics`

**Change type**
- docs | validation automation

**Definition of done**
- Story 6 defines one stable unsupported-boundary taxonomy for descriptor
  generation.
- The taxonomy covers the minimum Task 2 failure categories needed by the
  contract.
- Descriptor-generation failures are separated clearly from later runtime or
  numerical failures.

**Execution checklist**
- [ ] Freeze the minimum unsupported categories for Task 2:
      dropped operations, boundary-only noise treatment, ambiguous
      parameter routing, incomplete remapping, undocumented reordering across
      noise boundaries, malformed descriptor request, and disallowed mode or
      support-surface violations where they appear at descriptor generation
      time.
- [ ] Freeze one first-unsupported-condition vocabulary for
      machine-reviewable artifacts.
- [ ] Distinguish descriptor-generation failure stages from later runtime or
      benchmark failures.
- [ ] Align the Task 2 taxonomy with the Task 1 unsupported vocabulary wherever
      the meanings overlap.

**Evidence produced**
- One stable Story 6 unsupported-descriptor taxonomy.
- One reviewable first-unsupported-condition vocabulary for Task 2.

**Risks / rollback**
- Risk: if unsupported categories remain ad hoc, later validation and paper
  packaging will struggle to compare negative evidence.
- Rollback/mitigation: define one shared unsupported taxonomy before wiring the
  failure surfaces.

### Engineering Task 2: Add A Descriptor-Generation Preflight Or Validation Gate

**Implements story**
- `Story 6: Lossy Or Unsupported Descriptor Generation Fails Before Runtime With Structured Diagnostics`

**Change type**
- code | tests

**Definition of done**
- Supported Task 2 requests pass through one explicit descriptor-generation
  validation surface.
- The validator can reject lossy or unsupported conditions before runtime
  begins.
- The validator does not rely on hidden fallback or partial descriptor
  substitution to make an unsupported request succeed.

**Execution checklist**
- [ ] Introduce one Task 2 descriptor-generation validation surface in the
      planner or descriptor module, with `squander/partitioning/` as the
      default implementation home.
- [ ] Route supported continuity, methods, and optional exact-lowering requests
      through that gate before claiming descriptor success.
- [ ] Reject unsupported descriptor-generation conditions with stable category,
      first-condition, and failure-stage metadata.
- [ ] Keep descriptor-generation validation separate from later runtime
      execution checks.

**Evidence produced**
- One explicit Task 2 descriptor-generation validation gate.
- Focused regression coverage for pre-runtime rejection behavior.

**Risks / rollback**
- Risk: unsupported cases may still reach downstream code paths where loss or
  fallback becomes hard to reason about.
- Rollback/mitigation: make the descriptor-generation gate mandatory before any
  Task 2 support is claimed.

### Engineering Task 3: Detect Lossy Semantic Transformations Explicitly

**Implements story**
- `Story 6: Lossy Or Unsupported Descriptor Generation Fails Before Runtime With Structured Diagnostics`

**Change type**
- code | tests

**Definition of done**
- Descriptor generation can detect the main lossy semantic failures defined by
  Task 2.
- The checks are explicit enough to expose why a case failed rather than only
  that it failed.
- The supported positive contract remains protected against silent semantic
  erosion.

**Execution checklist**
- [ ] Add explicit checks for dropped operations and incomplete descriptor
      membership.
- [ ] Add explicit checks for boundary-only or otherwise hidden treatment of
      noise placement.
- [ ] Add explicit checks for ambiguous parameter-routing metadata.
- [ ] Add explicit checks for incomplete or non-invertible remapping metadata.
- [ ] Add explicit checks for undocumented reordering across noise boundaries.

**Evidence produced**
- One reviewable set of lossy-transformation checks for Task 2.
- Focused regression coverage tying concrete negative cases to concrete failure
  categories.

**Risks / rollback**
- Risk: generic failure handling may hide which semantic guarantee was actually
  violated.
- Rollback/mitigation: detect and label each lossy transformation class
  explicitly.

### Engineering Task 4: Align Structured Diagnostics With The Shared Audit Surface

**Implements story**
- `Story 6: Lossy Or Unsupported Descriptor Generation Fails Before Runtime With Structured Diagnostics`

**Change type**
- code | docs

**Definition of done**
- Story 6 negative evidence fits alongside the shared audit surface from Story
  5 where fields overlap.
- Unsupported descriptor records stay machine-reviewable and comparable.
- Task 2 does not introduce a disconnected error-only reporting format.

**Execution checklist**
- [ ] Reuse the shared case-level provenance tuple on unsupported records where
      it still applies.
- [ ] Keep overlapping fields aligned with the Story 5 audit surface for easy
      comparison between supported and unsupported cases.
- [ ] Add the structured unsupported fields without replacing the shared
      artifact vocabulary.
- [ ] Document how unsupported descriptor records relate to supported audit
      records.

**Evidence produced**
- One aligned Task 2 negative-evidence record shape.
- One explicit mapping between supported and unsupported descriptor artifacts.

**Risks / rollback**
- Risk: unsupported evidence may remain structurally incomparable to supported
  evidence even if both look reasonable on their own.
- Rollback/mitigation: align overlapping fields with the shared audit surface.

### Engineering Task 5: Block Mislabeling In Validation, Harnesses, And Artifact Labels

**Implements story**
- `Story 6: Lossy Or Unsupported Descriptor Generation Fails Before Runtime With Structured Diagnostics`

**Change type**
- code | validation automation | docs

**Definition of done**
- No validation or benchmark surface can claim supported Task 2 descriptor
  behavior after a failed descriptor-generation attempt.
- Artifact labels make unsupported status visible rather than silently
  substituting a different path.
- The no-mislabeling rule is enforced even before the full partitioned runtime
  exists.

**Execution checklist**
- [ ] Review validation and artifact-emission entry points that may later claim
      Task 2 descriptor behavior.
- [ ] Add explicit status or support labels that prevent failed descriptor
      cases from being recorded as supported.
- [ ] Ensure any sequential-density oracle path remains labeled as reference
      behavior rather than fallback success.
- [ ] Reuse the Task 2 audit-bundle style established in Story 5 where
      practical instead of inventing local failure semantics.

**Evidence produced**
- Reviewable artifact-labeling rules for Task 2 unsupported cases.
- Focused checks proving failed descriptor-generation cases are not mislabeled.

**Risks / rollback**
- Risk: even correct validation logic can be undermined if harnesses relabel
  failed cases as supported output.
- Rollback/mitigation: enforce status labeling wherever Task 2 behavior is
  recorded.

### Engineering Task 6: Add A Representative Unsupported-Descriptor Matrix

**Implements story**
- `Story 6: Lossy Or Unsupported Descriptor Generation Fails Before Runtime With Structured Diagnostics`

**Change type**
- tests | validation automation

**Definition of done**
- Story 6 covers representative unsupported cases across the Task 2 boundary.
- Negative cases produce stable category, first-condition, and failure-stage
  metadata.
- Unsupported cases are tested at descriptor generation time rather than only
  through later runtime failure.

**Execution checklist**
- [ ] Add representative negative cases for dropped operations, hidden noise
      placement, ambiguous parameter routing, incomplete remapping, and
      undocumented reordering.
- [ ] Reuse or extend Task 1 unsupported-case patterns where they provide a good
      model for structured diagnostics.
- [ ] Add a fast regression slice in `tests/partitioning/test_planner_surface_descriptors.py`
      that asserts failure occurs before runtime.
- [ ] Keep the matrix representative and contract-driven rather than exhaustive
      over every impossible combination.

**Evidence produced**
- Focused Story 6 regression coverage for representative unsupported descriptor
  requests.
- A representative unsupported-descriptor matrix tied to the Task 2 contract.

**Risks / rollback**
- Risk: without representative negative coverage, unsupported behavior may be
  tested only opportunistically and semantic-loss bugs can hide.
- Rollback/mitigation: freeze a small but contract-complete negative matrix and
  run it routinely.

### Engineering Task 7: Emit And Document A Stable Story 6 Unsupported-Descriptor Bundle

**Implements story**
- `Story 6: Lossy Or Unsupported Descriptor Generation Fails Before Runtime With Structured Diagnostics`

**Change type**
- validation automation | docs

**Definition of done**
- Story 6 emits one stable machine-reviewable unsupported-descriptor bundle or
  rerunnable checker.
- The bundle records unsupported category, first unsupported condition, failure
  stage, and no-mislabeling outcome for representative cases.
- Developer-facing notes explain the Task 2 unsupported boundary concretely.

**Execution checklist**
- [ ] Add a dedicated Story 6 artifact location
      (for example `benchmarks/density_matrix/artifacts/planner_surface/unsupported_descriptor/`).
- [ ] Emit representative unsupported cases through one stable schema.
- [ ] Record rerun commands, software metadata, and the no-mislabeling
      interpretation with the emitted bundle.
- [ ] Document how the Story 6 negative bundle complements the Story 5 positive
      audit bundle.

**Evidence produced**
- One stable Story 6 unsupported-descriptor bundle or checker.
- One stable developer-facing reference for the Task 2 negative boundary.

**Risks / rollback**
- Risk: unsupported behavior may remain documented only in prose, leaving later
  reviewers unable to tell how the Task 2 boundary was actually enforced.
- Rollback/mitigation: emit one structured negative bundle and document it
  together with the rule.

## Exit Criteria

Story 6 is complete only when all of the following are true:

- lossy or unsupported Task 2 descriptor requests fail before runtime with one
  stable unsupported taxonomy,
- descriptor-generation failures expose stable category, first unsupported
  condition, and failure-stage metadata,
- validation and artifact surfaces cannot silently relabel failed
  descriptor-generation cases as supported Task 2 behavior,
- one stable Story 6 unsupported-descriptor bundle or checker exists for
  negative evidence,
- and later runtime numerical failures and performance diagnosis remain
  separated from Task 2 unsupported-boundary closure.

## Implementation Notes

- Prefer one shared unsupported-descriptor vocabulary over multiple
  script-specific error taxonomies.
- Treat negative evidence as a required output of Task 2, not as an optional
  appendix to supported cases.
- Keep the Task 2 unsupported boundary enforceable before the full partitioned
  runtime exists.
