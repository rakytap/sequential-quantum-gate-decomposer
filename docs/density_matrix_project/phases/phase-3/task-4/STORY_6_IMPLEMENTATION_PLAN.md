# Story 6 Implementation Plan

## Story Being Implemented

Story 6: Fused Results And Provenance Stay Comparison-Ready Across Supported
Cases

This is a Layer 4 engineering plan for implementing the sixth behavioral slice
from `TASK_4_STORIES.md`.

## Scope

This story turns Task 4 fused execution into one stable result and audit surface:

- supported fused or near-fused cases emit comparison-ready outputs that later
  sequential and Aer checks can consume directly,
- fused-path provenance extends the shared Task 3 runtime-audit vocabulary
  rather than replacing it,
- fused-coverage and deferral summaries remain structurally stable across
  supported cases,
- and Story 6 closes stable fused output and provenance packaging without yet
  claiming phase-level performance closure.

Out of scope for this story:

- eligibility definition already owned by Story 1,
- the first positive structured fused-runtime slice already owned by Story 2,
- shared fused-capable reuse already owned by Story 3,
- positive semantic-preservation closure already owned by Story 4,
- explicit fused versus unfused versus deferred classification closure already
  owned by Story 5,
- and threshold-or-diagnosis benchmark closure owned by Story 7.

## Dependencies And Assumptions

- Stories 1 through 5 already define the eligibility, execution, shared reuse,
  semantics, and classification surfaces Story 6 must package consistently.
- The frozen source-of-truth contract is `TASK_4_MINI_SPEC.md`,
  `TASK_4_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-005`,
  `P3-ADR-008`, and `P3-ADR-009`.
- Task 3 already defines the positive runtime result and audit substrate Story 6
  must extend through:
  - `NoisyRuntimeExecutionResult`,
  - `NoisyRuntimeExecutionResult.to_dict()`,
  - `NoisyRuntimeExecutionResult.build_exact_output_record()`,
  - `build_runtime_audit_record()`,
  - `runtime_output_validation.py`,
  - and `runtime_audit_validation.py`.
- The existing Task 3 artifact bundles already provide the exact baseline for
  stable field naming and schema shape:
  - `benchmarks/density_matrix/artifacts/partitioned_runtime/story5_results/`,
  - `benchmarks/density_matrix/artifacts/partitioned_runtime/story6_audit/`,
  - and `benchmarks/density_matrix/artifacts/partitioned_runtime/story7_unsupported/`.
- Story 6 should preserve direct compatibility with the sequential density
  comparison path and, where fused microcase coverage is exercised, with the
  later Task 6 Qiskit Aer checks.
- Story 6 should prefer additive schema evolution over renaming shared Task 3
  fields unless a versioned successor is explicitly required and documented.
- The most conservative implementation path is to keep the Task 3 runtime schema
  version stable and add Task 4 fused-region and classification fields
  additively so the existing Task 3 validators remain comparable.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 4 Fused Provenance Tuple And Result-Surface Rule

**Implements story**
- `Story 6: Fused Results And Provenance Stay Comparison-Ready Across Supported Cases`

**Change type**
- docs | validation automation

**Definition of done**
- Story 6 defines one stable fused provenance tuple and one result-surface rule
  for supported Task 4 outputs.
- The rule is explicit enough that later validation and benchmark work can rely
  on it safely.
- The story distinguishes output stability from final performance conclusions.

**Execution checklist**
- [ ] Freeze the minimum fused provenance tuple around planner schema version,
      descriptor schema version, requested mode, source type, entry route,
      workload family, workload ID, runtime-path classification, and fused-
      coverage summary.
- [ ] Freeze the minimum result-surface fields every supported fused or near-
      fused case must expose.
- [ ] Define which fields remain shared with Task 3 unchanged and which are
      additive Task 4 extensions.
- [ ] Prefer additive fused-region fields on the existing Task 3 runtime payload
      before introducing a successor schema version.
- [ ] Keep final performance interpretation outside the Story 6 bar.

**Evidence produced**
- One stable Task 4 fused provenance tuple.
- One explicit fused result-surface rule for supported outputs.

**Risks / rollback**
- Risk: if fused provenance and result fields remain loose, later bundles will
  be hard to compare and easy to misread.
- Rollback/mitigation: freeze the fused audit vocabulary before broadening
  output production.

### Engineering Task 2: Reuse The Task 3 Exact-Output And Audit Surfaces As The Base

**Implements story**
- `Story 6: Fused Results And Provenance Stay Comparison-Ready Across Supported Cases`

**Change type**
- docs | code

**Definition of done**
- Task 4 fused packaging builds on the existing Task 3 exact-output and runtime-
  audit surfaces where fields overlap.
- Task 4-specific extensions are explicit and reviewable.
- Story 6 avoids creating a disconnected fourth result language.

**Execution checklist**
- [ ] Review the Task 3 exact-output and audit fields already emitted by
      `NoisyRuntimeExecutionResult.to_dict()` and `build_runtime_audit_record()`.
- [ ] Reuse overlapping field names directly where they already match the Task 4
      contract.
- [ ] Add only the Task 4-specific fields needed for fused coverage, deferral
      summaries, and reused path labeling.
- [ ] Document where Task 4 intentionally extends the Task 3 output vocabulary.

**Evidence produced**
- One reviewable mapping from Task 3 output fields to Task 4 fused output fields.
- One explicit boundary between reused vocabulary and Task 4-specific
  extensions.

**Risks / rollback**
- Risk: Task 4 may create a disconnected result language that later reviewers
  must translate mentally against Task 3 outputs.
- Rollback/mitigation: align Task 4 output packaging with Task 3 wherever
  practical.

### Engineering Task 3: Define A Shared Fused Runtime-Audit Record And Summary Surface

**Implements story**
- `Story 6: Fused Results And Provenance Stay Comparison-Ready Across Supported Cases`

**Change type**
- code | tests

**Definition of done**
- Supported fused or near-fused cases emit one shared fused runtime-audit record
  shape.
- The record separates case-level provenance, runtime-summary metadata, fused-
  coverage or deferral metadata, and exact-output references cleanly.
- The shape is stable across all supported Task 4 cases.

**Execution checklist**
- [ ] Define one shared top-level Task 4 fused-audit record shape.
- [ ] Record case-level provenance separately from runtime-summary fields,
      fused-coverage summaries, and exact-output references.
- [ ] Keep fused or near-fused cases machine-readable and structurally stable.
- [ ] Add regression checks for top-level schema stability.

**Evidence produced**
- One stable shared Task 4 fused-audit record shape.
- Regression checks for schema stability across supported cases.

**Risks / rollback**
- Risk: later Task 4 outputs may remain individually plausible but structurally
  incomparable.
- Rollback/mitigation: freeze one shared fused record shape before broadening
  bundle emission.

### Engineering Task 4: Cross-Check Structured, Continuity, And Microcase Slices Against The Shared Fused Surface

**Implements story**
- `Story 6: Fused Results And Provenance Stay Comparison-Ready Across Supported Cases`

**Change type**
- tests

**Definition of done**
- Structured, continuity, and microcase fused-capable slices emit records
  through the same shared Task 4 fused surface where they overlap.
- Schema drift across supported cases is caught early.
- The checks stay focused on output and provenance stability rather than on final
  benchmark interpretation.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_partitioned_runtime.py` for fused
      output and audit stability across representative supported cases.
- [ ] Compare top-level provenance, runtime-path classification, exact-output
      shape, fused-coverage summary presence, and summary-field presence across
      the supported cases.
- [ ] Keep the checks narrow to output and audit structure rather than to final
      benchmark thresholds.
- [ ] Fail quickly when supported cases diverge from the shared Task 4 surface.

**Evidence produced**
- Fast regression coverage for cross-case fused output stability.
- Reviewable comparison checks for the shared fused surface.

**Risks / rollback**
- Risk: output drift may remain hidden until paper packaging or broad benchmark
  work.
- Rollback/mitigation: enforce cross-case output checks early.

### Engineering Task 5: Preserve Direct Compatibility With Sequential And Aer Validation Consumers

**Implements story**
- `Story 6: Fused Results And Provenance Stay Comparison-Ready Across Supported Cases`

**Change type**
- code | tests | validation automation

**Definition of done**
- Supported fused outputs remain directly consumable by later sequential-baseline
  validation.
- Where fused microcase coverage is exercised, the outputs remain directly
  consumable by later Task 6 Aer checks without relabeling the runtime path.
- Story 6 does not require rerunning the same case through a different interface
  just to validate it.

**Execution checklist**
- [ ] Keep exact-output shape aligned with the Task 3 result surface.
- [ ] Keep path labeling explicit so later validators can distinguish fused and
      unfused execution honestly.
- [ ] Add focused checks that fused or near-fused microcase outputs can still be
      consumed by downstream exact-baseline tooling without schema conversion.
- [ ] Document any additive Task 4 fields so later validators can ignore them
      safely if they are not needed.

**Evidence produced**
- One reviewable compatibility rule for downstream sequential and Aer consumers.
- Focused regression coverage for compatibility with later validation tooling.

**Risks / rollback**
- Risk: later validation may require ad hoc conversion scripts if Story 6 changes
  result shape too aggressively.
- Rollback/mitigation: keep the exact-output surface compatible and additive.

### Engineering Task 6: Emit A Stable Story 6 Fused-Audit Bundle

**Implements story**
- `Story 6: Fused Results And Provenance Stay Comparison-Ready Across Supported Cases`

**Change type**
- validation automation | docs

**Definition of done**
- Story 6 emits one stable machine-reviewable fused-audit bundle or rerunnable
  checker.
- The bundle records representative supported Task 4 cases through one shared
  schema.
- The bundle is reusable by later Task 6 correctness work and Story 7 benchmark
  work.

**Execution checklist**
- [ ] Add a Story 6 validator under
      `benchmarks/density_matrix/partitioned_runtime/`, with
      `fused_runtime_audit_validation.py` as the primary checker.
- [ ] Add a dedicated Story 6 artifact location
      (for example `benchmarks/density_matrix/artifacts/partitioned_runtime/fused_runtime_audit/`).
- [ ] Emit representative structured, continuity, and microcase cases through one
      stable shared schema.
- [ ] Record rerun commands and software metadata with the emitted bundle.

**Evidence produced**
- One stable Story 6 fused-audit bundle or checker.
- One reusable output and provenance surface for later Task 4 work.

**Risks / rollback**
- Risk: if Story 6 emits only prose commitments, later tasks will not have a
  stable output surface to cite or validate.
- Rollback/mitigation: emit one machine-reviewable fused-audit bundle and keep
  it schema-stable.

### Engineering Task 7: Document And Run The Story 6 Fused Output Gate

**Implements story**
- `Story 6: Fused Results And Provenance Stay Comparison-Ready Across Supported Cases`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the supported Task 4 fused result surface.
- Fast regression checks and the Story 6 fused-audit bundle run successfully.
- Story 6 closes with a stable review path for fused outputs and provenance.

**Execution checklist**
- [ ] Document the shared Task 4 fused output and audit rule.
- [ ] Explain how Story 6 differs from Story 5 classification closure and Story
      7 performance closure.
- [ ] Run focused Story 6 regression coverage and verify
      `benchmarks/density_matrix/partitioned_runtime/fused_runtime_audit_validation.py`.
- [ ] Record stable test and artifact references for later Task 4 and Task 6
      work.

**Evidence produced**
- Passing Story 6 fused output and audit regression checks.
- One stable Story 6 fused-audit bundle or checker reference.

**Risks / rollback**
- Risk: Story 6 may appear complete while still leaving implementers unsure how
  fused outputs are reviewed consistently across cases.
- Rollback/mitigation: document the rule and require a rerunnable fused-audit
  bundle.

## Exit Criteria

Story 6 is complete only when all of the following are true:

- supported fused or near-fused cases emit one shared comparison-ready result
  and provenance surface,
- overlapping fields remain aligned with the Task 3 output and audit surfaces,
- downstream sequential validation, and where exercised Aer validation, can
  consume Task 4 outputs without rerunning cases through a different interface,
- one stable Story 6 fused-audit bundle or checker exists for later reuse,
- and threshold-or-diagnosis benchmark interpretation remains clearly assigned
  to Story 7.

## Implementation Notes

- Prefer additive schema extension over renaming shared Task 3 fields unless a
  versioned successor is truly necessary.
- Keep Story 6 focused on output and provenance stability, not on final
  benchmark storytelling.
- Treat direct compatibility with later validation tooling as a required output,
  not as a nice-to-have.
