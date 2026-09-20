# Engineering Task P31-S10-E02 Implementation Plan

## Engineering Task Being Implemented

`Engineering Task P31-S10-E02: Emit the versioned correctness package with Aer rows and Phase 3.1 slices`

This is a Layer 4 file-level implementation plan for the second engineering task
under Story `P31-S10` from
`../FOURTH_VERTICAL_SLICE_REMAINING_COUNTED_CORRECTNESS_STORIES_AND_ENGINEERING_TASKS.md`.
It turns the fourth-slice Task 3 package-migration wording into a concrete plan
against the current correctness-evidence builders in:

- `task-3/TASK_3_MINI_SPEC.md`,
- `task-3/ENGINEERING_TASK_P31_S10_E01_IMPLEMENTATION_PLAN.md`,
- `benchmarks/density_matrix/correctness_evidence/case_selection.py`,
- `benchmarks/density_matrix/correctness_evidence/records.py`,
- `benchmarks/density_matrix/correctness_evidence/common.py`,
- `benchmarks/density_matrix/correctness_evidence/bundle.py`,
- `benchmarks/density_matrix/correctness_evidence/validation_support.py`,
- `benchmarks/density_matrix/correctness_evidence/correctness_matrix_validation.py`,
- `benchmarks/density_matrix/correctness_evidence/sequential_correctness_validation.py`,
- `benchmarks/density_matrix/correctness_evidence/external_correctness_validation.py`,
- `benchmarks/density_matrix/correctness_evidence/output_integrity_validation.py`,
- `benchmarks/density_matrix/correctness_evidence/runtime_classification_validation.py`,
- `benchmarks/density_matrix/correctness_evidence/correctness_bundle_validation.py`,
- `benchmarks/density_matrix/correctness_evidence/summary_consistency_validation.py`,
- `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`,
- `benchmarks/density_matrix/evidence_core.py`,
- and, for supporting runtime semantics only,
  `squander/partitioning/noisy_runtime.py`,
  `squander/partitioning/noisy_runtime_core.py`,
  `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`,
  `tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py`.

## Scope

This engineering task promotes the now-completed counted correctness gates into a
**versioned correctness-evidence successor surface** for the bounded Phase 3.1
fourth slice:

- keep the frozen counted Phase 3.1 correctness IDs unchanged:
  - `phase31_microcase_1q_u3_local_noise_chain`,
  - `phase31_microcase_2q_cnot_local_noise_pair`,
  - `phase31_microcase_2q_multi_noise_entangler_chain`,
  - `phase31_microcase_2q_dense_same_support_motif`,
  - `phase2_xxz_hea_q4_continuity`,
  - `phase2_xxz_hea_q6_continuity`,
- emit a bounded Stage-A correctness package for the fourth slice that includes:
  - the four counted microcases,
  - the two counted continuity anchors,
  - the required external slice on the same bounded subset
    (`q4`, not `q6`),
- route microcases through the **strict** public Phase 3.1 path and continuity
  anchors through the **hybrid** public Phase 3.1 path,
- add the required Phase 3.1 counted-case fields and slices:
  - `claim_surface_id`,
  - `runtime_class`,
  - `representation_primary`,
  - `fused_block_support_qbits`,
  - `contains_noise`,
  - `counted_phase31_case`,
  - `channel_invariants`,
  - `partition_route_summary` for hybrid-counted rows,
- expand the external validation bundle from the old 4-row assumption to the
  bounded 5-row Phase 3.1 external slice,
- and provide a sibling validation pipeline that can run the Phase 3.1 package
  without yet mutating the current default Phase 3 correctness pipeline.

Out of scope for this engineering task:

- the counted structured performance carry-forward rows from `P31-ADR-010`,
- the Task 4 performance matrix and `break_even_table` / `justification_map`,
- publication/doc closure work from Task 5,
- switching the repository-wide default correctness pipeline from Phase 3 to
  Phase 3.1 before the sibling Phase 3.1 package is validated,
- changes to the runtime API or runtime-path naming,
- and any host-acceleration or Task 6 work.

## Current Evidence-Package Gap To Close

`P31-S10-E01` is designed to finish the remaining runtime-level correctness
gates. The broader correctness-evidence package still reflects the old Phase 3
surface and semantics.

### `benchmarks/density_matrix/correctness_evidence/case_selection.py`

The current file already contains planning helpers for the Phase 3.1 slice:

- `iter_phase31_correctness_microcase_cases()`,
- `iter_phase31_correctness_continuity_cases()`,
- and `iter_phase31_correctness_structured_cases()`.

However, the active builder remains:

- `build_correctness_evidence_case_contexts()`,

and it still returns the old Phase 3 default matrix:

- 4 continuity cases,
- 3 microcases,
- 18 structured cases.

So the Phase 3.1 selectors exist, but the package does not yet execute them.

### `benchmarks/density_matrix/correctness_evidence/records.py`

The current positive record builder still routes everything through:

- `execute_fused_with_reference(...)`

rather than through the explicit Phase 3.1 strict / hybrid public paths.

That means the current correctness package does **not** yet:

- distinguish strict versus hybrid execution for Phase 3.1 counted cases,
- emit the required `channel_invariants` slice,
- emit `partition_route_summary` for hybrid rows,
- or add the required Phase 3.1 counted-case fields from planning §10.6.

It also means the shared bridge-field helper in `evidence_core.py` is still
Phase-3-shaped. That is acceptable, but the required Phase 3.1 fields need to
be added **alongside** the existing bridge fields rather than silently mutating
that shared bridge contract.

### `benchmarks/density_matrix/correctness_evidence/external_correctness_validation.py`

The current external validation bundle still assumes:

- `len(cases) == 4`

and therefore matches the old Phase 3 external slice, not the bounded Phase 3.1
external slice:

- 4 microcases,
- plus `phase2_xxz_hea_q4_continuity`,
- excluding `phase2_xxz_hea_q6_continuity`.

That makes this file a required migration surface for `P31-S10-E02`.

### Other validation slices still assume the old default package

The following modules all consume the current default positive records or
default case matrix:

- `correctness_matrix_validation.py`
- `sequential_correctness_validation.py`
- `output_integrity_validation.py`
- `runtime_classification_validation.py`
- `correctness_bundle_validation.py`
- `summary_consistency_validation.py`
- `validation_pipeline.py`

Some of them hard-code old expected counts, for example
`correctness_matrix_validation.py` currently expects:

- continuity cases = 4,
- microcases = 3,
- structured cases = 18,
- external slice cases = 4.

So `P31-S10-E02` must decide whether to mutate these defaults or to add a
bounded Phase 3.1 sibling pipeline. The preferred plan below is the sibling
pipeline.

## Dependencies And Assumptions

- The source-of-truth contract remains:
  - `TASK_3_MINI_SPEC.md`,
  - `ENGINEERING_TASK_P31_S10_E01_IMPLEMENTATION_PLAN.md`,
  - `../FOURTH_VERTICAL_SLICE_REMAINING_COUNTED_CORRECTNESS_STORIES_AND_ENGINEERING_TASKS.md`,
  - `../DETAILED_PLANNING_PHASE_3_1.md`,
  - `../ADRs_PHASE_3_1.md`
    (`P31-ADR-003`, `P31-ADR-009`, `P31-ADR-011`, `P31-ADR-013`,
    `P31-ADR-015`).
- `P31-S10-E01` is assumed complete or at least has explicit blocker outcomes for:
  - `phase31_microcase_2q_multi_noise_entangler_chain`,
  - `phase31_microcase_2q_dense_same_support_motif`,
  - `phase2_xxz_hea_q6_continuity`.
- Planning §10.8 staged migration policy applies here:
  - **Stage A:** implement Phase 3.1 sibling builders and pipelines without
    mutating current Phase 3 defaults,
  - **Stage B:** switch defaults only after the Phase 3.1 flow is validated.
- The preferred `P31-S10-E02` plan is therefore **Stage A only**.
- The bounded fourth-slice package intentionally excludes the structured
  performance carry-forward rows; those remain downstream of `P31-S11`.
- `RUNTIME_CORRECTNESS_BRIDGE_FIELD_NAMES` in `benchmarks/density_matrix/evidence_core.py`
  should remain stable in this task unless a deliberate cross-pipeline contract
  change is needed later. Phase 3.1-specific fields should be additive outside
  that existing bridge set.
- Unsupported-boundary evidence can remain on the current shared negative-record
  path in this task unless the Phase 3.1 sibling package needs a dedicated
  wrapper for clarity.

## Target Files And Responsibilities

### Primary selector surface: `benchmarks/density_matrix/correctness_evidence/case_selection.py`

This file should grow a **bounded Phase 3.1 sibling selector** for the fourth
slice rather than reusing the old default builder directly.

#### What this file should provide after `P31-S10-E02`

- one Phase 3.1 bounded case-context builder for:
  - the four counted microcases,
  - the two counted continuity anchors,
  - and **no** structured cases in this fourth-slice package,
- stable metadata for:
  - `claim_surface_id`,
  - `representation_primary`,
  - `fused_block_support_qbits` where applicable,
  - `contains_noise`,
  - `counted_phase31_case`,
  - `external_reference_required`,
- and a cache / helper name that makes it obvious this is a Stage-A bounded
  Phase 3.1 package, not yet the default Phase 3 replacement.

#### Recommended implementation constraint

Keep `build_correctness_evidence_case_contexts()` unchanged in this task. Add a
new Phase 3.1 sibling builder instead, for example:

- `build_phase31_correctness_evidence_case_contexts()`

or an equally explicit bounded name.

### Primary positive-record surface: `benchmarks/density_matrix/correctness_evidence/records.py`

This file should gain a Phase 3.1 sibling positive-record builder that dispatches
to the correct public runtime path per bounded case type:

- strict for the four microcases,
- hybrid for the two continuity anchors.

#### What this file should provide after `P31-S10-E02`

- a bounded Phase 3.1 positive-record builder that uses:
  - `execute_partitioned_density_channel_native(...)` on the counted microcases,
  - `execute_partitioned_density_channel_native_hybrid(...)` on the continuity
    anchors,
- additive required fields:
  - `runtime_class`,
  - `claim_surface_id`,
  - `representation_primary`,
  - `fused_block_support_qbits`,
  - `contains_noise`,
  - `counted_phase31_case`,
- a `channel_invariants` slice for the counted strict microcases,
- a `partition_route_summary` slice for the counted hybrid rows,
- and a Phase 3.1 sibling positive-record list builder.

#### Recommended implementation constraint

Do **not** overwrite the existing `build_correctness_evidence_positive_record()`
behavior in this task. Keep the old default path stable and add a sibling
builder, for example:

- `build_phase31_correctness_evidence_positive_record()`
- `build_phase31_correctness_evidence_positive_records()`

#### Important bridge-field constraint

`build_runtime_correctness_bridge_fields(...)` and
`RUNTIME_CORRECTNESS_BRIDGE_FIELD_NAMES` should stay stable in this task.
Phase-3.1-specific fields and slices should be appended **after** the shared
bridge fields are materialized, not stuffed into the bridge-field set.

### Shared schema/constants surface: `benchmarks/density_matrix/correctness_evidence/common.py`

This file should define successor schema identifiers or equivalent shared names
for the bounded Phase 3.1 sibling package.

#### What this file should provide after `P31-S10-E02`

- version-bumped schema strings for any case or package payload whose required
  fields differ from the current v2/v1 Phase 3 defaults,
- shared output-root helpers that allow Phase 3.1 sibling bundles to write into
  distinct artifact slice directories if needed,
- and no silent reuse of the old schema identifiers when required Phase 3.1
  fields are added.

### Shared bundle surface: `benchmarks/density_matrix/correctness_evidence/bundle.py`

This file should gain a Phase 3.1 sibling payload builder for the bounded
fourth-slice package, for example:

- `build_phase31_correctness_package_payload()`

That payload should combine:

- the Phase 3.1 sibling positive records,
- the existing negative boundary records unless a dedicated sibling wrapper is
  needed later,
- and summary counts appropriate to the bounded fourth-slice package.

### Preferred Stage-A sibling validation entrypoints

The preferred `P31-S10-E02` plan is to keep the current default validation
modules stable and add explicit Phase 3.1 sibling entrypoints, for example:

- `phase31_correctness_matrix_validation.py`
- `phase31_sequential_correctness_validation.py`
- `phase31_external_correctness_validation.py`
- `phase31_output_integrity_validation.py`
- `phase31_runtime_classification_validation.py`
- `phase31_correctness_bundle_validation.py`
- `phase31_summary_consistency_validation.py`
- `phase31_validation_pipeline.py`

#### Why sibling entrypoints are preferred here

- They match planning §10.8 Stage A.
- They avoid destabilizing the current Phase 3 default evidence flow.
- They make the Phase 3.1 bounded package reviewable before any default switch.

### Files that should remain stable if possible

- `benchmarks/density_matrix/correctness_evidence/validation_support.py`
  should remain the shared generic envelope layer.
- `benchmarks/density_matrix/correctness_evidence/unsupported_boundary_validation.py`
  can remain unchanged in this task unless the sibling pipeline needs a
  dedicated wrapper for clarity.
- Runtime files under `squander/partitioning/` should not change in this task
  unless a genuine blocker is exposed by package migration.

## Implementation Sequence

### Step 1: Freeze the Stage-A sibling migration boundary

**Goal**

Make it explicit that `P31-S10-E02` emits a bounded Phase 3.1 sibling package
without yet switching the repository-wide default correctness pipeline.

**Execution checklist**

- [ ] Keep the current default `build_correctness_evidence_case_contexts()`
      surface unchanged in this task.
- [ ] Keep the current default `build_correctness_evidence_positive_record()`
      surface unchanged in this task.
- [ ] Prefer new Phase 3.1 sibling builders / wrappers over in-place mutation of
      the old default entrypoints.
- [ ] Keep the bounded package scope explicit:
      - microcases = 4,
      - continuity anchors = 2,
      - structured cases = 0 in this slice.

**Why first**

Without a stable Stage-A boundary, the task risks becoming an uncontrolled
default-switch instead of a reviewable bounded package migration.

### Step 2: Add the bounded Phase 3.1 case-context builder

**Goal**

Materialize the fourth-slice bounded correctness case set as one reusable
builder surface.

**Execution checklist**

- [ ] Add a Phase 3.1 bounded case-context builder in `case_selection.py`.
- [ ] Include exactly:
      - four counted microcases,
      - `phase2_xxz_hea_q4_continuity`,
      - `phase2_xxz_hea_q6_continuity`.
- [ ] Exclude structured carry-forward rows from this bounded package.
- [ ] Preserve `external_reference_required` on:
      - the four microcases,
      - `phase2_xxz_hea_q4_continuity`,
      - and not `phase2_xxz_hea_q6_continuity`.
- [ ] Keep all frozen workload IDs unchanged.

**Recommended expected bounded counts**

- `total_cases == 6`
- `microcases == 4`
- `continuity_cases == 2`
- `structured_cases == 0`
- `external_slice_cases == 5`

### Step 3: Add the Phase 3.1 sibling positive-record builder with required fields and slices

**Goal**

Generate bounded Phase 3.1 correctness records through the explicit strict /
hybrid runtime interpretations and add the required Phase 3.1 record fields.

**Execution checklist**

- [ ] Add a Phase 3.1 sibling positive-record builder in `records.py`.
- [ ] Dispatch by case type:
      - microcases -> `execute_partitioned_density_channel_native(...)`
      - continuity anchors -> `execute_partitioned_density_channel_native_hybrid(...)`
- [ ] Reuse `build_runtime_correctness_bridge_fields(...)` for the shared
      correctness metrics.
- [ ] Add the required Phase 3.1 counted-case fields after the shared bridge
      fields are built.
- [ ] Emit `channel_invariants` for the strict microcases.
- [ ] Emit `partition_route_summary` for the hybrid rows.
- [ ] Keep `build_correctness_evidence_positive_record()` unchanged in this task.

**Recommended record-shape rule**

Treat Phase 3.1-specific fields as **additive sibling fields**. Do not silently
change the old bridge-field contract unless later performance-pipeline work
explicitly requires the same fields downstream.

### Step 4: Add sibling validation slices and bounded package/bundle builders

**Goal**

Make the bounded Phase 3.1 correctness package executable end-to-end through a
parallel validation stack.

**Execution checklist**

- [ ] Add a bounded Phase 3.1 correctness-matrix validation entrypoint with the
      expected counts:
      - microcases = 4
      - continuity cases = 2
      - structured cases = 0
      - external slice cases = 5
- [ ] Add a bounded Phase 3.1 external validation entrypoint expecting:
      - `len(cases) == 5`
      - `microcases == 4`
      - `continuity_cases == 1`
- [ ] Add sibling sequential, output-integrity, runtime-classification, package,
      summary-consistency, and pipeline entrypoints as needed to run the bounded
      Phase 3.1 package in one command.
- [ ] Add or expose a Phase 3.1 sibling package payload builder in `bundle.py`.
- [ ] Version-bump schema identifiers for the sibling package where required.

**Recommended implementation constraint**

Prefer Phase 3.1 sibling module names rather than parameterizing the old
default modules in-place. This keeps Phase 3 and Stage-A Phase 3.1 outputs
separable during review.

### Step 5: Validate the bounded Phase 3.1 package without disturbing the current default pipeline

**Goal**

Prove that the bounded sibling package is internally consistent and ready for
later handoff to the performance matrix without yet flipping defaults.

**Execution checklist**

- [ ] Run the new bounded Phase 3.1 sibling validation pipeline end-to-end.
- [ ] Confirm the external Phase 3.1 bundle reports 5 required cases.
- [ ] Confirm the bounded matrix bundle reports:
      - total cases = 6
      - microcases = 4
      - continuity cases = 2
      - structured cases = 0
      - external slice cases = 5
- [ ] Confirm the sibling package and summary-consistency bundles pass.
- [ ] Re-run the current default correctness pipeline only if a shared helper was
      touched unexpectedly.

**Why this matters**

`P31-S10-E02` is successful only if the new package can be reviewed on its own
terms without destabilizing the current default correctness flow.

## Acceptance Evidence

`P31-S10-E02` is ready to hand off when all of the following are true:

- a bounded Phase 3.1 sibling correctness package exists for:
  - the four counted microcases,
  - `phase2_xxz_hea_q4_continuity`,
  - `phase2_xxz_hea_q6_continuity`,
- the sibling correctness-matrix bundle reports:
  - `total_cases == 6`
  - `microcases == 4`
  - `continuity_cases == 2`
  - `structured_cases == 0`
  - `external_slice_cases == 5`
- the sibling external-correctness bundle reports:
  - `total_cases == 5`
  - `microcases == 4`
  - `continuity_cases == 1`
  - all required external rows pass
- sibling positive records carry the required Phase 3.1 fields:
  - `claim_surface_id`
  - `runtime_class`
  - `representation_primary`
  - `fused_block_support_qbits`
  - `contains_noise`
  - `counted_phase31_case`
- sibling positive records also carry:
  - `channel_invariants` on the counted strict microcases
  - `partition_route_summary` on the counted hybrid rows
- the current default Phase 3 correctness pipeline remains unchanged in behavior
  unless a deliberate later Stage-B switch is approved
- and the sibling package produces a review-ready Case ID -> internal pass ->
  Aer pass summary aligned to the frozen fourth-slice IDs

## Handoff To The Next Engineering Tasks

After `P31-S10-E02` lands:

- `P31-S11-E01` should build the full counted performance matrix on top of the
  same strict/hybrid execution interpretations.
- The bounded Phase 3.1 sibling correctness package should become the input
  evidence surface for the later pre-closure review.
- The structured performance carry-forward rows can then be added deliberately
  rather than as placeholders in the fourth slice.
- A later Stage-B migration decision can choose whether to:
  - switch the default correctness builders/pipeline to the Phase 3.1 surfaces,
  - or keep explicit Phase 3 and Phase 3.1 sibling entrypoints side by side.

## Risks / Rollback

- Risk: in-place mutation of the current default builders destabilizes Phase 3
  historical evidence.
  Rollback/mitigation: keep Stage-A work in explicit Phase 3.1 sibling builders
  and entrypoints.

- Risk: the external-validation bundle keeps the old 4-row assumption.
  Rollback/mitigation: freeze the bounded Phase 3.1 external slice explicitly as
  5 rows and assert the count in the sibling external bundle.

- Risk: additive Phase 3.1 fields accidentally leak into the shared
  bridge-field contract and break downstream performance reuse.
  Rollback/mitigation: keep `build_runtime_correctness_bridge_fields(...)`
  stable and append Phase 3.1 fields outside that bridge set.

- Risk: the bounded sibling package is mistaken for full Task 3 closure,
  including structured carry-forward rows.
  Rollback/mitigation: keep the bounded package scope explicit and defer
  structured rows to `P31-S11`.

- Risk: package migration exposes real mismatches between the counted pytest
  gates and the record builders.
  Rollback/mitigation: treat the pytest gates from `P31-S10-E01` as the runtime
  truth and adjust the record builders to match them, not vice versa.
