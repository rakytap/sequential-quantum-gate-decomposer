# Engineering Task P31-S10-E01 Implementation Plan

## Engineering Task Being Implemented

`Engineering Task P31-S10-E01: Promote the remaining counted strict microcases and q6 hybrid continuity row into claim-bearing correctness gates`

This is a Layer 4 file-level implementation plan for the first engineering task
under Story `P31-S10` from
`../FOURTH_VERTICAL_SLICE_REMAINING_COUNTED_CORRECTNESS_STORIES_AND_ENGINEERING_TASKS.md`.
It turns the fourth-slice Task 3 wording into a concrete plan against the
current strict/hybrid correctness and evidence surfaces in:

- `task-3/TASK_3_MINI_SPEC.md`,
- `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`,
- `tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py`,
- `tests/partitioning/fixtures/workloads.py`,
- `tests/partitioning/fixtures/continuity.py`,
- `tests/partitioning/fixtures/runtime.py`,
- `squander/partitioning/noisy_runtime_core.py`,
- `squander/partitioning/noisy_runtime.py`,
- and, for scope control only,
  `benchmarks/density_matrix/correctness_evidence/case_selection.py`,
  `benchmarks/density_matrix/correctness_evidence/records.py`, and
  `benchmarks/density_matrix/correctness_evidence/external_correctness_validation.py`.

## Scope

This engineering task promotes the remaining counted correctness rows into
**deterministic claim-bearing pytest evidence** before the broader correctness
package migration:

- keep the remaining counted strict microcase IDs unchanged:
  - `phase31_microcase_2q_multi_noise_entangler_chain`,
  - `phase31_microcase_2q_dense_same_support_motif`,
- keep the remaining counted hybrid continuity ID unchanged:
  - `phase2_xxz_hea_q6_continuity`,
- reuse the existing public strict helper
  `execute_partitioned_density_channel_native(...)` for the remaining strict
  microcases,
- reuse the existing public hybrid helper
  `execute_partitioned_density_channel_native_hybrid(...)` for the counted `q6`
  continuity anchor,
- assert the frozen internal exactness family from `P31-C-03` on the final
  density state:
  - Frobenius difference,
  - maximum absolute difference,
  - `trace_deviation`,
  - `rho_is_valid`,
- add reviewer-facing route-summary assertions for the counted `q6` hybrid
  anchor using the already-landed additive per-partition route metadata,
- preserve the completed counted `q4` hybrid anchor and earlier strict slice as
  regression anchors,
- and record any correctness blocker explicitly rather than weakening the frozen
  counted slice.

Out of scope for this engineering task:

- Aer-row wiring and the versioned correctness package migration under
  `P31-S10-E02`,
- changes to active correctness-evidence builders or schema versions,
- publication/doc updates beyond local test/module intent,
- Task 4 performance matrix work,
- new runtime labels, new route reasons, or new public runtime result fields,
- and any broadening of the frozen support matrix or counted case set.

## Current Runtime And Evidence Gap To Close

The fourth slice begins from a stronger state than earlier Task 3 work:

- the strict path is already public and exact on the counted 1q and counted 2q
  anchor cases,
- the hybrid path is already public and exact on the counted `q4` continuity
  anchor,
- and the first structured pilot already exists.

What is still missing is the **rest of the counted correctness surface** named
by `P31-ADR-009` and planning §10.2.

### `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`

The current second-slice module already contains useful substrate for this task:

- public strict end-to-end exactness on
  `phase31_microcase_2q_cnot_local_noise_pair`,
- helper-level `CNOT` orientation / lowering evidence on
  `phase31_microcase_2q_multi_noise_entangler_chain`,
- strict local-support smoke coverage,
- and explicit public boundary checks.

However, it still stops short of `P31-S10` claim-bearing correctness closure:

- `phase31_microcase_2q_multi_noise_entangler_chain` appears only as a
  helper-level lowering/composition surface, not as a public end-to-end strict
  correctness gate,
- `phase31_microcase_2q_dense_same_support_motif` currently exists only as a
  fixture definition and is not yet covered by any public runtime correctness
  test,
- the module docstring still lists both of these strict counted IDs as deferred,
- and the strict counted surface after `P31-S06-E01` still reads as “minimal
  gate complete” rather than “remaining counted strict rows closed.”

That makes this file the natural primary target for the **remaining strict**
correctness promotion.

### `tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py`

The current hybrid module already covers:

- the counted `q4` continuity anchor with frozen aggregated route summary,
- the unsupported-by-both negative case,
- and a non-counted structured hybrid smoke.

However, the module still explicitly names `phase2_xxz_hea_q6_continuity` as
deferred, and there is no counted `q6` hybrid gate yet. That makes this file
the natural primary target for the remaining **hybrid** continuity closure in
`P31-S10-E01`.

### The broader correctness-evidence package is still on the old Phase 3 default path

The wider correctness-evidence harness already knows about the **planned**
Phase 3.1 counted rows, but it is not yet the right claim-bearing surface for
this task:

- `benchmarks/density_matrix/correctness_evidence/case_selection.py` already has
  `iter_phase31_correctness_microcase_cases()` and
  `iter_phase31_correctness_continuity_cases()`,
- but `build_correctness_evidence_case_contexts()` still returns the old Phase 3
  continuity / microcase / structured matrix,
- `benchmarks/density_matrix/correctness_evidence/records.py` still builds
  positive records through `execute_fused_with_reference(...)` rather than the
  explicit Phase 3.1 strict/hybrid paths,
- and `external_correctness_validation.py` still assumes the old 4-row external
  bundle shape rather than the bounded 5-row Phase 3.1 external slice.

So `P31-S10-E01` should stay in **deterministic pytest evidence** and leave all
package migration to `P31-S10-E02`.

## Dependencies And Assumptions

- The source-of-truth contract remains:
  - `TASK_3_MINI_SPEC.md`,
  - `../FOURTH_VERTICAL_SLICE_REMAINING_COUNTED_CORRECTNESS_STORIES_AND_ENGINEERING_TASKS.md`,
  - `../DETAILED_PLANNING_PHASE_3_1.md`,
  - `../ADRs_PHASE_3_1.md`
    (`P31-ADR-003`, `P31-ADR-008`, `P31-ADR-009`, `P31-ADR-011`,
    `P31-ADR-013`).
- `P31-S06-E01` is assumed complete:
  - the strict counted `phase31_microcase_2q_cnot_local_noise_pair` gate is the
    current public exactness template,
  - the helper substrate for bundle construction and invariant checks is already
    present in the second-slice module.
- `P31-S08-E01` is assumed complete:
  - the hybrid counted `phase2_xxz_hea_q4_continuity` gate is the current public
    whole-workload exactness template,
  - `_hybrid_partition_route_summary(...)` already exists and should be reused.
- The remaining counted strict microcase IDs already exist in
  `tests/partitioning/fixtures/workloads.py` and should be reused unchanged.
- `build_phase2_continuity_vqe(6)` plus
  `build_phase3_continuity_partition_descriptor_set(...)` already provide the
  deterministic `q6` continuity input surface.
- Route-summary stability for `q6` should prefer **aggregated counts** over
  partition-index assertions, just as `q4` currently does.
- If the first observed `q6` aggregated summary differs from expectation, freeze
  the summary deliberately in the same change rather than weakening the test
  into existence-only assertions.
- No new runtime API should be introduced just to close this task.

## Target Files And Responsibilities

### Primary strict evidence file: `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`

This file should become the public strict evidence home for the two remaining
counted 2-qubit microcases.

#### What this file should cover after `P31-S10-E01`

- a clearly labeled correctness subsection for the remaining strict counted
  microcases,
- end-to-end exactness for
  `phase31_microcase_2q_multi_noise_entangler_chain`,
- first public end-to-end exactness for
  `phase31_microcase_2q_dense_same_support_motif`,
- the frozen internal exactness family:
  - Frobenius difference,
  - maximum absolute difference,
  - `trace_deviation`,
  - `rho_is_valid`,
- at least one representation-invariant acceptance check on each newly promoted
  strict microcase using the already-imported helper substrate,
- and explicit wording that the remaining deferred work is now hybrid `q6`,
  external-reference closure, and package migration.

#### Recommended implementation constraint

Prefer to extend the existing second-slice module rather than creating a third
strict-slice test file. The helper substrate and public exactness idiom already
live there.

### Primary hybrid evidence file: `tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py`

This file should become the public hybrid evidence home for the remaining
counted continuity anchor.

#### What this file should cover after `P31-S10-E01`

- keep the counted `q4` continuity gate unchanged as the first hybrid anchor,
- add the counted `q6` continuity gate with the same public exactness family,
- reuse the existing `_hybrid_partition_route_summary(...)` helper,
- freeze one aggregated `q6` route summary in terms of:
  - `partition_count`,
  - `runtime_class_counts`,
  - `route_reason_counts`,
- keep the unsupported-by-both negative case unchanged,
- and keep the structured `q8` hybrid smoke explicitly non-counted.

#### Recommended implementation constraint

Do **not** add a new production route-summary builder just for `q6`. The
existing test-local summary approach used by `q4` is sufficient.

### Stable fixture files: `tests/partitioning/fixtures/workloads.py`, `tests/partitioning/fixtures/continuity.py`, `tests/partitioning/fixtures/runtime.py`

These files should remain unchanged unless a genuine fixture defect is exposed
while writing the new correctness gates.

#### Reuse as-is if possible

- `build_phase31_microcase_descriptor_set(...)`
- `build_phase2_continuity_vqe(6)`
- `PHASE3_RUNTIME_DENSITY_TOL`
- `build_density_comparison_metrics(...)`
- `build_initial_parameters(...)`

### Stable runtime files: `squander/partitioning/noisy_runtime_core.py` and `squander/partitioning/noisy_runtime.py`

The preferred `P31-S10-E01` plan is **no public API change** in these files.

#### Keep stable

- `execute_partitioned_density_channel_native(...)`
- `execute_partitioned_density_channel_native_hybrid(...)`
- `PHASE31_RUNTIME_PATH_CHANNEL_NATIVE`
- `PHASE31_RUNTIME_PATH_CHANNEL_NATIVE_HYBRID`
- `NoisyRuntimePartitionRecord.partition_runtime_class`
- `NoisyRuntimePartitionRecord.partition_route_reason`

#### Avoid in this task

- adding new runtime result fields,
- adding new route reasons,
- changing strict or hybrid naming,
- or reopening downgrade / fallback semantics.

### Explicitly out-of-scope bundle files

These files matter for `P31-S10-E02`, but the preferred `P31-S10-E01` plan is
**not** to edit them:

- `benchmarks/density_matrix/correctness_evidence/case_selection.py`
- `benchmarks/density_matrix/correctness_evidence/records.py`
- `benchmarks/density_matrix/correctness_evidence/external_correctness_validation.py`
- related bundle / validation scripts under
  `benchmarks/density_matrix/correctness_evidence/`

That wider package migration belongs to `P31-S10-E02` after the remaining
counted correctness gates are frozen in pytest.

## Implementation Sequence

### Step 1: Recast the current strict and hybrid test modules as the remaining counted correctness gate surface

**Goal**

Turn the current strict and hybrid modules from “completed partial Phase 3.1
proofs with deferred rows” into “completed partial proofs plus the remaining
counted correctness closure surface.”

**Execution checklist**

- [ ] Update the module docstring in
      `test_partitioned_channel_native_phase31_second_slice.py`.
- [ ] Update the module docstring in
      `test_partitioned_channel_native_phase31_hybrid_slice.py`.
- [ ] Add clearly labeled correctness subsections for the remaining counted
      strict microcases and the counted `q6` hybrid anchor.
- [ ] Name the still-deferred work explicitly:
      - Aer on the frozen external slice,
      - versioned correctness package migration,
      - full Task 4 matrix work.

**Why first**

This task is primarily about **evidence shape and claim boundary**. The modules
should describe the new fourth-slice boundary honestly before stronger
assertions are added.

### Step 2: Promote the remaining strict counted microcases to explicit end-to-end exactness evidence

**Goal**

Make the two remaining strict 2-qubit microcases part of the public counted
exactness surface rather than helper-only or fixture-only objects.

**Execution checklist**

- [ ] Keep the counted case ID strings unchanged:
      - `phase31_microcase_2q_multi_noise_entangler_chain`,
      - `phase31_microcase_2q_dense_same_support_motif`.
- [ ] Reuse `execute_partitioned_density_channel_native(...)` vs
      `execute_sequential_density_reference(...)`.
- [ ] Assert the frozen full-density exactness family for both cases:
      - Frobenius difference `<= PHASE3_RUNTIME_DENSITY_TOL`,
      - maximum absolute difference `<= PHASE3_RUNTIME_DENSITY_TOL`,
      - `result.trace_deviation <= PHASE3_RUNTIME_DENSITY_TOL`,
      - `result.rho_is_valid is True`.
- [ ] Preserve public runtime-path identity:
      - `result.runtime_path == "phase31_channel_native"`,
      - `result.requested_runtime_path == "phase31_channel_native"`.
- [ ] Add at least one representation-invariant acceptance check for each newly
      promoted strict microcase using the existing helper substrate.
- [ ] Keep the already-landed counted `phase31_microcase_2q_cnot_local_noise_pair`
      gate unchanged as the earlier strict anchor.

**Recommended evidence strategy**

Prefer tiny test-local helpers that reuse the already imported internal
functions in the second-slice module:

- `_build_partition_parameter_vector(...)`
- `_segment_parameter_vector(...)`
- `_identity_kraus_bundle_for_support_qubit_count(...)`
- `_member_to_kraus_bundle(...)`
- `_compose_kraus_bundles(...)`
- `_check_kraus_bundle_invariants(...)`

That keeps invariant evidence inside deterministic pytest rather than adding a
new public runtime field just for this task.

### Step 3: Promote `phase2_xxz_hea_q6_continuity` to explicit end-to-end hybrid correctness evidence

**Goal**

Make the counted `q6` continuity anchor the second and final counted
whole-workload correctness gate on the frozen Phase 3.1 continuity slice.

**Execution checklist**

- [ ] Keep the counted case ID string unchanged:
      `phase2_xxz_hea_q6_continuity`.
- [ ] Reuse
      `execute_partitioned_density_channel_native_hybrid(...)`
      vs `execute_sequential_density_reference(...)`.
- [ ] Assert the frozen full-density exactness family:
      - Frobenius difference `<= PHASE3_RUNTIME_DENSITY_TOL`,
      - maximum absolute difference `<= PHASE3_RUNTIME_DENSITY_TOL`,
      - `result.trace_deviation <= PHASE3_RUNTIME_DENSITY_TOL`,
      - `result.rho_is_valid is True`.
- [ ] Preserve runtime-path identity:
      - `result.runtime_path == "phase31_channel_native_hybrid"`,
      - `result.requested_runtime_path == "phase31_channel_native_hybrid"`.
- [ ] Reuse `_hybrid_partition_route_summary(...)`.
- [ ] Freeze one aggregated `q6` route summary in terms of:
      - `partition_count`,
      - `runtime_class_counts`,
      - `route_reason_counts`.
- [ ] Keep route-summary assertions aggregated rather than partition-index based.

**Recommended implementation constraint**

Do **not** guess the `q6` summary in advance in production code. Observe the
stable deterministic summary once, then freeze it deliberately in the test
constants in the same change.

### Step 4: Keep correctness-package migration and Aer-bundle work explicitly deferred to `P31-S10-E02`

**Goal**

Land the remaining counted correctness gates without pretending the broader
correctness-evidence package has already migrated to the Phase 3.1 slice.

**Execution checklist**

- [ ] Do not wire the Phase 3.1 iterators into
      `build_correctness_evidence_case_contexts()` in this task.
- [ ] Do not change `build_correctness_evidence_positive_record(...)` to call the
      explicit strict / hybrid Phase 3.1 paths in this task.
- [ ] Do not change `external_correctness_validation.py` in this task.
- [ ] Do not add schema version bumps, `channel_invariants`, or
      `partition_route_summary` bundle fields in this task.
- [ ] Keep the deferred package-migration note visible in the updated test
      modules and in this implementation plan.

**Reasoning**

`P31-S10-E01` closes the remaining **runtime-level correctness gates**. The
artifact migration belongs to `P31-S10-E02`.

### Step 5: Validate the remaining correctness gates and preserve slice continuity

**Goal**

Prove that the tightened strict and hybrid correctness evidence passes while the
already-landed earlier slice anchors remain intact.

**Execution checklist**

- [ ] Run:
      `conda run -n qgd python -m pytest tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py -q`
- [ ] Re-run:
      `conda run -n qgd python -m pytest tests/partitioning/test_partitioned_channel_native_phase31_slice.py tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py -q`
- [ ] If any shared continuity fixture or runtime helper changed unexpectedly,
      run:
      `conda run -n qgd python -m pytest tests/partitioning/evidence/test_phase31_hybrid_pilot_validation.py -q`
- [ ] Leave the broader correctness-evidence bundle scripts untouched unless a
      genuine blocker is exposed.

**Why this matters**

`P31-S10-E01` should be a narrow counted-evidence promotion task, not a
regression on the already-completed strict or hybrid surfaces.

## Acceptance Evidence

`P31-S10-E01` is ready to hand off when all of the following are true:

- `phase31_microcase_2q_multi_noise_entangler_chain` passes end-to-end through
  the strict path against the sequential oracle with:
  - Frobenius difference `<= 1e-10`
  - maximum absolute difference `<= 1e-10`
  - `trace_deviation <= 1e-10`
  - `rho_is_valid is True`
  - and at least one passing representation-invariant check
- `phase31_microcase_2q_dense_same_support_motif` passes the same strict
  end-to-end exactness family with at least one passing representation-invariant
  check
- `phase2_xxz_hea_q6_continuity` passes end-to-end through the hybrid path
  against the sequential oracle with:
  - Frobenius difference `<= 1e-10`
  - maximum absolute difference `<= 1e-10`
  - `trace_deviation <= 1e-10`
  - `rho_is_valid is True`
- the counted `q6` hybrid route summary is frozen and reviewable through
  aggregated counts
- the existing counted `q4` hybrid anchor still passes unchanged
- the existing strict first-slice and second-slice anchors still pass unchanged
- and no changes are required in the active correctness-evidence bundle builders
  just to close this task

## Handoff To The Next Engineering Tasks

After `P31-S10-E01` lands:

- `P31-S10-E02` should wire the bounded Aer subset and the versioned
  correctness package migration.
- The external bundle should reuse the same frozen IDs already established here
  rather than inventing a parallel registry.
- The package migration should carry forward:
  - the remaining strict microcase IDs,
  - the counted `q6` continuity ID,
  - the frozen hybrid route-summary vocabulary,
  - and the required `P31-ADR-013` fields.
- `P31-S11-E01` should remain downstream of this task and should not start
  claiming full performance interpretation until the counted correctness surface
  is green or explicitly blocked.

## Risks / Rollback

- Risk: the counted `q6` route summary may over-lock harmless partitioning
  details.
  Rollback/mitigation: assert aggregated counts by runtime class and route
  reason, not raw partition indices.

- Risk: the dense same-support motif may expose a real support-surface or
  invariant failure that the current partial slice never exercised.
  Rollback/mitigation: keep the failure explicit and do not narrow the frozen
  counted slice to preserve progress optics.

- Risk: touching active correctness-evidence builders in this task would blur
  the line between runtime-level correctness-gate closure and package migration.
  Rollback/mitigation: keep this task in deterministic pytest evidence unless a
  genuine blocker is exposed.

- Risk: broad runtime changes could accidentally reopen already-closed strict or
  hybrid behavior.
  Rollback/mitigation: prefer test-surface promotion first; if runtime fixes are
  required, keep them minimal and rerun the earlier slice anchors immediately.
