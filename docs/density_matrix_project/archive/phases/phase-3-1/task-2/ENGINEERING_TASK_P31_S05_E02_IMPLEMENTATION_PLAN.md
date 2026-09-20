# Engineering Task P31-S05-E02 Implementation Plan

## Engineering Task Being Implemented

`Engineering Task P31-S05-E02: Add the local-support negative matrix and audit assertions for the second slice`

This is a Layer 4 file-level implementation plan for the second engineering task
under Story `P31-S05` from
`../SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`.
It turns the second-slice Task 2 wording into a concrete evidence plan against
the public local-support runtime exposed by `P31-S05-E01` in:

- `tests/partitioning/test_partitioned_channel_native_phase31_slice.py`,
- `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`,
- `tests/partitioning/fixtures/workloads.py`,
- and, only if the tests reveal a real contract bug,
  `squander/partitioning/noisy_runtime_channel_native.py` or
  `squander/partitioning/noisy_runtime_core.py`.

## Scope

This engineering task tightens the **public** second-slice evidence after the
runtime is reopened:

- add a deterministic negative matrix around the frozen local-support boundary,
- assert the public fused-region audit fields rather than only inspecting final
  density equality,
- keep the assertions focused on the documented taxonomy and public audit
  surface,
- and preserve the completed 1-qubit regression gate plus the positive local-
  support runtime path from `P31-S05-E01`.

Out of scope for this engineering task:

- widening the supported motif surface beyond the frozen `1`- or `2`-qubit same-
  support slice,
- new runtime result fields unless a missing public contract defect is exposed,
- correctness-bundle emission (`P31-S06-E01` / Task 3),
- Aer rows, performance packaging, and publication packaging,
- revisiting the Task 1 representation or Task 2 positive-path design unless the
  new tests expose a concrete bug.

## Current Runtime Gap To Close

After `P31-S05-E01`, the public path can be correct without yet being fully
auditable or boundary-pinned.

### Public test surface

The current public evidence stack is still incomplete for Story `P31-S05`:

- `tests/partitioning/test_partitioned_channel_native_phase31_slice.py` covers
  only the completed first slice,
- `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`
  currently emphasizes helper-level E01/E02 substrate checks,
- Story `P31-S05` still requires deterministic **public** failures for:
  - support `> 2` qubits,
  - pure-unitary motifs,
  - unsupported operation or off-support patterns,
- and positive public tests must assert fused-region audit data, not merely
  infer correct support from final-state equality.

### Runtime audit surface

The good news is that most of the audit plumbing already exists:

- `NoisyRuntimeFusedRegionRecord` already exposes:
  - `partition_member_indices`,
  - `operation_names`,
  - `global_target_qbits`,
  - `local_target_qbits`,
- `phase31_channel_native` is already the frozen runtime-path label under
  `P31-ADR-012`,
- and the public result already returns `fused_regions`.

The missing piece is not a new schema. It is **tests that assert the existing
schema**.

### Fixture inventory

This task may need one or more deterministic negative cases beyond the existing
microcases.

Preferred rule:

- add named fixtures only when the same case is likely to be reused by later
  correctness or evidence work,
- otherwise, construct the smallest negative workloads directly in the test
  module to avoid overgrowing the frozen fixture inventory with one-off failure
  shapes.

## Dependencies And Assumptions

- The source-of-truth contract remains:
  - `task-2/TASK_2_MINI_SPEC.md`,
  - `../SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`,
  - `../DETAILED_PLANNING_PHASE_3_1.md`,
  - `../ADRs_PHASE_3_1.md`
    (`P31-ADR-007`, `P31-ADR-009`, `P31-ADR-012`).
- `P31-S05-E01` is assumed to have reopened the public local-support runtime and
  frozen the larger-workload smoke workload (or an explicitly documented
  deterministic substitute).
- The public error taxonomy remains the existing `NoisyRuntimeValidationError`
  family with `first_unsupported_condition` vocabulary frozen by `P31-ADR-012`.
- The public fused-region audit contract is the current
  `NoisyRuntimeFusedRegionRecord`; this task should not invent a second audit
  payload.

## Target Files And Responsibilities

### Primary evidence file: `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`

This should become the main second-slice public evidence module once local-
support admission is exposed.

#### Expand this module to cover

- deterministic public negative cases for the frozen local-support boundary,
- positive runtime-path and fused-region audit assertions on the bounded
  supported slice,
- and continued separation between:
  - helper-level E01/E02 substrate checks,
  - public runtime integration checks.

#### Recommended structure

- keep E01/E02 helper-level tests grouped first or in a clearly labeled section,
- add a second public-runtime section for:
  - positive public cases,
  - negative matrix cases,
  - fused-region audit assertions.

### Regression anchor: `tests/partitioning/test_partitioned_channel_native_phase31_slice.py`

This file should remain unchanged as the first-slice public regression gate.

#### Keep unchanged as the regression bar

- `test_phase31_channel_native_1q_microcase_matches_sequential_reference()`
- `test_phase31_channel_native_parametric_noise_clamped_matches_sequential()`
- `test_phase31_channel_native_rejects_pure_unitary_motif()`

### Optional fixture file: `tests/partitioning/fixtures/workloads.py`

Use this file only when a negative or smoke workload deserves a stable ID beyond
one test module.

#### Preferred fixture policy

- Reuse the frozen positive IDs from `P31-ADR-009` where possible.
- Keep negative one-off cases inline if they exist only to pin a single
  `first_unsupported_condition`.
- Add a named negative fixture only if the same unsupported boundary is expected
  to appear in later correctness bundles or audit tooling.

### Production files: `squander/partitioning/noisy_runtime_channel_native.py` and `squander/partitioning/noisy_runtime_core.py`

The preferred plan is **no functional change** in these files.

#### Allowed adjustments

- A minimal production fix is allowed only if one of the new tests exposes:
  - the wrong `first_unsupported_condition`,
  - missing or incorrect fused-region metadata,
  - or an accidental silent-fallback path.
- Do not widen the supported slice or add new public result fields just to make
  the tests easier to write.

## Implementation Sequence

### Step 1: Freeze the public negative matrix around the local-support boundary

**Goal**

Make the boundary behavior of the public second-slice runtime deterministic and
reviewable.

**Execution checklist**

- [ ] Add a `> 2` local-support negative case and assert
      `channel_native_qubit_span`.
- [ ] Add a pure-unitary 2-qubit negative case and assert
      `channel_native_noise_presence`.
- [ ] Add one unsupported operation or unsupported off-support-pattern negative
      case and assert `channel_native_support_surface`.
- [ ] Assert `first_unsupported_condition`, not brittle full-message wording.

**Recommended case shapes**

1. **Support-width negative**

   - Build or reuse a small motif whose fused local support would be `3` qubits,
   - request channel-native mode publicly,
   - assert `channel_native_qubit_span`.

2. **Pure-unitary negative**

   - Build or reuse a 2-qubit same-support motif with only `U3` / `CNOT`,
   - request channel-native mode publicly,
   - assert `channel_native_noise_presence`.

3. **Unsupported-surface negative**

   - Use the smallest deterministic unsupported operation or unsupported support
     pattern that reaches the documented public gate,
   - request channel-native mode publicly,
   - assert `channel_native_support_surface`.

### Step 2: Make fused-region audit assertions part of the positive acceptance evidence

**Goal**

Prove not only that the public runtime got the right answer, but also **which
support it fused**.

**Execution checklist**

- [ ] Assert `result.runtime_path == "phase31_channel_native"` on positive public
      cases.
- [ ] Assert at least one fused region with
      `candidate_kind == PHASE31_FUSION_KIND_CHANNEL_NATIVE_MOTIF`.
- [ ] Assert fused-region `global_target_qbits` and `local_target_qbits` on the
      positive 2-qubit microcase and larger-workload smoke case.
- [ ] Assert `partition_member_indices` and `operation_names` on the same fused
      region so the applied motif is reviewable.

**Recommended audit granularity**

- Assert the fields that are already public and stable.
- Avoid pinning incidental text such as `reason=` strings unless a later doc
  explicitly freezes them.

### Step 3: Keep the negative and audit coverage narrow, stable, and reusable

**Goal**

Avoid turning the second-slice public tests into a brittle lock on incidental
implementation detail.

**Execution checklist**

- [ ] Prefer deterministic small workloads where the fused support is
      unambiguous.
- [ ] Keep taxonomy assertions at the level of
      `first_unsupported_condition`, not full exception prose.
- [ ] Keep audit assertions at the level of the existing public fields, not
      internal helper names or temporary record reasons.
- [ ] If a negative workload is only useful once, build it inline rather than
      promoting it immediately into the shared fixture inventory.

### Step 4: Preserve the existing slice stack and prepare the correctness handoff

**Goal**

Land the public audit/boundary evidence without blurring adjacent task
responsibilities.

**Execution checklist**

- [ ] Re-run the first-slice public regression file unchanged.
- [ ] Keep the helper-level E01/E02 tests passing alongside the new public
      second-slice assertions.
- [ ] Limit any production changes to contract bugs revealed by the new tests.
- [ ] Leave counted correctness-package promotion to `P31-S06-E01`.

## Acceptance Evidence

`P31-S05-E02` is ready to hand off when all of the following are true:

- at least three deterministic public negative tests cover:
  - support `> 2` qubits -> `channel_native_qubit_span`,
  - pure unitary motif -> `channel_native_noise_presence`,
  - unsupported operation or off-support pattern ->
    `channel_native_support_surface`,
- positive public second-slice tests assert:
  - runtime path `phase31_channel_native`,
  - fused-region kind `channel_native_motif`,
  - fused-region `global_target_qbits`,
  - fused-region `local_target_qbits`,
  - fused-region `partition_member_indices`,
  - fused-region `operation_names`,
- the first-slice 1-qubit public regression file still passes unchanged,
- and the implementation does not add a second audit schema or a new runtime
  label just to satisfy the tests.

## Handoff To The Next Engineering Tasks

After `P31-S05-E02` lands:

- `P31-S06-E01` should promote the counted 2-qubit microcase plus the larger-
  workload smoke case into the second-slice correctness gate,
- Task 3 correctness-evidence work should reuse these frozen taxonomy and audit
  assertions when building machine-checkable Phase 3.1 bundles,
- later API-reference work can document the already-frozen runtime-path and
  `first_unsupported_condition` vocabulary without inventing new names.

## Risks / Rollback

- Risk: negative tests can over-freeze incidental implementation detail rather
  than the documented contract.
  Rollback/mitigation: assert only the frozen taxonomy and existing public audit
  fields, not full exception prose or helper-local ordering.

- Risk: a nominally “unsupported surface” test may actually fail earlier through
  a different legitimate guard.
  Rollback/mitigation: choose the smallest unambiguous case for each taxonomy
  row and validate the intended guard before broadening the matrix.

- Risk: positive audit assertions may reveal a real metadata propagation bug.
  Rollback/mitigation: fix the propagation with the smallest possible production
  patch rather than widening the public result schema.
