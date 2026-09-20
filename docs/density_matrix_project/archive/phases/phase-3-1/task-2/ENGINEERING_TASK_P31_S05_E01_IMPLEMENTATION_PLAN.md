# Engineering Task P31-S05-E01 Implementation Plan

## Engineering Task Being Implemented

`Engineering Task P31-S05-E01: Replace whole-workload gating with local-support eligibility and mapped application`

This is a Layer 4 file-level implementation plan for the first engineering task
under Story `P31-S05` from
`../SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`.
It turns the second-slice Task 2 wording into a concrete plan against the
current runtime integration layer in:

- `squander/partitioning/noisy_runtime_channel_native.py`,
- `squander/partitioning/noisy_runtime_core.py`,
- `tests/partitioning/test_partitioned_channel_native_phase31_slice.py`,
- `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`,
- and, if the larger-workload smoke remains frozen as written,
  `tests/partitioning/fixtures/workloads.py`.

## Scope

This engineering task exposes the already-landed bounded 1-qubit / 2-qubit
channel-native substrate through the **public** Phase 3.1 runtime path:

- remove the remaining whole-workload-width gate from channel-native motif
  validation and replace it with local-support eligibility,
- reuse the existing local-support to global-target mapping when applying the
  fused channel inside a larger global density state,
- keep the distinct runtime/API identity
  `execute_partitioned_density_channel_native(...)` /
  `phase31_channel_native` stable,
- add positive public integration tests for the minimal counted 2-qubit case and
  one larger-workload local-support smoke case,
- and preserve loud failures for unsupported channel-native requests.

Out of scope for this engineering task:

- new primitive gate or noise families,
- support beyond `2` qubits, correlated noise, or spectator participation inside
  the fused block itself,
- the broader negative local-support matrix and stricter fused-region audit
  assertions (`P31-S05-E02`),
- counted correctness-bundle emission (`P31-S06-E01` / Task 3),
- Aer rows, performance packaging, and publication-surface work.

## Current Runtime Gap To Close

Task 1 is now strong enough to execute bounded local-support motifs, but the
public runtime still does not **admit** them.

### `squander/partitioning/noisy_runtime_channel_native.py`

The main `P31-S05-E01` gap is concentrated in public eligibility:

- `_validate_whole_partition_motif()` already collects `local_support`, enforces
  allowed operation names, and requires at least one noise operation.
- The same helper still hard-fails unless `descriptor_set.qbit_num == 1` and
  `local_support == (0,)`, which keeps every counted 2-qubit case and every
  larger-workload local-support case out of the public runtime.
- `execute_partition_channel_native()` already threads `local_support` through
  member lowering, bundle composition, fused-region recording, and
  `_apply_kraus_bundle(...)`.
- `_apply_kraus_bundle()` already knows how to apply valid 1-qubit and 2-qubit
  bundles to arbitrary global targets, so the mapped-application substrate is
  present; it is simply hidden behind the old gate.

### `squander/partitioning/noisy_runtime_core.py`

The public runtime shell is already in the right shape:

- `execute_partitioned_density()` already routes channel-native requests through
  `execute_partition_channel_native(...)`,
- `execute_partitioned_density_channel_native()` already provides the distinct
  Phase 3.1 runtime surface frozen by `P31-ADR-012`,
- and the current branch already avoids silent fallback once
  `phase31_channel_native` is requested.

This task should therefore prefer **minimal or no change** in this file.

### Existing tests and fixture inventory

The current tests are split cleanly:

- `tests/partitioning/test_partitioned_channel_native_phase31_slice.py` is still
  the completed 1-qubit public-runtime regression gate,
- `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`
  currently proves the E01/E02 substrate and helper-level 2-qubit lowering, not
  public runtime admission.

There is one planning/runtime gap worth calling out explicitly:

- the story names the larger-workload smoke ID
  `phase31_local_support_q4_spectator_embedding_smoke`,
- but that case is **not** currently present in
  `tests/partitioning/fixtures/workloads.py`.

`P31-S05-E01` should either:

- add that exact frozen case ID to the fixture inventory,
- or deliberately ratify an equivalent deterministic larger-workload smoke case
  before implementation starts.

## Dependencies And Assumptions

- The source-of-truth contract remains:
  - `task-2/TASK_2_MINI_SPEC.md`,
  - `../SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`,
  - `../DETAILED_PLANNING_PHASE_3_1.md`,
  - `../ADRs_PHASE_3_1.md`
    (`P31-ADR-007`, `P31-ADR-009`, `P31-ADR-012`, plus the continuing
    sequential-reference and conservative-fusion baselines under
    `P31-ADR-003` / `P31-ADR-005`).
- Task 1 closure is assumed:
  - size-aware bundle representation,
  - bounded 1q / 2q invariant checks,
  - support-aware apply semantics,
  - `CNOT` lowering and ordered 2-qubit same-support helper coverage.
- `phase31_microcase_2q_cnot_local_noise_pair` already exists and should be the
  smallest public positive runtime case for this task.
- `NoisyRuntimeFusedRegionRecord` already exposes:
  - `partition_member_indices`,
  - `operation_names`,
  - `global_target_qbits`,
  - `local_target_qbits`,
  so this task should reuse the existing audit surface rather than add a second
  result schema.
- Pure unitary islands remain part of the shipped Phase 3 fused path, not part
  of the counted Phase 3.1 channel-native surface.

## Target Files And Responsibilities

### Primary file: `squander/partitioning/noisy_runtime_channel_native.py`

This file should carry almost all substantive code changes for `P31-S05-E01`.

#### Public eligibility logic to change

- Refactor `_validate_whole_partition_motif()` so positive admission is based on
  the **motif's local support**, not on the whole workload width.
- Preserve:
  - allowed operation-name filtering,
  - required noise presence,
  - support width cap `<= 2`,
  - explicit hard failures under the existing taxonomy,
  - and ascending `local_support` order as the canonical mapping rule.
- Remove the remaining whole-workload assumptions:
  - `descriptor_set.qbit_num == 1`,
  - `local_support == (0,)`.

#### Mapped application path to preserve

- Reuse the `local_support` returned by validation as the exact support threaded
  into:
  - `_member_to_kraus_bundle(...)`,
  - `_apply_kraus_bundle(...)`,
  - `_global_targets_for_local_support(...)`,
  - and `_build_channel_native_region_record(...)`.
- Keep `execute_partition_channel_native()` as a **whole-partition** executor for
  partitions that are themselves valid same-support motifs.
- Do **not** broaden this task into searching for a smaller eligible sub-motif
  inside a larger invalid partition. That is a different runtime-planning
  problem than the current story contract.

### Stable-callsite file: `squander/partitioning/noisy_runtime_core.py`

The preferred plan is **no change** in this file.

#### Keep stable

- `execute_partitioned_density()`
- `execute_partitioned_density_channel_native()`
- `NoisyRuntimeExecutionResult`
- `NoisyRuntimeFusedRegionRecord`

#### Allowed internal adjustments

- A tiny internal call-boundary adjustment is acceptable only if the new public
  validation split forces it.
- Do not add new runtime-path labels, new result fields, or silent fallback
  behavior in this task.

### Regression anchor: `tests/partitioning/test_partitioned_channel_native_phase31_slice.py`

This file should remain the completed first-slice public regression bar.

#### Keep unchanged as the regression gate

- `test_phase31_channel_native_1q_microcase_matches_sequential_reference()`
- `test_phase31_channel_native_parametric_noise_clamped_matches_sequential()`
- `test_phase31_channel_native_rejects_pure_unitary_motif()`

### Integration evidence file: `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`

This is the most natural place to expand second-slice public runtime evidence,
provided the module docstring is updated honestly once it stops being
helper-only.

#### Expand this module to cover

- public positive runtime execution on
  `phase31_microcase_2q_cnot_local_noise_pair`,
- larger-workload public runtime execution on the frozen 4-qubit spectator smoke
  case,
- runtime-path and fused-region presence assertions,
- and exact state comparison against `execute_sequential_density_reference(...)`.

### Fixture file: `tests/partitioning/fixtures/workloads.py`

This file becomes part of `P31-S05-E01` only if the larger-workload smoke case
is kept as a named frozen fixture.

#### Preferred fixture responsibility

- Add `phase31_local_support_q4_spectator_embedding_smoke` as a deterministic
  non-counted smoke workload if it does not already exist elsewhere.
- Keep the fused motif itself within the frozen `1`- or `2`-qubit same-support
  contract; the “larger workload” part should come from spectator qubits outside
  the fused support, not from widening the supported block.

## Implementation Sequence

### Step 1: Replace whole-workload gating with local-support eligibility

**Goal**

Make public channel-native admission depend on the actual motif support that will
be fused, not on the total workload width.

**Execution checklist**

- [ ] Remove the `descriptor_set.qbit_num == 1` admission rule from the positive
      path.
- [ ] Remove the hard `local_support == (0,)` admission rule from the positive
      path.
- [ ] Keep non-empty members, allowed operation names, noise presence, and
      support width `<= 2` as explicit guards.
- [ ] Preserve the existing `first_unsupported_condition` vocabulary:
      `channel_native_support_surface`,
      `channel_native_noise_presence`,
      `channel_native_qubit_span`.
- [ ] Keep ascending `local_support` order as the canonical mapping contract for
      later apply/audit steps.

**Why first**

Until this step lands, the already-tested E01/E02 substrate cannot become
visible through the public runtime at all.

### Step 2: Reuse the existing mapped-application substrate without redesigning the runtime shell

**Goal**

Expose the local-support execution path that already exists internally, while
keeping the public runtime/API identity stable.

**Execution checklist**

- [ ] Keep `execute_partition_channel_native()` structured around the existing
      ordered member loop and support-aware apply path.
- [ ] Ensure the same validated `local_support` feeds:
      lowering,
      global-target mapping,
      bundle application,
      and fused-region recording.
- [ ] Keep `phase31_channel_native` as the reported runtime path when this branch
      executes successfully.
- [ ] Do not add fallback behavior to the Phase 3 fused or sequential paths when
      channel-native mode was explicitly requested.

**Why this step matters**

`P31-S05-E01` is a public runtime-admission task, not a second rewrite of the
Task 1 bundle mathematics.

### Step 3: Add positive public integration coverage for bounded local-support execution

**Goal**

Prove that the public runtime now executes the bounded same-support channel on
the correct global qubits.

**Execution checklist**

- [ ] Add an end-to-end public runtime test for
      `phase31_microcase_2q_cnot_local_noise_pair`.
- [ ] Add or freeze the larger-workload smoke case
      `phase31_local_support_q4_spectator_embedding_smoke`.
- [ ] Add a public runtime smoke test for that larger-workload case.
- [ ] Compare the resulting density matrix against
      `execute_sequential_density_reference(...)` using the frozen
      `P31-ADR-008` tolerance policy.
- [ ] Assert that the runtime result reports `phase31_channel_native` and
      contains at least one `PHASE31_FUSION_KIND_CHANNEL_NATIVE_MOTIF` fused
      region.

**Recommended positive assertions**

- counted 2-qubit microcase:
  - final-state agreement,
  - runtime-path identity,
  - at least one channel-native fused region.
- larger-workload 4-qubit smoke:
  - final-state agreement,
  - fused execution on a 2-qubit local support embedded inside the larger state,
  - no accidental widening of the fused support itself.

### Step 4: Preserve loud unsupported behavior while reopening the public surface

**Goal**

Keep the runtime honest at the boundaries while positive admission is widened.

**Execution checklist**

- [ ] Keep pure-unitary same-support motifs as explicit
      `channel_native_noise_presence` failures.
- [ ] Keep support widths above `2` as explicit
      `channel_native_qubit_span` failures.
- [ ] Keep unsupported operations or unsupported support patterns as explicit
      `channel_native_support_surface` failures.
- [ ] Avoid broadening the supported motif matrix beyond the written contract in
      `P31-ADR-007`.

**Boundary note**

The focused negative matrix and stricter audit assertions belong to
`P31-S05-E02`; this step is about preserving the hard-failure contract while the
positive surface is reopened.

### Step 5: Keep slice boundaries explicit and prepare handoff

**Goal**

Land the public runtime widening without blurring the surrounding story stack.

**Execution checklist**

- [ ] Re-run the existing first-slice pytest file unchanged as the 1-qubit public
      regression anchor.
- [ ] Keep the second-slice module docstring honest if it now contains public
      runtime evidence in addition to helper-level substrate checks.
- [ ] Leave the exhaustive negative local-support matrix and richer audit
      assertions to `P31-S05-E02`.
- [ ] Leave counted correctness-bundle promotion to `P31-S06-E01`.

## Acceptance Evidence

`P31-S05-E01` is ready to hand off when all of the following are true:

- `execute_partitioned_density_channel_native(...)` succeeds on
  `phase31_microcase_2q_cnot_local_noise_pair` and matches
  `execute_sequential_density_reference(...)` under the frozen exactness
  tolerance,
- a larger-workload local-support smoke case executes through the public
  channel-native path and matches the same sequential reference,
- successful runs report runtime path `phase31_channel_native` and at least one
  `PHASE31_FUSION_KIND_CHANNEL_NATIVE_MOTIF` fused region,
- the existing first-slice 1-qubit public-runtime regression file still passes
  unchanged,
- and unsupported channel-native requests still fail loudly rather than
  downgrading silently.

## Handoff To The Next Engineering Tasks

After `P31-S05-E01` lands:

- `P31-S05-E02` should freeze the public negative matrix and assert fused-region
  audit fields more tightly,
- `P31-S06-E01` should promote the counted 2-qubit microcase plus the
  larger-workload smoke case into the second-slice correctness gate,
- later Task 3 / Task 4 work should reuse the same runtime-path and fused-region
  audit surface rather than inventing a parallel attribution layer.

## Risks / Rollback

- Risk: public admission may succeed while the mapped application still targets
  the wrong global qubits on larger workloads.
  Rollback/mitigation: compare to the full sequential density reference on both
  the counted 2-qubit microcase and the larger-workload smoke case before
  widening further.

- Risk: removing the whole-workload gate may accidentally imply support for
  sub-motif mining or broader unsupported patterns.
  Rollback/mitigation: keep whole-partition execution unchanged and preserve the
  existing unsupported taxonomy for everything outside the frozen same-support
  slice.

- Risk: the missing 4-qubit smoke fixture could cause ad hoc test creation and
  drift in naming.
  Rollback/mitigation: freeze the exact workload ID before implementation, or
  explicitly document the substitute deterministic smoke case if a rename is
  chosen.
