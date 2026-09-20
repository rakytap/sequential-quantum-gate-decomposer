# Engineering Task P31-S04-E02 Implementation Plan

## Engineering Task Being Implemented

`Engineering Task P31-S04-E02: Add CNOT lowering and ordered 2-qubit composition tests`

This is a Layer 4 file-level implementation plan for the second engineering task
under Story `P31-S04` from
`../SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`.
It turns the second-slice task wording into a concrete plan against the current
runtime substrate in:

- `squander/partitioning/noisy_runtime_channel_native.py`,
- `squander/partitioning/noisy_runtime_fusion.py`,
- `tests/partitioning/test_partitioned_channel_native_phase31_slice.py`,
- `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`,
- and the frozen microcase definitions in
  `tests/partitioning/fixtures/workloads.py`.

## Scope

This engineering task lands the remaining lowering and composition substrate
needed before the bounded 2-qubit slice can be exposed more broadly:

- lower `CNOT` into the primary `kraus_bundle` representation on 2-qubit local
  support,
- lift 1-qubit `U3` and local-noise members onto the active 2-qubit support so
  ordered composition stays inside one consistent `d = 4` Kraus space,
- prove ordered compositions of `U3`, `CNOT`, and local single-qubit noise on
  either qubit against the sequential density oracle,
- keep the current public runtime entrypoints stable,
- and preserve the completed E01 substrate plus the 1-qubit public regression
  gate.

Out of scope for this engineering task:

- reopening the whole-workload-width gate in `_validate_whole_partition_motif()`
  or otherwise making the public runtime positively admit larger-workload local
  support motifs (`P31-S05-E01`),
- public end-to-end promotion of the counted 2-qubit microcase through
  `execute_partitioned_density_channel_native(...)` (`P31-S06-E01`),
- wider same-support smoke cases, fused-region public-audit expansion, and the
  negative local-support matrix (`P31-S05-E02`),
- Aer evidence rows, correctness-bundle packaging, and performance packaging.

## Current Runtime Gap To Close

E01 delivered the size-aware substrate, but the member-lowering layer is still
stuck in the 1-qubit world.

### `squander/partitioning/noisy_runtime_channel_native.py`

The main E02-local gaps are:

- `_member_to_kraus_bundle()` still lowers `U3` and all local-noise members only
  as `2x2` bundles, regardless of the active motif support width.
- `_compose_kraus_bundles()` now correctly enforces equal dimensions, so any
  2-qubit same-support motif would immediately fail when the `4x4` identity seed
  is composed with `2x2` single-qubit member bundles.
- `_member_to_kraus_bundle()` still rejects `CNOT`, which is the explicit defer
  point left by E01.
- `execute_partition_channel_native()` already has `local_support`, so the
  missing piece is support-aware member lowering and ordered composition, not a
  redesign of the execution loop.
- `_validate_whole_partition_motif()` still keeps the public positive path at
  the 1-qubit whole-workload boundary; that defer is still correct for E02.

### `squander/partitioning/noisy_runtime_fusion.py`

This file already exposes the canonical control/target and bit-ordering
semantics the new channel-native lowering should match:

- `_kernel_indices_for_fused_cnot()`,
- `_embed_single_qubit_gate()`,
- `_embed_cnot_gate()`.

E02 should prefer reusing these semantics rather than inventing a second local
ordering rule inside the channel-native path.

### Existing second-slice tests

`tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`
currently proves only the E01 substrate:

- 4x4 invariant checks,
- 1-qubit bundle embed into a 2-qubit density matrix,
- 2-qubit identity sanity,
- 4x4 apply ordering via a product-unitary check,
- and explicit representation failures.

That module does **not** yet prove:

- direct `CNOT` lowering,
- support-aware lifting of `U3` and local-noise members into `4x4` bundles,
- ordered mixed-member composition on the counted 2-qubit motifs,
- or descriptor-order dependence for 2-qubit same-support compositions.

## Dependencies And Assumptions

- The source-of-truth contract remains:
  - `task-1/TASK_1_MINI_SPEC.md`,
  - `../SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`,
  - `../DETAILED_PLANNING_PHASE_3_1.md`,
  - `../ADRs_PHASE_3_1.md`
    (`P31-ADR-004`, `P31-ADR-007`, `P31-ADR-008`, `P31-ADR-009`,
    `P31-ADR-012`).
- E01 is already in place:
  - size-aware bundle contract (`d in {2, 4}`),
  - dimension-aware invariant checks,
  - support-aware apply semantics,
  - and the frozen 2-qubit subsystem indexing rule
    `sub(state) = b_g0 + 2 * b_g1`.
- `tests/partitioning/fixtures/workloads.py` already exposes the counted
  2-qubit cases required for E02:
  - `phase31_microcase_2q_cnot_local_noise_pair`,
  - `phase31_microcase_2q_multi_noise_entangler_chain`,
  - `phase31_microcase_2q_dense_same_support_motif`.
- The canonical local-support ordering remains ascending local-qubit order,
  carried consistently through member lowering, composition, and later apply.
- Public positive 2-qubit runtime exposure remains a later task. E02 may use
  direct helper imports and helper-level sequential comparisons to prove the new
  substrate without reopening `P31-S05-E01` early.

## Target Files And Responsibilities

### Primary file: `squander/partitioning/noisy_runtime_channel_native.py`

This file should carry almost all substantive code changes for `P31-S04-E02`.

#### Member lowering to generalize

- Extend `_member_to_kraus_bundle()` so it lowers each member onto the **active
  motif support width**, not just onto the primitive gate/noise width.
- Recommended signature change:

```python
def _member_to_kraus_bundle(
    descriptor_set: NoisyPartitionDescriptorSet,
    member: NoisyPartitionDescriptorMember,
    segment_parameters: np.ndarray,
    *,
    local_support: tuple[int, ...],
    runtime_path: str,
) -> np.ndarray:
    ...
```

- Derive `local_qbit_to_support_index = {lq: idx for idx, lq in enumerate(local_support)}`
  once per call, or precompute and thread the mapping from
  `execute_partition_channel_native()`.

#### New helpers to add

- Add a helper that lifts a 1-qubit Kraus bundle onto bounded local support, for
  example:
  - `_embed_single_qubit_bundle_on_local_support(bundle_1q, *, support_qbit_count, support_target_qbit, ...)`
- For support width `1`, that helper should return the original `2x2` bundle.
- For support width `2`, it should embed each Kraus operator into a `4x4`
  matrix using `_embed_single_qubit_gate(total_kernel_qbits=2, ...)`.
- Add a dedicated `CNOT` lowering helper, for example:
  - `_kraus_bundle_cnot(*, support_control_qbit: int, support_target_qbit: int, ...) -> np.ndarray`
- Prefer deriving that unitary from `_kernel_indices_for_fused_cnot(...)` plus
  `_embed_cnot_gate(total_kernel_qbits=2, ...)` so the control/target
  convention stays identical to the fused-kernel path.

#### Lowering behavior to preserve

- `U3` on a 1-qubit motif still lowers to one `2x2` unitary Kraus operator.
- Local single-qubit noise on a 1-qubit motif still lowers to the existing `2x2`
  bundle families.
- Support widths above `2` still fail explicitly under the existing unsupported
  taxonomy.

#### Lowering behavior to add

- `U3` inside a 2-qubit same-support motif lowers to a single-operator `4x4`
  bundle on the correct support line.
- Local single-qubit noise inside a 2-qubit same-support motif lowers to a `4x4`
  Kraus bundle with each `2x2` Kraus operator embedded on the correct local line.
- `CNOT` inside a 2-qubit same-support motif lowers to a single-operator `4x4`
  bundle with the correct control/target orientation.

### Reference/oracle file: `squander/partitioning/noisy_runtime_fusion.py`

The preferred plan is **no functional change** in this file.

Treat it as the orientation oracle for:

- `_embed_single_qubit_gate()`,
- `_kernel_indices_for_fused_cnot()`,
- `_embed_cnot_gate()`.

If reuse by import becomes awkward, copy the semantics exactly and cite the
reference helpers in comments/tests. Do not create a second orientation rule.

### Stable-callsite file: `squander/partitioning/noisy_runtime_core.py`

The preferred plan is **no change** in this file.

#### Keep stable

- `execute_partitioned_density()`
- `execute_partitioned_density_channel_native()`
- `NoisyRuntimeExecutionResult`
- `NoisyRuntimeFusedRegionRecord`

#### Allowed internal adjustments

- Passing `local_support` (or a derived mapping) deeper into
  `execute_partition_channel_native()` is fine if it remains internal to
  `noisy_runtime_channel_native.py`.
- Do not add new runtime-path labels or result fields in this task.
- Do not weaken the current no-silent-fallback contract.

### Regression anchor: `tests/partitioning/test_partitioned_channel_native_phase31_slice.py`

This file should remain the completed 1-qubit public-runtime regression bar.

#### Keep unchanged as the regression gate

- `test_phase31_channel_native_1q_microcase_matches_sequential_reference()`
- `test_phase31_channel_native_parametric_noise_clamped_matches_sequential()`
- `test_phase31_channel_native_rejects_pure_unitary_motif()`

### E02 evidence file: `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`

This is the right place for `P31-S04-E02` evidence.

#### Expand this module to cover

- direct `CNOT` lowering,
- reverse-orientation sanity where available,
- ordered composition on counted 2-qubit microcases,
- descriptor-order dependence,
- and continued documentation that the public 2-qubit runtime surface remains
  deferred after E02.

### Fixture file: `tests/partitioning/fixtures/workloads.py`

The preferred plan is **no change**.

Reuse the frozen counted cases already present in the Phase 3.1 microcase
inventory rather than inventing fresh identifiers.

## Implementation Sequence

### Step 1: Freeze the 2-qubit local-order and `CNOT` lowering contract

**Goal**

Turn `CNOT` from the remaining defer point into a concrete `4x4` Kraus/unitary
lowering that matches the fused-kernel semantics already shipped in Phase 3.

**Execution checklist**

- [ ] Add a helper that lowers `CNOT` on 2-qubit local support into a
      `(1, 4, 4)` bundle.
- [ ] Map descriptor `local_control_qbit` / `local_target_qbit` into support
      indices using ascending `local_support`.
- [ ] Reuse `_kernel_indices_for_fused_cnot()` and `_embed_cnot_gate()`, or
      match them exactly if duplication is unavoidable.
- [ ] Fail clearly if `CNOT` is asked to lower without both local wires or
      without 2-qubit support.

**Why first**

Everything else in E02 depends on a frozen control/target convention; the
composition tests are not meaningful until the 4x4 `CNOT` itself is
unambiguous.

### Step 2: Make `_member_to_kraus_bundle()` support-aware for 2-qubit motifs

**Goal**

Ensure every member in a 2-qubit same-support motif contributes a `4x4` bundle,
so ordered composition proceeds in one consistent primary space.

**Execution checklist**

- [ ] Extend `_member_to_kraus_bundle()` to receive `local_support` or an
      equivalent local-to-support index mapping.
- [ ] Keep 1-qubit-support lowering unchanged for `U3` and the existing noise
      families.
- [ ] Lift `U3` members to `4x4` bundles when `len(local_support) == 2`.
- [ ] Lift local single-qubit noise members to `4x4` bundles when
      `len(local_support) == 2`, embedding each Kraus operator on the correct
      support line.
- [ ] Lower `CNOT` to `4x4` only when `len(local_support) == 2`; do not imply
      any broader gate-family support.

**Recommended implementation rule**

For support width `2`, prefer `_embed_single_qubit_gate()` per Kraus operator
over raw `np.kron(...)` unless the factor order is explicitly audited. E01
already proved the bit-order contract; E02 should reuse it.

**Why this step matters**

After E01, `_compose_kraus_bundles()` rejects mixed dimensions by design. E02
must therefore upgrade member lowering, not composition, to make 2-qubit motifs
legal.

### Step 3: Thread the new lowering path through the internal composition loop

**Goal**

Make the existing channel-native composition loop capable of building a 2-qubit
same-support fused bundle internally, without yet reopening the public 2-qubit
runtime surface.

**Execution checklist**

- [ ] Pass `local_support` into `_member_to_kraus_bundle()` from
      `execute_partition_channel_native()`.
- [ ] Keep the identity seed bundle support-sized as already implemented in E01.
- [ ] Keep `_compose_kraus_bundles()`, `_check_kraus_bundle_invariants()`, and
      `_apply_kraus_bundle()` as the E01 substrate rather than reopening them.
- [ ] Keep `_validate_whole_partition_motif()` whole-workload gating unchanged
      unless the task boundary is deliberately widened beyond the current docs.

**Recommended implementation constraint**

Do not weaken the current `channel_native_qubit_span` / 1-qubit whole-workload
gate in this task. The current story stack still reserves that exposure for
`P31-S05-E01`.

### Step 4: Add direct lowering and ordered-composition tests

**Goal**

Prove the new 2-qubit lowering semantics before public runtime exposure.

**Execution checklist**

- [ ] Add a direct lowering test comparing channel-native `CNOT` lowering to the
      fused-kernel oracle on `phase31_microcase_2q_cnot_local_noise_pair`.
- [ ] Add at least one reverse-orientation or mixed-orientation check, ideally
      using `phase31_microcase_2q_multi_noise_entangler_chain`, which already
      contains both `CNOT(1,0)` and `CNOT(0,1)` forms.
- [ ] Add a helper-level composition test where a 2-qubit mixed motif (`U3`,
      `CNOT`, local noise on either qubit, later `U3`) matches sequential
      density evolution on a bounded 2-qubit density matrix.
- [ ] Add a test/assertion that descriptor order, not an internal shortcut
      order, defines the fused result.

**Recommended test shapes**

1. **Direct lowering**

   - inspect the `CNOT` member from
     `phase31_microcase_2q_cnot_local_noise_pair`,
   - lower via `_member_to_kraus_bundle(..., local_support=(0, 1), ...)`,
   - compare to
     `np.array([_embed_cnot_gate(total_kernel_qbits=2, kernel_control_qbit=0, kernel_target_qbit=1)])`.

2. **Ordered same-support composition**

   - build the bundle by iterating members in descriptor order,
   - compose with `_compose_kraus_bundles(...)`,
   - validate with `_check_kraus_bundle_invariants(...)`,
   - apply with `_apply_kraus_bundle(...)`,
   - compare to `execute_sequential_density_reference(...)` or an equivalent
     dense sequential oracle on the same descriptor set and parameter vector.

3. **Descriptor-order dependence**

   - compare the descriptor-ordered fused output to an intentionally reordered
     composition of the same lowered members,
   - assert the descriptor-ordered version matches the sequential oracle and the
     reordered version does not (or at minimum differs numerically on a
     deterministic state).

**Why helper-level tests are enough here**

The story wording asks for lowering/composition evidence **before** full public
2-qubit runtime wiring. That matches the current defer boundary where positive
2-qubit runtime admission is still reserved for `P31-S05-E01`.

### Step 5: Keep the regression boundary explicit and prepare handoff

**Goal**

Land E02 evidence without blurring the task boundary.

**Execution checklist**

- [ ] Re-run the existing first-slice pytest file unchanged as the 1-qubit
      public regression anchor.
- [ ] Keep the second-slice test module docstring honest: helper substrate and
      direct-comparison evidence only, not full public 2-qubit runtime
      admission.
- [ ] Avoid adding new runtime-path labels or result fields.
- [ ] Leave larger-workload same-support admission and fused-region public audit
      to `P31-S05-E01` / `P31-S05-E02`.
- [ ] Leave counted public microcase promotion to `P31-S06-E01`.

## Acceptance Evidence

`P31-S04-E02` is ready to hand off when all of the following are true:

- channel-native can lower `CNOT` into a `4x4` bundle with a control/target
  orientation that matches the fused-kernel oracle,
- 2-qubit same-support members (`U3`, `CNOT`, local single-qubit noise on
  either qubit) lower to one consistent `d = 4` bundle space and compose
  without dimension mismatch,
- helper-level tests show ordered 2-qubit mixed-motif composition matches
  sequential density evolution on the counted microcases,
- the existing first-slice 1-qubit public-runtime regression file still passes
  unchanged,
- and the public 2-qubit runtime surface is still not claimed ahead of
  `P31-S05-E01`.

## Handoff To The Next Engineering Tasks

After `P31-S04-E02` lands:

- `P31-S05-E01` should remove the remaining whole-workload-width gating and make
  the already-tested 2-qubit same-support channel-native apply path visible
  inside larger workloads.
- `P31-S05-E02` should add the negative local-support matrix and fused-region
  audit coverage once the public runtime surface is reopened.
- `P31-S06-E01` should promote `phase31_microcase_2q_cnot_local_noise_pair` and
  the next counted 2-qubit microcases into end-to-end sequential-reference
  comparisons through the public channel-native entrypoint.

## Risks / Rollback

- Risk: `CNOT` control / target orientation may silently flip if local-support
  index mapping diverges from the fused-kernel semantics.
  Rollback/mitigation: use `_embed_cnot_gate()` as the oracle and test both
  direct lowering and final-state agreement.

- Risk: support-aware member lowering may embed `U3` or local noise on the wrong
  local line inside a 2-qubit motif.
  Rollback/mitigation: build `4x4` single-qubit bundles with
  `_embed_single_qubit_gate()` per Kraus operator and include “noise on either
  qubit” composition tests.

- Risk: helper-level tests could accidentally imply full public runtime support.
  Rollback/mitigation: keep the docstring and acceptance language explicit that
  public 2-qubit admission remains deferred to `P31-S05-E01`.
