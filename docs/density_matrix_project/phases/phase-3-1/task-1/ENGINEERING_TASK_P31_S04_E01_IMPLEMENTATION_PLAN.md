# Engineering Task P31-S04-E01 Implementation Plan

## Engineering Task Being Implemented

`Engineering Task P31-S04-E01: Generalize the primary representation and invariant helpers to support size-aware 1q/2q blocks`

This is a Layer 4 file-level implementation plan for the first engineering task
under Story `P31-S04` from
`../SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`.
It turns the second-slice task wording into a concrete plan against the current
runtime substrate in:

- `squander/partitioning/noisy_runtime_channel_native.py`,
- `squander/partitioning/noisy_runtime_core.py`,
- `tests/partitioning/test_partitioned_channel_native_phase31_slice.py`,
- and a recommended successor test module
  `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`.

## Scope

This engineering task lands the internal runtime substrate needed before the
counted 2-qubit slice can be exposed cleanly:

- remove hard-coded `2x2` / 1-qubit assumptions from the primary
  `kraus_bundle` helpers,
- make invariant checks dimension-aware for bounded 1-qubit and 2-qubit
  objects,
- prepare local-support application of a bounded channel to a larger global
  density matrix through explicit support metadata,
- keep the current public runtime entrypoints stable,
- and preserve the completed 1-qubit slice as a regression gate.

Out of scope for this engineering task:

- `CNOT` lowering and ordered 2-qubit composition (`P31-S04-E02`),
- public positive admission of larger-workload local-support motifs
  (`P31-S05-E01`),
- the counted 2-qubit end-to-end correctness gate (`P31-S06-E01`),
- Aer rows, correctness bundle schema work, and performance packaging.

## Current Runtime Gap To Close

The current runtime code already proves the 1-qubit slice, but the internal
representation layer is still fixed to that narrow case.

### `squander/partitioning/noisy_runtime_channel_native.py`

The main task-local gaps are:

- `_IDENTITY_KRAUS_BUNDLE` is fixed to one `2x2` identity operator.
- `_compose_kraus_bundles()` allocates `(..., 2, 2)` and therefore assumes
  1-qubit channels.
- `_check_kraus_bundle_invariants()` accumulates against `I_2` and hard-codes
  `d = 2` when building the Choi matrix.
- `_apply_kraus_bundle()` rejects every workload except `qbit_num == 1`.
- `_validate_whole_partition_motif()` still ties admissible motifs to the
  whole-workload width rather than only to the motif's local support.
- `_member_to_kraus_bundle()` still rejects `CNOT`, which is correct for this
  task because `CNOT` lowering belongs to `P31-S04-E02`.

### `squander/partitioning/noisy_runtime_core.py`

The public runtime entrypoints are already adequate for this task:

- `execute_partitioned_density()` already threads the `partition`, the current
  `DensityMatrix`, and the `runtime_path` into
  `execute_partition_channel_native(...)`.
- `NoisyRuntimeFusedRegionRecord` already carries `global_target_qbits` and
  `local_target_qbits`, so no new public result fields are required for
  `P31-S04-E01`.
- `execute_partitioned_density_channel_native()` does not need a signature
  change.

### Existing slice tests

`tests/partitioning/test_partitioned_channel_native_phase31_slice.py` is
currently a 1-qubit public-runtime regression gate:

- positive 1-qubit microcase,
- parametric-noise clamp parity,
- pure-unitary rejection.

That file should remain the regression anchor for the completed first slice
rather than becoming a mixed 1q/2q internal-helper catch-all.

## Dependencies And Assumptions

- The source-of-truth contract remains:
  - `task-1/TASK_1_MINI_SPEC.md`,
  - `../SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`,
  - `../DETAILED_PLANNING_PHASE_3_1.md`,
  - `../ADRs_PHASE_3_1.md` (`P31-ADR-004`, `P31-ADR-007`, `P31-ADR-008`).
- `tests/partitioning/fixtures/workloads.py` already exposes the frozen counted
  2-qubit case ID `phase31_microcase_2q_cnot_local_noise_pair`, so this task
  should not invent a second identifier scheme.
- `NoisyRuntimeFusedRegionRecord` already exposes the audit fields needed for
  later local-support runtime tests; this task should reuse that surface rather
  than creating a parallel record.
- The canonical local-support ordering for the second slice should remain an
  explicit, auditable rule. The safest default is ascending local-qubit order
  carried consistently through bundle construction, invariant checks, and later
  runtime application.
- This task may prepare a support-aware apply helper before the runtime publicly
  admits all wider supported motifs. Public exposure belongs to `P31-S05-E01`.

## Target Files And Responsibilities

### Primary file: `squander/partitioning/noisy_runtime_channel_native.py`

This file should carry almost all substantive code changes for `P31-S04-E01`.

#### Representation helpers to change

- Replace `_IDENTITY_KRAUS_BUNDLE` with a helper that builds the identity bundle
  for the active support size, for example:
  - `_identity_kraus_bundle_for_support(support_qbit_count: int) -> np.ndarray`
- Replace hard-coded `2x2` allocation in `_compose_kraus_bundles()` with
  dimension-aware allocation derived from the operand shapes.
- Add explicit shape guards so bundle composition fails clearly if mixed support
  dimensions or non-square Kraus operators are passed accidentally.

#### Invariant helpers to change

- Refactor `_check_kraus_bundle_invariants()` so it derives the matrix dimension
  from the bundle shape rather than from a hard-coded `d = 2`.
- Build the completeness identity as `I_d`, where `d = 2 ** support_qbit_count`
  or the equivalent inferred bundle dimension.
- Build the Choi matrix with the same `d`, not with the old 1-qubit constant.
- Keep the Phase 3.1 thresholds unchanged:
  - completeness-style residuals `<= 1e-10`,
  - positivity-style eigenvalue floors `>= -1e-12`.

#### Apply helper to change

- Refactor `_apply_kraus_bundle()` so the internal contract becomes support-aware,
  for example:

```python
def _apply_kraus_bundle(
    bundle: np.ndarray,
    rho: DensityMatrix,
    *,
    qbit_num: int,
    local_support: tuple[int, ...],
    global_target_qbits: tuple[int, ...],
) -> DensityMatrix:
    ...
```

- Preserve exact 1-qubit behavior as the simplest supported special case.
- Add the minimum pure-NumPy embedding logic needed to apply a 1-qubit or
  2-qubit bounded channel to the full global density matrix.
- Keep support sizes above 2 qubits as explicit hard failures.

#### Validation split to prefer

To keep `P31-S04-E01` narrow and auditable, prefer splitting
`_validate_whole_partition_motif()` into two conceptual responsibilities:

- **support collection / structural validation**
  - collect `local_support`,
  - confirm allowed operation names and noise presence,
  - confirm support width `<= 2`,
- **public positive-surface gating**
  - keep the current whole-workload-width restriction until `P31-S05-E01`
    deliberately reopens it.

This lets `P31-S04-E01` land size-aware helpers without accidentally claiming
the full `P31-S05` runtime surface early.

### Stable-callsite file: `squander/partitioning/noisy_runtime_core.py`

The preferred plan is **minimal change** in this file.

#### Keep stable

- `execute_partitioned_density()`
- `execute_partitioned_density_channel_native()`
- `NoisyRuntimeExecutionResult`
- `NoisyRuntimeFusedRegionRecord`

#### Allowed internal adjustments

- If `execute_partition_channel_native()` needs an internal signature or return
  tweak to thread support-aware helper inputs, keep that change internal to the
  import-and-call boundary in `execute_partitioned_density()`.
- Do not add new public runtime-path labels or result fields in this task.
- Do not weaken the current no-silent-fallback contract.

### Regression anchor: `tests/partitioning/test_partitioned_channel_native_phase31_slice.py`

This file should stay focused on the completed first slice.

#### Keep unchanged as the regression bar

- `test_phase31_channel_native_1q_microcase_matches_sequential_reference()`
- `test_phase31_channel_native_parametric_noise_clamped_matches_sequential()`
- `test_phase31_channel_native_rejects_pure_unitary_motif()`

#### Recommended testing expansion strategy

- Keep the current file as the mandatory 1-qubit public-runtime regression gate.
- Add `P31-S04-E01` internal-substrate tests to a successor file:
  `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`
- If direct helper imports are used for this task, keep them narrowly scoped and
  explain in the module docstring that they validate representation substrate,
  not the full counted public 2-qubit runtime surface yet.

## Implementation Sequence

### Step 1: Freeze a size-aware bundle contract in `noisy_runtime_channel_native.py`

**Goal**

Turn the primary `kraus_bundle` object from a fixed 1-qubit internal shape into
an auditable bounded object for support widths 1 and 2.

**Execution checklist**

- [ ] Add one helper that converts support width into bundle dimension.
- [ ] Replace the fixed identity bundle constant with a size-aware helper.
- [ ] Add explicit bundle-shape guards for:
      rank-3 bundles,
      square Kraus matrices,
      and supported dimensions `{2, 4}` only.
- [ ] Keep failures inside the existing runtime validation taxonomy rather than
      introducing a new error family.

**Why first**

Everything else in this task depends on a correct internal shape contract.

### Step 2: Generalize the invariant suite to 1q and 2q bounded objects

**Goal**

Make `P31-ADR-008` checks work on `2x2` and `4x4` channels without changing the
threshold policy.

**Execution checklist**

- [ ] Refactor `_check_kraus_bundle_invariants()` to derive `d` from the bundle
      shape.
- [ ] Build the completeness residual against `np.eye(d)`.
- [ ] Build the Choi matrix with `d * d` flattening driven by the inferred
      dimension.
- [ ] Add at least one focused 4x4 positive invariant test and one focused
      broken-bundle negative test.

**Recommended narrow test cases**

- Positive: identity 2-qubit bundle.
- Positive: tensor product of two 1-qubit valid channels.
- Negative: deliberately non-trace-preserving 4x4 bundle with residual above the
  frozen tolerance.

### Step 3: Prepare a support-aware apply helper without widening the public surface yet

**Goal**

Allow the runtime to apply a bounded 1q or 2q channel to a global density matrix
through explicit support metadata.

**Execution checklist**

- [ ] Change `_apply_kraus_bundle()` to accept local-support and global-target
      metadata explicitly.
- [ ] Preserve the current 1-qubit exact behavior as a regression-preserving
      fast path.
- [ ] Add a generic embedding path for 1q/2q bundles on the full density matrix.
- [ ] Keep support sizes above 2 qubits as explicit hard failures.

**Recommended implementation constraint**

For `P31-S04-E01`, prefer a simple exact NumPy implementation over premature
optimization. The scientific value here is semantic correctness and explicit
support mapping, not speed.

**Recommended helper-level evidence**

- A direct apply test where a valid 1-qubit bundle acts on one qubit inside a
  2-qubit global density matrix and matches an explicitly embedded dense
  reference result.
- A direct apply test where a valid 2-qubit identity bundle leaves a larger
  density matrix unchanged on the targeted support.

### Step 4: Keep `noisy_runtime_core.py` stable while threading the new helper contract

**Goal**

Avoid unnecessary public-surface churn while the internal channel-native
substrate is generalized.

**Execution checklist**

- [ ] Keep `execute_partitioned_density_channel_native()` unchanged.
- [ ] Keep the `execute_partitioned_density()` channel-native branch structure
      unchanged unless an internal helper signature forces a tiny callsite edit.
- [ ] Reuse `NoisyRuntimeFusedRegionRecord` as-is for later support-audit tests.
- [ ] Avoid adding new runtime result fields during this task.

**Reasoning**

`P31-S04-E01` is a substrate task, not a public API redesign task.

### Step 5: Land tests in the right place and preserve first-slice clarity

**Goal**

Add the minimum regression and helper-level evidence without blurring the
completed first slice and the planned second slice.

**Execution checklist**

- [ ] Re-run the existing first-slice pytest file unchanged as the primary
      regression gate.
- [ ] Add a new second-slice helper-oriented test module rather than mixing
      internal-substrate coverage into the first-slice public-runtime file.
- [ ] Keep public positive 2-qubit runtime tests for `P31-S04-E02` and
      `P31-S05-E01`, not for this engineering task.
- [ ] Document clearly in the new test module which behaviors are still deferred
      after `P31-S04-E01`.

**Recommended test inventory for this task**

- `test_phase31_channel_native_invariants_accept_2q_identity_bundle`
- `test_phase31_channel_native_invariants_reject_broken_2q_bundle`
- `test_phase31_channel_native_apply_helper_embeds_1q_bundle_into_2q_density`
- `test_phase31_channel_native_apply_helper_rejects_support_above_2q`

These tests may import narrow internal helpers from
`squander.partitioning.noisy_runtime_channel_native` because they validate the
representation substrate directly rather than the final counted public slice.

## Acceptance Evidence

`P31-S04-E01` is ready to hand off when all of the following are true:

- the existing first-slice regression file still passes unchanged:
  `conda run -n qgd python -m pytest tests/partitioning/test_partitioned_channel_native_phase31_slice.py -q`
- new focused helper-level tests for 4x4 invariant checks and support-aware
  application pass in the successor second-slice test module,
- `noisy_runtime_core.py` keeps the public channel-native entrypoints stable,
- the implementation still does **not** claim `CNOT` lowering or public positive
  2-qubit runtime execution,
- and the next handoff to `P31-S04-E02` is explicit: `CNOT` lowering plugs into
  a now size-aware `kraus_bundle` substrate rather than reopening the 1q-only
  helpers.

## Handoff To The Next Engineering Tasks

After `P31-S04-E01` lands:

- `P31-S04-E02` should add `CNOT` lowering and ordered 2-qubit composition on
  top of the generalized bundle substrate.
- `P31-S05-E01` should remove the remaining whole-workload-width gating and make
  the local-support apply path visible through the public runtime on bounded
  larger workloads.
- `P31-S06-E01` should promote the smallest counted 2-qubit microcase and one
  larger-workload smoke case into end-to-end sequential-reference tests.

## Risks / Rollback

- Risk: shape-generalization bugs can preserve dimensions while silently
  changing semantics.
  Rollback/mitigation: validate invariants and apply semantics first on tiny
  deterministic helper-level cases before enabling the public 2-qubit path.

- Risk: this task may accidentally widen the public runtime surface before
  `P31-S05-E01` is ready.
  Rollback/mitigation: keep public admission rules narrow and treat wider
  support exposure as the next engineering task's responsibility.

- Risk: direct private-helper tests may over-lock internal function names.
  Rollback/mitigation: keep helper-level tests narrow, local to this substrate
  task, and move behaviorally significant public-runtime tests to the later
  second-slice story work.
