# Engineering Task P31-S07-E01 Implementation Plan

## Engineering Task Being Implemented

`Engineering Task P31-S07-E01: Add explicit hybrid runtime entrypoint and partition-route policy`

This is a Layer 4 file-level implementation plan for the first engineering task
under Story `P31-S07` from
`../THIRD_VERTICAL_SLICE_STRICT_PLUS_HYBRID_WHOLE_WORKLOAD_STORIES_AND_ENGINEERING_TASKS.md`.
It turns the third-slice Task 2 wording into a concrete plan against the
current runtime orchestration and evidence substrate in:

- `squander/partitioning/noisy_runtime_core.py`,
- `squander/partitioning/noisy_runtime_channel_native.py`,
- `squander/partitioning/noisy_runtime_fusion.py`,
- `tests/partitioning/test_partitioned_channel_native_phase31_slice.py`,
- `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`,
- and a recommended successor hybrid test module
  `tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py`.

It may also reuse existing workload builders without mutating them:

- `tests/partitioning/fixtures/workloads.py`,
- `benchmarks/density_matrix/planner_surface/workloads.py`,
- `benchmarks/density_matrix/planner_surface/common.py`,
- and `squander.partitioning.noisy_planner`
  (`build_phase3_continuity_partition_descriptor_set`).

## Scope

This engineering task lands the **runtime split** needed before counted hybrid
continuity or structured-performance work can proceed cleanly:

- add a distinct public Phase 3.1 **hybrid** runtime identity:
  `execute_partitioned_density_channel_native_hybrid(...)` /
  `phase31_channel_native_hybrid`,
- preserve the existing **strict** motif-proof identity
  `execute_partitioned_density_channel_native(...)` /
  `phase31_channel_native` unchanged,
- classify each partition as:
  - eligible for the channel-native path,
  - or routable to the shipped Phase 3 exact path for a frozen support-surface
    reason,
  - or unsupported-by-both and therefore a hard failure,
- emit partition-level route attribution on successful hybrid runs,
- add one positive hybrid continuity case and one unsupported-by-both negative
  case as the minimum public evidence for this task,
- and keep the completed strict 1-qubit / bounded 2-qubit slices as regression
  anchors rather than reopening them.

Out of scope for this engineering task:

- the counted hybrid continuity correctness gate itself (`P31-S08-E01`),
- the representative structured pilot benchmark row and baseline-trio timing
  (`P31-S09-E01`),
- full `correctness_evidence` / `performance_evidence` builder migration under
  `P31-ADR-013`,
- Aer rows from `P31-ADR-011`,
- new gate/noise families,
- support beyond `2` local qubits for channel-native execution,
- Task 6 host-acceleration work.

## Current Runtime Gap To Close

The Phase 3.1 strict substrate is now strong enough to execute bounded 1-qubit
and 2-qubit same-support motifs, but the public runtime still only supports an
all-or-nothing channel-native interpretation.

### `squander/partitioning/noisy_runtime_core.py`

The main `P31-S07-E01` gap is concentrated in the top-level execution loop:

- `execute_partitioned_density()` currently distinguishes only:
  - Phase 3 baseline,
  - Phase 3 fused unitary-island mode,
  - and strict `phase31_channel_native`,
- once `phase31_channel_native` is requested, **every** partition is routed
  through `execute_partition_channel_native(...)`,
- there is no hybrid runtime-path label,
- there is no public helper for a hybrid path,
- and there is no partition-level route attribution beyond the existing
  `NoisyRuntimePartitionRecord` shell.

This is the key architectural blocker for:

- `phase2_xxz_hea_q4_continuity`,
- `phase2_xxz_hea_q6_continuity`,
- and the frozen Phase 3.1 structured performance families.

### `squander/partitioning/noisy_runtime_channel_native.py`

The strict channel-native executor already has the right mathematics for the
eligible case:

- `_validate_whole_partition_motif()` performs the support-surface checks,
- `execute_partition_channel_native()` composes the ordered Kraus bundle,
- `_check_kraus_bundle_invariants()` enforces the current invariant policy,
- `_apply_kraus_bundle()` already knows how to apply a valid bounded 1q/2q
  channel on local support inside a larger global density state.

The missing piece is **classification**:

- the current code raises on every ineligible partition,
- but hybrid routing needs a preflight distinction between:
  - support-surface ineligibility that is still Phase-3-supported and therefore
    routable,
  - and non-routable failures such as representation mismatch, invariant
    failure, or execution failure.

### `squander/partitioning/noisy_runtime_fusion.py`

This file already contains the shipped Phase 3 exact per-partition executor:

- `_execute_partition_with_optional_fusion(...)`

That is the natural execution target for hybrid-routed partitions and should be
reused rather than reimplemented.

### Existing slice tests

The current public tests are split cleanly:

- `tests/partitioning/test_partitioned_channel_native_phase31_slice.py`
  remains the strict 1-qubit regression bar,
- `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`
  proves the bounded 2-qubit strict runtime and local-support smoke path,
- there is currently **no** dedicated hybrid-runtime test module.

That suggests a clean testing strategy:

- keep the strict regression files stable,
- add a new hybrid-focused successor module rather than overloading the existing
  second-slice file with a second execution interpretation.

## Dependencies And Assumptions

- The source-of-truth contract remains:
  - `task-2/TASK_2_MINI_SPEC.md`,
  - `../THIRD_VERTICAL_SLICE_STRICT_PLUS_HYBRID_WHOLE_WORKLOAD_STORIES_AND_ENGINEERING_TASKS.md`,
  - `../DETAILED_PLANNING_PHASE_3_1.md`,
  - `../ADRs_PHASE_3_1.md`
    (`P31-ADR-012`, `P31-ADR-013`, `P31-ADR-003`, `P31-ADR-005`).
- The strict substrate is assumed complete:
  - size-aware `kraus_bundle` representation,
  - bounded 1q/2q invariant checks,
  - ordered `CNOT` lowering,
  - strict local-support public execution for the second slice.
- `phase2_xxz_hea_q4_continuity` is already a frozen counted continuity ID and
  should be the smallest positive hybrid case for this task.
- `build_phase3_continuity_partition_descriptor_set(...)` already builds the
  continuity descriptor surface and should be reused rather than replaced.
- `build_phase31_structured_descriptor_set(...)` already exists for later hybrid
  structured pilot work; this task may use one case as a non-claim-bearing smoke
  if route-policy sanity requires it, but baseline-trio timing belongs to
  `P31-S09-E01`.
- `NoisyRuntimePartitionRecord` is currently too thin to carry hybrid route
  attribution as frozen in `P31-ADR-013`; this task is allowed to extend it
  additively.
- `NoisyRuntimeFusedRegionRecord` should remain the region-level audit surface;
  partition routing should not be encoded indirectly by inventing fake fused
  regions.

## Target Files And Responsibilities

### Primary orchestration file: `squander/partitioning/noisy_runtime_core.py`

This file should carry the top-level hybrid-runtime implementation.

#### Runtime identities to add

- Add a new runtime-path constant:
  - `PHASE31_RUNTIME_PATH_CHANNEL_NATIVE_HYBRID = "phase31_channel_native_hybrid"`
- Add a new public helper:
  - `execute_partitioned_density_channel_native_hybrid(...)`
- Keep the existing strict helper unchanged.

#### Execution-loop refactor to prefer

- Keep `execute_partitioned_density()` as the single orchestration entry.
- Refactor the channel-native branch from one boolean
  `channel_native_path = requested_runtime_path == PHASE31_RUNTIME_PATH_CHANNEL_NATIVE`
  into an explicit mode split:
  - strict channel-native,
  - hybrid channel-native,
  - Phase 3 baseline/fused.
- In the hybrid branch, for each partition:
  1. build the partition circuit and local parameter vector exactly as today,
  2. classify the partition for hybrid routing,
  3. if strict channel-native eligible:
     - execute `execute_partition_channel_native(...)`,
  4. if Phase-3-supported but Phase-3.1-ineligible for a frozen support-surface
     reason:
     - execute `_execute_partition_with_optional_fusion(...)` using the shipped
       Phase 3 exact path with `allow_fusion=True`,
  5. if unsupported-by-both or non-routable:
     - fail loudly.

#### Partition-route attribution to add

- Extend `NoisyRuntimePartitionRecord` and `runtime_partition_audit_dict()` with
  additive hybrid fields, such as:
  - `partition_runtime_class`,
  - `partition_route_reason`.
- The intended vocabulary is the frozen `P31-ADR-013` / `P31-ADR-012` contract:
  - runtime classes:
    - `phase31_channel_native`,
    - `phase3_unitary_island_fused`,
    - `phase3_supported_unfused`,
  - route reasons:
    - `eligible_channel_native_motif`,
    - `pure_unitary_partition`,
    - `channel_native_noise_presence`,
    - `channel_native_qubit_span`,
    - `channel_native_support_surface`.

#### Runtime-path reporting to keep explicit

- Successful hybrid runs should report:
  - `requested_runtime_path == "phase31_channel_native_hybrid"`,
  - `runtime_path == "phase31_channel_native_hybrid"`.
- Strict runs should remain unchanged.

### Secondary classifier file: `squander/partitioning/noisy_runtime_channel_native.py`

This file should provide the preflight boundary between strict execution and
hybrid routing.

#### Preferred change

Add one narrow helper that classifies a partition for hybrid routing without
weakening the strict path, for example:

```python
def classify_partition_channel_native_route(
    descriptor_set,
    partition,
    *,
    runtime_path: str,
) -> tuple[bool, tuple[int, ...] | None, str]:
    ...
```

The key behavior should be:

- reuse `_validate_whole_partition_motif()` or its internal logic,
- if the partition is eligible, return:
  - eligible = `True`,
  - `local_support`,
  - route reason `eligible_channel_native_motif`,
- if it is ineligible for a **routable support-surface reason**, return:
  - eligible = `False`,
  - no `local_support`,
  - one frozen route reason,
- do **not** convert representation, invariant, or runtime-execution errors into
  routable reasons.

#### Mapping rule to freeze

Prefer a reviewer-friendly mapping from strict failures to hybrid route reasons:

- pure unitary partition -> `pure_unitary_partition`,
- local support span above `2` -> `channel_native_qubit_span`,
- other frozen support-surface ineligibility -> `channel_native_support_surface`.

If `channel_native_noise_presence` must be preserved separately for traceability,
document exactly when it is emitted instead of overloading it as a synonym for
pure-unitary routing.

#### Keep unchanged where possible

- `execute_partition_channel_native(...)` should remain the strict whole-partition
  executor for eligible motifs.
- `_apply_kraus_bundle(...)`, `_compose_kraus_bundles(...)`, and invariant logic
  should not be reopened here except for classifier reuse.

### Reused Phase 3 executor: `squander/partitioning/noisy_runtime_fusion.py`

Preferred plan: **no semantic changes**.

#### Reuse

- Reuse `_execute_partition_with_optional_fusion(...)` for hybrid-routed
  partitions.

#### Optional tiny helper

- If needed, add one tiny summary helper to determine whether a routed partition
  ended up as:
  - `phase3_unitary_island_fused`,
  - or `phase3_supported_unfused`,
  based on the returned `NoisyRuntimeFusedRegionRecord`s.

Do **not** redesign the Phase 3 fused-region model in this task.

### Recommended new test module:
`tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py`

This should become the primary evidence file for `P31-S07-E01`.

#### What it should cover

- positive hybrid execution on `phase2_xxz_hea_q4_continuity`,
- route-attribution assertions on that continuity case,
- one unsupported-by-both negative case under the hybrid helper,
- and, if runtime cost is acceptable, one structured hybrid execution smoke
  without timing claims.

#### Why a new file

- It keeps the strict second-slice module honest and stable.
- It gives the hybrid contract one focused home instead of scattering
  whole-workload semantics across strict motif files.

### Strict regression anchors to keep unchanged

- `tests/partitioning/test_partitioned_channel_native_phase31_slice.py`
- `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`

These should remain the strict regression bars, not become hybrid runtime tests.

## Implementation Sequence

### Step 1: Add the hybrid runtime identity and additive audit surface

**Goal**

Create a public Phase 3.1 whole-workload execution identity that is distinct
from the strict motif-proof path and from the shipped Phase 3 exact path.

**Execution checklist**

- [ ] Add `PHASE31_RUNTIME_PATH_CHANNEL_NATIVE_HYBRID`.
- [ ] Add `execute_partitioned_density_channel_native_hybrid(...)`.
- [ ] Extend `NoisyRuntimePartitionRecord` additively with route-attribution
      fields.
- [ ] Extend `runtime_partition_audit_dict()` to emit those new fields.
- [ ] Keep the strict helper and strict runtime-path behavior unchanged.

**Why first**

Without a distinct runtime identity and additive audit surface, the hybrid path
would collapse into silent downgrade by construction.

### Step 2: Split routable support-surface ineligibility from hard failures

**Goal**

Make the hybrid runtime able to decide whether a partition should:

- execute channel-natively,
- route to the shipped Phase 3 exact path,
- or fail.

**Execution checklist**

- [ ] Add a preflight route-classifier helper in
      `noisy_runtime_channel_native.py`.
- [ ] Reuse the existing `_validate_whole_partition_motif()` logic where
      possible.
- [ ] Freeze one mapping from strict failure categories to hybrid route reasons.
- [ ] Keep representation mismatch, invariant failure, and runtime-execution
      failure as hard errors rather than reroute conditions.

**Recommended implementation constraint**

Do not route by catching broad exceptions from `execute_partition_channel_native`
itself. Classify **before** execution so routable support-surface ineligibility
is separated cleanly from true execution failures.

### Step 3: Add per-partition hybrid routing in the execution loop

**Goal**

Execute a mixed whole workload through the correct exact path partition by
partition.

**Execution checklist**

- [ ] In the hybrid branch of `execute_partitioned_density()`, classify each
      partition before execution.
- [ ] Eligible partition:
      - call `execute_partition_channel_native(...)`.
- [ ] Routed partition:
      - call `_execute_partition_with_optional_fusion(...)` with the shipped
        Phase 3 exact settings.
- [ ] Unsupported-by-both partition:
      - fail loudly.
- [ ] Derive `partition_runtime_class` from the actual routed execution result.
- [ ] Record `partition_route_reason` on every hybrid partition record.
- [ ] Keep the result-level `runtime_path` equal to the hybrid runtime label.

**Recommended route-summary rule**

For the first implementation, prefer one route record per partition over a more
ambitious nested schema. Task 3 / Task 4 can aggregate these later into:

- `channel_native_partition_count`,
- `phase3_routed_partition_count`,
- `channel_native_member_count`,
- `phase3_routed_member_count`.

### Step 4: Add focused hybrid tests without blurring strict regression files

**Goal**

Prove that the hybrid runtime executes a mixed workload exactly and records why
each partition used its path.

**Execution checklist**

- [ ] Add a new hybrid test module.
- [ ] Positive continuity case:
      - build `phase2_xxz_hea_q4_continuity`,
      - run `execute_partitioned_density_channel_native_hybrid(...)`,
      - compare to `execute_sequential_density_reference(...)`,
      - assert runtime-path identity,
      - assert at least one channel-native-routed partition and at least one
        Phase-3-routed partition.
- [ ] Unsupported-by-both negative case:
      - use one relaxed-surface unsupported gate/noise case,
      - assert the run fails rather than being absorbed into hybrid routing.
- [ ] Optional structured route smoke:
      - run one `phase31_pair_repeat_q8_dense_seed20260318` hybrid smoke only if
        it stays lightweight enough for the test suite,
      - do **not** attach timing expectations here.

**Recommended positive assertions**

- continuity case:
  - `result.runtime_path == "phase31_channel_native_hybrid"`,
  - full-density exactness against the sequential oracle,
  - at least one `partition_runtime_class == "phase31_channel_native"`,
  - at least one `partition_runtime_class in {"phase3_unitary_island_fused",
    "phase3_supported_unfused"}`,
  - route reasons are drawn only from the frozen vocabulary.

### Step 5: Preserve slice boundaries and prepare handoff

**Goal**

Land the hybrid runtime without pretending the continuity correctness gate or
the structured performance pilot are already complete.

**Execution checklist**

- [ ] Keep strict regression files unchanged.
- [ ] Leave Aer rows to later Task 3 work.
- [ ] Leave baseline-trio timing and diagnosis rows to `P31-S09-E01`.
- [ ] Document the current hybrid runtime as enabling:
      - `P31-S08-E01`,
      - `P31-S09-E01`,
      not replacing them.

## Acceptance Evidence

`P31-S07-E01` is ready to hand off when all of the following are true:

- `execute_partitioned_density_channel_native_hybrid(...)` exists and returns
  runtime-path label `phase31_channel_native_hybrid`,
- `phase2_xxz_hea_q4_continuity` executes through the hybrid path and matches
  the sequential oracle under the frozen exactness tolerance,
- the continuity run records at least:
  - one channel-native-routed partition,
  - one Phase-3-routed partition,
  - stable route reasons from the frozen vocabulary,
- one unsupported-by-both hybrid negative case still fails loudly,
- the strict 1-qubit and bounded 2-qubit regression files still pass unchanged,
- and no full bundle migration is required just to close this runtime task.

## Handoff To The Next Engineering Tasks

After `P31-S07-E01` lands:

- `P31-S08-E01` should promote `phase2_xxz_hea_q4_continuity` into the first
  counted hybrid correctness gate with explicit route-summary assertions.
- `P31-S09-E01` should add one representative structured pilot row with the
  baseline trio and route coverage.
- Later Task 3 / Task 4 bundle work should reuse the additive partition-route
  fields from this task rather than inventing a parallel attribution layer.

## Risks / Rollback

- Risk: hybrid routing becomes an undocumented silent downgrade.
  Rollback/mitigation: give the hybrid path its own runtime label, helper, and
  partition-route records from the first patch.

- Risk: support-surface ineligibility and true execution failures get conflated.
  Rollback/mitigation: classify routable cases before channel-native execution
  and keep invariant/runtime failures as hard errors.

- Risk: the hybrid path accidentally changes strict behavior.
  Rollback/mitigation: keep strict helpers and strict regression files unchanged,
  and gate all new behavior behind the new hybrid runtime path.

- Risk: Phase 3 routed partitions are executed through settings that do not
  match the shipped Phase 3 exact baseline.
  Rollback/mitigation: reuse `_execute_partition_with_optional_fusion(...)` with
  the shipped Phase 3 exact semantics instead of introducing a second fallback
  executor.
