# Engineering Task P31-S06-E01 Implementation Plan

## Engineering Task Being Implemented

`Engineering Task P31-S06-E01: Register the second-slice fixtures and end-to-end comparisons`

This is a Layer 4 file-level implementation plan for the first engineering task
under Story `P31-S06` from
`../SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`.
It turns the second-slice Task 3 wording into a concrete plan against the
current correctness/evidence surfaces in:

- `task-3/TASK_3_MINI_SPEC.md`,
- `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`,
- `tests/partitioning/test_partitioned_channel_native_phase31_slice.py`,
- `tests/partitioning/fixtures/workloads.py`,
- `benchmarks/density_matrix/planner_surface/workloads.py`,
- `tests/partitioning/fixtures/runtime.py`,
- and, for scope control only,
  `benchmarks/density_matrix/correctness_evidence/case_selection.py` plus
  `benchmarks/density_matrix/correctness_evidence/records.py`.

## Scope

This engineering task promotes the already-landed second-slice public runtime
proof into the **minimum Task 3 correctness gate** for this slice:

- keep the counted 2-qubit case ID
  `phase31_microcase_2q_cnot_local_noise_pair` unchanged,
- reuse the frozen larger-workload smoke ID
  `phase31_local_support_q4_spectator_embedding_smoke`,
- extend deterministic pytest evidence so the counted 2-qubit case asserts the
  full internal exactness family required by `P31-C-03`,
- add explicit representation-invariant evidence for the counted fused 2-qubit
  block using the already-landed helper substrate,
- keep the 4-qubit smoke case as **non-counted** slice evidence while still
  asserting exactness on the full density matrix,
- and preserve the first-slice 1-qubit public regression gate unchanged.

Out of scope for this engineering task:

- full `P31-ADR-009` correctness-package closure across all counted IDs,
- Aer rows or any external-reference expansion from `P31-ADR-011`,
- correctness bundle/schema wiring under `P31-ADR-013`,
- `correctness_evidence` package-builder migration to the Phase 3.1 runtime
  path,
- new runtime result fields, new runtime labels, or new fixture ID families,
- and structured performance / diagnosis work from Task 4.

## Current Runtime And Evidence Gap To Close

The second slice is now publicly executable, but it is not yet recorded as the
minimal counted correctness gate promised by Story `P31-S06`.

### `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`

The current second-slice module already covers:

- helper-level substrate checks from `P31-S04`,
- public positive runtime execution from `P31-S05-E01`,
- public negative boundary/audit checks from `P31-S05-E02`,
- and the frozen larger-workload smoke fixture layout.

However, the current public positives still stop short of the Story `P31-S06`
acceptance signals:

- they assert runtime path, fused-region metadata, and density agreement against
  the sequential reference,
- but they do **not** yet assert
  `result.trace_deviation <= 1e-10`,
  `result.rho_is_valid is True`,
  or a representation-invariant check on the counted fused 2-qubit block,
- and the module docstring still says counted end-to-end 2-qubit gates remain
  deferred to `P31-S06-E01`.

That makes this file the natural primary target for `P31-S06-E01`.

### Stable fixture inventory is already in place

Unlike `P31-S05-E01`, this task does **not** start from missing workload IDs:

- `phase31_microcase_2q_cnot_local_noise_pair` already exists and is the frozen
  counted 2-qubit microcase for the second slice,
- `phase31_local_support_q4_spectator_embedding_smoke` already exists in
  `tests/partitioning/fixtures/workloads.py`,
- and that same smoke case is already mirrored in
  `benchmarks/density_matrix/planner_surface/workloads.py`.

The preferred plan is therefore to **reuse** those IDs, not invent new names or
alternate builders.

### The first-slice regression file is already the right exactness pattern

`tests/partitioning/test_partitioned_channel_native_phase31_slice.py` already
shows the intended public exactness idiom for a counted slice:

- runtime-path identity,
- density agreement against the sequential reference,
- `trace_deviation <= PHASE3_RUNTIME_DENSITY_TOL`,
- `rho_is_valid is True`.

`P31-S06-E01` should reuse that exact style for the counted 2-qubit case rather
than broadening the first-slice file itself.

### Full Task 3 package wiring is present only as planning helpers

The broader correctness-evidence harness is **not** yet the right place for this
slice closure:

- `benchmarks/density_matrix/correctness_evidence/case_selection.py` already has
  `iter_phase31_correctness_microcase_cases()` and related Phase 3.1 planning
  helpers,
- but `build_correctness_evidence_case_contexts()` still returns the old Phase 3
  case matrix,
- and `benchmarks/density_matrix/correctness_evidence/records.py` still builds
  positive records through `execute_fused_with_reference(...)`, not through the
  `phase31_channel_native` public path.

So `P31-S06-E01` should stay in **deterministic pytest evidence** and leave the
full artifact/bundle migration to later Task 3 work.

## Dependencies And Assumptions

- The source-of-truth contract remains:
  - `TASK_3_MINI_SPEC.md`,
  - `../SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`,
  - `../DETAILED_PLANNING_PHASE_3_1.md`,
  - `../ADRs_PHASE_3_1.md`
    (`P31-ADR-008`, `P31-ADR-009`, `P31-ADR-011`, `P31-ADR-013`).
- `P31-S04-E01` and `P31-S04-E02` are assumed complete:
  - size-aware bundle representation,
  - support-aware application,
  - `CNOT` lowering,
  - ordered 2-qubit same-support composition,
  - and helper-level invariant checks.
- `P31-S05-E01` and `P31-S05-E02` are assumed complete:
  - public local-support admission is open,
  - the frozen 4-qubit smoke case exists,
  - public fused-region audit fields are already asserted,
  - negative matrix coverage is already in place.
- `NoisyRuntimeExecutionResult` already exposes the public exactness fields this
  task needs:
  - `trace_deviation`,
  - `rho_is_valid`,
  - `runtime_path`,
  - `fused_regions`.
- The counted-vs-non-counted distinction for this task lives in **test/module
  intent**, not in new runtime metadata:
  - counted case:
    `phase31_microcase_2q_cnot_local_noise_pair`,
  - non-counted smoke:
    `phase31_local_support_q4_spectator_embedding_smoke`.
- Representation-invariant evidence should reuse the existing private helper
  substrate already imported by the second-slice test module rather than adding
  new production-facing audit fields.

## Target Files And Responsibilities

### Primary evidence file: `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`

This file should become the second-slice **minimal correctness gate** while
remaining honest about the helper/runtime/boundary layers that landed earlier.

#### What this file should cover after `P31-S06-E01`

- a clearly labeled public correctness subsection for Story `P31-S06`,
- the counted 2-qubit microcase as the minimal claim-bearing end-to-end proof,
- the larger-workload 4-qubit smoke case as non-counted local-support evidence,
- the exactness family required by `P31-C-03`:
  - Frobenius difference,
  - trace validity,
  - density validity,
- and at least one representation-invariant acceptance check on the counted
  fused 2-qubit block.

#### Expected changes in this file

- Update the module docstring so it no longer says counted end-to-end 2-qubit
  proof remains deferred.
- Preserve the existing `P31-S04` helper substrate and `P31-S05` public runtime
  / boundary sections.
- Add or upgrade a final section for the `P31-S06-E01` minimal correctness gate.

### Stable fixture inventory: `tests/partitioning/fixtures/workloads.py`

This file is expected to stay unchanged unless the current smoke-case ID or
partition shape proves unstable while writing the correctness assertions.

#### Keep stable if possible

- `phase31_microcase_2q_cnot_local_noise_pair`
- `phase31_local_support_q4_spectator_embedding_smoke`

### Benchmark mirror inventory: `benchmarks/density_matrix/planner_surface/workloads.py`

This file should also remain unchanged unless fixture parity drifts while the
test module is tightened.

### Regression anchor: `tests/partitioning/test_partitioned_channel_native_phase31_slice.py`

This file remains the completed first-slice public regression gate.

#### Keep unchanged as the 1q bar

- `test_phase31_channel_native_1q_microcase_matches_sequential_reference()`
- `test_phase31_channel_native_parametric_noise_clamped_matches_sequential()`
- `test_phase31_channel_native_rejects_pure_unitary_motif()`

### Explicitly out-of-scope harness files

These files are important for the broader Task 3 package, but the preferred
`P31-S06-E01` plan is **not** to edit them:

- `benchmarks/density_matrix/correctness_evidence/case_selection.py`
- `benchmarks/density_matrix/correctness_evidence/records.py`
- related bundle/validation scripts under
  `benchmarks/density_matrix/correctness_evidence/`

That wider correctness-package migration belongs to later Task 3 work after the
second-slice pytest gate is frozen.

## Implementation Sequence

### Step 1: Recast the second-slice module as a minimal correctness gate

**Goal**

Turn the current second-slice module from “helper substrate + public runtime
admission” into “helper substrate + public runtime/boundary + minimal Task 3
correctness gate”.

**Execution checklist**

- [ ] Update the module docstring in
      `test_partitioned_channel_native_phase31_second_slice.py`.
- [ ] Add a clearly labeled public correctness section for `P31-S06-E01`.
- [ ] Keep the first-slice file named as the unchanged 1-qubit regression anchor.
- [ ] Name the remaining deferred counted IDs and broader Task 3 / Task 4 work
      clearly enough that the module does not over-claim full Phase 3.1 closure.

**Why first**

This task is primarily about **evidence shape**. The module should describe the
new slice boundary honestly before it gains stronger assertions.

### Step 2: Promote the counted 2-qubit microcase to explicit end-to-end correctness evidence

**Goal**

Make `phase31_microcase_2q_cnot_local_noise_pair` the minimal counted
end-to-end proof for the second slice.

**Execution checklist**

- [ ] Keep the counted case ID string unchanged:
      `phase31_microcase_2q_cnot_local_noise_pair`.
- [ ] Reuse `execute_partitioned_density_channel_native(...)` vs
      `execute_sequential_density_reference(...)`.
- [ ] Assert the full internal exactness family on the final density state:
      - Frobenius difference `<= PHASE3_RUNTIME_DENSITY_TOL`,
      - maximum absolute difference `<= PHASE3_RUNTIME_DENSITY_TOL`,
      - `result.trace_deviation <= PHASE3_RUNTIME_DENSITY_TOL`,
      - `result.rho_is_valid is True`.
- [ ] Preserve the already-landed runtime-path and fused-region audit assertions
      from `P31-S05`.
- [ ] Add at least one representation-invariant acceptance check on the fused
      2-qubit bundle using the existing helper substrate.

**Recommended evidence strategy**

Prefer a tiny **test-local** helper that reuses the already imported internal
functions:

- `_build_partition_parameter_vector(...)`
- `_segment_parameter_vector(...)`
- `_identity_kraus_bundle_for_support_qubit_count(...)`
- `_member_to_kraus_bundle(...)`
- `_compose_kraus_bundles(...)`
- `_check_kraus_bundle_invariants(...)`

That keeps invariant evidence inside the test module rather than adding a new
public runtime field just for this task.

### Step 3: Keep the larger-workload smoke case stable and explicitly non-counted

**Goal**

Use the existing 4-qubit local-support smoke case as the minimal “larger
workload” proof without turning it into a broader continuity or bundle claim.

**Execution checklist**

- [ ] Reuse the existing stable smoke ID:
      `phase31_local_support_q4_spectator_embedding_smoke`.
- [ ] Keep the partition-layout precondition explicit:
      partitions map to `(0, 1)` and `(2, 3)`.
- [ ] Extend the smoke test to assert the same full-density exactness family:
      - Frobenius difference,
      - maximum absolute difference,
      - trace deviation,
      - density validity.
- [ ] Preserve fused-region audit checks so the applied local support remains
      reviewable.
- [ ] Keep this case documented as **non-counted** slice evidence, not as a new
      claim-bearing Phase 3.1 counted ID.

**Important boundary**

This smoke case should prove that a bounded 2-qubit motif works inside a larger
4-qubit workload. It should **not** become an accidental continuity anchor or a
surrogate for the deferred `phase2_xxz_hea_q4_continuity` / `q6` counted cases.

### Step 4: Preserve Task 3 boundaries and avoid premature package wiring

**Goal**

Land the minimal correctness gate without pretending the broader Phase 3.1
artifact/bundle surface is already migrated.

**Execution checklist**

- [ ] Keep `tests/partitioning/test_partitioned_channel_native_phase31_slice.py`
      unchanged.
- [ ] Prefer no changes to `tests/partitioning/fixtures/workloads.py` or the
      benchmark workload mirror unless fixture drift is discovered.
- [ ] Do not wire `iter_phase31_correctness_*` into the active
      `correctness_evidence` bundle builders in this task.
- [ ] Do not add Aer rows, schema version bumps, or the remaining counted IDs:
      - `phase31_microcase_2q_multi_noise_entangler_chain`,
      - `phase31_microcase_2q_dense_same_support_motif`,
      - `phase2_xxz_hea_q4_continuity`,
      - `phase2_xxz_hea_q6_continuity`.

**Reasoning**

Story `P31-S06` explicitly closes only the **minimal second-slice correctness
loop**. Full Task 3 closure remains larger than this task.

### Step 5: Validate the minimal correctness gate and preserve slice continuity

**Goal**

Prove that the tightened second-slice evidence passes while the earlier slice
anchors remain intact.

**Execution checklist**

- [ ] Run:
      `conda run -n qgd python -m pytest tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py -q`
- [ ] Re-run:
      `conda run -n qgd python -m pytest tests/partitioning/test_partitioned_channel_native_phase31_slice.py tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py -q`
- [ ] If any fixture inventory file changed, run a small parity or iterator check
      against the benchmark mirror.

**Why this matters**

`P31-S06-E01` should be a narrow evidence-promotion task, not a regression on
the already-completed first slice or `P31-S05` public runtime work.

## Acceptance Evidence

`P31-S06-E01` is ready to hand off when all of the following are true:

- `phase31_microcase_2q_cnot_local_noise_pair` passes end-to-end against the
  sequential oracle with:
  - Frobenius difference `<= 1e-10`,
  - maximum absolute difference `<= 1e-10`,
  - `trace_deviation <= 1e-10`,
  - `rho_is_valid is True`,
  - and at least one passing representation-invariant check on the fused block,
- `phase31_local_support_q4_spectator_embedding_smoke` passes the same
  full-density exactness family while remaining explicitly non-counted,
- the second-slice module documents what remains deferred after this minimal
  correctness gate,
- the first-slice public regression file still passes unchanged,
- and no code changes are required in the broader `correctness_evidence` bundle
  builders just to close this slice.

## Handoff To The Next Engineering Tasks

After `P31-S06-E01` lands:

- `P31-S06-E02` should record second-slice completion and point cleanly to the
  remaining deferred counted IDs and broader Task 3 / Task 4 work.
- Later Task 3 work should decide when to wire
  `iter_phase31_correctness_microcase_cases()` and related Phase 3.1 builders
  into the active `correctness_evidence` package.
- External-reference rows (`P31-ADR-011`) and bundle/schema closure
  (`P31-ADR-013`) should reuse the same stable case IDs established here rather
  than inventing a parallel registry.

## Risks / Rollback

- Risk: the second-slice module may over-lock internal helper mechanics when
  adding invariant evidence.
  Rollback/mitigation: reuse the existing narrow helper stack already imported in
  the module and keep any new test-local helpers tiny and local.

- Risk: the 4-qubit smoke case may be mistaken for a counted correctness or
  continuity claim.
  Rollback/mitigation: keep its non-counted status explicit in module wording
  and avoid wiring it into broader Task 3 package builders here.

- Risk: touching `correctness_evidence` builders in this task would blur the line
  between the minimal second-slice gate and the remaining full Task 3 work.
  Rollback/mitigation: keep this task in deterministic pytest evidence unless a
  genuine blocker is exposed.
