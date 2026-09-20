# Engineering Task P31-S08-E01 Implementation Plan

## Engineering Task Being Implemented

`Engineering Task P31-S08-E01: Add the counted q4 hybrid continuity correctness gate`

This is a Layer 4 file-level implementation plan for the first engineering task
under Story `P31-S08` from
`../THIRD_VERTICAL_SLICE_STRICT_PLUS_HYBRID_WHOLE_WORKLOAD_STORIES_AND_ENGINEERING_TASKS.md`.
It turns the third-slice Task 3 wording into a concrete plan against the
current hybrid-runtime and correctness/evidence surfaces in:

- `task-3/TASK_3_MINI_SPEC.md`,
- `tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py`,
- `tests/partitioning/fixtures/runtime.py`,
- `tests/partitioning/fixtures/continuity.py`,
- `squander/partitioning/noisy_runtime_core.py`,
- `squander/partitioning/noisy_runtime.py`,
- and, for scope control only,
  `benchmarks/density_matrix/correctness_evidence/case_selection.py` plus
  `benchmarks/density_matrix/correctness_evidence/records.py`.

## Scope

This engineering task promotes the already-landed hybrid runtime from
`P31-S07-E01` into the **minimum counted Task 3 correctness gate** for the
third slice:

- keep the counted hybrid continuity ID `phase2_xxz_hea_q4_continuity`
  unchanged,
- reuse the explicit public helper
  `execute_partitioned_density_channel_native_hybrid(...)`,
- assert end-to-end exactness against the sequential reference under the frozen
  numeric policy from `P31-C-03`,
- freeze one reviewable hybrid route summary for the counted `q4` anchor using
  the additive per-partition attribution from `P31-S07-E01`,
- keep the negative unsupported-by-both hybrid case as the no-silent-fallback
  regression anchor,
- keep the representative structured hybrid smoke as non-claim-bearing support
  evidence for the next performance-design loop,
- and record explicitly what remains deferred after this counted continuity
  gate:
  - `phase2_xxz_hea_q6_continuity`,
  - Aer on the frozen external slice,
  - full `correctness_evidence` package migration.

Out of scope for this engineering task:

- `phase2_xxz_hea_q6_continuity`,
- full `P31-ADR-011` Aer closure,
- wiring `iter_phase31_correctness_continuity_cases()` into the active
  correctness-package builders,
- updating
  `benchmarks/density_matrix/correctness_evidence/records.py` to execute the
  hybrid runtime directly,
- schema version bumps or full `channel_invariants` package migration under
  `P31-ADR-013`,
- representative structured timing / diagnosis rows from `P31-S09-E01`,
- new runtime labels, new route reasons, or new public runtime result fields.

## Current Runtime And Evidence Gap To Close

`P31-S07-E01` already made the hybrid path real, but the current evidence still
stops short of the counted correctness interpretation promised by Story
`P31-S08`.

### `tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py`

The current hybrid module already covers:

- positive `q4` continuity execution through
  `phase31_channel_native_hybrid`,
- exactness against the sequential oracle via Frobenius distance,
  `trace_deviation`, and `rho_is_valid`,
- mixed routing through:
  - `phase31_channel_native`,
  - and a Phase 3 routed class,
- one unsupported-by-both negative case,
- and one representative structured hybrid smoke.

However, it still stops short of the Story `P31-S08` acceptance signals:

- the module docstring still frames the file primarily as the
  `P31-S07-E01` runtime / route-attribution surface rather than as the first
  counted hybrid correctness gate,
- the counted `q4` continuity test currently uses **existence** checks for route
  attribution (`any(...)`) rather than a stable aggregated route summary,
- the exactness idiom does not yet include `max_abs_diff`, which is optional for
  the story but preferred for consistency with earlier counted correctness
  slices,
- and the file does not yet name the deferred Task 3 items strongly enough that
  a reviewer can distinguish “first counted hybrid continuity gate” from “full
  Task 3 closure.”

That makes this file the natural primary target for `P31-S08-E01`.

### Stable helper surfaces already exist

Unlike `P31-S07-E01`, this task does **not** start from missing runtime or
fixture infrastructure:

- `execute_partitioned_density_channel_native_hybrid(...)` already exists,
- `PHASE31_RUNTIME_PATH_CHANNEL_NATIVE_HYBRID` already exists,
- `NoisyRuntimePartitionRecord` already exposes:
  - `partition_runtime_class`,
  - `partition_route_reason`,
- `build_phase3_continuity_partition_descriptor_set(...)` already builds the
  frozen `phase2_xxz_hea_q4_continuity` anchor,
- `build_phase2_continuity_vqe(4)` already provides deterministic continuity
  inputs,
- and `tests/partitioning/fixtures/runtime.py` already exposes:
  - `PHASE3_RUNTIME_DENSITY_TOL`,
  - `build_density_comparison_metrics(...)`,
  - `build_initial_parameters(...)`.

The preferred `P31-S08-E01` plan is therefore to **tighten and freeze the
pytest correctness gate**, not to redesign the runtime.

### The broader correctness-package builders are not ready to carry this slice

The wider correctness-evidence harness is already aware of the **planned**
Phase 3.1 continuity slice, but it is not yet the right claim-bearing surface
for this task:

- `benchmarks/density_matrix/correctness_evidence/case_selection.py` already has
  `iter_phase31_correctness_continuity_cases()` for `q4` / `q6`,
- but the active `build_correctness_evidence_case_contexts()` still returns the
  old Phase 3 continuity/microcase/structured matrix,
- and `benchmarks/density_matrix/correctness_evidence/records.py` still builds
  positive records through `execute_fused_with_reference(...)`, not through the
  explicit hybrid Phase 3.1 path.

So `P31-S08-E01` should stay in **deterministic pytest evidence** and leave
bundle / artifact migration to later Task 3 work.

## Dependencies And Assumptions

- The source-of-truth contract remains:
  - `TASK_3_MINI_SPEC.md`,
  - `../THIRD_VERTICAL_SLICE_STRICT_PLUS_HYBRID_WHOLE_WORKLOAD_STORIES_AND_ENGINEERING_TASKS.md`,
  - `../DETAILED_PLANNING_PHASE_3_1.md`,
  - `../ADRs_PHASE_3_1.md`
    (`P31-ADR-009`, `P31-ADR-011`, `P31-ADR-012`, `P31-ADR-013`).
- `P31-S07-E01` is assumed complete:
  - explicit hybrid runtime helper,
  - explicit hybrid runtime-path label,
  - additive partition-route attribution,
  - one unsupported-by-both negative hybrid case,
  - and one representative structured hybrid smoke.
- The counted-vs-deferred distinction for this task lives in **test/module/doc
  intent**, not in a new runtime metadata field:
  - counted case:
    `phase2_xxz_hea_q4_continuity`,
  - deferred continuity:
    `phase2_xxz_hea_q6_continuity`.
- Route-summary stability should prefer **aggregate counts** over fragile
  partition-index assertions.
- The current landed `q4` hybrid route summary, observed against the existing
  descriptor partitioning, is:
  - `partition_count == 5`,
  - runtime-class counts:
    - `phase31_channel_native: 2`,
    - `phase3_unitary_island_fused: 3`,
  - route-reason counts:
    - `eligible_channel_native_motif: 2`,
    - `pure_unitary_partition: 3`.
- If future planner or partitioning work changes that summary intentionally, the
  expected route summary should be updated **deliberately** in the same change
  rather than weakened into an existence-only check.

## Target Files And Responsibilities

### Primary evidence file: `tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py`

This file should become the third-slice **minimal counted hybrid correctness
gate** while preserving the earlier `P31-S07-E01` boundary checks already
landed there.

#### What this file should cover after `P31-S08-E01`

- a clearly labeled public correctness subsection for Story `P31-S08`,
- the counted hybrid continuity anchor `phase2_xxz_hea_q4_continuity`,
- the frozen exactness family on the final density state:
  - Frobenius difference,
  - maximum absolute difference,
  - trace deviation,
  - density validity,
- a stable hybrid route summary for the counted `q4` anchor,
- the existing unsupported-by-both negative case as a no-silent-fallback
  regression anchor,
- and the structured `q8` hybrid smoke as non-counted support evidence for the
  later performance-design slice.

#### Recommended helper to add

Prefer one tiny **test-local** route-summary helper rather than a new production
API, for example:

```python
def _hybrid_partition_route_summary(result) -> dict[str, dict[str, int]]:
    ...
```

The preferred summary shape is:

- `partition_count`,
- `runtime_class_counts`,
- `route_reason_counts`.

That is strong enough to support reviewer-facing route stability without
forcing `P31-S08-E01` to invent a second runtime-output schema.

#### Recommended current frozen summary for the counted `q4` anchor

Given the currently landed hybrid runtime and continuity descriptor surface, the
preferred first counted summary to freeze is:

- `partition_count == 5`,
- `runtime_class_counts`:
  - `phase31_channel_native: 2`,
  - `phase3_unitary_island_fused: 3`,
- `route_reason_counts`:
  - `eligible_channel_native_motif: 2`,
  - `pure_unitary_partition: 3`.

Do **not** assert raw partition indices unless a later audit task explicitly
requires index-level stability.

### Stable helper file: `tests/partitioning/fixtures/runtime.py`

This file should remain unchanged unless the hybrid test module needs one tiny
reusable helper for route-summary comparison.

#### Keep stable if possible

- `PHASE3_RUNTIME_DENSITY_TOL`,
- `build_density_comparison_metrics(...)`,
- `build_initial_parameters(...)`.

### Stable continuity fixture file: `tests/partitioning/fixtures/continuity.py`

This file should remain unchanged.

#### Reuse

- `build_phase2_continuity_vqe(4)` for the counted `q4` anchor.

### Stable runtime files: `squander/partitioning/noisy_runtime_core.py` and `squander/partitioning/noisy_runtime.py`

The preferred `P31-S08-E01` plan is **no semantic change** in these files.

#### Keep stable

- `execute_partitioned_density_channel_native_hybrid(...)`,
- `PHASE31_RUNTIME_PATH_CHANNEL_NATIVE_HYBRID`,
- `NoisyRuntimePartitionRecord.partition_runtime_class`,
- `NoisyRuntimePartitionRecord.partition_route_reason`.

#### Avoid in this task

- adding an aggregated route-summary field to `NoisyRuntimeExecutionResult`,
- adding a new hybrid correctness runtime label,
- changing the strict runtime path or the shipped Phase 3 path,
- reopening `requested_runtime_path` or runtime downgrade semantics.

### Explicitly out-of-scope bundle files

These files are important for the broader Task 3 package, but the preferred
`P31-S08-E01` plan is **not** to edit them:

- `benchmarks/density_matrix/correctness_evidence/case_selection.py`,
- `benchmarks/density_matrix/correctness_evidence/records.py`,
- related bundle / validation scripts under
  `benchmarks/density_matrix/correctness_evidence/`.

That broader correctness-package migration belongs to later Task 3 work after
the counted hybrid continuity pytest gate is frozen.

## Implementation Sequence

### Step 1: Recast the hybrid test module as the first counted hybrid correctness gate

**Goal**

Turn the current hybrid module from “runtime split + route smoke” into “runtime
split + route smoke + first counted hybrid continuity correctness gate”.

**Execution checklist**

- [ ] Update the module docstring in
      `test_partitioned_channel_native_phase31_hybrid_slice.py`.
- [ ] Add a clearly labeled correctness subsection for Story `P31-S08`.
- [ ] Name the remaining deferred Task 3 items explicitly:
      - `phase2_xxz_hea_q6_continuity`,
      - Aer on the frozen external slice,
      - full `correctness_evidence` package migration.
- [ ] Keep the unsupported-by-both negative case and the structured smoke case
      explicitly distinguished from the counted `q4` correctness gate.

**Why first**

This task is primarily about **evidence shape and claim boundary**. The module
should describe the slice honestly before the assertions are tightened.

### Step 2: Promote `phase2_xxz_hea_q4_continuity` to explicit end-to-end hybrid correctness evidence

**Goal**

Make the counted `q4` continuity anchor the minimal claim-bearing whole-workload
proof for the third-slice hybrid interpretation.

**Execution checklist**

- [ ] Keep the counted case ID string unchanged:
      `phase2_xxz_hea_q4_continuity`.
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
- [ ] Keep the current mixed-path interpretation explicit:
      - at least one channel-native-routed partition,
      - and at least one Phase 3 routed partition.

**Recommended exactness style**

Prefer the same public exactness idiom already used by earlier counted pytest
gates:

- `frobenius_norm_diff`,
- `max_abs_diff`,
- `trace_deviation`,
- `rho_is_valid`.

That keeps the counted `q4` hybrid gate comparable to earlier exactness slices
without waiting for bundle migration.

### Step 3: Freeze a stable hybrid route summary without adding new runtime API

**Goal**

Turn the current existence-only route checks into a reviewer-friendly,
machine-checkable summary for the counted hybrid continuity anchor.

**Execution checklist**

- [ ] Add one tiny test-local route-summary helper driven from
      `result.partitions`.
- [ ] Aggregate:
      - partition count,
      - runtime-class counts,
      - route-reason counts.
- [ ] Assert the current frozen `q4` summary:
      - `partition_count == 5`,
      - `phase31_channel_native == 2`,
      - `phase3_unitary_island_fused == 3`,
      - `eligible_channel_native_motif == 2`,
      - `pure_unitary_partition == 3`.
- [ ] Keep route-summary assertions aggregated rather than index-based.

**Recommended implementation constraint**

Do **not** add a new production summary builder just for this task. The current
additive hybrid partition fields are enough for the first counted gate.

**Why this matters**

Story `P31-S08` needs more than “some mixed routing happened.” It needs a route
summary that stays reviewable as the evidence surface grows.

### Step 4: Keep the broader correctness-package migration explicitly deferred

**Goal**

Land the counted `q4` hybrid gate without pretending the full Task 3 artifact
surface is already migrated to the hybrid interpretation.

**Execution checklist**

- [ ] Do not wire `iter_phase31_correctness_continuity_cases()` into the active
      correctness-package builders in this task.
- [ ] Do not change
      `build_correctness_evidence_positive_record(...)`
      to call the hybrid runtime in this task.
- [ ] Do not add Aer rows, schema version bumps, or `q6` continuity closure in
      this task.
- [ ] Keep the reviewer-facing deferred-work note visible in the hybrid test
      module and in this implementation plan.

**Reasoning**

`P31-S08-E01` closes the **first counted hybrid continuity loop**, not the full
Task 3 evidence-package migration.

### Step 5: Validate the counted hybrid gate and preserve slice continuity

**Goal**

Prove that the tightened hybrid correctness evidence passes while earlier strict
and hybrid slice anchors remain intact.

**Execution checklist**

- [ ] Run:
      `conda run -n qgd python -m pytest tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py -q`
- [ ] Re-run:
      `conda run -n qgd python -m pytest tests/partitioning/test_partitioned_channel_native_phase31_slice.py tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py -q`
- [ ] If any shared runtime helper changed unexpectedly, re-run:
      `conda run -n qgd python -m pytest tests/partitioning/test_partitioned_runtime.py -q`
- [ ] Leave the broader correctness-evidence bundle scripts untouched unless a
      genuine blocker is exposed.

**Why this matters**

`P31-S08-E01` should be a narrow counted-evidence promotion task, not a
regression on the already-landed strict or hybrid runtime surfaces.

## Acceptance Evidence

`P31-S08-E01` is ready to hand off when all of the following are true:

- `phase2_xxz_hea_q4_continuity` passes end-to-end through the hybrid path
  against the sequential oracle with:
  - Frobenius difference `<= 1e-10`,
  - maximum absolute difference `<= 1e-10`,
  - `trace_deviation <= 1e-10`,
  - `rho_is_valid is True`,
- the counted `q4` continuity route summary is frozen and reviewable:
  - `partition_count == 5`,
  - `phase31_channel_native == 2`,
  - `phase3_unitary_island_fused == 3`,
  - `eligible_channel_native_motif == 2`,
  - `pure_unitary_partition == 3`,
- the hybrid module documents what remains deferred after this first counted
  hybrid gate:
  - `phase2_xxz_hea_q6_continuity`,
  - Aer on the frozen external slice,
  - full `correctness_evidence` package migration,
- the unsupported-by-both hybrid negative case still fails loudly,
- the strict first-slice and second-slice regression files still pass unchanged,
- and no changes are required in the active correctness-evidence bundle builders
  just to close this slice.

## Handoff To The Next Engineering Tasks

After `P31-S08-E01` lands:

- `P31-S09-E01` should add one representative hybrid structured pilot row with
  baseline-trio timing, route coverage, and one diagnosis tag.
- Later Task 3 work should decide when to wire:
  - `iter_phase31_correctness_continuity_cases()`,
  - the hybrid runtime path,
  - and the required `P31-ADR-013` Phase 3.1 fields
  into the active correctness-evidence package.
- External-reference rows from `P31-ADR-011` should reuse the same frozen `q4`
  continuity ID and the same hybrid route vocabulary rather than inventing a
  parallel registry.
- `phase2_xxz_hea_q6_continuity` should remain the next deferred correctness
  anchor after the `q4` gate is green.

## Risks / Rollback

- Risk: the counted `q4` route summary may over-lock harmless partition ordering
  details.
  Rollback/mitigation: assert aggregated counts by runtime class and route
  reason, not raw partition indices.

- Risk: `P31-S08-E01` may be mistaken for full Task 3 closure.
  Rollback/mitigation: keep the deferred `q6` / Aer / package-migration note
  explicit in the module docstring, acceptance evidence, and handoff section.

- Risk: touching the active correctness-evidence bundle builders in this task
  would blur the line between the first counted hybrid gate and the later full
  package migration.
  Rollback/mitigation: keep this task in deterministic pytest evidence unless a
  genuine blocker appears.

- Risk: future planner partitioning changes may legitimately alter the `q4`
  route summary and tempt a silent weakening of the test.
  Rollback/mitigation: if the route summary changes intentionally, update the
  frozen expected summary and the third-slice planning docs in the same change.
