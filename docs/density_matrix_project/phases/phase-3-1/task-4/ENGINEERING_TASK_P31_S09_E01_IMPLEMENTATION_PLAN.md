# Engineering Task P31-S09-E01 Implementation Plan

## Engineering Task Being Implemented

`Engineering Task P31-S09-E01: Add one representative hybrid structured benchmark pilot`

This is a Layer 4 file-level implementation plan for the first engineering task
under Story `P31-S09` from
`../THIRD_VERTICAL_SLICE_STRICT_PLUS_HYBRID_WHOLE_WORKLOAD_STORIES_AND_ENGINEERING_TASKS.md`.
It turns the third-slice Task 4 wording into a concrete plan against the
current hybrid-runtime and performance/evidence surfaces in:

- `task-4/TASK_4_MINI_SPEC.md`,
- `tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py`,
- `benchmarks/density_matrix/performance_evidence/case_selection.py`,
- `benchmarks/density_matrix/performance_evidence/records.py`,
- `benchmarks/density_matrix/performance_evidence/common.py`,
- and, for scope control only,
  `tests/partitioning/evidence/test_performance_evidence.py`,
  `benchmarks/density_matrix/performance_evidence/benchmark_matrix_validation.py`,
  and `benchmarks/density_matrix/performance_evidence/metric_surface_validation.py`.

It should also introduce one narrow pilot-specific validation surface rather
than reopening the full performance package, for example:

- `benchmarks/density_matrix/performance_evidence/phase31_hybrid_pilot_validation.py`,
- and a matching focused test module
  `tests/partitioning/evidence/test_phase31_hybrid_pilot_validation.py`.

## Scope

This engineering task promotes the already-landed hybrid runtime and counted
`q4` continuity gate into the **minimum Task 4 pilot benchmark** for the third
slice:

- freeze the representative primary-family pilot ID
  `phase31_pair_repeat_q8_periodic_seed20260318` (documented replacement: the
  original dense row routed zero channel-native partitions, so it was not a true
  mixed-route hybrid pilot),
- reuse the explicit whole-workload hybrid runtime
  `execute_partitioned_density_channel_native_hybrid(...)`,
- record the **baseline trio** on one reproducible pilot row:
  - sequential density reference,
  - Phase 3 partitioned+fused,
  - hybrid Phase 3.1 channel-native,
- record hybrid route coverage on that same pilot row using the additive
  partition attribution from `P31-S07-E01`,
- emit one initial diagnosis tag and one `decision_class` using the frozen
  `P31-ADR-010` / `P31-ADR-013` vocabulary,
- keep the existing structured hybrid smoke as a runtime sanity anchor rather
  than a substitute for benchmark evidence,
- and record explicitly what remains deferred after this pilot row:
  - the remaining counted primary-family rows,
  - the control-family closure,
  - and the full `break_even_table` / `justification_map` matrix.

Out of scope for this engineering task:

- wiring the full Phase 3.1 counted performance matrix into the active
  `performance_evidence` package builders,
- replacing the current 34-case Phase 3 performance inventory,
- full `P31-C-06` closure across all `26` counted Phase 3.1 performance rows,
- full `break_even_table` / `justification_map` closure across the whole frozen
  matrix,
- new host-parallel / SIMD / Task 6 build variants,
- changing the strict runtime or the shipped Phase 3 fused runtime,
- and new planner workload families or new structured workload IDs.

## Current Runtime And Evidence Gap To Close

`P31-S07-E01` and `P31-S08-E01` made the hybrid whole-workload path real and
counted it once on `q4` continuity, but the current performance surfaces still
stop at the old Phase 3 package design.

### `tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py`

The current hybrid test module already covers:

- one representative structured hybrid smoke on
  `phase31_pair_repeat_q8_dense_seed20260318`,
- exactness against the sequential oracle on that smoke case,
- and frozen route-reason vocabulary checks.

However, that file still stops short of Story `P31-S09`:

- it has **no timing surface**,
- it does **not** record the baseline trio,
- it does **not** aggregate route coverage into performance-facing counters,
- it does **not** emit a diagnosis tag or `decision_class`,
- and it is correctly framed as runtime / correctness evidence rather than as a
  performance pilot bundle.

That makes it a useful regression anchor, but **not** the primary implementation
surface for `P31-S09-E01`.

### `benchmarks/density_matrix/performance_evidence/case_selection.py`

This file already contains two important pieces:

- the active Phase 3 performance inventory and context builders:
  - `build_performance_evidence_inventory_cases()`,
  - `build_performance_evidence_case_contexts()`,
- and the **planning helper** Phase 3.1 structured inventory:
  - `build_phase31_performance_inventory_cases()`,
  - `iter_phase31_performance_cases()`.

The key gap is that the active builder still returns only:

- Phase 3 continuity cases,
- and the older Phase 3 structured matrix,

while the Phase 3.1 counted structured families remain planning helpers only.

That means this file is the natural place to add one narrow helper for the
pilot case, but **not** the place to migrate the whole package yet.

### `benchmarks/density_matrix/performance_evidence/records.py`

This file already contains the active timing and record-shaping logic, but it is
still Phase 3-shaped:

- `_measure_review_timings(...)` records only:
  - sequential timings,
  - and Phase 3 fused timings,
- `build_performance_evidence_core_benchmark_record(...)` bridges correctness
  through:
  - the correctness-evidence reference index, or
  - `execute_fused_with_reference(...)`,
- and the active benchmark record has no place for:
  - the hybrid Phase 3.1 timing baseline,
  - hybrid route-coverage counters,
  - `decision_class`,
  - or a pilot-specific diagnosis tag.

This file is therefore the natural place to add a **dedicated pilot record
builder** that reuses existing helpers without rewriting the active full-package
record path.

### Active performance validation tests are deliberately Phase 3-shaped

`tests/partitioning/evidence/test_performance_evidence.py` currently freezes the
active package surface:

- `34` total cases,
- `4` continuity + `30` structured rows,
- representative review surfaces based on the older structured families,
- and repeated-timing fields for sequential vs Phase 3 fused only.

Those tests are valuable regression anchors for the existing package, but they
should **not** be repurposed as the primary evidence surface for this one pilot
task.

So `P31-S09-E01` should prefer a **new narrow pilot validation slice** rather
than reopening the existing full-package tests.

## Dependencies And Assumptions

- The source-of-truth contract remains:
  - `TASK_4_MINI_SPEC.md`,
  - `../THIRD_VERTICAL_SLICE_STRICT_PLUS_HYBRID_WHOLE_WORKLOAD_STORIES_AND_ENGINEERING_TASKS.md`,
  - `../DETAILED_PLANNING_PHASE_3_1.md`,
  - `../ADRs_PHASE_3_1.md`
    (`P31-ADR-010`, `P31-ADR-012`, `P31-ADR-013`, `P31-ADR-014`).
- `P31-S07-E01` is assumed complete:
  - explicit hybrid runtime helper,
  - explicit hybrid runtime-path label,
  - additive partition-route attribution,
  - and the representative structured hybrid smoke case already executes.
- `P31-S08-E01` is assumed complete:
  - the counted `q4` hybrid continuity gate is green,
  - hybrid route summary is reviewable,
  - and the third-slice correctness loop is closed narrowly enough to support a
    pilot performance row.
- The frozen pilot ID is:
  - `phase31_pair_repeat_q8_periodic_seed20260318`.
- If implementation order forces a replacement pilot ID, the replacement must:
  - come from the frozen primary-family inventory,
  - remain a hybrid whole-workload case,
  - and be documented in the story doc and this plan in the same change.
- The pilot timing mode should reuse the existing repeated-timing policy from
  `performance_evidence.common`:
  - `PERFORMANCE_EVIDENCE_REPETITIONS = 3`,
  - `timing_mode = "median_3"`.
- Exactness guards for the pilot row should reuse
  `build_runtime_correctness_bridge_fields(...)` where practical rather than
  inventing a new performance-only correctness schema.
- Route coverage should derive from the already-landed hybrid partition
  metadata:
  - `partition_runtime_class`,
  - `partition_route_reason`,
  - and the descriptor partition member counts.

## Target Files And Responsibilities

### Primary case-selection file: `benchmarks/density_matrix/performance_evidence/case_selection.py`

This file should provide the narrow frozen pilot context without reopening the
active full-package matrix.

#### Recommended helper to add

Add one explicit pilot-context helper, for example:

```python
def build_phase31_hybrid_pilot_case_context(
    *,
    workload_id: str = "phase31_pair_repeat_q8_periodic_seed20260318",
) -> PerformanceEvidenceCaseContext:
    ...
```

#### What it should do

- Freeze the pilot ID by default:
  - family `phase31_pair_repeat`,
  - `qbit_num = 8`,
  - `noise_pattern = "periodic"`,
  - `seed = 20260318`.
- Reuse `build_phase31_structured_descriptor_set(...)`.
- Reuse the existing metadata contract from `_base_metadata_from_descriptor(...)`.
- Add the already-frozen Phase 3.1 counted metadata needed by `P31-ADR-013`:
  - `claim_surface_id = "phase31_bounded_mixed_motif_v1"`,
  - `representation_primary = "kraus_bundle"`,
  - `contains_noise = True`,
  - `counted_phase31_case = True`.

#### Keep unchanged where possible

- `build_performance_evidence_case_contexts()`,
- `build_performance_evidence_inventory_cases()`,
- and the active 34-case Phase 3 package inventory.

### Primary record file: `benchmarks/density_matrix/performance_evidence/records.py`

This file should carry the pilot-row measurement and record shaping.

#### Recommended helpers to add

Prefer a narrow pilot-only helper stack, for example:

```python
def _measure_phase31_hybrid_pilot_timings(case_context) -> dict:
    ...

def _hybrid_route_coverage(runtime_result, descriptor_set) -> dict[str, int]:
    ...

def build_phase31_hybrid_pilot_record(case_context) -> dict:
    ...
```

#### Baseline trio to record

The pilot record should time all three baselines:

- sequential density reference,
- Phase 3 partitioned+fused,
- Phase 3.1 hybrid channel-native.

Preferred repeated fields:

- `sequential_runtime_ms_samples`,
- `phase3_fused_runtime_ms_samples`,
- `phase31_hybrid_runtime_ms_samples`,
- and matching peak-RSS sample lists plus median fields.

#### Correctness guard to keep explicit

Even though this is a performance pilot, the row should still keep a semantic
guardrail:

- one exactness comparison for the hybrid result against the sequential oracle,
- and preferably one exactness comparison for the Phase 3 fused result as well,
- using the existing shared bridge helper
  `build_runtime_correctness_bridge_fields(...)` where practical.

This keeps the pilot row scientifically interpretable without pretending full
Task 3 package migration is already done.

#### Route coverage to emit

The pilot row should aggregate the required hybrid route-coverage counters from
the current partition records:

- `channel_native_partition_count`,
- `phase3_routed_partition_count`,
- `channel_native_member_count`,
- `phase3_routed_member_count`,
- and the raw partition-level route records for auditability.

Recommended class mapping:

- `phase31_channel_native` counts as channel-native coverage,
- `phase3_unitary_island_fused` and `phase3_supported_unfused` count as
  Phase 3 routed coverage.

#### Decision artifact fields to emit

The pilot row should emit:

- `decision_class` from the frozen vocabulary:
  - `phase3_sufficient`,
  - `phase31_justified`,
  - `phase31_not_justified_yet`,
- and one initial `diagnosis_tag`.

#### Recommended minimal `diagnosis_tag` vocabulary

Keep the first pilot vocabulary intentionally small and measurable, for example:

- `phase31_positive_gain`,
- `limited_channel_native_coverage`,
- `hybrid_overhead_dominant`.

Recommended deterministic mapping:

- if hybrid beats the Phase 3 fused baseline by wall-clock speedup
  (hybrid vs fused median runtime, ratio \(\ge 1.2\); peak-RSS deltas are **not**
  used here because `ru_maxrss` is process-wide and order-biased in repeated
  in-process runs):
  - `diagnosis_tag = "phase31_positive_gain"`,
  - `decision_class = "phase31_justified"`,
- elif channel-native coverage is effectively absent:
  - `diagnosis_tag = "limited_channel_native_coverage"`,
  - `decision_class = "phase3_sufficient"`,
- else:
  - `diagnosis_tag = "hybrid_overhead_dominant"`,
  - `decision_class = "phase31_not_justified_yet"`.

This is strong enough for a one-row pilot without pretending the full
`break_even_table` / `justification_map` already exists.

### Recommended new validation module:
`benchmarks/density_matrix/performance_evidence/phase31_hybrid_pilot_validation.py`

This should become the primary pilot artifact surface for `P31-S09-E01`.

#### What it should do

- build exactly one pilot case,
- materialize one benchmark record for that case,
- assemble a small artifact bundle and summary,
- and write it through the same validation scaffold style used by the existing
  performance-evidence scripts.

#### Recommended bundle summary

At minimum, summarize:

- `total_cases`,
- `pilot_case_name`,
- `timing_mode`,
- `decision_class`,
- `diagnosis_tag`,
- `channel_native_partition_count`,
- `phase3_routed_partition_count`.

#### Why a new file

- It keeps the active Phase 3 performance package honest and unchanged.
- It gives the Phase 3.1 pilot one focused, reviewable home.
- It lets later Task 4 work decide deliberately when to merge the pilot into the
  broader matrix and summary surfaces.

### Recommended new test module:
`tests/partitioning/evidence/test_phase31_hybrid_pilot_validation.py`

This should become the focused regression surface for `P31-S09-E01`.

#### What it should cover

- the pilot case ID is frozen,
- the baseline trio fields are present and comparable,
- route coverage fields are present and non-negative,
- `decision_class` is from the frozen vocabulary,
- `diagnosis_tag` is present,
- and the pilot bundle summary is internally consistent.

### Stable regression anchors to keep unchanged

- `tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py`
  should remain the runtime / correctness smoke anchor,
- `tests/partitioning/evidence/test_performance_evidence.py`
  should remain the active Phase 3 package regression anchor,
- `benchmarks/density_matrix/performance_evidence/benchmark_matrix_validation.py`
  and
  `benchmarks/density_matrix/performance_evidence/metric_surface_validation.py`
  should remain unchanged in this task unless a genuine shared-helper edit forces
  a tiny compatibility patch.

## Implementation Sequence

### Step 1: Freeze one representative pilot case without reopening the full matrix

**Goal**

Select one stable primary-family structured row that can carry the first hybrid
performance interpretation honestly.

**Execution checklist**

- [ ] Add one narrow pilot-context helper in
      `performance_evidence/case_selection.py`.
- [ ] Freeze the default pilot ID:
      `phase31_pair_repeat_q8_periodic_seed20260318`.
- [ ] Reuse the existing Phase 3.1 planning-helper metadata surface.
- [ ] Keep the active `build_performance_evidence_case_contexts()` unchanged.
- [ ] If a replacement pilot is required, document that replacement in:
      - this implementation plan,
      - and the third-slice story doc.

**Why first**

Without one explicit pilot case, the task will drift toward an unbounded partial
matrix migration.

### Step 2: Add one pilot benchmark record with the baseline trio and correctness guard

**Goal**

Turn the recommended pilot case into one comparable benchmark row with enough
semantic evidence to support later paper claims honestly.

**Execution checklist**

- [ ] Measure sequential timings with the existing sequential helper.
- [ ] Measure Phase 3 fused timings with `execute_partitioned_density_fused(...)`.
- [ ] Measure hybrid timings with
      `execute_partitioned_density_channel_native_hybrid(...)`.
- [ ] Use repeated runs and median aggregation:
      - `PERFORMANCE_EVIDENCE_REPETITIONS = 3`,
      - `timing_mode = "median_3"`.
- [ ] Record median runtime and peak-RSS values for all three baselines.
- [ ] Add one correctness guard against the sequential oracle for the hybrid
      result, and preferably for the Phase 3 fused result as well.

**Recommended output fields**

- `sequential_median_runtime_ms`,
- `phase3_fused_median_runtime_ms`,
- `phase31_hybrid_median_runtime_ms`,
- `sequential_median_peak_rss_kb`,
- `phase3_fused_median_peak_rss_kb`,
- `phase31_hybrid_median_peak_rss_kb`,
- `hybrid_vs_phase3_speedup`,
- optional informational peak-RSS deltas (not used to set `decision_class`).

**Reasoning**

Story `P31-S09` is about the first **whole-workload** performance object, so
the hybrid baseline must be measured directly, not inferred from the earlier
strict or fused paths.

### Step 3: Emit route coverage and a one-row decision artifact

**Goal**

Make the pilot row useful for later design decisions rather than only as a raw
timing sample.

**Execution checklist**

- [ ] Aggregate route coverage from the hybrid runtime partition records.
- [ ] Emit:
      - `channel_native_partition_count`,
      - `phase3_routed_partition_count`,
      - `channel_native_member_count`,
      - `phase3_routed_member_count`.
- [ ] Include the raw `partitions` route records in the pilot artifact.
- [ ] Emit one `decision_class` using the frozen `P31-ADR-010` vocabulary.
- [ ] Emit one initial `diagnosis_tag`.

**Recommended implementation constraint**

Do **not** attempt the full matrix-wide `break_even_table` in this task. Emit a
single-row pilot decision artifact using the same decision vocabulary instead.

**Why this matters**

Story `P31-S09` is valuable only if the pilot row explains **why** the richer
hybrid method did or did not help on that case.

### Step 4: Add a dedicated pilot validation slice instead of rewriting the active package

**Goal**

Land one reproducible pilot artifact without destabilizing the current Phase 3
benchmark package.

**Execution checklist**

- [ ] Add a dedicated pilot validation module under
      `benchmarks/density_matrix/performance_evidence/`.
- [ ] Add a focused pytest module for the pilot validation output.
- [ ] Keep the active full-package validation files unchanged unless a tiny
      shared-helper compatibility patch is required.
- [ ] Keep the current hybrid smoke test free of timing assertions.

**Reasoning**

The pilot row is intentionally **smaller** than full Task 4 closure, so it
should have a dedicated surface rather than being smuggled into the old matrix.

### Step 5: Validate the pilot and preserve slice boundaries

**Goal**

Prove that the pilot row is reproducible and that earlier runtime / package
surfaces remain intact.

**Execution checklist**

- [ ] Run the focused pilot pytest module, for example:
      `conda run -n qgd python -m pytest tests/partitioning/evidence/test_phase31_hybrid_pilot_validation.py -q`
- [ ] Re-run the hybrid runtime smoke module:
      `conda run -n qgd python -m pytest tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py -q`
- [ ] If shared performance helpers changed, re-run:
      `conda run -n qgd python -m pytest tests/partitioning/evidence/test_performance_evidence.py -q`
- [ ] If the pilot validation module writes an artifact bundle, verify the
      summary fields and stable pilot case ID.

**Why this matters**

`P31-S09-E01` should be a narrow benchmark-pilot task, not a regression on the
already-landed hybrid runtime or on the existing Phase 3 performance package.

## Acceptance Evidence

`P31-S09-E01` is ready to hand off when all of the following are true:

- one benchmark row exists for
  `phase31_pair_repeat_q8_periodic_seed20260318` (or a further documented
  replacement frozen pilot ID),
- that row records the baseline trio:
  - sequential,
  - Phase 3 fused,
  - Phase 3.1 hybrid,
- the row includes repeated-timing samples and medians for the three baselines,
- the row includes hybrid route coverage:
  - `channel_native_partition_count`,
  - `phase3_routed_partition_count`,
  - `channel_native_member_count`,
  - `phase3_routed_member_count`,
- the row carries one `decision_class` from:
  - `phase3_sufficient`,
  - `phase31_justified`,
  - `phase31_not_justified_yet`,
- the row carries one non-empty `diagnosis_tag`,
- the pilot artifact or companion note names the remaining deferred Task 4 work:
  - the remaining counted primary-family rows,
  - the control-family closure,
  - the full `break_even_table` / `justification_map`,
- the hybrid runtime smoke still passes unchanged,
- and the active full-package Phase 3 performance tests still pass unchanged if
  shared helpers were touched.

## Handoff To The Next Engineering Tasks

After `P31-S09-E01` lands:

- later Task 4 work should decide when to wire:
  - `iter_phase31_performance_cases()`,
  - the hybrid baseline,
  - route-coverage aggregation,
  - and the `decision_class` fields
  into the active `performance_evidence` package builders,
- the one-row pilot decision artifact should expand into the full
  `break_even_table` / `justification_map` across the frozen 26-case matrix,
- control-family closure should stay explicit rather than being inferred from one
  primary-family pilot row,
- and any Task 6 build variants should layer on top of the same pilot-row
  vocabulary rather than inventing a second diagnosis surface.

## Risks / Rollback

- Risk: the pilot row is mistaken for full Task 4 closure.
  Rollback/mitigation: keep the dedicated pilot validation surface separate from
  the active full-package matrix and name the deferred matrix work explicitly.

- Risk: touching the active `performance_evidence` builders accidentally rewrites
  the current Phase 3 benchmark package.
  Rollback/mitigation: add pilot-only helpers and a pilot-only validation module
  instead of replacing `build_performance_evidence_case_contexts()` or the
  existing record builders.

- Risk: one pilot row may be too noisy to support a stable diagnosis tag.
  Rollback/mitigation: reuse the existing median-of-3 timing policy and keep the
  first diagnosis vocabulary deliberately small and deterministic.

- Risk: future partitioning changes may alter route coverage on the pilot row and
  tempt silent weakening of the benchmark interpretation.
  Rollback/mitigation: if the pilot route coverage changes intentionally, update
  the pilot validation expectations and the third-slice docs in the same change.
