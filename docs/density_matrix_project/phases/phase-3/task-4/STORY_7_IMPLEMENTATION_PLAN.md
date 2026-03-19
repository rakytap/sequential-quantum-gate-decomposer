# Story 7 Implementation Plan

## Story Being Implemented

Story 7: Representative Structured Fused Runs Close The Phase 3
Threshold-Or-Diagnosis Rule

This is a Layer 4 engineering plan for implementing the seventh behavioral slice
from `TASK_4_STORIES.md`.

## Scope

This story turns Task 4 fused execution into the phase-level performance evidence
surface required by the Phase 3 contract:

- representative required 8 and 10 qubit structured workloads with real fused
  coverage are benchmarked through one stable evidence package,
- runtime, memory, partition context, and fused-coverage metrics are recorded
  alongside correctness-preserving outcomes,
- the benchmark package closes the phase-level threshold-or-diagnosis rule by
  either showing measurable benefit or by producing a benchmark-grounded
  diagnosis,
- and Story 7 closes Task 4 benchmark interpretation without reopening deferred
  channel-native fusion or broader Phase 4 growth.

Out of scope for this story:

- eligibility definition already owned by Story 1,
- the first positive structured fused-runtime slice already owned by Story 2,
- shared fused-capable reuse already owned by Story 3,
- positive exact semantic-preservation closure already owned by Story 4,
- explicit fused classification closure already owned by Story 5,
- and stable fused output and provenance packaging already owned by Story 6.

## Dependencies And Assumptions

- Stories 1 through 6 already define the eligibility, execution, reuse,
  semantics, classification, and output surfaces that Story 7 must benchmark
  honestly.
- The frozen source-of-truth contract is `TASK_4_MINI_SPEC.md`,
  `TASK_4_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-005`,
  `P3-ADR-009`, `P3-ADR-010`, and the performance-claim-boundary item in
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- The required threshold-or-diagnosis rule remains:
  - either at least one representative required 8 or 10 qubit structured case
    with real fused coverage shows median wall-clock speedup `>= 1.2x` or
    peak-memory reduction `>= 15%` versus the sequential baseline without
    correctness loss,
  - or the benchmark package plus profiling evidence explicitly explains why the
    native fused baseline does not yet accelerate that case and justifies the
    follow-on architecture decision gate.
- Task 3 already records the core runtime metrics Story 7 should reuse through
  `NoisyRuntimeExecutionResult`, including `runtime_ms`, `peak_rss_kb`,
  partition count, max partition span, and exact-output summaries.
- The existing structured workload builders Story 7 should reuse already exist in
  `benchmarks/density_matrix/planner_surface/workloads.py`, especially
  `iter_story2_structured_descriptor_sets()`, `STRUCTURED_FAMILY_NAMES`,
  `STRUCTURED_QUBITS`, and `MANDATORY_NOISE_PATTERNS`.
- Story 7 should treat the existing Task 3 mandatory-workload bundle under
  `benchmarks/density_matrix/artifacts/phase3_task3/story2_workloads/` as the
  plain partitioned baseline evidence surface it must extend rather than
  replace.
- Profiling artifacts are required only when they materially support the
  diagnosis path of the rule; Story 7 should not make profiling a blanket
  requirement for already clear wins.
- Fully channel-native fused noisy blocks remain explicitly deferred beyond the
  minimum Phase 3 contract.

## Engineering Tasks

### Engineering Task 1: Freeze The Representative Structured Benchmark Inventory And Decision Rule

**Implements story**
- `Story 7: Representative Structured Fused Runs Close The Phase 3 Threshold-Or-Diagnosis Rule`

**Change type**
- docs | validation automation

**Definition of done**
- Story 7 defines the exact representative structured benchmark inventory it
  owns.
- Story 7 defines the threshold-or-diagnosis rule in one explicit benchmark
  review surface.
- The inventory stays tied to the required structured Phase 3 workload matrix.

**Execution checklist**
- [ ] Freeze the required Story 7 benchmark inventory around representative 8
      and 10 qubit structured workloads with real fused coverage.
- [ ] Define which family, size, noise pattern, and seed combinations are the
      representative review set.
- [ ] Freeze the threshold-or-diagnosis decision rule in one benchmark-facing
      contract description.
- [ ] Keep channel-native follow-on work and broader Phase 4 growth outside the
      Story 7 bar.

**Evidence produced**
- One stable Story 7 structured benchmark inventory.
- One explicit threshold-or-diagnosis review rule for Task 4.

**Risks / rollback**
- Risk: Story 7 may drift into a broad benchmark zoo that weakens the phase
  claim instead of clarifying it.
- Rollback/mitigation: freeze a representative benchmark inventory tied to the
  required structured matrix.

### Engineering Task 2: Reuse The Shared Structured Workload Builders And Baseline Runtime Surface

**Implements story**
- `Story 7: Representative Structured Fused Runs Close The Phase 3 Threshold-Or-Diagnosis Rule`

**Change type**
- docs | code

**Definition of done**
- Story 7 reuses the existing structured workload builders and the Task 3 plain
  partitioned baseline surface.
- Benchmark comparisons stay anchored to the supported descriptor and runtime
  contracts.
- Story 7 avoids inventing a second benchmark workload language.

**Execution checklist**
- [ ] Reuse `iter_story2_structured_descriptor_sets()` or the smallest auditable
      successor as the benchmark workload source.
- [ ] Reuse the plain Task 3 partitioned runtime surface as the non-fused
      baseline where that comparison is needed.
- [ ] Keep workload identity, path labels, and metric naming aligned with the
      existing Task 3 benchmark surface.
- [ ] Document any Story 7-specific benchmark metadata explicitly rather than
      renaming shared fields.

**Evidence produced**
- One reviewable mapping from Task 3 structured workload evidence to Task 4
  fused benchmark evidence.
- One explicit baseline comparison rule for Story 7.

**Risks / rollback**
- Risk: benchmark comparisons may become hard to trust if Story 7 quietly uses a
  different workload or baseline vocabulary than Task 3.
- Rollback/mitigation: reuse the shared workload builders and baseline surface
  directly.

### Engineering Task 3: Build The Representative Fused Structured Benchmark Harness

**Implements story**
- `Story 7: Representative Structured Fused Runs Close The Phase 3 Threshold-Or-Diagnosis Rule`

**Change type**
- code | validation automation

**Definition of done**
- Story 7 has one benchmark harness for representative structured workloads with
  real fused coverage.
- The harness records runtime, memory, partition context, fused coverage, and
  correctness-preserving outputs.
- The harness is reusable across both the threshold and diagnosis paths.

**Execution checklist**
- [ ] Add a Story 7 benchmark harness under
      `benchmarks/density_matrix/partitioned_runtime/`, with
      `fused_performance_validation.py` as the primary driver.
- [ ] Record runtime, peak memory, partition count, max partition span, and
      fused-coverage summaries for each representative case.
- [ ] Record the plain Task 3 baseline metrics and the fused-path metrics in one
      comparable schema.
- [ ] Keep the harness rooted in the supported fused runtime and sequential
      reference surfaces rather than in synthetic kernel timing alone.

**Evidence produced**
- One reusable Story 7 fused structured benchmark harness.
- One comparable metric schema for fused and baseline runs.

**Risks / rollback**
- Risk: benchmark logic may become entangled with one-off scripts that are hard
  to rerun or audit later.
- Rollback/mitigation: centralize the representative benchmark harness in one
  stable validation entry point.

### Engineering Task 4: Record Correctness-Preserving Benchmark Metrics And Coverage

**Implements story**
- `Story 7: Representative Structured Fused Runs Close The Phase 3 Threshold-Or-Diagnosis Rule`

**Change type**
- code | tests | validation automation

**Definition of done**
- Story 7 benchmark cases record the metrics needed by the phase-level rule.
- Measurable benefit is interpreted only on correctness-preserving fused cases.
- The benchmark package is explicit about which cases had real fused coverage.

**Execution checklist**
- [ ] Record density and trace agreement metrics alongside runtime and memory
      metrics for representative fused benchmark cases.
- [ ] Record partition count or span context and fused-coverage summaries for
      each case.
- [ ] Ensure a benchmark case only counts toward the positive threshold path if
      it had real fused coverage and preserved correctness.
- [ ] Add focused checks proving required metric fields are always present.

**Evidence produced**
- One reviewable metric surface tying performance data to correctness-preserving
  fused coverage.
- Focused regression coverage for required Story 7 benchmark fields.

**Risks / rollback**
- Risk: benchmark results may appear stronger than they are if correctness or
  real fused coverage is not tied directly to the measured metrics.
- Rollback/mitigation: record correctness and fused coverage alongside every
  performance metric.

### Engineering Task 5: Add The Diagnosis Path With Explicit Profiling Or Bottleneck Evidence

**Implements story**
- `Story 7: Representative Structured Fused Runs Close The Phase 3 Threshold-Or-Diagnosis Rule`

**Change type**
- code | validation automation | docs

**Definition of done**
- Story 7 can close honestly even when the representative fused cases do not yet
  cross the positive threshold.
- The diagnosis path records benchmark-grounded reasons and, where material,
  profiling evidence for the remaining bottleneck.
- The diagnosis path feeds the deferred follow-on decision gate explicitly.

**Execution checklist**
- [ ] Add one diagnosis path in the Story 7 harness for representative cases
      that preserve correctness but do not meet the positive threshold.
- [ ] Record benchmark-grounded reasons such as limited fused coverage,
      persistent noise-boundary fragmentation, or remaining runtime bottlenecks.
- [ ] Add profiling artifact references only when they materially clarify the
      diagnosis.
- [ ] Document how a diagnosis result maps to the follow-on architecture
      decision gate rather than to hidden incompleteness.

**Evidence produced**
- One explicit Story 7 diagnosis path for non-winning representative cases.
- One structured explanation surface for benchmark-grounded follow-on decisions.

**Risks / rollback**
- Risk: negative or mixed performance results may be hidden or hand-waved instead
  of becoming useful scientific evidence.
- Rollback/mitigation: make the diagnosis path a first-class closure route.

### Engineering Task 6: Emit A Stable Story 7 Performance Bundle

**Implements story**
- `Story 7: Representative Structured Fused Runs Close The Phase 3 Threshold-Or-Diagnosis Rule`

**Change type**
- validation automation | docs

**Definition of done**
- Story 7 emits one stable machine-reviewable performance bundle or rerunnable
  checker.
- The bundle supports both the positive threshold path and the diagnosis path.
- The output is stable enough for later paper and review work to cite directly.

**Execution checklist**
- [ ] Add a dedicated Story 7 artifact location
      (for example `benchmarks/density_matrix/artifacts/phase3_task4/story7_performance/`).
- [ ] Emit representative structured benchmark cases through one stable schema.
- [ ] Record the threshold-or-diagnosis verdict explicitly in the bundle summary.
- [ ] Record rerun commands, software metadata, and any referenced profiling
      artifact locations with the emitted bundle.

**Evidence produced**
- One stable Story 7 performance bundle or checker.
- One direct citation surface for the Task 4 benchmark verdict.

**Risks / rollback**
- Risk: if Story 7 evidence remains scattered across notebooks or console logs,
  the phase-level claim boundary will be hard to audit later.
- Rollback/mitigation: emit one stable performance bundle and cite it directly.

### Engineering Task 7: Document The Follow-On Decision Gate And Run The Story 7 Benchmark Surface

**Implements story**
- `Story 7: Representative Structured Fused Runs Close The Phase 3 Threshold-Or-Diagnosis Rule`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the Story 7 threshold-or-diagnosis rule and the
  follow-on decision gate.
- The Story 7 benchmark harness and bundle run successfully.
- Deferred channel-native fusion remains documented as future work rather than as
  a hidden prerequisite.

**Execution checklist**
- [ ] Document the representative benchmark inventory and the threshold-or-
      diagnosis rule.
- [ ] Explain how positive-threshold and diagnosis-path outcomes should be
      interpreted in Phase 3.
- [ ] Run focused Story 7 regression coverage and verify
      `benchmarks/density_matrix/partitioned_runtime/fused_performance_validation.py`.
- [ ] Record stable references to the Story 7 tests, bundle, and any referenced
      profiler artifacts.

**Evidence produced**
- Passing Story 7 benchmark regression checks.
- One stable Story 7 performance-bundle or checker reference.

**Risks / rollback**
- Risk: later reviewers may mistake deferred channel-native fusion for an
  unfinished hidden prerequisite instead of an explicit benchmark-driven branch.
- Rollback/mitigation: document the decision gate and deferred boundary
  explicitly.

## Exit Criteria

Story 7 is complete only when all of the following are true:

- representative required 8 and 10 qubit structured workloads with real fused
  coverage are benchmarked through one stable evidence package,
- runtime, memory, partition context, fused coverage, and correctness-preserving
  metrics are recorded together,
- the performance bundle explicitly closes either the positive threshold path or
  the diagnosis path of the phase-level rule,
- one stable Story 7 performance bundle or checker exists for direct citation,
- and channel-native fused noisy blocks remain clearly documented as a deferred,
  benchmark-driven follow-on branch.

## Implementation Notes

- Prefer a small representative structured benchmark package over a large but
  weakly interpretable benchmark zoo.
- Treat diagnosis-quality negative or mixed results as valid scientific evidence
  when they are benchmark-grounded and explicit.
- Do not let Story 7 quietly expand the minimum Phase 3 architecture beyond the
  frozen baseline.
