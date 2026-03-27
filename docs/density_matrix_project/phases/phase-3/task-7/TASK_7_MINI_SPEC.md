# Task 7: Performance And Sensitivity Benchmark Package

**Implementation Status: COMPLETE**

This mini-spec defined the Phase 3 Task 7 implementation contract. Performance
and sensitivity benchmark evidence is implemented in
`benchmarks/density_matrix/performance_evidence/` with machine-checkable bundles
emitted under `benchmarks/density_matrix/artifacts/performance_evidence/`.
Performance closure was achieved through the diagnosis branch.

This document inherits the frozen phase decisions from `DETAILED_PLANNING_PHASE_3.md`,
`P3-ADR-003`, `P3-ADR-004`, `P3-ADR-005`, `P3-ADR-006`, `P3-ADR-007`,
`P3-ADR-008`, `P3-ADR-009`, and `P3-ADR-010`, plus the closed canonical
planner-surface, semantic-preservation, runtime-minimum, cost-model
sequencing, support-matrix, validation-baseline, benchmark-anchor, and
performance-claim-boundary items in
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. It does not reopen the supported
workload surface, the frozen numeric thresholds, the bounded Task 5 calibrated
planner claim, or the deferred channel-native / broader Phase 4 branches.

## Given / When / Then
- Given a supported `CanonicalNoisyPlannerSurface`, an auditable
  `NoisyPartitionDescriptorSet`, an executable partitioned density runtime with
  at least one real fused execution mode on representative workloads, a bounded
  benchmark-calibrated Task 5 planner claim, and a machine-reviewable Task 6
  correctness package with explicit unsupported-boundary evidence.
- When Phase 3 uses benchmark results to support a performance conclusion, a
  sensitivity conclusion, or a benchmark-grounded follow-on architecture
  decision for exact noisy mixed-state partitioning.
- Then the benchmark package must characterize runtime, memory, planning
  overhead, partition structure, and fused-coverage behavior on the frozen
  workload matrix, preserve stable per-case provenance and counted-status
  rules, measure the required sensitivity knobs, and close the Phase 3
  threshold-or-diagnosis rule without overstating universal acceleration.

## Assumptions and dependencies
- Task 1 provides the canonical Phase 3 planner-entry contract as a
  schema-versioned `CanonicalNoisyPlannerSurface` with stable provenance fields
  for `requested_mode`, `source_type`, and `workload_id`, plus explicit
  planner-entry unsupported evidence (entry route and workload family are
  implied by `source_type`, not stored on the surface).
- Task 2 provides the schema-versioned partition handoff contract through
  `NoisyPartitionDescriptorSet`, `NoisyPartitionDescriptor`, and
  `NoisyPartitionDescriptorMember`, including canonical operation references,
  exact gate/noise order, qubit-remapping metadata, and parameter-routing
  metadata.
- Task 3 provides the executable partitioned density runtime surface and the
  core runtime metrics Task 7 must benchmark honestly, including
  `NoisyRuntimeExecutionResult`, `runtime_ms`, `peak_rss_kb`, partition-count
  summaries, partition-span summaries, and stable runtime-path labels.
- Task 4 provides the minimum real fused-execution baseline, explicit fused /
  supported-but-unfused / deferred classification, and the benchmark harnesses
  under `benchmarks/density_matrix/partitioned_runtime/` that Task 7 may reuse
  or extend.
- Task 5 provides the benchmark-calibrated planner surface or bounded candidate
  family plus machine-reviewable calibration bundles or rerunnable checkers.
  Task 7 interprets runtime, memory, planning-overhead, and sensitivity
  behavior over that supported calibration surface rather than inventing a new
  planner claim.
- Task 6 provides the machine-reviewable correctness and unsupported-boundary
  package under `benchmarks/density_matrix/artifacts/correctness_evidence/`. Task 7
  depends on that package for counted-status rules, correctness thresholds, the
  required external slice, and explicit negative-boundary visibility.
- Task 8 depends on Task 7 by turning the representative benchmark package into
  the final Paper 2 evidence and documentation bundle. Task 7 therefore closes
  benchmark interpretation, but not the final publication narrative itself.
- The mandatory workload classes remain:
  - 2 to 4 qubit micro-validation circuits,
  - 4, 6, 8, and 10 qubit Phase 2 noisy XXZ `HEA` continuity cases,
  - and the mandatory 8 and 10 qubit structured noisy `U3` / `CNOT` families
    with stable seed rules and sparse, periodic, and dense local-noise
    patterns.
- The frozen required gate surface remains `U3` and `CNOT`. The frozen required
  noise surface remains local single-qubit depolarizing, local amplitude
  damping, and local phase damping or dephasing.
- Sequential `NoisyCircuit` execution remains the required internal exact
  baseline for every counted supported case. Qiskit Aer remains the required
  external exact reference on the mandatory 2 to 4 qubit microcases and the
  representative small continuity subset. They are validation baselines, not
  hidden benchmark fallback paths.
- The current delivered Task 4 finding is that representative structured 8- and
  10-qubit cases now exercise a real fused path, but the currently observed
  fused baseline closes the phase rule via the diagnosis branch rather than via
  the positive-threshold branch because supported islands remain partially
  unfused and the present Python-level fused path adds overhead.
- The current delivered Task 5 result narrows the supported planner claim to a
  bounded auditable `max_partition_qubits` span-budget family on the existing
  noisy planner surface. If winners inside that family remain close or
  rerun-sensitive, Task 7 reports that sensitivity explicitly rather than
  pretending a permanently frozen universal winner exists.
- The current delivered Task 6 package contains one shared positive record
  surface plus one shared negative-boundary surface, with `25` counted supported
  cases, `4` required external-reference cases, and `17` explicit negative
  boundary cases across planner-entry, descriptor-generation, and runtime-stage
  evidence. Task 7 must reuse those boundaries rather than reconstructing them
  informally in summary scripts.
- The phase-level performance rule remains:
  - either at least one required 8- or 10-qubit structured case shows median
    wall-clock speedup `>= 1.2x` or peak-memory reduction `>= 15%` versus the
    sequential baseline without correctness loss,
  - or the required benchmark package plus profiling evidence explicitly shows
    why the native Phase 3 baseline does not yet accelerate that case and
    justifies the follow-on architecture decision gate.
- Profiling artifacts are required only when they materially support the
  diagnosis branch or a follow-on architecture decision. They are not a blanket
  replacement for representative benchmark evidence.
- This task defines the minimum representative performance and sensitivity
  package for Phase 3. It does not widen the support matrix, replace the Task 6
  correctness gate, reopen the Task 5 planner claim, or by itself authorize
  deferred channel-native, approximate, or broader Phase 4 workflow growth.

## Required behavior
- Task 7 freezes the representative benchmark contract for Phase 3: one stable
  benchmark package or rerunnable checker must exist for the frozen noisy
  workload matrix and must be strong enough to support Paper 2 performance
  interpretation honestly.
- The benchmark package must preserve the dual-anchor structure frozen at the
  phase level:
  - the Phase 2 noisy XXZ `HEA` workflow remains the continuity anchor,
  - and the structured noisy `U3` / `CNOT` families remain the methods stress
    matrix.
- The continuity anchor may contribute context, reproducibility continuity, and
  workload comparability, but the phase-level positive performance threshold is
  judged on required representative 8- and 10-qubit structured cases with
  supported `partitioned_density` execution.
- Mandatory structured benchmark coverage must stay rooted in the frozen methods
  matrix:
  - 8 and 10 qubit required structured families,
  - at least `3` seed-fixed instances per mandatory structured family and size,
  - and sparse, periodic, and dense local-noise placement sensitivity.
- Task 7 must measure the benchmark dimensions frozen in Phase 3 planning. At
  minimum, for each counted representative case the package must preserve:
  - runtime,
  - peak memory,
  - planner runtime or planning overhead,
  - partition count and qubit-span summaries,
  - runtime-path classification,
  - fused-path coverage and supported-but-unfused or deferred-region counts
    where applicable,
  - and correctness-gate status joined from the shared Task 6 evidence surface.
- Performance claims may count only correctness-preserving supported cases. A
  case that fails the frozen Task 6 exactness thresholds, lacks a required
  external-baseline verdict when that slice applies, uses silent fallback, or
  has incomplete provenance cannot count toward a positive benchmark conclusion.
- The package must preserve stable case-level provenance all the way from the
  Task 1 planner surface through the Task 6 correctness gate. At minimum, a
  counted benchmark case must remain auditable by workload family, workload ID,
  source type, entry route, requested mode, seed or deterministic construction
  rule, noise-pattern label, planner-setting reference, partition summary, and
  runtime-path classification.
- Sensitivity is part of the scientific identity of Phase 3 rather than optional
  appendix material. At minimum, Task 7 must make benchmark behavior auditable
  across:
  - supported planner-setting or partition-size choices within the bounded Task
    5 claim surface,
  - sparse, periodic, and dense local-noise placement,
  - and the frozen workload identity knobs of family, qubit count, and seed-
    fixed instance.
- If benchmark behavior differs materially across those sensitivity knobs, the
  package must record that difference explicitly rather than compressing the
  matrix into one averaged claim.
- Additional benchmark comparisons such as fused-versus-unfused or
  partitioned-versus-sequential are allowed when they clarify the bottleneck,
  but they do not replace the phase-level threshold rule, which remains defined
  against the sequential baseline on representative required cases.
- The benchmark package must distinguish clearly between:
  - counted supported benchmark evidence,
  - diagnosis-only or bottleneck-explaining benchmark evidence,
  - and excluded, unsupported, or deferred cases that define the boundary of the
    claim.
- The summary layer must be machine-checkable. Rolled-up counts, representative
  case selections, sensitivity summaries, speedup summaries, and diagnosis
  labels must be validated against the underlying per-case records and the Task
  6 counted-status rules rather than treated as prose-only interpretation.
- Repeated-timing interpretation must remain auditable. If Task 7 cites median
  wall-clock behavior or median peak-memory behavior, the underlying repeated
  runs or a rerunnable timing procedure must be recorded well enough to
  recompute that median later.
- If at least one representative required 8- or 10-qubit structured case meets
  the positive threshold without correctness loss, Task 7 must report that as a
  bounded benchmark result rather than as a universal acceleration claim.
- If no representative required case meets the positive threshold, Task 7 must
  close honestly through the diagnosis branch by emitting benchmark-grounded
  limitation reporting. That reporting must identify the main bottleneck in
  benchmark terms, preserve the supporting profile or artifact evidence when
  profiling materially affects the conclusion, and map the result to the
  follow-on decision gate without pretending Phase 3 is a failed or undefined
  phase.
- Task 7 may reuse Task 3, Task 4, Task 5, and Task 6 artifacts directly, but
  it must emit one stable benchmark package or rerunnable checker that
  downstream Task 8 consumers can use without reverse-engineering benchmark
  status from disconnected outputs.
- Task 7 completion means one representative performance and sensitivity package
  exists, is auditable, and is publication-usable. It does not require
  universal speedup, closure of the deferred channel-native branch, or broader
  workflow growth beyond the frozen Phase 3 boundary.

## Unsupported behavior
- Claiming a Phase 3 performance win from cases that lack Task 6 correctness
  closure, rely on silent fallback, or are missing stable case provenance.
- Cherry-picking favorable seeds, noise placements, planner settings, or one-off
  reruns while presenting them as the frozen representative benchmark matrix.
- Replacing the frozen workload matrix with ad hoc exploratory circuits,
  synthetic kernels, or benchmark-only manual case labels and still presenting
  the result as the supported Task 7 package.
- Treating one structured case at one qubit size as sufficient closure while
  skipping the frozen sensitivity dimensions of noise placement, qubit scale, or
  bounded planner-setting variation.
- Reporting runtime or speedup without peak memory, planning overhead,
  partition-context, runtime-path, or fused-coverage metadata when those fields
  are required to interpret the benchmark honestly.
- Using only fused-versus-unfused comparison, only partitioned-versus-
  partitioned comparison, or only profiler traces while omitting the phase-level
  comparison rule against the sequential baseline.
- Presenting diagnosis-only cases as if they were positive-threshold wins, or
  presenting positive-threshold wins as if they proved universal speedup across
  all noisy workloads.
- Collapsing counted supported cases, diagnosis-only cases, planner-entry
  unsupported cases, descriptor-generation unsupported cases, runtime-stage
  unsupported cases, and deferred fusion candidates into one ambiguous benchmark
  bucket.
- Omitting rerun metadata or repeated-run details while citing median runtime or
  memory behavior as a stable result.
- Treating rerun-sensitive Task 5 planner-setting outcomes as a permanently
  frozen universal winner instead of recording the supported bounded claim and
  the observed sensitivity.
- Using Task 7 to imply support for correlated noise, broader circuit-source
  parity, calibration-aware workflow features, gradients, broader noisy VQE/VQA
  growth, channel-native fused noisy blocks, or approximate scaling branches.
- Treating favorable benchmark slices as sufficient reason to weaken the frozen
  correctness thresholds, external-baseline obligations, or explicit limitation
  reporting required by the phase contract.

## Acceptance evidence
- A documented benchmark matrix identifies the continuity-anchor slices and the
  representative structured performance slices, gives each case a stable
  workload ID, records family name, qubit count, seed rule, noise-pattern label,
  planner-setting reference, and states which cases are counted, diagnosis-only,
  or excluded.
- The representative structured matrix covers:
  - required 8- and 10-qubit structured workloads,
  - at least `3` seed-fixed instances per mandatory structured family and size,
  - and sparse, periodic, and dense local-noise placement.
- One stable machine-reviewable benchmark package or rerunnable checker exists
  and records, for each counted or diagnosis-only case:
  - benchmark-package or schema version,
  - planner, descriptor, runtime, and supporting artifact schema versions where
    applicable,
  - workload family and workload ID,
  - requested mode,
  - source type and entry route,
  - seed or deterministic construction rule,
  - noise-pattern label,
  - planner-setting or Task 5 claim-selection reference,
  - partition count and partition-span summary,
  - runtime-path and fused-coverage classification,
  - runtime, peak memory, and planning time,
  - repeated-run timing or an auditable median-computation procedure,
  - correctness-gate references and raw comparison metrics where required,
  - external-baseline references where required,
  - counted versus diagnosis-only versus excluded status,
  - exclusion reason when not counted,
  - and software version or commit.
- The package includes one summary-consistency checker or equivalent validation
  rule proving that rolled-up benchmark counts, sensitivity summaries, and any
  representative-case claims agree with the underlying per-case records and the
  Task 6 counted-status surface.
- The benchmark evidence shows the required sensitivity dimensions directly:
  - variation across sparse, periodic, and dense local-noise placement,
  - variation across the supported planner-setting or partition-size choices
    inside the bounded Task 5 claim surface,
  - and variation across the frozen workload identity knobs of family, qubit
    count, and seed-fixed instance.
- At least one representative required 8- or 10-qubit structured case with
  supported `partitioned_density` execution either:
  - shows median wall-clock speedup `>= 1.2x` versus the sequential baseline
    without correctness loss,
  - or shows peak-memory reduction `>= 15%` versus the sequential baseline
    without correctness loss,
  - or is included in a diagnosis package whose benchmark-grounded explanation
    justifies the follow-on decision gate.
- Positive-threshold cases satisfy the frozen Task 6 correctness thresholds:
  - maximum Frobenius-norm density difference `<= 1e-10` against the sequential
    density baseline on required internal cases,
  - `|Tr(rho) - 1| <= 1e-10`,
  - `rho.is_valid(tol=1e-10)` on recorded outputs,
  - and maximum absolute energy error `<= 1e-8` on required continuity-anchor
    cases where the observable is part of the benchmark slice.
- Where the representative benchmark package reuses required 2 to 4 qubit
  microcases or the small continuity external slice, the recorded outputs remain
  joinable to the Qiskit Aer evidence without relabeling the runtime path or
  rebuilding the workload identity.
- If the diagnosis branch closes the rule, benchmark-grounded bottleneck
  artifacts explicitly identify the dominant limitation and preserve profiler
  artifacts when profiling materially affects the conclusion.
- Reproducibility artifacts record rerun commands, representative-case selection
  rules, software metadata, and the mapping from Task 5 calibration outputs plus
  Task 6 correctness records into the final Task 7 benchmark package.
- Traceability target: satisfy the Phase 3 Task 7 evidence requirements in
  `DETAILED_PLANNING_PHASE_3.md`.
- Traceability target: support the full-phase acceptance criteria requiring the
  benchmark package to record runtime, memory, planning overhead, and fusion
  coverage on representative noisy workloads, while documenting achieved benefit
  and remaining limitations honestly.
- Traceability target: satisfy the canonical planner-surface and semantic-
  preservation dependencies in `P3-ADR-003` and `P3-ADR-004`, the executable
  runtime and real-fused baseline in `P3-ADR-005`, the benchmark-calibrated
  planner interpretation boundary in `P3-ADR-006`, the frozen support-matrix
  boundary in `P3-ADR-007`, the two-baseline validation rule in `P3-ADR-008`,
  the benchmark-anchor and sensitivity decision in `P3-ADR-009`, and the
  deferred follow-on rule in `P3-ADR-010`.

## Affected interfaces
- `CanonicalNoisyPlannerSurface` and the planner-entry provenance surface that
  defines benchmark case identity, supported mode selection, and unsupported
  entry outcomes.
- `NoisyPartitionDescriptorSet`, `NoisyPartitionDescriptor`, and
  `NoisyPartitionDescriptorMember`, plus any descriptor-audit surfaces that
  carry partition-count, qubit-span, and semantic-preservation context into
  benchmark consumers.
- `NoisyRuntimeExecutionResult` and related partitioned density runtime outputs,
  including `runtime_ms`, `peak_rss_kb`, runtime-path classification, and
  fused-region summaries.
- Benchmark harnesses under `benchmarks/density_matrix/partitioned_runtime/`,
  including representative structured fused-performance validation surfaces and
  any successor Task 7 package builders.
- Task 5 calibration bundles or rerunnable checkers under
  `benchmarks/density_matrix/artifacts/planner_calibration/` that define the bounded
  supported planner-setting surface Task 7 interprets.
- Task 6 correctness, unsupported-boundary, and summary-consistency bundles
  under `benchmarks/density_matrix/artifacts/correctness_evidence/` that define counted
  status and correctness-gate reuse for Task 7.
- Summary-consistency, sensitivity-rollup, and reproducibility tooling that must
  preserve stable case identities and benchmark interpretations in
  machine-reviewable form.
- Publication-facing benchmark tables, limitation summaries, and architecture-
  decision references later consumed by Task 8.
- Change classification: additive for benchmark packaging, summary validation,
  and sensitivity reporting surfaces, but stricter for claim labeling because
  performance language now requires counted correctness-preserving benchmark
  evidence and honest diagnosis handling.

## Publication relevance
- Supports Paper 2's open performance question directly by turning the delivered
  Task 4 to Task 6 backend and correctness surfaces into one auditable
  representative benchmark package.
- Supplies the runtime, memory, planning-overhead, fused-coverage, and
  sensitivity evidence needed for the Phase 3 abstract, short paper, and full
  paper to make a bounded methods claim rather than a prose-only promise.
- Preserves scientific honesty by allowing Phase 3 to close through either the
  measurable-benefit path or the diagnosis-grounded limitation path, as long as
  both are benchmark-backed and auditable.
- Makes noise-placement sensitivity publishable rather than incidental, which is
  central to the scientific identity frozen for Phase 3.
- Prevents overclaiming by forcing Task 8 and later publication layers to reuse
  one benchmark package with stable workload identities, explicit counted-status
  rules, and machine-checkable summary consistency.
