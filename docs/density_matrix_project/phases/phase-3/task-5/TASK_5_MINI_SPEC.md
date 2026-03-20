# Task 5: Noise-Aware Planning Heuristic And Calibration

This mini-spec turns Phase 3 Task 5 into an implementation-ready contract. It
inherits the frozen phase decisions from `DETAILED_PLANNING_PHASE_3.md`,
`P3-ADR-003`, `P3-ADR-004`, `P3-ADR-005`, `P3-ADR-006`, `P3-ADR-007`,
`P3-ADR-008`, `P3-ADR-009`, and `P3-ADR-010`, plus the closed canonical
planner-surface, semantic-preservation, runtime-minimum, cost-model
sequencing, support-matrix, validation-baseline, benchmark-anchor, and
performance-claim-boundary items in
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. It does not reopen the supported
workload surface, the frozen exactness thresholds, or the deferred
channel-native / broader Phase 4 branches.

## Given / When / Then
- Given a supported `CanonicalNoisyPlannerSurface`, an auditable
  `NoisyPartitionDescriptorSet` contract, and an executable partitioned density
  runtime with real fused execution available on at least one eligible
  substructure inside the frozen workload matrix.
- When Phase 3 planning claims a density-aware heuristic or objective for
  selecting among supported partitioning alternatives on mandatory workloads.
- Then that claim is benchmark-calibrated rather than purely analytic, responds
  to mixed-state cost and explicit noise placement rather than to an unchanged
  state-vector proxy, remains auditable through stable planner settings and
  workload provenance, and keeps its approximation and support boundary
  explicit.

## Assumptions and dependencies
- Task 1 provides the canonical Phase 3 planner-entry contract as a
  schema-versioned `CanonicalNoisyPlannerSurface` with stable provenance fields
  for `requested_mode`, `source_type`, `entry_route`, `workload_family`, and
  `workload_id`.
- Task 2 provides the schema-versioned partition handoff contract through
  `NoisyPartitionDescriptorSet`, `NoisyPartitionDescriptor`, and
  `NoisyPartitionDescriptorMember`, including canonical operation references,
  exact gate/noise order, qubit-remapping metadata, and parameter-routing
  metadata.
- Task 3 provides the first executable partitioned density runtime that turns
  supported descriptor sets into comparison-ready outputs and auditable runtime
  metrics without silent fallback.
- Task 4 provides the minimum real fused-execution baseline plus explicit
  classification of fused, supported-but-unfused, and deferred fusion
  candidates. Task 5 may use fused-path coverage as one measured signal, but it
  does not require fully channel-native fusion to define the calibrated
  planning claim.
- Task 6 depends on Task 5 by validating that the calibrated planner's positive
  benchmark cases still preserve the frozen exact noisy semantics and keep
  unsupported boundaries explicit.
- Task 7 depends on Task 5 by turning the calibration outputs into rolled-up
  runtime, memory, planning-overhead, and sensitivity summaries for the
  methods-paper evidence package.
- The mandatory workload classes remain:
  - 2 to 4 qubit micro-validation circuits,
  - 4, 6, 8, and 10 qubit Phase 2 noisy XXZ `HEA` continuity cases,
  - and the mandatory 8 and 10 qubit structured noisy `U3` / `CNOT` families
    with stable seed rules and sparse, periodic, and dense local-noise
    patterns.
- The frozen required gate surface remains `U3` and `CNOT`. The frozen required
  noise surface remains local single-qubit depolarizing, local amplitude
  damping, and local phase damping or dephasing.
- The sequential `NoisyCircuit` path remains the internal exact oracle, and
  Qiskit Aer remains the required external reference on the frozen microcase
  slice and representative small continuity subset. They are validation
  baselines, not hidden planner fallbacks.
- The existing state-vector FLOP model in `squander/partitioning/tools.py` may
  be reused as scaffolding or as an explicit comparison baseline, but not as
  the unmodified Phase 3 scientific claim.
- Task 5 may calibrate a structural heuristic, a weighted objective, an
  optimization-guided planner, or a bounded candidate-setting policy, but the
  contract-defining result is the benchmark-calibrated behavior on the frozen
  workload matrix rather than the implementation style used to compute it.
- In the current delivered Task 5 result, the supported calibration surface is a
  bounded family of auditable `max_partition_qubits` span-budget settings on
  the existing noisy planner surface. Broader adapted `kahn` / `tdag` /
  `gtqcp` / `ilp` / `ilp-fusion-ca` families remain design-space or comparison
  references until they are separately implemented on the noisy planner path.
- Stable workload IDs, deterministic seed rules, noise-pattern labels, planner
  settings, and machine-reviewable artifact bundles are already part of the
  frozen Phase 3 evidence contract and must remain reusable here.
- Planner-time overhead is part of the scientific interpretation surface. It is
  not optional metadata that can be omitted when a planner appears to improve
  runtime or peak memory.
- This task does not widen the support matrix to correlated multi-qubit noise,
  readout or shot-noise workflow features, calibration-aware noise, broader
  noisy VQE/VQA growth, or approximate scaling branches.

## Required behavior
- Task 5 freezes the minimum density-aware planning claim for Phase 3: at least
  one declared planner heuristic or objective must be calibrated on the frozen
  benchmark matrix and then used to rank, score, or select among supported
  partitioning alternatives on the canonical noisy planner surface.
- The supported calibration target may be a heuristic rule set, a weighted
  score, an optimization objective, or another auditable decision surface, but
  it must be explicit enough that later validation, benchmarking, and paper
  work can rerun or inspect what was calibrated.
- A Phase 3 planner is density-aware only if its supported claim responds to
  mixed-state execution signals that are absent from the unchanged state-vector
  FLOP model. At minimum, the calibrated behavior must be explainable in terms
  of auditable information drawn from:
  - partition support size or qubit-span cost,
  - explicit noise placement, density, or boundary interaction,
  - and measured benchmark behavior on the frozen workload matrix, including
    runtime, peak memory, planning time, partition count or span, and where
    relevant fused-path coverage.
- Calibration must be anchored to the frozen mandatory workload inventory and
  its stable workload IDs, seed rules, and noise-pattern labels. Ad hoc
  exploratory cases may help development, but they do not define the supported
  Phase 3 claim.
- Calibration must preserve the canonical-surface and descriptor-level semantic
  contract. Planner scoring changes may alter partition choice, but they must
  not weaken exact gate/noise order, parameter-routing, remapping, runtime-path
  labeling, or unsupported-case obligations.
- Positive calibration evidence may count only supported runs that execute
  through the documented partitioned runtime surface and satisfy the frozen
  correctness thresholds against the sequential density baseline. Required
  external Aer comparisons remain applicable on the frozen microcase slice and
  any other external-baseline cases explicitly used in the calibration package.
- The supported calibration surface must record enough provenance to keep the
  claim auditable. At minimum, calibration artifacts should retain:
  - planner schema identity or heuristic version,
  - workload family and workload ID,
  - requested mode,
  - source type,
  - entry route,
  - seed or deterministic construction rule,
  - noise-pattern label,
  - planner settings,
  - partition count and partition-span summary,
  - runtime-path classification,
  - runtime, peak memory, and planning time,
  - correctness verdicts against the frozen thresholds,
  - and calibration outputs such as score, ranking, selected configuration, or
    an auditable equivalent.
- If multiple planner candidates or multiple parameter settings are evaluated,
  the final supported calibration must be selected through one documented rule
  grounded in the frozen benchmark data rather than through post hoc narrative
  preference or hand-picked favorable plots.
- If that benchmark-grounded rule yields close or rerun-sensitive winners inside
  the bounded supported candidate family, the supported claim is the auditable
  selection rule plus explicit claim boundary, not one permanently frozen winner
  identity.
- The calibration package must keep the claim boundary explicit. At minimum, it
  must distinguish:
  - the benchmark-calibrated planner settings or method that define the
    supported Phase 3 claim,
  - structural or state-vector-oriented comparison baselines used only for
    reference,
  - and exploratory, out-of-scope, or diagnosis-only cases that do not enlarge
    the supported claim boundary.
- If the calibrated planner remains approximate on some workload classes, noise
  densities, or fusion opportunities, those approximation areas must be
  documented explicitly rather than hidden behind one generic density-aware
  label.
- The unchanged state-vector FLOP model may remain present as scaffolding or a
  comparison line, but it must not be relabeled as the Phase 3 density-aware
  objective unless the calibration evidence shows which density-specific signals
  were added and how the claim differs from the state-vector baseline.
- Planner-time overhead must be measured and reported as part of the supported
  calibration package. A planner that appears favorable only because planning
  cost is omitted does not satisfy the Task 5 claim.
- Task 5 completion means at least one benchmark-calibrated density-aware
  planner surface or bounded candidate family is defined, auditable, and
  reusable by later validation, benchmark, and paper-packaging work. It does
  not by itself require global optimality, universal acceleration, one
  permanently stable winning candidate identity, or closure of the separate Task
  6 correctness package or Task 7 rolled-up performance interpretation.

## Unsupported behavior
- Porting the unchanged state-vector FLOP model into Phase 3 and relabeling it
  as density-aware without benchmark-calibrated mixed-state evidence.
- Claiming Task 5 closure from algorithm-landscape discussion, structural
  heuristic intuition, or planner-internal score definitions alone without
  calibration on the frozen benchmark matrix.
- Calibrating only on ad hoc favorable workloads, only on synthetic kernels
  disconnected from the supported runtime, or only on one narrow workload slice
  while presenting the result as the full Phase 3 claim.
- Using hidden per-workload overrides, benchmark-only allowlists, manually
  chosen partitions, or another non-auditable tuning path that later consumers
  cannot rerun from the recorded planner settings and workload provenance.
- Counting cases that fail the frozen correctness thresholds, rely on silent
  fallback, or use an unlabeled runtime path as positive calibration evidence.
- Omitting planning time, workload identity, or planner-setting provenance from
  the claimed calibration package.
- Treating exploratory broader support such as correlated noise,
  calibration-aware workflow features, readout-oriented features, or approximate
  scaling branches as if they were part of the frozen minimum Task 5 surface.
- Claiming global optimality, universal acceleration, or broad workflow
  generality from a benchmark-calibrated heuristic or objective that is only
  validated on the frozen Phase 3 workload matrix.
- Folding deferred channel-native fusion or other post-benchmark architecture
  branches into the minimum density-aware planning claim.
- Treating structural heuristic baselines or state-vector-oriented comparison
  baselines as the supported calibrated model without explicitly distinguishing
  them from the benchmark-calibrated claim surface.

## Acceptance evidence
- A documented calibration matrix identifies the mandatory micro-validation,
  continuity, and structured-workload cases that participate in calibration,
  gives them stable workload IDs, records the seed rules and noise-pattern
  labels, and states which planner settings or candidate methods were evaluated.
- One stable machine-reviewable calibration bundle or rerunnable checker exists
  for the supported calibration surface and records planner version or heuristic
  ID, workload provenance, selected-plan summary, score or ranking output,
  runtime-path classification, partition count or span, runtime, peak memory,
  planning time, and the calibration verdict for each counted case.
- Positive calibration cases satisfy the frozen internal exactness thresholds
  against the sequential density baseline on the mandatory correctness surface:
  - maximum Frobenius-norm density difference `<= 1e-10`,
  - `|Tr(rho) - 1| <= 1e-10`,
  - and recorded outputs satisfy `rho.is_valid(tol=1e-10)`.
- Any continuity-anchor cases counted in the calibration package satisfy the
  frozen maximum absolute energy error threshold `<= 1e-8`.
- Where 2 to 4 qubit microcases or representative small continuity cases are
  used as external-baseline anchors in the calibration package, recorded outputs
  remain comparable to Qiskit Aer and satisfy the frozen external exactness
  rule.
- The calibration evidence shows that planner behavior is not noise-blind. At
  least one reproducible comparison slice across the frozen workload inventory
  demonstrates that explicit noise placement, mixed-state support cost, or
  fused-coverage opportunity changes the recorded planner score, ranking, or
  selected configuration in a machine-reviewable way.
- At least one auditable comparison against a structural-only or
  state-vector-oriented baseline is preserved so the Phase 3 density-aware claim
  is distinguishable from pre-calibration scaffolding.
- Reproducibility artifacts for mandatory calibration evidence record planner
  schema or heuristic version, workload family, workload ID, source type, entry
  route, seed or deterministic construction rule, noise-pattern label, planner
  settings, partition summaries, raw benchmark outputs, correctness verdicts,
  and software version or commit.
- The calibration summary documents the supported claim boundary, any diagnosis-
  only or exploratory cases, and the main approximation areas that remain after
  calibration.
- Traceability target: satisfy the Phase 3 Task 5 evidence requirements in
  `DETAILED_PLANNING_PHASE_3.md`.
- Traceability target: support the full-phase acceptance criteria requiring a
  benchmark-calibrated density-aware planning objective or heuristic without
  overstating optimality, support scope, or performance conclusions beyond the
  frozen workload matrix.
- Traceability target: satisfy the canonical planner-surface and
  semantic-preservation dependencies in `P3-ADR-003` and `P3-ADR-004`, the
  runtime-minimum and real-fused baseline dependence in `P3-ADR-005`, the
  correctness-first calibration rule in `P3-ADR-006`, the frozen support-matrix
  boundary in `P3-ADR-007`, the validation-baseline rule in `P3-ADR-008`, the
  benchmark-anchor decision in `P3-ADR-009`, and the deferred follow-on
  boundary in `P3-ADR-010`.

## Affected interfaces
- `CanonicalNoisyPlannerSurface` and any equivalent canonical noisy planner
  input boundary that feeds partition planning on supported Phase 3 workloads.
- Any planner cost-model, scoring, ranking, or strategy-selection surface that
  claims density-aware behavior for noisy mixed-state workloads, including
  structural baselines such as `kahn`, `tdag`, and `gtqcp`, or optimization-
  guided variants such as adapted `ilp`, `ilp-fusion`, or `ilp-fusion-ca` when
  those are in scope.
- `NoisyPartitionDescriptorSet`, `NoisyPartitionDescriptor`, and any plan
  summary or metadata surface that carries partition-count, qubit-span, or
  planner-setting information into runtime and benchmark consumers.
- The partitioned density runtime and fused-capable execution surfaces that
  supply the measured runtime, peak-memory, runtime-path, and fused-coverage
  signals used in calibration.
- Validation, benchmark, and reproducibility tooling that must preserve stable
  workload provenance, correctness verdicts, planner settings, and calibration
  outputs in machine-reviewable form.
- Any machine-reviewable calibration manifest, artifact bundle, or rerunnable
  checker that later Task 6, Task 7, and publication packaging can consume
  directly.
- Optional profiler-integration surfaces when profiling materially supports the
  interpretation of planner settings or follow-on architecture decisions.
- Change classification: additive for planner-selection behavior and audit
  metadata, but stricter for claim labeling because density-aware language now
  requires benchmark-calibrated provenance rather than heuristic intuition
  alone.

## Publication relevance
- Supports Paper 2's core claim that Phase 3 partition planning is density-aware
  and benchmark-calibrated rather than a renamed state-vector cost model.
- Provides the calibration evidence needed for the Phase 3 abstract, short
  paper, and full paper to describe why the planner responds to noisy
  mixed-state workloads and where the claim boundary stops.
- Keeps publication claims scientifically defensible by tying the planner result
  to the frozen workload IDs, seed rules, noise-pattern vocabulary, correctness
  baselines, and auditable runtime measurements.
- Prevents the paper narrative from overstating optimality or generality by
  separating:
  - the supported benchmark-calibrated planner claim,
  - structural or state-vector comparison baselines,
  - and deferred post-benchmark architecture branches.
