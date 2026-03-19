# Task 4: Real Fused Execution On Eligible Substructures

This mini-spec turns Phase 3 Task 4 into an implementation-ready contract. It
inherits the frozen phase decisions from `DETAILED_PLANNING_PHASE_3.md`,
`P3-ADR-004`, `P3-ADR-005`, `P3-ADR-007`, `P3-ADR-008`, `P3-ADR-009`, and
`P3-ADR-010`, plus the closed runtime-minimum, support-matrix,
validation-baseline, benchmark-anchor, and performance-claim-boundary items in
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. It does not reopen canonical
planner-entry scope, descriptor semantics, the Task 3 executable partitioned
baseline, Task 5 density-aware heuristic calibration, or the deferred
channel-native / broader Phase 4 branches.

## Given / When / Then
- Given a supported `NoisyPartitionDescriptorSet` from the canonical Phase 3
  noisy planner surface, an executable Task 3 partitioned density runtime, and
  at least one descriptor-local region that satisfies the documented Task 4
  fusion-eligibility rules.
- When the runtime executes that workload under explicit
  `partitioned_density` behavior with auditable runtime settings and
  fused-capable execution enabled for eligible substructures.
- Then at least one eligible descriptor-level substructure runs through a real
  fused execution path inside the noisy partitioned runtime, surrounding noisy
  semantics remain exact relative to the sequential `NoisyCircuit` baseline
  within the frozen thresholds, and any supported-but-unfused or deferred
  fusion candidates remain explicitly classified rather than silently entering a
  fused path or being misreported as fused coverage.

## Assumptions and dependencies
- Task 1 provides the canonical Phase 3 planner-entry contract as a
  schema-versioned `CanonicalNoisyPlannerSurface` with stable provenance fields
  for `requested_mode`, `source_type`, `entry_route`, `workload_family`, and
  `workload_id`.
- Task 2 provides the schema-versioned partition handoff contract through
  `NoisyPartitionDescriptorSet`, `NoisyPartitionDescriptor`, and
  `NoisyPartitionDescriptorMember`, including canonical operation references,
  partition-local qubit remapping, and parameter-routing metadata.
- Task 3 provides the first executable partitioned density runtime that
  consumes the Task 2 descriptor contract directly and records an auditable
  runtime-path classification. Task 4 extends that same runtime boundary with
  real fused execution rather than replacing it with a separate incompatible
  executor.
- Task 6 depends on Task 4 by validating that fused-path outputs preserve exact
  noisy semantics against the sequential `NoisyCircuit` baseline and, on the
  required external slice, against Qiskit Aer.
- Task 7 depends on Task 4 by measuring runtime, peak memory, planning
  overhead, partition count, qubit span, fused-path coverage, and profiler
  evidence on workloads that actually exercise the fused path.
- The mandatory workload classes remain:
  - 2 to 4 qubit micro-validation circuits,
  - 4, 6, 8, and 10 qubit Phase 2 noisy XXZ `HEA` continuity cases,
  - and the mandatory 8 and 10 qubit structured noisy `U3` / `CNOT` families
    with stable seed rules and sparse, periodic, and dense local-noise
    patterns.
- The frozen required gate surface remains `U3` and `CNOT`. The frozen required
  noise surface remains local single-qubit depolarizing, local amplitude
  damping, and local phase damping or dephasing.
- The minimum required fused baseline is conservative unitary-island fusion
  inside noisy partitions. Control-aware refinements or very small-support local
  channel fusion may be explored only as optional extensions and do not define
  the minimum Task 4 closure bar.
- Fully channel-native fused noisy blocks remain explicitly deferred beyond the
  minimum Phase 3 contract.
- Fused execution may introduce internal kernels, compiled blocks, caches, or
  schedule summaries, but supported semantics and benchmark evidence must remain
  auditable back to the Task 2 descriptor contract plus documented runtime
  settings rather than hidden planner state or benchmark-only annotations.

## Required behavior
- Task 4 freezes the minimum real fused-execution contract inside the Task 3
  partitioned density runtime. Planner-only fusion markings, symbolic
  fusibility, or offline kernel experiments are not enough.
- Supported fusion eligibility must be defined against the Task 2
  descriptor-level contract rather than opaque planner internals. The minimum
  positive-path eligibility class for Phase 3 closure is at least one
  descriptor-local substructure that is:
  - contiguous in the ordered descriptor-member sequence,
  - contained within one supported partition runtime context,
  - composed only of supported unitary gate members on the frozen `U3` /
    `CNOT` surface,
  - auditable through the descriptor qubit-remapping and parameter-routing
    metadata,
  - and bounded by partition edges or explicit noise members unless a separate
    exact equivalence rule is documented and validated.
- Eligibility detection may use internal analysis or scheduling logic, but the
  contract-defining result must remain auditable in runtime artifacts. Hidden
  workload-specific allowlists or benchmark-only manual tagging are not the
  supported contract.
- A supported fused path must actually execute at least one eligible
  substructure through a real fused runtime path or fused kernel inside the
  partitioned density executor. Merely emitting a fused annotation while
  executing the unfused Task 3 baseline does not satisfy this task.
- The fused path must consume descriptor ordering, canonical operation
  references, partition-local remapping, and parameter-routing metadata
  directly from the Task 2 contract or from an auditable derivation of it. A
  second private fusion-only contract is not supported.
- Exact noisy semantics around the fused substructure must remain preserved. In
  the minimum baseline, explicit noise operations remain first-class boundaries
  and must not be silently absorbed into fused blocks, crossed by undocumented
  reordering, or reduced to metadata-only markers.
- Supported runtime execution may mix fused and unfused substructures inside the
  same workload, but artifacts and diagnostics must keep the following classes
  distinct:
  - supported and actually exercised through the fused path,
  - supported but intentionally left unfused,
  - and deferred or unsupported as fusion candidates for the minimum Phase 3
    contract.
- If a supported workload contains no eligible fused region, or contains
  regions intentionally left unfused, the workload may still execute through the
  plain Task 3 partitioned path. Those cases must be recorded as unfused or
  deferred rather than being misreported as fused coverage.
- If a workload or substructure is reported as fused coverage in validation,
  benchmark, or publication-facing artifacts, the recorded runtime-path
  classification must show actual fused execution rather than the plain
  partitioned baseline.
- Fused-path execution must emit the same comparison-ready density outputs and,
  where applicable, continuity observable outputs required for the Task 3
  correctness package. Enabling fusion must not force a separate unlabeled
  output route.
- Runtime artifacts for fused cases must record enough provenance to keep the
  fusion claim auditable. At minimum, the evidence should retain:
  - planner schema identity,
  - descriptor schema identity,
  - requested mode,
  - source type,
  - entry route,
  - workload family,
  - workload ID,
  - runtime-path classification,
  - partition count,
  - partition-span summary,
  - fused-substructure count,
  - fused gate count or span summary,
  - and stable reasons for any eligible-looking region that is left unfused or
    deferred.
- Unsupported fusion candidates that still belong to an otherwise supported
  workload must not silently take the fused path. They must either remain on the
  supported unfused Task 3 path with explicit classification, or hard-fail with
  stable structured diagnostics when the runtime cannot preserve the supported
  semantics at all.
- Task 4 completion means at least one eligible descriptor-level substructure on
  representative real Phase 3 workloads executes through a real fused path,
  preserves exact noisy semantics within the frozen thresholds, and produces
  auditable coverage and deferral evidence. It does not by itself require fully
  channel-native noisy-block fusion, universal acceleration, or Task 5
  heuristic-calibration closure.

## Unsupported behavior
- Claiming Task 4 closure because partitions are marked as fusible, because a
  planner emits fusion candidates, or because isolated synthetic kernels run
  outside the real noisy partitioned runtime.
- Fusing across explicit noise boundaries or absorbing supported noise
  operations into a unitary-only fused block without a separately documented and
  validated exact contract.
- Reinterpreting descriptor semantics through a second private fusion-only
  representation that becomes the real supported fused path.
- Hidden workload-specific eligibility heuristics, benchmark-only manual
  tagging, or post hoc relabeling that cannot be audited back to the descriptor
  members actually executed.
- Silent fallback from an artifact labeled as fused execution to the plain Task
  3 partitioned path.
- Collapsing supported-but-unfused cases, deferred fusion candidates, and true
  runtime-stage unsupported failures into one ambiguous evidence bucket.
- Treating fully channel-native fused noisy blocks, broader noisy VQE/VQA
  workflow growth, or approximate scaling branches as part of the minimum Task 4
  contract.
- Widening the required support surface beyond the frozen `U3` / `CNOT` plus
  local-noise baseline as part of fused-execution closure.
- Using only synthetic microbenchmarks or profiler traces as evidence that the
  real fused path exists on representative Phase 3 workloads.
- Claiming speedup or memory benefit from cases that do not record real
  fused-path coverage and correctness-preserving runtime provenance.

## Acceptance evidence
- Supported integration or benchmark runs show at least one real fused
  execution mode on eligible descriptor-level substructures inside the Task 3
  partitioned density runtime, and those runs use representative real Phase 3
  workloads rather than only synthetic kernels.
- For every benchmarked fused case on the mandatory correctness surface, fused
  execution preserves the frozen exactness thresholds against the sequential
  density baseline:
  - maximum Frobenius-norm density difference `<= 1e-10`,
  - `|Tr(rho) - 1| <= 1e-10`,
  - and recorded outputs satisfy `rho.is_valid(tol=1e-10)`.
- Any continuity-anchor case exercised through the fused path satisfies the
  frozen maximum absolute energy error threshold `<= 1e-8`.
- Validation and benchmark artifacts classify fusion results explicitly across
  the required three-way evidence split:
  - supported and exercised through fused execution,
  - supported but intentionally left unfused,
  - and deferred or unsupported as fusion candidates.
- Reproducibility artifacts for fused and near-fused cases record planner schema
  version, descriptor schema version, requested mode, source type, entry route,
  workload family, workload ID, runtime-path classification, partition count,
  partition-span summary, fused-path coverage, and stable deferral or
  unsupported-category labels where applicable.
- The required benchmark package records runtime, peak memory, planning time,
  partition count, qubit span, and fused-path coverage for workloads that
  exercise the fused path.
- The phase-level performance-evidence rule is fed by Task 4 artifacts:
  - either at least one required 8- or 10-qubit structured case with real fused
    coverage shows median wall-clock speedup `>= 1.2x` or peak-memory reduction
    `>= 15%` versus the sequential baseline without correctness loss,
  - or the required benchmark package plus profiling evidence explicitly shows
    why the native fused baseline does not yet accelerate that case and
    justifies the follow-on architecture decision gate.
- Where fused coverage is exercised on the required 2 to 4 qubit microcases,
  recorded outputs remain consumable by the Task 6 Qiskit Aer checks without
  rerunning the case through a different interface or relabeling the runtime
  path.
- Traceability target: satisfy the Phase 3 Task 4 evidence requirements in
  `DETAILED_PLANNING_PHASE_3.md`.
- Traceability target: support the full-phase acceptance criteria requiring an
  executable partitioned density runtime with at least one benchmarked real
  fused execution mode on eligible substructures.
- Traceability target: satisfy the exact-order-preservation decision in
  `P3-ADR-004`, the runtime-minimum decision in `P3-ADR-005`, the frozen
  support-matrix boundary in `P3-ADR-007`, the validation-baseline and
  benchmark-anchor decisions in `P3-ADR-008` and `P3-ADR-009`, and the deferred
  follow-on boundary in `P3-ADR-010`.

## Affected interfaces
- `NoisyPartitionDescriptorSet`, `NoisyPartitionDescriptor`, and
  `NoisyPartitionDescriptorMember`, plus any serialization or adapter boundary
  that carries the Task 2 descriptor contract into fused-capable runtime
  execution.
- Any partitioned density execution-plan builder, eligibility classifier, or
  runtime adapter that identifies descriptor-local fused substructures and maps
  them into real fused runtime work.
- Any fused kernel, compiled block, or partition-local executor that runs a
  supported eligible substructure while preserving the surrounding noisy
  semantics.
- The plain Task 3 partitioned runtime-path classification surface, which must
  now distinguish actual fused execution from supported unfused execution and
  from runtime-stage unsupported outcomes.
- The sequential `NoisyCircuit` execution boundary and any comparison harness
  that treats it as the internal exact baseline for fused-path validation.
- External microcase validation surfaces that compare required fused-capable
  cases against Qiskit Aer.
- Benchmark, validation, and reproducibility tooling that must record fused-path
  coverage, partition count, partition span, runtime metrics, and stable
  deferral or unsupported-candidate diagnostics.
- Change classification: additive for the supported fused runtime baseline, but
  stricter for ambiguous fusion labeling because cases that were previously
  only "fusible in principle" now must either execute through a real fused path
  or remain explicitly labeled as unfused or deferred.

## Publication relevance
- Supports Paper 2's central claim that Phase 3 delivers limited real fused
  execution on noisy mixed-state workloads rather than stopping at partitioning
  and scheduling.
- Supplies the auditable evidence needed for the abstract, short paper, and full
  paper to distinguish:
  - benchmarked real fused coverage,
  - supported workloads intentionally left unfused,
  - and explicitly deferred channel-native fusion work.
- Keeps publication claims scientifically defensible by tying the fused result
  to the schema-versioned descriptor contract, the sequential exact baseline,
  and the frozen workload matrix rather than to synthetic kernel stories.
- Supports honest performance framing by feeding the threshold-or-diagnosis rule
  for representative 8- and 10-qubit structured workloads instead of implying a
  universal speedup claim.
