# Task 3: Executable Partitioned Density Runtime

This mini-spec turns Phase 3 Task 3 into an implementation-ready contract. It
inherits the frozen phase decisions from `DETAILED_PLANNING_PHASE_3.md`,
`P3-ADR-003`, `P3-ADR-004`, `P3-ADR-005`, `P3-ADR-006`, `P3-ADR-007`,
`P3-ADR-008`, and `P3-ADR-009`, plus the closed runtime-minimum,
validation-baseline, benchmark-anchor, and performance-claim-boundary items in
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. It does not reopen
planner-surface scope, descriptor semantics, the real fused-execution closure
owned by Task 4, density-aware cost-model calibration, or broader Phase 4
workflow-growth decisions.

## Given / When / Then
- Given a supported `NoisyPartitionDescriptorSet` produced from the canonical
  Phase 3 noisy planner surface and a request that claims
  `partitioned_density` behavior on a mandatory workload inside the frozen
  support matrix.
- When the runtime executes that descriptor set under explicit partitioned mode
  with a concrete parameter vector and documented runtime settings.
- Then the case runs end to end through a real partitioned density execution
  path that consumes the auditable Task 2 descriptor contract directly,
  preserves exact noisy semantics relative to the sequential `NoisyCircuit`
  baseline within the frozen thresholds, records stable runtime provenance, and
  hard-fails runtime-stage unsupported cases without silent fallback or
  relabeling them as supported results.

## Assumptions and dependencies
- Task 1 provides the canonical Phase 3 planner-entry contract as a
  schema-versioned `CanonicalNoisyPlannerSurface` with stable provenance fields
  for `requested_mode`, `source_type`, and `workload_id` (entry route and
  workload family implied by `source_type`, not stored on the surface).
- Task 2 provides the schema-versioned partition handoff contract through
  `NoisyPartitionDescriptorSet`, `NoisyPartitionDescriptor`, and
  `NoisyPartitionDescriptorMember`, including canonical operation references,
  partition-local qubit remapping, and parameter-routing metadata.
- Task 3 defines the first executable partitioned density runtime that consumes
  that Task 2 contract directly. It does not by itself close the separate Task
  4 requirement for at least one real fused execution mode on eligible
  substructures.
- Task 4 depends on Task 3 by extending the same runtime boundary with fused
  execution on eligible descriptor-level substructures rather than by replacing
  the Task 3 runtime contract with a separate incompatible execution path.
- Task 6 depends on Task 3 by comparing runtime outputs against the sequential
  `NoisyCircuit` baseline and by validating that runtime-stage failures remain
  distinct from planner-entry and descriptor-generation failures.
- Task 7 depends on Task 3 by measuring runtime, peak memory, planning
  overhead, partition count, qubit span, and later fused-path coverage on the
  executable runtime surface created here.
- The sequential `NoisyCircuit` execution path remains the internal exact
  semantic oracle for every mandatory Phase 3 case. Qiskit Aer remains the
  required external reference on the frozen microcase slice and representative
  small continuity subset, but it is a validation baseline rather than a
  fallback execution path.
- The mandatory workload classes remain:
  - 2 to 4 qubit micro-validation circuits,
  - 4, 6, 8, and 10 qubit Phase 2 noisy XXZ `HEA` continuity cases,
  - and the mandatory 8 and 10 qubit structured noisy `U3` / `CNOT` families
    with stable seed rules and sparse, periodic, and dense local-noise
    patterns.
- The frozen required gate surface remains `U3` and `CNOT`. The frozen required
  noise surface remains local single-qubit depolarizing, local amplitude
  damping, and local phase damping or dephasing.
- Runtime implementation may introduce internal execution plans, partition-local
  kernels, or schedule objects, but supported semantics must remain auditable
  back to the descriptor contract plus documented runtime configuration rather
  than hidden planner state or workload-specific adapters.
- Task 1 and Task 2 already established separate structured unsupported
  vocabularies for planner-entry and descriptor-generation failures. Task 3
  should extend that evidence model with distinct runtime-stage failure
  reporting rather than collapsing all failures into one generic bucket.
- The sequential density path is the correctness reference only. It must not
  remain as a silent fallback for requests labeled as supported
  `partitioned_density` execution.
- This task does not widen the frozen support matrix to correlated multi-qubit
  noise, readout or shot-noise workflow features, calibration-aware noise,
  broader noisy VQE/VQA feature growth, or channel-native fused noisy blocks.

## Required behavior
- Task 3 freezes the minimum executable runtime contract between
  `NoisyPartitionDescriptorSet` and end-to-end partitioned density execution.
- A supported runtime request must consume the schema-versioned Task 2
  descriptor set directly. A second private runtime-only descriptor schema, or
  a workload-specific ad hoc reinterpretation of the same partitioned case, is
  not the contract-defining supported path.
- The runtime must execute supported mandatory cases under explicit
  `partitioned_density` mode selection. A case must not reach the sequential
  density path, the state-vector path, or another hidden non-partitioned path
  while still being labeled as supported partitioned execution.
- Runtime execution must derive partition-local qubit layout from the Task 2
  descriptor metadata such as `local_to_global_qbits` or an auditable
  equivalent derived from it. Hidden workload-specific qubit remapping is not
  supported.
- Global parameter vectors must be routed into partition-local execution
  according to the descriptor `parameter_routing` contract or an equivalent
  auditable mapping derived from it. Implicit parameter-order assumptions are
  not supported.
- Exact gate and noise order defined by descriptor membership and canonical
  operation references must remain preserved through runtime execution. Noise
  operations remain first-class runtime content rather than boundary-only or
  out-of-band metadata.
- If the runtime batches, caches, or otherwise transforms partition-local work
  internally, those transformations must not weaken the exact descriptor-level
  semantics or obscure how the supported result relates to the validated Task 2
  contract.
- Supported runtime execution must produce a final density result or equivalent
  exact-output record for every mandatory correctness case so the partitioned
  result can be compared directly against the sequential density baseline.
- Supported continuity-anchor execution must also emit observable-comparison
  outputs sufficient to verify the frozen energy-agreement threshold without
  requiring a separate unlabeled execution route.
- Required microcase runtime outputs must be emitted in a stable,
  comparison-ready form that later external validation against Qiskit Aer can
  consume without relabeling the workload, changing the runtime path, or
  regenerating results through a different execution surface.
- Supported runtime results must reuse the Task 1 and Task 2 case-level
  provenance tuple and add runtime-specific metadata sufficient to keep the
  execution auditable. At minimum, the recorded runtime evidence should retain:
  - planner schema identity,
  - descriptor schema identity,
  - requested mode,
  - source type,
  - entry route,
  - workload family,
  - workload ID,
  - qubit count,
  - partition count,
  - partition-span summary,
  - and runtime-path classification such as the plain partitioned baseline and
    any later fused extension.
- The runtime must expose descriptor-to-runtime handoff evidence, such as an
  auditable execution-plan summary, per-partition execution record, or
  rerunnable checker, showing that the runtime consumed the supported Task 2
  contract rather than hidden planner state.
- Runtime-stage unsupported or incomplete-execution cases that arise after
  successful descriptor generation must hard-fail with stable structured
  diagnostics. At minimum, the emitted failure record must preserve a distinct
  runtime failure stage plus unsupported category, first unsupported condition,
  requested mode, workload identity, and descriptor identity or an auditable
  equivalent.
- Supported runtime execution must be reproducible for a fixed descriptor set,
  parameter vector, and runtime configuration. Reproducibility here means the
  runtime path label, case provenance, and exact comparison outputs remain
  stable enough for later validation and benchmark packaging.
- Task 3 completion means supported Phase 3 workloads can execute end to end
  from the validated descriptor contract with no silent fallback and with
  comparison-ready outputs against the sequential baseline. It does not by
  itself require the separate Task 4 fused-execution closure, Task 5
  density-aware heuristic calibration, or the full Task 7 sensitivity package.

## Unsupported behavior
- Reinterpreting `NoisyPartitionDescriptorSet` through a second private
  runtime-only contract that becomes the real supported execution surface.
- Running a case through the sequential density executor, state-vector path, or
  another hidden non-partitioned path while still recording the case as
  supported `partitioned_density` execution.
- Using workload-specific adapters that bypass the validated descriptor contract
  for continuity cases, microcases, or structured benchmark families.
- Applying implicit qubit remapping, implicit parameter routing, or another
  runtime-only semantic assumption that cannot be audited back to the Task 2
  descriptor metadata.
- Dropping noise operations, silently coalescing descriptor members, or
  reordering operations across unsupported noise boundaries during execution.
- Warning-only or best-effort handling for runtime-stage unsupported cases when
  the request claims supported `partitioned_density` behavior.
- Collapsing runtime-stage failures into planner-entry or descriptor-generation
  failure buckets in a way that makes the real failure layer unauditable.
- Treating Task 3 as complete because descriptors can be emitted or because a
  benchmark script can run, even if the runtime result is not compared against
  the sequential density baseline on the mandatory correctness surface.
- Claiming Task 3 closure implies the separate Task 4 real fused-execution
  requirement, Task 5 benchmark-calibrated heuristic claim, or broader Phase 4
  workflow growth are already complete.
- Widening the required support surface beyond the frozen `U3` / `CNOT` plus
  local-noise baseline as part of Task 3 closure.
- Using Qiskit Aer or another external simulator as an execution fallback while
  still claiming the supported internal partitioned runtime was exercised.

## Acceptance evidence
- Supported integration or validation runs show that the mandatory workload
  matrix executes through an explicit `partitioned_density` runtime path from
  `NoisyPartitionDescriptorSet` inputs rather than from hidden planner or
  workload-specific adapters.
- Descriptor-to-runtime handoff evidence shows that the runtime consumes the
  Task 2 schema-versioned descriptor contract directly and records auditable
  execution metadata without requiring a second private runtime schema.
- Internal correctness evidence shows that partitioned-versus-sequential
  density agreement satisfies the frozen Phase 3 thresholds on the mandatory
  correctness matrix:
  - maximum Frobenius-norm density difference `<= 1e-10`,
  - `|Tr(rho) - 1| <= 1e-10`,
  - and recorded outputs satisfy `rho.is_valid(tol=1e-10)`.
- Continuity-anchor evidence shows that the required 4, 6, 8, and 10 qubit
  noisy XXZ `HEA` cases satisfy the frozen maximum absolute energy error
  threshold `<= 1e-8`.
- Runtime-result artifacts for the required 2 to 4 qubit microcases are emitted
  in one stable comparison-ready shape that later Task 6 external-baseline
  checks can consume directly without re-running the case through a different
  interface.
- Negative tests show that runtime-stage unsupported cases or incomplete
  execution fail explicitly with one stable structured failure vocabulary and do
  not appear in artifacts as supported partitioned-runtime success cases.
- Reproducibility artifacts for mandatory runtime cases record planner schema
  version, descriptor schema version, requested mode, source type, entry route,
  workload family, workload ID, partition count, partition-span summary,
  runtime-path classification, raw or summarized comparison outputs, and the
  runtime and memory metrics needed by later benchmark packaging.
- Traceability target: satisfy the Phase 3 Task 3 evidence requirements in
  `DETAILED_PLANNING_PHASE_3.md`.
- Traceability target: support the full-phase acceptance criteria requiring that
  the partitioned density runtime executes the mandatory benchmark matrix end to
  end and preserves exact semantics within the thresholds frozen in Section
  10.1.
- Traceability target: satisfy the runtime-minimum decision in `P3-ADR-005`,
  the correctness-first sequencing rule in `P3-ADR-006`, the frozen
  support-matrix boundary in `P3-ADR-007`, and the validation-baseline and
  benchmark-anchor decisions in `P3-ADR-008` and `P3-ADR-009`.

## Affected interfaces
- `NoisyPartitionDescriptorSet`, `NoisyPartitionDescriptor`, and
  `NoisyPartitionDescriptorMember`, plus any serialization or adapter boundary
  that carries the Task 2 schema-versioned descriptor contract into runtime
  execution.
- Any partitioned density executor, execution-plan builder, or runtime adapter
  that converts supported descriptor sets plus a parameter vector into
  partition-local execution requests.
- The sequential `NoisyCircuit` execution boundary and any comparison harness
  that treats it as the internal exact baseline for supported runtime cases.
- Continuity-anchor workload builders, mandatory microcase builders, and
  structured-family harness entry points that feed Task 3 runtime runs.
- Runtime-stage validation, preflight, and structured error-reporting surfaces
  for unsupported or incomplete partitioned execution requests.
- Benchmark, validation, and reproducibility metadata surfaces that must record
  case provenance, runtime-path classification, partition count, partition
  span, comparison outputs, and raw runtime metrics.
- Change classification: additive for supported partitioned-density workloads,
  but stricter for cases that previously might have relied on hidden sequential
  behavior or ambiguous execution labeling, which now become explicit hard
  failures or clearly labeled non-partitioned results.

## Publication relevance
- Supports Paper 2's core claim that Phase 3 delivers an executable
  partitioned density runtime rather than stopping at planner and descriptor
  representation.
- Makes correctness and performance evidence scientifically defensible by tying
  runtime results to the schema-versioned descriptor contract and to the
  sequential exact baseline.
- Provides the runtime evidence layer needed for the Phase 3 abstract, short
  paper, and full paper sections that discuss end-to-end partitioned execution,
  benchmark completeness, and honest reporting of limitations.
- Prevents the paper narrative from overstating Phase 3 by separating the
  executable partitioned baseline defined here from Task 4 fused-execution
  coverage, Task 5 calibration claims, and deferred channel-native follow-on
  work.
