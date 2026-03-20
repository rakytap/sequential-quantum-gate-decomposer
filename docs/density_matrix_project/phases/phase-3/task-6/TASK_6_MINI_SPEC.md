# Task 6: Correctness Validation And Unsupported-Boundary Evidence

This mini-spec turns Phase 3 Task 6 into an implementation-ready contract. It
inherits the frozen phase decisions from `DETAILED_PLANNING_PHASE_3.md`,
`P3-ADR-003`, `P3-ADR-004`, `P3-ADR-005`, `P3-ADR-006`, `P3-ADR-007`,
`P3-ADR-008`, `P3-ADR-009`, and `P3-ADR-010`, plus the closed canonical
planner-surface, semantic-preservation, runtime-minimum, cost-model
sequencing, support-matrix, validation-baseline, benchmark-anchor, and
performance-claim-boundary items in
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. It does not reopen the frozen
support surface, the frozen numeric thresholds, the bounded Task 5 calibrated
planner claim, or the deferred channel-native / broader Phase 4 branches.

## Given / When / Then
- Given a supported `CanonicalNoisyPlannerSurface`, an auditable
  `NoisyPartitionDescriptorSet`, an executable partitioned density runtime with
  real fused execution available on at least one eligible substructure, and a
  benchmark-calibrated Task 5 planner surface or bounded candidate family on
  the frozen workload matrix.
- When Phase 3 counts a `partitioned_density` case as positive evidence or uses
  it to support an acceleration, calibration, benchmark, or publication claim.
- Then the evidence package must prove exact agreement with the sequential
  `NoisyCircuit` baseline on the required internal correctness matrix, prove
  agreement with Qiskit Aer on the required external slice, and keep
  planner-entry, descriptor-generation, and runtime-only unsupported or
  deferred behavior explicitly separated in one machine-reviewable correctness
  package.

## Assumptions and dependencies
- Task 1 provides the canonical Phase 3 planner-entry contract as a
  schema-versioned `CanonicalNoisyPlannerSurface` with stable provenance fields
  for `requested_mode`, `source_type`, `entry_route`, `workload_family`, and
  `workload_id`, plus explicit planner-entry unsupported evidence.
- Task 2 provides the schema-versioned partition handoff contract through
  `NoisyPartitionDescriptorSet`, `NoisyPartitionDescriptor`, and
  `NoisyPartitionDescriptorMember`, including canonical operation references,
  qubit-remapping metadata, parameter-routing metadata, and the structured
  descriptor-generation unsupported or lossy vocabulary.
- Task 3 provides the first executable partitioned density runtime that
  consumes the Task 2 contract directly, records auditable runtime-path
  classification, and reports runtime-stage unsupported cases without silent
  fallback.
- Task 4 provides the minimum real fused-execution baseline plus explicit
  classification of fused, supported-but-unfused, and deferred fusion
  candidates. Task 6 validates those runtime-path labels rather than replacing
  them with one generic correctness status.
- Task 5 provides the benchmark-calibrated planner surface or bounded candidate
  family plus reusable calibration bundles or rerunnable checkers. Task 6
  validates that the counted positive cases from that calibrated surface still
  preserve the frozen exact noisy semantics and honest claim boundary.
- Task 7 depends on Task 6 by rolling per-case correctness and
  unsupported-boundary records into benchmark summaries without losing stable
  provenance or failure-stage separation.
- Task 8 depends on Task 6 by packaging positive supported evidence and
  negative unsupported-boundary evidence into a manifest-driven Paper 2 bundle.
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
  oracle on every mandatory Phase 3 case. Qiskit Aer density-matrix simulation
  remains the required external exact reference on the 2 to 4 qubit microcases
  and representative small continuity subset. Both are validation baselines,
  not hidden fallback paths.
- Stable workload IDs, deterministic seed rules, noise-pattern labels,
  planner-setting references, runtime-path labels, and machine-reviewable
  bundles or rerunnable checkers are already part of the frozen Phase 3
  evidence contract and must remain reusable here.
- Rolled-up validation or benchmark summaries that drive architecture or paper
  conclusions must be checked against the underlying per-case records rather
  than treated as self-justifying.
- This task defines the minimum correctness package required before acceleration
  or calibrated-planner conclusions are treated as valid. It does not by itself
  widen the support matrix, replace Task 7 benchmark interpretation, or
  replace Task 8 publication packaging.

## Required behavior
- Task 6 freezes the minimum correctness-evidence contract required before any
  Phase 3 runtime benefit, fusion benefit, or density-aware planning claim is
  treated as scientifically valid.
- Every counted positive `partitioned_density` case must be validated against
  the sequential `NoisyCircuit` baseline. A run that lacks an internal exactness
  verdict cannot contribute to a positive benchmark, calibration, or publication
  summary.
- The required internal correctness matrix spans every mandatory Phase 3
  workload class used for supported evidence, including the 2 to 4 qubit
  micro-validation circuits, the 4 / 6 / 8 / 10 qubit continuity-anchor cases,
  and the required 8 / 10 qubit structured noisy partitioning families.
- The required external correctness slice covers all mandatory 2 to 4 qubit
  microcases and the representative small continuity subset frozen by the phase
  contract. Any additional case used as external-baseline evidence must preserve
  the same provenance and threshold rules.
- Frozen numeric thresholds must remain explicit and machine-checked:
  - maximum Frobenius-norm density difference `<= 1e-10` against Qiskit Aer on
    the required external slice,
  - maximum Frobenius-norm density difference `<= 1e-10` between partitioned
    and sequential execution on all mandatory internal correctness cases,
  - `|Tr(rho) - 1| <= 1e-10` and `rho.is_valid(tol=1e-10)` on recorded required
    outputs,
  - and maximum absolute energy error `<= 1e-8` on the required 4 / 6 / 8 /
    10 qubit continuity-anchor cases.
- Pass or fail interpretation is strict rather than percentile-based. The
  required external microcases and the mandatory internal correctness matrix
  must achieve `100%` pass among counted supported cases.
- Positive correctness evidence may count only runs that execute through the
  documented partitioned runtime surface under explicit `partitioned_density`
  behavior. Silent fallback to the sequential density path, the state-vector
  path, or another hidden non-partitioned substitute is not valid evidence.
- Fused-path cases that are counted as positive evidence must satisfy the same
  correctness thresholds as unfused supported cases. Supported-but-unfused and
  deferred fusion candidates must remain explicitly labeled rather than being
  reported as fused success by implication.
- Unsupported-boundary evidence must remain layered. At minimum, the correctness
  package must distinguish:
  - planner-entry unsupported cases,
  - descriptor-generation unsupported or lossy cases,
  - and runtime-stage unsupported or deferred cases.
- Every negative evidence record must preserve one stable machine-reviewable
  vocabulary for failure stage, unsupported category, first unsupported
  condition, case provenance, and exclusion reason so later benchmark and paper
  layers do not need to infer why a case was excluded.
- A supported case may be counted only if its provenance is auditable. At
  minimum that means the evidence records the workload family, workload ID,
  source type, entry route, requested mode, seed or deterministic construction
  rule, noise-pattern label, planner-setting reference, partition summary, and
  runtime-path classification together with the comparison verdicts.
- Task 6 may consume Task 5 calibration bundles, Task 4 fused audit bundles, or
  equivalent earlier artifacts as inputs, but it must emit one stable
  correctness package or rerunnable checker that downstream consumers can use
  directly without reconstructing counted-status rules from unrelated outputs.
- Summary tables, rolled-up benchmark counts, and publication-facing figures
  that depend on Task 6 must be checked against the underlying per-case records.
  Summary-only evidence is insufficient for supported correctness claims.
- Task 6 completion means one reusable correctness package exists that gates
  later Task 7 benchmark conclusions and Task 8 publication packaging. It does
  not require universal simulator parity, broader support-matrix expansion, or
  closure of the deferred channel-native / approximate branches.

## Unsupported behavior
- Claiming Phase 3 acceleration, fused-runtime benefit, or calibrated-planner
  success from cases that lack recorded sequential-baseline correctness verdicts.
- Treating the Task 5 calibration bundle by itself as the Task 6 correctness
  package without explicit internal and external validation results.
- Validating only against the internal sequential baseline or only against
  Qiskit Aer instead of preserving the required two-baseline model.
- Counting cases that fail the frozen Frobenius, trace, density-validity, or
  continuity-anchor energy thresholds as positive supported evidence.
- Counting runs with silent fallback, unlabeled runtime path, incomplete
  provenance, or summary-only status as supported correctness evidence.
- Collapsing planner-entry, descriptor-generation, and runtime-stage
  unsupported or deferred behavior into one generic failure bucket, or hiding
  those cases by filtering them out of the emitted evidence package.
- Reporting supported-but-unfused or deferred fusion candidates as if they were
  positive fused-path correctness evidence.
- Cherry-picking favorable subsets, ad hoc exploratory workloads, or manually
  selected reruns while presenting them as the frozen mandatory correctness
  matrix.
- Letting rolled-up benchmark counts or paper figures drift from the underlying
  per-case records without a machine-checkable summary-consistency rule.
- Using Task 6 to imply support for correlated noise, readout or shot-noise
  workflow features, calibration-aware workflow growth, broader noisy VQE/VQA
  scope, channel-native fused noisy blocks, or approximate scaling branches.

## Acceptance evidence
- A documented correctness matrix identifies the mandatory micro-validation,
  continuity, and structured-family cases, gives them stable workload IDs,
  records seed rules and noise-pattern labels, and marks which cases belong to
  the required internal-only versus internal-plus-external validation slices.
- One stable machine-reviewable correctness package or rerunnable checker exists
  and records, for each case:
  - correctness-package or schema version,
  - workload family and workload ID,
  - requested mode,
  - source type and entry route,
  - seed or deterministic construction rule,
  - noise-pattern label,
  - planner-setting or calibration reference,
  - partition-count or partition-span summary,
  - runtime-path and fusion classification,
  - internal baseline verdicts and raw comparison metrics,
  - external baseline verdicts and raw comparison metrics where required,
  - continuity-anchor energy comparison where required,
  - case status,
  - failure stage,
  - unsupported category,
  - first unsupported condition,
  - exclusion reason when not counted,
  - and software version or commit.
- Counted supported cases satisfy the frozen internal exactness thresholds
  against the sequential density baseline on the mandatory correctness matrix:
  - maximum Frobenius-norm density difference `<= 1e-10`,
  - `|Tr(rho) - 1| <= 1e-10`,
  - and recorded outputs satisfy `rho.is_valid(tol=1e-10)`.
- The required 2 to 4 qubit external microcases and representative small
  continuity cases that belong to the external slice satisfy maximum
  Frobenius-norm density difference `<= 1e-10` against Qiskit Aer.
- The required 4 / 6 / 8 / 10 qubit continuity-anchor cases satisfy the frozen
  maximum absolute energy error threshold `<= 1e-8`.
- Positive cases that exercise a real fused path satisfy the same correctness
  thresholds as unfused supported cases and preserve explicit fused,
  supported-but-unfused, and deferred classification in the emitted artifacts.
- Unsupported-boundary artifacts cover planner-entry unsupported cases,
  descriptor-generation unsupported or lossy cases, and runtime-stage
  unsupported or deferred behavior while preserving failure stage, unsupported
  category, first unsupported condition, case provenance, and exclusion reason.
- A pass-rate or completeness checker confirms `100%` pass on the mandatory
  external microcases and the mandatory internal correctness matrix for counted
  supported cases.
- A summary-consistency checker or equivalent manifest rule validates that any
  rolled-up counts, benchmark summaries, or publication-facing status tables
  agree with the underlying per-case records.
- Reproducibility artifacts make Task 5 calibration outputs and Task 6
  correctness judgments joinable through stable workload and planner-setting
  identifiers rather than manual relabeling.
- Traceability target: satisfy the Phase 3 Task 6 evidence requirements in
  `DETAILED_PLANNING_PHASE_3.md`.
- Traceability target: support the full-phase acceptance criteria requiring
  exact agreement with the sequential density baseline on the mandatory
  correctness matrix and agreement with Qiskit Aer on the required external
  slice before any benchmark claim is treated as valid.
- Traceability target: satisfy the canonical planner-surface and
  semantic-preservation dependencies in `P3-ADR-003` and `P3-ADR-004`, the
  runtime and real-fused baseline in `P3-ADR-005`, the correctness-first claim
  sequencing in `P3-ADR-006`, the frozen support-matrix boundary in
  `P3-ADR-007`, the two-baseline validation rule in `P3-ADR-008`, the benchmark
  anchor decision in `P3-ADR-009`, and the deferred follow-on boundary in
  `P3-ADR-010`.

## Affected interfaces
- `CanonicalNoisyPlannerSurface` and any planner-entry provenance or
  unsupported-evidence surface that determines whether a request is eligible for
  counted Task 6 correctness evidence.
- `NoisyPartitionDescriptorSet`, `NoisyPartitionDescriptor`, and
  `NoisyPartitionDescriptorMember`, plus any descriptor-generation audit or
  unsupported-boundary surfaces that must remain joinable to runtime evidence.
- The partitioned density runtime outputs, runtime-path classification, fused
  execution classification, and per-case comparison records consumed by the
  correctness package.
- Sequential `NoisyCircuit` comparison tooling and the Qiskit Aer comparison
  harness used for the required two-baseline validation model.
- Task 5 calibration bundles or rerunnable checkers that Task 6 must consume
  through stable workload and planner-setting identifiers.
- Validation manifests, summary-consistency checkers, and machine-reviewable
  correctness or unsupported-boundary bundles later consumed by Task 7 and Task
  8.
- Benchmark and publication reporting surfaces that need explicit counted versus
  excluded status plus supported versus unsupported boundary labels.
- Change classification: additive for validation, checker, and artifact-package
  surfaces, but stricter for case-status semantics because only thresholded
  supported cases may close correctness or performance claims.

## Publication relevance
- Supports Paper 2's core claim that native noisy partitioning and fused
  execution preserve exact mixed-state semantics on the frozen workload matrix
  before any acceleration conclusion is drawn.
- Supplies the two-baseline validation package needed for the Phase 3 abstract,
  short paper, and full paper to remain scientifically defensible.
- Makes negative evidence publishable by separating planner-entry,
  descriptor-generation, and runtime-only unsupported or deferred boundaries
  instead of hiding them inside one generic failure category.
- Prevents overclaiming by forcing benchmark summaries and calibrated-planner
  conclusions to reuse one auditable correctness package with stable case
  identities and machine-checkable counted-status rules.
