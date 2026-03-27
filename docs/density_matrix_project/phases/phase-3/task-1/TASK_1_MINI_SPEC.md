# Task 1: Canonical Noisy Planner Surface

**Implementation Status: COMPLETE**

This mini-spec defined the Phase 3 Task 1 implementation contract. The canonical
noisy planner surface is now implemented in `squander/partitioning/noisy_planner.py`
as the `CanonicalNoisyPlannerSurface` dataclass and associated builder functions.

This document inherits the frozen phase decisions from `DETAILED_PLANNING_PHASE_3.md`,
`P3-ADR-003`, `P3-ADR-007`, and `P3-ADR-009`, plus the closed canonical
planner-surface, support-matrix, and benchmark-anchor items in
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. It does not reopen
semantic-preservation, runtime-minimum, cost-model, or broader Phase 4
workflow-growth decisions.

## Given / When / Then
- Given a request that claims `partitioned_density` behavior and a circuit
  source inside the frozen Phase 3 support surface.
- When that request is normalized into the Phase 3 planner input contract
  before partition planning begins.
- Then the planner receives an auditable ordered noisy mixed-state operation
  sequence equivalent to `NoisyCircuit` content built from `GateOperation` and
  `NoiseOperation`, and unsupported requests fail before execution with
  deterministic diagnostics and no silent fallback.

## Assumptions and dependencies
- Task 1 defines the canonical planner entry surface and the supported entry
  paths into it. It does not by itself define partition descriptor metadata,
  execution scheduling, fused kernels, or performance claims.
- Task 2 depends on Task 1 by adding the semantic-preservation metadata and
  exactness obligations that partition descriptors must retain once the
  canonical surface exists.
- Task 3 depends on Task 1 by executing partitioned density workloads from this
  canonical surface rather than from source-specific ad hoc adapters.
- The required continuity input is the frozen Phase 2 noisy XXZ `HEA`
  workflow, which must lower exactly into the canonical surface without
  implying new Phase 4 workflow growth.
- The required methods-oriented inputs are the mandatory Phase 3 structured
  noisy `U3` / `CNOT` families and the required 2 to 4 qubit micro-validation
  cases.
- The frozen mandatory gate surface is `U3` and `CNOT`; the frozen mandatory
  noise surface is local single-qubit depolarizing, local amplitude damping,
  and local phase damping or dephasing.
- Additional exact lowering from `qgd_Circuit` or `Gates_block` sources is
  allowed when it stays inside the documented support matrix, but full direct
  parity for every circuit source is not assumed.
- The sequential `NoisyCircuit` density path remains the internal exact
  baseline for validation, but it is a reference oracle rather than an implicit
  fallback path for `partitioned_density`.
- Internal planner data structures may introduce auxiliary DAG, graph, or
  partition-preparation views, but the contract-defining surface must remain
  traceable to the canonical ordered noisy operation sequence.
- This task freezes representational scope for Phase 3. It does not widen
  support to correlated noise, readout or shot-noise workflows,
  calibration-aware noise, approximate scaling, or broader noisy VQE/VQA
  feature growth.

## Required behavior
- Task 1 freezes a canonical internal planner input contract for Phase 3 rather
  than leaving planner semantics implicit in legacy state-vector structures or
  source-specific adapters.
- The canonical Phase 3 planner surface is an ordered noisy mixed-state
  operation sequence equivalent to `NoisyCircuit` operations built from
  `GateOperation` and `NoiseOperation`.
- Gate operations and noise operations are first-class planner objects. Noise
  placement must not be reduced to barrier-only annotations, opaque side
  metadata, or undocumented boundary markers.
- Every mandatory Phase 3 workload must reach the canonical surface before
  partition planning begins.
- The required positive entry paths include:
  - exact lowering of the frozen Phase 2 noisy XXZ `HEA` continuity workflow,
  - representation of the mandatory structured noisy `U3` / `CNOT` benchmark
    families,
  - and representation of the required 2 to 4 qubit micro-validation cases.
- Lowering from existing source surfaces such as `qgd_Circuit` or
  `Gates_block` is acceptable only when the lowered result is exact, auditable,
  and remains inside the documented support matrix. The Phase 3 claim is judged
  on the resulting canonical noisy surface rather than on source parity.
- Unsupported circuit sources, unsupported gate families, unsupported noise
  models, or unsupported `partitioned_density` requests must hard-error before
  execution begins.
- No benchmark or validation case may claim `partitioned_density` behavior if
  execution silently falls back to the sequential density path, the
  state-vector path, or a hidden non-partitioned substitute.
- The supported-path contract must make the entry route auditable in tests,
  validation outputs, and benchmark artifacts, for example whether a case came
  from Phase 2 continuity lowering, structured family construction, or another
  exact lowering path.
- Task 1 completion means supported Phase 3 entry paths converge onto one
  canonical planner surface with explicit unsupported boundaries. It does not
  require partition execution, fused runtime behavior, or density-aware cost
  modeling to be complete.

## Unsupported behavior
- Defining the Phase 3 planner contract implicitly around `qgd_Circuit`,
  `Gates_block`, or other state-vector-first source semantics alone.
- Treating noise channels as external annotations, partition-boundary markers,
  or other non-first-class metadata outside the canonical planner surface.
- Claiming full direct parity for every circuit source as part of Task 1
  closure.
- Silently partially lowering a source circuit, silently dropping unsupported
  operations, or silently rewriting a request into a different supported form.
- Silent fallback from `partitioned_density` to sequential density execution,
  state-vector execution, or any undocumented compatibility path.
- Warning-only or best-effort handling for unsupported source, gate, or noise
  requests when the request claims supported Phase 3 planner behavior.
- Using an internal DAG or auxiliary representation in a way that obscures the
  canonical ordered gate/noise sequence or makes the planner input contract
  unauditable.
- Counting broader Phase 4 workflows, calibration-aware noise, correlated
  multi-qubit noise, readout/shot-noise workflows, or approximate scaling
  methods as required Task 1 support.

## Acceptance evidence
- Planner-normalization tests or equivalent validation show that the frozen
  Phase 2 noisy XXZ `HEA` workflow lowers into the canonical noisy surface
  without changing the supported gate/noise order.
- Positive representability tests show that required 2 to 4 qubit
  micro-validation cases and at least one required structured noisy `U3` /
  `CNOT` family instance enter the same canonical planner surface.
- Interface or integration evidence shows that supported `partitioned_density`
  requests expose auditable gate and noise operations as first-class planner
  objects rather than as boundary-only metadata.
- Negative tests show that unsupported circuit sources, unsupported gates,
  unsupported noise models, or malformed `partitioned_density` requests fail
  before execution and do not fall back silently.
- Reproducibility artifacts for mandatory cases record the entry path, support
  labels, canonical-surface summary or equivalent audit trail, and
  unsupported-case diagnostics for negative evidence.
- Traceability target: satisfy the Phase 3 Task 1 evidence requirements in
  `DETAILED_PLANNING_PHASE_3.md`.
- Traceability target: support the full-phase acceptance criteria requiring
  exact supported-case semantics, explicit unsupported boundaries, and no
  silent fallback for `partitioned_density`.
- Traceability target: satisfy the canonical planner-surface decision in
  `P3-ADR-003`, the frozen support-matrix boundary in `P3-ADR-007`, and the
  continuity-plus-structured-workload anchor in `P3-ADR-009`.

## Affected interfaces
- `NoisyCircuit`, `GateOperation`, and `NoiseOperation` or the equivalent
  canonical planner-input adapter boundary.
- Any lowering or normalization surface that routes the frozen Phase 2
  continuity workflow into partition planning.
- Phase 3 structured noisy circuit builders, fixtures, or harness entry points
  that must create or lower required benchmark families into the canonical
  surface.
- Pre-execution validation and error-reporting paths for unsupported source,
  gate, noise, or mode requests.
- Benchmark and reproducibility metadata surfaces that must record how a case
  reached the canonical planner contract.
- Change classification: additive for supported workloads that now converge on a
  shared canonical planner surface, but stricter for ambiguous or unsupported
  requests, which become explicit hard failures rather than undocumented
  behavior.

## Publication relevance
- Supports Paper 2's core methods claim that Phase 3 partitions canonical noisy
  mixed-state circuits rather than state-vector circuits with externally
  attached noise annotations.
- Makes benchmark attribution scientifically defensible by ensuring every
  reported `partitioned_density` result is tied to one auditable planner-input
  contract.
- Provides contract evidence for the canonical noisy planner-surface sections in
  the Phase 3 abstract, short paper, and full paper.
- Prevents the paper narrative from overstating circuit-source generality by
  clearly separating the guaranteed canonical support surface from deferred
  full-source parity.
