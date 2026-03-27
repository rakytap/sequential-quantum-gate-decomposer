# Pre-Implementation Completion Checklist

**Status: IMPLEMENTATION COMPLETE**

This document validated that the Phase 3 planning set was detailed enough to
start implementation. Phase 3 implementation is now complete. This checklist
serves as the historical record of the planning closure that enabled
implementation.

The Phase 3 source-of-truth contract was:

- `DETAILED_PLANNING_PHASE_3.md`
- `ADRs_PHASE_3.md`

Its purpose was:

> show that Phase 3 is specified tightly enough to begin implementation for the
> documented baseline methods claim, while keeping explicit follow-on branches
> visible.

## Current Readiness Verdict

**Phase 3 implementation is now complete.**

The planning specification was detailed enough to guide successful
implementation. The delivered implementation in `noisy_planner.py` and
`noisy_runtime.py` satisfies the frozen baseline contract documented here.

Deliberately deferred items remain explicit trade-offs documented in the ADRs
rather than hidden specification gaps.

## Closure Map

| Contract item | Contract decision that closes it | Primary decision record | Status |
|---|---|---|---|
| Canonical planner surface | canonical noisy mixed-state operation stream equivalent to `NoisyCircuit`, exact lowering allowed, no silent fallback in `partitioned_density` mode | planning planner-input decision; `P3-ADR-003` | Closed |
| Semantic preservation | partition descriptors retain gate/noise order, qubit support, parameter mapping, and noise placement as first-class semantics | planning semantic-preservation decision; `P3-ADR-004` | Closed |
| Runtime minimum and real fused execution | executable partitioned density runtime plus at least one real fused execution mode on eligible substructures | planning runtime/fused-execution decision; `P3-ADR-005` | Closed |
| Cost-model sequencing | correctness-first structural heuristics followed by benchmark-calibrated density-aware modeling | planning cost-model decision; `P3-ADR-006` | Closed |
| Support matrix | workload-driven gate/noise surface centered on the frozen Phase 2 baseline | planning support-matrix decision; `P3-ADR-007` | Closed |
| Validation baselines | sequential `NoisyCircuit` required internally and Qiskit Aer required externally on microcases | planning benchmark-minimum decision; `P3-ADR-008` | Closed |
| Benchmark anchors | Phase 2 continuity workflow plus structured noisy partitioning families and noise-placement sensitivity matrix | planning workflow/benchmark-anchor decision; `P3-ADR-009` | Closed |
| Performance claim boundary and follow-on branch rule | explicit threshold-or-diagnosis rule and deferred channel-native / Phase 4 growth branch | planning numeric-acceptance decision; `P3-ADR-010` | Closed |

## Checklist Closure Details

### 1. Canonical Planner Surface

Status: `Closed`

Closure decision:

- Phase 3 defines a canonical noisy mixed-state planner surface equivalent to an
  ordered `NoisyCircuit` operation stream.
- Mandatory workloads are judged through that canonical surface even when they
  originate from Phase 2 continuity lowering or other exact lowering paths.
- Unsupported `partitioned_density` requests hard-error explicitly before
  execution.
- Silent fallback to the sequential path is excluded from the Phase 3 contract.

Trade-off recorded:

- The planner contract is precise and aligned with the real density backend.
- Full direct parity for every source representation is deferred rather than
  implied.

### 2. Semantic Preservation

Status: `Closed`

Closure decision:

- Partition descriptors must preserve:
  - exact gate/noise order,
  - qubit support,
  - parameter routing metadata,
  - and any remapping data needed for faithful execution.
- Noise is part of the semantic model, not barrier-only metadata.
- Reordering across noise boundaries is not part of the required baseline unless
  separately documented as exact and validated.

Trade-off recorded:

- The contract is strict enough to make Paper 2 correctness claims defensible.
- More aggressive transformations remain future work rather than hidden Phase 3
  assumptions.

### 3. Runtime Minimum And Real Fused Execution

Status: `Closed`

Closure decision:

- Phase 3 must deliver an executable partitioned density runtime on the required
  benchmark matrix.
- Planner-only or representation-only closure is explicitly insufficient.
- At least one real fused execution mode is required on eligible substructures.
- Fully channel-native fused noisy blocks are deferred beyond the minimum
  closure bar.

Trade-off recorded:

- This is strong enough for a publishable methods result.
- It avoids forcing the most invasive architecture before benchmark evidence
  justifies it.

### 4. Cost-Model Sequencing

Status: `Closed`

Closure decision:

- Phase 3 implementation sequence is correctness first, structural heuristic
  second, benchmark-calibrated density model third.
- The state-vector FLOP model may be reused only as scaffolding or comparison,
  not as the Phase 3 scientific claim.
- Optional kernel tuning is allowed only when profiling shows it materially
  affects the benchmark outcome.

Trade-off recorded:

- This protects the methods claim from premature optimality language.
- It accepts that the strongest density-aware planning claim may arrive later in
  the implementation sequence than the first runnable baseline.

### 5. Support Matrix

Status: `Closed`

Closure decision:

- Required gate surface for the minimum claim remains centered on `U3` and
  `CNOT`.
- Required noise surface remains local depolarizing, local amplitude damping,
  and local phase damping or dephasing.
- Additional gates already exposed by `NoisyCircuit` are optional only when a
  mandatory microcase or clearly secondary benchmark needs them.
- Full circuit-source parity, correlated noise, calibration-aware noise, and
  readout/shot-noise workflow features are deferred.

Trade-off recorded:

- The support surface is narrow enough to keep Phase 3 about methods rather than
  feature growth.
- Broader coverage remains possible later without changing the minimum closure
  rule.

### 6. Validation Baselines

Status: `Closed`

Closure decision:

- Sequential `NoisyCircuit` execution is mandatory as the internal exact
  baseline on every required case.
- Qiskit Aer density-matrix simulation is mandatory as the external exact
  baseline on the required 2 to 4 qubit microcases and representative small
  continuity cases.
- Profiling artifacts are required when they materially support the benchmark
  interpretation or architecture decision.

Trade-off recorded:

- Internal and external correctness are both covered.
- The benchmark package remains bounded by requiring external comparison only
  where it is most valuable.

### 7. Benchmark Anchors

Status: `Closed`

Closure decision:

- The frozen Phase 2 noisy XXZ `HEA` workflow is the required continuity anchor.
- Structured noisy `U3` / `CNOT` circuit families are the required methods
  stress workloads.
- Mandatory scale coverage is:
  - external micro-validation at 2 to 4 qubits,
  - internal correctness at 4, 6, 8, and 10 qubits,
  - and mandatory performance recording at 8 and 10 qubits for the structured
    families.
- Noise-placement sensitivity is frozen as sparse, periodic, and dense local
  placement patterns.

Trade-off recorded:

- The benchmark package stays connected to the Phase 2 workflow while still
  being broad enough for Paper 2.
- It deliberately stops short of a fully general noisy algorithm benchmark zoo.

### 8. Performance Claim Boundary And Follow-On Branch Rule

Status: `Closed`

Closure decision:

- Phase 3 correctness thresholds must pass on every mandatory case.
- The performance-evidence threshold is closed by a two-path rule:
  - at least one representative required case shows measurable benefit,
  - or the benchmark plus profiling package explains why the native baseline
    still falls short and justifies the follow-on decision gate.
- Channel-native fusion, broader Phase 4 workflow growth, and approximate
  scaling remain explicit deferred branches rather than implicit Phase 3
  incompleteness.

Trade-off recorded:

- The phase keeps a meaningful performance expectation without forcing an
  unrealistic universal speedup claim.
- Negative or mixed benchmark results can still produce a valid Phase 3 methods
  outcome if they are benchmark-grounded and honestly framed.

## Resolution Outcome

The Phase 3 planning set now closes the main Layer 1 questions directly inside
the detailed planning and ADR records instead of leaving them as vague future
TODOs.

This is better for implementation because:

- the source-of-truth planning document contains the frozen contract,
- the ADRs capture the trade-offs and rejected alternatives,
- and this checklist can point to accepted decisions instead of open design
  ambiguity.

## Go / No-Go Rule

**Implementation is complete.**

The Phase 3 baseline contract has been fully implemented. The following
interpretation remains relevant for future work:

- requests outside the documented support matrix should be treated as new
  scope decisions or follow-on ADRs,
- channel-native fused noisy blocks remain a benchmark-driven follow-on branch,
  not a hidden prerequisite,
- and broader VQE/VQA workflow growth belongs to Phase 4.

## Final Practical Assessment

The checklist was solidly closed for the Phase 3 baseline contract, and
implementation proceeded successfully.

Phase 3 delivered the documented methods scope. The benchmark outcome confirmed
diagnosis-grounded performance closure rather than positive-threshold closure on
representative workloads.
