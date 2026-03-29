# Full Paper Outline: Phase 3.1 Bounded Channel-Native / Superoperator Fusion

## Draft Status

Skeleton only. Sections will be expanded after implementation and evidence
bundles exist. No venue-specific formatting yet.

## Title Candidates

- `Bounded Exact Channel-Native Fusion for Noise-Dense Motifs in Partitioned Density-Matrix Simulation`
- `Beyond Unitary Islands: Exact Mixed-Motif Channel Fusion After Noise-Aware Partitioning`

## 1. Introduction

- Motivation: exact noisy simulation cost; Phase 3 contribution recap.
- Gap: unitary-island fusion stops at noise boundaries; dense same-support noisy
  motifs remain fragmented.
- Contributions (planned and now contract-aligned):
  - a bounded exact fused object for 1- and 2-qubit mixed gate+noise motifs,
  - a channel-native execution path inside the partitioned noisy runtime,
  - exactness plus CPTP-invariant validation,
  - and a counted benchmark slice designed to show benefit beyond the Phase 3
    fused baseline.

## 2. Background

- Density-matrix simulation and local noise models.
- Quantum-channel representations: Kraus, Choi, PTM, Liouville / superoperator.
- Primary counted-claim choice for Phase 3.1:
  - `kraus_bundle`, with Liouville / superoperator matrices as internal
    acceleration views only after equivalence is demonstrated.
- Partitioning and fusion literature pointer (see project `REFERENCES.md`).
- Phase 3 summary and explicit deferral of channel-native fusion.

## 3. Semantics and Representation

- Ordered noisy circuits as reference.
- Bounded v1 claim surface:
  - primitive gates `U3`, `CNOT`,
  - primitive noise: local depolarizing, local amplitude damping, local phase
    damping / dephasing,
  - contiguous 1- and 2-qubit mixed motifs,
  - at least one noise operation per counted fused block.
- Choice of primary channel representation:
  - `kraus_bundle` for the counted claim surface,
  - Liouville / superoperator matrices allowed as non-primary apply/cache forms.
- Composition, trace preservation, positivity / CPTP invariants, and numerical
  stability considerations.

## 4. Architecture (Planned)

- Integration with canonical planner surface and descriptors (high level).
- Eligibility and unsupported tiers.
- No silent fallback for advertised channel-native modes.
- Relationship to the shipped Phase 3 fused path:
  - pure unitary islands stay on the Phase 3 mechanism,
  - mixed noisy motifs are the new counted object.

## 5. Experimental Setup

- Baselines:
  - sequential `NoisyCircuit`,
  - Phase 3 partitioned+fused,
  - Phase 3.1 channel-native.
- Counted correctness slice:
  - `phase31_microcase_1q_u3_local_noise_chain`,
  - `phase31_microcase_2q_cnot_local_noise_pair`,
  - `phase31_microcase_2q_multi_noise_entangler_chain`,
  - `phase31_microcase_2q_dense_same_support_motif`,
  - `phase2_xxz_hea_q4_continuity`,
  - `phase2_xxz_hea_q6_continuity`,
  - and every counted performance case.
- Counted performance slice:
  - primary families:
    - `phase31_pair_repeat`,
    - `phase31_alternating_ladder`,
  - control family:
    - `layered_nearest_neighbor`,
  - qubit counts `{8, 10}`,
  - primary-family patterns `{periodic, dense}`,
  - control pattern `sparse`,
  - seed policy `{20260318, 20260319, 20260320}` for primary families and
    `20260318` for control cases.
- Metrics:
  - runtime,
  - peak memory,
  - planning time,
  - exactness tolerances,
  - CPTP-invariant checks,
  - and decision-surface classification.

## 6. Results (Placeholder)

- Correctness summary table:
  - internal exactness,
  - external micro-validation where required,
  - CPTP invariant checks.
- Performance summary:
  - all three baselines,
  - positive-method threshold check relative to Phase 3 fused baseline
    (`>= 1.2x` runtime or `>= 15%` memory on at least one representative
    primary-family case),
  - `break_even_table` / `justification_map`.
- Negative-result branch (if needed):
  - where the richer fused object is not justified yet.

## 7. Discussion

- When bounded mixed-motif channel fusion helps vs does not.
- Why the counted win, if present, is a methods result rather than merely a
  representation change.
- Implications for training-scale studies and Phase 4 priorities.

## 8. Related Work

- Defer to `REFERENCES.md` curation; align with Side Paper A framing.

## 9. Conclusion

- Planned closing claims tied to evidence only.
- Primary intended closing claim:
  - bounded exact channel-native fusion helps on a narrow same-support noisy
    motif slice beyond the shipped Phase 3 fused baseline.
- Fallback closing claim:
  - the richer fused object does not yet justify its complexity on the counted
    slice, with benchmark-grounded diagnosis.

## Reproducibility

- Point to benchmark harness paths and configuration logging (to be filled
  after implementation).
- Mandatory bundles remain scalar-only for v1; Task 6 host acceleration, if it
  exists later, stays out of the main counted paper claim.

## Traceability

- `DETAILED_PLANNING_PHASE_3_1.md`
- `ADRs_PHASE_3_1.md`
- `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`
