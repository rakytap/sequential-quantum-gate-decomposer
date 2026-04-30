# Abstract for Phase 3.1

## Draft Status

Decision-study closure draft aligned to
`PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md`, which now records the frozen
Phase 3.1 v1 slice as `decision-study-ready`. This abstract is therefore the
Task 5 claim-closure version for the bounded decision-study outcome, not a
placeholder awaiting further evidence.

## Text (Draft)

Exact density-matrix simulation is the most reliable classical baseline for
studying how realistic local noise changes variational quantum circuits, yet
most partitioning and fusion methods still stop at unitary subcircuits. This
work studies a bounded exact alternative: channel-native fusion of small mixed
gate-noise motifs inside partitioned density-matrix execution. The method uses
two execution interpretations. A **strict** motif-proof path treats same-
support 1- and 2-qubit motifs built from `U3`, `CNOT`, and local depolarizing,
amplitude-damping, and phase-damping channels as exact Kraus-bundle CPTP
objects. An explicit **hybrid** whole-workload path routes eligible partitions
channel-natively and keeps Phase-3-supported but Phase-3.1-ineligible
partitions on the shipped exact baseline with route attribution. Validation now
spans both layers. On the strict path, the bounded counted microcase surface
agrees with the sequential oracle below `1e-10`, while invariant checks confirm
trace preservation and positivity under the frozen numerical policy. On the
hybrid path, both counted continuity anchors also match the sequential oracle
within the same threshold, and the required bounded Qiskit Aer slice is present
for the four strict microcases plus `phase2_xxz_hea_q4_continuity`. The full
frozen 26-row whole-workload matrix now records route-aware comparisons against
the sequential reference and the shipped Phase 3 fused baseline. That matrix
supports a bounded **decision-study** conclusion rather than a positive-methods
closure: `17` rows are classified `phase3_sufficient`, `9` rows are classified
`phase31_not_justified_yet`, and `0` rows are classified `phase31_justified`.
The present result therefore establishes a bounded exactness-and-decision-study
contribution: exact channel-native fusion is feasible and scientifically
auditable on the frozen motif slice, but the stronger performance-justification
threshold is not met on the frozen whole-workload matrix.

## Keywords (Draft)

quantum simulation, density matrix, quantum channels, open quantum systems,
circuit partitioning, gate fusion, Kraus representation, exact classical
emulation

## Traceability

- `PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md`
- `CLOSURE_PLAN_PHASE_3_1.md`
- `SHORT_PAPER_PHASE_3_1.md`
- `PAPER_PHASE_3_1.md`
- `DETAILED_PLANNING_PHASE_3_1.md`
