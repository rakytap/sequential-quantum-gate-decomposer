# Abstract for Phase 3.1

## Draft Status

Implementation-backed abstract revised to match the current Phase 3.1 state:
bounded channel-native execution is real and internally validated on slice
cases, while the full counted correctness and performance packages are still
pending.

## Text (Draft)

Exact density-matrix simulation is the most reliable classical baseline for
studying how realistic local noise changes variational quantum circuits, yet
most partitioning and fusion methods still stop at unitary subcircuits. This
work studies a bounded exact alternative: channel-native fusion of small mixed
gate-noise motifs inside partitioned density-matrix execution. The current
method uses two execution interpretations. A **strict** motif-proof path treats
same-support 1- and 2-qubit motifs built from `U3`, `CNOT`, and local
depolarizing, amplitude-damping, and phase-damping channels as exact Kraus-
bundle CPTP objects. An explicit **hybrid** whole-workload path is the intended
evaluation vehicle for continuity and structured benchmark cases, where eligible
partitions run channel-natively and Phase-3-supported but Phase-3.1-ineligible
partitions remain on the shipped exact baseline with route attribution.
Current implementation-backed validation covers the strict path: Frobenius-norm
agreement below `1e-10` is observed on a 1-qubit motif, one counted 2-qubit
motif, and a 4-qubit fully eligible local-support smoke case, while invariant
checks confirm trace preservation and positivity under the frozen numerical
policy. The present result therefore establishes the feasibility of bounded
exact channel-native fusion and a reproducible methodology for validating such
blocks, rather than a completed acceleration claim. Publication-backed coverage
of the remaining counted rows, external Qiskit Aer comparison, and the hybrid
8- and 10-qubit structured performance study are still pending, so broader
performance conclusions are intentionally withheld.

## Keywords (Draft)

quantum simulation, density matrix, quantum channels, open quantum systems,
circuit partitioning, gate fusion, Kraus representation, exact classical
emulation

## Traceability

- `SHORT_PAPER_PHASE_3_1.md`
- `PAPER_PHASE_3_1.md`
- `DETAILED_PLANNING_PHASE_3_1.md`
