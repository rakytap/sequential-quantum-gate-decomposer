# Abstract for Phase 3.1

## Draft Status

Implementation-backed abstract revised to match the current Phase 3.1 state:
bounded channel-native execution is real on the strict slice, and the hybrid
whole-workload path now has an initial correctness anchor plus one structured
pilot row. Broader correctness closure and the full structured performance
matrix are still pending.

## Text (Draft)

Exact density-matrix simulation is the most reliable classical baseline for
studying how realistic local noise changes variational quantum circuits, yet
most partitioning and fusion methods still stop at unitary subcircuits. This
work studies a bounded exact alternative: channel-native fusion of small mixed
gate-noise motifs inside partitioned density-matrix execution. The current
method uses two execution interpretations. A **strict** motif-proof path treats
same-support 1- and 2-qubit motifs built from `U3`, `CNOT`, and local
depolarizing, amplitude-damping, and phase-damping channels as exact Kraus-
bundle CPTP objects. An explicit **hybrid** whole-workload path routes eligible
partitions channel-natively and keeps Phase-3-supported but Phase-3.1-ineligible
partitions on the shipped exact baseline with route attribution. Current
implementation-backed validation now spans both layers. On the strict path,
Frobenius-norm agreement below `1e-10` is observed on a 1-qubit motif, one
counted 2-qubit motif, and a 4-qubit fully eligible local-support smoke case,
while invariant checks confirm trace preservation and positivity under the
frozen numerical policy. On the hybrid path, a counted 4-qubit continuity
anchor also matches the sequential oracle within the same threshold, and one
frozen 8-qubit structured pilot row records route coverage plus sequential,
Phase 3 fused, and hybrid timings. That pilot currently remains slower than the
existing Phase 3 fused baseline, so the present result is a bounded exactness-
and-decision-study contribution rather than a closed acceleration claim.
Publication-backed coverage of the remaining strict rows, the 6-qubit
continuity anchor, external Qiskit Aer comparison, and the full structured
8- and 10-qubit matrix are still pending.

## Keywords (Draft)

quantum simulation, density matrix, quantum channels, open quantum systems,
circuit partitioning, gate fusion, Kraus representation, exact classical
emulation

## Traceability

- `SHORT_PAPER_PHASE_3_1.md`
- `PAPER_PHASE_3_1.md`
- `DETAILED_PLANNING_PHASE_3_1.md`
