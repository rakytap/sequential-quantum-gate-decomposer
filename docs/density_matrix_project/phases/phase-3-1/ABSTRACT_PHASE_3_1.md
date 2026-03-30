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
implementation composes same-support 1- and 2-qubit motifs built from `U3`,
`CNOT`, and local depolarizing, amplitude-damping, and phase-damping channels
into Kraus bundles, preserves ordered noisy-circuit semantics, and applies the
resulting CPTP object on local support within a larger density state. Internal
validation against sequential exact execution confirms Frobenius-norm agreement
below `1e-10` on a 1-qubit motif, one counted 2-qubit motif, and a 4-qubit
spectator-support smoke case, while invariant checks confirm trace preservation
and positivity under the frozen numerical policy. The present result therefore
establishes the feasibility of bounded exact channel-native fusion and a
reproducible methodology for validating such blocks, rather than a completed
acceleration claim. The remaining counted microcases, Qiskit Aer
cross-validation, and 8- and 10-qubit structured performance study are still
pending, so broader performance conclusions are intentionally withheld. The
current outcome is best read as a methods-grounding result and a reusable
foundation for future exact noisy acceleration studies.

## Keywords (Draft)

quantum simulation, density matrix, quantum channels, open quantum systems,
circuit partitioning, gate fusion, Kraus representation, exact classical
emulation

## Traceability

- `SHORT_PAPER_PHASE_3_1.md`
- `PAPER_PHASE_3_1.md`
- `DETAILED_PLANNING_PHASE_3_1.md`
