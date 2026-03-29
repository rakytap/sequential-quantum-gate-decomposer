# Abstract for Phase 3.1

## Draft Status

Planning-phase abstract draft for conference submission or thesis chapter
framing. Revise after Phase 3.1 evidence exists.

## Text (Draft)

Phase 3 established that noisy mixed-state circuits can be first-class inputs to
partition planning and partitioned exact density execution in SQUANDER, with a
real fused path based on unitary islands inside noisy partitions. Phase 3.1
targets the next bounded methods step: **exact channel-native /
superoperator-native fusion of contiguous 1- and 2-qubit mixed gate+noise
motifs** built from `U3`, `CNOT`, and local single-qubit noise channels on the
same support. The intended contribution is not merely a new representation, but
an exact fused execution path for small CPTP blocks that preserves ordered
`NoisyCircuit` semantics while reducing the overhead created when noise
boundaries fragment the Phase 3 fused baseline. The counted evaluation slice is
therefore narrow and reproducible: exact agreement with sequential density
simulation under `<= 1e-10` Frobenius-norm thresholds, representation-level
CPTP invariants, bounded continuity anchors, and motif-dense 8- and 10-qubit
benchmark families compared against sequential execution and the shipped Phase 3
partitioned+fused baseline. The primary paper target is a bounded positive
methods result: at least one representative motif-dense case should show
`>= 1.2x` median wall-clock speedup or `>= 15%` peak-memory reduction versus
the Phase 3 fused baseline without correctness loss. If this threshold is not
met, the same package closes honestly as a diagnosis-grounded negative result.
Broader workflow expansion, correlated noise, support beyond 2-qubit fused
motifs, and approximate scaling methods remain out of scope.

## Keywords (Draft)

quantum simulation, density matrix, noise channels, circuit partitioning, gate
fusion, superoperator, CPTP maps, exact classical emulation

## Traceability

- `SHORT_PAPER_PHASE_3_1.md`
- `PAPER_PHASE_3_1.md`
- `DETAILED_PLANNING_PHASE_3_1.md`
