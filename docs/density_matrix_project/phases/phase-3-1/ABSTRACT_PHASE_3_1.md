# Abstract for Phase 3.1

## Draft Status

Pre-Task-5 abstract draft aligned to `CLOSURE_PLAN_PHASE_3_1.md`. This
document is intentionally **not** yet submission-ready because
`PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md` now records the closure state as
`decision-study-ready`, but the final Task 5 manuscript closure has not yet
rewritten the publication surfaces into their bounded decision-study form. The
current text reflects the stronger implementation-backed evidence boundary now
present in the repo: the bounded counted correctness package spans all four
`phase31_microcase_*` rows plus `phase2_xxz_hea_q4_continuity` and
`phase2_xxz_hea_q6_continuity`, the required five-row external slice is already
present on the current Stage-A evidence path, and the full structured
performance matrix plus its matrix-wide `break_even_table` /
`justification_map` are now emitted. The remaining work is publication closure,
not evidence closure.

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
implementation-backed validation now spans both layers. On the strict path, the
bounded counted microcase surface now agrees with the sequential oracle below
`1e-10`, while invariant checks confirm trace preservation and positivity under
the frozen numerical policy. On the hybrid path, both counted continuity
anchors also match the sequential oracle within the same threshold, and the
required bounded Qiskit Aer slice is already present for the four strict
microcases plus `phase2_xxz_hea_q4_continuity`. One frozen 8-qubit structured
pilot row records route coverage plus sequential, Phase 3 fused, and hybrid
timings. That pilot currently remains slower than the existing Phase 3 fused
baseline, so the present result is a bounded exactness result with initial
decision-study evidence rather than a closed acceleration claim. The remaining
claim gate is the full structured 8- and 10-qubit matrix and its decision
artifact, followed by the formal pre-publication evidence review.

## Keywords (Draft)

quantum simulation, density matrix, quantum channels, open quantum systems,
circuit partitioning, gate fusion, Kraus representation, exact classical
emulation

## Traceability

- `CLOSURE_PLAN_PHASE_3_1.md`
- `SHORT_PAPER_PHASE_3_1.md`
- `PAPER_PHASE_3_1.md`
- `DETAILED_PLANNING_PHASE_3_1.md`
