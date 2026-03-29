# Beyond Unitary Islands: Why Channel-Native Fusion Matters for Exact Noisy Simulation

## Draft Status

Planning-phase narrative companion to `SHORT_PAPER_PHASE_3_1.md`. It now follows
the same bounded **positive-methods-first** contract: Phase 3.1 is the attempt
to show that exact noisy fusion can do more than unitary islands on a narrow,
scientifically chosen slice. Wording will still tighten against delivered
evidence after implementation.

## Abstract

Researchers often treat noise as something that happens *between* unitary
chunks of a quantum circuit. Phase 3 showed that SQUANDER can do better than
that at the planning level: noise is a first-class object in the partitioned
runtime contract, even though the shipped fused primitive is still a unitary
island. Phase 3.1 is the next, more ambitious step. It targets **exact
channel-native fusion of bounded 1- and 2-qubit mixed gate+noise motifs** so
that dense same-support noisy regions can be executed as single CPTP blocks
rather than repeatedly broken apart at noise boundaries.

That matters scientifically because the project is no longer asking only
whether noise-aware partitioning is possible. It is asking whether a stronger,
still exact fused object can beat the already delivered Phase 3 fused baseline
on the kind of noise-dense local motifs that are most likely to matter in
training-relevant workloads. The expected paper story is therefore a bounded
positive methods result first, with an honest fallback to a negative or mixed
diagnosis if the benchmark slice does not justify the added complexity.

## Publication Surface Role

Narrative positioning for a PhD-conference or general technical audience within
the Phase 3.1 package. Answers “why this phase matters in the research arc”
alongside the technical short paper.

## Research Arc Position

- Phase 2: exact noisy workflows are real in SQUANDER.
- Phase 3: partitioning and fusion respect noisy semantics; limited fusion ships.
- Phase 3.1: asks whether **bounded exact fusion of mixed noisy motifs** is the
  right next lever for performance beyond unitary islands, with benchmark
  discipline against the shipped Phase 3 fused baseline.
- Phase 4: returns to training-facing features once backend questions are
  settled.

## What Success Looks Like (Narratively)

Success is first a **bounded positive answer**: at least one representative
motif-dense case shows a real benefit beyond the Phase 3 fused baseline while
preserving exactness. More broadly, success is a reviewable explanation of
where that richer fused object is justified and where it is not, backed by
exact checks and comparative benchmarks against both sequential simulation and
the Phase 3 fused baseline.

## Non-Goals (Audience Clarity)

Phase 3.1 is not the thesis trainability paper, not the optimizer paper, and
not the approximate scaling paper. It is also not a broad gate-coverage or
correlated-noise paper. Those remain later milestones.

## Traceability

- `DETAILED_PLANNING_PHASE_3_1.md`
- `SHORT_PAPER_PHASE_3_1.md`
- `docs/density_matrix_project/planning/PUBLICATIONS.md` (Side Paper A / Phase 3.1)
