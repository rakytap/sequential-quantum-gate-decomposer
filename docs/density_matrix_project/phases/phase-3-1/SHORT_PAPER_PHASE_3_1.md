# Bounded Exact Channel-Native Fusion for Noise-Dense Motifs in Partitioned Density-Matrix Simulation

## Draft Status

Planning-phase technical short-paper surface aligned with the current Phase 3.1
contract. It now targets a **bounded positive-methods paper first** on the
frozen v1 slice; fallback to a diagnosis-grounded negative result remains
allowed if the counted slice does not justify the stronger claim. No API-level
claims until `API_REFERENCE` exists post-implementation.

## Abstract

Phase 3 delivered an exact partitioned noisy density runtime with real
unitary-island fusion, but it intentionally stopped at noise boundaries.
Phase 3.1 targets the next bounded methods step: **exact channel-native /
superoperator-native fusion of contiguous 1- and 2-qubit mixed gate+noise
motifs** built from `U3`, `CNOT`, and local single-qubit noise channels on the
same support. The intended contribution is not merely a new representation, but
an exact fused execution path for small CPTP blocks that preserves ordered
`NoisyCircuit` semantics and delivers a measurable benefit beyond the shipped
Phase 3 fused baseline on a frozen workload slice. The counted claim surface is
therefore a positive-methods result first: if at least one representative
motif-dense 8- or 10-qubit case shows `>= 1.2x` median wall-clock speedup or
`>= 15%` peak-memory reduction relative to the Phase 3 fused baseline, with no
correctness loss, the paper closes as a bounded methods contribution. If not,
the same evidence closes honestly as a diagnosis-grounded negative result.

## Publication Surface Role

Technical methods / systems short paper for Phase 3.1 (follow-on to Paper 2;
aligns with `PUBLICATIONS.md` Side Paper A, but now framed as a bounded
positive-methods contribution first).

## Claim Boundary (Planning Draft)

**In scope for the counted v1 Phase 3.1 claim:**

- Exact channel-native / superoperator-native fusion for **contiguous 1- and
  2-qubit mixed motifs** on a fixed support, with **at least one noise
  operation** in each counted Phase 3.1 fused block.
- Primary counted-claim representation:
  - `kraus_bundle`, with Liouville / superoperator matrices allowed only as
    internal apply/cache views after equivalence is shown.
- Primitive surface frozen to `U3`, `CNOT`, local depolarizing, local amplitude
  damping, and local phase damping / dephasing.
- Multiple successive gates and multiple same-support local noise insertions
  inside one fused block.
- Exactness versus sequential `NoisyCircuit` under the frozen numeric policy:
  `<= 1e-10` Frobenius agreement, validity checks, and representation-level
  CPTP invariants.
- Comparative benchmarking against:
  - sequential `NoisyCircuit`,
  - Phase 3 partitioned+fused,
  - Phase 3.1 channel-native.

**Explicit non-claims until evidenced:**

- Universal speedup across all noisy workloads.
- Support beyond 2 qubits, correlated noise, spectator-qubit effects, or
  arbitrary unbounded CPTP fusion.
- Phase 4 gradients, optimizers, or broader circuit sources.
- Task 6 host-acceleration claims as part of the main paper story; mandatory
  bundles remain scalar-only.

**Evidence-closure rule (intent):**

Only mandatory, complete, supported correctness evidence plus counted
performance evidence closes the main positive-methods claim. The benchmark
package must also emit a `break_even_table` / `justification_map` classifying
where Phase 3 remains sufficient and where Phase 3.1 is actually justified.

## Traceability Table (Planning Draft)

| Claim / paper question | Planned evidence anchor | Contract source |
|---|---|---|
| What is the new fused scientific object? | `phase31_bounded_mixed_motif_v1`; Task 1 representation contract; invariant-aware microcases | `P31-ADR-004`, `P31-ADR-007`, `task-1/TASK_1_MINI_SPEC.md` |
| Is the new path still exact? | counted correctness slice + `channel_invariants` + Aer external slice | `P31-ADR-008`, `P31-ADR-009`, `P31-ADR-011`, `task-3/TASK_3_MINI_SPEC.md` |
| Does it beat the shipped Phase 3 fused baseline anywhere meaningful? | counted performance slice + `break_even_table` / `justification_map` | `P31-ADR-010`, `task-4/TASK_4_MINI_SPEC.md` |
| How is the new path distinguished from the old one? | `phase31_channel_native` runtime path and additive API surface | `P31-ADR-012`, `task-2/TASK_2_MINI_SPEC.md` |
| How does the evidence flow evolve? | scalar-only counted v1 bundles; Phase 3.1 may later become the default evidence surface while Phase 3 remains explicit legacy scripts/functions | `P31-ADR-014`, `P31-ADR-015`, `task-6/TASK_6_MINI_SPEC.md` |
| What is not part of the main claim? | non-claims list, scalar-only v1 evidence builds, Task 6 later-branch status | this file §Claim Boundary, `task-6/TASK_6_MINI_SPEC.md` |

## 1. Introduction and Motivation

Exact density-matrix simulation remains the strongest classical anchor for noisy
variational research in this project. Phase 3 connected partitioning to that
anchor while deferring fusion through noise itself. Phase 3.1 asks whether the
next exact fusion layer—operating natively on small noisy CPTP motifs rather
than only on unitary islands—changes the performance story while preserving the
ordered semantics that justify the anchor.

## 2. Relationship to Phase 3

Phase 3 closed with a bounded methods result and diagnosis-grounded performance
closure. Phase 3.1 **adds** a new counted fused object: bounded mixed noisy
motifs on 1- and 2-qubit support. It does not restate or weaken Phase 3’s
minimum deliverable. Comparisons therefore name three baselines where relevant:
sequential density, Phase 3 partitioned with unitary-island fusion, and Phase
3.1 channel-native fusion.

## 3. Technical Scope (Planning)

- Representation and composition of fused noisy blocks, with `kraus_bundle` as
  the primary counted-claim form.
- Eligibility rules for the v1 slice:
  - contiguous 1- and 2-qubit same-support mixed motifs,
  - at least one noise operation per counted fused block,
  - multiple same-support local noise insertions allowed,
  - pure unitary islands continue to use the Phase 3 fused path.
- Explicit unsupported behavior beyond the bounded slice.

## 4. Evaluation Plan (Planning)

- Correctness:
  - counted microcases:
    - `phase31_microcase_1q_u3_local_noise_chain`,
    - `phase31_microcase_2q_cnot_local_noise_pair`,
    - `phase31_microcase_2q_multi_noise_entangler_chain`,
    - `phase31_microcase_2q_dense_same_support_motif`,
  - counted continuity anchors:
    - `phase2_xxz_hea_q4_continuity`,
    - `phase2_xxz_hea_q6_continuity`,
  - sequential exactness `<= 1e-10`, validity checks, and representation-level
    CPTP invariants.
- Performance:
  - primary families:
    - `phase31_pair_repeat`,
    - `phase31_alternating_ladder`,
  - control family:
    - `layered_nearest_neighbor`,
  - counted positive-method threshold:
    - `>= 1.2x` median wall-clock speedup or `>= 15%` peak-memory reduction
      versus the Phase 3 fused baseline on at least one representative primary
      family case,
  - required decision artifact:
    - `break_even_table` / `justification_map`.

## 5. Limitations

Support matrix is intentionally narrow at first. Approximate simulators and
trajectory methods remain out of scope for the core Phase 3.1 exact claim. If
the frozen positive-method threshold is not met, this short paper falls back to
an honest diagnosis-grounded negative result rather than quietly expanding the
claim surface.

## Traceability

- `DETAILED_PLANNING_PHASE_3_1.md`
- `ADRs_PHASE_3_1.md`
- `docs/density_matrix_project/planning/PLANNING.md` §5.1
