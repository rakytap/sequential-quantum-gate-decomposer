# Bounded Exact Channel-Native Fusion for Local Noisy Motifs in Partitioned Density-Matrix Simulation

## Draft Status

Pre-closure boundary-sync draft aligned to
`CLOSURE_PLAN_PHASE_3_1.md`. The required
`PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md` now exists and currently records
the closure state `not-ready-yet`, so this surface is intentionally **not** yet
a venue-ready short paper. The current draft reflects the stronger implementation-backed evidence
boundary now present in the repo: the bounded counted correctness package spans
all four `phase31_microcase_*` rows plus `phase2_xxz_hea_q4_continuity` and
`phase2_xxz_hea_q6_continuity`, the required five-row external slice is already
present on the current Stage-A evidence path, and one structured hybrid pilot
row provides initial whole-workload decision-study evidence. The remaining
claim-closing gate is the full structured performance matrix plus the
`break_even_table` / `justification_map`, followed by the pre-publication
evidence review.

## Abstract

Exact density-matrix simulation is the most trustworthy classical baseline for
studying variational quantum circuits under realistic local noise, but existing
partitioning and fusion techniques largely stop at unitary subcircuits. This
work studies a bounded exact alternative: channel-native fusion of small mixed
gate-noise motifs inside partitioned density-matrix execution. The current
method now has two execution interpretations. A **strict** motif-proof path uses
Kraus bundles as the primary exact representation for 1- and 2-qubit
same-support motifs built from `U3`, `CNOT`, and local depolarizing,
amplitude-damping, and phase-damping channels. An explicit **hybrid**
whole-workload path is the current evaluation vehicle for continuity and
structured benchmark cases, where eligible partitions execute channel-natively
and Phase-3-supported but Phase-3.1-ineligible partitions remain on the shipped
exact baseline with route attribution. Current implementation-backed validation
now spans both layers. On the strict path, `<= 1e-10` Frobenius-norm agreement
with sequential exact execution is now present across the bounded counted
microcase surface, together with a 4-qubit fully eligible local-support smoke
case and explicit hard failures on out-of-scope motifs. On the hybrid path,
both counted continuity anchors (`phase2_xxz_hea_q4_continuity` and
`phase2_xxz_hea_q6_continuity`) match the sequential oracle within the same
threshold, and the bounded external-reference slice against Qiskit Aer is
already present for the four strict microcases plus `q4` continuity. One frozen
8-qubit structured pilot row records route coverage plus sequential, Phase 3
fused, and hybrid timings. In the current pilot, hybrid execution remains
slower than the existing Phase 3 fused baseline. The present result therefore
establishes a bounded exactness result with initial decision-study evidence,
but not yet a closed positive-methods acceleration claim because the full
structured 8/10-qubit matrix and its decision artifact remain outstanding.

## Publication Surface Role

Concise methods short paper for the Phase 3.1 publication surface. It follows a
short-paper structure centered on problem, method, current results, and
limitations rather than on a roadmap. Until the pre-publication evidence review
exists, this document functions as a boundary-sync technical surface rather
than as a submission-ready manuscript.

## Current Claim Boundary

**Implemented and validated on the current evidence boundary**

- Exact channel-native fusion for the four counted `phase31_microcase_*` rows on
  the bounded 1- and 2-qubit same-support mixed-motif surface, carried by the
  **strict** motif-proof runtime interpretation.
- Kraus-bundle composition as the primary counted representation, with
  representation-level completeness and positivity checks.
- Local-support application of the fused CPTP object inside a larger global
  density matrix, including a larger but still fully eligible 4-qubit smoke
  workload.
- Explicit **hybrid** whole-workload interpretation with partition-level route
  attribution and no silent fallback for unsupported-by-both partitions.
- Counted hybrid continuity exactness for
  `phase2_xxz_hea_q4_continuity` and `phase2_xxz_hea_q6_continuity` under the
  same `1e-10` density threshold.
- The required bounded external Qiskit Aer slice on the four strict
  `phase31_microcase_*` rows plus hybrid `phase2_xxz_hea_q4_continuity`.
- One frozen structured hybrid pilot row,
  `phase31_pair_repeat_q8_periodic_seed20260318`, with route coverage and
  baseline-trio timing. The current row supports only the
  negative-to-inconclusive conclusion that hybrid overhead still dominates the
  shipped Phase 3 fused baseline.
- Explicit no-silent-fallback behavior for out-of-scope motifs.

**Implemented but not yet claim-closing**

- The bounded counted correctness and external slices are already stronger than
  earlier drafts, but they have not yet been turned into a formal closure state
  through `PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md`.
- The frozen hybrid pilot row proves route-aware whole-workload execution and
  provides an initial decision-study signal, but one row cannot close the
  whole-workload justification question.

**Still missing for Phase 3.1 closure**

- The full structured 8- and 10-qubit performance matrix, including
  control-family closure and the required `break_even_table` /
  `justification_map`.
- The pre-publication evidence review and explicit closure-state record
  (`positive-methods-ready`, `decision-study-ready`, or `not-ready-yet`).
- General support beyond 2 qubits, correlated noise, or arbitrary unbounded
  CPTP fusion.

### Claim-to-evidence traceability

Compact mapping from publication claims to runtime interpretation, frozen case
identifiers, and the primary regression surface. Thresholds match the frozen
partitioned-runtime density policy (`1e-10` unless noted).

| Claim (concise) | Path | Case ID / workload | Evidence surface | Threshold / notes |
|------------------|------|--------------------|------------------|-------------------|
| 1q mixed motif exact vs sequential oracle | strict | `phase31_microcase_1q_u3_local_noise_chain` | `tests/partitioning/test_partitioned_channel_native_phase31_slice.py` | Frobenius, max-abs ≤ `1e-10`; trace, validity |
| Counted 2q mixed motif exact vs sequential | strict | `phase31_microcase_2q_cnot_local_noise_pair` | `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py` | Same; fused-region audit |
| 4q eligible smoke (two disjoint fused blocks) | strict | `phase31_local_support_q4_spectator_embedding_smoke` | same file | Full-matrix agreement ≤ `1e-10` |
| Kraus completeness / Choi floor on counted 2q fuse | strict | (same partition as counted 2q case) | `test_phase31_s06_e01_counted_2q_fused_kraus_bundle_satisfies_invariants` in second-slice module | Internal invariants (`1e-10` completeness; Choi floor) |
| Ordered composition matters (2q microcase) | strict | `phase31_microcase_2q_cnot_local_noise_pair` | `test_phase31_channel_native_ordered_fusion_matches_sequential_on_2q_cnot_microcase` | Reversal changes state ≫ `1e-10` |
| Bounded counted correctness package present | strict + hybrid | four `phase31_microcase_*` rows plus `phase2_xxz_hea_q4_continuity` and `phase2_xxz_hea_q6_continuity` | `tests/partitioning/evidence/test_phase31_correctness_evidence.py`; `phase31_correctness_package_bundle.json` | 6 counted supported rows in the current Stage-A package |
| Bounded external slice present | strict + hybrid | four `phase31_microcase_*` rows plus `phase2_xxz_hea_q4_continuity` | same test; `phase31_external_correctness_bundle.json` | 5 required Aer rows on the frozen external slice |
| Hybrid continuity whole-workload exactness (`q4`) | hybrid | `phase2_xxz_hea_q4_continuity` | `tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py` | Density metrics ≤ `1e-10`; stable aggregated route summary |
| Hybrid continuity whole-workload exactness (`q6`) | hybrid | `phase2_xxz_hea_q6_continuity` | same module | Density metrics ≤ `1e-10`; stable aggregated route summary |
| Hybrid unsupported gate fails loudly | hybrid | `hybrid_negative_rx_relaxed_surface` | same module | `NoisyRuntimeValidationError`; no silent route |
| Hybrid structured q8 smoke (exactness) | hybrid | `phase31_pair_repeat_q8_dense_seed20260318` | `test_phase31_hybrid_structured_pair_repeat_q8_dense_smoke` in `test_partitioned_channel_native_phase31_hybrid_slice.py` | Frobenius ≤ `1e-10`; route vocabulary |
| Frozen hybrid pilot: baseline trio + routes + diagnosis | hybrid | `phase31_pair_repeat_q8_periodic_seed20260318` | `tests/partitioning/evidence/test_phase31_hybrid_pilot_validation.py`; `benchmarks/density_matrix/performance_evidence/phase31_hybrid_pilot_validation.py` | `median_3` samples; `decision_class` / `diagnosis_tag`; not a matrix closure claim |

## 1. Problem and Gap

Partitioning and gate-fusion methods are well developed for unitary or
state-vector simulation, as illustrated by TDAG, GTQCP, QGo, and more recent
gate-fusion work such as QMin. Open-system simulation, by contrast, is usually
discussed through channel representations such as Kraus, Choi, or Liouville
forms, as in standard quantum-information and open-system references.

The gap is that these two traditions do not automatically compose. A noisy
region of a circuit is not just a unitary block with annotations attached to
its boundary; the order of gates and channels is itself part of the semantics.
The scientific question addressed here is therefore narrow but important:

> Can small noisy motifs be fused exactly, without breaking ordered
> density-matrix semantics, so that noise-dense local structure becomes a real
> optimization object rather than a mandatory fusion barrier?

## 2. Bounded Method

The current method adopts five design rules.

First, the primary mathematical object is a **Kraus bundle** on bounded support.
This keeps the counted claim in a standard exact channel representation rather
than in a hidden internal optimization view.

Second, the support surface is intentionally narrow: contiguous 1- and 2-qubit
same-support motifs built from `U3`, `CNOT`, and local depolarizing,
amplitude-damping, or phase-damping channels. Each fused motif must contain at
least one noise operation.

Third, exactness is protected by construction. Composition follows descriptor
order, not an implementation shortcut order; fused objects are checked through
trace-preservation and positivity-style invariants before they are treated as
valid counted objects.

Fourth, interpretability matters as much as exactness. The current path hard
fails on unsupported motifs instead of silently reverting to a weaker execution
mode, which keeps future benchmark comparisons scientifically meaningful.

Fifth, scientific interpretation now distinguishes **strict** and **hybrid**
execution. The strict path proves the exact fused object on fully eligible
motif-dense workloads. The hybrid path is the current whole-workload path for
continuity and structured benchmark cases, where eligible partitions execute
channel-natively and Phase-3-supported but Phase-3.1-ineligible partitions stay
on the shipped exact baseline with route attribution.

## 3. Current Results

The current implemented result is deliberately bounded but already scientific.
Current implementation-backed evidence now spans both the **strict** and the
initial **hybrid** layer.

- All four counted `phase31_microcase_*` motifs now execute through the strict
  channel-native path and match the sequential exact reference within the
  frozen `1e-10` density threshold.
- The strict microcase package also carries the current representation-level
  invariant checks and the bounded external-reference slice on those four
  microcases.
- A 4-qubit spectator-support smoke case shows that a bounded 2-qubit fused
  noisy block can be embedded into a larger global density state without losing
  correctness on the full matrix, provided the whole workload remains fully
  eligible for the strict path.
- The counted continuity anchors `phase2_xxz_hea_q4_continuity` and
  `phase2_xxz_hea_q6_continuity` execute through the **hybrid** path, match the
  sequential exact reference within the same threshold, and record explicit
  mixed route attribution between channel-native and shipped Phase 3
  execution.
- The required external slice also includes hybrid
  `phase2_xxz_hea_q4_continuity`, keeping the bounded package tied to an
  external reference without turning into a broad simulator bake-off.
- One frozen 8-qubit structured pilot row,
  `phase31_pair_repeat_q8_periodic_seed20260318`, records a complete baseline
  trio plus route coverage. In the current pilot, hybrid execution remains
  slower than the existing Phase 3 fused baseline, so the row supports a
  decision-study outcome rather than a positive performance claim.
- Ordered composition is claim-bearing: reversing the composed sequence changes
  the result, confirming that ordered noisy semantics must remain explicit.
- Boundary behavior is explicit: pure unitary motifs, support larger than two
  qubits, and unsupported gate or noise families fail deterministically instead
  of being silently downgraded.

Together, these results support a sharper but still bounded statement:
channel-native fusion is feasible for local noisy motifs, semantically
executable inside mixed whole workloads, and auditable in a reproducible way.
What they do **not** yet support is a general performance justification.

## 4. Current Answer to the Guiding Question

The guiding question for this publication surface is:

> When does the existing exact partitioned baseline stop being enough, and when
> would more invasive channel-native fusion become justified?

The current answer is now sharper than it was at planning time.

The existing unitary-island baseline stops being enough when repeated local
noise insertions fragment a same-support motif that is still small enough to be
represented exactly as one CPTP object. Channel-native fusion is therefore
**mathematically justified** on bounded 1- and 2-qubit noisy motifs today, and
the current bounded correctness package now shows that such motifs can be
exercised both as strict counted microcases and inside larger exact workloads
through the hybrid continuity anchors without losing correctness or route
auditability.

However, the broader whole-workload decision is currently negative-to-
inconclusive rather than positive. The first frozen 8-qubit structured pilot
row yields explicit channel-native coverage and a complete baseline trio, but
it is currently slower than the existing Phase 3 fused baseline. On the current
evidence, channel-native fusion is therefore mathematically justified and
semantically executable, but not yet performance-justified as a broader
workload method.

## 5. Limitations and Remaining Claim Gate

This short paper should stay honest about what is still missing.

- The current result is a bounded exactness result with initial decision-study
  evidence, not a closed acceleration paper.
- The bounded counted correctness package and required five-row external slice
  are already present in current evidence, but they still need to be turned
  into a formal closure-state decision through
  `PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md`.
- The current hybrid evidence includes only one frozen structured pilot row.
  That row is informative precisely because it is not positive: it shows that
  mathematical feasibility does not automatically translate into workload-level
  speedup.
- The broader 8- and 10-qubit structured matrix, control-family closure, final
  `break_even_table` / `justification_map`, and the resulting closure-state
  decision remain undelivered.

If those missing layers later show a real advantage relative to the existing
partitioned exact baseline, this short paper can close as a bounded positive
methods result. If they do not, the scientifically honest closure is a
decision-study paper explaining where the richer fused object is and is not
justified.

## Selected References

- Joseph Clark, Travis S. Humble, and Himanshu Thapliyal, *TDAG: Tree-based
  Directed Acyclic Graph Partitioning for Quantum Circuits*, ACM GLSVLSI 2023.
- Joseph Clark, Travis S. Humble, and Himanshu Thapliyal, *GTQCP: Greedy
  Topology-Aware Quantum Circuit Partitioning*, `arXiv:2410.02901`.
- Xin-Chuan Wu, Marc Grau Davis, Frederic T. Chong, and Costin Iancu, *QGo:
  Scalable Quantum Circuit Optimization Using Automated Synthesis*,
  `arXiv:2012.09835`.
- Longshan Xu, Edwin Hsing-Mean Sha, Yuhong Song, and Qingfeng Zhu, *QMin:
  Quantum Circuit Minimization via Gate Fusions for Efficient State Vector
  Simulation*, `Quantum Information Processing 25, 6 (2026)`.
- Michael A. Nielsen and Isaac L. Chuang, *Quantum Computation and Quantum
  Information*, Cambridge University Press (2010).
- Christopher J. Wood, Jacob D. Biamonte, and David G. Cory, *Tensor networks
  and graphical calculus for open quantum systems*, `Quantum Information and
  Computation 15, 759-811 (2015)`.
- Ang Li, Omer Subasi, Xiu Yang, and Sriram Krishnamoorthy, *Density Matrix
  Quantum Circuit Simulation via the BSP Machine on Modern GPU Clusters*, SC20.
- Tyson Jones, Anna Brown, Ian Bush, and Simon C. Benjamin, *QuEST and High
  Performance Simulation of Quantum Computers*, `Scientific Reports 9, 10736
  (2019)`.

## Traceability

- `DETAILED_PLANNING_PHASE_3_1.md`
- `ADRs_PHASE_3_1.md`
- `CLOSURE_PLAN_PHASE_3_1.md`
- `task-5/TASK_5_MINI_SPEC.md`
