# Bounded Exact Channel-Native Fusion for Local Noisy Motifs in Partitioned Density-Matrix Simulation

## Draft Status

Implementation-backed technical short-paper surface revised to match the actual
Phase 3.1 state. The current evidence supports a bounded exactness result and a
reusable methods contribution; it does **not** yet support the originally
planned broader performance claim.

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
whole-workload path is the intended evaluation vehicle for continuity and
structured benchmark cases, where eligible partitions execute channel-natively
and Phase-3-supported but Phase-3.1-ineligible partitions remain on the shipped
exact baseline with route attribution. Current implementation-backed validation
covers the strict path: `<= 1e-10` Frobenius-norm agreement with sequential
exact execution is shown on a 1-qubit motif, one counted 2-qubit motif, and a
4-qubit fully eligible local-support smoke case, together with explicit hard
failures on out-of-scope motifs. The present result therefore establishes that
bounded exact channel-native fusion is feasible and scientifically auditable,
but it is not yet a closed acceleration claim: publication-backed coverage of
the remaining counted rows, Qiskit Aer cross-validation, and hybrid 8- and
10-qubit benchmarking are still pending.

## Publication Surface Role

Concise methods short paper for the Phase 3.1 publication surface. It follows a
short-paper structure centered on problem, method, current results, and
limitations rather than on a roadmap.

## Current Claim Boundary

**Supported by current implementation-backed evidence**

- Exact channel-native fusion for bounded 1- and 2-qubit same-support mixed
  motifs containing at least one noise operation, carried by the **strict**
  motif-proof runtime interpretation.
- Kraus-bundle composition as the primary counted representation, with
  representation-level completeness and positivity checks.
- Local-support application of the fused CPTP object inside a larger global
  density matrix, including a larger but still fully eligible 4-qubit smoke
  workload.
- Explicit no-silent-fallback behavior for out-of-scope motifs.

**Not yet supported by current evidence**

- Publication-backed coverage of the remaining counted correctness rows:
  `phase31_microcase_2q_multi_noise_entangler_chain`,
  `phase31_microcase_2q_dense_same_support_motif`,
  and the **hybrid** continuity anchors `phase2_xxz_hea_q4_continuity` and
  `phase2_xxz_hea_q6_continuity`.
- External Qiskit Aer validation for the frozen strict-plus-hybrid Phase 3.1
  slice.
- The structured 8- and 10-qubit performance study, which is contractually
  assigned to the explicit **hybrid** whole-workload interpretation and its
  required `break_even_table` / `justification_map`.
- General support beyond 2 qubits, correlated noise, or arbitrary unbounded
  CPTP fusion.

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
motif-dense workloads. The hybrid path is the intended whole-workload path for
continuity and structured benchmark cases, where eligible partitions execute
channel-natively and Phase-3-supported but Phase-3.1-ineligible partitions stay
on the shipped exact baseline with route attribution.

## 3. Current Results

The current implemented result is deliberately bounded but already scientific.
Current implementation-backed evidence is carried by the **strict** path.

- A 1-qubit mixed motif executes through the strict channel-native path and
  matches the sequential exact reference within the frozen `1e-10` density
  threshold.
- One counted 2-qubit motif, built around `CNOT` plus local noise on the same
  support, also matches the sequential exact reference within the same
  threshold.
- A 4-qubit spectator-support smoke case shows that a bounded 2-qubit fused
  noisy block can be embedded into a larger global density state without losing
  correctness on the full matrix, provided the whole workload remains fully
  eligible for the strict path.
- Ordered composition is claim-bearing: reversing the composed sequence changes
  the result, confirming that ordered noisy semantics must remain explicit.
- Boundary behavior is explicit: pure unitary motifs, support larger than two
  qubits, and unsupported gate or noise families fail deterministically instead
  of being silently downgraded.

By contrast, the broader counted continuity and structured whole-workload story
now belongs to the explicit **hybrid** interpretation. That interpretation is
part of the Phase 3.1 contract, but it is not yet publication-backed evidence.

Together, these results support a narrow but concrete statement: bounded exact
channel-native fusion is feasible for local noisy motifs and can be validated in
a way that is auditable and reproducible.

## 4. Current Answer to the Guiding Question

The guiding question for this publication surface is:

> When does the existing exact partitioned baseline stop being enough, and when
> would more invasive channel-native fusion become justified?

The current answer is now sharper than it was at planning time.

The existing unitary-island baseline stops being enough when repeated local
noise insertions fragment a same-support motif that is still small enough to be
represented exactly as one CPTP object. Channel-native fusion is therefore
**mathematically justified** on bounded 1- and 2-qubit noisy motifs today.

However, the broader whole-workload answer now requires two layers. The strict
path establishes the fused object itself. The explicit hybrid path is what will
decide whether that object is performance-justified on continuity and structured
benchmark workloads. That stronger claim still requires publication-backed
coverage of the remaining counted rows, external cross-validation, and the
structured 8- and 10-qubit hybrid benchmark matrix.

## 5. Limitations and Remaining Claim Gate

This short paper should stay honest about what is still missing.

- The current result is a bounded exactness study, not a closed acceleration
  paper.
- The full counted correctness surface has not yet been exercised through the
  publication-backed strict-plus-hybrid evidence package.
- Qiskit Aer has not yet been wired into the current strict-plus-hybrid Phase
  3.1 evidence slice.
- The planned motif-dense 8- and 10-qubit performance families exist as
  deterministic workload inventories, but the claim-closing whole-workload
  interpretation for them is the explicit hybrid path and that evidence bundle
  is not yet delivered.

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
- `task-5/TASK_5_MINI_SPEC.md`
