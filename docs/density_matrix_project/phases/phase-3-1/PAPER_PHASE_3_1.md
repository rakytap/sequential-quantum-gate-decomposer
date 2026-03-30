# Full Paper Draft: Bounded Exact Channel-Native Fusion for Local Noisy Motifs

## Draft Status

Implementation-backed full-paper surface revised to reflect the current Phase
3.1 state. This draft now separates what is already demonstrated from what is
still required before a broader performance claim can be defended.

## Title Candidates

- `Bounded Exact Channel-Native Fusion for Local Noisy Motifs in Partitioned Density-Matrix Simulation`
- `Beyond Unitary Fusion: Exact CPTP Motif Composition for Noisy Quantum Simulation`
- `Exact Local Noisy-Motif Fusion in a Partitioned Density-Matrix Runtime`

## Abstract

Exact density-matrix simulation is the most reliable classical baseline for
studying variational quantum circuits under realistic local noise, yet most
partitioning and fusion techniques are still centered on unitary or
state-vector-style subcircuits. We study a bounded exact alternative inside the
SQUANDER mixed-state execution stack: channel-native fusion of small same-
support gate-noise motifs. The current method treats 1- and 2-qubit motifs
built from `U3`, `CNOT`, and local depolarizing, amplitude-damping, or phase-
damping channels as exact CPTP objects represented primarily by Kraus bundles.
Composition follows ordered noisy-circuit semantics, invariant checks enforce
trace preservation and positivity, and the fused object can be applied on local
support within a larger density state. Current implementation-backed validation
shows `<= 1e-10` Frobenius-norm agreement with sequential exact execution on a
1-qubit motif, one counted 2-qubit motif, and a 4-qubit spectator-support smoke
case, together with deterministic rejection of out-of-scope motifs. These
results establish that bounded exact channel-native fusion is feasible and
scientifically auditable. They do not yet close a broader acceleration claim:
the remaining counted microcases, external Qiskit Aer validation, and
structured 8- and 10-qubit benchmarking are still pending. The current outcome
is therefore best interpreted as a methods-grounding result and a reusable
framework for future exact noisy acceleration studies.

## Publication Surface Role

This document is the full-paper draft surface for the Phase 3.1 publication
track. It follows a research-paper structure with explicit problem statement,
method, current results, limitations, and reproducibility posture.

## Current Claim Boundary

**Current supported claim**

- Small same-support noisy motifs can be fused exactly as CPTP objects inside a
  partitioned density-matrix execution flow.
- The fused object can be validated through sequential-reference agreement plus
  channel-invariant checks.
- Local-support embedding into a larger density state is feasible on the current
  bounded slice.
- Unsupported motifs remain explicit rather than silently falling back.

**Claim-bearing gate that remains open**

- The full counted correctness slice,
- external Qiskit Aer validation on the frozen external slice,
- and the structured 8- and 10-qubit performance matrix with decision-surface
  reporting.

## 1. Introduction

Exact noisy simulation matters because many scientifically relevant questions in
quantum computing and quantum machine learning depend on how realistic local
noise reshapes circuit behavior. This is especially true for variational
workflows, where the interaction among ansatz structure, optimization, and noise
can change convergence behavior, effective trainability, and the interpretation
of benchmark results.

Density-matrix simulation provides the cleanest exact reference for these
questions, but it is also expensive. That expense has made reuse-oriented
acceleration strategies such as partitioning and fusion attractive in the
unitary setting. The challenge is that noisy evolution is governed by channels,
not just unitary subcircuits, and therefore the usual unitary-island viewpoint
does not fully capture the structure of a noisy mixed-state computation.

This paper studies a bounded exact alternative inside SQUANDER: can small noisy
motifs themselves become the fused object? The current answer is intentionally
narrow. We focus on 1- and 2-qubit same-support motifs containing both gates and
local noise, and we ask whether they can be fused exactly, validated
rigorously, and embedded into a larger partitioned density-matrix execution
flow.

The current contribution is therefore a methods-grounding result rather than a
finished acceleration paper. It establishes the exact fused object, the current
validation discipline, and the scientific boundary of the claim.

## 2. Related Work and Scientific Gap

### 2.1 Partitioning and Fusion in the Unitary Setting

Graph-based partitioning and gate-fusion methods such as TDAG, GTQCP, QGo, and
QMin show that substantial structure can be exploited when quantum circuits are
treated as dependency objects rather than as flat gate lists. These methods are
highly relevant for identifying reusable motifs and for motivating bounded
fusion.

However, most of this literature is developed around unitary or state-vector
simulation, where the fused object is itself unitary. That assumption becomes
insufficient when local noise channels are interleaved with gates in a way that
is semantically important.

### 2.2 Open-System and Density-Matrix Background

Quantum operations theory provides the language needed for exact noisy
simulation. Nielsen and Chuang formalize Kraus maps and CPTP structure, while
Wood, Biamonte, and Cory make explicit the relations among Kraus, Choi,
Liouville, and related channel representations. High-performance density-matrix
simulation work, including Li et al., QuEST, and Qulacs, shows that exact noisy
simulation is already a nontrivial systems problem.

These references provide the mathematical and computational background, but they
do not by themselves answer the present question: how to fuse a bounded noisy
motif inside an exact partitioned runtime while keeping ordered circuit
semantics explicit.

### 2.3 Scientific Gap

The gap addressed here lies between these two literatures. The unitary fusion
literature identifies reusable local structure, while the open-system literature
defines valid noisy evolution. The missing bridge is a scientifically auditable
bounded exact fused object for mixed gate-noise motifs.

## 3. Problem Formulation

The current study focuses on three research questions.

**RQ1.** Can a small mixed gate-noise motif be represented as one exact CPTP
object without changing the ordered semantics of the original noisy circuit?

**RQ2.** Can that fused object be applied on local support inside a larger
density state without breaking correctness on the full global matrix?

**RQ3.** What validation discipline is needed before any performance claim about
exact noisy fusion becomes scientifically credible?

The current bounded support surface is deliberately narrow:

- same-support 1- and 2-qubit motifs,
- `U3` and `CNOT` gates,
- local depolarizing, amplitude-damping, and phase-damping channels,
- and at least one noise operation per fused motif.

This scope is a design decision, not a claim of generality.

## 4. Method

### 4.1 Primary Exact Representation

The current counted representation is a **Kraus bundle**. This choice keeps the
claim tied to a standard exact channel formalism rather than to an opaque
execution cache. Other views such as Liouville or superoperator matrices may
still be useful internally later, but the current claim is anchored in Kraus
form.

### 4.2 Ordered Composition

The fused object is constructed in the exact operation order of the noisy motif.
This is essential because a noisy circuit is not invariant under arbitrary
reordering of gates and channels. In the current method, descriptor order is
part of the scientific object, not merely a software detail.

### 4.3 Physical-Invariant Checks

Before a fused object is counted as valid, it must satisfy representation-level
checks consistent with the current numerical policy:

- trace-preservation via Kraus completeness,
- positivity-style checks through the associated Choi object,
- and state-level agreement with the sequential exact reference.

This is the minimal discipline needed to distinguish an exact noisy method from
an implementation shortcut.

### 4.4 Local-Support Embedding and Failure Semantics

The current method also requires that a fused local object can be embedded into
a larger density state without weakening the claim to a reduced-subsystem check.
For this reason, the current validation uses full-matrix comparison on the
larger-workload smoke case.

Equally important, unsupported motifs fail explicitly. This no-silent-fallback
rule keeps future benchmark interpretation honest.

## 5. Current Implementation-Backed Evaluation

### 5.1 Current Implemented Slice

The current implementation-backed slice includes:

- one 1-qubit mixed motif,
- one counted 2-qubit mixed motif,
- and one non-counted 4-qubit spectator-support smoke case.

Deterministic workload inventories also already exist for the remaining counted
microcases and for the future 8- and 10-qubit structured performance families,
but those inventories are not yet wired into claim-closing Phase 3.1 evidence
bundles.

### 5.2 Metrics and Thresholds

The current bounded slice uses:

- Frobenius-norm density-matrix agreement with the sequential exact reference,
- maximum absolute matrix difference,
- trace deviation,
- density validity checks,
- and representation-level invariant residuals.

The current exactness threshold for the slice is `<= 1e-10` on the density
comparison metrics.

### 5.3 Current Results

The current results support four statements.

First, the 1-qubit mixed motif matches the sequential exact reference within the
frozen threshold.

Second, the counted 2-qubit motif, built around `CNOT` plus local noise on the
same support, also matches the sequential exact reference within the same
threshold and satisfies the current invariant checks.

Third, the 4-qubit spectator-support smoke case shows that a bounded 2-qubit
fused noisy object can be embedded into a larger density state while preserving
the correctness of the full global output.

Fourth, out-of-scope motifs fail deterministically. Pure unitary motifs, motifs
with support above two qubits, and motifs using unsupported operations remain
visible as unsupported behavior rather than being silently absorbed into a
different path.

### 5.4 What Is Still Missing

The current full-paper claim remains incomplete because three layers are still
missing:

- the remaining counted correctness rows,
- the external Qiskit Aer slice,
- and the structured 8- and 10-qubit benchmark matrix together with the
  required decision-surface reporting.

As a result, this paper should currently report a bounded exactness result, not
yet a broader performance result.

## 6. Discussion

The scientific importance of the current result is not that it proves noisy
fusion is broadly beneficial. It does not. The important result is that it
changes the status of a noisy motif from "mandatory fusion barrier" to
"candidate exact fused object" on a narrow but nontrivial support surface.

This already answers part of the motivating scientific question. The existing
unitary-only baseline stops being enough when repeated local noise insertions
fragment a motif that is still small enough to admit exact bounded channel
composition. The current study shows that such motifs exist and can be handled
exactly.

What remains unknown is whether this mathematically justified object is also a
performance-justified object on the workload families that matter most. That is
the next claim-bearing gate, and the paper should say so directly.

## 7. Threats to Validity and Limitations

Several limitations must remain explicit.

### 7.1 Narrow Support Surface

The current result is deliberately restricted to bounded 1- and 2-qubit local
motifs with specific gate and noise families. It should not be interpreted as a
claim about correlated noise, larger supports, or arbitrary CPTP fusion.

### 7.2 Missing External and Performance Closure

The current paper does not yet have the full external Qiskit Aer slice or the
structured 8- and 10-qubit performance matrix. This is the main reason broader
acceleration language must remain withheld.

### 7.3 Dense-Regime Scale Limits

Even a successful bounded exact fusion method remains inside the scaling limits
of dense density-matrix simulation. The present result improves structure, not
the asymptotic memory law.

### 7.4 Representation Choice

Kraus form is the current primary exact representation, but it may not be the
only useful representation for later larger-scale studies. The present paper
should treat it as the current counted choice, not as a universal theorem about
best representation.

## 8. Reproducibility and Reporting

The current reproducibility posture follows four rules.

- Use deterministic microcase definitions and stable case identifiers.
- Report explicit numerical thresholds for exactness and invariant checks.
- Keep unsupported cases visible.
- Separate current slice evidence from future claim-bearing bundles.

At the current state of the work, reproducibility is strongest at the level of
deterministic workload definitions and exact-threshold regression tests. The
later full evidence package should add external-reference rows, structured
benchmark reporting, and explicit decision-surface artifacts.

## 9. Conclusion

This paper surface now supports a clear, bounded conclusion. Exact channel-
native fusion of local noisy motifs is feasible, auditable, and already
demonstrated on a narrow implementation-backed slice. That is scientifically
useful because it establishes a concrete bridge between unitary fusion ideas and
open-system channel semantics. At the same time, the work is not yet a closed
performance paper. The remaining counted correctness rows, external validation,
and structured benchmarking are still required before stronger acceleration
claims can be made.

## Selected References

1. Peter Rakyta and Zoltan Zimboras, *Approaching the theoretical limit in
   quantum gate decomposition*, `Quantum 6, 710 (2022)`.
2. Joseph Clark, Travis S. Humble, and Himanshu Thapliyal, *TDAG: Tree-based
   Directed Acyclic Graph Partitioning for Quantum Circuits*, ACM GLSVLSI 2023.
3. Joseph Clark, Travis S. Humble, and Himanshu Thapliyal, *GTQCP: Greedy
   Topology-Aware Quantum Circuit Partitioning*, `arXiv:2410.02901`.
4. Xin-Chuan Wu, Marc Grau Davis, Frederic T. Chong, and Costin Iancu, *QGo:
   Scalable Quantum Circuit Optimization Using Automated Synthesis*,
   `arXiv:2012.09835`.
5. Longshan Xu, Edwin Hsing-Mean Sha, Yuhong Song, and Qingfeng Zhu, *QMin:
   Quantum Circuit Minimization via Gate Fusions for Efficient State Vector
   Simulation*, `Quantum Information Processing 25, 6 (2026)`.
6. Michael A. Nielsen and Isaac L. Chuang, *Quantum Computation and Quantum
   Information*, Cambridge University Press (2010).
7. Christopher J. Wood, Jacob D. Biamonte, and David G. Cory, *Tensor networks
   and graphical calculus for open quantum systems*, `Quantum Information and
   Computation 15, 759-811 (2015)`.
8. Ang Li, Omer Subasi, Xiu Yang, and Sriram Krishnamoorthy, *Density Matrix
   Quantum Circuit Simulation via the BSP Machine on Modern GPU Clusters*, SC20.
9. Tyson Jones, Anna Brown, Ian Bush, and Simon C. Benjamin, *QuEST and High
   Performance Simulation of Quantum Computers*, `Scientific Reports 9, 10736
   (2019)`.
10. Yasunari Suzuki et al., *Qulacs: a fast and versatile quantum circuit
    simulator for research purpose*, `Quantum 5, 559 (2021)`.

## Traceability

- `DETAILED_PLANNING_PHASE_3_1.md`
- `ADRs_PHASE_3_1.md`
- `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`
- `task-5/TASK_5_MINI_SPEC.md`
