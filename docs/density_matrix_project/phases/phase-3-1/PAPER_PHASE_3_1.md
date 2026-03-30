# Full Paper Draft: Bounded Exact Channel-Native Fusion for Local Noisy Motifs

## Draft Status

Implementation-backed full-paper surface revised to reflect the current Phase
3.1 state. This draft now includes initial hybrid whole-workload evidence:
exact `q4` continuity plus one structured `q8` pilot row whose current outcome
is overhead-dominant relative to the Phase 3 fused baseline.

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
support gate-noise motifs. The current design now has two execution
interpretations. A **strict** motif-proof path treats 1- and 2-qubit motifs
built from `U3`, `CNOT`, and local depolarizing, amplitude-damping, or phase-
damping channels as exact CPTP objects represented primarily by Kraus bundles.
An explicit **hybrid** whole-workload path is the current evaluation vehicle
for continuity and structured benchmark cases, where eligible partitions run
channel-natively and Phase-3-supported but Phase-3.1-ineligible partitions stay
on the shipped exact baseline with route attribution. Composition follows
ordered noisy-circuit semantics, invariant checks enforce trace preservation and
positivity, and the fused object can be applied on local support within a larger
density state. Current implementation-backed validation now spans both layers:
`<= 1e-10` Frobenius-norm agreement with sequential exact execution is shown on
a 1-qubit motif, one counted 2-qubit motif, a 4-qubit fully eligible local-
support smoke case, and the counted hybrid continuity anchor
`phase2_xxz_hea_q4_continuity`, together with deterministic rejection of
out-of-scope motifs. A first frozen 8-qubit structured hybrid pilot row also
records route coverage plus sequential, Phase 3 fused, and hybrid timings; on
the current evidence, that row remains slower than the existing Phase 3 fused
baseline. These results establish a bounded exactness-and-decision-study
contribution. They do not yet close a broader acceleration claim: publication-
backed coverage of the remaining counted rows, external Qiskit Aer validation,
and the full structured 8- and 10-qubit matrix are still pending.

## Publication Surface Role

This document is the full-paper draft surface for the Phase 3.1 publication
track. It follows a research-paper structure with explicit problem statement,
method, current results, limitations, and reproducibility posture.

## Current Claim Boundary

**Current supported claim**

- Small same-support noisy motifs can be fused exactly as CPTP objects inside a
  partitioned density-matrix execution flow, carried by the **strict**
  motif-proof runtime interpretation.
- The fused object can be validated through sequential-reference agreement plus
  channel-invariant checks.
- Local-support embedding into a larger density state is feasible on the current
  bounded slice, including a larger but still fully eligible smoke workload.
- Explicit **hybrid** whole-workload execution with partition-level route
  attribution is implemented and scientifically interpretable.
- The counted hybrid continuity anchor `phase2_xxz_hea_q4_continuity` executes
  exactly under the hybrid interpretation.
- One frozen structured hybrid pilot row,
  `phase31_pair_repeat_q8_periodic_seed20260318`, records baseline-trio timing
  and route coverage. The current row supports only the
  negative-to-inconclusive conclusion that hybrid overhead still dominates the
  shipped Phase 3 fused baseline.
- Unsupported motifs remain explicit rather than silently falling back.

**Claim-bearing gate that remains open**

- Publication-backed coverage of the remaining strict mixed-motif rows,
- publication-backed coverage of the remaining hybrid continuity anchor
  `phase2_xxz_hea_q6_continuity`,
- external Qiskit Aer validation on the frozen strict-plus-hybrid external
  slice,
- and the full structured 8- and 10-qubit performance matrix with
  control-family closure and decision-surface reporting.

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

The runtime interpretation now has two levels. The **strict** path proves the
exact fused object on fully eligible motif workloads. The explicit **hybrid**
path is the current whole-workload evaluation vehicle for continuity and
structured benchmark cases, where some partitions are eligible for the new exact
channel-native treatment and others remain on the shipped exact Phase 3 path.

The current contribution can therefore be stated in three parts:

- a bounded exact fused object for 1- and 2-qubit mixed motifs under a strict
  motif-proof interpretation,
- a route-attributed hybrid whole-workload interpretation together with an
  exact counted `q4` continuity anchor,
- and a first structured `q8` pilot row showing that whole-workload performance
  justification remains open and may be narrower than motif-level feasibility.

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

The current execution interpretation is likewise split deliberately:

- **strict** execution proves the bounded fused object on fully eligible
  workloads,
- **hybrid** execution is the current whole-workload path for the
  counted continuity and structured benchmark slice.

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

### 4.5 Strict and Hybrid Execution Interpretation

The current contract separates two execution interpretations.

In **strict** execution, every partition in the workload must be fully eligible
for the bounded 1- and 2-qubit mixed-motif contract. This is the correct path
for proving the fused object itself and for small counted motif cases. Any
ineligible partition is a hard failure.

In **hybrid** execution, eligible partitions use the channel-native exact path,
while partitions that remain supported by the shipped Phase 3 exact runtime but
fall outside the bounded Phase 3.1 eligibility surface stay on that baseline
with explicit route attribution. This interpretation is required for continuity
and structured whole-workload benchmarking because many such workloads mix
strictly eligible and merely Phase-3-supported partitions.

The scientific importance of this split is interpretability: the field should be
able to distinguish "the new fused object was proved here" from "the new fused
object was exercised inside a larger exact workload here."

## 5. Current Implementation-Backed Evaluation

### 5.1 Current Implemented Slice

The current implementation-backed slice now spans both the **strict** and the
initial **hybrid** layer. It includes:

- one 1-qubit mixed motif,
- one counted 2-qubit mixed motif,
- and one non-counted 4-qubit spectator-support smoke case whose partitions all
  remain fully eligible for the strict path,
- one counted 4-qubit hybrid continuity anchor,
- and one frozen 8-qubit structured hybrid pilot row.

Deterministic workload inventories also already exist for the remaining counted
microcases and for the future 8- and 10-qubit structured performance families.
Under the current contract, those remaining whole-workload rows are to be
carried by the explicit **hybrid** path rather than by the strict path, but
they are not yet wired into claim-closing Phase 3.1 evidence bundles.

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

The current results support six statements.

First, the 1-qubit mixed motif matches the sequential exact reference within the
frozen threshold.

Second, the counted 2-qubit motif, built around `CNOT` plus local noise on the
same support, also matches the sequential exact reference within the same
threshold and satisfies the current invariant checks.

Third, the 4-qubit spectator-support smoke case shows that a bounded 2-qubit
fused noisy object can be embedded into a larger density state while preserving
the correctness of the full global output.

Fourth, the counted `q4` continuity anchor shows that the new fused object can
be exercised inside a larger exact workload that mixes channel-native and
shipped Phase 3 partitions without losing full-matrix correctness or route
auditability.

Fifth, the first frozen `q8` structured pilot row records the baseline trio and
explicit route coverage. On the current evidence, this row is overhead-dominant
relative to the existing Phase 3 fused baseline, so it supports a negative-to-
inconclusive whole-workload decision result rather than a positive acceleration
claim.

Sixth, out-of-scope motifs fail deterministically. Pure unitary motifs, motifs
with support above two qubits, and motifs using unsupported operations remain
visible as unsupported behavior rather than being silently absorbed into a
different path.

Across the strict slice, ordered composition remains claim-bearing: reversing
the composed sequence changes the result, so ordered noisy semantics must
remain explicit.

### 5.4 What Is Still Missing

The current full-paper claim remains incomplete because four layers are still
missing:

- publication-backed coverage of the remaining counted correctness rows,
- the remaining counted `q6` hybrid continuity anchor,
- the external Qiskit Aer slice,
- and the structured 8- and 10-qubit benchmark matrix together with the
  required control-family and decision-surface reporting.

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
performance-justified object on the workload families that matter most. Under
the current contract, that question is now explicitly assigned to the **hybrid**
whole-workload interpretation rather than to the strict motif-proof path alone.
The first hybrid pilot already shows why this separation matters: the exact
fused object can be sound and scientifically useful even when the current whole-
workload row does not yet justify using it as a faster runtime. That is the
next claim-bearing gate, and the paper should say so directly.

## 7. Threats to Validity and Limitations

Several limitations must remain explicit.

### 7.1 Narrow Support Surface

The current result is deliberately restricted to bounded 1- and 2-qubit local
motifs with specific gate and noise families. It should not be interpreted as a
claim about correlated noise, larger supports, or arbitrary CPTP fusion.

### 7.2 Partial Whole-Workload Evidence and Missing Matrix Closure

The current paper does not yet have the full external Qiskit Aer slice, the
remaining counted hybrid continuity rows, or the structured 8- and 10-qubit
performance matrix. It does have one hybrid pilot row, but one row is not a
matrix, and the current row is not positive. This is the main reason broader
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

The current reproducibility posture follows five rules.

- Use deterministic microcase definitions and stable case identifiers.
- Report explicit numerical thresholds for exactness and invariant checks.
- Keep unsupported cases visible.
- Separate strict motif-proof evidence from future hybrid whole-workload
  evidence.
- For hybrid rows, report route coverage and the baseline trio explicitly rather
  than treating all Phase 3.1 execution as one opaque class.

At the current state of the work, reproducibility is strongest at the level of
deterministic workload definitions and exact-threshold regression tests. The
later full evidence package should add the remaining external-reference rows,
the broader structured benchmark matrix, and explicit decision-surface
artifacts.

## 9. Conclusion

This paper surface now supports a clear, bounded conclusion. Exact channel-
native fusion of local noisy motifs is feasible, auditable, and already
demonstrated on a narrow implementation-backed **strict** slice. That is
scientifically useful because it establishes a concrete bridge between unitary
fusion ideas and open-system channel semantics. The hybrid layer is no longer
only planned: a counted `q4` continuity anchor shows whole-workload exactness
with explicit route attribution, and a first frozen `q8` pilot row provides the
initial decision-study evidence. That pilot does not yet justify broader
acceleration language. The remaining counted correctness rows, external
validation, and the full structured benchmark matrix are still required before
stronger workload-level claims can be made.

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
