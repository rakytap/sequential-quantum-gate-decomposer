# Beyond Unitary Fusion in Exact Noisy Quantum Simulation

## Draft Status

Pre-closure science-first narrative companion aligned to
`CLOSURE_PLAN_PHASE_3_1.md`. The required
`PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md` now exists and currently records
the closure state as `decision-study-ready`, so this document should now be
read as a boundary-synchronized narrative companion for the bounded
decision-study outcome rather than as a not-ready draft. The current text reflects the stronger implementation-backed
evidence boundary now present in the repo while keeping the narrative
scientific rather than task-oriented.

## Abstract

Partitioning and gate fusion are standard tools in unitary quantum simulation,
while open-system theory provides the language of Kraus maps, Choi matrices,
and superoperators for noisy evolution. What remains underdeveloped is the
bridge between these two traditions: how to treat a small noisy region of a
circuit as an exact reusable object without losing the ordered semantics of
open-system evolution. The present study provides a bounded answer. It shows
that same-support 1- and 2-qubit noisy motifs can be composed as exact CPTP
objects and validated against sequential density-matrix evolution, rather than
being treated only as barriers between unitary islands. It also sharpens the
methodological picture: a scientifically credible path appears to need both a
strict motif-proof layer, where the fused object itself is validated, and a
separate hybrid whole-workload layer, where that object is tested inside larger
exact workloads with explicit route attribution. The current evidence now
reaches both layers more strongly than earlier drafts suggested: the bounded
counted correctness package spans the strict microcase surface plus two hybrid
continuity anchors, the bounded external-reference slice is already present on
the frozen required rows, and a first frozen 8-qubit pilot row shows that
feasibility does not automatically imply performance benefit. The scientific
value is therefore methodological and decision-oriented: it identifies a
reusable path toward exact noisy acceleration while also showing that broader
whole-workload justification remains an empirical question rather than an
assumption.

## Publication Surface Role

Scientific narrative short paper for a quantum-computing or quantum-machine-
learning audience. Its purpose is to explain why the current bounded result
matters, where it fits in the literature, and what reusable insight it offers.
Until the pre-publication evidence review exists, this document functions as a
boundary-sync narrative companion rather than as a submission-ready narrative
paper.

## 1. Why This Problem Matters

Exact noisy simulation matters because realistic local noise can qualitatively
change the behavior of variational algorithms. For questions about trainability,
loss landscapes, or optimizer robustness, a state-vector-only picture is often
not enough. Density-matrix simulation provides the cleanest exact reference, but
it is computationally expensive and therefore forces careful choices about what
kind of structure can be reused safely.

This tension creates a natural scientific target: identify classes of noisy
structure that can be fused exactly, so that noisy simulation becomes more
structured than "unitaries plus interruptions" without sacrificing physical
correctness.

## 2. Where the Literature Is Split

The current literature offers strong ingredients, but not yet a full answer.

Graph-based partitioning and gate-fusion papers such as TDAG, GTQCP, QGo, and
QMin show how much can be gained when circuits are treated as structured
dependency objects. However, these methods are typically developed for unitary
or state-vector simulation.

Open-system and density-matrix literature, by contrast, gives the mathematical
language needed for exact noisy evolution. Nielsen and Chuang formalize quantum
channels and CPTP structure; Wood, Biamonte, and Cory make explicit the
relations among Kraus, Choi, and Liouville forms; Li et al., QuEST, and Qulacs
show that exact mixed-state simulation is a serious high-performance computing
problem in its own right.

What is still relatively unexplored is the exact noisy middle ground: bounded
fusion of mixed gate-noise motifs inside a partitioned execution flow.

## 3. Current Scientific Result

The current result is intentionally narrow but already meaningful.

It shows that a small same-support noisy motif can be promoted to a first-class
exact object: a bounded CPTP block composed in operation order and validated
against sequential density-matrix evolution. In the present bounded slice, this
has been demonstrated for 1-qubit and 2-qubit motifs and for local-support
application inside a larger density state.

Just as importantly, the current interpretation now separates two scientific
roles that are often blurred together. One role is to prove that the exact noisy
object itself is sound. The other is to test whether that object helps inside
larger exact workloads that still contain structure outside the bounded fused
slice. The former belongs to a strict motif-proof setting; the latter belongs
to an explicit hybrid whole-workload setting.

This two-layer story is no longer only conceptual. The hybrid layer now has two
counted whole-workload correctness anchors at 4 and 6 qubits that match the
sequential oracle while preserving explicit route attribution between
channel-native and existing exact execution. The bounded external-reference
slice is also already present on the frozen required rows, which keeps the
current narrative tied to a real external check without turning the study into a
broad simulator comparison. In addition, the phase now has a first frozen
8-qubit structured pilot row with route coverage and comparative timing. That
pilot currently indicates overhead-dominant behavior relative to the existing
Phase 3 fused baseline, which is scientifically useful because it separates
mathematical feasibility from workload-level justification.

Scientifically, this matters because it changes the role of noise in the
simulation narrative. Noise is no longer treated only as a point where fusion
must stop. Instead, some noisy motifs can themselves become the fused object,
provided the representation, ordering, and invariant checks are all exact.

This is a result about **feasibility and methodology**, not yet about universal
performance. That distinction is important and should remain explicit.

## 4. Reusable Working Principles

The current bounded study suggests five reusable principles for future exact
noisy acceleration work.

- Start from the **ordered noisy semantics**, not from the representation alone.
  A fused object is only meaningful if it preserves the exact order of gates and
  channels.
- Choose one **primary exact representation** for the counted claim. Here the
  useful lesson is not that Kraus form is universally best, but that one primary
  representation should anchor the scientific claim.
- Require **physical invariant checks** before making performance claims.
  Trace-preservation and positivity are not optional bookkeeping; they are part
  of the evidence that the fused object remains a valid channel.
- Enforce **no silent fallback**. If a method advertises noisy fusion, then
  unsupported cases must remain visible. Otherwise benchmarks become difficult
  to interpret scientifically.
- Separate **proof mode** from **whole-workload evaluation mode**. The exact
  fused object can be scientifically established on a strict bounded slice
  before broader workload-level claims are made through an explicit hybrid
  interpretation.

These principles are potentially reusable beyond the present bounded support
surface. The first hybrid pilot now provides an empirical reason for this
separation: exactness can close before performance justification does.

## 5. What Remains Hard

Several hard questions remain open.

First, mathematical feasibility and performance usefulness are not the same.
Showing that a noisy motif can be fused exactly does not yet show that the
resulting method improves runtime or memory on the workloads that matter most.
The first hybrid pilot already illustrates this separation: a method can be
exact and auditable on a whole workload while still failing to outperform the
existing baseline on that row.

Second, exact noisy acceleration remains scale-limited by dense mixed-state
representation itself. Even a successful bounded fusion method operates inside a
regime where memory and data movement remain fundamental constraints.

Third, broader noisy structure is still unresolved: correlated noise, larger
supports, and more heterogeneous motif families may require different
representations or different validation strategies.

The full structured 8- and 10-qubit matrix, its decision artifact, and the
formal pre-publication evidence review still remain before one can say how
general the current negative-to-inconclusive pilot result really is.

The right scientific conclusion is therefore not "noisy fusion is solved," but
"a bounded exact path now exists and can be studied honestly."

## Selected References

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
- Yasunari Suzuki et al., *Qulacs: a fast and versatile quantum circuit
  simulator for research purpose*, `Quantum 5, 559 (2021)`.
- Joseph Clark, Travis S. Humble, and Himanshu Thapliyal, *TDAG: Tree-based
  Directed Acyclic Graph Partitioning for Quantum Circuits*, ACM GLSVLSI 2023.
- Joseph Clark, Travis S. Humble, and Himanshu Thapliyal, *GTQCP: Greedy
  Topology-Aware Quantum Circuit Partitioning*, `arXiv:2410.02901`.
- Longshan Xu, Edwin Hsing-Mean Sha, Yuhong Song, and Qingfeng Zhu, *QMin:
  Quantum Circuit Minimization via Gate Fusions for Efficient State Vector
  Simulation*, `Quantum Information Processing 25, 6 (2026)`.

## Traceability

- `CLOSURE_PLAN_PHASE_3_1.md`
- `SHORT_PAPER_PHASE_3_1.md`
- `PAPER_PHASE_3_1.md`
- `task-5/TASK_5_MINI_SPEC.md`
