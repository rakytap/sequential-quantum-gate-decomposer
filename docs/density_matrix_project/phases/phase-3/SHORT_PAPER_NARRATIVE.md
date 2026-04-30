# Making Circuit Partitioning Native to Noisy Density-Matrix Workflows in SQUANDER

## Draft Status

This is now an implementation-backed narrative short-paper surface for the
Phase 3 Paper 2 package. The wording should track the delivered fused-runtime,
planner-calibration, correctness-evidence, and performance-evidence surfaces
directly, while final publication packaging remains open.

## Abstract

Exact noisy simulation is the strongest classical reference for understanding
how realistic noise changes the behavior of variational quantum circuits, but
its computational cost rises quickly and limits the scale of exact studies.
SQUANDER already closes the workflow-integration problem for one canonical noisy
XXZ `HEA` workflow through its Phase 2 density-matrix backend. The next
research question is therefore not whether exact noisy workflows are possible,
but whether their execution can be accelerated without weakening the exact
mixed-state semantics that make them scientifically valuable. Phase 3 addresses
that question by making noisy mixed-state circuits first-class objects of
partition planning and runtime execution. The current delivered contribution
combines a canonical noisy planner surface, partition descriptors that preserve
explicit gate/noise ordering, an executable partitioned density runtime with at
least one real fused execution mode, and a benchmark package that compares the
result against both the sequential `NoisyCircuit` baseline and Qiskit Aer. In the
current delivered planner-calibration result, the benchmark-facing planning
layer is a
benchmark-calibrated policy over a bounded family of auditable
`max_partition_qubits` span-budget settings on the existing noisy planner
surface rather than a broad family of separately implemented noisy planner
variants. The scope is deliberately bounded: fully channel-native fused noisy
blocks, broader noisy VQE/VQA growth, and approximate scaling methods remain
future work.

## Publication Surface Role

This document is the narrative short-paper surface for a general
PhD-conference audience within the Phase 3 Paper 2 package.

## Paper 2 Claim Boundary

Main claim:
SQUANDER can extend partitioning and limited fusion to exact noisy mixed-state
circuits without reducing noise to planner-external metadata, while preserving
exact semantics and yielding a scientifically useful benchmarked backend.

Current planner-calibration refinement:
the supported planning claim is presently a benchmark-calibrated selection rule
over a bounded noisy-planner candidate family, with broader adapted planner
families retained as design-space or comparison references until separately
implemented.

Current correctness-evidence refinement:
the delivered correctness package is now pinned to the currently selected
`span_budget_q2` surface and emitted through one shared `correctness_evidence` bundle
family. It records `25` counted supported cases, a bounded Qiskit Aer slice of
`4` cases, and `17` explicit unsupported-boundary cases that remain visible
instead of being filtered away.

Explicit non-claims:
- fully channel-native fused noisy blocks are not part of the baseline Paper 2
  claim
- broader noisy VQE/VQA feature growth and density-backend gradients are Phase
  4+ work
- approximate scaling methods remain future branches rather than current Paper 2
  results
- full direct `qgd_Circuit` parity is not a current Paper 2 claim

Supported-path boundary:
The guaranteed Paper 2 path is the canonical noisy mixed-state planner surface
plus exact lowering for the frozen Phase 2 continuity workflow and the required
Phase 3 structured benchmark families.

No-fallback rule:
No silent sequential fallback is part of the Phase 3 contract for benchmarks
that claim partitioned density behavior.

Evidence-closure rule:
Only mandatory, complete, supported correctness and reproducibility evidence,
plus either measured benefit or benchmark-grounded limitation reporting, closes
the main Paper 2 claim.

Phase positioning:
Paper 2 is the Phase 3 methods milestone between exact noisy integration and
broader noisy workflow science.

## 1. Introduction

The density-matrix track in SQUANDER exists because realistic noisy training
cannot be studied honestly with ideal state-vector simulation alone. Phase 2
already established the first exact noisy workflow result by integrating the
density backend into one canonical noisy XXZ `HEA` workflow. That solved the
workflow-integration problem, but it did not solve the execution-cost problem.

The project now sits at a clearer post-correctness-evidence transition point.
SQUANDER now
has:

- a mature state-vector partitioning and gate-fusion subsystem,
- and a validated exact noisy mixed-state backend.

What remained missing at the start of Phase 3 was a backend architecture in
which noisy mixed-state circuits are partitioned in their own right rather than
treated as unitary regions interrupted by noise boundaries that the planner did
not understand. The delivered planner, descriptor, runtime, fusion,
calibration, correctness, and benchmark layers now close that native-backend
and benchmark surface. The main open work is now the final publication-
packaging layer that sits on top of the delivered backend.

## 2. Why Phase 3 Matters

The main scientific value of Phase 3 is not just speed. It is the quality of the
backend contract that later noisy optimizer and trainability studies inherit.
If noisy mixed-state execution is accelerated by weakening semantics or by
ignoring noise placement in the planner, then Phase 4 and Phase 5 will stand on
a fragile foundation.

Phase 3 therefore matters because it asks a sharper question:

> can exact noisy mixed-state circuits become native objects of partition
> planning and runtime execution in a way that remains scientifically
> trustworthy?

That is the central Paper 2 narrative.

## 3. Planned Contribution

The contribution is deliberately bounded.

First, Phase 3 defines a canonical noisy mixed-state planner surface equivalent
to an ordered `NoisyCircuit` operation stream. This gives the planner direct
access to gate order, noise placement, qubit support, and parameter structure.

Second, the phase defines partition descriptors that preserve exact gate/noise
order rather than reducing noise to boundary-only metadata.

Third, the delivered result includes an executable partitioned density runtime
with at least one real fused execution mode on eligible substructures. This is
stronger than a planner-only story, but more bounded than a fully channel-native
fusion architecture.

Current fused-runtime findings sharpen that boundary. The baseline fused
result is now a concrete descriptor-local unitary-island path built on the
density backend's local-unitary primitive, not an abstract future possibility.
On representative 8- and 10-qubit structured workloads this path is real and
semantically exact, but it currently supports the diagnosis branch of the Phase
3 performance rule rather than a positive speedup claim. That is still a
meaningful methods result because it exposes where limited exact fusion helps,
where supported islands remain outside the fused core, and where more invasive
follow-on work would need benchmark justification.

Fourth, the benchmark package studies representative noisy workloads
rather than only synthetic kernels. The continuity anchor is the frozen Phase 2
noisy XXZ `HEA` workflow, and the methods stress matrix is built from structured
noisy `U3` / `CNOT` families under sparse, periodic, and dense local-noise
placement. In the current delivered performance-evidence package, that
benchmark layer now
records `34` counted supported cases:

- `4` continuity-anchor cases,
- and `30` structured cases across the three required families at 8 and 10
  qubits.

The representative review set contains `6` primary-seed sparse structured
cases, one per family and size. All `6` preserve exact semantics and exercise a
real fused path, but none reaches the positive threshold. The current diagnosis
therefore points less to an absence of fusion than to the fact that many
supported islands still remain unfused and the present fused path stays slower
than the sequential reference with no peak-memory reduction.

## 4. Validation Story

The paper should stay exact-first.

The internal exact reference is sequential `NoisyCircuit` execution. The
external exact reference is Qiskit Aer on the mandatory microcases and the
current 4-qubit continuity slice. The key paper requirement is not just that the partitioned
path runs, but that it preserves the exact semantics of the sequential density
baseline within strict thresholds.

The current benchmark package can now answer three questions:

1. Does the partitioned runtime preserve exact noisy mixed-state semantics?
2. Does the backend provide at least one real fused execution path rather than
   only a partition schedule?
3. Where do the resulting partitioning choices help, and where does the native
   baseline still leave a visible bottleneck?

For the currently delivered planner-calibration surface, the third question
should be read as
"how does the bounded noisy-planner candidate family behave under one
benchmark-grounded selection rule?" rather than as "which broad planner family
has now been permanently settled as the final winner?"

For the currently delivered correctness-evidence and performance-evidence
surfaces, that story is now implementation-backed end to end: one shared
correctness package records `25`
counted supported cases, `4` required external-reference cases, and explicit
negative boundary evidence, while the performance-evidence benchmark package
records `34` counted supported benchmark cases and keeps the `17`
carried-forward correctness-evidence boundary cases visible instead of relying
on prose-only performance claims.

The representative review set is deliberately smaller and more interpretable
than the full benchmark matrix. It contains `6` primary-seed sparse structured
cases, one for each required family and size. All `6` remain exact and exercise
the real fused path, but none reaches the positive threshold. The current
benchmark layer therefore closes through diagnosis-grounded limitation reporting
rather than through a speedup claim, which is still scientifically valuable
because it makes the next architecture question benchmark-rooted instead of
speculative.

## 5. Scientific Positioning

The Paper 2 claim should sit between two weaker stories.

It should be stronger than:

- "we represented noisy circuits in a planner,"
- or "we sped up a few kernels."

It should be more bounded than:

- "we solved fully general noisy-block fusion,"
- or "we built a completely general noisy workflow framework."

This middle position is exactly what makes the Phase 3 paper a clean methods
result in the publication ladder.

## 6. Expected Limitations

Even the successful baseline result will still have visible limitations:

- the exact dense regime remains scale-limited,
- the support surface remains workload-driven,
- fully channel-native fused noisy blocks remain deferred,
- and broader noisy VQE/VQA growth remains outside the paper's scope.

Those limitations are acceptable because they preserve a clear claim boundary
and make later phases scientifically distinct rather than muddled together.

## 7. Follow-On Phases

If the Phase 3 baseline closes well, the handoff is straightforward:

- Phase 4 broadens noisy workflows, gradients, and optimizer studies on top of a
  stabilized backend,
- and more invasive channel-native or approximate-scaling branches remain
  benchmark-driven follow-ons rather than hidden dependencies.

That sequencing keeps the density-matrix track coherent:

1. exact noisy workflow integration,
2. exact noisy partitioning and limited fusion,
3. broader noisy optimizer science,
4. then trainability analysis and optional scaling branches.

## References

Suggested narrative-facing reference shortlist:

- Peter Rakyta and Zoltan Zimboras, *Approaching the theoretical limit in
  quantum gate decomposition*, `Quantum 6, 710 (2022)`.
- Peter Rakyta, Gregory Morse, Jakab Nadori, Zita Majnay-Takacs, Oskar Mencer,
  and Zoltan Zimboras, *Highly optimized quantum circuits synthesized via
  data-flow engines*, `Journal of Computational Physics 500, 112756 (2024)`.
- Joseph Clark, Travis S. Humble, and Himanshu Thapliyal, *TDAG: Tree-based
  Directed Acyclic Graph Partitioning for Quantum Circuits*, ACM GLSVLSI 2023.
- Xin-Chuan Wu, Marc Grau Davis, Frederic T. Chong, and Costin Iancu, *QGo:
  Scalable Quantum Circuit Optimization Using Automated Synthesis*,
  `arXiv:2012.09835`.
- Ang Li, Omer Subasi, Xiu Yang, and Sriram Krishnamoorthy, *Density Matrix
  Quantum Circuit Simulation via the BSP Machine on Modern GPU Clusters*, SC20.
- Tyson Jones, Anna Brown, Ian Bush, and Simon C. Benjamin, *QuEST and High
  Performance Simulation of Quantum Computers*, `Scientific Reports 9, 10736
  (2019)`.
- Yasunari Suzuki et al., *Qulacs: a fast and versatile quantum circuit
  simulator for research purpose*, `Quantum 5, 559 (2021)`.
