# Paper 2 Draft for Phase 3

## Title Candidates

- `Noise-Aware Partitioning and Gate Fusion for Exact Mixed-State Quantum Circuit Simulation`
- `Making Circuit Partitioning Native to Noisy Density-Matrix Workflows in SQUANDER`
- `Exact Noise-Aware Partitioning for Mixed-State Quantum Circuit Simulation in SQUANDER`

## Draft Status

This document is a planning-facing full-paper draft for Phase 3. It defines the
target narrative, evidence structure, and claim boundaries for Paper 2, but it
is not yet an implementation-backed paper surface. Any results language should
be converted to implementation-backed wording only after the mandatory Phase 3
benchmark and validation package exists.

## Abstract Summary

This document is the full-paper draft surface for the planned Phase 3 methods
and systems paper. A compact conference abstract version is maintained
separately in `ABSTRACT_PHASE_3.md`.

## Publication Surface Role

This document is the planning-facing full-paper draft surface for the Phase 3
Paper 2 package.

## Paper 2 Claim Boundary

Main claim:
SQUANDER extends partitioning and limited fusion to exact noisy mixed-state
circuits by making noisy operations first-class planner inputs, preserving exact
gate/noise semantics across partition descriptors and runtime execution, and
validating the resulting partitioned density path on representative noisy
workloads.

Supporting claims:
- a canonical noisy mixed-state planner surface aligned with the `NoisyCircuit`
  execution contract
- exact semantic-preservation rules for gate/noise ordering, parameter routing,
  and partition execution
- an executable partitioned density runtime with at least one real fused
  execution mode on eligible substructures
- a benchmark-calibrated density-aware planning objective or heuristic
- a publication-grade correctness, performance, and reproducibility package

Explicit non-claims:
- fully channel-native or superoperator-native fused noisy blocks are outside
  the baseline Paper 2 claim
- broader noisy VQE/VQA workflow growth, density-backend gradients, and
  optimizer studies remain Phase 4+ work
- approximate scaling methods such as trajectories or MPDO-style approaches are
  outside the current Paper 2 claim
- full `qgd_Circuit` parity is not part of the baseline Paper 2 claim
- universal speedup across every noisy workload is not required for Paper 2

Supported-path boundary:
The guaranteed Paper 2 path is the canonical noisy mixed-state planner surface
plus the documented exact lowering required for the frozen Phase 2 continuity
workflow and the required Phase 3 structured benchmark families.

No-fallback rule:
No silent sequential fallback is part of the Phase 3 contract for any benchmark
that claims partitioned density behavior.

Exact-regime boundary:
Mandatory internal correctness coverage spans 4, 6, 8, and 10 qubits, with
external micro-validation at 2 to 4 qubits and required performance recording
on representative 8- and 10-qubit structured families.

Evidence-closure rule:
Only mandatory, complete, supported correctness and reproducibility evidence,
plus either measured benefit or benchmark-grounded limitation reporting, closes
the main Paper 2 claim.

Phase positioning:
Paper 2 is the Phase 3 noise-aware partitioning and fusion milestone in the
density-matrix publication ladder.

## 1. Introduction

Exact density-matrix simulation remains the cleanest classical reference for
studying quantum circuits under realistic noise, especially when the scientific
goal is not only simulation but noisy variational training. Phase 2 of the
SQUANDER density-matrix program already closed the first major integration gap:
the project now has a validated exact noisy backend for one canonical noisy XXZ
`HEA` workflow together with a publication-grade evidence bundle. That result
turned the density-matrix module into a usable scientific instrument.

At the same time, Phase 2 also made the next bottleneck impossible to ignore.
Exact dense mixed-state simulation is expensive, and the current partitioning
and gate-fusion stack remains structurally tied to a state-vector execution
model. SQUANDER therefore now contains two strong assets that are not yet
properly connected:

- a sequential exact noisy mixed-state runtime centered on `NoisyCircuit`,
- and a mature partitioning/fusion subsystem centered on `qgd_Circuit`,
  `Gates_block`, and state-vector-oriented planning heuristics.

The purpose of Phase 3 is to bridge that gap. The question is no longer whether
SQUANDER can support an exact noisy workflow at all. The question is whether
partitioning and limited fusion can be extended so noisy mixed-state circuits
become native planner objects while exact noisy semantics remain intact.

This paper is therefore not a workflow-surface paper and not an approximate
scaling paper. It is a methods and systems paper about making noisy
mixed-state circuits first-class inputs to partition planning and runtime
execution.

### 1.1 Main Contribution

The planned Phase 3 contribution has five parts:

- a canonical noisy mixed-state planner surface aligned with the sequential
  density backend,
- partition descriptors that preserve exact gate/noise order, qubit support,
  and parameter-routing semantics,
- an executable partitioned density runtime with at least one real fused
  execution mode on eligible substructures,
- a density-aware planning heuristic or objective calibrated on representative
  noisy workloads,
- and a benchmark and validation package strong enough to support a major
  methods paper rather than only an internal architecture note.

### 1.2 Why This Contribution Matters

This contribution matters for two reasons.

First, it creates the missing backend step between exact noisy integration and
later noisy optimizer science. Without a stabilized Phase 3 backend, Phase 4
workflow expansion would sit on top of an execution-cost bottleneck that has not
yet been studied in the right architectural terms.

Second, it is a scientifically meaningful methods contribution in its own right.
Many existing partitioning and fusion methods assume unitary circuits or
state-vector semantics. Extending partition planning so noise channels and
mixed-state execution are native rather than external is a stronger and more
research-relevant result than a narrow kernel-only optimization story.

### 1.3 Contract Learnings That Shape Paper 2

The current Phase 3 contract bakes several lessons from the Phase 2 workflow and
planning process directly into the planned paper narrative:

- exact semantic preservation is more important than aggressive but weakly
  specified optimization,
- silent fallback behavior is scientifically unacceptable,
- support-tier classification (`required`, `optional`, `deferred`,
  `unsupported`) is necessary to keep claims auditable,
- the sequential density path is the internal exact ground truth that the Phase
  3 backend must preserve,
- and negative or mixed performance results can still produce a valid methods
  paper when the benchmark package explains the limiting factor honestly.

## 2. Background and Related Work

### 2.1 SQUANDER Foundations

SQUANDER already has strong lineage in gate decomposition, optimization, and
high-performance quantum circuit execution. That foundation matters for Phase 3
because the density-matrix track is not being developed in isolation. It is
being integrated into an existing framework that already has:

- a mature partitioning stack,
- optimized circuit representations,
- and workflow infrastructure designed for variational research.

The foundational SQUANDER papers provide the platform context for why Phase 3 is
a natural methods milestone rather than an isolated simulator project.

### 2.2 Partitioning and Gate-Fusion Literature

The current partitioning subsystem already reflects a clear research lineage
through TDAG, GTQCP, and QGo, while QMin and the Nguyen et al. gate-fusion work
provide useful context for control-aware and runtime-oriented fusion thinking.
These papers motivate several important Phase 3 choices:

- planner structure matters,
- gate fusion should be evaluated in runtime terms, not only symbolic terms,
- and workload-aware partitioning is a better scientific target than broad but
  weakly grounded parity claims.

However, most of this prior art is still unitary- or state-vector-oriented.
Phase 3 differs by asking how noise placement and mixed-state semantics enter
the planner and runtime contract directly.

### 2.3 Exact Density-Matrix and HPC Context

The density-matrix and HPC literature, including Li et al., Doi and Horii,
QuEST, Qulacs, and Atlas, makes two things clear. First, exact mixed-state
simulation is a serious systems problem with rapidly rising computational and
memory cost. Second, memory locality, planning overhead, and execution structure
matter strongly even before very large scale.

That context is essential for Paper 2 because it justifies why exact noisy
partitioning is not merely an implementation convenience. It is the main route
to making the exact backend scientifically useful beyond the narrowest workflow
regime.

### 2.4 Why This Paper Is Not About Approximate Scaling

Approximate scaling directions such as stochastic trajectories or MPDO-style
methods remain scientifically important, but the planning set deliberately keeps
them outside the early critical path. The Phase 3 paper is therefore about
extending exact noisy mixed-state simulation, not replacing it.

### 2.5 Broader Noisy-Training Motivation

The broader PhD motivation remains noisy variational training and trainability
under realistic noise. That motivation matters here because Phase 3 is judged
not only by raw simulator throughput, but by its value as the backend that later
optimizer and trainability studies will depend on.

## 3. Current SQUANDER Baseline After Phase 2

The baseline for this paper is the system state after the frozen Phase 2
contract.

### 3.1 What Exists Today

At the start of Phase 3, the codebase already provides:

- an exact dense mixed-state backend with `DensityMatrix`, `GateOperation`,
  `NoiseOperation`, and `NoisyCircuit`,
- a validated exact noisy workflow slice for one canonical noisy XXZ `HEA`
  workflow,
- explicit backend selection and exact `Re Tr(H*rho)` support in the frozen
  Phase 2 scope,
- and a mature state-vector partitioning and gate-fusion subsystem in
  `squander/partitioning` together with `Gates_block`.

### 3.2 What Does Not Yet Exist

What remains open is precisely what defines the Phase 3 problem:

- noisy mixed-state circuits are not yet native planner inputs,
- partition descriptors do not yet preserve exact noisy semantics as part of the
  planner/runtime contract,
- the density backend does not yet have an executable partitioned runtime with a
  real fused path,
- and the current planning objective remains state-vector-oriented and
  noise-blind.

### 3.3 Why This Gap Matters

Phase 2 already established the exact noisy workflow baseline that later science
needs. The remaining problem is therefore a backend methods problem, not a
workflow-existence problem. Solving that methods problem is what makes Phase 4
and Phase 5 scientifically stronger rather than merely broader.

## 4. Phase 3 Problem Definition

The Phase 3 problem can be stated as:

> how to make partitioning and limited fusion native to exact noisy mixed-state
> circuits in SQUANDER without sacrificing the exact semantics that define the
> value of the density backend.

This problem has both scientific and architectural dimensions.

Scientifically, the backend must remain exact enough to serve as a trusted
reference. Architecturally, the planner must become aware of noise placement and
mixed-state execution structure rather than treating them as external boundary
conditions.

The core design tension is therefore:

- broad optimization freedom versus strict semantic preservation,
- and planner/runtime generality versus a bounded publishable baseline.

Phase 3 resolves this tension by choosing a correctness-first, workload-driven
baseline rather than a maximally invasive architecture from the start.

## 5. Planned Architecture and Method

### 5.1 Canonical Noisy Mixed-State Planner Surface

Phase 3 defines a canonical internal planner surface equivalent to an ordered
`NoisyCircuit` operation stream built from `GateOperation` and `NoiseOperation`.
This decision makes noisy mixed-state operations native planner objects.

The intended benefit is clarity:

- planner inputs are aligned with the real density backend,
- the frozen Phase 2 continuity workflow can be lowered into that surface
  exactly,
- and the methods claim no longer depends on pretending that noise exists only
  at partition boundaries.

### 5.2 Partition Descriptor and Semantic-Preservation Contract

The planner is not sufficient on its own. The partition descriptor must retain
enough information to execute partitioned circuits faithfully.

The required descriptor contract includes:

- exact gate/noise order,
- qubit support,
- parameter-routing metadata,
- and remapping information when partition-local execution changes indexing.

This is the main reason Phase 3 is stronger than a planner-only story. The
paper's value depends on showing that partitioning is not achieved by weakening
the exact noisy semantics.

### 5.3 Executable Partitioned Density Runtime

Phase 3 minimum closure requires an end-to-end partitioned density execution
path on the mandatory benchmark matrix. This runtime must be explicit in the
benchmark harness and must not silently substitute the sequential density path
for cases that claim partitioned execution.

This requirement makes the methods claim operational rather than symbolic.

### 5.4 Real Fused Execution on Eligible Substructures

The planned Paper 2 claim also requires at least one real fused execution mode
inside the noisy partitioned runtime. This is intentionally weaker than
requiring fully channel-native fused noisy blocks, but it is stronger than
claiming only partition scheduling or partition descriptors.

The fused path may still remain unitary-island-based internally if:

- the surrounding noisy semantics remain exact,
- and the fused path is benchmarked on representative noisy workloads.

Current Task 4 implementation findings now make this branch concrete. The
delivered minimum fused path is descriptor-local unitary-island fusion on 1- and
2-qubit spans using the density backend's local-unitary primitive. The fused
runtime extends the shared Task 3 surface additively through an explicit fused
runtime-path label plus auditable fused, supported-but-unfused, and deferred
region records rather than through a second private runtime schema.

### 5.5 Density-Aware Planning Heuristic and Calibration

The current cost model in `squander/partitioning/tools.py` is not sufficient to
close a density-aware methods claim. Phase 3 therefore follows a staged
approach:

1. establish native noisy-circuit correctness,
2. deliver the executable partitioned runtime,
3. apply structural noise-aware planning heuristics,
4. calibrate a density-aware objective or heuristic on the benchmark matrix.

This keeps Paper 2 honest. The planner is allowed to begin from structural
reuse of the state-vector machinery, but the final methods claim must be backed
by density-aware evidence rather than by a renamed state-vector cost model.

## 6. Validation Methodology

Because the contribution is exact-first, validation is central to the planned
paper.

### 6.1 Primary Internal Baseline

Sequential `NoisyCircuit` execution is the required internal exact reference for
every mandatory Phase 3 case. This preserves continuity with the delivered Phase
2 semantics and ensures that any acceleration claim is also a semantic claim.

### 6.2 Primary External Baseline

Qiskit Aer density-matrix simulation remains the required external reference on
the mandatory 2 to 4 qubit microcases and representative small continuity
cases. This keeps the exactness story anchored to a strong external baseline
without turning Paper 2 into a broad simulator bake-off.

### 6.3 Validation Targets

The planned validation package has three layers:

- micro-validation at 2 to 4 qubits stressing partition boundaries, noise
  placement, and required local noise classes,
- continuity validation on the frozen Phase 2 noisy XXZ `HEA` workflow at 4, 6,
  8, and 10 qubits,
- structured partitioning-family validation on representative 8- and 10-qubit
  noisy `U3` / `CNOT` families.

### 6.4 Numeric Acceptance Thresholds

The planned Phase 3 thresholds are intentionally strict:

- Frobenius-norm density difference `<= 1e-10` against Qiskit Aer on required
  microcases,
- Frobenius-norm density difference `<= 1e-10` between partitioned and
  sequential execution on mandatory correctness cases,
- `|Tr(rho) - 1| <= 1e-10` and `rho.is_valid(tol=1e-10)` on required outputs,
- energy agreement `<= 1e-8` on the Phase 2 continuity anchor cases.

The final paper should state these thresholds explicitly rather than relying on
qualitative language such as "matches the baseline."

### 6.5 Why Validation Must Be Both Internal and External

Using only the external baseline would weaken continuity with the exact backend
that the project itself is supposed to preserve. Using only the internal
baseline would weaken the paper's scientific credibility. The two-baseline model
is therefore part of the planned claim structure, not a convenience.

## 7. Benchmark Design

### 7.1 Workload Classes

The benchmark package is intentionally built around both continuity and methods
coverage.

Continuity anchor:

- the frozen Phase 2 noisy XXZ `HEA` workflow at 4, 6, 8, and 10 qubits.

Structured methods families:

- layered nearest-neighbor noisy `U3` / `CNOT` circuits,
- seed-fixed random noisy `U3` / `CNOT` circuits,
- and one structured partitioning stress family in the same required gate
  surface.

### 7.2 Noise Classes and Placements

Mandatory scientific noise classes remain:

- local depolarizing,
- local amplitude damping,
- local phase damping or dephasing.

The planned sensitivity study covers:

- sparse placement,
- periodic layer-boundary placement,
- dense layer-wise placement.

This design ensures the paper is about noise-aware partitioning rather than
partitioning in the presence of a single convenient noise schedule.

### 7.3 Metrics

The final paper should report:

- density agreement with the sequential baseline,
- exactness against Qiskit Aer where required,
- energy agreement on continuity-anchor cases,
- runtime,
- peak memory,
- planner runtime,
- partition count and qubit span,
- and fused-path coverage on eligible substructures.

### 7.4 Performance Interpretation

The paper should not overcommit to a universal speedup claim. Instead, it should
make a stronger methodological promise:

- exact semantics must hold on the mandatory matrix,
- and the benchmark package must either demonstrate useful benefit on
  representative cases or provide a benchmark-grounded diagnosis of where the
  current native baseline still falls short.

That second outcome is still scientifically valuable if the diagnosis is
structured, reproducible, and clearly linked to the follow-on architecture
decision gate.

The current Task 4 baseline already illustrates that second outcome. On
representative 8- and 10-qubit layered nearest-neighbor sparse workloads, the
real fused path preserves exact semantics and exercises substantial fused
coverage, but the measured median runtime remains slower than the plain
partitioned baseline and peak memory does not improve. The current benchmarked
diagnosis points primarily to supported islands that still remain unfused plus
Python-level fused-kernel overhead in the present baseline implementation. Paper
2 should therefore frame the current performance result as diagnosis-grounded
and architecture-informing rather than as a positive acceleration claim.

## 8. Expected Scientific Claims

If the planned benchmark package closes successfully, the final Phase 3 paper
should be able to claim:

- noisy mixed-state circuits are first-class partitioning inputs in SQUANDER,
- the partitioned runtime preserves exact noisy semantics relative to the
  sequential density baseline,
- the system delivers more than planner-only representation by executing
  partitioned noisy circuits end to end with at least one real fused path,
- performance and memory behavior on representative workloads are characterized
  and scientifically interpretable,
- and fully channel-native fused noisy blocks remain an explicit follow-on
  branch rather than an unacknowledged missing piece.

The paper should avoid stronger claims unless the evidence genuinely supports
them. In particular, it should not overstate generality, universal speedup, or
full parity across all circuit sources.

## 9. Threats to Validity and Limitations

### 9.1 Limited Exact Scale

Dense density matrices remain fundamentally expensive. Even with successful
partitioning and limited fusion, the exact regime stays bounded. Paper 2 should
therefore present Phase 3 as a methods advance within an honest exact regime,
not as a claim to arbitrarily large exact simulation.

### 9.2 Workload-Driven Support Surface

The support matrix remains deliberately narrow and workload-driven. This is
appropriate for a bounded methods paper, but it also limits generality. The
final paper should state that explicitly.

### 9.3 Planner-Calibrated but Not Fully Architecture-Final

The staged heuristic-to-calibration approach means the Phase 3 baseline may not
yet be the final word in density-aware planning. This is acceptable, but only
if the paper explains which claims are benchmark-grounded today and which are
left to later refinement.

### 9.4 No Fully Channel-Native Fused Noisy Blocks

The baseline Phase 3 paper intentionally stops short of fully channel-native
fused noisy blocks. This should be framed as a deliberate architecture boundary,
not as an oversight.

### 9.5 Benchmark Selection Bias

If the structured families are too favorable to the chosen support surface, the
methods paper weakens. Benchmark design must therefore stay visibly tied to the
declared Phase 3 workload classes and noise-placement sensitivity matrix.

## 10. Connection to Later Phases

The planned Paper 2 narrative should end with a clear handoff to later phases.

### 10.1 Connection to Phase 4

Phase 4 broadens noisy VQE/VQA surface, gradients, and optimizer studies. Those
results are stronger if they build on a backend whose execution-cost structure
and correctness properties have already been studied in Phase 3.

### 10.2 Connection to the Channel-Native Follow-On Branch

If the Phase 3 benchmark package shows that the native baseline still leaves the
dominant bottleneck unresolved, the paper should explicitly justify a dedicated
follow-on branch for more invasive channel-native fusion rather than quietly
stretching the baseline claim.

### 10.3 Connection to Later Scaling Branches

Approximate scaling methods such as trajectories or MPDOs remain valuable only
after the exact backend has been extended and benchmarked strongly enough to
serve as a trustworthy comparison point.

## 11. Conclusion

Phase 3 is the natural methods follow-on to the exact noisy integration result
of Phase 2. The planned contribution is not merely to schedule noisy circuits
more cleverly, and not to broaden the application surface prematurely. It is to
make noisy mixed-state circuits native objects of partition planning and runtime
execution while preserving the exact semantics that define the density backend's
scientific value. If the planned validation and benchmark package closes
successfully, Paper 2 will provide the central methods and systems result in the
density-matrix publication ladder and the backend foundation needed for later
noisy optimizer and trainability studies.

## References

Suggested core references for the full-paper version:

1. Peter Rakyta and Zoltan Zimboras, *Approaching the theoretical limit in
   quantum gate decomposition*, `Quantum 6, 710 (2022)`.
2. Peter Rakyta, Gregory Morse, Jakab Nadori, Zita Majnay-Takacs, Oskar Mencer,
   and Zoltan Zimboras, *Highly optimized quantum circuits synthesized via
   data-flow engines*, `Journal of Computational Physics 500, 112756 (2024)`.
3. Joseph Clark, Travis S. Humble, and Himanshu Thapliyal, *TDAG: Tree-based
   Directed Acyclic Graph Partitioning for Quantum Circuits*, ACM GLSVLSI 2023.
4. Joseph Clark, Travis S. Humble, and Himanshu Thapliyal, *GTQCP:
   Greedy Topology-Aware Quantum Circuit Partitioning*, `arXiv:2410.02901`.
5. Xin-Chuan Wu, Marc Grau Davis, Frederic T. Chong, and Costin Iancu, *QGo:
   Scalable Quantum Circuit Optimization Using Automated Synthesis*,
   `arXiv:2012.09835`.
6. Longshan Xu, Edwin Hsing-Mean Sha, Yuhong Song, and Qingfeng Zhu, *QMin:
   Quantum Circuit Minimization via Gate Fusions for Efficient State Vector
   Simulation*, `Quantum Information Processing 25, 6 (2026)`.
7. Nguyen et al., *Gate Fusion Optimization for Quantum Simulation*, OSTI
   technical report.
8. Fang et al., *Efficient Hierarchical State Vector Simulation of Quantum
   Circuits via Acyclic Graph Partitioning*, IEEE CLUSTER 2022.
9. Felix Burt, Kuan-Cheng Chen, and Kin K. Leung, *A Multilevel Framework for
   Partitioning Quantum Circuits*, `Quantum 10, 1984 (2026)`.
10. Ang Li, Omer Subasi, Xiu Yang, and Sriram Krishnamoorthy, *Density Matrix
    Quantum Circuit Simulation via the BSP Machine on Modern GPU Clusters*,
    SC20.
11. Jun Doi and Hiroshi Horii, *Cache Blocking Technique to Large Scale Quantum
    Computing Simulation on Supercomputers*, IEEE QCE 2020.
12. Tyson Jones, Anna Brown, Ian Bush, and Simon C. Benjamin, *QuEST and High
    Performance Simulation of Quantum Computers*, `Scientific Reports 9, 10736
    (2019)`.
13. Yasunari Suzuki et al., *Qulacs: a fast and versatile quantum circuit
    simulator for research purpose*, `Quantum 5, 559 (2021)`.
14. Mingkuan Xu et al., *Atlas: Hierarchical Partitioning for Quantum Circuit
    Simulation on GPUs*, `arXiv:2408.09055`.
15. Damian S. Steiger, Thomas Haener, and Matthias Troyer, *ProjectQ: An Open
    Source Software Framework for Quantum Computing*, `Quantum 2, 49 (2018)`.
