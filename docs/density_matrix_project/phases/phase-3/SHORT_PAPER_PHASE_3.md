# Noise-Aware Partitioning and Gate Fusion for Exact Mixed-State Quantum Circuit Simulation

## Draft Status

This document is a planning-facing short-paper draft for Phase 3. It should be
treated as a target publication surface derived from the Phase 3 contract, not
as an implementation-backed paper. Claims must be tightened to benchmark-backed
language once the mandatory evidence package exists.

## Abstract

Exact density-matrix simulation is the cleanest classical reference for studying
variational quantum circuits under realistic noise, but its cost quickly limits
the scale of exact experiments. SQUANDER already provides a validated exact
mixed-state backend for one canonical noisy workflow and a separate mature
state-vector partitioning and gate-fusion subsystem. Phase 3 targets the next
methods step: extending partitioning and limited fusion so noisy mixed-state
circuits become first-class planner inputs rather than unitary islands with
noise treated as external barriers. The planned contribution combines a
canonical noisy mixed-state planner surface, partition descriptors that preserve
explicit gate/noise ordering and parameter routing, an executable partitioned
density runtime with at least one real fused execution mode, and a
benchmark-calibrated density-aware planning objective. The evaluation surface is
anchored on the frozen Phase 2 noisy XXZ `HEA` workflow together with structured
noisy `U3` / `CNOT` partitioning families, validated against the sequential
`NoisyCircuit` baseline and Qiskit Aer. The intended Phase 3 claim is a bounded
methods result: exact noisy semantics remain central, fully channel-native fused
noisy blocks are deferred, and broader noisy VQE/VQA growth remains Phase 4
work.

## Publication Surface Role

This document is the compact short-paper surface for the Phase 3 Paper 2
package.

## Paper 2 Claim Boundary

Main claim:
SQUANDER extends partitioning and limited fusion to exact noisy mixed-state
circuits by making noisy operations first-class planner inputs, preserving exact
gate/noise semantics across partition descriptors and runtime execution, and
validating the resulting partitioned density path on representative noisy
workloads.

Explicit non-claims:
- fully channel-native fused noisy blocks are future work beyond the baseline
  Paper 2 claim
- density-backend gradients, optimizer-comparison studies, and broader noisy
  VQE/VQA workflow growth are Phase 4+ work
- approximate scaling methods remain later work beyond the current Paper 2 claim
- full `qgd_Circuit` parity is not part of the baseline Paper 2 claim
- a universal speedup claim across all noisy workloads is not required for Paper
  2; honest limitation reporting is part of the claim boundary

Supported-path boundary:
The guaranteed Paper 2 path is the canonical noisy mixed-state planner surface
plus the exact lowering needed for the frozen Phase 2 continuity workflow and
the required Phase 3 structured benchmark families.

No-fallback rule:
No silent sequential fallback is part of the Phase 3 contract for any benchmark
that claims partitioned density behavior.

Exact-regime boundary:
Mandatory internal correctness coverage spans 4, 6, 8, and 10 qubits, with
external micro-validation at 2 to 4 qubits and required performance recording
on representative 8- and 10-qubit structured families.

Evidence-closure rule:
Only mandatory, complete, supported correctness and reproducibility evidence,
plus either measurable benefit or benchmark-grounded limitation reporting,
closes the main Paper 2 claim.

Phase positioning:
Paper 2 is the Phase 3 methods and systems milestone in the density-matrix
publication ladder.

## 1. Introduction and Motivation

Exact noisy simulation is central to the broader PhD theme of scalable training
under realistic noise. Phase 2 already delivered a crucial integration step by
turning SQUANDER's density-matrix module into a usable backend for one canonical
noisy XXZ `HEA` workflow. That result established the exact noisy workflow
baseline, but it did not address the main remaining bottleneck: exact dense
mixed-state execution is still expensive, and the current partitioning/fusion
stack remains state-vector-oriented.

This creates a natural Phase 3 methods question. SQUANDER already has:

- a sequential exact noisy mixed-state path built around `NoisyCircuit`,
- and a mature state-vector partitioning and fusion subsystem.

What it does not yet have is a native way to partition and partially fuse noisy
mixed-state circuits while preserving the exact semantics that made the Phase 2
backend scientifically valuable in the first place.

The purpose of Phase 3 is therefore not to broaden noisy VQE/VQA surface or to
introduce approximate scaling. It is to make partitioning and limited fusion
native to exact noisy mixed-state workloads.

## 2. Problem Statement

The current gap is not the absence of exact noisy simulation. That gap was
closed for one canonical workflow in Phase 2. The current gap is that the
existing planner and fusion machinery still assumes a circuit model closer to
state-vector execution than to exact noisy mixed-state execution.

More concretely, the Phase 3 problem has four parts:

1. Noisy mixed-state circuits are not yet first-class planner inputs.
2. Partition descriptors do not yet preserve explicit gate/noise order and
   parameter-routing semantics as part of the contract.
3. The density backend does not yet have an executable partitioned runtime with
   real fused execution on representative noisy workloads.
4. The current cost model is state-vector-oriented and noise-blind.

This means that the main scientific question for Phase 3 is:

> can noisy mixed-state circuits be partitioned and partially fused in a way
> that remains semantically exact and yields useful performance behavior on
> representative noisy workloads?

## 3. Phase 3 Contribution and Scope

The planned Phase 3 contribution has four parts.

### 3.1 Canonical Noisy Planner Surface

Phase 3 defines a canonical noisy mixed-state planner surface equivalent to an
ordered `NoisyCircuit` operation stream. This makes gate operations and noise
operations first-class planner objects rather than external annotations.

### 3.2 Exact Semantic Preservation

Partition descriptors must preserve:

- explicit gate/noise order,
- qubit support,
- parameter-routing metadata,
- and any remapping information needed for exact execution.

Noise placement is part of the semantics, not boundary-only metadata.

### 3.3 Executable Partitioned Runtime Plus Real Fused Execution

The minimum publishable Phase 3 result is stronger than a planner-only
representation. The target runtime must:

- execute partitioned noisy mixed-state circuits end to end,
- include at least one real fused execution mode on eligible substructures,
- and preserve exact agreement with the sequential density baseline.

### 3.4 Density-Aware Planning Objective

Phase 3 should move beyond a state-vector FLOP model by using:

- structural noise-aware planning heuristics first,
- followed by a benchmark-calibrated density-aware objective or heuristic.

This keeps the methods claim grounded in measured behavior rather than in a
premature analytic optimality claim.

## 4. Validation and Benchmark Surface

Because the contribution is exact-first, validation is central to Paper 2.

### 4.1 Baselines

The planned validation baseline has two layers:

- sequential `NoisyCircuit` execution as the required internal exact reference,
- Qiskit Aer density-matrix simulation as the required external reference on the
  mandatory microcases and representative small continuity cases.

### 4.2 Workload Classes

The benchmark package is designed around both continuity and methods-oriented
coverage:

- the frozen Phase 2 noisy XXZ `HEA` workflow at 4, 6, 8, and 10 qubits,
- structured noisy `U3` / `CNOT` partitioning families at 8 and 10 qubits,
- and 2 to 4 qubit micro-validation cases stressing partition boundaries and the
  required noise families.

### 4.3 Noise Classes and Placements

The mandatory scientific noise surface remains:

- local depolarizing,
- local amplitude damping,
- local phase damping or dephasing.

Sensitivity should be recorded across:

- sparse local-noise placement,
- periodic layer-boundary placement,
- dense layer-wise placement.

### 4.4 Metrics

The final paper should report:

- density agreement with the sequential baseline,
- exactness against Qiskit Aer on the required external microcases,
- energy agreement on the continuity anchor,
- runtime,
- peak memory,
- planner runtime,
- partition count and qubit span,
- and fused-path coverage.

### 4.5 Success Interpretation

Paper 2 does not require a universal speedup claim across every noisy workload.
It requires:

- exact semantic preservation on the mandatory correctness matrix,
- and either measured benefit on representative cases or a benchmark-grounded
  diagnosis of why the native Phase 3 baseline still leaves a dominant
  bottleneck.

## 5. Scientific Contribution

The planned scientific value of Phase 3 is that it moves beyond two weaker
alternatives:

- a strong exact noisy backend with no acceleration story,
- or a partitioning story that still treats noise as an external exception.

If successful, Phase 3 produces a methods contribution in which noisy
mixed-state circuits become natural objects of partition planning and runtime
execution. That is stronger than a workflow-only speedup story and more aligned
with the broader PhD than a narrow kernel-only optimization result.

## 6. Scope Boundaries and Non-Goals

The final paper should stay bounded.

In scope:

- native noisy mixed-state planner inputs,
- exact semantic-preservation rules,
- executable partitioned runtime,
- limited real fused execution,
- benchmark-calibrated density-aware planning,
- and publication-grade correctness and performance evidence.

Out of scope for the baseline claim:

- fully channel-native fused noisy blocks,
- broader noisy VQE/VQA surface growth,
- density-backend gradients,
- approximate scaling methods,
- and full `qgd_Circuit` parity.

These exclusions are not weaknesses. They are scope controls that preserve a
clear methods claim and keep later phases scientifically distinct.

## 7. Expected Limitations

Even the successful baseline Paper 2 result will still have important
limitations:

- the exact dense regime remains fundamentally scale-limited,
- the support surface remains workload-driven rather than fully general,
- early density-aware planning may begin with structural heuristics before more
  refined calibration is complete,
- and fully channel-native fused noisy blocks remain future work rather than the
  baseline architecture.

These limitations should be explicit in the final paper because they are part of
what makes the claim honest and publication-ready.

## 8. Follow-On Phases

The Phase 3 handoff should remain clean:

- Phase 4 broadens noisy VQE/VQA workflow surface, gradients, and optimizer
  studies on top of the stabilized Phase 3 backend,
- and later follow-on branches can revisit channel-native fusion or approximate
  scaling only if the Phase 3 benchmark package shows that those directions are
  justified.

The sequence remains important:

1. integrate exact noisy workflows,
2. make partitioning and limited fusion native to those workflows,
3. broaden optimizer-facing workflow science,
4. then move into larger trainability studies and optional scaling branches.

## References

Selected references most directly relevant to the Phase 3 paper:

- Peter Rakyta and Zoltan Zimboras, *Approaching the theoretical limit in
  quantum gate decomposition*, `Quantum 6, 710 (2022)`.
- Peter Rakyta, Gregory Morse, Jakab Nadori, Zita Majnay-Takacs, Oskar Mencer,
  and Zoltan Zimboras, *Highly optimized quantum circuits synthesized via
  data-flow engines*, `Journal of Computational Physics 500, 112756 (2024)`.
- Joseph Clark, Travis S. Humble, and Himanshu Thapliyal, *TDAG: Tree-based
  Directed Acyclic Graph Partitioning for Quantum Circuits*, ACM GLSVLSI 2023.
- Joseph Clark, Travis S. Humble, and Himanshu Thapliyal, *GTQCP:
  Greedy Topology-Aware Quantum Circuit Partitioning*, `arXiv:2410.02901`.
- Xin-Chuan Wu, Marc Grau Davis, Frederic T. Chong, and Costin Iancu, *QGo:
  Scalable Quantum Circuit Optimization Using Automated Synthesis*,
  `arXiv:2012.09835`.
- Longshan Xu, Edwin Hsing-Mean Sha, Yuhong Song, and Qingfeng Zhu, *QMin:
  Quantum Circuit Minimization via Gate Fusions for Efficient State Vector
  Simulation*, `Quantum Information Processing 25, 6 (2026)`.
- Nguyen et al., *Gate Fusion Optimization for Quantum Simulation*, OSTI
  technical report.
- Ang Li, Omer Subasi, Xiu Yang, and Sriram Krishnamoorthy, *Density Matrix
  Quantum Circuit Simulation via the BSP Machine on Modern GPU Clusters*, SC20.
- Jun Doi and Hiroshi Horii, *Cache Blocking Technique to Large Scale Quantum
  Computing Simulation on Supercomputers*, IEEE QCE 2020.
- Tyson Jones, Anna Brown, Ian Bush, and Simon C. Benjamin, *QuEST and High
  Performance Simulation of Quantum Computers*, `Scientific Reports 9, 10736
  (2019)`.
- Yasunari Suzuki et al., *Qulacs: a fast and versatile quantum circuit
  simulator for research purpose*, `Quantum 5, 559 (2021)`.
