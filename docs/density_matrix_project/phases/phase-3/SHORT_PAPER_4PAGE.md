# Noise-Aware Partitioning and Limited Fusion for Exact Mixed-State Quantum Circuit Simulation

## Draft Status

This document is a planning-facing 4-page paper surface for the Phase 3 Paper 2
package. It is intended to guide venue shaping and implementation evidence
collection before the paper becomes submission-ready.

## Abstract

Exact density-matrix simulation is the strongest classical reference for noisy
variational quantum circuit research, but its cost quickly limits the scale of
exact experiments. SQUANDER already provides a validated exact noisy backend for
one canonical workflow and a separate mature state-vector partitioning and
gate-fusion subsystem. Phase 3 targets the next methods step: extending
partitioning and limited fusion so noisy mixed-state circuits become first-class
planner inputs rather than unitary islands separated by opaque noise
boundaries. The planned contribution combines a canonical noisy mixed-state
planner surface, partition descriptors that preserve explicit gate/noise order
and parameter routing, an executable partitioned density runtime with at least
one real fused execution mode on eligible substructures, and a benchmark package
anchored on both the frozen Phase 2 noisy XXZ `HEA` workflow and structured
noisy `U3` / `CNOT` circuit families. The validation story is exact-first:
partitioned execution must match the sequential `NoisyCircuit` baseline within
strict thresholds, while Qiskit Aer remains the required external reference on
the mandatory microcases. The baseline Paper 2 claim is deliberately bounded:
fully channel-native fused noisy blocks, broader noisy VQE/VQA growth, and
approximate scaling methods remain future work.

## Publication Surface Role

This document is the 4-page conference-paper surface for the Phase 3 Paper 2
package.

## Paper 2 Claim Boundary

Main claim:
SQUANDER extends partitioning and limited fusion to exact noisy mixed-state
circuits by making noisy operations first-class planner inputs, preserving exact
gate/noise semantics across partition descriptors and runtime execution, and
validating the resulting partitioned density path on representative noisy
workloads.

Explicit non-claims:
- fully channel-native fused noisy blocks are not part of the baseline Paper 2
  claim
- broader noisy VQE/VQA workflow growth and density-backend gradients remain
  Phase 4+ work
- approximate scaling methods remain later branches beyond the Paper 2 claim
- full direct `qgd_Circuit` parity is not required for baseline Paper 2 closure

Evidence-closure rule:
Only mandatory, complete, supported correctness and reproducibility evidence,
plus either measured benefit or benchmark-grounded limitation reporting, closes
the main Paper 2 claim.

## 1. Introduction

Phase 2 already established the exact noisy workflow baseline for SQUANDER by
integrating the density-matrix backend into one canonical noisy XXZ `HEA`
workflow. The next bottleneck is now architectural rather than integrational:
the project has both an exact noisy mixed-state backend and a mature
state-vector partitioning/fusion subsystem, but noisy mixed-state circuits are
not yet native objects of partition planning.

This matters because the broader PhD theme is scalable training under realistic
noise. If partitioning and fusion remain effectively unitary-first, later noisy
optimizer and trainability studies will be built on a backend whose main
execution-cost question is still unresolved.

The central Phase 3 question is therefore:

> can partitioning and limited fusion be extended so noisy mixed-state circuits
> remain semantically exact while becoming native planner and runtime objects?

## 2. Planned Method

The planned Phase 3 method has four components.

### 2.1 Canonical Noisy Mixed-State Planner Surface

The planner operates on a canonical noisy operation stream aligned with the
`NoisyCircuit` execution model. This makes gate operations and noise operations
first-class planner inputs.

### 2.2 Exact Semantic-Preservation Contract

Partition descriptors must preserve:

- explicit gate/noise order,
- qubit support,
- parameter-routing metadata,
- and any remapping information needed for exact execution.

Noise is not allowed to collapse into planner-external boundary metadata.

### 2.3 Executable Partitioned Runtime With Real Fused Execution

The baseline result is stronger than a planner representation. The runtime must:

- execute partitioned noisy mixed-state circuits end to end,
- include at least one real fused execution mode on eligible substructures,
- and avoid silent fallback to sequential execution on claimed partitioned runs.

### 2.4 Density-Aware Planning Calibration

The planner is expected to move beyond the existing state-vector FLOP model via:

- structural noise-aware heuristics first,
- followed by benchmark-calibrated density-aware planning.

This sequencing keeps the methods claim evidence-driven rather than purely
analytic.

## 3. Validation and Benchmark Plan

### 3.1 Baselines

- Internal exact baseline: sequential `NoisyCircuit` execution.
- External exact baseline: Qiskit Aer density-matrix simulation on the mandatory
  2 to 4 qubit microcases and representative small continuity cases.

### 3.2 Workload Matrix

The planned benchmark surface combines:

- the frozen Phase 2 noisy XXZ `HEA` continuity workflow at 4, 6, 8, and 10
  qubits,
- structured noisy `U3` / `CNOT` partitioning families at 8 and 10 qubits,
- mandatory micro-validation cases stressing partition boundaries and noise
  placement.

### 3.3 Noise Matrix

The required scientific noise surface remains:

- local depolarizing,
- local amplitude damping,
- local phase damping or dephasing,
- with sparse, periodic, and dense local-noise placement sensitivity.

### 3.4 Metrics and Thresholds

The planned paper must report:

- Frobenius-norm density difference against the sequential baseline,
- external exactness against Qiskit Aer where required,
- energy agreement on continuity-anchor cases,
- runtime,
- peak memory,
- planner runtime,
- partition count and qubit span,
- fused-path coverage.

The baseline thresholds remain strict:

- density agreement `<= 1e-10` on mandatory internal correctness cases,
- external microcase agreement `<= 1e-10`,
- continuity-anchor energy agreement `<= 1e-8`.

## 4. Expected Paper Claim

If the required evidence closes successfully, the final Paper 2 claim should be
that SQUANDER now supports:

- noisy mixed-state circuits as first-class partitioning inputs,
- exact semantic preservation across partitioned execution,
- an executable partitioned density runtime with at least one real fused path,
- and a benchmarked, scientifically interpretable backend for later noisy
  workflow science.

The paper does not need to claim fully channel-native fusion or universal
speedup. It does need to claim more than planner-only representation.

## 5. Limitations and Future Work

The final paper should state four baseline limitations explicitly:

- the exact dense regime remains fundamentally scale-limited,
- the support surface remains workload-driven,
- fully channel-native fused noisy blocks remain future work,
- and broader noisy VQE/VQA workflow growth remains outside Phase 3.

These limitations motivate the natural follow-on ladder:

- broader noisy workflow and optimizer studies in Phase 4,
- optional channel-native follow-on work if the Phase 3 benchmark package
  justifies it,
- and later approximate-scaling branches only after the exact backend remains a
  strong reference point.

## References

Suggested compact reference set for the 4-page surface:

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
- Ang Li, Omer Subasi, Xiu Yang, and Sriram Krishnamoorthy, *Density Matrix
  Quantum Circuit Simulation via the BSP Machine on Modern GPU Clusters*, SC20.
- Tyson Jones, Anna Brown, Ian Bush, and Simon C. Benjamin, *QuEST and High
  Performance Simulation of Quantum Computers*, `Scientific Reports 9, 10736
  (2019)`.
- Yasunari Suzuki et al., *Qulacs: a fast and versatile quantum circuit
  simulator for research purpose*, `Quantum 5, 559 (2021)`.
