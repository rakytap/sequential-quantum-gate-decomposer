# Curated References

This document collects the main literature relevant to the density-matrix,
partitioning, fusion, and noisy-training agenda discussed in this project.

The list is curated for usefulness, not completeness. Each entry includes a short
note on why it matters for this PhD direction.

Tag legend:

- `[Current code]`: directly reflected in the existing SQUANDER codebase.
- `[Near-term]`: strongly relevant to the current manuscript and follow-on
  implementation path.
- `[Future branch]`: useful for later scaling or optional architecture branches.
- `[Trainability]`: central to the noise-aware optimization and barren-plateau
  agenda.

## 1. SQUANDER Foundations

- `[Current code]` Peter Rakyta and Zoltan Zimboras, *Approaching the
  theoretical limit in quantum gate decomposition*, `Quantum 6, 710 (2022)`.
  Relevance: core SQUANDER decomposition background and the main foundation for
  why the project already has strong compilation / synthesis infrastructure.

- `[Current code]` Peter Rakyta and Zoltan Zimboras, *Efficient quantum gate
  decomposition via adaptive circuit compression*, `arXiv:2203.04426`.
  Relevance: useful context for SQUANDER's wider decomposition design and its
  optimizer-first philosophy.

- `[Current code]` Peter Rakyta, Gregory Morse, Jakab Nadori, Zita
  Majnay-Takacs, Oskar Mencer, and Zoltan Zimboras, *Highly optimized quantum
  circuits synthesized via data-flow engines*, `Journal of Computational Physics
  500, 112756 (2024)`.
  Relevance: important SQUANDER performance and hardware-acceleration context,
  especially for heterogeneous execution ambitions in the PhD plan.

- `[Trainability]` Jakab Nadori, Gregory Morse, Barna Fulop Villam, Zita
  Majnay-Takacs, Zoltan Zimboras, and Peter Rakyta, *Batched Line Search
  Strategy for Navigating through Barren Plateaus in Quantum Circuit Training*,
  `Quantum 9, 1841 (2025)`.
  Relevance: directly relevant to the optimizer track and to Phase 4 optimizer
  comparisons under noise.

## 2. Quantum Circuit Partitioning And Gate Fusion

- `[Current code]` Joseph Clark, Travis S. Humble, and Himanshu Thapliyal,
  *TDAG: Tree-based Directed Acyclic Graph Partitioning for Quantum Circuits*,
  `ACM GLSVLSI 2023`, DOI `10.1145/3583781.3590234`.
  Relevance: directly aligned with `squander/partitioning/tdag.py`.

- `[Current code]` Joseph Clark, Travis S. Humble, and Himanshu Thapliyal,
  *GTQCP: Greedy Topology-Aware Quantum Circuit Partitioning*, `arXiv:2410.02901`,
  conference DOI `10.1109/QCE57702.2023.00089`.
  Relevance: directly aligned with the `gtqcp` variant exposed by the current
  partitioning API.

- `[Current code]` Xin-Chuan Wu, Marc Grau Davis, Frederic T. Chong, and Costin
  Iancu, *QGo: Scalable Quantum Circuit Optimization Using Automated Synthesis*,
  `arXiv:2012.09835`.
  Relevance: the best conceptual match for SQUANDER's wide-circuit
  optimization-by-partitioning workflow.

- `[Near-term]` Longshan Xu, Edwin Hsing-Mean Sha, Yuhong Song, and Qingfeng
  Zhuge, *QMin: Quantum Circuit Minimization via Gate Fusions for Efficient
  State Vector Simulation*, `Quantum Information Processing 25, 6 (2026)`,
  DOI `10.1007/s11128-025-05028-6`.
  Relevance: especially important for control-aware fusion ideas and for
  interpreting the existing `ilp-fusion-ca` strategy.

- `[Near-term]` Fang et al., *Efficient Hierarchical State Vector Simulation of
  Quantum Circuits via Acyclic Graph Partitioning*, `IEEE CLUSTER 2022`,
  DOI `10.1109/CLUSTER51413.2022.00041`.
  Relevance: useful background for hierarchical partitioning, stage formation,
  and simulation-oriented graph cuts.

- `[Near-term]` Felix Burt, Kuan-Cheng Chen, and Kin K. Leung, *A Multilevel
  Framework for Partitioning Quantum Circuits*, `Quantum 10, 1984 (2026)`,
  DOI `10.22331/q-2026-01-22-1984`.
  Relevance: modern multilevel partitioning framework; highly relevant if
  single-level partitioning becomes the bottleneck later.

- `[Near-term]` Nguyen et al., *Gate Fusion Optimization for Quantum
  Simulation*, `OSTI Technical Report`, available at
  <https://www.osti.gov/servlets/purl/1985363>.
  Relevance: strongly aligned with the gate-fusion viewpoint in the current
  planner/runtime split.

## 3. Graph Algorithms Used In The Partitioning Stack

- `[Current code]` Esko Nuutila and Eljas Soisalon-Soininen, *On Finding the
  Strongly Connected Components in a Directed Graph*, `Information Processing
  Letters (1994)`, DOI `10.1016/0020-0190(94)00128-K`.
  Relevance: directly relevant to the SCC and reachability machinery used in
  `ilp.py`.

- `[Current code]` Paul Balister et al., *Algorithms for Generating Convex Sets
  in Acyclic Digraphs*, `Journal of Discrete Algorithms`, DOI
  `10.1016/j.jda.2008.07.008`.
  Relevance: directly relevant to feasible partition enumeration in the ILP path.

## 4. Density-Matrix And Open-System Simulation On Classical Hardware

- `[Near-term]` Ang Li, Omer Subasi, Xiu Yang, and Sriram Krishnamoorthy,
  *Density Matrix Quantum Circuit Simulation via the BSP Machine on Modern GPU
  Clusters*, `SC20`, DOI `10.1109/SC41405.2020.00017`.
  Relevance: one of the most important density-matrix HPC references; useful for
  communication-aware scaling and for understanding the real cost of exact
  mixed-state simulation.

- `[Near-term]` Jun Doi and Hiroshi Horii, *Cache Blocking Technique to Large
  Scale Quantum Computing Simulation on Supercomputers*, `IEEE QCE 2020`,
  DOI `10.1109/QCE49297.2020.00035`, `arXiv:2102.02957`.
  Relevance: important for memory-system thinking, cache blocking, and kernel
  locality; useful for density-matrix fusion cost reasoning even though it is not
  specific to density matrices.

- `[Near-term]` Tyson Jones, Anna Brown, Ian Bush, and Simon C. Benjamin,
  *QuEST and High Performance Simulation of Quantum Computers*, `Scientific
  Reports 9, 10736 (2019)`, DOI `10.1038/s41598-019-47174-9`.
  Relevance: strong benchmark and reference simulator for both state-vector and
  density-matrix simulation.

- `[Near-term]` Yasunari Suzuki et al., *Qulacs: a fast and versatile quantum
  circuit simulator for research purpose*, `Quantum 5, 559 (2021)`, DOI
  `10.22331/q-2021-10-06-559`.
  Relevance: practical reference simulator with noisy and parametric support;
  useful as a comparison point for performance-oriented papers.

- `[Future branch]` J. R. Johansson, P. D. Nation, and Franco Nori, *QuTiP: An
  open-source Python framework for the dynamics of open quantum systems*,
  `Computer Physics Communications 183, 1760 (2012)`, DOI
  `10.1016/j.cpc.2012.02.021`.
  Relevance: foundational open-system and superoperator reference; especially
  relevant if the project later explores channel-native fusion or Liouville-space
  formulations.

- `[Future branch]` Song Cheng et al., *Simulating Noisy Quantum Circuits with
  Matrix Product Density Operators*, `Physical Review Research 3, 023005 (2021)`,
  DOI `10.1103/PhysRevResearch.3.023005`.
  Relevance: important reference for MPDO-style approximate scaling methods; a
  likely comparison point if exact density matrices hit their practical limit.

- `[Future branch]` Lukas Burgholzer, Hartwig Bauer, and Robert Wille, *Hybrid
  Schroedinger-Feynman Simulation of Quantum Circuits With Decision Diagrams*,
  `IEEE QCE 2021`, DOI `10.1109/QCE52317.2021.00037`, `arXiv:2105.07045`.
  Relevance: useful for hybrid and compressed simulation alternatives once the
  exact backend is mature.

- `[Near-term]` Mingkuan Xu et al., *Atlas: Hierarchical Partitioning for Quantum
  Circuit Simulation on GPUs*, `arXiv:2408.09055`.
  Relevance: modern hierarchical simulation reference; useful for stage/kernel
  separation and hardware-aware planning ideas.

## 5. Quantum Software Framework References

- `[Current code]` Damian S. Steiger, Thomas Haener, and Matthias Troyer,
  *ProjectQ: An Open Source Software Framework for Quantum Computing*, `Quantum
  2, 49 (2018)`.
  Relevance: broad reference point for simulator/compiler design and a useful
  software comparison in review discussions.

## 6. Noisy VQA, Trainability, And Barren Plateaus

### 6.1 SQUANDER Core Papers (VQE Built On This Infrastructure)

- `[Current code]` Peter Rakyta and Zoltan Zimboras, *Approaching the theoretical
  limit in quantum gate decomposition*, `Quantum 6, 710 (2022)`.
  The foundational SQUANDER paper. VQE inherits the circuit representation,
  gate kernels, and optimization infrastructure described here.

- `[Current code]` Peter Rakyta and Zoltan Zimboras, *Efficient quantum gate
  decomposition via adaptive circuit compression*, `arXiv:2203.04426`.
  Describes the adaptive decomposition algorithm. VQE shares the
  Optimization_Interface base class with N_Qubit_Decomposition_adaptive.

- `[Current code]` Peter Rakyta, Gregory Morse, Jakab Nadori, Zita Majnay-Takacs,
  Oskar Mencer, and Zoltan Zimboras, *Highly optimized quantum circuits
  synthesized via data-flow engines*, `Journal of Computational Physics 500,
  112756 (2024)`.
  Covers hardware acceleration (DFE/Groq). VQE contains `#ifdef __GROQ__`
  conditional paths for accelerated state-vector simulation
  (optimization_problem_Groq).

- `[Trainability]` Jakab Nadori, Gregory Morse, Barna Fulop Villam, Zita
  Majnay-Takacs, Zoltan Zimboras, and Peter Rakyta, *Batched Line Search
  Strategy for Navigating through Barren Plateaus in Quantum Circuit Training*,
  `Quantum 9, 1841 (2025)`.
  Directly describes the AGENTS and COSINE optimizers that VQE uses. The batched
  line-search technique is the core of the barren-plateau-resilient optimization.

### 6.2 Cost Function and Error Correction

- `[Current code]` arXiv:2210.09191 — Referenced in Optimization_Interface.h for
  bitflip error corrections in the cost function (Eq. 21). The
  correction1_scale and correction2_scale parameters in Optimization_Interface
  implement this paper's error model.

### 6.3 VQE/VQA Theory and Trainability

- `[Trainability]` Daochen Wang, Oscar Higgott, and Stephen Brierley,
  *Accelerated Variational Quantum Eigensolver*, `Physical Review Letters 122,
  140504 (2019)`, DOI `10.1103/PhysRevLett.122.140504`.
  Accelerated VQE workflow reference, relevant to SQUANDER's optimizer/runtime
  trade-offs.

- `[Trainability]` Jarrod R. McClean et al., *Barren Plateaus in Quantum Neural
  Network Training*, `Nature Communications 9, 4812 (2018)`, DOI
  `10.1038/s41467-018-07090-4`.
  Foundational barren-plateau reference; motivates SQUANDER's gradient-free
  AGENTS optimizer for VQE.

- `[Trainability]` M. Cerezo et al., *Cost Function Dependent Barren Plateaus in
  Shallow Parametrized Quantum Circuits*, `Nature Communications 12, 1791
  (2021)`, DOI `10.1038/s41467-021-21728-w`.
  Explains why locality-preserving cost functions matter; the VQE cost function
  (local Hamiltonian expectation) is specifically designed to be trainable.

- `[Trainability]` Giacomo De Palma, Milad Marvian, Cambyse Rouze, and Daniel
  Stilck Franca, *Limitations of Variational Quantum Algorithms: A Quantum
  Optimal Transport Approach*, `PRX Quantum 4, 010309 (2023)`, DOI
  `10.1103/PRXQuantum.4.010309`.
  Theoretical limitations of noisy VQAs; motivates the density-matrix VQE
  backend for studying noise effects.

- `[Trainability]` Stefan H. Sack, Raimel A. Medina, Alexios A. Michailidis,
  Richard Kueng, and Maksym Serbyn, *Avoiding Barren Plateaus Using Classical
  Shadows*, `PRX Quantum 3, 020365 (2022)`, DOI `10.1103/PRXQuantum.3.020365`.
  Mitigation reference; relevant to Renyi entropy monitoring in VQE.

### 6.4 Random Unitary and Ansatz Construction

- `[Current code]` arXiv:1303.5904 — Parametrization of random unitary matrices.
  Referenced in Random_Unitary.cpp for generating initial states and test
  unitaries.

### 6.5 Density-Matrix Simulation (for Noisy VQE)

- `[Near-term]` Ang Li, Omer Subasi, Xiu Yang, and Sriram Krishnamoorthy,
  *Density Matrix Quantum Circuit Simulation via the BSP Machine on Modern GPU
  Clusters*, `SC20`, DOI `10.1109/SC41405.2020.00017`.
  Reference simulator for the density-matrix backend's performance and
  correctness benchmarking.

- `[Near-term]` Tyson Jones, Anna Brown, Ian Bush, and Simon C. Benjamin,
  *QuEST and High Performance Simulation of Quantum Computers*, `Scientific
  Reports 9, 10736 (2019)`, DOI `10.1038/s41598-019-47174-9`.
  Reference simulator for the density-matrix backend's performance and
  correctness benchmarking.

- `[Near-term]` Yasunari Suzuki et al., *Qulacs: a fast and versatile quantum
  circuit simulator*, `Quantum 5, 559 (2021)`, DOI `10.22331/q-2021-10-06-559`.
  Reference simulator for the density-matrix backend's performance and
  correctness benchmarking.

### 6.6 Bayesian Optimization

- `[Current code]` arXiv:1807.02811 — Referenced in Bayes_Opt.h for the Bayesian
  optimization algorithm used as one of VQE's supported optimizers.

## 7. Phase-Specific Shortlists

### 7.1 Phase 3 Methods / Systems Paper

Use this document as the only planning source of truth for the Phase 3 paper
literature.

Core citation set for the Phase 3 paper:

- `Section 1` for SQUANDER platform lineage and internal technical grounding.
- `Section 2` for partitioning and gate-fusion prior art.
- `Section 4` for exact density-matrix, HPC, and memory-locality context.
- `Section 5` for software-framework positioning when comparisons to other
  platforms are needed.

Priority entries for Paper 2:

- `Approaching the theoretical limit in quantum gate decomposition` for the
  core SQUANDER platform lineage.
- `TDAG`, `GTQCP`, and `QGo` for the strongest partition-planning lineage.
- `QMin` and Nguyen et al. for the gate-fusion side of the story.
- Fang et al. and Burt et al. for hierarchical or multilevel partitioning
  context.
- Li et al. and Doi & Horii for exact mixed-state/HPC and memory-locality
  motivation.
- `QuEST` and `Qulacs` for practical external simulator baselines.
- `Atlas` as optional modern hierarchical GPU context when hardware-aware
  planning becomes relevant.
- `QuTiP` only when discussing channel-native, superoperator, or Liouville-space
  follow-on branches rather than the minimum Phase 3 claim.

## 8. Practical Reading Order

For the current manuscript and follow-on planning path after delivered Phase 3
work, the most useful reading order is:

1. SQUANDER density-matrix docs already in this repository,
2. `TDAG`, `GTQCP`, and `QGo`,
3. `QMin` and Nguyen et al. on gate fusion,
4. `DM-Sim`, `Atlas`, `Doi & Horii`, `QuEST`, and `Qulacs`,
5. `Cerezo`, `McClean`, `Sack`, and `De Palma`,
6. `Cheng et al.` and other approximate scaling references once exact density
   matrices become the limiting factor.

## 9. How To Use This List

- Use the partitioning/fusion papers to justify the planner/runtime split and the
  density-aware performance work.
- Use the density-matrix and HPC papers to motivate why exact mixed-state
  acceleration is scientifically non-trivial.
- Use the trainability papers to keep the project focused on the PhD theme rather
  than drifting into simulator engineering alone.
- Use the software references when framing SQUANDER relative to other platforms.
