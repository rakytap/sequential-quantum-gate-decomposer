# Exact Noisy Variational Quantum Circuit Emulation in SQUANDER

## Abstract

Variational quantum algorithms are a central paradigm for near-term quantum
applications, but their realistic study depends on accurate noisy simulation and
on software frameworks that expose that simulation inside practical training
workflows. SQUANDER already offers a high-performance state-vector workflow and
a standalone density-matrix module for exact mixed-state simulation, yet the
current mixed-state backend is not integrated into the framework's main
variational pipeline. This Phase 2 paper defines the first research-grade
integration step for that gap. The proposed contribution is an exact noisy
density-matrix backend path for SQUANDER that supports backend selection,
observable evaluation via `Tr(H*rho)`, and a bridge from existing circuit/gate
representations into mixed-state execution. The scope is intentionally narrow:
the work targets exact noisy backend integration and validation, while
density-aware partitioning, gate fusion, gradients, and approximate scaling are
deferred to later phases. The paper outlines the validation methodology,
benchmark matrix, and acceptance criteria needed to establish the backend as a
usable scientific instrument for noisy variational studies. This integration is
expected to enable reproducible exact noisy workflows, support the first major
publication in the density-matrix track, and provide the foundation for later
acceleration and trainability research.

## 1. Introduction and Motivation

Variational quantum algorithms (VQAs) remain among the most important candidates
for extracting useful behavior from noisy intermediate-scale quantum devices.
They are flexible enough to cover applications in chemistry, optimization,
machine learning, and state preparation, but they are also unusually sensitive
to the details of circuit architecture, optimizer choice, and noise model.
Because of this, classical emulation remains indispensable: it is needed not
only to test algorithmic ideas, but also to understand how realistic noise
changes trainability, expressivity, and optimizer behavior.

SQUANDER is already strong on the state-vector side. It contains:

- a high-performance state-vector execution path,
- strong decomposition and optimization machinery,
- and a mature circuit partitioning and gate-fusion subsystem.

At the same time, a density-matrix module has recently been added, providing:

- exact mixed-state representation,
- unified gate and noise execution through `NoisyCircuit`,
- and reference validation against Qiskit Aer.

This is an important milestone, but it is not yet sufficient for the main PhD
research agenda. The current density-matrix path is a standalone capability. It
is not yet integrated into the main variational workflow, and the code does not
yet implement the now-documented Phase 2 backend-selection contract or the
observable-evaluation path needed for noisy training studies. As a result, the
project currently has a capable exact noisy simulator and a frozen Phase 2
contract, but not yet a delivered noisy variational training backend.

This short paper defines the first major integration step needed to close that
gap. Its purpose is to turn the density-matrix path into a usable backend for
exact noisy variational workflows while keeping the scientific claim narrow,
defensible, and clearly separated from later acceleration work.

## 2. Problem Statement

The core problem addressed in Phase 2 is not the lack of a mixed-state
representation. That already exists. The problem is that the existing exact
mixed-state backend is not yet part of the research workflow that the broader
project actually needs.

More concretely, the current implementation gap has four parts:

1. The documented backend-selection path between state-vector and
   density-matrix execution is not yet implemented in the main variational
   workflow.
2. The documented expectation-value path based on `Tr(H*rho)` is not yet
   delivered and validated in running code for noisy variational objectives.
3. The documented bridge from the current circuit and gate representations into
   the density-matrix backend is not yet implemented at the workflow level.
4. The frozen support surface, benchmark gate, and publication evidence package
   are not yet delivered in executable form.

This means that even though exact noisy evolution is available in isolation, the
project cannot yet claim to support research-grade exact noisy VQA workflows.
That missing integration layer is the real Phase 2 problem.

## 3. Current Baseline and Research Gap

The current density-matrix baseline is already meaningful. It includes:

- `DensityMatrix` as a mixed-state container,
- `NoisyCircuit` as a unified gate-plus-noise execution path,
- baseline noise channels such as depolarizing, amplitude damping, and phase
  damping,
- Python bindings,
- and test and benchmark scaffolding.

This is enough to support standalone exact noisy simulations and reference
comparisons. However, the existing architecture documents already identify the
next integration targets:

- a density-matrix backend path inside the variational workflow,
- support for expectation values of the form `Tr(H*rho)`,
- and routing through the optimization interface.

From a scientific perspective, the gap is equally clear. The central PhD
research theme is scalable training under realistic noise models. Without a
usable exact noisy backend in the training workflow, the project cannot yet
produce the later optimizer and trainability results in a clean, reproducible
way.

It is also important to state what this gap is not. It is not yet a partitioning
or fusion problem. The existing planning set already separates those concerns
into a later methods phase. Phase 2 is first about making exact noisy training
possible and trustworthy.

## 4. Phase 2 Contribution and Scope

The proposed Phase 2 contribution is an exact noisy backend integration for
SQUANDER's variational workflows. The contribution has five parts.

### 4.1 Backend Selection

Phase 2 defines the density-matrix execution path as a selectable backend rather
than an isolated module. This gives the project a documented contract for when
mixed-state evaluation is requested and what that means for downstream workflow
behavior.

### 4.2 Exact Noisy Observable Evaluation

Phase 2 establishes the expectation-value path needed for noisy training. The
core requirement is support for evaluation of observables through `Tr(H*rho)`.
This is the minimum scientific contract needed for exact noisy VQA studies.

### 4.3 Circuit-to-Backend Bridging

Phase 2 defines how the existing circuit and gate structures reach the
density-matrix backend in a documented and reproducible way. This includes
supported cases, unsupported cases, and explicit scope boundaries.

### 4.4 Workload-Driven Noise Support

Phase 2 prioritizes local, physically meaningful noise models needed for the
first noisy training studies. Global whole-register toy noise may still be used
as a baseline or regression check, but it is not treated as the main scientific
workload.

### 4.5 Publication-Oriented Validation

The integration is not considered complete unless it is supported by a benchmark
and validation package strong enough for the first major paper in the
density-matrix track.

## 5. Scope Boundaries and Non-Goals

The Phase 2 claim must remain narrow in order to stay defensible.

In scope:

- exact noisy backend integration,
- observable evaluation,
- workflow bridging,
- training-relevant noise support,
- validation and reproducibility.

Out of scope:

- density-aware partitioning,
- gate fusion for density matrices,
- channel-native fusion,
- gradients for the density backend,
- approximate scaling methods,
- and full gate parity with the entire state-vector circuit surface.

These exclusions are not weaknesses in the plan. They are deliberate scope
controls that protect the scientific clarity of the phase and preserve a clean
handoff to the later acceleration phase.

## 6. Validation and Benchmark Plan

Because Phase 2 is exact-first, validation is central to the contribution.

### 6.1 Primary Reference

Qiskit Aer density-matrix simulation is the primary external baseline for noisy
observable and workflow validation.

### 6.2 Workload Types

The benchmark and validation suite should include:

- small-to-medium exact noisy circuits,
- at least one variational workflow relevant to the later PhD studies,
- representative circuits whose gate and noise content match the Phase 2 support
  surface.

### 6.3 Noise Types

The baseline evaluation should emphasize realistic local noise classes:

- local depolarizing noise,
- local phase damping or dephasing,
- local amplitude damping,
- and selected additional local models if required by the target workflow.

### 6.4 Metrics

Phase 2 evidence should include:

- agreement of observables with the reference backend,
- exact noisy workflow stability,
- runtime and memory characterization,
- and reproducibility of configuration and results.

### 6.5 Acceptance Point

The current planning and roadmap documents point to a practical exact regime on
the order of roughly 10 qubits for the Phase 2 milestone. This should be treated
as the anchor for workflow validation rather than as a universal claim about the
backend's ultimate limit.

## 7. Expected Scientific Contribution

The scientific contribution of Phase 2 is not raw algorithmic novelty at the
same level as a new fusion or partitioning method. Its value is different and
still substantial:

- it turns exact mixed-state simulation into a practical workflow capability
  within SQUANDER,
- it creates a trustworthy backend for future noisy optimizer and trainability
  studies,
- and it establishes the first complete publication in the density-matrix track
  of the project.

This is exactly the kind of contribution that can enable stronger later science.
Without it, later claims about optimizer behavior or trainability under noise
would either be impossible or much harder to defend.

## 8. Expected Limitations

The expected limitations of the Phase 2 contribution should be stated openly.

First, the phase is not intended to maximize performance. It aims to make exact
noisy workflows usable and validated, not to solve density-aware acceleration.

Second, the gate support surface will remain workload-driven. This is acceptable
for the phase, but it means that some workflows or ansatz classes may remain out
of scope.

Third, the exact operating regime is limited by the cost of dense density
matrices. This is not a defect of the plan; it is the reason later phases exist.

Finally, Phase 2 does not yet address density-matrix gradients, partitioning,
fusion, or approximate scaling. Those remain later opportunities, not current
promises.

## 9. Follow-On Phases

The Phase 2 handoff is straightforward:

- Phase 3 takes the integrated exact backend and asks how to accelerate it
  through density-aware partitioning and fusion.
- Phase 4 uses the integrated and accelerated backend to study noisy optimizer
  behavior in a more systematic way.
- Phase 5 uses the resulting workflow to study trainability, entropy growth, and
  barren-plateau behavior under realistic noise.

This sequence is important. It keeps the scientific narrative coherent:

1. integrate the exact noisy backend,
2. accelerate it,
3. use it to answer training questions.

## 10. Conclusion

Phase 2 is the first point where the density-matrix effort becomes central to
the PhD research narrative. The contribution is to establish an exact noisy
backend integration for SQUANDER's variational workflows, grounded in
publication-grade validation and realistic local noise models. By keeping the
scope focused on backend selection, observable evaluation, workflow bridging,
and reproducibility, the phase delivers a clear and defensible scientific result
while preparing the project for the more performance- and trainability-oriented
phases that follow.

## References

Selected references most directly relevant to the Phase 2 contribution:

- Joseph Clark, Travis S. Humble, and Himanshu Thapliyal, *TDAG: Tree-based
  Directed Acyclic Graph Partitioning for Quantum Circuits*, ACM GLSVLSI 2023.
- Xin-Chuan Wu, Marc Grau Davis, Frederic T. Chong, and Costin Iancu, *QGo:
  Scalable Quantum Circuit Optimization Using Automated Synthesis*,
  `arXiv:2012.09835`.
- Ang Li, Omer Subasi, Xiu Yang, and Sriram Krishnamoorthy, *Density Matrix
  Quantum Circuit Simulation via the BSP Machine on Modern GPU Clusters*, SC20.
- J. R. Johansson, P. D. Nation, and Franco Nori, *QuTiP: An open-source Python
  framework for the dynamics of open quantum systems*, Computer Physics
  Communications 2012.
- M. Cerezo et al., *Cost Function Dependent Barren Plateaus in Shallow
  Parametrized Quantum Circuits*, Nature Communications 2021.
- Giacomo De Palma, Milad Marvian, Cambyse Rouze, and Daniel Stilck Franca,
  *Limitations of Variational Quantum Algorithms: A Quantum Optimal Transport
  Approach*, PRX Quantum 2023.
