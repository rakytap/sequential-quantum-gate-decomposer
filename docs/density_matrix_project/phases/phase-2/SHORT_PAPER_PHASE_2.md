# Exact Noisy Variational Quantum Circuit Emulation in SQUANDER

## Abstract

Variational quantum algorithms are a central paradigm for near-term quantum
applications, but their realistic study depends on accurate noisy simulation and
on software frameworks that expose that simulation inside practical training
workflows. SQUANDER already offers a high-performance state-vector workflow and
a standalone density-matrix module for exact mixed-state simulation. Phase 2
now delivers the integrated-backend slice needed for a research-grade noisy VQE
anchor workflow: explicit backend selection, exact Hermitian energy evaluation
via `Re Tr(H*rho)`, generated-`HEA` bridge support, ordered fixed local noise,
and structured unsupported-boundary handling. The scope remains intentionally
narrow: exact noisy backend integration and validation are in scope, while
density-aware partitioning, gate fusion, gradients, and approximate scaling are
deferred to later phases. The delivered evidence package includes mandatory
micro-validation, mandatory workflow-scale exact-regime coverage, bounded
optimization-trace evidence, and a machine-checkable publication manifest. This
positions Phase 2 as a complete integration milestone for the first major paper
in the density-matrix track and as the validated baseline for later
acceleration and trainability phases.

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
- a unified gate-and-noise execution layer,
- and reference validation against Qiskit Aer.

This is an important milestone, and the current implementation now closes the
core integration slice needed for one canonical exact noisy variational
workflow. The delivered result provides explicit backend selection, a generated
hardware-efficient-ansatz bridge with ordered fixed local noise, exact
real-valued Hermitian energy evaluation, a mandatory 1 to 3 qubit
micro-validation matrix, required end-to-end 4- and 6-qubit workflow cases, a
mandatory 4/6/8/10-qubit workflow-scale exact-regime matrix with 10 fixed
parameter vectors per required size, one bounded 4-qubit optimization trace
driven by a deterministic parameter policy, and explicit hard-error unsupported
behavior. As a result, the project now has a delivered research-grade exact
noisy backend slice for the frozen canonical workflow scope.

That slice is now backed by a complete workflow-facing publication package
archived in `benchmarks/density_matrix/artifacts/phase2_task6/`, with
`task6_story6_publication_bundle.json` as the top-level manifest. The final
Task 6 manifest links the canonical workflow contract
`story1_canonical_workflow_contract.json`, the end-to-end plus trace bundle
`story2_end_to_end_trace_bundle.json`, the workflow matrix bundle
`story3_matrix_baseline_bundle.json`, the unsupported-workflow bundle
`story4_unsupported_workflow_bundle.json`, and the interpretation-guardrail
bundle `story5_interpretation_bundle.json` in one machine-readable evidence
surface while preserving traceability to the underlying Task 5 validation
layers. The publication bundle now checks semantic closure across those layers,
including contract completeness, contract-aligned workflow gates, unsupported-
boundary integrity, and interpretation guardrails, rather than only artifact
presence and shared workflow identity.

This short paper documents the first major integration step that closes that
gap. Its purpose is to present the density-matrix path as a usable backend for
one exact noisy variational workflow while keeping the scientific claim narrow,
defensible, and clearly separated from later acceleration work.

## 2. Problem Statement

The core problem addressed in Phase 2 is not the lack of a mixed-state
representation. That already exists. The problem was integrating the exact
mixed-state backend into the research workflow that the broader project needs.

More concretely, the deliberately bounded scientific scope has four defining
features:

1. The delivered density path is generated-hardware-efficient-ansatz-only rather
   than a broad circuit-source bridge.
2. The observable path is intentionally limited to exact Hermitian energy
   evaluation rather than a broad measurement framework.
3. The executed evidence now covers the mandatory micro-validation matrix, the
   mandatory 4/6/8/10 workflow-scale benchmark floor, and one bounded 4-qubit
   optimization trace.
4. The publication-ready reproducibility package is complete for the current
   integrated-backend slice, while broader later-phase application and acceleration
   studies remain outside this paper's claim.

This means that Phase 2 is no longer blocked by workflow integration, local
correctness evidence, workflow-scale exactness evidence, or reproducibility of
the integrated backend slice. The remaining boundaries are deliberate scientific
scope boundaries rather than missing foundations.

## 3. Current Baseline and Research Gap

The current density-matrix baseline is already meaningful. It includes:

- exact dense mixed-state representation,
- a unified gate-plus-noise execution path,
- baseline noise channels such as depolarizing, amplitude damping, and phase
  damping,
- Python-facing workflow access,
- and test and benchmark scaffolding.

This is enough to support standalone exact noisy simulations and reference
comparisons. However, the existing architecture documents already identify the
next integration targets:

- a density-matrix backend path inside the variational workflow,
- support for exact real-valued Hermitian energies via `Re Tr(H*rho)`,
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

The delivered Phase 2 contribution is an exact noisy backend integration for
SQUANDER's variational workflows. The contribution has five parts.

### 4.1 Backend Selection

Phase 2 defines the density-matrix execution path as a selectable backend rather
than an isolated module. This gives the project a documented contract for when
mixed-state evaluation is requested and what that means for downstream workflow
behavior.

### 4.2 Exact Noisy Observable Evaluation

Phase 2 establishes the expectation-value path needed for noisy training. The
core requirement is exact real-valued Hermitian-energy evaluation through
`Re Tr(H*rho)`. This is the minimum scientific contract needed for the
canonical exact noisy VQE study delivered in Phase 2.

### 4.3 Circuit-to-Backend Bridging

Phase 2 defines how the generated default `HEA` circuit reaches the
density-matrix backend in a documented and reproducible way. This includes the
supported canonical workflow path, unsupported cases, and explicit scope
boundaries rather than broad circuit-source parity.

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

## 6. Validation and Benchmark Baseline (Delivered)

Because Phase 2 is exact-first, validation is central to the contribution.

### 6.1 Primary Reference

Qiskit Aer density-matrix simulation is the primary external baseline for noisy
observable and workflow validation.

### 6.2 Workload Types

The delivered benchmark and validation suite includes:

- mandatory 1 to 3 qubit micro-validation cases for the required local-noise
  baseline,
- mandatory 4 / 6 / 8 / 10 qubit anchor-workflow fixed-parameter matrix with 10
  parameter vectors per required size,
- one bounded 4-qubit optimization trace for workflow-level completion evidence.

### 6.3 Noise Types

The delivered baseline emphasizes realistic local noise classes:

- local depolarizing noise,
- local phase damping or dephasing,
- local amplitude damping,
- whole-register depolarizing retained only as optional classification evidence,
- deferred and unsupported families captured explicitly as negative evidence.

### 6.4 Metrics

Phase 2 evidence includes:

- agreement of observables with the reference backend under frozen thresholds,
- exact noisy workflow completion and bridge-support checks,
- runtime and peak-memory characterization on mandatory workflow cases,
- reproducibility of configuration, provenance, and artifact status,
- and the rule that only mandatory, complete, supported evidence closes the
  main Phase 2 claim while optional evidence remains supplemental.

### 6.5 Acceptance Point

The practical exact regime for Phase 2 is anchored at 10 qubits, and the
delivered mandatory evidence package includes documented 10-qubit anchor cases.
This remains a workflow-validation anchor, not a universal claim about the
backend's ultimate limit.

## 7. Scientific Contribution

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

## 8. Current Limitations

The current limitations of the Phase 2 contribution are:

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
