# Paper 1 Draft for Phase 2

## Title Candidates

- `Exact Noisy Variational Quantum Circuit Emulation in SQUANDER`
- `Integrating Density-Matrix Simulation into a Large-Scale Quantum Training Framework`
- `A Research-Grade Exact Noisy Backend for Variational Quantum Workflows in SQUANDER`

## Abstract Summary

This document is the implementation-backed full-paper draft for Phase 2. A
compact conference abstract version is maintained separately in
`ABSTRACT_PHASE_2.md`.

In one sentence, the paper's claim is:

> SQUANDER's standalone exact density-matrix module can be elevated into a
> validated noisy variational backend by adding backend selection, exact
> real-valued Hermitian energy evaluation via `Re Tr(H*rho)`, and a documented
> generated-`HEA` bridge for one canonical noisy XXZ VQE workflow into
> mixed-state execution.

## 1. Introduction

Variational quantum algorithms (VQAs) remain one of the most active directions
for near-term quantum computing because they provide a flexible hybrid
quantum-classical framework for chemistry, optimization, and machine learning.
Yet realistic progress in this area depends critically on two capabilities that
are often treated separately: accurate noisy circuit emulation and practical
integration of that emulation into trainable workflows.

The need for mixed-state simulation is straightforward. Ideal state-vector
simulation is insufficient for studying physically relevant noise channels such
as depolarizing noise, dephasing, amplitude damping, readout errors, and later
calibration-informed noise models. At the same time, density-matrix simulation
is substantially more expensive than state-vector simulation, which makes it
essential to think carefully about where exact noisy simulation should be used,
what scale is realistic, and what scientific questions it should answer.

SQUANDER already provides a strong foundation for this work. It offers:

- a mature state-vector-oriented variational and decomposition framework,
- high-performance state-vector execution with partitioning and fusion support,
- and a recently introduced standalone density-matrix module based on exact
mixed-state representation, a unified noisy-circuit execution layer, and
explicit noise channels.

The density-matrix module is already useful in isolation, and the current
implementation now closes the core integration slice needed for one canonical
exact noisy variational workflow. The delivered result provides explicit backend
selection, a generated hardware-efficient-ansatz density path with ordered fixed
local noise, exact real-valued Hermitian energy evaluation, a mandatory 1 to 3
qubit micro-validation matrix, required end-to-end 4- and 6-qubit workflow
cases, a mandatory 4/6/8/10-qubit workflow-scale exact-regime matrix with 10
fixed parameter vectors per required size, one bounded 4-qubit optimization
trace, and explicit hard-error unsupported behavior. This means the project now
supports a research-grade exact noisy variational workflow slice at the
research-workflow level for the frozen canonical workflow scope.

This paper addresses that integration gap. Its purpose is not to introduce a new
fusion algorithm or a new approximation method. Instead, it defines and justifies
the first major step required to make exact noisy mixed-state simulation a
usable scientific instrument inside SQUANDER. The focus is on backend
integration, exact noisy observable evaluation, workflow bridging, realistic
local noise support, and publication-grade validation.

The scope is deliberately constrained. Density-aware partitioning and fusion,
gradient support for the density backend, and approximate scaling methods are all
important future directions, but they are deferred to later phases in order to
keep the scientific claim narrow and defensible. This is a paper about turning a
standalone exact mixed-state capability into a usable exact noisy variational
backend.

### 1.1 Main Contribution

The paper proposes the following Phase 2 contribution:

- a selectable density-matrix backend path for SQUANDER's variational workflow,
- a validated exact Hermitian-energy contract based on `Re Tr(H*rho)`,
- a documented generated-`HEA` bridge from the current VQE entry path into the
  density-matrix execution path,
- a workload-driven Phase 2 support surface for realistic local noise inside
  one canonical noisy XXZ VQE workflow,
- and a benchmark and validation package strong enough to support a major paper
rather than only an internal engineering milestone.

Implementation-backed status for the completed integrated-backend slice:

- implemented and validated:
  - explicit backend selection with no silent fallback,
  - a generated hardware-efficient-ansatz bridge into the density path,
  - fixed ordered local noise insertion,
  - exact real-valued Hermitian energy evaluation through the sparse-Hamiltonian
  interface,
  - a stable canonical workflow contract with explicit workflow ID, input/output
    contract fields, and supported/optional/deferred/unsupported boundary
    classes,
  - a mandatory 1 to 3 qubit micro-validation matrix,
  - required end-to-end 4- and 6-qubit workflow execution cases,
  - a mandatory 4/6/8/10-qubit workflow-scale exact-regime benchmark matrix with
  10 fixed parameter vectors per required size,
  - one bounded 4-qubit optimization trace,
  - structured unsupported-workflow outcomes,
  - runtime and peak-memory characterization for workflow-scale cases,
  - and a workflow-facing validation package archived in
  `benchmarks/density_matrix/artifacts/phase2_task6/` with
  `task6_story6_publication_bundle.json` as the top-level manifest. That
  manifest links the canonical workflow contract
  `story1_canonical_workflow_contract.json`, the Story 2 end-to-end plus trace
  bundle `story2_end_to_end_trace_bundle.json`, the Story 3 matrix baseline
  bundle `story3_matrix_baseline_bundle.json`, the Story 4 unsupported-workflow
  bundle `story4_unsupported_workflow_bundle.json`, and the Story 5
  interpretation-guardrail bundle `story5_interpretation_bundle.json` while
  preserving traceability to the underlying Task 5 validation bundles.

### 1.2 Why This Contribution Matters

This contribution matters for two reasons.

First, it is the enabling step for the later science of the PhD. Optimizer
studies under noise, trainability experiments, and density-aware acceleration
all depend on having a usable and trusted exact noisy backend inside the actual
workflow.

Second, it is scientifically valuable on its own. Many projects possess either:

- a density-matrix simulator disconnected from broader workflow infrastructure,
or
- a training framework that remains effectively state-vector-first.

Bridging that gap in a well-validated and workflow-oriented way is a meaningful
methods contribution, especially when the target use case is realistic noisy
variational research.

### 1.3 Implementation Learnings From Tasks 1-6

Phase 2 implementation produced several learnings that materially shape Paper 1
claim quality:

- explicit support-tier classification (`required`, `optional`, `deferred`,
  `unsupported`) is necessary to avoid over-claiming mandatory baseline support,
- hard-error boundaries at Python normalization, C++ noise-spec validation, and
  density-anchor preflight are required to prevent silent fallback behavior,
- phase-level validation should remain layered: the workflow matrix should close
  independently of the bounded optimization trace, with the trace and documented
  10-qubit anchor packaged as a separate evidence layer,
- the workflow-facing publication bundle must preserve one stable workflow ID and
  contract version across the contract, end-to-end, matrix, unsupported, and
  interpretation artifacts,
- only mandatory, complete, supported evidence may close the main Phase 2 claim;
  optional whole-register depolarizing remains supplemental, and deferred or
  unsupported evidence remains boundary-only,
- metric-completeness and interpretation guardrails should be preserved as
  first-class machine-checkable evidence layers rather than left implicit in raw
  benchmark payloads,
- machine-checkable manifests that verify artifact presence, expected status
  alignment, and workflow identity are required for reproducible publication
  evidence.

## 2. Background and Related Work

### 2.1 Variational Quantum Workflows Under Noise

VQAs are optimized through repeated evaluation of observables with respect to
parameterized quantum circuits. In ideal simulations this reduces to repeated
state-vector evolution and observable evaluation. Under realistic noise,
however, circuit states become mixed, and observable evaluation must be
formulated through density matrices, typically as `Tr(H*rho)`.

This changes both the representation and the scientific interpretation of the
workflow. Mixed-state evolution enables realistic emulation of noise channels,
but it also shifts the computational bottleneck and constrains the accessible
qubit regime. As a result, an exact density-matrix backend must be used
strategically and with clear scope boundaries.

### 2.2 SQUANDER's Existing Strengths

SQUANDER already has strong foundations in:

- circuit decomposition,
- circuit optimization,
- variational workflows,
- and state-vector simulation with partitioning and fusion support.

Its recent density-matrix track provides a solid Phase 1 base:

- exact mixed-state representation,
- a unified noisy-circuit execution path,
- baseline noise channels,
- and reference validation.

The remaining challenge is integration depth rather than raw existence of the
mixed-state module.

### 2.3 Exact Mixed-State Simulation Frameworks

Several prior frameworks are relevant as scientific and engineering context.

QuTiP provides foundational open-system simulation concepts and is especially
important as a superoperator and open-quantum-systems reference. QuEST and
Qulacs provide high-performance simulation context and are useful comparison
points when discussing practical simulation support. DM-Sim and related HPC
work demonstrate that density-matrix simulation is a serious systems problem in
its own right and that communication-aware execution matters quickly once the
problem grows.

These works establish that mixed-state simulation is feasible and scientifically
valuable, but they do not directly solve the integration problem addressed
here: turning a standalone exact density-matrix engine into a validated noisy
variational backend inside an existing training-oriented framework.

### 2.4 Partitioning and Fusion Literature

The current SQUANDER codebase already reflects a strong partitioning and fusion
research lineage through TDAG, GTQCP, QGo, and related work. That literature is
highly relevant to later phases. However, Paper 1 is not a fusion paper. It
depends on those future directions only in the sense that it prepares the exact
noisy backend that later acceleration phases will operate on.

### 2.5 Trainability and Barren Plateaus

The broader PhD motivation is shaped by trainability work, including the
barren-plateau literature and more recent work on local cost functions,
trainability under noise, and mitigation strategies. These papers justify why
an exact noisy backend is worth integrating: without it, later claims about
noise-induced optimizer behavior or trainability would rest on weaker workflow
assumptions.

## 3. Current SQUANDER Baseline

The baseline for this paper is the exact mixed-state capability available at the
end of Phase 1.

### 3.1 What Exists Today

The delivered mixed-state path includes:

- exact dense mixed-state representation,
- a unified gate-plus-noise execution path,
- initial channels including depolarizing, amplitude damping, and phase damping,
- Python bindings and examples,
- and baseline validation against Qiskit Aer.

This is enough to support standalone exact noisy circuit execution and mixed-state
observables at the module level.

### 3.2 What Remains Intentionally Outside The Implemented Scope

The current integrated-backend implementation closes the frozen backend-integration and
validation package. What remains outside the implemented scope is therefore a
matter of deliberate scientific boundary-setting rather than missing foundation:

- broader circuit-source support beyond the generated hardware-efficient-ansatz
path,
- broader observable support beyond exact Hermitian energy evaluation,
- later-phase density-aware acceleration, fusion, and partitioning methods,
- later-phase optimizer and trainability studies,
- and broader multi-framework comparison beyond the mandatory Aer-centered
evidence package.

### 3.3 Why This Gap Matters

With implementation of these documented pieces, the project now has both an
exact noisy module and a delivered exact noisy training workflow slice with a
complete integrated-backend evidence package. For the PhD theme, that closes the main
integration gap and turns the backend into a usable scientific instrument.

## 4. Phase 2 Problem Definition

The Phase 2 problem is:

> how to integrate exact density-matrix simulation into SQUANDER's variational
> workflow in a way that is scientifically useful, exact, validated, and
> publication-ready.

The key challenge is not merely software plumbing. The integration must be
designed in a way that:

- preserves exactness,
- documents the supported use cases,
- avoids premature entanglement with later acceleration work,
- and creates a clean basis for later optimizer and trainability studies.

This paper therefore frames Phase 2 as a specification-led integration problem
with scientific, architectural, and validation dimensions.

## 5. Architectural Integration Goals

Phase 2 has five architectural goals.

### 5.1 Goal 1: Backend Selection

The framework must define how density-matrix execution is selected relative to
existing state-vector execution. This includes:

- the workflow-level contract,
- the intended entry points,
- and the semantics of choosing the exact noisy path.

The purpose of this goal is not to maximize flexibility but to make the density
backend usable and explicit.

### 5.2 Goal 2: Exact Observable Evaluation

The backend must support exact real-valued Hermitian energy evaluation through
`Re Tr(H*rho)` in a way that is suitable for the supported noisy variational
workflow.

This goal is central because observable evaluation is the scientific minimum
needed for noisy VQA studies.

### 5.3 Goal 3: Circuit-to-Backend Bridge

The framework must define how the generated default `HEA` circuit reaches the
density backend. That bridge must be:

- documented,
- bounded by a support matrix,
- and clear about unsupported cases.

### 5.4 Goal 4: Workload-Driven Noise Support

Phase 2 must define a finite, justified noise scope centered on realistic local
noise needed for the first studies. The phase should not try to become a
maximally broad noise framework all at once.

### 5.5 Goal 5: Validation and Publication Evidence

The integration must be justified by an evidence package that can support the
first major paper in the density-matrix track. This includes:

- external reference validation,
- benchmark design,
- and a workflow-level demonstration.

## 6. Scope Boundaries

This paper intentionally excludes several attractive but later-stage directions.

### 6.1 Excluded From Paper 1

- density-aware partitioning,
- gate fusion for density matrices,
- channel-native or superoperator fusion,
- gradient routing and density-backend gradient support,
- approximate scaling via trajectories or MPDOs,
- and full gate parity as a primary milestone.

### 6.2 Why Exclusion Is Necessary

These directions are excluded because they would weaken the clarity of the Phase
2 claim. Paper 1 should establish:

- exact noisy backend integration,
- not density-aware acceleration,
- and not large-scale approximate simulation.

### 6.3 Relationship to Later Papers

These excluded directions are not discarded. They are explicitly reserved for
later phases:

- Phase 3 for density-aware partitioning and fusion,
- Phase 4 for optimizer studies under noise,
- and Phase 5 for trainability science.

## 7. Validation Methodology

Because the contribution is exact-first, validation is central to the paper.

### 7.1 Primary Baseline

Qiskit Aer density-matrix simulation is the primary external reference for
workflow-level validation.

### 7.2 Internal Baseline

The current standalone sequential density-matrix path acts as the internal exact
baseline for consistency within SQUANDER.

### 7.3 Validation Targets

Validation demonstrated in the current integrated-backend bundle includes:

- exact noisy observable agreement on the mandatory 1 to 3 qubit micro matrix,
- required end-to-end support for the canonical noisy workflow at 4 and 6
  qubits together with mandatory 4 / 6 / 8 / 10 qubit fixed-parameter matrix
  evidence with 10 parameter vectors per required size,
- one bounded 4-qubit optimization trace with runtime and peak-memory capture,
- and reproducible behavior under the required local-noise baseline.

### 7.4 Acceptance Thresholds

Paper 1 should not rely on vague statements like “works correctly.” It must
translate correctness into explicit validation outcomes, such as agreement with
trusted references and stable end-to-end execution within the documented exact
regime.

### 7.5 Why Validation Must Be Workflow-Centered

It is possible to validate a simulator module without validating the surrounding
workflow. That is not enough here. The paper is about turning a module into a
scientific backend, so workflow-level validation is required.

## 8. Benchmark Design

### 8.1 Workload Classes

The delivered benchmark suite includes:

- exact noisy small-to-medium circuits,
- one canonical noisy XXZ VQE workflow that depends on mixed-state evaluation,
- and training-relevant circuits consistent with the documented support matrix.

### 8.2 Noise Classes

The delivered benchmark suite prioritizes:

- local depolarizing noise,
- local phase damping or dephasing,
- local amplitude damping,
- whole-register depolarizing only as an optional regression or stress-test
  baseline.

Generalized amplitude damping and coherent local error remain possible later
justified extensions, but they are not part of the delivered evidence cited by
this paper. Whole-register depolarizing can be included as a baseline or
regression case, but it does not count toward closure of the main claim.

### 8.3 Metrics

The delivered benchmark suite collects:

- observable agreement,
- runtime,
- memory footprint,
- workflow stability,
- and reproducibility artifacts.

### 8.4 Exact Regime

The currently documented acceptance anchor for Phase 2 is stability around
roughly 10 qubits. Paper 1 should use this as the exact-regime reference point
for workflow demonstrations rather than making broader unsupported scale claims.

## 9. Observed Results and Publication Claims

Observed Phase 2 results are not framed as performance breakthroughs. They are
framed as backend-enabling scientific infrastructure with publication-quality
evidence.

The main publication claims should be:

- SQUANDER now supports a validated exact noisy backend path for one canonical
  generated-`HEA` noisy XXZ VQE workflow.
- That backend supports exact real-valued Hermitian energy evaluation through
  `Re Tr(H*rho)`.
- The support surface is explicitly documented and tied to realistic local-noise
use cases.
- Only mandatory, complete, supported evidence closes the main Paper 1 claim;
  optional whole-register depolarizing remains supplemental, and deferred or
  unsupported evidence remains boundary-only.
- The resulting workflow is reproducible and sufficient to support the next
phases of noisy optimizer and trainability research.

The paper may include runtime and memory characterization, but those results
should support the usability claim, not replace it.

## 10. Threats to Validity

### 10.1 Limited Exact Scale

Because dense density matrices scale poorly, the exact regime is constrained.
This is a real limitation. The paper should acknowledge it and position the
integration as a high-confidence exact backend rather than a scale-maximization
result.

### 10.2 Support-Surface Limitations

A workload-driven support surface may exclude some gates, noise models, or
workflow variants. This is acceptable if it is documented clearly, but it is
still a limitation of generality.

### 10.3 Benchmark Selection Bias

If the paper only benchmarks workflows that are unusually favorable to the
existing support surface, the result will be weaker. Benchmark design must
therefore remain visibly tied to realistic training use cases.

### 10.4 Reference Dependence

The primary external reference is Qiskit Aer. This is appropriate, but the paper
should still note that validation claims are strongest when they are supported by
clear methodological transparency and, where practical, additional reference
points.

### 10.5 Pre-Acceleration Positioning

Because the paper intentionally defers density-aware acceleration, readers may
question performance ambition. This should be handled by clearly positioning
Paper 1 as the integration step that makes later acceleration scientifically
useful.

## 11. Connection to Later Phases

Paper 1 should explicitly explain how it enables the next stages of the PhD.

### 11.1 Connection to Phase 3

Phase 3 asks whether density-aware partitioning and fusion can accelerate exact
noisy simulation. That question only becomes useful once the exact noisy backend
is already integrated into the research workflow.

### 11.2 Connection to Phase 4

Optimizer comparisons under noise require a stable exact backend so that noisy
effects are not confounded with workflow fragility or missing observable paths.

### 11.3 Connection to Phase 5

The main trainability paper depends on exact noisy workflows, reproducible
instrumentation, and realistic local-noise studies. Phase 2 is the first point
where those become structurally possible.

## 12. Conclusion

This paper defines the first major research-phase transition in the
density-matrix track of SQUANDER. The contribution is not a new acceleration
algorithm and not an approximate scaling method. It is the disciplined
integration of exact noisy density-matrix simulation into the variational
workflow of an existing high-performance quantum framework. By keeping the scope
focused on backend selection, exact observable evaluation, workflow bridging,
realistic local-noise support, and publication-grade validation, the paper
delivers a complete and defensible scientific step that underpins the later
performance and trainability phases of the PhD.

## References

Suggested core references for the full-paper version:

1. Peter Rakyta and Zoltan Zimboras, *Approaching the theoretical limit in
  quantum gate decomposition*, `Quantum 6, 710 (2022)`.
2. Peter Rakyta et al., *Highly optimized quantum circuits synthesized via
  data-flow engines*, `Journal of Computational Physics 500, 112756 (2024)`.
3. Joseph Clark, Travis S. Humble, and Himanshu Thapliyal, *TDAG: Tree-based
  Directed Acyclic Graph Partitioning for Quantum Circuits*, ACM GLSVLSI 2023.
4. Xin-Chuan Wu, Marc Grau Davis, Frederic T. Chong, and Costin Iancu, *QGo:
  Scalable Quantum Circuit Optimization Using Automated Synthesis*,
   `arXiv:2012.09835`.
5. Ang Li, Omer Subasi, Xiu Yang, and Sriram Krishnamoorthy, *Density Matrix
  Quantum Circuit Simulation via the BSP Machine on Modern GPU Clusters*, SC20.
6. J. R. Johansson, P. D. Nation, and Franco Nori, *QuTiP: An open-source
  Python framework for the dynamics of open quantum systems*, Computer Physics
   Communications 2012.
7. Tyson Jones, Anna Brown, Ian Bush, and Simon C. Benjamin, *QuEST and High
  Performance Simulation of Quantum Computers*, Scientific Reports 2019.
8. Yasunari Suzuki et al., *Qulacs: a fast and versatile quantum circuit
  simulator for research purpose*, `Quantum 5, 559 (2021)`.
9. M. Cerezo et al., *Cost Function Dependent Barren Plateaus in Shallow
  Parametrized Quantum Circuits*, Nature Communications 2021.
10. Giacomo De Palma, Milad Marvian, Cambyse Rouze, and Daniel Stilck Franca,
  *Limitations of Variational Quantum Algorithms: A Quantum Optimal Transport
    Approach*, PRX Quantum 2023.
11. Jakab Nadori et al., *Batched Line Search Strategy for Navigating through
  Barren Plateaus in Quantum Circuit Training*, `Quantum 9, 1841 (2025)`.

