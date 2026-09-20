# Integrating Exact Density-Matrix Simulation into a Variational Quantum Training Framework

## Abstract

Variational quantum algorithms require noisy circuit simulation to study
training behavior under realistic conditions, yet integrating exact
density-matrix backends into existing training frameworks presents nontrivial
design challenges around backend selection, observable evaluation, and noise
scheduling. We describe the integration of an exact density-matrix backend
into SQUANDER, a high-performance quantum circuit optimization framework
with mature state-vector partitioning and fusion capabilities. The
integration provides explicit backend selection with no silent fallback,
exact observable evaluation via Re Tr(H rho) using sparse Hamiltonian
structure, and a circuit-to-density bridge with ordered local noise
insertion at specified gate positions. Validation against Qiskit Aer
density-matrix simulation shows energy agreement within 1e-10 on
micro-validation cases (1-3 qubits) and within 1e-8 across the mandatory
4/6/8/10-qubit workflow regime. Per-evaluation performance benchmarks show
the integrated backend is 5-70x faster than Qiskit Aer at 4-8 qubits due
to reduced per-call overhead, while Qiskit Aer is approximately 1.7x faster
at 10 qubits. We report one end-to-end optimization trace on a 4-qubit
noisy XXZ VQE problem demonstrating full training-loop functionality. The
integrated backend establishes the validated foundation for planned
density-aware acceleration through partitioning and fusion in later work.

## Publication Surface Role

This document is the narrative short-paper surface for a general
PhD-conference audience within the Phase 2 Paper 1 package.

## Paper 1 Claim Boundary

Main claim:
SQUANDER's exact density-matrix backend is integrated into one canonical noisy
XXZ VQE workflow through explicit backend selection, exact Hermitian-energy
evaluation via `Re Tr(H*rho)`, a generated-`HEA` bridge, realistic local-noise
support, and a publication-grade validation package.

Explicit non-claims:
- density-aware partitioning and fusion are future work for Phase 3, not current
  Paper 1 results
- density-matrix gradients and approximate scaling are future work beyond the
  current Paper 1 claim
- broad noisy-VQA workflow generality beyond the canonical supported path is not
  a current Paper 1 claim
- broad manual circuit parity or full `qgd_Circuit` parity is not a current
  Paper 1 claim
- optimizer-comparison studies and trainability analysis belong to later phases
  rather than to the delivered Phase 2 result

Supported-path boundary:
The guaranteed Paper 1 path is the generated-`HEA` VQE-facing density route
rather than broad standalone `NoisyCircuit` capability or full `qgd_Circuit`
parity.

No-fallback rule:
No implicit `auto` mode or silent fallback is part of the Phase 2 contract.

Exact-regime boundary:
Full end-to-end workflow execution is required at 4 and 6 qubits,
benchmark-ready fixed-parameter evaluation is required at 8 and 10 qubits, and
the documented 10-qubit case is the acceptance anchor for the current exact
regime.

Evidence-closure rule:
Only mandatory, complete, supported evidence closes the main Paper 1 claim.

Phase positioning:
Paper 1 is the Phase 2 exact noisy backend integration milestone in the
density-matrix publication ladder.

## 1. Introduction

Variational quantum algorithms (VQAs) are among the most studied approaches
for near-term quantum computation [1,2]. Their practical study depends on
accurate classical emulation, and recent work has shown that noise
qualitatively changes the training landscape: Wang et al. [3] proved that
local Pauli noise can induce barren plateaus that do not arise in ideal
simulation, while Fontana et al. [4] demonstrated that non-unital noise
models such as amplitude damping do not necessarily exhibit this effect.
These results mean that the *type* of noise model used in simulation
directly determines the conclusions one can draw about trainability --- and
that training under noise requires a noise-aware simulation backend inside
the optimizer loop.

Several major frameworks already support density-matrix simulation within
variational workflows. Qiskit Aer [5] integrates noise models through its
Estimator primitives and can automatically switch to density-matrix
simulation when noise is present. PennyLane [6] provides a dedicated
`default.mixed` device for density-matrix VQE with differentiable noise.
Cirq [7] offers `DensityMatrixSimulator` for noisy parameter sweeps, and
Qulacs [8] supports both `StateVector` and `DensityMatrix` classes for VQE.
The problem addressed here is therefore not the absence of noisy VQE
capability in the field, but a specific integration gap within SQUANDER.

SQUANDER [9,10] is a high-performance quantum circuit optimization framework
with strong capabilities in gate decomposition, variational optimization,
and state-vector simulation with circuit partitioning and gate fusion.
A standalone density-matrix module was recently added, providing exact
mixed-state representation, a unified gate-and-noise execution layer, and
noise channels including local depolarizing, amplitude damping, and phase
damping. However, this module was not connected to the main variational
workflow: the VQE optimizer could not use density-matrix evaluation as its
cost function. Phase 2 of the density-matrix project closes this gap.

The contribution has three components. First, the density-matrix backend is
now a selectable execution path inside the VQE workflow, with explicit
backend choice and hard-error rejection of unsupported configurations.
Second, exact observable evaluation via Re Tr(H rho) is available using the
existing sparse Hamiltonian interface. Third, a documented bridge lowers the
generated hardware-efficient ansatz (HEA) circuit into the density-matrix
execution path with ordered local noise insertion at specified gate
positions. The integration is validated against Qiskit Aer across the 1-10
qubit exact regime and includes one end-to-end optimization trace
demonstrating training-loop functionality.

## 2. Design and Implementation

### 2.1 Backend Selection

The VQE entry point (`qgd_Variational_Quantum_Eigensolver_Base`) accepts an
explicit `backend` keyword argument. Two modes are supported: `state_vector`
(default, preserving backward compatibility) and `density_matrix`. When
`density_matrix` is selected, the cost function evaluation follows the
density-matrix path described below. Crucially, no implicit automatic mode
or silent fallback exists: if `density_matrix` is selected but the circuit,
ansatz, or noise configuration falls outside the documented support surface,
the system raises a hard error before any computation. Similarly, if
`state_vector` is selected but density-only features (such as ordered local
noise) are configured, a hard error is raised. This design prevents
scientifically ambiguous benchmark results --- every stored result is
attributable to a specific backend.

This differs from Qiskit Aer's `method="automatic"`, which silently
switches between state-vector and density-matrix simulation based on circuit
and noise content without requiring explicit user choice.

### 2.2 Cost Function Evaluation

When the density backend is active, the optimizer's cost function evaluation
proceeds in three steps:

1. **State initialization.** A density matrix rho is initialized to the
   pure state |0...0><0...0| (or a user-supplied initial state).

2. **Circuit evolution.** The generated HEA circuit is lowered into a
   `NoisyCircuit` --- an ordered list of `GateOperation` and
   `NoiseOperation` entries sharing a common interface
   (`IDensityOperation`). Each gate is applied using local kernel
   application: the small 2x2 unitary kernel is applied directly to pairs
   of density-matrix elements differing on the target qubit's index bit,
   at O(4^N) cost per gate instead of O(8^N) for full-matrix U rho U^dag.
   Noise channels reuse the same local structure: local depolarizing
   applies the three Pauli terms via the same 2x2 kernel method, and
   amplitude damping applies two Kraus operators via bit-index logic.
   These are standard techniques in density-matrix simulation [11,12], but
   they are critical enablers for exact VQE in the 4-10 qubit regime.

3. **Observable evaluation.** The energy E(theta) = Re Tr(H rho(theta)) is
   computed by iterating over the nonzero entries of the sparse Hamiltonian
   H (in CSR format) and reading the corresponding density-matrix elements.
   For physically structured Hamiltonians such as the XXZ spin chain used in
   our benchmarks, this costs O(nnz) where nnz is the number of nonzero
   Hamiltonian entries, avoiding a dense matrix product.

The existing optimizer infrastructure (convergence logic, parameter updates,
gradient-free methods such as COSINE and BAYES_OPT) is reused without
modification. The density backend replaces only the cost function
evaluation, which is standard practice in variational frameworks.

### 2.3 Noise Specification

Noise is specified as an ordered list of local insertions, each defined by a
channel type (local depolarizing, amplitude damping, or phase damping), a
target qubit, a gate index after which the noise is applied, and a fixed
noise parameter. During circuit lowering, each noise operation is appended
to the `NoisyCircuit` immediately after the corresponding gate, producing a
fully deterministic and auditable execution order. This differs from
rule-based noise models (as used in Qiskit and Cirq), where noise insertion
is resolved at circuit compilation time according to pattern-matching rules.

## 3. Validation

### 3.1 Methodology

All validation compares SQUANDER's density-matrix backend against Qiskit Aer
(version 0.17.2) density-matrix simulation on identical circuits and noise
configurations. Validation is structured in two tiers:

- **Micro-validation (1-3 qubits):** 7 cases covering U3 and CNOT gates
  with each required noise channel individually and in mixed sequences.
  Acceptance threshold: maximum absolute energy error <= 1e-10, density
  matrix validity (Tr(rho) = 1, rho >= 0, rho = rho^dag) to tolerance 1e-10,
  and |Im Tr(H rho)| <= 1e-10.

- **Workflow-scale (4, 6, 8, 10 qubits):** 10 fixed parameter vectors per
  qubit count, evaluated on the XXZ spin-chain Hamiltonian with HEA ansatz
  and mixed local noise (depolarizing, amplitude damping, phase damping).
  Acceptance threshold: maximum absolute energy error <= 1e-8.

### 3.2 Results

All 7 micro-validation cases pass with maximum energy error below 2.5e-18.
All 40 workflow-scale cases (10 per size at 4, 6, 8, and 10 qubits) pass
within the 1e-8 threshold. Density matrices remain physically valid across
all cases. Unsupported configurations (readout noise, correlated noise,
non-HEA ansatze, state-vector backend with density noise) are confirmed to
fail before execution with named error conditions.

One end-to-end optimization trace on a 4-qubit XXZ problem with mixed local
noise demonstrates full training-loop functionality: the COSINE optimizer
converges from initial energy 0.936 to final energy -4.259 over 18
parameters, with all intermediate evaluations performed through the
density-matrix backend.

### 3.3 Performance

Per-evaluation runtime was measured on the mandatory workflow-scale cases
(Table 1). SQUANDER is faster at 4-8 qubits because it calls the C++
density kernel directly with a pre-lowered circuit, avoiding per-evaluation
overhead from circuit compilation and noise-model resolution that Qiskit Aer
incurs on each call. At 10 qubits, the core matrix operations dominate and
Qiskit Aer's more optimized kernel overtakes SQUANDER's current
implementation.

**Table 1.** Per-evaluation runtime comparison (milliseconds, median over 10
parameter vectors).

| Qubits | SQUANDER (ms) | Qiskit Aer (ms) | Ratio                       |
|--------|---------------|------------------|-----------------------------|
| 4      | 0.06          | 4.1              | SQUANDER 70x faster         |
| 6      | 0.87          | 4.6              | SQUANDER 5x faster          |
| 8      | 26            | 211              | SQUANDER 8x faster          |
| 10     | 435           | 263              | Qiskit Aer 1.7x faster      |

In a VQE training loop with hundreds of evaluations, the per-call advantage
at 4-8 qubits translates directly: a 500-evaluation 8-qubit run completes
in approximately 13 seconds in SQUANDER versus 106 seconds in Qiskit Aer.
The crossover at 10 qubits motivates the planned kernel-level and
partitioning-based acceleration in subsequent phases.

Peak resident memory at 10 qubits is approximately 854 MB, confirming the
practical limit of the exact dense approach at this scale.

## 4. Limitations

The limitations of this work should be stated explicitly.

**Scale.** Dense density matrices require O(4^N) memory. The practical exact
regime is limited to approximately 10 qubits. This is inherent to the exact
approach and is the primary motivation for the planned density-aware
partitioning work.

**Circuit support.** Only the generated HEA ansatz bridge is validated.
Custom circuits, alternative ansatz families, and arbitrary gate structures
are out of scope.

**No density gradients.** The density-matrix backend currently supports only
gradient-free optimizers (COSINE, BAYES_OPT). Gradient-based optimization
through the density path is deferred to future work.

**Kernel performance at scale.** At 10 qubits, Qiskit Aer's density-matrix
kernel is faster than SQUANDER's current implementation. The SQUANDER kernel
does not yet use BLAS or parallelism at the density-matrix operation level;
optimization is reserved for the acceleration phase.

**Standard techniques.** The individual techniques used --- local kernel
gate application, Kraus-operator noise, sparse trace evaluation, backend
dispatch --- are all well-known in the quantum simulation literature. The
contribution is in their integration into a specific framework with
validated workflow-level evidence, not in any single algorithmic novelty.

## 5. Future Work

This integration establishes the exact baseline for three planned research
directions:

- **Noise-aware partitioning and fusion (Phase 3).** SQUANDER's existing
  state-vector partitioning and gate-fusion subsystem will be extended so noisy
  density-matrix circuits are first-class inputs rather than unitary islands
  separated by external noise boundaries. This research direction has no
  equivalent in other frameworks and is expected to close the performance gap
  observed at 10 qubits.

- **Optimizer studies under noise (Phase 4).** With an exact noisy backend
  in the training loop, systematic comparison of optimizer behavior under
  realistic local noise becomes possible.

- **Trainability analysis (Phase 5).** The exact backend provides the
  trusted reference needed for gradient-variance studies, barren-plateau
  diagnostics, and entropy-based trainability metrics under realistic noise.

## 6. Conclusion

We have integrated an exact density-matrix backend into SQUANDER's
variational quantum eigensolver workflow, providing explicit backend
selection, exact observable evaluation, and a documented circuit-to-density
bridge with ordered local noise. The implementation uses standard
density-matrix simulation techniques and is validated against Qiskit Aer
across the 1-10 qubit exact regime with floating-point-level agreement. At
4-8 qubits, the integrated backend is 5-70x faster than Qiskit Aer per
evaluation due to reduced per-call overhead, while Qiskit Aer is faster at
10 qubits. The work establishes a validated exact noisy training backend
and the foundation for density-aware acceleration through partitioning and
fusion in subsequent phases.

## References

[1] A. Peruzzo et al., "A variational eigenvalue solver on a photonic
quantum processor," Nature Communications 5, 4213 (2014).

[2] M. Cerezo et al., "Variational quantum algorithms," Nature Reviews
Physics 3, 625-644 (2021).

[3] S. Wang et al., "Noise-induced barren plateaus in variational quantum
algorithms," Nature Communications 12, 6961 (2021).

[4] E. Fontana et al., "Beyond unital noise in variational quantum
algorithms: noise-induced barren plateaus and limit sets," Quantum 9,
1617 (2025).

[5] Qiskit contributors, "Qiskit Aer: An Aer provider for Qiskit,"
https://github.com/Qiskit/qiskit-aer (2024).

[6] V. Bergholm et al., "PennyLane: Automatic differentiation of hybrid
quantum-classical computations," arXiv:1811.04968 (2018).

[7] Cirq Developers, "Cirq: A Python framework for creating, editing, and
invoking Noisy Intermediate Scale Quantum (NISQ) circuits,"
https://quantumai.google/cirq (2024).

[8] Y. Suzuki et al., "Qulacs: a fast and versatile quantum circuit
simulator for research purpose," Quantum 5, 559 (2021).

[9] P. Rakyta and Z. Zimboras, "Approaching the theoretical limit in
quantum gate decomposition," Quantum 6, 710 (2022).

[10] P. Rakyta et al., "Highly optimized quantum circuits synthesized via
data-flow engines," Journal of Computational Physics 500, 112756 (2024).

[11] A. Li et al., "Density Matrix Quantum Circuit Simulation via the BSP
Machine on Modern GPU Clusters," SC20 (2020).

[12] J. R. Johansson, P. D. Nation, and F. Nori, "QuTiP: An open-source
Python framework for the dynamics of open quantum systems," Computer
Physics Communications 183, 1760-1772 (2012).

[13] J. Nadori et al., "Batched Line Search Strategy for Navigating through
Barren Plateaus in Quantum Circuit Training," Quantum 9, 1841 (2025).
