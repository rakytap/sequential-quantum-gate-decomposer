# Architecture overview — density-matrix track (current state)

> **Status:** current-state reference · **Owner skill:** `spec-driven-development` ·
> **Scope:** the density-matrix / noisy-simulation stack inside SQUANDER as it exists after
> delivered Phases 1–3.1 · **Not:** product intent (`PRODUCT_STATEMENT.md`), sequencing
> (`ROADMAP.md`), or decision rationale (ADRs, linked below).
> Update at every milestone close that changes a boundary, flow, integration, or ADR status.

## 1. Context (C4 level 1)

SQUANDER is a C++/Python library for training, synthesising, and simulating quantum
circuits. The density-matrix track adds **exact mixed-state simulation with noise** as an
additive backend, so noisy variational workflows and noise-aware partitioning can be
studied without disturbing the default state-vector path.

| Actor / system | Relationship |
|----------------|--------------|
| Researcher (Python) | Builds `NoisyCircuit`s, runs noisy VQE on the density backend, drives planner/runtime and evidence pipelines |
| SQUANDER state-vector core (`Gates_block`, `qgd_Circuit`, optimizers) | Reused: gate definitions via adapters; optimizer loop via the VQE base class; **not** modified in behavior |
| Qiskit Aer (density-matrix simulator) | External **reference** for correctness and external-validation evidence; optional dependency |
| Evidence pipelines (`benchmarks/density_matrix/`) | Machine-checkable proofs of exactness, bounded support, and performance claims; consumed by closeouts and papers |

## 2. Containers (C4 level 2)

| Container | Location | Responsibility |
|-----------|----------|----------------|
| **C++ density module** | `squander/src-cpp/density_matrix/` (built into the shared C++ tree via `squander_common`) | Exact dense `DensityMatrix`; ordered gate+noise execution via `NoisyCircuit`; `GateOperation` / `NoiseOperation` adapters; legacy standalone `NoiseChannel` API |
| **Python bindings** | `squander/density_matrix/` (`bindings.cpp` → `_density_matrix_cpp`) | pybind11 exposure of `DensityMatrix`, `NoisyCircuit`, `OperationInfo`, noise classes; NumPy interop |
| **Variational integration** | `squander/src-cpp/variational_quantum_eigensolver/` + `squander/VQA/qgd_Variational_Quantum_Eigensolver_Base.py` | Backend selection (state-vector default, density on request), ordered fixed local-noise specification, exact `Re Tr(Hρ)` energy, `describe_density_bridge()` audit metadata |
| **Noisy planner** | `squander/partitioning/noisy_planner.py`, `noisy_planner_surface_builders.py`, `noisy_descriptor.py`, `noisy_types.py`, `noisy_validation_errors.py` | Canonical, schema-versioned ordered operation surface; strict validation; partition descriptors preserving order, noise placement, qubit remap, and parameter routing |
| **Partitioned density runtime** | `squander/partitioning/noisy_runtime.py`, `noisy_runtime_core.py`, `noisy_runtime_fusion.py`, `noisy_runtime_channel_native.py`, `noisy_runtime_errors.py` | Per-partition `NoisyCircuit` execution on one global `DensityMatrix`; descriptor-local unitary-island fusion; bounded strict/hybrid channel-native (Kraus-bundle) execution on the frozen Phase 3.1 slice; `execute_sequential_density_reference` |
| **State-vector partitioners** | `squander/partitioning/{kahn,tdag,ilp,partition,split,tools}.py` | Mature ideal-circuit planners; a **parallel** contract, not replaced by the noisy planner |
| **Evidence pipelines** | `benchmarks/density_matrix/<tree>/` | Workflow, bridge-scope, noise-support, planner-surface, partitioned-runtime, planner-calibration, correctness, performance and publication bundles with validators and `validation_pipeline.py` entry points |
| **Tests** | `tests/density_matrix/`, `tests/partitioning/`, `tests/VQE/`, `squander/src-cpp/density_matrix/tests/test_basic.cpp` | Python and optional C++ suites aligned with the containers above |

## 3. Bounded contexts and ownership

| Context | Owns | Ubiquitous language |
|---------|------|---------------------|
| **Mixed-state core** (C++) | Density-matrix state, exact evolution, noise channel semantics | `DensityMatrix`, `NoisyCircuit`, `IDensityOperation`, `GateOperation`, `NoiseOperation`, purity, entropy, partial trace |
| **Noisy variational workflow** | Backend choice, supported workflow surface, energy evaluation, bridge metadata | backend, density path, ordered fixed local noise, canonical noisy XXZ workflow, `Re Tr(Hρ)`, unsupported → hard error |
| **Noisy planning and execution** | Canonical operation surface, descriptors, partitioned/fused execution, reference execution | canonical surface, partition descriptor, parameter routing, unitary island, fused execution, channel-native (strict / hybrid), sequential density reference |
| **Evidence** | Case ids, bundles, validators, manifests | counted case, continuity anchor, evidence bundle, diagnosis-grounded closure, claim boundary |

Contexts talk through explicit surfaces: the workflow context hands a lowered gate/noise
sequence to planning via `describe_density_bridge()`; planning hands validated descriptor
sets to the runtime; the runtime emits execution records the evidence context consumes.

## 4. Ports, adapters, and the dependency rule

- `IDensityOperation` (`density_operation.h`) is the port every executable operation
  implements; `GateOperation` adapts existing SQUANDER `Gate*` types so gate definitions stay
  centralised; `NoiseOperation` subclasses implement channels in the circuit model.
- The **sequential `NoisyCircuit` executor is the exact semantic baseline**. Every
  partitioned, fused, channel-native, or future backend path is validated against it within
  frozen tolerances; Qiskit Aer is the external reference, never the internal oracle.
- Dependency direction: evidence → runtime → planner → bindings → C++ core. Python contracts
  (planner, runtime) may evolve their schemas; the C++ core does not depend on them.
- The density path is **strict**: unsupported circuit sources, gates, noise names, or
  optimizer modes raise explicit, structured errors. There is no silent fallback to the
  state-vector path or to an unfused execution.

## 5. Major flows

1. **Exact noisy energy evaluation** — `qgd_Variational_Quantum_Eigensolver_Base`
   (density backend) → lowers the generated-`HEA` ansatz plus ordered local noise →
   `NoisyCircuit` → `DensityMatrix` → `Re Tr(Hρ)` with the sparse Hamiltonian.
2. **Partitioned execution** — canonical operation list → `noisy_planner` validation →
   descriptor set (order, noise placement, remap, parameter routing) → `noisy_runtime`
   executes per partition on one global `DensityMatrix`, optionally fusing eligible
   unitary islands or running the bounded channel-native modes → execution record.
3. **Evidence generation** — a pipeline under `benchmarks/density_matrix/<tree>/` builds
   counted cases, runs runtime and reference paths (and Aer where required), validates
   bundles against schemas and claim rules, and writes artifacts under
   `benchmarks/density_matrix/artifacts/<tree>/`.

## 6. Runtime and deployment

Single-process library: CPU only, dense complex128 state (`matrix_base<QGD_Complex16>`),
TBB for the existing C++ parallelism, exponential memory in qubit count (exact regime
validated at 4–10 qubits). Built with scikit-build/CMake into an editable Python package in
the `qgd` conda environment; no services, network, or persistent storage beyond evidence
artifacts on disk. Commands and lanes: `TECH_STACK.md`.

## 7. Current risks and constraints

- **Exactness vs scale** — dense exact simulation bounds the reachable qubit count;
  acceleration is additive and must match the sequential reference (ADR-005).
- **Two noise representations** — in-circuit `NoiseOperation` vs legacy standalone
  `NoiseChannel` duplicates some logic; behavioral truth follows the circuit-ordered model.
- **Python-level fused kernels** add overhead; delivered performance closure is
  diagnosis-grounded (where time goes), not a blanket speedup. Channel-native fusion closed
  as a bounded decision study: `17` rows `phase3_sufficient`, `9` `phase31_not_justified_yet`,
  `0` `phase31_justified` on the frozen 26-row matrix.
- **Bounded support surface** — gate/noise names, circuit sources, and optimizer modes are
  intentionally restricted; full `qgd_Circuit` parity and density-backend gradient routing
  are not delivered.
- **Conservative fusion** — unitary-island fusion is exact on eligible substructures and is
  not a general superoperator fusion architecture.

## 8. Decisions and further reading

Program-level ADRs (in force until superseded by a milestone ADR under `docs/specs/`):
[`ADR-001 … ADR-008`](../density_matrix_project/archive/planning/ADRs.md). Phase-level
decision records and the delivered contracts:
[`archive/phases/phase-2/ADRs_PHASE_2.md`](../density_matrix_project/archive/phases/phase-2/ADRs_PHASE_2.md),
[`phase-3/ADRs_PHASE_3.md`](../density_matrix_project/archive/phases/phase-3/ADRs_PHASE_3.md),
[`phase-3-1/ADRs_PHASE_3_1.md`](../density_matrix_project/archive/phases/phase-3-1/ADRs_PHASE_3_1.md),
and the Phase 3.1 evidence review
[`PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md`](../density_matrix_project/archive/phases/phase-3-1/PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md).

API as of the last closures:
[`API_REFERENCE_PHASE_1.md`](../density_matrix_project/archive/phases/phase-1/API_REFERENCE_PHASE_1.md)
(core types) and
[`API_REFERENCE_PHASE_3.md`](../density_matrix_project/archive/phases/phase-3/API_REFERENCE_PHASE_3.md)
(variational integration, planner, runtime). Planner source inventory:
[`CANONICAL_NOISY_PLANNER_SURFACE_SOURCES.md`](../density_matrix_project/CANONICAL_NOISY_PLANNER_SURFACE_SOURCES.md).
