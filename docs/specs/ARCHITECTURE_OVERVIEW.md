# Architecture overview — density-matrix track (current state)

> **Status:** current-state reference · **Owner skill:** `spec-driven-development` ·
> **Scope:** the density-matrix / noisy-simulation stack inside SQUANDER as it exists after
> delivered Phases 1–3.1 and the recorded M-F1a denominator (G-05 docs `11795eed`; RM admin
> `completeness_claim` true for handoff 2026-10-06; frozen bundle JSON may still record false at
> generation) · **Not:** product intent (`PRODUCT_STATEMENT.md`), sequencing
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

As of `031996f4`, the recorded M-F1a advertised-route boundary is four routes at anchors 4, 6, 8,
and 10, each with `max_partition_qubits` 2: `partitioned_density_descriptor_baseline`
(`execute_partitioned_density`), `partitioned_density_descriptor_fused_unitary_islands`
(`execute_partitioned_density_fused`), `phase31_channel_native`
(`execute_partitioned_density_channel_native`), and `phase31_channel_native_hybrid`
(`execute_partitioned_density_channel_native_hybrid`). The oracle is
`execute_sequential_density_reference` and is not a route under test. The counted bundle is
`benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/counted/mf1a_counted_bundle.json`
(sha256 `a22ee685038170cb0991a9ae2195b20ad4c977b111409b312cc708fa9e6872f9`, recorded at task-9
C2 `0a66f974`). **M-F1a exactness reconfirmed** for the ADR-F1A-001 counted denominator (16/16
QA-001), with stated disclosures: baseline shared-kernel agreement with the sequential oracle;
strict cells are products of disjoint pair states. Admin close at docs HEAD `11795eed` (G-04
rocky PASS recorded at `031996f4`). Docs record `completeness_claim` true for handoff; the frozen
counted bundle JSON may still show `completeness_claim` false as recorded at generation time.
No Aer, energy, or frozen-matrix claim beyond the counted denominator and disclosures in
`EXACTNESS_RECONFIRMATION_CLOSEOUT.md` §4.

Counted M-F1a rows use the ADR-F1A-002 predicate only: Frobenius norm, max-abs entry, and
absolute trace deviation at most `1e-10`, and `lambda_min` at least `-1e-12`, with every state
entry and residual finite. `lambda_min` is one-sided: a finding is reported only below
`-1e-13`, and a positive value is neither a finding nor a marker.

Evaluation mode is `phase31_channel_native_hybrid` only. Every hybrid partition carries one
runtime-class label and one route-reason label from the frozen vocabulary, and each label agrees
with a fused-region witness. Baseline, fused, and strict keep requested and realized paths and do
not gain per-partition labels.

Baseline cells and the oracle share `_build_runtime_circuit` and the C++ `NoisyCircuit` kernels,
so agreement there is bitwise and a kernel-level bug appears on both sides. The strict cells at 4,
6, 8, and 10 are products of disjoint pair states and do not test inputs correlated across a
partition boundary. Both limits are recorded in `EXACTNESS_RECONFIRMATION_CLOSEOUT.md` §4.

The M-F1a exactness fitness lane is
`benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` together with
`pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts=""`. The fast
`pytest -m "density_matrix and not slow"` lane is not that fitness lane.

Regeneration command:
`PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`

M-F1a artifacts live under `benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/`.
The counted file is `mf1a/counted/mf1a_counted_bundle.json`. The other five directories are the
provisional siblings (`milestone_counted` false): `q4_baseline`, `fused`, `hybrid`, `strict`,
and `baseline`. The eight pre-M-F1a bundles are verified and not written.

The M-F1a state-vector gate is rocky-local project CI (G-04; Tester recipe; GitHub Actions out
of semester path per Zoltán 2026-10-06) at HEAD `031996f4`. It is not `.github/workflows/ci.yml`
`workflow_dispatch`. G-05 does not run that job. O-10 push at `031996f4` is sync-only. Recorded
outcome at write time: pass, `/tmp/mf1a-g04-rocky-ci/REPORT.md` (1067 passed, 1 deselected QX2,
exit 0, wall 40m56s; pytest.log `/tmp/mf1a-g04-rocky-ci/pytest.log`). A missing or failed job is
not described as green.

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
   counted cases, runs each advertised route and the sequential oracle (Aer is an optional
   external reference and is not in the M-F1a count), validates bundles against schemas and
   claim rules, and writes artifacts under `benchmarks/density_matrix/artifacts/<tree>/`.

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
