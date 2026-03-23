# Density Matrix Changelog and Roadmap

Primary audience: maintainers tracking delivered scope vs planned milestones.

## Phase 1 (Complete)

### Delivered

- Added new C++ density matrix module:
  - `DensityMatrix`,
  - `NoisyCircuit`,
  - operation adapters and noise operations,
  - legacy standalone noise channel API.
- Added Python package integration:
  - `squander.density_matrix`,
  - pybind11 extension `_density_matrix_cpp`.
- Added initial noise model support:
  - depolarizing,
  - amplitude damping (T1),
  - phase damping (T2).
- Added dedicated test suites:
  - `tests/density_matrix/test_density_matrix.py` (35 test functions plus
    parametrized cases),
  - `squander/src-cpp/density_matrix/tests/test_basic.cpp` (20 C++ tests).
- Added benchmark utilities and Qiskit Aer comparison scripts in `benchmarks/`.
- Integrated build through CMake with `squander_common` and module subdirectory.

### Scope Notes

- Integration with existing SQUANDER VQA/decomposition internals is limited in
  phase 1.
- Density matrix backend exists as a dedicated module and API, not yet a
  selectable backend in current VQE training loops.

---

## Phase 2 (Complete)

- Deep integration entry points:
  - backend selection in VQE configuration,
  - expectation value path `Tr(H*rho)`,
  - bridge from existing gate sequences to density-matrix circuit execution,
  - canonical noisy XXZ workflow contract and publication-facing validation
    bundle.
- Extend noise stack with additional noise channels:
  - generalized amplitude damping,
  - coherent unitary error model.
- Improve automated validation against Qiskit for noisy small/medium circuits.

Implementation acceptance criteria:
- VQE backend switch executes the density-matrix path.
- `Tr(H*rho)` expectation-value path is validated.
- Noisy emulation is stable at 10 qubits.

---

## Phase 3 (Complete)

### Delivered

- Added a canonical noisy mixed-state planner surface and schema-backed
  partition descriptor contract in `squander/partitioning/noisy_planner.py`.
- Added descriptor metadata that preserves explicit gate/noise order, qubit
  remapping, and parameter-routing semantics on the supported Phase 3 surface.
- Added an executable partitioned density runtime in
  `squander/partitioning/noisy_runtime.py`.
- Added a conservative real fused execution baseline via descriptor-local
  unitary-island fusion on eligible supported substructures.
- Added machine-checkable correctness, performance, and publication-evidence
  pipelines under `benchmarks/density_matrix/`.

Documented closure points:
- Noisy mixed-state circuits enter the planner as first-class supported inputs
  without reducing noise to boundary-only metadata.
- Phase 3 delivers more than planner-only representation: the shipped runtime
  executes partitioned workloads and exercises at least one real fused path.
- Partitioned and fused execution are validated against the sequential density
  baseline on the required workloads.

### Scope Notes

- Sequential `NoisyCircuit` execution and Qiskit Aer remain the required
  validation baselines.
- The delivered planner-calibration result is bounded to the audited Phase 3
  support surface; it does not claim full density-aware parity for every
  planner variant or circuit source.
- Phase 3 performance closure is diagnosis-grounded rather than a blanket
  speedup claim.
- Fully channel-native fused noisy blocks, full `qgd_Circuit` parity, broader
  Phase 4 workflow growth, and approximate scaling remain deferred.

---

## Phase 4 (Planned)

- Broaden noisy VQE/VQA integration beyond the frozen Phase 2 canonical
  workflow.
- Add density-backend gradient and optimizer routing for the supported Phase 4
  workflow surface.
- Optimizer evaluations under noise (including BLS workflows).
- Scaling targets for noisy VQA experiments in the 16-20 qubit range.
- Experiment runner and reproducible configuration logging.

Implementation acceptance criteria:
- End-to-end noisy VQE/VQA workflows beyond the Phase 2 baseline are
  functional.
- Density-backend gradient and optimizer support exist for the supported Phase 4
  workflow surface.
- Optimizer comparison experiments are reproducible.

---

## Phase 5 (Planned)

- Trainability analysis framework under noise:
  - gradient variance,
  - entropy metrics,
  - barren plateau diagnostics.
- Controlled unital vs non-unital noise studies.
- Batch experiment pipeline for statistical analyses and publication-ready data.

Implementation acceptance criteria:
- Complete trainability analysis dataset is produced.
- Publication-grade figures and documented conclusions are generated.

---

## Status Policy

- `Complete`: implemented, tested, and merged in branch scope.
- `Planned`: agreed roadmap item not fully implemented yet.
- `In Progress`: active development in the current milestone window.

