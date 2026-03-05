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

## Phase 2 (Planned)

- Deep integration entry points:
  - backend selection in VQE configuration,
  - expectation value path `Tr(H*rho)`,
  - bridge from existing gate sequences to density-matrix circuit execution.
- Extend noise stack with additional noise channels:
  - generalized amplitude damping,
  - coherent unitary error model.
- Improve automated validation against Qiskit for noisy small/medium circuits.

Implementation acceptance criteria:
- VQE backend switch executes the density-matrix path.
- `Tr(H*rho)` expectation-value path is validated.
- Noisy emulation is stable at 10 qubits.

---

## Phase 3 (Planned)

- Complete density-matrix noise module features required by experiments.
- Add gradient path for density-matrix optimization.
- Add AVX-focused optimization for key density operations.
- Add calibration and fidelity validation workflows for noise channels.

Implementation acceptance criteria:
- Gradient support is wired for the density backend.
- Expanded noise channels are validated against Qiskit Aer.

---

## Phase 4 (Planned)

- Full noisy VQA integration through the baseline SQUANDER training loop.
- Optimizer evaluations under noise (including BLS workflows).
- Scaling targets for noisy VQA experiments in the 16-20 qubit range.
- Experiment runner and reproducible configuration logging.

Implementation acceptance criteria:
- End-to-end noisy VQA training loop is functional.
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

