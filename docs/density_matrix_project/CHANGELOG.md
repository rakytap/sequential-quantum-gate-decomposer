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

## Phase 3 (Planned)

- Extend the partitioning and gate-fusion subsystem so noisy mixed-state circuits
  are first-class inputs, not just unitary regions with noise left outside the
  partition model.
- Represent noise channels and density-matrix execution semantics inside the
  partitioning/fusion contract.
- Introduce a noise-aware partitioning objective / heuristic and benchmark
  calibration for mixed-state workloads.
- Add mixed-state partitioned/fused execution plus any optional,
  benchmark-driven density-kernel optimizations required by that runtime, such
  as AVX work when profiling justifies it.
- Add correctness and performance validation against the sequential density
  baseline on representative noisy circuit families.

Implementation acceptance criteria:
- Partitioning can represent circuits that contain both gates and noise
  operations while preserving exact execution order.
- Partitioning decisions are calibrated on noisy density workloads rather than
  state-vector-only costs.
- Phase 3 delivers an executable partitioned path with at least one real fused
  execution mode; planner-only representation is not sufficient.
- Partitioned/fused execution matches the unfused density baseline on
  representative noisy circuits.
- Fully channel-native fused noisy blocks are optional follow-on work rather
  than the minimum Phase 3 completion bar.

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

