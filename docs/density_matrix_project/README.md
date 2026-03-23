# Density Matrix Project for SQUANDER

This directory documents the density-matrix track of SQUANDER.

The objective is to move SQUANDER from ideal pure-state simulation to
noise-aware mixed-state simulation, then make noise a first-class concern in
the partitioning and gate-fusion stack, and only after that broaden noisy
VQE/VQA workflow capabilities.

## Why This Project Exists

SQUANDER's existing simulation flow is state-vector based. That is efficient for
ideal circuits but cannot represent mixed states produced by noise channels.

Density matrices are required for:

- realistic noisy emulation (depolarizing, T1, T2, and future channels),
- trainability studies under noise (gradient collapse, barren plateaus),
- reproducible noisy VQA experiments.

## Current Status

Phase 1 is complete and established the foundation for the density-matrix project:

- `DensityMatrix` C++ core with quantum properties and partial trace,
- `NoisyCircuit` unified gate + noise execution path,
- three implemented noise channels (depolarizing, amplitude damping, phase damping),
- Python bindings in `squander.density_matrix`,
- dedicated tests and Qiskit comparison benchmarks.

Phase 2 is complete and established the current exact noisy-workflow baseline:

- backend selection between state-vector and density-matrix execution,
- exact Hermitian-energy evaluation via `Re Tr(H*rho)`,
- one canonical noisy XXZ workflow contract with explicit supported and
  deferred boundaries,
- and machine-checkable validation/publication bundles for that frozen support
  surface.

Phase 3 is complete and established the noise-aware partitioning/fusion foundation:

- canonical noisy mixed-state planner and descriptor surfaces that treat gates
  and noise as first-class planner inputs,
- an executable partitioned density runtime with limited real fused execution on
  eligible substructures,
- machine-checkable correctness, performance, and publication-evidence bundles
  grounded in sequential `NoisyCircuit` and Qiskit Aer baselines,
- and a bounded planner-calibration result with diagnosis-grounded performance
  closure, while channel-native fusion and broader workflow growth remain
  deferred.

Phase 4 is planned and will broaden the noisy VQE/VQA surface, gradients, and
optimizer studies.

Phase 5 is planned and will focus on trainability analysis under noise
(gradient collapse, barren plateaus).


## 5-Phase Roadmap


| Phase | Goal                                                            | Status   |
| ----- | --------------------------------------------------------------- | -------- |
| 1     | Foundation: density matrices + initial noise channels           | Complete |
| 2     | Exact noisy backend integration for one canonical workflow       | Complete |
| 3     | Noise-aware partitioning and gate fusion for mixed-state circuits | Complete |
| 4     | Broader noisy VQE/VQA features, gradients, and optimizer studies | Planned |
| 5     | Trainability analysis under noise (BP and expressivity studies) | Planned  |


Notes:

- Scope here is density-matrix-specific work.
- GPU kernel development is tracked separately and can be integrated per phase as
available.
- Stochastic trajectory methods are deferred to later project stages.

## Documentation Map

- `[CHANGELOG.md](CHANGELOG.md)`: delivered phase outputs and upcoming phase
targets.
- `[ARCHITECTURE.md](ARCHITECTURE.md)`: implementation structure and integration
extension points.
- `[SETUP.md](SETUP.md)`: environment setup, build, verification, troubleshooting.

## Minimal Hello World

```python
from squander.density_matrix import DensityMatrix, NoisyCircuit
import numpy as np

rho = DensityMatrix(qbit_num=2)
circuit = NoisyCircuit(2)
circuit.add_H(0)
circuit.add_CNOT(1, 0)
circuit.apply_to(np.array([]), rho)
```

