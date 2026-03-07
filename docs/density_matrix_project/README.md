# Density Matrix Project for SQUANDER

This directory documents the density-matrix track of SQUANDER.

The objective is to move SQUANDER from ideal pure-state simulation to  
noise-aware mixed-state simulation, then integrate that capability into the VQA  
training flow.

## Why This Project Exists

SQUANDER's existing simulation flow is state-vector based. That is efficient for
ideal circuits but cannot represent mixed states produced by noise channels.

Density matrices are required for:

- realistic noisy emulation (depolarizing, T1, T2, and future channels),
- trainability studies under noise (gradient collapse, barren plateaus),
- reproducible noisy VQA experiments.

## Current Status

Phase 1 is complete on `feature/density-matrix-phase1`:

- `DensityMatrix` C++ core with quantum properties and partial trace,
- `NoisyCircuit` unified gate + noise execution path,
- three implemented noise channels (depolarizing, amplitude damping, phase damping),
- Python bindings in `squander.density_matrix`,
- dedicated tests and Qiskit comparison benchmarks.

The remaining work is deeper baseline integration and noisy VQA feature completion.

## 5-Phase Roadmap


| Phase | Goal                                                            | Status   |
| ----- | --------------------------------------------------------------- | -------- |
| 1     | Foundation: density matrices + initial noise channels           | Complete |
| 2     | Deep baseline integration + noise completion + basic VQA hooks  | Planned  |
| 3     | Full noise stack + density-matrix gradients + AVX optimization  | Planned  |
| 4     | Full noisy VQA training loop for 16-20 qubits                   | Planned  |
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
- `[API_REFERENCE.md](API_REFERENCE.md)`: complete phase-1 Python API surface.
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

