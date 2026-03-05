# Density Matrix API Reference (Phase 1)

This is the API source of truth for `squander.density_matrix` on
`feature/density-matrix-phase1`.

Primary audience: developers and advanced users writing code against phase-1 APIs.

## Import

```python
from squander.density_matrix import (
    DensityMatrix,
    NoisyCircuit,
    NoiseChannel,
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel,
)
```

---

## `DensityMatrix`

Quantum mixed-state container with validation, observables, and evolution
operations.

### Constructors

- `DensityMatrix(qbit_num: int)`
  - Initializes to `|0...0><0...0|`.
- `DensityMatrix(state_vector: np.ndarray[complex])`
  - Builds `rho = |psi><psi|` from a 1D state vector.
- `DensityMatrix.from_numpy(array: np.ndarray[complex]) -> DensityMatrix`
  - Builds from a square `2^n x 2^n` matrix.
- `DensityMatrix.maximally_mixed(qbit_num: int) -> DensityMatrix`
  - Builds `I / 2^n`.

### Properties

- `qbit_num: int`
- `dim: int`

### Methods

State properties:
- `trace() -> complex`
- `purity() -> float`
- `entropy() -> float`
- `is_valid(tol: float = 1e-10) -> bool`
- `eigenvalues() -> list[float]`

State operations:
- `apply_unitary(U: np.ndarray[complex]) -> None`
- `apply_single_qubit_unitary(u_2x2: np.ndarray[complex], target_qbit: int) -> None`
- `partial_trace(trace_out: list[int]) -> DensityMatrix`
- `clone() -> DensityMatrix`
- `to_numpy() -> np.ndarray[complex]`

### Example

```python
from squander.density_matrix import DensityMatrix
import numpy as np

rho = DensityMatrix(qbit_num=2)
print(rho.purity())   # 1.0

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
rho.apply_single_qubit_unitary(H, target_qbit=0)
print(rho.is_valid()) # True
```

---

## `NoisyCircuit`

Circuit builder for density-matrix evolution with both unitary gates and noise
channels.

### Constructor

- `NoisyCircuit(qbit_num: int)`

### Properties

- `qbit_num: int`
- `parameter_num: int` (total number of circuit parameters)
- `len(circuit)` returns operation count

### Gate APIs

Single-qubit constant gates:
- `add_H(target)`
- `add_X(target)`
- `add_Y(target)`
- `add_Z(target)`
- `add_S(target)`
- `add_Sdg(target)`
- `add_T(target)`
- `add_Tdg(target)`
- `add_SX(target)`

Single-qubit parametric gates:
- `add_RX(target)` (1 parameter)
- `add_RY(target)` (1 parameter)
- `add_RZ(target)` (1 parameter)
- `add_U1(target)` (1 parameter)
- `add_U2(target)` (2 parameters)
- `add_U3(target)` (3 parameters)

Two-qubit constant gates:
- `add_CNOT(target, control)`
- `add_CZ(target, control)`
- `add_CH(target, control)`

Two-qubit parametric gates:
- `add_CRY(target, control)` (1 parameter)
- `add_CRZ(target, control)` (1 parameter)
- `add_CRX(target, control)` (1 parameter)
- `add_CP(target, control)` (1 parameter)

### Noise APIs

- `add_depolarizing(qbit_num, error_rate=None)`
  - `error_rate=None` -> parametric (adds 1 parameter)
  - `error_rate=float` -> fixed channel (adds 0 parameters)

- `add_amplitude_damping(target, gamma=None)`
  - `gamma=None` -> parametric (adds 1 parameter)
  - `gamma=float` -> fixed channel (adds 0 parameters)

- `add_phase_damping(target, lambda_param=None)`
  - `lambda_param=None` -> parametric (adds 1 parameter)
  - `lambda_param=float` -> fixed channel (adds 0 parameters)

### Execution and inspection

- `apply_to(parameters: np.ndarray[float], density_matrix: DensityMatrix) -> None`
- `get_operation_info() -> list[OperationInfo]`

Parameter ordering rule:
- Parameters are consumed in the same order parametric operations were added.

### Example

```python
from squander.density_matrix import DensityMatrix, NoisyCircuit
import numpy as np

rho = DensityMatrix(qbit_num=2)
circuit = NoisyCircuit(2)
circuit.add_H(0)
circuit.add_CNOT(1, 0)
circuit.add_RZ(0)                 # parameter 0
circuit.add_phase_damping(0)      # parameter 1

params = np.array([0.7, 0.03], dtype=float)
circuit.apply_to(params, rho)
print(circuit.parameter_num)       # 2
print(rho.purity())                # < 1 due to noise
```

---

## `OperationInfo`

Returned by `NoisyCircuit.get_operation_info()`.

Fields:
- `name: str`
- `is_unitary: bool`
- `param_count: int`
- `param_start: int`

### Example

```python
info = circuit.get_operation_info()
for op in info:
    print(op.name, op.is_unitary, op.param_count, op.param_start)
```

---

## Legacy Standalone Noise Channels

These classes apply noise directly to an existing `DensityMatrix`.

### `NoiseChannel`

- Base class
- Methods:
  - `apply(density_matrix: DensityMatrix) -> None`
  - `get_name() -> str`

### `DepolarizingChannel`

- `DepolarizingChannel(qbit_num: int, error_rate: float)`
- Read-only properties:
  - `qbit_num`
  - `error_rate`

### `AmplitudeDampingChannel`

- `AmplitudeDampingChannel(target_qbit: int, gamma: float)`
- Read-only properties:
  - `target_qbit`
  - `gamma`

### `PhaseDampingChannel`

- `PhaseDampingChannel(target_qbit: int, lambda: float)`
  - Use positional arguments in Python because `lambda` is a reserved keyword.
- Read-only properties:
  - `target_qbit`
  - `lambda_param`

### Example

```python
from squander.density_matrix import DensityMatrix, DepolarizingChannel

rho = DensityMatrix(qbit_num=2)
noise = DepolarizingChannel(qbit_num=2, error_rate=0.01)
noise.apply(rho)
```

---

## Common Pitfalls

- Use method calls with parentheses:
  - correct: `rho.purity()`
  - incorrect: `rho.purity`
- For parametric circuits, ensure parameter vector length equals
  `circuit.parameter_num`.
- `NoisyCircuit` gate signatures follow `(target, control)` for controlled gates.

