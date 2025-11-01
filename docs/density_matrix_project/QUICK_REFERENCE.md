# Density Matrix Quick Reference

**Phase 1**

---

## Installation

```bash
# Setup environment
conda create -n qgd python=3.13 -c conda-forge -y
conda activate qgd
conda install -y numpy scipy pytest scikit-build tbb-devel ninja cmake -c conda-forge
pip install pybind11

# Build and install
cd sequential-quantum-gate-decomposer
git checkout feature/density-matrix-phase1
export TBB_INC_DIR=~/.conda/envs/qgd/include
export TBB_LIB_DIR=~/.conda/envs/qgd/lib
python setup.py build_ext
pip install -e .

# Verify
python -c "from squander.density_matrix import DensityMatrix; print('✅ Working')"
```

**Full setup:** [SETUP.md](SETUP.md)

---

## Import

```python
from squander.density_matrix import (
    DensityMatrix,           # Core density matrix class
    DensityCircuit,          # Circuit builder
    DepolarizingChannel,     # Uniform noise
    AmplitudeDampingChannel, # T1 relaxation
    PhaseDampingChannel,     # T2 dephasing
)
import numpy as np
```

---

## Creating Density Matrices

```python
# Ground state |00⟩⟨00|
rho = DensityMatrix(qbit_num=2)

# From state vector |ψ⟩ → ρ = |ψ⟩⟨ψ|
psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho = DensityMatrix(psi)

# Maximally mixed ρ = I/2^n
rho = DensityMatrix.maximally_mixed(qbit_num=2)
```

---

## Properties (⚠️ All are methods, not properties!)

```python
rho.purity()         # Tr(ρ²) ∈ [0, 1], returns float
rho.entropy()        # Von Neumann entropy, returns float
rho.trace()          # Should be 1.0, returns complex
rho.is_valid(tol)    # Validate density matrix, returns bool
rho.is_hermitian(tol) # Check ρ = ρ†, returns bool
```

**⚠️ Use parentheses:** `rho.purity()` NOT `rho.purity`

---

## Operations

```python
# Apply unitary: ρ → UρU†
rho.apply_unitary(U)

# Partial trace (keep specified qubits)
rho_A = rho.partial_trace([0, 1])  # Keep qubits 0,1

# Export to NumPy
arr = rho.to_numpy()

# Deep copy
rho2 = rho.clone()
```

---

## Building Circuits

```python
circuit = DensityCircuit(qbit_num=2)

# Single-qubit gates
circuit.add_H(target)    # Hadamard
circuit.add_X(target)    # Pauli-X
circuit.add_Y(target)    # Pauli-Y
circuit.add_Z(target)    # Pauli-Z
circuit.add_RX(target)   # Rotation-X (parameterized)
circuit.add_RY(target)   # Rotation-Y (parameterized)
circuit.add_RZ(target)   # Rotation-Z (parameterized)

# Two-qubit gates
circuit.add_CNOT(target, control)
circuit.add_CZ(target, control)

# Apply to density matrix
parameters = np.array([])  # Empty if no parameterized gates
circuit.apply_to(parameters, rho)
```

---

## Noise Channels

### Depolarizing Noise

**Model:** ρ → (1-p)ρ + p·I/2^n

```python
noise = DepolarizingChannel(qbit_num=2, error_rate=0.01)
noise.apply(rho)
```

### Amplitude Damping (T1)

**Model:** Energy relaxation |1⟩ → |0⟩

```python
noise = AmplitudeDampingChannel(qbit_num=2, gamma=0.1)
noise.apply(rho)
```

### Phase Damping (T2)

**Model:** Loss of coherence

```python
noise = PhaseDampingChannel(qbit_num=2, gamma=0.1)
noise.apply(rho)
```

---

## Complete Example

```python
from squander.density_matrix import (
    DensityMatrix, 
    DensityCircuit,
    DepolarizingChannel,
)
import numpy as np

# Create Bell state
rho = DensityMatrix(qbit_num=2)
circuit = DensityCircuit(2)
circuit.add_H(0)
circuit.add_CNOT(1, 0)
circuit.apply_to(np.array([]), rho)

print(f"Pure state purity: {rho.purity()}")  # 1.0

# Add 1% noise
noise = DepolarizingChannel(qbit_num=2, error_rate=0.01)
noise.apply(rho)

print(f"After noise purity: {rho.purity()}")  # ~0.99
print(f"Entropy: {rho.entropy()}")  # > 0

# Check entanglement via partial trace
rho_A = rho.partial_trace([0])
print(f"Reduced purity: {rho_A.purity()}")  # ~0.5 (entangled!)
```

---

## Integration with Existing SQUANDER

```python
from squander.gates.qgd_Circuit import qgd_Circuit  # Existing
from squander.density_matrix import DensityMatrix    # New

# Use existing circuit
sv_circuit = qgd_Circuit(2)
sv_circuit.add_H(0)
sv_circuit.add_CNOT(1, 0)

# Get unitary and apply to density matrix
U = sv_circuit.get_Matrix(np.array([]))
rho = DensityMatrix(qbit_num=2)
rho.apply_unitary(U)
```

---

## Testing

```bash
# Run all tests
pytest tests/density_matrix/ -v

# Expected: 22 passed

# Run examples
python examples/density_matrix/basic_usage.py
```

---

## Common Patterns

### Create Maximally Mixed State

```python
rho = DensityMatrix.maximally_mixed(qbit_num=n)
# or manually:
rho = DensityMatrix(qbit_num=n)
noise = DepolarizingChannel(qbit_num=n, error_rate=1.0)
noise.apply(rho)
```

### Check if Pure State

```python
is_pure = abs(rho.purity() - 1.0) < 1e-10
```

### Check if Entangled (for bipartite systems)

```python
# For pure states: reduced density matrix is mixed ⟺ entangled
rho_A = rho.partial_trace([0])  # Keep first subsystem
is_entangled = abs(rho_A.purity() - 1.0) > 1e-10
```

---

## What's NOT in Phase 1

```python
# ❌ These DO NOT exist in Phase 1:
from squander.density_matrix.core import DensityMatrix  # No core.py
from squander.density_matrix import bell_state_density_matrix  # No convenience functions

# ❌ Properties don't work (use methods instead):
purity = rho.purity   # Wrong
purity = rho.purity()  # Correct ✅
```

---

## Performance Limits (Phase 1)

| Qubits | Memory | Time per gate |
|--------|--------|---------------|
| 5 | 8 KB | ~1 μs |
| 10 | 8 MB | ~1 ms |
| 12 | 128 MB | ~16 ms |

**Note:** Phase 2 will add AVX optimizations for 2-5x speedup.

---

## Troubleshooting

**Can't import module:**
```bash
# Check if built:
ls squander/density_matrix/_density_matrix_cpp*.so

# Rebuild:
python setup.py build_ext
pip install -e .
```

**TBB errors:**
```bash
export TBB_INC_DIR=~/.conda/envs/qgd/include
export TBB_LIB_DIR=~/.conda/envs/qgd/lib
```

**Full troubleshooting:** [SETUP.md](SETUP.md#troubleshooting)

---

## Documentation

- **This guide:** Quick reference for common tasks
- **[README.md](README.md):** Documentation index
- **[SETUP.md](SETUP.md):** Installation guide
- **[phase1-isolated/README.md](phase1-isolated/README.md):** Examples with explanations
- **[phase1-isolated/PHASE1_IMPLEMENTATION.md](phase1-isolated/PHASE1_IMPLEMENTATION.md):** Complete API reference
- **[phase1-isolated/PHASE1_DESIGN.md](phase1-isolated/PHASE1_DESIGN.md):** Design rationale

---

*Last Updated: November 1, 2025*  


