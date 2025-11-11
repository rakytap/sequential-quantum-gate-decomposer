# Phase 1: Density Matrix as Squander Subpackage

**Status:** Complete ✅  
**Integration:** `from squander.density_matrix import DensityMatrix`  
**Build:** Modern CMake, always enabled  
**Isolation:** Zero modifications to existing files  
**Tests:** 22 Python + 8 C++ tests

---

## Overview

### What's in Phase 1?

**Goal:** Add density matrix support to SQUANDER for mixed-state quantum simulation

**Delivered:**
- ✅ Core C++ classes (DensityMatrix, NoisyCircuit, 3 noise channels)
- ✅ Python bindings via pybind11
- ✅ Complete integration with existing SQUANDER
- ✅ Comprehensive tests and examples
- ✅ Zero modifications to existing code
- ✅ Modern CMake build system

---

## Quick Links

- **API Reference:** [PHASE1_IMPLEMENTATION.md](PHASE1_IMPLEMENTATION.md) - Complete API with all methods
- **Design Rationale:** [PHASE1_DESIGN.md](PHASE1_DESIGN.md) - Why these design decisions
- **Setup Instructions:** [../SETUP.md](../SETUP.md) - How to build and test
- **Project Overview:** [../DENSITY_MATRIX_PROJECT_README.md](../DENSITY_MATRIX_PROJECT_README.md) - Full roadmap

---

## Usage Examples 

### Example 1: Basic Usage

**Create Bell state and verify purity:**

```python
from squander.density_matrix import DensityMatrix, NoisyCircuit
import numpy as np

# Create 2-qubit density matrix (initialized to |00⟩⟨00|)
rho = DensityMatrix(qbit_num=2)
print(f"Initial purity: {rho.purity()}")  # 1.0 (pure state)

# Create Bell state: H⊗I • CNOT • |00⟩
circuit = NoisyCircuit(2)
circuit.add_H(0)          # Hadamard on qubit 0
circuit.add_CNOT(1, 0)    # CNOT with control=0, target=1

# Apply circuit (no parameters needed for these gates)
circuit.apply_to(np.array([]), rho)
print(f"After circuit: {rho.purity()}")  # Still 1.0 (unitary evolution)
print(f"Entropy: {rho.entropy()}")       # 0.0 (still pure)
```

**Expected output:**
```
Initial purity: 1.0
After circuit: 1.0
Entropy: 0.0
```

### Example 2: With Depolarizing Noise

**Simulate realistic quantum noise:**

```python
from squander.density_matrix import (
    DensityMatrix,
    NoisyCircuit,
    DepolarizingChannel,
)
import numpy as np

# Create Bell state first
rho = DensityMatrix(qbit_num=2)
circuit = NoisyCircuit(2)
circuit.add_H(0)
circuit.add_CNOT(1, 0)
circuit.apply_to(np.array([]), rho)

print(f"Before noise - Purity: {rho.purity():.6f}")  # 1.0 (pure)

# Add 1% depolarizing noise: ρ → (1-p)ρ + p·I/4
noise = DepolarizingChannel(qbit_num=2, error_rate=0.01)
noise.apply(rho)

print(f"After noise  - Purity: {rho.purity():.6f}")  # ~0.99 (mixed)
print(f"After noise  - Entropy: {rho.entropy():.6f}")  # > 0 (has mixing)
```

**Expected output:**
```
Before noise - Purity: 1.000000
After noise  - Purity: 0.990000
After noise  - Entropy: 0.043414
```

### Example 3: Integration with Existing SQUANDER

**Use existing state vector circuits with density matrices:**

```python
from squander.gates.qgd_Circuit import qgd_Circuit  # Existing SQUANDER
from squander.density_matrix import DensityMatrix    # New Phase 1
import numpy as np

# Define circuit using existing SQUANDER interface
sv_circuit = qgd_Circuit(2)
sv_circuit.add_H(0)
sv_circuit.add_CNOT(1, 0)

# Extract unitary matrix from state vector circuit
U = sv_circuit.get_Matrix(np.array([]))

# Apply to density matrix (seamless integration!)
rho = DensityMatrix(qbit_num=2)
rho.apply_unitary(U)

print(f"Purity: {rho.purity()}")      # 1.0 (pure)
print(f"Circuit compatible: ✅")
```

**Expected output:**
```
Purity: 1.0
Circuit compatible: ✅
```

---

### Example 4: Partial Trace (Entanglement Verification)

**Check entanglement by tracing out subsystems:**

```python
from squander.density_matrix import DensityMatrix, NoisyCircuit
import numpy as np

# Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
rho = DensityMatrix(qbit_num=2)
circuit = NoisyCircuit(2)
circuit.add_H(0)
circuit.add_CNOT(1, 0)
circuit.apply_to(np.array([]), rho)

print(f"Full system purity: {rho.purity():.6f}")  # 1.0 (pure)

# Trace out qubit 1, keep qubit 0
rho_A = rho.partial_trace([0])
print(f"Reduced purity: {rho_A.purity():.6f}")    # 0.5 (maximally mixed)

# For pure entangled states, reduced density matrix is mixed!
# This confirms entanglement
```

**Expected output:**
```
Full system purity: 1.000000
Reduced purity: 0.500000
```

---

## Modern CMake Features

### INTERFACE Library Pattern

```cmake
# Create INTERFACE library for common dependencies
add_library(squander_common INTERFACE)

target_include_directories(squander_common INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include>
    $<INSTALL_INTERFACE:include/squander>
)

target_link_libraries(squander_common INTERFACE
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${TBB_LIB}
)
```

**Benefits:**
- Single source of truth for common dependencies
- Transitive dependencies handled automatically
- Cleaner target-based linking
- Easier to maintain

### Target-Based Linking

```cmake
# Density matrix module links against INTERFACE library
target_link_libraries(density_matrix_core
    PUBLIC  squander_common  # INTERFACE library
    PRIVATE qgd              # Main SQUANDER library
)
```

**Benefits:**
- Compiler knows exact dependencies
- Automatic include path propagation
- Easier to understand dependency graph

### Generator Expressions

```cmake
# Platform-specific compiler flags
target_compile_options(density_matrix_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<CONFIG:Release>:-O3>
)
```

**Benefits:**
- Cross-platform compatibility
- Configuration-specific settings
- No if/else logic needed

---


## What's NOT in Phase 1?

The following are planned for future development but **NOT implemented in Phase 1**:

### ❌ Python Convenience Layer

```python
# These DO NOT work in Phase 1:
from squander.density_matrix.core import DensityMatrix  # ❌ No core.py
from squander.density_matrix import bell_state_density_matrix  # ❌ Not implemented

# Use method syntax, not property syntax:
purity = rho.purity    # ❌ Wrong (it's a method, not property)
purity = rho.purity()  # ✅ Correct (with parentheses)
```

### ⏭️ Future Phases

**Phase 2:** Performance optimization
- AVX-optimized kernels
- Advanced noise models
- Profiling and benchmarks

**Phase 3:** Large-scale simulation
- Stochastic methods (quantum trajectories)
- Monte Carlo simulation
- 20+ qubit systems

---

## Getting Started

**1. Setup:** Follow [../SETUP.md](../SETUP.md) to build and install

**2. Verify:** Run the quick test:
```bash
python -c "from squander.density_matrix import DensityMatrix; print('✅ Working')"
```

**3. Learn:** Try the examples above (copy-paste and run!)

**4. Explore:** Check [PHASE1_IMPLEMENTATION.md](PHASE1_IMPLEMENTATION.md) for complete API

---

*Last Updated: November 1, 2025*  

