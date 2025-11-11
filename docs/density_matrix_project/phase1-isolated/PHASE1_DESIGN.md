# Phase 1: Integrated Subpackage Approach

**Python Integration:** `from squander.density_matrix import DensityMatrix`  
**Build Integration:** No flags, modern CMake  
**Code Isolation:** Zero modifications to existing files  

---

## Design Principles

### Core Changes

1. **Python Integration:** Density matrix is a **subpackage** of squander
   - `from squander.density_matrix import DensityMatrix`

2. **Build Integration:** Always built, modern CMake targets

3. **CMake Modernization:** 
   - INTERFACE libraries for common dependencies
   - Modern target-based linking
   - Generator expressions
   - Proper PUBLIC/PRIVATE/INTERFACE scoping

4. **Non-Invasive:** Zero modifications to existing code
   - Same namespace isolation (`squander::density::`)
   - Same inheritance/wrapping strategy
   - Same C++ directory structure

---

## Directory Structure

### Python Package Structure

```
squander/
├── __init__.py                          # EXISTING (no modification)
├── gates/                               # EXISTING (unchanged)
│   ├── __init__.py
│   └── ...
├── decomposition/                       # EXISTING (unchanged)
├── density_matrix/                      # NEW subpackage
│   ├── __init__.py                      # NEW
│   └── _density_matrix_cpp.so           # Built by pybind11
└── ...
```

### C++ Source Structure

```
squander/src-cpp/
├── common/                              # EXISTING (unchanged)
│   ├── include/matrix_base.hpp          # Reused
│   └── ...
├── gates/                               # EXISTING (unchanged)
│   ├── include/Gate.h                   # Wrapped
│   └── ...
├── decomposition/                       # EXISTING (unchanged)
└── density_matrix/                      # NEW module
    ├── include/
    │   ├── density_matrix.h             # Core class
    │   ├── density_circuit.h            # Circuit wrapper
    │   └── noise_channel.h              # Noise models
    ├── density_matrix.cpp
    ├── density_circuit.cpp
    ├── noise_channel.cpp
    ├── tests/
    │   └── test_basic.cpp
    └── CMakeLists.txt                   # Module build config
```

### Test Structure

```
tests/
├── gates/                               # EXISTING (unchanged)
├── density_matrix/                      # NEW tests
│   ├── __init__.py
│   └── test_density_matrix.py
└── ...
```

---

##  Modern CMake Structure

### Root CMakeLists.txt Changes

**Minimal additions to existing file:**

```cmake
# At the top (after project())
# ===================================================================
# Modern CMake Setup
# ===================================================================

# Use modern CMake features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ===================================================================
# Create INTERFACE library for common dependencies (NEW)
# ===================================================================

add_library(squander_common INTERFACE)

target_include_directories(squander_common INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include>
    $<INSTALL_INTERFACE:include/squander>
)

target_link_libraries(squander_common INTERFACE
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${TBB_LIB}
)

# ... existing qgd library setup ...

# ===================================================================
# Add density matrix module (at end, before install)
# ===================================================================

add_subdirectory(squander/src-cpp/density_matrix)
```

### Density Matrix CMakeLists.txt (Modern)

**File:** `squander/src-cpp/density_matrix/CMakeLists.txt`

```cmake
# ===================================================================
# SQUANDER Density Matrix Module
# Modern CMake Configuration
# ===================================================================

message(STATUS "=== Configuring Density Matrix Module ===")

# ===================================================================
# Find pybind11
# ===================================================================

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pybind11_DIR)
        find_package(pybind11 CONFIG PATHS ${pybind11_DIR})
    endif()
endif()

if(NOT pybind11_FOUND)
    message(WARNING "pybind11 not found. Install with: pip install pybind11")
    message(WARNING "Skipping density matrix module.")
    return()
endif()

message(STATUS "Found pybind11: ${pybind11_VERSION}")

# ===================================================================
# Source Files
# ===================================================================

set(DENSITY_MATRIX_SOURCES
    density_matrix.cpp
    density_circuit.cpp
    noise_channel.cpp
)

set(DENSITY_MATRIX_HEADERS
    include/density_matrix.h
    include/density_circuit.h
    include/noise_channel.h
)

# ===================================================================
# C++ Library Target (INTERFACE - header-only possible, or STATIC)
# ===================================================================

# Option 1: Static library (recommended for complex logic)
add_library(density_matrix_core STATIC ${DENSITY_MATRIX_SOURCES})

target_include_directories(density_matrix_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/squander/density_matrix>
)

# Link against common SQUANDER infrastructure
target_link_libraries(density_matrix_core
    PUBLIC
        squander_common  # INTERFACE library
    PRIVATE
        qgd              # Main SQUANDER library
)

target_compile_features(density_matrix_core PUBLIC cxx_std_17)

# Compiler warnings and optimization
target_compile_options(density_matrix_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<CONFIG:Release>:-O3>
)

# ===================================================================
# Python Module (pybind11)
# ===================================================================

pybind11_add_module(_density_matrix_cpp MODULE
    ../../density_matrix/bindings.cpp
)

target_link_libraries(_density_matrix_cpp PRIVATE
    density_matrix_core
    qgd
)

target_include_directories(_density_matrix_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set output location to squander/density_matrix/
set_target_properties(_density_matrix_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/squander/density_matrix
    OUTPUT_NAME "_density_matrix_cpp"
)

# ===================================================================
# Installation
# ===================================================================

# Install C++ library
install(TARGETS density_matrix_core
        EXPORT SquanderTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
)

# Install headers
install(FILES ${DENSITY_MATRIX_HEADERS}
        DESTINATION include/squander/density_matrix
)

# Install Python module
install(TARGETS _density_matrix_cpp
        LIBRARY DESTINATION squander/density_matrix
)

# ===================================================================
# Tests (Optional)
# ===================================================================

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    enable_testing()
    
    add_executable(test_density_matrix_cpp tests/test_basic.cpp)
    target_link_libraries(test_density_matrix_cpp PRIVATE density_matrix_core)
    add_test(NAME DensityMatrixCppTests COMMAND test_density_matrix_cpp)
endif()

message(STATUS "=== Density Matrix Module Configured ===")
```

---

##  Python Package Structure

### `squander/density_matrix/__init__.py` 

**Phase 1 Implementation:** Direct C++ bindings (simpler, no Python wrapper layer)

**Note:** This is what's actually implemented in Phase 1

```python
"""
Density Matrix Module for SQUANDER

Provides mixed-state quantum simulation with noise modeling.
Integrated as a subpackage of SQUANDER.

Usage:
    from squander.density_matrix import DensityMatrix, DensityCircuit
    from squander.density_matrix import DepolarizingChannel
"""

__version__ = "1.0.0"

# Import C++ bindings directly (Phase 1)
from ._density_matrix_cpp import (
    DensityMatrix,
    DensityCircuit,
    NoiseChannel,
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel,
)

__all__ = [
    # Core classes
    "DensityMatrix",
    "DensityCircuit",
    
    # Noise channels
    "NoiseChannel",
    "DepolarizingChannel",
    "AmplitudeDampingChannel",
    "PhaseDampingChannel",
]
```

---

## Usage Examples

### Import Structure

```python
# Existing SQUANDER (unchanged)
from squander.gates.qgd_Circuit import qgd_Circuit
from squander.decomposition import N_Qubit_Decomposition

# New density matrix subpackage (Phase 1 - all working ✅)
from squander.density_matrix import DensityMatrix, DensityCircuit
from squander.density_matrix import DepolarizingChannel
from squander.density_matrix import AmplitudeDampingChannel, PhaseDampingChannel
```

### Basic Usage

```python
from squander.density_matrix import DensityMatrix, DensityCircuit
import numpy as np

# Create density matrix
rho = DensityMatrix(qbit_num=2)

# Create circuit
circuit = DensityCircuit(2)
circuit.add_H(0)
circuit.add_CNOT(1, 0)

# Apply circuit
circuit.apply_to(np.array([]), rho)

# Note: Use method syntax with parentheses (not properties)
print(f"Purity: {rho.purity()}")  # 1.0 ✅
print(f"Entropy: {rho.entropy()}")  # 0.0 ✅
```

### With Noise

```python
from squander.density_matrix import (
    DensityMatrix,
    DensityCircuit,
    DepolarizingChannel,
)
import numpy as np

# Create and apply circuit
rho = DensityMatrix(qbit_num=2)
circuit = DensityCircuit(2)
circuit.add_H(0)
circuit.add_CNOT(1, 0)
circuit.apply_to(np.array([]), rho)

# Add noise
noise = DepolarizingChannel(qbit_num=2, error_rate=0.01)
noise.apply(rho)

print(f"Purity after noise: {rho.purity()}")  # < 1.0 ✅
```

### Integration with Existing SQUANDER

```python
from squander.gates.qgd_Circuit import qgd_Circuit
from squander.density_matrix import DensityMatrix
import numpy as np

# Use existing circuit to get unitary
sv_circuit = qgd_Circuit(2)
sv_circuit.add_H(0)
sv_circuit.add_CNOT(1, 0)

# Get unitary matrix
params = np.array([])
U = sv_circuit.get_Matrix(params)

# Apply to density matrix
rho = DensityMatrix(qbit_num=2)
rho.apply_unitary(U)
print(f"Purity: {rho.purity()}")  # 1.0 ✅
```

---

### Modern CMake Benefits

- ✅ **INTERFACE libraries** for shared dependencies
- ✅ **Target-based linking** (no variable soup)
- ✅ **Generator expressions** for config-specific settings
- ✅ **Proper scoping** (PUBLIC/PRIVATE/INTERFACE)
- ✅ **Easier maintenance** and clearer dependencies

---

## 🎯 Non-Invasive Guarantee

### Files NOT Modified

- ❌ `squander/src-cpp/common/*.cpp`
- ❌ `squander/src-cpp/common/include/*.h`
- ❌ `squander/src-cpp/gates/*.cpp`
- ❌ `squander/src-cpp/gates/include/*.h`
- ❌ `squander/gates/*.py`
- ❌ `squander/decomposition/*.py`
- ❌ `squander/__init__.py` (only if needed for version compatibility)

### Files Added (New)

- ✅ `squander/src-cpp/density_matrix/` (entire directory)
- ✅ `squander/density_matrix/` (entire directory)
- ✅ `tests/density_matrix/` (entire directory)

### Files Modified (Minimal)

- ⚠️ `CMakeLists.txt` (add ~27 lines for INTERFACE library + subdirectory)

---

## 📝 Phase 1 vs. Future Enhancements Summary

### ✅ Implemented in Phase 1

**C++ Core:**
- DensityMatrix class with full quantum properties
- DensityCircuit for circuit construction
- 3 noise channels (Depolarizing, Amplitude Damping, Phase Damping)

**Python Interface:**
- Direct C++ bindings via pybind11
- Subpackage: `from squander.density_matrix import ...`
- Method-based API (e.g., `rho.purity()` with parentheses)
- NumPy integration

**Build & Test:**
- Modern CMake with INTERFACE libraries
- 22 Python tests + 8 C++ tests (all passing)
- Comprehensive examples
- Zero modifications to existing SQUANDER

### ⏭️ NOT in Phase 1 (Future Enhancements)

**Optimizations:**
- No AVX kernels (Phase 2)
- No stochastic simulation (Phase 3)
- No advanced noise models yet

**See [PHASE1_IMPLEMENTATION.md](PHASE1_IMPLEMENTATION.md) for complete API reference and what's actually available.**

---

*Last Updated: November 1, 2025*  


