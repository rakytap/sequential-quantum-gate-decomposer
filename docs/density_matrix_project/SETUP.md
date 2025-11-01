# Phase 1 Density Matrix - Setup & Testing Guide

**For:** Users wanting to build and test the density matrix module  
**Branch:** `feature/density-matrix-phase1`  
**Python:** 3.13 (also compatible with 3.8-3.12)

---

## 📋 Prerequisites

- Anaconda or Miniconda installed
- CMake >= 3.15
- C++17 compiler (GCC >= 7, Clang >= 5, MSVC 2017+)
- Git

---

## 🚀 Quick Setup (5 Minutes)

### 1. Create Conda Environment

```bash
conda create -n qgd python=3.13 -c conda-forge -y
conda activate qgd
```

### 2. Install Dependencies

```bash
# Core dependencies (required)
conda install -y numpy scipy pytest scikit-build tbb-devel ninja cmake -c conda-forge

# Additional packages (optional, for integration tests)
conda install -y qiskit qiskit-aer matplotlib networkx -c conda-forge

# pybind11 (required for density matrix module)
pip install pybind11
```

### 3. Clone and Build

```bash
cd /path/to/sequential-quantum-gate-decomposer
git checkout feature/density-matrix-phase1

# Set TBB paths (required)
export TBB_INC_DIR=~/.conda/envs/qgd/include
export TBB_LIB_DIR=~/.conda/envs/qgd/lib

# Build
rm -rf _skbuild
python setup.py build_ext
python -m pip install -e .
```

### 4. Verify Installation

```bash
python << 'EOF'
from squander.density_matrix import DensityMatrix, DensityCircuit
import numpy as np

rho = DensityMatrix(qbit_num=2)
circuit = DensityCircuit(2)
circuit.add_H(0)
circuit.add_CNOT(1, 0)
circuit.apply_to(np.array([]), rho)

print(f"✅ Density Matrix Module Working!")
print(f"   Purity: {rho.purity():.6f}")
print(f"   Entropy: {rho.entropy():.6f}")
EOF
```

**Expected output:**
```
✅ Density Matrix Module Working!
   Purity: 1.000000
   Entropy: 0.000000
```

---

## 🧪 Running Tests

### Python Tests (Comprehensive)

```bash
# All density matrix tests (22 tests, ~1 second)
pytest tests/density_matrix/ -v

# Expected: 22 passed
```

### Example Scripts

```bash
# Run comprehensive examples
python examples/density_matrix/basic_usage.py

# Expected: 5 examples, all pass
```

### C++ Tests (Optional)

If you want to run C++ unit tests:

```bash
# Build with testing enabled
export QGD_CTEST=1
export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
python setup.py build_ext

# Run C++ tests
./test_standalone/test_density_matrix_cpp
```

> These linker environment variables ensure the build picks up the conda-provided
> `libstdc++`/TBB libraries. You can unset them afterwards if you do not need
> them globally (`unset LDFLAGS LIBRARY_PATH`).

---

## 🔧 Troubleshooting

### Issue: pybind11 not found

**Error:**
```
pybind11 not found. Install with: pip install pybind11
```

**Solution:**
```bash
conda activate qgd
pip install pybind11
```

### Issue: TBB headers not found

**Error:**
```
fatal error: tbb/scalable_allocator.h: No such file or directory
```

**Solution:**
```bash
# Install TBB
conda install tbb-devel -c conda-forge

# Set environment variables
export TBB_INC_DIR=~/.conda/envs/qgd/include
export TBB_LIB_DIR=~/.conda/envs/qgd/lib

# Rebuild
rm -rf _skbuild
python setup.py build_ext
```

### Issue: Cannot import density_matrix

**Error:**
```
ModuleNotFoundError: No module named 'squander.density_matrix'
```

**Solution:**
```bash
# Ensure module was built
ls squander/density_matrix/_density_matrix_cpp*.so

# If not found, rebuild:
python setup.py build_ext

# Reinstall in editable mode
python -m pip install -e .
```

### Issue: Qiskit installation fails (Python 3.13)

**Error:**
```
ValueError: `py_limited_api='cp39'` not supported
```

**Solution:**
```bash
# Install from conda-forge (not pip)
conda install qiskit qiskit-aer -c conda-forge
```

---

## 📊 Test Results

After successful setup, you should see:

**Pytest:**
```
tests/density_matrix/test_density_matrix.py ... 22 passed in 0.43s
```

**Example script:**
```
Example 1: Pure State Evolution ✅
Example 2: Noise Simulation ✅
Example 3: T1 and T2 Noise ✅
Example 4: Maximally Mixed State ✅
Example 5: Partial Trace ✅

All examples completed successfully!
```

---

## 🎓 Next Steps

**After successful setup, you can:**

1. **Try examples:** See [phase1-isolated/README.md](phase1-isolated/README.md) for working code examples
2. **Learn the API:** Check [phase1-isolated/PHASE1_IMPLEMENTATION.md](phase1-isolated/PHASE1_IMPLEMENTATION.md) for complete API reference
3. **Understand design:** Read [phase1-isolated/PHASE1_DESIGN.md](phase1-isolated/PHASE1_DESIGN.md) for design rationale
4. **See roadmap:** Review [DENSITY_MATRIX_PROJECT_README.md](DENSITY_MATRIX_PROJECT_README.md) for the full project vision

---

## 📦 Package Information

**SQUANDER:** v1.9.3  
**Density Matrix Module:** v1.0.0 (Phase 1)  
**Python Compatibility:** 3.8-3.13  
**Build System:** CMake 3.15+ with scikit-build  
**Bindings:** pybind11 3.0+

---

*Last Updated: November 1, 2025*  
