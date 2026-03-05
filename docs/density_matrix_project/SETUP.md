# Density Matrix Setup and Verification

This guide covers installation and validation for the density matrix module on
`feature/density-matrix-phase1`.

Primary audience: developers and users who need to build and run the module.

## Prerequisites

- Python 3.10+ (3.13 tested on this branch)
- CMake >= 3.15
- C++11-capable compiler (for example GCC >= 4.8.1, Intel >= 14.0.1, or MSVC equivalent)
- Conda (recommended)
- Git

## 1) Create Environment

```bash
conda create -n qgd python=3.13 -c conda-forge -y
conda activate qgd
```

## 2) Install Dependencies

```bash
# Core build/runtime dependencies
conda install -y numpy scipy pytest scikit-build tbb-devel ninja cmake -c conda-forge

# Optional validation dependencies (Qiskit comparison scripts)
conda install -y qiskit qiskit-aer matplotlib networkx -c conda-forge

# pybind11 is required by the density matrix module build
pip install pybind11
```

## 3) Build and Install

```bash
cd /path/to/sequential-quantum-gate-decomposer
git checkout feature/density-matrix-phase1

# Required in many conda-based setups
export TBB_INC_DIR=~/.conda/envs/qgd/include
export TBB_LIB_DIR=~/.conda/envs/qgd/lib

# Clean and build
rm -rf _skbuild
python setup.py build_ext
python -m pip install -e .
```

## 4) Quick Verification

```bash
python - << 'EOF'
from squander.density_matrix import DensityMatrix, NoisyCircuit
import numpy as np

rho = DensityMatrix(qbit_num=2)
circuit = NoisyCircuit(2)
circuit.add_H(0)
circuit.add_CNOT(1, 0)
circuit.apply_to(np.array([]), rho)

print("Density matrix module import and execution: OK")
print("Purity:", rho.purity())
print("Entropy:", rho.entropy())
EOF
```

Expected:
- import succeeds,
- circuit execution succeeds,
- purity is `1.0` for this unitary-only circuit.

## 5) Run Tests

Always activate the same environment first:

```bash
conda activate qgd
```

Python tests:

```bash
pytest tests/density_matrix/ -v
```

Examples:

```bash
python examples/density_matrix/basic_usage.py
```

Optional C++ tests:

```bash
export QGD_CTEST=1
export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"

python setup.py build_ext
./test_standalone/test_density_matrix_cpp

unset LDFLAGS LIBRARY_PATH
```

## Troubleshooting

### `pybind11` not found

Symptom:
- CMake reports pybind11 missing and skips the density matrix module.

Fix:

```bash
conda activate qgd
pip install pybind11
python setup.py build_ext
```

### TBB header/library not found

Symptom:
- missing `tbb/...` headers or link errors.

Fix:

```bash
conda install -y tbb-devel -c conda-forge
export TBB_INC_DIR=~/.conda/envs/qgd/include
export TBB_LIB_DIR=~/.conda/envs/qgd/lib
rm -rf _skbuild
python setup.py build_ext
```

### `ModuleNotFoundError: No module named 'squander.density_matrix'`

Fix:

```bash
python setup.py build_ext
python -m pip install -e .
```

Then verify that a built extension exists:

```bash
ls squander/density_matrix/_density_matrix_cpp*.so
```

### Qiskit install issues on Python 3.13

Prefer conda-forge packages instead of pip wheels:

```bash
conda install -y qiskit qiskit-aer -c conda-forge
```

## Next Documents

- Project overview and roadmap: [`README.md`](README.md)
- API details: [`API_REFERENCE.md`](API_REFERENCE.md)
- Architecture details: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- External context: [`RESEARCH_ALIGNMENT.md`](RESEARCH_ALIGNMENT.md)
- Delivered and planned work: [`CHANGELOG.md`](CHANGELOG.md)

