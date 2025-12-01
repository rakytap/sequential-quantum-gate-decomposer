[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4508680.svg)](https://doi.org/10.5281/zenodo.4508680)
<a href="https://trackgit.com">
<img src="https://us-central1-trackgit-analytics.cloudfunctions.net/token/ping/l0sfey1m19at85951dwl" alt="trackgit-views" />
</a>

<div align="center">

![SQUANDER Logo](Doxygen/images/layers.png)

# Sequential Quantum Gate Decomposer (SQUANDER)

**A High-Performance C++/Python Library for Quantum Circuit Decomposition and Optimization**

[Installation](#installation) • [Documentation](#documentation) • [Features](#features) • [Citation](#citation)

</div>

---

## Overview

The **Sequential Quantum Gate Decomposer (SQUANDER)** is a state-of-the-art computational library designed for training parametric quantum circuits and performing quantum gate synthesis. SQUANDER provides a comprehensive Python interface that enables researchers and developers to conduct advanced numerical experiments in quantum computing, including quantum gate synthesis, variational quantum eigensolver (VQE) applications, and quantum state preparation.

### Key Capabilities

SQUANDER excels in decomposing n-qubit unitaries into sequences of one-qubit rotations and two-qubit controlled gates using advanced synthesis methods. The library leverages a high-performance parallel C/C++ framework with vectorized AVX gate kernels, delivering exceptional computational efficiency for quantum circuit simulations.

### Optimization Techniques

Beyond conventional gradient-based optimizers (gradient descent, ADAM, and BFGS), SQUANDER incorporates an innovative gradient-free optimization technique that demonstrates robust numerical behavior and is particularly effective in circumventing barren plateaus—a common challenge in quantum circuit training. The library's handcrafted optimization strategies are specifically designed to accommodate the periodicity inherent in quantum optimization landscapes, ensuring resilient numerical efficiency.

---

## Installation

SQUANDER is available as pre-built Python wheels for **Windows**, **Linux**, and **macOS**, making installation straightforward for most users. The package can be installed directly from the Python Package Index (PyPI):

```bash
pip install numpy tbb-devel wheel scikit-build ninja qiskit
pip install squander
```

### System Requirements

- **Python**: 3.8 or higher (tested with Python 3.6-3.13)
- **Operating Systems**: Windows, Linux, macOS

### Python Dependencies

The following packages are automatically installed as dependencies:

- [NumPy](https://numpy.org/install/)
- [SciPy](https://www.scipy.org/install.html)
- [Qiskit](https://qiskit.org/documentation/install.html)
- [NetworkX](https://networkx.org/)
- [Matplotlib](https://matplotlib.org/)

---

## Development Installation

For developers who wish to build SQUANDER from source or contribute to the project, the following development installation instructions are provided.

### Prerequisites

The following dependencies are required to compile and build SQUANDER from source:

#### Build Tools
- [CMake](https://cmake.org/) (>=3.10.2)
- C++/C compiler:
  - [Intel Compiler](https://software.intel.com/content/www/us/en/develop/tools/compilers/c-compilers.html) (>=14.0.1), or
  - [GNU Compiler Collection](https://gcc.gnu.org/) (>=v4.8.1)

#### Libraries
- [TBB (Threading Building Blocks)](https://github.com/oneapi-src/oneTBB) library
  - Can be installed via Python package: `pip install tbb-devel`
  - Or system package: `sudo apt install libtbb-dev` (Linux)
- [LAPACKE](https://www.netlib.org/lapack/lapacke.html)
- **BLAS Library** (choose one):
  - [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) (optional)
  - [OpenBLAS](https://www.openblas.net/) (optional, recommended)
  - GNU Scientific Library CBLAS

#### Optional Tools
- [Doxygen](https://www.doxygen.nl/index.html) (for generating documentation)
- [Ninja](https://ninja-build.org/) (speeds up compilation)

### Development Build on Unix/Linux/macOS

#### 1. Clone the Repository

```bash
git clone https://github.com/rakytap/sequential-quantum-gate-decomposer.git
cd sequential-quantum-gate-decomposer
```

#### 2. Set Up Environment Variables (if needed)

If TBB is installed at a non-standard location or using GNU compiler:

```bash
export TBB_LIB_DIR=path/to/TBB/lib(64)
export TBB_INC_DIR=path/to/TBB/include
```

**Note**: When TBB is installed via the `tbb-devel` Python package, these environment variables are not necessary.

#### 3. Build from Source

The SQUANDER package uses a Python build script (`setup.py`) that automatically detects the CBLAS library used by NumPy and configures CMake accordingly:

```bash
python3 setup.py build_ext
```

This command compiles the SQUANDER C++ library and builds the Python interface extensions in place.

#### 4. Install in Development Mode

After a successful build, install the package in editable (development) mode:

```bash
python -m pip install -e .
```

### Development Build with Conda (Recommended)

We recommend using Anaconda/Miniconda for development environments:

#### 1. Create Environment from Configuration File

```bash
conda env create -f conda_env_example.yaml
```

#### 2. Activate Environment

```bash
conda activate qgd
```

#### 3. Build and Install

```bash
python3 setup.py build_ext
python -m pip install -e .
```

### Development Build on Windows

#### Prerequisites

- CMake must be in the system PATH
- Microsoft Visual C++ compiler

#### Build Steps

```cmd
set PATH=%PATH%;C:\Program Files\cmake\bin

set TBB_LOCATION=<Python_Folder>/LocalCache/local-packages
set TBB_INC_DIR=%TBB_LOCATION%/Library/include
set TBB_LIB_DIR=%TBB_LOCATION%/Library/lib
set LIB=<BLAS_Location>/lib;<LAPACK_Location>/lib

python setup.py build_ext -DTBB_HEADER=<TBB_Location>\Library\include\
```

#### Install DLL Files

Copy required DLL files to the package directory:

```cmd
copy _skbuild\win-amd64-3.9\cmake-install\bin .\squander\decomposition
copy "%TBB_LOCATION%\Library\bin\tbb12.dll" .\squander\decomposition
copy "%TBB_LOCATION%\Library\bin\tbbmalloc.dll" .\squander\decomposition
```

#### Verify Installation

```cmd
python -m pytest
```

### Building Distribution Packages

#### Binary Wheel Distribution

To build a wheel binary for distribution:

```bash
python3 setup.py bdist_wheel
```

The wheel will be created in the `dist/` directory. Note that the created wheel is not portable, as it contains hard-coded links to external libraries (TBB and CBLAS).

#### Source Distribution

To create a portable source distribution:

```bash
python3 setup.py sdist
```

The source distribution tarball will be created in the `dist/` directory.

---

## Features

SQUANDER provides a comprehensive suite of tools for quantum circuit manipulation and optimization:

### Core Functionality

- **Unitary Decomposition**: Decompose unitaries into quantum circuits using multiple methods:
  - Standard decomposition (`N_Qubit_Decomposition`)
  - Adaptive decomposition with circuit compression (`N_Qubit_Decomposition_adaptive`)
  - Custom topology decomposition (`N_Qubit_Decomposition_custom`)
  - Tree search decomposition (`N_Qubit_Decomposition_Tree_Search`)
  - Tabu search decomposition (`N_Qubit_Decomposition_Tabu_Search`)

- **Circuit Optimization**: Optimize wide quantum circuits using the `Wide_Circuit_Optimization` class

- **State Preparation**: Prepare quantum states via adaptive state preparation (`N_Qubit_State_Preparation_adaptive`)

- **Variational Quantum Algorithms**:
  - Variational Quantum Eigensolver (VQE) (`Variational_Quantum_Eigensolver`)
  - Generative Quantum Machine Learning (GQML) (`Generative_Quantum_Machine_Learning`)

- **Circuit Simulation**: High-performance state vector evolution for quantum circuit simulation

- **Circuit Synthesis**: SABRE algorithm for qubit routing and mapping

- **Circuit Partitioning**: Partition large circuits for efficient decomposition and optimization

- **Qiskit Integration**: Seamless integration with Qiskit through the `Qiskit_IO` module

---

## Python Interface

SQUANDER exposes its C++ functionality through a comprehensive Python interface. The main modules include:

| Module | Description |
|--------|-------------|
| `squander.decomposition` | Quantum gate decomposition classes for decomposing unitaries and preparing quantum states |
| `squander.gates` | Quantum gate implementations and circuit building blocks (CNOT, H, RX, RY, RZ, and custom gates) |
| `squander.VQA` | Classes for VQE and generative quantum machine learning algorithms |
| `squander.synthesis` | Circuit synthesis tools including SABRE algorithm for qubit routing and mapping |
| `squander.partitioning` | Circuit partitioning utilities for breaking down large circuits into manageable sub-circuits |
| `squander.IO_interfaces` | Input/output interfaces including Qiskit integration (`Qiskit_IO`) |
| `squander.utils` | Utility functions for working with quantum circuits and unitaries |
| `squander.nn` | Experimental neural network interface for quantum machine learning |

### Example Usage

Comprehensive examples demonstrating SQUANDER's capabilities are available in the `examples/` directory:

- **`examples/decomposition/`**: Examples of unitary decomposition with various methods
- **`examples/VQE/`**: Variational quantum eigensolver examples
- **`examples/state_preparation/`**: Quantum state preparation examples
- **`examples/simulation/`**: Quantum circuit simulation benchmarks
- **`examples/partitioning/`**: Circuit partitioning examples

Additional usage patterns and test cases can be found in the `tests/` directory.

---

## Documentation

Comprehensive documentation for the SQUANDER package is available at:

**[CodeDocs[xyz]](https://codedocs.xyz/rakytap/sequential-quantum-gate-decomposer/)**

---

## Citation

If you use SQUANDER in your research, please cite the following publications:

**[1]** Péter Rakyta, Zoltán Zimborás, *Approaching the theoretical limit in quantum gate decomposition*, Quantum **6**, 710 (2022).  
**[2]** Péter Rakyta, Zoltán Zimborás, *Efficient quantum gate decomposition via adaptive circuit compression*, arXiv:2203.04426.  
**[3]** Peter Rakyta, Gregory Morse, Jakab Nádori, Zita Majnay-Takács, Oskar Mencer, Zoltán Zimborás, *Highly optimized quantum circuits synthesized via data-flow engines*, Journal of Computational Physics **500**, 112756 (2024).  
**[4]** Jakab Nádori, Gregory Morse, Barna Fülöp Villám, Zita Majnay-Takács, Zoltán Zimborás, Péter Rakyta, *Batched Line Search Strategy for Navigating through Barren Plateaus in Quantum Circuit Training*, Quantum **9**, 1841 (2025).

---

## Acknowledgments

This project was supported by:
- Grant OTKA PD123927
- The Ministry of Innovation and Technology and the National Research, Development and Innovation Office within the Quantum Information National Laboratory of Hungary

---

## Contact

For questions, support, or collaboration inquiries, please contact:

- **Zoltán Zimborás** (Researcher): zimboras.zoltan@wigner.hu
- **Peter Rakyta** (Developer): peter.rakyta@ttk.elte.hu

---

## License

SQUANDER is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
