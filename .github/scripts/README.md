# Build Scripts

This directory contains scripts for building and packaging the squander package.

## build_wheel.sh

Creates a minimal conda environment, builds a wheel, and fixes it with auditwheel (Linux only).

### Prerequisites

- Conda (Miniconda or Anaconda)
- Linux system (for auditwheel)
- System packages (if not installed via conda):
  - `cmake` (>=3.10.2)
  - `libtbb-dev` (Intel TBB)
  - `libopenblas-dev` (BLAS/LAPACK)
  - `liblapacke-dev` (LAPACKE)
  - `patchelf` (required by auditwheel)

### Usage

```bash
# Basic usage (uses default Python 3.10)
./.github/scripts/build_wheel.sh

# Custom Python version
PYTHON_VERSION=3.11 ./.github/scripts/build_wheel.sh

# Custom environment name
ENV_NAME=my-build-env ./.github/scripts/build_wheel.sh

# Both custom
PYTHON_VERSION=3.11 ENV_NAME=squander-py311 ./.github/scripts/build_wheel.sh
```

### What it does

1. Creates a fresh conda environment with the specified Python version
2. Installs build dependencies (cmake, TBB, BLAS, LAPACK, patchelf)
3. Installs Python build tools (build, setuptools, wheel, scikit-build, auditwheel)
4. Builds the wheel using `python -m build --wheel`
5. Repairs the wheel with `auditwheel repair` to bundle dependencies
6. Outputs the repaired wheel to `dist/` directory

### Output

The script creates:
- Original wheel: `dist/squander-<version>-<platform>.whl`
- Repaired wheel: `dist/squander-<version>-manylinux_<version>_<arch>.whl`

The repaired wheel contains all necessary shared library dependencies bundled inside.

### Environment Variables

- `ENV_NAME`: Conda environment name (default: `squander-build`)
- `PYTHON_VERSION`: Python version to use (default: `3.10`)
- `CONDA_BASE`: Base conda directory (auto-detected)

### Troubleshooting

**Error: "Cannot find required utility `patchelf`"**
- Install patchelf: `sudo apt-get install patchelf` or `conda install -c conda-forge patchelf`

**Error: "TBB not found"**
- Set `TBB_LIB_DIR` and `TBB_INC_DIR` environment variables before running
- Or install TBB via conda: `conda install -c conda-forge tbb-devel`

**Error: "BLAS not found"**
- Install OpenBLAS: `sudo apt-get install libopenblas-dev` or `conda install -c conda-forge openblas`

**auditwheel fails**
- Make sure you're on Linux (auditwheel is Linux-only)
- For macOS, use `delocate` instead
- For Windows, use `delvewheel` instead

### Notes

- The script cleans previous builds before building
- The conda environment is removed and recreated each time (ensures clean build)
- System packages are preferred if available, conda packages are used as fallback

