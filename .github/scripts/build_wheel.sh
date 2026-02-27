#!/bin/bash
# Build wheel script for squander package
# Creates a minimal conda environment, builds the wheel, and fixes it with auditwheel

set -e  # Exit on error

# Configuration
ENV_NAME="${ENV_NAME:-squander-build}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on Linux (auditwheel is Linux-only)
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    log_error "This script is designed for Linux. auditwheel only works on Linux."
    log_info "For macOS, use 'delocate' instead."
    log_info "For Windows, use 'delvewheel' instead."
    exit 1
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    log_error "conda is not installed or not in PATH"
    exit 1
fi

log_info "Building wheel for squander package"
log_info "Project root: ${PROJECT_ROOT}"
log_info "Python version: ${PYTHON_VERSION}"
log_info "Environment name: ${ENV_NAME}"

# Activate conda base
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    log_warn "Environment ${ENV_NAME} already exists. Removing it..."
    conda env remove -n "${ENV_NAME}" -y
fi

# Create minimal conda environment
log_info "Creating conda environment: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# Activate environment
log_info "Activating environment: ${ENV_NAME}"
conda activate "${ENV_NAME}"

# Install system dependencies via conda-forge (if available) or apt
log_info "Installing system dependencies..."

# Try to install via conda first (works in conda environments)
if conda install -c conda-forge cmake tbb-devel openblas lapack patchelf -y 2>/dev/null; then
    log_info "Installed dependencies via conda"
else
    log_warn "Could not install all dependencies via conda, trying system packages..."
    # Check if we can use apt (requires sudo)
    if command -v apt-get &> /dev/null && [ "$EUID" -eq 0 ]; then
        apt-get update -qq
        apt-get install -y --no-install-recommends cmake libtbb-dev libopenblas-dev liblapacke-dev patchelf
    else
        log_warn "Cannot install system packages (need root or apt-get not available)"
        log_info "Please ensure the following are installed:"
        log_info "  - cmake (>=3.10.2)"
        log_info "  - libtbb-dev (TBB libraries)"
        log_info "  - libopenblas-dev (BLAS/LAPACK)"
        log_info "  - liblapacke-dev (LAPACKE)"
        log_info "  - patchelf (for auditwheel)"
    fi
fi

# Install Python build dependencies
log_info "Installing Python build dependencies..."
pip install --upgrade pip
pip install build setuptools wheel scikit-build

# Install auditwheel
log_info "Installing auditwheel..."
pip install auditwheel

# Set environment variables for build
export TBB_LIB_DIR="${CONDA_PREFIX}/lib"
export TBB_INC_DIR="${CONDA_PREFIX}/include"

# If TBB not found in conda, try to find system TBB
if [ ! -d "${TBB_LIB_DIR}" ] || [ -z "$(find "${TBB_LIB_DIR}" -name '*tbb*' 2>/dev/null)" ]; then
    log_warn "TBB not found in conda environment, searching system paths..."
    if [ -d "/usr/lib/x86_64-linux-gnu" ]; then
        export TBB_LIB_DIR="/usr/lib/x86_64-linux-gnu"
        export TBB_INC_DIR="/usr/include"
        log_info "Using system TBB: ${TBB_LIB_DIR}"
    fi
fi

# Change to project root
cd "${PROJECT_ROOT}"

# Clean previous builds
log_info "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info _skbuild/

# Build the wheel
log_info "Building wheel..."
python -m build --wheel

# Check if wheel was created
WHEEL_FILE=$(find dist -name "*.whl" -type f | head -n 1)
if [ -z "${WHEEL_FILE}" ]; then
    log_error "No wheel file found in dist/"
    exit 1
fi

log_info "Wheel created: ${WHEEL_FILE}"

# Fix wheel with auditwheel
log_info "Fixing wheel with auditwheel..."
auditwheel repair "${WHEEL_FILE}" -w dist/

# List repaired wheels
log_info "Repaired wheels:"
ls -lh dist/*.whl

# Show auditwheel info for the repaired wheel
REPAIRED_WHEEL=$(find dist -name "*manylinux*.whl" -type f | head -n 1)
if [ -n "${REPAIRED_WHEEL}" ]; then
    log_info "Checking repaired wheel: ${REPAIRED_WHEEL}"
    auditwheel show "${REPAIRED_WHEEL}"
    log_info "${GREEN}Success! Repaired wheel: ${REPAIRED_WHEEL}${NC}"
else
    log_warn "No manylinux wheel found. Original wheel may not have been repaired."
    log_info "Original wheel: ${WHEEL_FILE}"
fi

log_info "Build complete!"
log_info "Wheels are in: ${PROJECT_ROOT}/dist/"

