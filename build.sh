#!/usr/bin/env bash
set -euo pipefail

# Build exllamav3 optimized for native architecture (sm_89) with CUDA 13.4
# Maximum performance configuration

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# CUDA 13.4 toolkit path
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.4}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"

# Target architecture: sm_89 (Ada Lovelace / RTX 40-series)
export TORCH_CUDA_ARCH_LIST="8.9"

# Host compiler path for nvcc (flags are in CXXFLAGS)
export CUDAHOSTCXX="$(which g++)"

# C++ compiler flags for host code
export CXXFLAGS="-march=native -O3 -funroll-loops -flto -pipe -Wno-unused-result"
export CFLAGS="-O3 -pipe -Wno-unused-result"

# Force PyTorch to use CUDA 13.4 nvcc (system nvcc on PATH may be different version)
export PYTORCH_NVCC="${CUDA_HOME}/bin/nvcc"

# nvcc flags for maximum GPU kernel performance
export TORCH_NVCC_FLAGS="-Xptxas=-O3 -Xptxas=-maxrregcount=0 -Xcompiler=-fPIC"

# Use all available cores for parallel build
export MAX_JOBS=10

echo "=== Building exllamav3 (max performance) ==="
echo "CUDA_HOME:      $CUDA_HOME"
echo "CUDA_PATH:      $CUDA_PATH"
echo "Target arch:    sm_89 (compute_89)"
echo "Host compiler:  $CUDAHOSTCXX"
echo "CXXFLAGS:       $CXXFLAGS"
echo "TORCH_NVCC:     $TORCH_NVCC_FLAGS"
echo "Max jobs:       $MAX_JOBS"
echo ""

# Verify CUDA toolkit
if [ ! -d "$CUDA_HOME" ]; then
    echo "ERROR: CUDA toolkit not found at $CUDA_HOME"
    echo "Set CUDA_HOME to the CUDA 13.4 installation path"
    exit 1
fi

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: nvcc not found at $CUDA_HOME/bin/nvcc"
    exit 1
fi

echo "nvcc version:"
"$CUDA_HOME/bin/nvcc" --version | head -1
echo ""

echo "g++ version:"
g++ --version | head -1
echo ""

# Full rebuild requested via ./build.sh --full ?
FULL_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --full) FULL_BUILD=1 ;;
    esac
done

# Ensure build directory exists (persistent object files for incremental builds)
mkdir -p build

# Python 3.12 virtual environment
PYTHON="python3.12"
VENV_DIR=".venv"

# Install Python dev headers if missing (requires sudo)
if ! dpkg -s python3.12-dev &>/dev/null; then
    echo "WARNING: python3.12-dev is not installed. Please run:"
    echo "  sudo apt-get install -y python3.12-dev"
    echo "Then re-run this script."
    exit 1
fi
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python 3.12 virtual environment in $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip inside venv
$PYTHON -m pip install --upgrade pip setuptools wheel ninja

# Build strategy:
# - "pip install -e ." runs setuptools' PEP 660 editable build, which compiles
#   into a throwaway temp dir on every run (build_editable -> TemporaryDirectory),
#   so it is always a from-scratch build. It only needs to run once: the editable
#   install is just a .pth pointing at this source tree, and exllamav3/ext.py
#   imports the top-level exllamav3_ext module resolved from the project root.
# - "setup.py build_ext --inplace" compiles into the persistent build/temp.* dir
#   (ninja keeps its log/deps there), so only changed TUs are recompiled, and
#   writes the .so in place at the project root where the editable install loads it.
if [ "$FULL_BUILD" -eq 1 ] || ! $PYTHON -m pip show exllamav3 &>/dev/null; then
    echo "Full editable install (from-scratch build)..."
    $PYTHON -m pip install -e . --no-build-isolation -v \
        --config-settings editable-verbose=true \
        2>&1 | tee build.log
else
    echo "Incremental build (build_ext --inplace)..."
    $PYTHON setup.py build_ext --inplace 2>&1 | tee build.log
fi

echo ""
echo "=== Build complete ==="
echo "Log: build.log"