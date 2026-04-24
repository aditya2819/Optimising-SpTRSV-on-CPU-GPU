#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$REPO_ROOT/src/gpu"
BUILD_DIR="$REPO_ROOT/build/gpu"
BINARY="$BUILD_DIR/sptrsv_bench"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
CUDA_LIB="${CUDA_HOME}/targets/x86_64-linux/lib"
CUDA_ARCH="${CUDA_ARCH:-89}"

mkdir -p "$BUILD_DIR"

"$NVCC" -O3 -std=c++17 \
    -diag-suppress 177 \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    -I"$CUDA_HOME/include" \
    "$SRC_DIR/sptrsv_benchmark.cu" \
    "$SRC_DIR/sptrsv_kernel.cu" \
    -L"$CUDA_LIB" \
    -lcusparse -lcudart \
    -o "$BINARY"

cd "$REPO_ROOT"
"$BINARY" "$@"
