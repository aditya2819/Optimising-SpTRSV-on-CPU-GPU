#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$REPO_ROOT/src/cpu"
BUILD_DIR="$REPO_ROOT/build/cpu"
BINARY="$BUILD_DIR/sptrsv_bench"
CXX="${CXX:-g++}"
MKLROOT="${MKLROOT:-/opt/intel/oneapi/mkl/latest}"

if [ -f "$MKLROOT/include/mkl.h" ]; then
    include_flags="-I$MKLROOT/include"
    compiler_rt_flags=""
    if [ -n "${CMPLR_ROOT:-}" ]; then
        compiler_rt_flags="-L$CMPLR_ROOT/lib -L$CMPLR_ROOT/opt/compiler/lib"
    fi
    link_flags="$compiler_rt_flags -L$MKLROOT/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl"
elif [ -f "/usr/include/mkl/mkl.h" ]; then
    include_flags="-I/usr/include/mkl"
    link_flags="-L/usr/lib/x86_64-linux-gnu -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
else
    echo "MKL headers not found." >&2
    echo "Checked:" >&2
    echo "  $MKLROOT/include/mkl.h" >&2
    echo "  /usr/include/mkl/mkl.h" >&2
    echo "Set MKLROOT to your oneAPI MKL installation if needed." >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

"$CXX" -O3 -fopenmp -std=c++17 \
    $include_flags \
    "$SRC_DIR/sptrsv_kernel.cpp" \
    "$SRC_DIR/sptrsv_benchmark.cpp" \
    $link_flags \
    -o "$BINARY"

cd "$REPO_ROOT"
"$BINARY" "$@"
