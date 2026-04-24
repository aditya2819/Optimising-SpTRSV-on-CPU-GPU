#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/cpu"
BINARY="$BUILD_DIR/sptrsv_bench"
MATRIX="${1:-web-Stanford}"
VTUNE_BIN="${VTUNE_BIN:-/opt/intel/oneapi/vtune/2025.10/bin64/vtune}"
ITERS="${ITERS:-500}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/generated/cpu/vtune_profiles}"

KERNELS=(
  basic
  branchred
  unrolled
  thresholdomp
  chunkedomp
  capchunkomp
  mkl-solve
)

cd "$REPO_ROOT"

if [ ! -x "$VTUNE_BIN" ]; then
  echo "VTune binary not found: $VTUNE_BIN" >&2
  exit 1
fi

"$SCRIPT_DIR/run_benchmark.sh" family-one random 1024 basic > /dev/null

rm -rf "$RESULT_DIR"
mkdir -p "$RESULT_DIR"

for kernel in "${KERNELS[@]}"; do
  safe_kernel="${kernel//-/_}"
  result_dir="$RESULT_DIR/vtune_${MATRIX//-/_}_${safe_kernel}"

  rm -rf "$result_dir"
  "$VTUNE_BIN" -collect hotspots -result-dir "$result_dir" -- "$BINARY" real-profile "$MATRIX" "$kernel" "$ITERS" > /dev/null
  "$VTUNE_BIN" -report summary -result-dir "$result_dir" > "${result_dir}_summary.txt"
  "$VTUNE_BIN" -report hotspots -result-dir "$result_dir" > "${result_dir}_hotspots.txt"
done

echo "Wrote VTune outputs under:"
echo "  $RESULT_DIR"
