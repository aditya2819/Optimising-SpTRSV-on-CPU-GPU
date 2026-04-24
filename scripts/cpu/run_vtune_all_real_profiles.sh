#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/cpu"
BINARY="$BUILD_DIR/sptrsv_bench"
VTUNE_BIN="${VTUNE_BIN:-/opt/intel/oneapi/vtune/2025.10/bin64/vtune}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/generated/cpu/vtune_profiles_all_real}"

KERNELS=(
  basic
  branchred
  unrolled
  thresholdomp
  chunkedomp
  capchunkomp
  mkl-solve
)

MATRICES=(
  ASIC_100k
  LFAT5000
  poli
  web-Stanford
  circuit204
  soc-sign-epinions
  msc04515
  pwtk
  helm2d03
  apache1
  bcsstk13
  bcsstk14
  bcsstk17
  bcsstk18
  bcsstk27
  bcsstm22
  scircuit
  fv1
  fv2
  fv3
  EPA
  ca-GrQc
  email-Enron
  soc-Epinions1
  roadNet-CA
  wiki-Vote
  ASIC_680k
  web-BerkStan
  roadNet-TX
  wiki-Talk
)

cd "$REPO_ROOT"

if [ ! -x "$VTUNE_BIN" ]; then
  echo "VTune binary not found: $VTUNE_BIN" >&2
  exit 1
fi

"$SCRIPT_DIR/run_benchmark.sh" family-one random 1024 basic > /dev/null

rm -rf "$RESULT_DIR"
mkdir -p "$RESULT_DIR"

for matrix in "${MATRICES[@]}"; do
  for kernel in "${KERNELS[@]}"; do
    safe_matrix="${matrix//-/_}"
    safe_kernel="${kernel//-/_}"
    result_dir="$RESULT_DIR/vtune_${safe_matrix}_${safe_kernel}"

    rm -rf "$result_dir"
    "$VTUNE_BIN" -collect hotspots -result-dir "$result_dir" -- "$BINARY" real-one "$matrix" "$kernel" > /dev/null
    "$VTUNE_BIN" -report summary -result-dir "$result_dir" > "${result_dir}_summary.txt"
    "$VTUNE_BIN" -report hotspots -result-dir "$result_dir" > "${result_dir}_hotspots.txt"
  done
done

echo "Wrote VTune outputs under:"
echo "  $RESULT_DIR"
