#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/cpu"
BINARY="$BUILD_DIR/sptrsv_bench"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/generated/cpu}"

mkdir -p "$RESULT_DIR"
cd "$REPO_ROOT"

# Build once, then use the binary directly to avoid recompiling per run.
"$SCRIPT_DIR/run_benchmark.sh" family-one random 1024 basic > /dev/null

"$BINARY" families > "$RESULT_DIR/synthetic_solve_only_table.txt"
"$BINARY" real > "$RESULT_DIR/real_sweep_solve_only_table.txt"

echo "Wrote:"
echo "  $RESULT_DIR/synthetic_solve_only_table.txt"
echo "  $RESULT_DIR/real_sweep_solve_only_table.txt"
