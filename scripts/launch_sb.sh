#!/usr/bin/env bash
#
# Launch the full StateBench suite: {hybrid,transformer} x {r-trivial,aperiodic,periodic}
# at the 60M size.
#
# Each invocation of state_bench.py's `launch` command already expands over
# model-type and distribution when both are omitted, so this script only needs
# to add an optional seed axis on top of that suite.
#
# Usage:
#   scripts/launch_sb.sh [--n_seeds N] [extra args passed to state_bench.py launch]
#
# Examples:
#   scripts/launch_sb.sh --max-gpus 8
#   scripts/launch_sb.sh --n_seeds 3 --max-gpus 8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_BENCH_PY="$SCRIPT_DIR/../src/scripts/train/ladder/state_bench.py"

n_seeds=1
extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_seeds)
            n_seeds="$2"
            shift 2
            ;;
        --n_seeds=*)
            n_seeds="${1#*=}"
            shift
            ;;
        *)
            extra_args+=("$1")
            shift
            ;;
    esac
done

for ((seed = 0; seed < n_seeds; seed++)); do
    if [[ "$n_seeds" -gt 1 ]]; then
        echo "=== Launching StateBench suite for init-seed $seed ==="
        uv run "$STATE_BENCH_PY" launch --size 60M --init-seed "$seed" "${extra_args[@]+"${extra_args[@]}"}"
    else
        uv run "$STATE_BENCH_PY" launch --size 60M "${extra_args[@]+"${extra_args[@]}"}"
    fi
done
