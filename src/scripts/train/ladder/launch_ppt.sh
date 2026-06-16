#! /bin/bash

set -euo pipefail

COMMAND="${1:-${COMMAND:-dry-run}}"
PPT_STEPS=500
MODEL_SIZE="60M"
CLUSTER="ai2/jupiter"
RUN_NAME="pptexp-ppt-${PPT_STEPS}steps-${MODEL_SIZE}"
WORKSPACE="ai2/linear-rnns"
BUDGET="ai2/oe-other"
PRIORITY="urgent"

case "$COMMAND" in
  dry-run | launch) ;;
  *)
    echo "COMMAND must be 'dry-run' or 'launch', got '$COMMAND'" >&2
    exit 1
    ;;
esac

max_gpus_for_size() {
  case "$1" in
    60M | 100M | 190M | 370M | 600M | 760M | 1B | 3B | 7B)
      echo 8
      ;;
    13B)
      echo 32
      ;;
    *)
      return 1
      ;;
  esac
}

if ! MAX_GPUS="$(max_gpus_for_size "$MODEL_SIZE")"; then
  echo "No max-gpus configured for MODEL_SIZE='$MODEL_SIZE'" >&2
  exit 1
fi

cmd=(
  uv run src/scripts/train/ladder/ppt_ladder.py "$COMMAND"
  --size "$MODEL_SIZE"
  --max-gpus "$MAX_GPUS"
  --name "$RUN_NAME"
  --chinchilla-multiple 8.0
  --cluster "$CLUSTER"
)

if [[ "$COMMAND" == "launch" ]]; then
  cmd+=(
    --workspace "$WORKSPACE"
    --budget "$BUDGET"
    --priority "$PRIORITY"
  )
fi

"${cmd[@]}"
