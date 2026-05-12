#!/bin/bash

uv run python -m olmo_core.launch.beaker \
  --gpus=2 \
  --weka=oe-training-default \
  --workspace=ai2/jacksonp \
  --shared-filesystem \
  -- src/examples/llm/train.py \
    tutorial-run-01 \
    --save-folder="/weka/oe-training-default/$USER/tutorial-run-01" \
    --work-dir="/weka/oe-training-default/$USER/dataset-cache" \
    --trainer.callbacks.lm_evaluator.enabled=false \
    --trainer.callbacks.downstream_evaluator.enabled=false \
    --trainer.no_checkpoints \
    --trainer.hard_stop='{value: 100, unit: steps}'