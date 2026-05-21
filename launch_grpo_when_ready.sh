#!/usr/bin/env bash
# Polls GPU free VRAM every 60s; launches GRPO phase-1 generation when
# at least 500 MB is available (enough for 4-bit 0.8B + KV cache).
# Phase-2 training is launched automatically after phase-1 completes.
#
# Usage: nohup bash launch_grpo_when_ready.sh &> logs/grpo_launch.log &

set -euo pipefail

VENV=/home/user/mamba-venv/bin/python
REPO=/home/user/spo-reasoning-training-regimen
ADAPTER=$REPO/output/full_run_qwen35_0.8b_3ep/base-plus-facts/adapter
DATASET=$REPO/data/train_structured_967.jsonl
GENERATED=$REPO/data/grpo_generated.jsonl
OUTPUT_DIR=$REPO/output/grpo_training

VRAM_THRESHOLD=500   # MB
POLL_SEC=60

echo "[$(date)] Waiting for >= ${VRAM_THRESHOLD} MB free VRAM ..."

while true; do
    FREE_MB=$($VENV -c "import torch; free,_=torch.cuda.mem_get_info(0); print(int(free/1024/1024))" 2>/dev/null || echo 0)
    echo "[$(date)] Free VRAM: ${FREE_MB} MB"

    if [ "$FREE_MB" -ge "$VRAM_THRESHOLD" ]; then
        echo "[$(date)] GPU ready. Launching phase 1 (generate) ..."
        break
    fi

    sleep "$POLL_SEC"
done

cd "$REPO"

# Phase 1: generate completions + scores
$VENV generate_grpo_data.py \
    --adapter-path "$ADAPTER" \
    --dataset-path "$DATASET" \
    --output-path "$GENERATED" \
    --group-size 8 \
    --max-new-tokens 256 \
    --seed 42 \
    --resume

echo "[$(date)] Phase 1 complete. Rows written: $(wc -l < "$GENERATED")"

# Phase 2: train from precomputed rewards
echo "[$(date)] Launching phase 2 (train) ..."
$VENV run_grpo_training.py \
    --adapter-path "$ADAPTER" \
    --precomputed-data-path "$GENERATED" \
    --output-dir "$OUTPUT_DIR" \
    --group-size 8 \
    --patience 3 \
    --max-epochs 20

echo "[$(date)] GRPO pipeline complete. Adapter saved to $OUTPUT_DIR"
