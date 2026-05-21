#!/usr/bin/env bash
# Polls GPU free VRAM every 60s; launches GRPO phase-1 generation when
# at least 500 MB is available (enough for 4-bit 0.8B + KV cache).
# After phase-1, builds a curated best-of-N SFT corpus, then runs phase-2.
#
# Corpus: data/train_full_corpus.jsonl (2366 unique quotes, merged from
# gen_verbatim_checkpoint.db + train_structured_967.jsonl)
# Group size 8 → 8 completions per quote, each scored with 4 confidence
# samples → rich signal; best-of-8 selected per quote for SFT.
#
# Usage: nohup bash launch_grpo_when_ready.sh &> logs/grpo_launch.log &

set -euo pipefail

VENV=/home/user/mamba-venv/bin/python
REPO=/home/user/spo-reasoning-training-regimen
ADAPTER=$REPO/output/full_run_qwen35_0.8b_3ep/base-plus-facts/adapter
DATASET=$REPO/data/train_full_corpus.jsonl
GENERATED=$REPO/data/grpo_generated.jsonl
SFT_CORPUS=$REPO/data/train_best_of_n.jsonl
OUTPUT_DIR=$REPO/output/grpo_training

VRAM_THRESHOLD=500   # MB
POLL_SEC=60

# Ensure full corpus is ready (idempotent)
if [ ! -f "$DATASET" ]; then
    echo "[$(date)] Building full corpus ..."
    $VENV "$REPO/prep_full_corpus.py" \
        --extra-jsonl "$REPO/data/train_structured_967.jsonl"
fi
echo "[$(date)] Corpus: $(wc -l < "$DATASET") quotes in $DATASET"

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

# Phase 1: generate 8 completions per quote with confidence scoring
# --confidence-samples 4: each completion rated 4 times → distribution signal
# --confidence-weight 0.3: 30% of reward from confidence distribution quality
$VENV generate_grpo_data.py \
    --adapter-path "$ADAPTER" \
    --dataset-path "$DATASET" \
    --output-path "$GENERATED" \
    --group-size 8 \
    --max-new-tokens 256 \
    --confidence-samples 4 \
    --confidence-weight 0.3 \
    --seed 42 \
    --resume

echo "[$(date)] Phase 1 complete. Rows: $(wc -l < "$GENERATED")"

# Best-of-N SFT corpus: strict schema validation + best reward per quote
echo "[$(date)] Building best-of-N SFT corpus ..."
$VENV build_sft_corpus.py \
    --input  "$GENERATED" \
    --output "$SFT_CORPUS" \
    --top-k 1 \
    --min-reward 0.0 \
    --min-entailed 1 \
    --min-non-entailed 1

echo "[$(date)] SFT corpus: $(wc -l < "$SFT_CORPUS") rows → $SFT_CORPUS"

# Phase 2: GRPO train from precomputed rewards
echo "[$(date)] Launching phase 2 (GRPO train) ..."
$VENV run_grpo_training.py \
    --adapter-path "$ADAPTER" \
    --precomputed-data-path "$GENERATED" \
    --output-dir "$OUTPUT_DIR" \
    --group-size 8 \
    --patience 3 \
    --max-epochs 20

echo "[$(date)] GRPO pipeline complete. Adapter → $OUTPUT_DIR"
echo "[$(date)] SFT corpus ready for independent SFT training → $SFT_CORPUS"
