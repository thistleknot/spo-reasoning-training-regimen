#!/usr/bin/env bash
# Waits for generation (PID 3174138) to finish, then:
#   1. Builds the best-of-N SFT corpus via build_sft_corpus.py
#   2. Runs a clean-GPU training benchmark (batch sizes 8/16/32)
#   3. Runs SPO training on train_best_of_n.jsonl at the optimal batch_size
#
# Usage: nohup bash post_gen_pipeline.sh &> logs/post_gen_pipeline.log &

set -euo pipefail

REPO=/home/user/spo-reasoning-training-regimen
VENV=/home/user/mamba-venv/bin/python
GEN_PID=3174138
TOTAL=2366
ADAPTER=$REPO/output/full_run_qwen35_0.8b_3ep/base-plus-facts/adapter
GENERATED=$REPO/data/grpo_generated.jsonl
SFT_CORPUS=$REPO/data/train_best_of_n.jsonl
OUTPUT_DIR=$REPO/output/spo_best_of_n

cd "$REPO"

# ── 1. Wait for generation ────────────────────────────────────────────────────
echo "[$(date)] Waiting for generation PID $GEN_PID to finish..."
while kill -0 $GEN_PID 2>/dev/null; do
    DONE=$(wc -l < "$GENERATED")
    echo "[$(date)] Generation: ${DONE}/${TOTAL} quotes done"
    sleep 60
done
echo "[$(date)] Generation PID $GEN_PID has exited."
echo "[$(date)] Final row count: $(wc -l < "$GENERATED")"

# ── 2. Build best-of-N SFT corpus ────────────────────────────────────────────
echo "[$(date)] Building best-of-N SFT corpus..."
$VENV build_sft_corpus.py \
    --input  "$GENERATED" \
    --output "$SFT_CORPUS" \
    --top-k 3 \
    --diversity-alpha 0.15 \
    --min-reward 0.0 \
    --min-entailed 1 \
    --min-non-entailed 1

echo "[$(date)] SFT corpus: $(wc -l < "$SFT_CORPUS") rows → $SFT_CORPUS"

# ── 3. Clean-GPU benchmark ────────────────────────────────────────────────────
echo "[$(date)] Running clean-GPU training benchmark..."
$VENV benchmark_training.py \
    --adapter-path "$ADAPTER" \
    --data-path "$GENERATED" \
    --steps 10 \
    --batch-sizes 8,16,32 \
    2>&1 | tee logs/benchmark_clean.log

echo "[$(date)] Benchmark complete."

# ── 4. Resolve best batch_size ────────────────────────────────────────────────
BEST_BS=$($VENV - <<'PYEOF'
import json, sys
from pathlib import Path

results_path = Path("benchmark_training_results.json")
if not results_path.exists():
    print(8)   # safe fallback
    sys.exit(0)

results = json.loads(results_path.read_text())
if not results:
    print(8)
    sys.exit(0)

best = max(results, key=lambda r: r["tokens_per_sec"])
print(best["batch_size"])
PYEOF
)
echo "[$(date)] Using batch_size=${BEST_BS} from benchmark"

# ── 5. SPO training on best-of-N corpus ──────────────────────────────────────
echo "[$(date)] Starting SPO training on $SFT_CORPUS ..."
$VENV -m src.run_spo_training \
    --adapter-path  "$ADAPTER" \
    --dataset-path  "$SFT_CORPUS" \
    --output-dir    "$OUTPUT_DIR" \
    --num-epochs    2 \
    --batch-size    "$BEST_BS" \
    --gradient-accumulation-steps 1 \
    --learning-rate 1e-5 \
    --skip-regression-gate

echo "[$(date)] SPO training complete."
echo "[$(date)] Trained adapter → $OUTPUT_DIR"
echo "[$(date)] SFT corpus      → $SFT_CORPUS"
