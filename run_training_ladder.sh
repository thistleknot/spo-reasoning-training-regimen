#!/bin/bash
# Full training ladder: runs all 4 tiers (0 → 3) with automatic go/no-go gates.
#
# Tier 0 — zero-shot structure     (~2 min)
# Tier 1 — annotation format       (~10 min, 50 rec, 1 epoch)
# Tier 2 — content quality         (~25 min, 200 rec, 2 epochs)
# Tier 3 — full convergence        (~90 min, 900 rec, 5 epochs)
#
# Usage:
#   bash run_training_ladder.sh [ADAPTER_PATH] [CORPUS_PATH] [MAX_TIER]
#
# Defaults:
#   ADAPTER_PATH = output/spo_verbatim_3ep_v8/adapter
#   CORPUS_PATH  = data/train_facts_verbatim_v9.jsonl
#   MAX_TIER     = 3  (full ladder)
#
# To run only through Tier 2 (skip the full 90-min train):
#   bash run_training_ladder.sh path/to/adapter data/corpus.jsonl 2

set -e
cd /home/user/spo-reasoning-training-regimen

PYTHON=/home/user/mamba-venv/bin/python3
ADAPTER_PATH="${1:-output/spo_verbatim_3ep_v8/adapter}"
CORPUS_PATH="${2:-data/train_facts_verbatim_v9.jsonl}"
MAX_TIER="${3:-3}"
OUTPUT_ROOT="output/ladder_$(date +%Y%m%d_%H%M%S)"

echo "=== Training Ladder ==="
echo "  base adapter : $ADAPTER_PATH"
echo "  corpus       : $CORPUS_PATH"
echo "  max_tier     : $MAX_TIER"
echo "  output_root  : $OUTPUT_ROOT"
echo ""

$PYTHON -m src.training_ladder \
    --base-adapter "$ADAPTER_PATH" \
    --corpus "$CORPUS_PATH" \
    --output-root "$OUTPUT_ROOT" \
    --max-tier "$MAX_TIER" \
    2>&1 | tee "$OUTPUT_ROOT/../ladder_$(date +%Y%m%d_%H%M%S).log"

EXIT=$?
echo ""
if [ $EXIT -eq 0 ]; then
    echo "=== LADDER COMPLETE — results in $OUTPUT_ROOT/ladder_summary.json ==="
else
    echo "=== LADDER FAILED — see $OUTPUT_ROOT/ladder_summary.json for which tier stopped ==="
fi
exit $EXIT
