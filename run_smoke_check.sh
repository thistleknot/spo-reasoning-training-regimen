#!/bin/bash
# Smoke check wrapper: runs Tier 0 (structure) and Tier 1 (annotation format)
# of the training ladder.  Total ~12-15 min vs 75-112 min for a full run.
#
# Usage:
#   bash run_smoke_check.sh [ADAPTER_PATH] [CORPUS_PATH]
#
# Defaults:
#   ADAPTER_PATH = output/spo_verbatim_3ep_v8/adapter
#   CORPUS_PATH  = data/train_facts_verbatim_v9.jsonl
#
# Set MAX_TIER=0 to run only Tier 0 (zero-shot, ~2 min).
# Set MAX_TIER=2 to extend through Tier 2 content quality (~25 min extra).
# Full ladder (Tier 3): use run_training_ladder.sh instead.

set -e
cd /home/user/spo-reasoning-training-regimen

PYTHON=/home/user/mamba-venv/bin/python3
ADAPTER_PATH="${1:-output/spo_verbatim_3ep_v8/adapter}"
CORPUS_PATH="${2:-data/train_facts_verbatim_v9.jsonl}"
MAX_TIER="${MAX_TIER:-1}"
OUTPUT_ROOT="output/smoke_$(date +%Y%m%d_%H%M%S)"

echo "=== Smoke Check (ladder tiers 0-${MAX_TIER}) ==="
echo "  base adapter : $ADAPTER_PATH"
echo "  corpus       : $CORPUS_PATH"
echo "  max_tier     : $MAX_TIER"
echo "  output_root  : $OUTPUT_ROOT"
echo ""

$PYTHON -m src.training_ladder \
    --base-adapter "$ADAPTER_PATH" \
    --corpus "$CORPUS_PATH" \
    --output-root "$OUTPUT_ROOT" \
    --max-tier "$MAX_TIER"

EXIT=$?
echo ""
if [ $EXIT -eq 0 ]; then
    echo "=== SMOKE CHECK PASSED — proceed to full training run ==="
else
    echo "=== SMOKE CHECK FAILED — investigate before full training ==="
fi
exit $EXIT
