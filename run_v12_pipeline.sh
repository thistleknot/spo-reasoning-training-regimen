#!/bin/bash
set -e
cd /home/user/spo-reasoning-training-regimen

PYTHON=/home/user/mamba-venv/bin/python3
CORPUS=data/train_facts_verbatim_v12.jsonl
BASE_ADAPTER=output/ladder_run_v11/tier3_convergence/adapter
OUTPUT_ROOT=output/ladder_run_v12

echo "=== v12 pipeline: 78% verbatim verb-phrase predicates (repair+bootstrap+tautology-filter) ==="
echo "Corpus: $CORPUS ($(wc -l < "$CORPUS") records)"
echo "Base adapter: $BASE_ADAPTER"
echo ""

mkdir -p "$OUTPUT_ROOT"
$PYTHON -m src.training_ladder \
    --base-adapter "$BASE_ADAPTER" \
    --corpus "$CORPUS" \
    --output-root "$OUTPUT_ROOT" \
    --max-tier 3 \
    2>&1 | tee "$OUTPUT_ROOT/ladder_run.log"

echo ""
echo "=== v12 pipeline complete ==="
cat "$OUTPUT_ROOT/ladder_summary.json" 2>/dev/null || echo "(no summary)"
