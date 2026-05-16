#!/usr/bin/env bash
# Run the full v13 transliteration training pipeline.
#
# Steps:
#   1. Generate transliteration triplets for every entailed premise in
#      data/train_structured_verbatim_v12.jsonl → v13 structured corpus
#   2. Rebuild the filtered training corpus via serialize_training_format
#   3. Run the training ladder (tier 3 only from v12d tier2 adapter)
#
# Usage:
#   bash run_v13_pipeline.sh [base_adapter_path]
#
# Default base adapter: output/ladder_run_v12d/tier2_content/adapter

set -euo pipefail

PYTHON=/home/user/mamba-venv/bin/python3

BASE_ADAPTER="${1:-output/ladder_run_v12c/tier2_content/adapter}"
STRUCTURED_IN="data/train_structured_verbatim_v12.jsonl"
STRUCTURED_V13="data/train_structured_verbatim_v13.jsonl"
FACTS_V13="data/train_facts_verbatim_v13.jsonl"
CHECKPOINT_DB="data/gen_translit_checkpoint.db"
OUTPUT_DIR="output/ladder_run_v13"

echo "=== Step 1: Generate transliterations ==="
$PYTHON -m src.generate_transliterations \
    --input  "$STRUCTURED_IN" \
    --output "$STRUCTURED_V13" \
    --checkpoint "$CHECKPOINT_DB"

echo ""
echo "=== Step 2: Rebuild filtered training corpus ==="
$PYTHON -m src.serialize_training_format \
    --input  "$STRUCTURED_V13" \
    --output "$FACTS_V13"

RECORD_COUNT=$($PYTHON -c "
import json, pathlib
count = sum(1 for _ in pathlib.Path('$FACTS_V13').open())
print(count)
")
echo "Training corpus: $RECORD_COUNT records → $FACTS_V13"

echo ""
echo "=== Step 3: Run training ladder (tier 3 only from v12d tier2 adapter) ==="
$PYTHON -m src.training_ladder \
    --base-adapter "$BASE_ADAPTER" \
    --corpus       "$FACTS_V13" \
    --output-root  "$OUTPUT_DIR" \
    --start-tier   3

echo ""
echo "=== v13 pipeline complete. Results: $OUTPUT_DIR ==="
