#!/usr/bin/env bash
# Run the v22 regimen training pipeline.
#
# Continues from the v19 tier3 adapter and runs tiers 4-5:
#   Tier 4 (facts_with_confidence):  200rec×3ep  → (tag, confidence=N) format
#   Tier 5 (syllogism_with_confidence): 200rec×3ep → Throughline + Confidence score
#
# Regimen datasets (200 records each) are already built:
#   data/train_facts_with_confidence_verbatim_v19.jsonl
#   data/train_syllogism_with_confidence_verbatim_v19.jsonl
#
# Usage:
#   bash run_v22_pipeline.sh [base_adapter_path]

set -euo pipefail

PYTHON=/home/user/mamba-venv/bin/python3

BASE_ADAPTER="${1:-output/ladder_run_v18/tier3_convergence/adapter}"
CORPUS="data/train_facts_verbatim_v14.jsonl"   # default corpus (overridden per-tier by corpus_override)
OUTPUT_DIR="output/ladder_run_v22"

echo "=== v22 Regimen Training ==="
echo "Base adapter: $BASE_ADAPTER"
echo "Output dir:   $OUTPUT_DIR"
echo ""

echo "=== Verifying regimen datasets ==="
$PYTHON - <<'PREFLIGHT'
import json
from pathlib import Path

facts_path = Path("data/train_facts_with_confidence_verbatim_v19.jsonl")
syl_path   = Path("data/train_syllogism_with_confidence_verbatim_v19.jsonl")

for path, name in [(facts_path, "facts_with_confidence"), (syl_path, "syllogism_with_confidence")]:
    if not path.exists():
        print(f"MISSING: {path}")
        import sys; sys.exit(1)
    records = [json.loads(l) for l in path.open()]
    na_count = sum(1 for r in records if "N/A" in r.get("output_text", ""))
    print(f"{name}: {len(records)} records, {na_count} N/A outputs")

print("Pre-flight PASSED.")
PREFLIGHT

echo ""
echo "=== Running ladder tiers 4-5 ==="
$PYTHON -m src.training_ladder \
    --base-adapter "$BASE_ADAPTER" \
    --corpus       "$CORPUS" \
    --output-root  "$OUTPUT_DIR" \
    --start-tier   4 \
    --max-tier     5

echo ""
echo "=== v22 pipeline complete. Results: $OUTPUT_DIR ==="
