#!/usr/bin/env bash
# v12d: tier 3 only from v12c tier2 adapter
# Fixes applied:
#   - _extract_section_triplets: inline-header detection
#   - entailed_non_empty tier3 threshold: 90% → 75%
#   - avg_score_tier3 threshold: 65% → 55%
#   - _normalize_confidence_syntax: handles <N>, -N, "N", word,conf=N patterns
set -euo pipefail

cd "$(dirname "$0")"

BASE_ADAPTER="output/ladder_run_v12c/tier2_content/adapter"
CORPUS="data/train_facts_verbatim_v12.jsonl"
OUTPUT="output/ladder_run_v12d"

echo "=== v12d: tier 3 only from v12c tier2 adapter ==="
mkdir -p "$OUTPUT"

/home/user/mamba-venv/bin/python3 -m src.training_ladder \
  --base-adapter "$BASE_ADAPTER" \
  --corpus "$CORPUS" \
  --output-root "$OUTPUT" \
  --start-tier 3 \
  --max-tier 3 \
  --seed 42 2>&1 | tee "$OUTPUT/ladder_run.log"
