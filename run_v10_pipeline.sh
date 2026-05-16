#!/bin/bash
# v10 pipeline: verbatim predicate — S|P(tag)|O format enforced throughout
#
# Root cause of v9 corpus defect: build_base_reasoning_prompt() description said
# "subject | (tag, confidence=N) | object" (no predicate), contradicting the
# inline example. 76.9% of v9 triplet lines had no predicate.
#
# This pipeline:
#   1. Waits for v10 corpus generator to complete (or exits early if not started)
#   2. Serializes structured JSONL to training format
#   3. Runs the full 4-tier training ladder against the v10 corpus
#
# Prerequisites:
#   - v10 corpus generator must have finished:
#       data/train_structured_verbatim_v10.jsonl  (structured output)
#   - Base adapter to start from:
#       output/ladder_run_latest/tier3_convergence/adapter/
#
set -e
cd /home/user/spo-reasoning-training-regimen

PYTHON=/home/user/mamba-venv/bin/python3
STRUCTURED=data/train_structured_verbatim_v10.jsonl
CORPUS=data/train_facts_verbatim_v10.jsonl
BASE_ADAPTER=output/ladder_run_latest/tier3_convergence/adapter
OUTPUT_ROOT=output/ladder_run_v10

if [ ! -f "$STRUCTURED" ]; then
    echo "ERROR: $STRUCTURED not found — corpus generator has not finished yet."
    echo "Check progress with: tail -f /tmp/gen_v10_corpus.log"
    exit 1
fi

echo "=== Step 1: Serialize v10 corpus to training format ==="
$PYTHON -m src.serialize_training_format \
    --input "$STRUCTURED" \
    --output "$CORPUS"
echo "Corpus written to $CORPUS ($(wc -l < "$CORPUS") records)"

echo "=== Step 2: Run 4-tier training ladder with v10 corpus ==="
mkdir -p "$OUTPUT_ROOT"
$PYTHON -m src.training_ladder \
    --base-adapter "$BASE_ADAPTER" \
    --corpus "$CORPUS" \
    --output-root "$OUTPUT_ROOT" \
    --max-tier 3 \
    2>&1 | tee "$OUTPUT_ROOT/ladder_run.log"

echo "=== v10 pipeline complete ==="
echo "Results in: $OUTPUT_ROOT/"
cat "$OUTPUT_ROOT/ladder_summary.json" 2>/dev/null || echo "(no ladder_summary.json — check log)"
