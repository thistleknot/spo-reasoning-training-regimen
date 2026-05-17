#!/usr/bin/env bash
# Run the full v13 transliteration training pipeline.
#
# Steps:
#   1. Generate transliteration triplets for every entailed premise in
#      data/train_structured_verbatim_v12.jsonl → v13 structured corpus
#   2. Rebuild the filtered training corpus via serialize_training_format
#   2b. PRE-FLIGHT corpus validation — fast sample check before committing to
#       the full training run; aborts early on bad corpus quality
#   3. Run the training ladder (tier 3 only from v12c tier2 adapter)
#
# Usage:
#   bash run_v13_pipeline.sh [base_adapter_path]

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

echo ""
echo "=== Step 2b: Pre-flight corpus validation (sample=50) ==="
$PYTHON - <<'PREFLIGHT'
import json, re, sys
from pathlib import Path

tag_re     = re.compile(r'\(\s*(observed|inferred)\s*\)', re.IGNORECASE)  # tag-only, no confidence
triplet_re = re.compile(r'^[^|]+\|[^|]+\|[^|]+$')

records = [json.loads(l) for l in Path("data/train_facts_verbatim_v13.jsonl").open()]
sample  = records[:50]

issues = []
for i, r in enumerate(sample):
    out = r.get("output_text", "")
    # Must have all 3 headers
    for hdr in ("Non-Entailed Premises:", "Entailed Premises:", "Throughline:"):
        if hdr not in out:
            issues.append(f"record {i}: missing header '{hdr}'")
    # All main triplet lines must have a tag annotation (confidence stripped by design)
    for line in out.splitlines():
        line = line.strip()
        if not line or line.endswith(":"):
            continue
        if line.startswith("(") and line.endswith(")"):
            continue  # transliteration line — handled separately
        if triplet_re.match(line):
            if not re.search(r'\(\s*(observed|inferred)', line, re.IGNORECASE):
                issues.append(f"record {i}: triplet missing tag: {line[:80]}")
                break

# Transliteration coverage
with_tl = sum(1 for r in sample if "\n(" in r.get("output_text",""))
tl_pct  = 100 * with_tl / len(sample)

print(f"Sample: {len(sample)} records")
print(f"Transliteration coverage: {with_tl}/{len(sample)} ({tl_pct:.1f}%)")
print(f"Format issues found: {len(issues)}")
for iss in issues[:5]:
    print(f"  ! {iss}")

if len(issues) > 5:
    print(f"  ... and {len(issues)-5} more")
    sys.exit(1)
print("Pre-flight PASSED — proceeding to training.")
PREFLIGHT

echo ""
RECORD_COUNT=$($PYTHON -c "
import json, pathlib
count = sum(1 for _ in pathlib.Path('$FACTS_V13').open())
print(count)
")
echo "Training corpus: $RECORD_COUNT records → $FACTS_V13"

echo ""
echo "=== Step 3: Run training ladder (tier 3 only from v12c tier2 adapter) ==="
$PYTHON -m src.training_ladder \
    --base-adapter "$BASE_ADAPTER" \
    --corpus       "$FACTS_V13" \
    --output-root  "$OUTPUT_DIR" \
    --start-tier   3

echo ""
echo "=== v13 pipeline complete. Results: $OUTPUT_DIR ==="
