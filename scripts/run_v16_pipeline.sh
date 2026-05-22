#!/usr/bin/env bash
# Run the v14 training pipeline.
#
# Re-uses the v13 structured corpus (transliterations already generated).
# Steps:
#   1. Re-serialize v13 structured corpus with confidence stripped from
#      base_reasoning targets (serialize_training_format.py now strips)
#   1b. PRE-FLIGHT corpus validation — confirm tag-only format throughout
#   2. Run training ladder tier3 from v12c tier2 adapter using v14 corpus
#
# Usage:
#   bash run_v14_pipeline.sh [base_adapter_path]

set -euo pipefail

PYTHON=/home/user/mamba-venv/bin/python3

BASE_ADAPTER="${1:-output/ladder_run_v12c/tier2_content/adapter}"
STRUCTURED_V13="data/train_structured_verbatim_v13.jsonl"
FACTS_V14="data/train_facts_verbatim_v14.jsonl"
OUTPUT_DIR="output/ladder_run_v14"

echo "=== Step 1: Serialize v13 structured corpus (confidence stripped) ==="
$PYTHON -m src.serialize_training_format \
    --input  "$STRUCTURED_V13" \
    --output "$FACTS_V14"

echo ""
echo "=== Step 1b: Pre-flight corpus validation (sample=50) ==="
$PYTHON - <<'PREFLIGHT'
import json, re, sys
from pathlib import Path

tag_re     = re.compile(r'\(\s*(observed|inferred)\s*\)', re.IGNORECASE)
triplet_re = re.compile(r'^[^|]+\|[^|]+\|[^|]+$')

records = [json.loads(l) for l in Path("data/train_facts_verbatim_v14.jsonl").open()]
sample  = records[:50]

issues = []
for i, r in enumerate(sample):
    out = r.get("output_text", "")
    for hdr in ("Non-Entailed Premises:", "Entailed Premises:", "Throughline:"):
        if hdr not in out:
            issues.append(f"record {i}: missing header '{hdr}'")
    for line in out.splitlines():
        line = line.strip()
        if not line or line.endswith(":"):
            continue
        if line.startswith("(") and line.endswith(")"):
            continue  # transliteration line
        if triplet_re.match(line):
            if not re.search(r'\(\s*(observed|inferred)', line, re.IGNORECASE):
                issues.append(f"record {i}: triplet missing tag: {line[:80]}")
                break
            if re.search(r'confidence\s*=', line, re.IGNORECASE):
                issues.append(f"record {i}: confidence NOT stripped: {line[:80]}")
                break

# Transliteration coverage
with_tl = sum(1 for r in sample if "\n(" in r.get("output_text",""))
tl_pct  = 100 * with_tl / len(sample)

print(f"Sample: {len(sample)} records")
print(f"Transliteration coverage: {with_tl}/{len(sample)} ({tl_pct:.1f}%)")
print(f"Format issues found: {len(issues)}")
for iss in issues[:10]:
    print(f"  ! {iss}")

if issues:
    sys.exit(1)
print("Pre-flight PASSED — proceeding to training.")
PREFLIGHT

echo ""
RECORD_COUNT=$($PYTHON -c "
import json, pathlib
count = sum(1 for _ in pathlib.Path('$FACTS_V14').open())
print(count)
")
echo "Training corpus: $RECORD_COUNT records → $FACTS_V14"

echo ""
echo "=== Step 2: Run training ladder (tier3 from v12c tier2 adapter) ==="
$PYTHON -m src.training_ladder \
    --base-adapter "$BASE_ADAPTER" \
    --corpus       "$FACTS_V14" \
    --output-root  "$OUTPUT_DIR" \
    --start-tier   3

echo ""
echo "=== v14 pipeline complete. Results: $OUTPUT_DIR ==="
