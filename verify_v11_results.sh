#!/bin/bash
# Post-tier3 verification: check holdout_examples.md and update ladder_run_latest
set -e
cd /home/user/spo-reasoning-training-regimen

PYTHON=/home/user/mamba-venv/bin/python3
V11_OUT=output/ladder_run_v11

echo "=== v11 Results Verification ==="
echo ""

# 1. Check ladder_summary.json
if [ -f "$V11_OUT/ladder_summary.json" ]; then
    echo "--- Ladder Summary ---"
    cat "$V11_OUT/ladder_summary.json"
    echo ""
fi

# 2. Show holdout_examples.md header (scores table)
if [ -f "$V11_OUT/holdout_examples.md" ]; then
    echo "--- holdout_examples.md (first 60 lines) ---"
    head -60 "$V11_OUT/holdout_examples.md"
    echo ""
fi

# 3. Show tier3 check rates
if [ -f "$V11_OUT/tier3_convergence/holdout_examples.md" ]; then
    echo "--- tier3 check rates ---"
    head -30 "$V11_OUT/tier3_convergence/holdout_examples.md"
    echo ""
fi

# 4. Sample a few triplets from generated outputs to verify verbatim predicates
echo "--- Sample entailed triplets with predicates ---"
$PYTHON3 -c "
import re, json
with open('$V11_OUT/tier3_convergence/holdout_examples.md') as f:
    content = f.read()
sections = content.split('**Generated**')
pred_re = re.compile(r'\|\s+\w[^|(]+\s+\((?:observed|inferred)')
count = 0
for sec in sections[1:]:
    m = re.search(r'\`\`\`text\n(.*?)\`\`\`', sec, re.DOTALL)
    if not m: continue
    for line in m.group(1).splitlines():
        if '|' in line and pred_re.search(line):
            print('  PRED:', line.strip()[:100])
            count += 1
            if count >= 10: break
    if count >= 10: break
if count == 0:
    print('  (no predicate triplets found — bare-tag only output)')
" 2>/dev/null || true

echo ""
echo "=== Done ==="
