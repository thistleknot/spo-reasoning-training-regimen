#!/bin/bash
# v6 pipeline: start from v3 (pre-tautological-contamination), filtered corpus,
# 5 epochs at 2e-5 lr to overwrite tautological bias and establish verbatim pattern
set -e
cd /home/user/spo-reasoning-training-regimen
PYTHON=/home/user/mamba-venv/bin/python3
LOG=output/spo_verbatim_3ep_v6/training.log

mkdir -p output/spo_verbatim_3ep_v6

echo "=== Step 1: v6 SPO training (5 epochs from v3 adapter, filtered corpus) ==="
$PYTHON -m src.run_spo_training \
    --adapter-path output/spo_verbatim_3ep_v3/adapter \
    --dataset-path data/train_facts_verbatim_filtered.jsonl \
    --output-dir output/spo_verbatim_3ep_v6 \
    --num-epochs 5 \
    --learning-rate 2e-5 \
    2>&1 | tee "$LOG"

echo "=== Step 2: Generate holdout (in-distribution: verbatim corpus) ==="
$PYTHON -m src.gen_spo_holdout \
    --adapter-path output/spo_verbatim_3ep_v6/adapter \
    --dataset-path data/train_facts_verbatim_filtered.jsonl \
    --output output/spo_verbatim_3ep_v6/holdout_examples.md \
    --n 20

echo "=== v6 pipeline complete ==="
cat output/spo_verbatim_3ep_v6/holdout_examples.md | grep "^## Example\|score:" | head -30

echo "=== Step 3: Commit and push v6 artifacts ==="
git add -f \
    output/spo_verbatim_3ep_v6/holdout_examples.md \
    output/spo_verbatim_3ep_v6/spo_summary.json \
    output/spo_verbatim_3ep_v6/regression_gate.json \
    output/spo_verbatim_3ep_v6/adapter/adapter_config.json \
    output/spo_verbatim_3ep_v6/adapter/tokenizer_config.json \
    output/spo_verbatim_3ep_v6/adapter/README.md \
    2>/dev/null || true

AVG_SCORE=$($PYTHON -c "import json; d=json.load(open('output/spo_verbatim_3ep_v6/spo_summary.json')); print(d.get('avg_score','?'))" 2>/dev/null || echo "?")

git commit -m "Add v6 SPO adapter: v3 restart + 5-epoch verbatim training

Rationale: v4/v5 were biased by 56% tautological corpus (E==NE).
v6 resets from v3 (pre-contamination) and trains directly on the
985-record filtered verbatim corpus (tautological records removed).

Training changes vs v5:
- Starting adapter: v3 (not v4/v5)
- Epochs: 5 (not 3)
- LR: 2e-5 (not 1e-5, to overwrite tautological bias)
- Holdout drawn from verbatim corpus (in-distribution)

avg_score=${AVG_SCORE} (holdout 20 examples from verbatim corpus)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" 2>/dev/null && git push || echo "Nothing to commit or push failed"
