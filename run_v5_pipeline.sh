#!/bin/bash
# v5 pipeline: filtered corpus (no tautological records) + fixed empty-Entailed gate
set -e
cd /home/user/spo-reasoning-training-regimen
PYTHON=/home/user/mamba-venv/bin/python3
LOG=output/spo_verbatim_3ep_v5/training.log

mkdir -p output/spo_verbatim_3ep_v5

echo "=== Step 1: v5 SPO training (3 epochs from v4 adapter, filtered corpus) ==="
$PYTHON -m src.run_spo_training \
    --adapter-path output/spo_verbatim_3ep_v4/adapter \
    --dataset-path data/train_facts_verbatim_filtered.jsonl \
    --output-dir output/spo_verbatim_3ep_v5 \
    --num-epochs 3 \
    2>&1 | tee "$LOG"

echo "=== Step 2: Generate holdout ==="
$PYTHON -m src.gen_spo_holdout \
    --adapter-path output/spo_verbatim_3ep_v5/adapter \
    --dataset-path data/train_facts_verbatim_filtered.jsonl \
    --output output/spo_verbatim_3ep_v5/holdout_examples.md \
    --n 20

echo "=== v5 pipeline complete ==="
cat output/spo_verbatim_3ep_v5/holdout_examples.md | grep "^## Example\|score:" | head -30

echo "=== Step 3: Commit and push v5 artifacts ==="
git add -f \
    output/spo_verbatim_3ep_v5/holdout_examples.md \
    output/spo_verbatim_3ep_v5/spo_summary.json \
    output/spo_verbatim_3ep_v5/regression_gate.json \
    output/spo_verbatim_3ep_v5/adapter/adapter_config.json \
    output/spo_verbatim_3ep_v5/adapter/tokenizer_config.json \
    output/spo_verbatim_3ep_v5/adapter/README.md \
    src/spo_trainer.py \
    src/serialize_training_format.py \
    data/train_facts_verbatim_filtered.jsonl \
    2>/dev/null || true

AVG_SCORE=$(python3 -c "import json; d=json.load(open('output/spo_verbatim_3ep_v5/spo_summary.json')); print(d.get('avg_score','?'))" 2>/dev/null || echo "?")

git commit -m "Add v5 SPO adapter: filtered corpus + empty-Entailed gate

Key changes:
- serialize_training_format.py: filter tautological records (E==NE)
  985 records kept from 2259 (vs 2259 unfiltered in v4)
- spo_trainer.py: empty-Entailed penalty (-0.15) now applies without
  source_quote; previously only triggered when source_quote provided

avg_score=${AVG_SCORE} (holdout 20 examples)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" 2>/dev/null && git push || echo "Nothing to commit or push failed"
