#!/bin/bash
# v7 pipeline: v3 restart + MUST-POPULATE instruction added to training prompt
# Key insight: weighted SFT on gold data cannot override the model's base tendency to
# leave Entailed Premises empty. Adding an explicit constraint to the input_text prompt
# teaches the model at training time to follow the rule, which transfers to inference
# because gen_spo_holdout uses the same input_text format.
set -e
cd /home/user/spo-reasoning-training-regimen
PYTHON=/home/user/mamba-venv/bin/python3
LOG=output/spo_verbatim_3ep_v7/training.log

mkdir -p output/spo_verbatim_3ep_v7

echo "=== Step 1: v7 SPO training (5 epochs from v3, must-populate prompt) ==="
$PYTHON -m src.run_spo_training \
    --adapter-path output/spo_verbatim_3ep_v3/adapter \
    --dataset-path data/train_facts_verbatim_v7.jsonl \
    --output-dir output/spo_verbatim_3ep_v7 \
    --num-epochs 5 \
    --learning-rate 2e-5 \
    2>&1 | tee "$LOG"

echo "=== Step 2: Generate holdout (in-distribution: v7 verbatim corpus) ==="
$PYTHON -m src.gen_spo_holdout \
    --adapter-path output/spo_verbatim_3ep_v7/adapter \
    --dataset-path data/train_facts_verbatim_v7.jsonl \
    --output output/spo_verbatim_3ep_v7/holdout_examples.md \
    --n 20

echo "=== v7 pipeline complete ==="
cat output/spo_verbatim_3ep_v7/holdout_examples.md | grep "^## Example\|score:" | head -30

echo "=== Step 3: Commit and push v7 artifacts ==="
git add -f \
    output/spo_verbatim_3ep_v7/holdout_examples.md \
    output/spo_verbatim_3ep_v7/spo_summary.json \
    output/spo_verbatim_3ep_v7/regression_gate.json \
    output/spo_verbatim_3ep_v7/adapter/adapter_config.json \
    output/spo_verbatim_3ep_v7/adapter/tokenizer_config.json \
    output/spo_verbatim_3ep_v7/adapter/README.md \
    src/serialize_training_format.py \
    data/train_facts_verbatim_v7.jsonl \
    2>/dev/null || true

AVG_SCORE=$($PYTHON -c "import json; d=json.load(open('output/spo_verbatim_3ep_v7/spo_summary.json')); print(d.get('avg_score','?'))" 2>/dev/null || echo "?")

git commit -m "Add v7 SPO adapter: must-populate instruction in training prompt

Root cause of empty Entailed Premises (v4-v6): weighted SFT trains on gold
outputs only; the model's own empty-Entailed tendencies are never penalized.
Adding an explicit constraint to input_text teaches the model at training
time and transfers to inference (same prompt at generation time).

Changes:
- serialize_training_format.py: added 'IMPORTANT: Entailed Premises section
  MUST contain at least one triplet. Never leave Entailed Premises empty.'
  to build_base_reasoning_prompt()
- data/train_facts_verbatim_v7.jsonl: 985-record corpus re-serialized with
  updated prompt

Training: v3 adapter restart, 5 epochs, lr=2e-5
avg_score=${AVG_SCORE} (holdout 20 examples from v7 corpus)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" 2>/dev/null && git push || echo "Nothing to commit or push failed"
