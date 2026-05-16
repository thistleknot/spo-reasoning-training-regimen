#!/bin/bash
# v9 pipeline: fix confidence=X prompt artifact from v8
# Root cause of confidence=X in v8 outputs: prompt said "confidence=X" literally.
# v9 changes prompt to "confidence=N" with example "confidence=1.0" to teach concrete values.
# Restarts from v8 adapter (v8 avg_score=0.8225, now targeting >0.90 once X is eliminated).
set -e
cd /home/user/spo-reasoning-training-regimen
PYTHON=/home/user/mamba-venv/bin/python3
LOG=output/spo_verbatim_3ep_v9/training.log

mkdir -p output/spo_verbatim_3ep_v9

echo "=== Step 1: v9 SPO training (v8 restart, fixed prompt no confidence=X) ==="
$PYTHON -m src.run_spo_training \
    --adapter-path output/spo_verbatim_3ep_v8/adapter \
    --dataset-path data/train_facts_verbatim_v9.jsonl \
    --output-dir output/spo_verbatim_3ep_v9 \
    --num-epochs 5 \
    --learning-rate 1e-5 \
    2>&1 | tee "$LOG"

echo "=== Step 2: Generate holdout ==="
$PYTHON -m src.gen_spo_holdout \
    --adapter-path output/spo_verbatim_3ep_v9/adapter \
    --dataset-path data/train_facts_verbatim_v9.jsonl \
    --output output/spo_verbatim_3ep_v9/holdout_examples.md \
    --n 20

echo "=== v9 pipeline complete ==="
cat output/spo_verbatim_3ep_v9/holdout_examples.md | grep "^## Example\|score:" | head -30

AVG_SCORE=$($PYTHON -c "import json; d=json.load(open('output/spo_verbatim_3ep_v9/regression_gate.json')); print(d.get('avg_score','?'))" 2>/dev/null || echo "?")

echo "=== Step 3: Commit and push v9 artifacts ==="
git add -f \
    output/spo_verbatim_3ep_v9/holdout_examples.md \
    output/spo_verbatim_3ep_v9/spo_summary.json \
    output/spo_verbatim_3ep_v9/regression_gate.json \
    output/spo_verbatim_3ep_v9/adapter/adapter_config.json \
    output/spo_verbatim_3ep_v9/adapter/tokenizer_config.json \
    output/spo_verbatim_3ep_v9/adapter/README.md \
    src/serialize_training_format.py \
    src/build_training_regimens.py \
    src/rebuild_training_corpora.py \
    tests/test_supervision_prompt_contracts.py \
    data/train_facts_verbatim_v9.jsonl \
    run_v9_pipeline.sh \
    2>/dev/null || true

git commit -m "Add v9 SPO adapter: fix confidence=X prompt artifact

v8 (avg_score=0.8225) had model outputs using literal 'confidence=X' in
~14 of 60 triplet lines because the training prompt said:
  'Format each premise as: subject | relation (tag, confidence=X) | object'
The model learned to echo the template placeholder instead of a numeric value.

Fix: prompt now says 'confidence=N' with concrete example confidence=1.0.
Both build_base_reasoning_prompt() and serialize_facts_with_confidence_record()
updated. Training uses half the lr (1e-5) since v8 was already well-converged.

Changes:
- src/serialize_training_format.py: prompt line 169 changed
- src/build_training_regimens.py: prompt line 82 changed
- tests/test_supervision_prompt_contracts.py: updated assertions
- data/train_facts_verbatim_v9.jsonl: 900 records, same filters as v8

Training: v8 adapter restart, 5 epochs, lr=1e-5
avg_score=${AVG_SCORE} (regression gate, 5 examples from v9 corpus)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" 2>/dev/null && git push || echo "Nothing to commit or push failed"
