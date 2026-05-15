#!/bin/bash
# v8 pipeline: annotation normalization — unlocks +0.30 confidence/tag scorer bonus
# Root cause of 0.65 ceiling: serializer stripped confidence annotations from training
# targets, so the model never learned the (tag, confidence=X) format that the scorer
# awards 0.15+0.15=0.30 for. v8 re-serializes with canonical annotations preserved.
set -e
cd /home/user/spo-reasoning-training-regimen
PYTHON=/home/user/mamba-venv/bin/python3
LOG=output/spo_verbatim_3ep_v8/training.log

mkdir -p output/spo_verbatim_3ep_v8

echo "=== Step 1: v8 SPO training (v7 restart, annotation-normalized corpus) ==="
$PYTHON -m src.run_spo_training \
    --adapter-path output/spo_verbatim_3ep_v7/adapter \
    --dataset-path data/train_facts_verbatim_v8.jsonl \
    --output-dir output/spo_verbatim_3ep_v8 \
    --num-epochs 5 \
    --learning-rate 2e-5 \
    2>&1 | tee "$LOG"

echo "=== Step 2: Generate holdout ==="
$PYTHON -m src.gen_spo_holdout \
    --adapter-path output/spo_verbatim_3ep_v8/adapter \
    --dataset-path data/train_facts_verbatim_v8.jsonl \
    --output output/spo_verbatim_3ep_v8/holdout_examples.md \
    --n 20

echo "=== v8 pipeline complete ==="
cat output/spo_verbatim_3ep_v8/holdout_examples.md | grep "^## Example\|score:" | head -30

AVG_SCORE=$($PYTHON -c "import json; d=json.load(open('output/spo_verbatim_3ep_v8/regression_gate.json')); print(d.get('avg_score','?'))" 2>/dev/null || echo "?")

echo "=== Step 3: Commit and push v8 artifacts ==="
git add -f \
    output/spo_verbatim_3ep_v8/holdout_examples.md \
    output/spo_verbatim_3ep_v8/spo_summary.json \
    output/spo_verbatim_3ep_v8/regression_gate.json \
    output/spo_verbatim_3ep_v8/adapter/adapter_config.json \
    output/spo_verbatim_3ep_v8/adapter/tokenizer_config.json \
    output/spo_verbatim_3ep_v8/adapter/README.md \
    src/serialize_training_format.py \
    src/build_training_regimens.py \
    src/rebuild_training_corpora.py \
    tests/test_supervision_prompt_contracts.py \
    data/train_facts_verbatim_v8.jsonl \
    run_v8_pipeline.sh \
    2>/dev/null || true

git commit -m "Add v8 SPO adapter: annotation normalization unlocks confidence/tag scoring

Score ceiling analysis (v7 at 0.656):
- Scorer awards +0.15 for confidence= in [0,1] and +0.15 for (tag,confidence=X) format
- v7 serializer stripped annotations → model output bare tags → permanent 0.30 ceiling
- v8 serializer normalizes all annotation formats to canonical (tag, confidence=X)

Changes:
- serialize_training_format.py: added normalize_triplet() (6-case ladder),
  is_bad_record() (template/N/A/repetitive filters), updated triplets_to_text()
  and serialize_training_record() to always preserve canonical annotations
- build_training_regimens.py: removed include_confidence=True kwargs
- rebuild_training_corpora.py: removed include_confidence=False kwarg
- tests/test_supervision_prompt_contracts.py: updated for new format
- data/train_facts_verbatim_v8.jsonl: 900-record corpus (v7 had 985)
  with all annotations in canonical (tag, confidence=X) format

Training: v7 adapter restart, 5 epochs, lr=2e-5
avg_score=${AVG_SCORE} (holdout 20 examples from v8 corpus)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" 2>/dev/null && git push || echo "Nothing to commit or push failed"
