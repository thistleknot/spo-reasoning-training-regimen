#!/bin/bash
# Post-generation v4 pipeline: rebuild JSONL → SPO train → holdout
set -e
cd /home/user/spo-reasoning-training-regimen
PYTHON=/home/user/mamba-venv/bin/python3

echo "=== Step 1: Wait for generation to complete ==="
while ps aux | grep "generate_verbatim_corpus" | grep -qv grep; do
    PROGRESS=$(strings data/gen_verbatim.log 2>/dev/null | grep "written=" | tail -1)
    echo "[$(date +%H:%M)] $PROGRESS"
    sleep 60
done
echo "Generation process done."

echo "=== Step 2: Build training JSONL from verbatim corpus ==="
$PYTHON -m src.serialize_training_format \
    --input data/train_structured_verbatim.jsonl \
    --output data/train_facts_with_confidence_verbatim.jsonl

VERBATIM_COUNT=$(wc -l < data/train_facts_with_confidence_verbatim.jsonl)
echo "Training JSONL: $VERBATIM_COUNT records"

echo "=== Step 3: v4 SPO training (3 epochs from v3 adapter) ==="
mkdir -p output/spo_verbatim_3ep_v4
$PYTHON -m src.run_spo_training \
    --adapter-path output/spo_verbatim_3ep_v3/adapter \
    --dataset-path data/train_facts_with_confidence_verbatim.jsonl \
    --output-dir output/spo_verbatim_3ep_v4 \
    --num-epochs 3 \
    2>&1 | tee output/spo_verbatim_3ep_v4/training.log

echo "=== Step 4: Generate holdout ==="
$PYTHON -m src.gen_spo_holdout \
    --adapter-path output/spo_verbatim_3ep_v4/adapter \
    --dataset-path data/train_facts_with_confidence_verbatim.jsonl \
    --output output/spo_verbatim_3ep_v4/holdout_examples.md \
    --n 20

echo "=== v4 pipeline complete ==="
echo "Holdout: output/spo_verbatim_3ep_v4/holdout_examples.md"
