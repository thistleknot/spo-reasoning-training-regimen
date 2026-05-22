# SPO Reasoning Training Regimen

Build quote -> structured reasoning dataset -> QLoRA adapter.

This repo packages the full workflow for turning raw quotes into structured reasoning examples and training a model to emit that reasoning in a pedagogical order. The root README is the front door; the deeper mechanics live in the docs folders linked below.

## What you get

- Synthetic reasoning dataset generation from quotes
- A format contract that separates generation order from training order
- Three complementary training regimens built from the same synthetic base
- QLoRA training guidance for small-to-mid-size reasoning models
- Optional downstream judging or calibration after the base reasoning model is trained
- "Seeing is believing" example artifacts under `data/`

## Workflow at a glance

```mermaid
flowchart LR
    A[Quotes] --> B[generate_grpo_data.py\nK=8 completions/quote\nfrozen-judge scored]
    B --> C[build_sft_corpus.py\ntop-k=3 greedy diverse\nbest-of-N selection]
    C --> D[benchmark_training.py\nfind optimal batch_size]
    D --> E[src/run_spo_training.py\nbf16 + grad-ckpt\n2 epochs]
    E --> F[output/spo_best_of_n/adapter]
    F --> G[Inference on new quotes]
```

### Automated pipeline

`post_gen_pipeline.sh` runs the full sequence unattended: waits for generation to
complete, benchmarks training throughput on the clean GPU, applies the fastest batch
size, builds the best-of-N corpus, and launches SPO training.

## Quick start

### Install

```bash
git clone https://github.com/thistleknot/spo-reasoning-training-regimen.git
cd spo-reasoning-training-regimen
pip install -r requirements.txt
```

### Generate a starter dataset

```python
from src.synthetic_generator import SyntheticReasoningGenerator

quotes = [
    "By the pricking of my thumbs, Something wicked this way comes.",
    "Call me Ishmael.",
]

gen = SyntheticReasoningGenerator()
examples = gen.generate_from_quotes(quotes)
gen.export_to_jsonl("data/my_dataset.jsonl")
```

That path creates template-backed examples immediately. If you want model-backed generation with OpenAI, Ollama, or Hugging Face, go straight to `docs/generation/README.md`.

### Then choose your next step

1. Inspect example outputs in `data/SEEING_IS_BELIEVING_EXAMPLES.md`
2. Learn the format contract in `docs/format/README.md`
3. Train a model via `docs/training/README.md`
4. Configure inference via `docs/inference/README.md`
5. Evaluate on held-out quotes to verify reasoning quality

## Example output

One holdout quote run through the original adapter and the SPO-trained adapter.
Full set (15 quotes): [`data/SEEING_IS_BELIEVING_EXAMPLES.md`](data/SEEING_IS_BELIEVING_EXAMPLES.md).

> *"Imagine smiling after a slap in the face. Then think of doing it twenty-four hours a day."*

*(holdout reward: 0.619)*

**Before** (original adapter):
```
Non-Entailed Premises:
slap in the face | is (observed) | an event
smiling | is (observed, implied) | a positive emotion
doing it twenty-four hours a daily | is (observed/implies) | a routine

Entailed Premises:
doing it twenty-four hours daily | is (inferred) | a way to maintain the positive emotion
slap in the faces | is (inferred, implied) | a source of the positive emotion
smiling | is an action | is (observed/implied) | taken after a slap in the faces
slap in the facing | is (observed/inferred) | a slap in the face
doing it twenty-four days a day | is (inferred/implies) | the same positive emotion
slap | is (observed/facts) | a slap in a face

Throughline:
Maintaining a positive emotional state through a daily routine of self-correction
(smiling after a slap) is a sustainable strategy.
```

**After** (SPO best-of-N adapter):
```
Non-Entailed Premises:
slap in the face | is (observed) | an event
smiling after a slap in the faces | is (observed) | a reaction
doing it twenty-four hours a daily | is (observed) | a habit

Entailed Premises:
doing it twenty-four hours daily | is (inferred) | implies a daily routine
smiling after a slip in the face | is an example of doing it twenty-four times a day
slap in the faces | is an example of smiling after a slip in the faces
smiling after a slp in the face | is a form of doing it twenty-four a day

Throughline:
Smiling after a slap in a face is a daily habit.
```

*Before: 6 entailed premises with garbled subject echoes ("slap in the faces", "slap in the facing"), bloated throughline. After: cleaner subject reference, tighter throughline.*

## Repo map

```text
spo-reasoning-training-regimen/
├── src/                              # library modules (import targets)
│   ├── run_spo_training.py           # SPO training entrypoint
│   ├── spo_trainer.py
│   ├── frozen_judge.py               # batch-scoring reward model
│   ├── grpo_trainer.py
│   ├── synthetic_generator.py
│   ├── build_training_regimens.py
│   ├── training_strategy.py
│   ├── evaluate_regimens.py
│   ├── run_ablation_matrix.py
│   ├── serialize_training_format.py
│   ├── chat_format.py
│   └── ...
├── scripts/                          # pipeline runners (call from repo root)
│   ├── generate_grpo_data.py         # Phase 1: quote → K completions
│   ├── build_sft_corpus.py           # Phase 1→2: best-of-N selection
│   ├── benchmark_training.py         # GPU batch-size benchmark
│   ├── gen_seeing_is_believing.py    # holdout before/after inference
│   ├── run_grpo_training.py
│   ├── prep_full_corpus.py
│   ├── generate_layered.py           # v2 two-stage generation (design)
│   ├── post_gen_pipeline.sh          # unattended full pipeline
│   ├── launch_grpo_when_ready.sh
│   ├── run_v15_pipeline.sh           # historical regimen runners
│   └── watch_and_build.sh
├── docs/
│   ├── generation/README.md
│   ├── training/README.md
│   ├── inference/README.md
│   ├── format/README.md
│   ├── architecture/README.md
│   ├── quickstart/README.md
│   └── history/                      # prior-version review artifacts
├── data/
│   ├── train_best_of_n.jsonl         # 6921-row best-of-N SFT corpus
│   ├── grpo_generated.jsonl          # 2366×8 raw generation output
│   ├── train_structured_967.jsonl    # original 967-quote corpus
│   ├── SEEING_IS_BELIEVING_EXAMPLES.md  # before/after holdout outputs
│   ├── benchmark_training_results.json
│   └── sample_quotes.txt
└── tests/
```

## Documentation

| Goal | Read |
|---|---|
| Get running quickly | [`docs/quickstart/README.md`](docs/quickstart/README.md) |
| Generate datasets from quotes | [`docs/generation/README.md`](docs/generation/README.md) |
| Understand data format and ordering | [`docs/format/README.md`](docs/format/README.md) |
| Train with QLoRA | [`docs/training/README.md`](docs/training/README.md) |
| Configure models for inference | [`docs/inference/README.md`](docs/inference/README.md) |
| Understand architecture and rationale | [`docs/architecture/README.md`](docs/architecture/README.md) |
| Lessons learned (failure modes + fixes) | [`docs/lessons.md`](docs/lessons.md) |
| Inspect example data and artifacts | [`data/README.md`](data/README.md) |

---

**Status:** Production-ready  
**Repository:** https://github.com/thistleknot/spo-reasoning-training-regimen
