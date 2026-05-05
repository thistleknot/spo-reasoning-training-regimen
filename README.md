# SPO Reasoning Training Regimen

Build quote -> structured reasoning dataset -> QLoRA adapter -> optional downstream confidence calibration.

This repo packages the full workflow for turning raw quotes into structured reasoning examples, training a model to emit that reasoning in a pedagogical order, and optionally calibrating confidence with Soft Policy Optimization (SPO). The root README is the front door; the deeper mechanics live in the docs folders linked below.

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
    A[Quotes] --> B[Generation<br/>LLM or templates]
    B --> C[Validated reasoning examples]
    C --> D[Training JSONL<br/>pedagogical order]
    D --> E[QLoRA adapter]
    E --> F[Inference on new quotes]
    F --> G[SPO calibration<br/>optional]
```

## Fast paths

| If you want to... | Start here |
|---|---|
| Understand the whole workflow fast | `docs/quickstart/README.md` |
| Configure an LLM for dataset generation | `docs/generation/README.md` |
| Train with QLoRA | `docs/training/README.md` |
| Configure model inference | `docs/inference/README.md` |
| Understand the exact format contract | `docs/format/README.md` |
| See finished examples before touching code | `data/SEEING_IS_BELIEVING_EXAMPLES.md` |

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
5. Add SPO calibration if you want confidence to mean something downstream

## Synthetic generation -> training example

The same quote moves through a few different shapes before it becomes a training row.

### 1. Synthetic generation scaffold

```text
Input:
"By the pricking of my thumbs, Something wicked this way comes."

Completion:
Throughline:
  When one feels a premonition or intuitive sense, something bad is approaching.

Entailed Premises:
  - something | is (inferred, confidence=0.75) | wicked
  - something | is (inferred, confidence=0.8) | coming

Non-Entailed Premises:
  - thumbs | are (observed, confidence=1.0) | pricking
```

### 2. Preprocessed structured record

The preprocessing step strips hybrid markdown wrappers, extracts named sections, normalizes missing values, and turns the scaffold into a clean structured record:

```json
{
  "quote": "By the pricking of my thumbs, Something wicked this way comes.",
  "entailed_premises": [
    "something | is (inferred, confidence=0.75) | wicked",
    "something | is (inferred, confidence=0.8) | coming"
  ],
  "non_entailed_premises": [
    "thumbs | are (observed, confidence=1.0) | pricking"
  ],
  "syllogism": "When one feels a premonition or intuitive sense, something bad is approaching."
}
```

### 3. Training row actually fed to the model

The serializer then converts that structured record into the pedagogical training format used for QLoRA. Evidence tags stay; numeric confidence is stripped so the base model learns premises and throughline text rather than frozen scores:

```text
"By the pricking of my thumbs, Something wicked this way comes."

Non-Entailed Premises:
thumbs | are (observed) | pricking

Entailed Premises:
something | is (inferred) | wicked
something | is (inferred) | coming

Throughline:
When one feels a premonition or intuitive sense, something bad is approaching.
```

## Training regimen families

The repo now supports three adjacent supervised tasks over the same synthetic source data:

| Regimen | Input | Output | Numeric confidence |
|---|---|---|---|
| Base reasoning | Quote | Non-entailed + entailed premises + throughline | Stripped |
| Facts with confidence | Quote | Non-entailed + entailed premises | Preserved |
| Syllogism with confidence | Quote + confidence-bearing facts | Throughline + aggregate confidence | Preserved |

The first regimen teaches the reasoning structure cleanly. The other two turn the original synthetic numerics into follow-on tasks instead of freezing them into the base target.

You can build the two follow-on datasets with:

```bash
python -m src.build_training_regimens \
  --input path/to/confidence_rich_source.jsonl \
  --output data/train_facts_with_confidence.jsonl \
  --regimen facts_with_confidence

python -m src.build_training_regimens \
  --input path/to/confidence_rich_source.jsonl \
  --output data/train_syllogism_with_confidence.jsonl \
  --regimen syllogism_with_confidence
```

## Why the format matters

The core design choice is that the repo uses different orders for different stages of the pipeline.

| Stage | Order | Why |
|---|---|---|
| Generation | Throughline -> Entailed -> Non-Entailed | Natural reasoning flow for the generating LLM |
| Training | Non-Entailed -> Entailed -> Throughline | Negative inference first; better contrastive signal |
| Inference | Non-Entailed -> Entailed -> Throughline | The model tends to emit what it was taught |

This matters because the training target is not just "state the answer." It teaches the model what does **not** support the conclusion before teaching what does. That is the job of **non-entailed premises**.

For the full specification, examples, and exact `Input` / `Completion` layout, read `docs/format/README.md`.

## How the data gets cleaned before training

The repo's filtering/cleaning path is conservative and explicit rather than magic:

1. **Parse the hybrid record** via `src/preprocess_training_data.py`
   - Pull the quote out of `input_text`
   - Extract `Entailed Premises`, `Non-Entailed Premises`, and `Conclusion` / `Syllogism`
2. **Normalize section values**
   - Empty or explicit `N/A` sections become `None` in the structured record
   - Missing markdown wrappers are handled by the section extractor when possible
3. **Preserve reasoning signal**
    - Triplets stay intact
    - Evidence tags are preserved
    - Numeric confidence can remain in structured synthetic records for audit, but it is stripped from training rows so downstream judges can assign scores later
    - The cleaned corpus in `data/train_clean_for_model_967.jsonl` reflects the confidence-free training target
4. **Drop malformed rows at preprocessing time**
   - If a line fails JSON parsing or record conversion, it increments the preprocessing error count and is not written to the cleaned output
5. **Serialize survivors into training format**
   - `src/serialize_training_format.py` writes the final pedagogical order:
     `Non-Entailed Premises -> Entailed Premises -> Throughline`
   - Empty sections are written back out as explicit `N/A`

In other words, the repo is not training directly on whatever the generator spit out. It parses, normalizes, preserves the reasoning structure, and only then serializes clean training rows without static numeric confidence labels.

## Model configuration: where it lives

One of the prior pain points was making generation-model and inference-model setup too easy to miss. The split is now:

| Need | Where to go |
|---|---|
| Configure an LLM to generate synthetic data | `docs/generation/README.md` |
| Load a fine-tuned adapter or set inference parameters | `docs/inference/README.md` |
| Configure QLoRA training knobs | `docs/training/README.md` |
| Understand why confidence should stay downstream of the base training target | `docs/architecture/README.md` |

## Repo map

```text
spo-reasoning-training-regimen/
├── src/
│   ├── synthetic_generator.py
│   ├── build_training_regimens.py
│   ├── spo_trainer.py
│   ├── pipeline.py
│   ├── training_config.py
│   ├── graph_ontology.py
│   └── ...
├── docs/
│   ├── generation/README.md
│   ├── training/README.md
│   ├── inference/README.md
│   ├── format/README.md
│   ├── architecture/README.md
│   └── quickstart/README.md
└── data/
    ├── sample_quotes.txt
    ├── examples_training_format.jsonl
    ├── examples_facts_with_confidence.jsonl
    ├── examples_syllogism_with_confidence.jsonl
    ├── SEEING_IS_BELIEVING_EXAMPLES.md
    ├── train_clean_for_model_967.jsonl
    ├── train_facts_with_confidence_967.jsonl
    └── train_syllogism_with_confidence_967.jsonl
```

## Seeing-is-believing artifacts

If you want to inspect the output shape before generating anything, start here:

| Artifact | Why it matters |
|---|---|
| `data/SEEING_IS_BELIEVING_EXAMPLES.md` | Human-readable examples of quote -> structured reasoning |
| `data/examples_training_format.jsonl` | The same examples in training-ready JSONL |
| `data/examples_facts_with_confidence.jsonl` | Example follow-on task for premise scoring |
| `data/examples_syllogism_with_confidence.jsonl` | Example follow-on task for throughline scoring |
| `data/sample_quotes.txt` | Easy starter input set |
| `data/train_clean_for_model_967.jsonl` | Larger cleaned training corpus used in prior work |
| `data/train_facts_with_confidence_967.jsonl` | Follow-on fact-confidence regimen |
| `data/train_syllogism_with_confidence_967.jsonl` | Follow-on syllogism-confidence regimen |

## Documentation guide

| Goal | Read |
|---|---|
| Generate datasets from quotes | `docs/generation/README.md` |
| Get running quickly | `docs/quickstart/README.md` |
| Configure models for inference | `docs/inference/README.md` |
| Train with QLoRA | `docs/training/README.md` |
| Understand data format and ordering | `docs/format/README.md` |
| Understand architecture and rationale | `docs/architecture/README.md` |
| Inspect example data and artifacts | `data/README.md` |

## The three phases in one paragraph each

### Phase 1: Generate

Turn raw quotes into structured reasoning examples. This can be done with a hosted LLM, a local model, or plain templates if you want to hand-fill the outputs.

### Phase 2: Train

Take the validated JSONL and fine-tune a base model so it learns the pedagogical reasoning format: non-entailed premises first, then entailed premises, then the throughline.

### Phase 3: Optimize

Apply downstream judging or calibration if you want numeric confidence later. The base model should first learn to emit the right premises and throughline; any scores can be assigned post hoc by a judge or calibration layer.

## Hardware by phase

| Phase | GPU | VRAM | Notes |
|---|---|---|---|
| Generation | Optional | Depends on model | Hosted APIs or local models both work |
| Training | Recommended | 8GB+ | QLoRA keeps smaller models practical |
| SPO | Recommended | 8GB+ | Useful when confidence calibration matters |

## Practical next steps

1. Read `data/SEEING_IS_BELIEVING_EXAMPLES.md` if you want output intuition first
2. Use `docs/generation/README.md` if you need help wiring up OpenAI, Ollama, or Hugging Face
3. Use `docs/format/README.md` if you need the exact prompt and serialization contract
4. Use `docs/training/README.md` when you are ready to fine-tune
5. Use `docs/inference/README.md` when you are ready to load adapters and run new quotes

---

**Status:** Production-ready  
**Repository:** https://github.com/thistleknot/spo-reasoning-training-regimen
