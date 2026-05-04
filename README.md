# SPO Reasoning Training Regimen

**Complete pipeline for generating synthetic reasoning datasets and training models with Soft Policy Optimization.**

## What This Does (5-Minute Overview)

Three-phase workflow to build reasoning models:

```
Phase 1: GENERATE              Phase 2: TRAIN                Phase 3: OPTIMIZE
─────────────────────────      ────────────────────────      ──────────────────
Quotes (text)                  Generated JSONL                Trained Model
    ↓                                ↓                            ↓
LLM extracts reasoning         QLoRA fine-tuning         SPO confidence optimization
(triplets + syllogism)         (pedagogical format)       (reward = correctness × confidence)
    ↓                                ↓                            ↓
ReasoningExamples              Tuned Adapter              Calibrated Model
    ↓                                ↓                            ↓
Training JSONL            Production-ready           Ready for deployment
```

## Quick Start (2 Minutes)

### Install
```bash
git clone https://github.com/thistleknot/spo-reasoning-training-regimen.git
cd spo-reasoning-training-regimen
pip install -r requirements.txt
```

### Generate Examples
```python
from src.synthetic_generator import SyntheticReasoningGenerator

# Without LLM (create templates)
gen = SyntheticReasoningGenerator()
quotes = ["Your quote here", "Another quote"]
examples = gen.generate_from_quotes(quotes)
gen.export_to_jsonl("data/examples.jsonl")
```

### With LLM (OpenAI)
```python
from openai import OpenAI
client = OpenAI(api_key="your-key")

def generate_fn(prompt):
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content

examples = gen.generate_from_quotes(quotes, llm_generate_fn=generate_fn)
```

### Train with QLoRA
```python
# Use the generated data/examples.jsonl with your QLoRA trainer
# See docs/training/ for complete setup
```

## Three Phases Explained

### Phase 1: Synthetic Dataset Generation (`docs/generation/README.md`)

Transform raw quotes into structured reasoning:

```
"By the pricking of my thumbs, Something wicked this way comes."
    ↓
[Model extracts reasoning]
    ↓
Non-Entailed Premises:
  thumbs | are (observed) | pricking

Entailed Premises:
  something | is (inferred, confidence=0.85) | wicked
  
Throughline:
  Physical sensations signal approaching danger
```

**You can use:**
- OpenAI GPT-4
- Local Qwen models
- Anthropic Claude
- HuggingFace models
- Or no LLM (manual templates)

See **`docs/generation/README.md`** for model configuration and examples.

### Phase 2: QLoRA Training (`docs/training/README.md`)

Fine-tune a language model on the generated reasoning data:

```python
# Takes: Generated JSONL + base model (Qwen, Llama, etc.)
# Returns: QLoRA adapter (~10MB) with learned reasoning patterns
# Time: 30 min - 2 hours depending on model size
```

**Features:**
- 4-bit quantization (fits in 8GB VRAM)
- Configurable rank, learning rate, epochs
- Multi-GPU support
- Monitoring with WandB/TensorBoard

See **`docs/training/README.md`** for full QLoRA setup.

### Phase 3: SPO Confidence Optimization (Optional)

Optimize the model's confidence calibration using downstream task rewards:

```python
# Training: Model learns to generate triplets
# SPO: Model learns accurate confidence scores
# Reward: correctness × confidence (only high-confidence correct outputs rewarded)
```

This ensures your model is **confident when right, uncertain when wrong**.

See `src/spo_trainer.py` for implementation.

## Key Insight: Architecture

### Confidence is NOT a Training Label

```
Training Phase:
  Model learns: Generate correct triplets
  Data contains: Evidence tags (observed/inferred) - NO confidence
  
Inference Phase:
  Model generates: Triplets WITH confidence (emergent)
  Example: confidence=0.8 on "something | is | wicked"
  
SPO Phase:
  Model learns: Assign accurate confidence
  Reward signal: Downstream task correctness
  Result: confidence ≈ actual accuracy
```

**Why this matters:**
- Separates structure learning from calibration
- Prevents confidence overfitting to training labels
- Enables better transfer to new tasks
- Model is interpretable (confidence has meaning)

## Data Format: Pedagogical Order

Training data uses: **Non-Entailed → Entailed → Throughline**

Why this order?
1. **Negative inference first** — Model learns what's irrelevant
2. **Contrastive learning** — Discriminate true from false facts
3. **Better convergence** — Explicit negatives improve training

Example:
```json
{
  "input_text": "\"By the pricking of my thumbs...\"",
  "output_text": "Non-Entailed Premises:\n  thumbs | are (observed, confidence=1.0) | pricking\n\nEntailed Premises:\n  something | is (inferred, confidence=0.85) | wicked\n\nThroughline:\n  When one feels a premonition, something bad approaches."
}
```

## Project Structure

```
spo-reasoning-training-regimen/
├── src/
│   ├── synthetic_generator.py    # Quote → ReasoningExample
│   ├── spo_trainer.py            # SPO confidence optimization
│   ├── pipeline.py               # End-to-end orchestration
│   ├── training_config.py        # Format configuration
│   ├── graph_ontology.py         # Triplet storage & retrieval
│   └── [4 more modules]
│
├── docs/
│   ├── generation/README.md      # Phase 1: Dataset generation & LLM config
│   ├── training/README.md        # Phase 2: QLoRA training setup
│   ├── inference/README.md       # Model inference configuration
│   ├── format/README.md          # Format specification
│   ├── architecture/README.md    # Design principles
│   └── quickstart/README.md      # Quick reference
│
├── data/
│   ├── sample_quotes.txt         # Example input quotes
│   ├── examples_training_format.jsonl  # 5 complete end-to-end examples
│   ├── SEEING_IS_BELIEVING_EXAMPLES.md # Example outputs with explanations
│   └── train_clean_for_model_967.jsonl # 967 production training records
│
├── README.md                     # This file (overview)
└── requirements.txt              # Dependencies
```

## Documentation Guide

| Goal | Read |
|------|------|
| Generate datasets from quotes | `docs/generation/README.md` |
| Configure models for inference | `docs/inference/README.md` |
| Train with QLoRA | `docs/training/README.md` |
| Understand data format | `docs/format/README.md` |
| See example outputs | `data/SEEING_IS_BELIEVING_EXAMPLES.md` |
| Understand architecture | `docs/architecture/README.md` |
| Quick reference | `docs/quickstart/README.md` |

## Complete Example Workflow

```python
# 1. GENERATE: Create examples from quotes
from src.synthetic_generator import SyntheticReasoningGenerator
from openai import OpenAI

# Configure LLM
client = OpenAI(api_key="your-key")
def generate_fn(prompt):
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content

# Generate
gen = SyntheticReasoningGenerator()
quotes = ["Your quote 1", "Your quote 2"]
examples = gen.generate_from_quotes(quotes, llm_generate_fn=generate_fn)
gen.export_to_jsonl("data/my_dataset.jsonl")

# 2. TRAIN: Fine-tune with QLoRA (see docs/training/README.md)
# This takes generated JSONL and produces a trained adapter

# 3. OPTIMIZE: Apply SPO (optional, for confidence calibration)
from src.spo_trainer import SPOTrainer, SPOEvaluator
trainer = SPOTrainer(model, tokenizer, SPOEvaluator.composite_score)
# ... SPO training loop ...

# 4. DEPLOY: Use trained model for inference
# Load adapter + use for new reasoning tasks
```

## What You Get

✅ **End-to-end pipeline** — Quotes to reasoning model in ~2 hours
✅ **Multiple model options** — OpenAI, Qwen, Llama, Anthropic, HF
✅ **Pedagogical training format** — Improves convergence and interpretability
✅ **SPO support** — Optimize confidence calibration
✅ **Graph ontology** — Enables fact retrieval and multi-hop reasoning
✅ **5 example outputs** — See exactly what the pipeline produces

## Hardware Requirements

| Phase | GPU | VRAM | Time |
|-------|-----|------|------|
| Generation | Optional | Varies | Minutes-hours (depends on LLM) |
| Training | Recommended | 8GB+ | 30 min - 2 hours |
| SPO | Recommended | 8GB+ | 1-4 hours |

**Note:** Training uses 4-bit quantization so Qwen 0.6B fits in 8GB VRAM.

## Common Patterns

### Template-Based (No LLM)
```python
gen = SyntheticReasoningGenerator()
examples = gen.generate_from_quotes(quotes)  # Creates empty templates
gen.export_to_json("data/templates.json")
# Then manually fill in each example
```

### Batch Generation (Process Large Datasets)
```python
for i in range(0, len(all_quotes), batch_size):
    batch = all_quotes[i:i+batch_size]
    examples = gen.generate_from_quotes(batch, llm_generate_fn=generate_fn)
    gen.export_to_jsonl(f"data/batch_{i}.jsonl")
```

### With Config System (Parametrize Everything)
```python
from src.training_config import PipelineConfig, TrainingFormat, PremiseOrdering
config = PipelineConfig(
    training_format=TrainingFormat(
        premise_ordering=PremiseOrdering.PEDAGOGICAL,
    )
)
# Use config throughout pipeline
```

## See Also

- **`docs/generation/README.md`** — How to configure models for generation
- **`docs/training/README.md`** — How to set up QLoRA training
- **`docs/inference/README.md`** — Model configuration details
- **`data/SEEING_IS_BELIEVING_EXAMPLES.md`** — Example outputs
- **`src/synthetic_generator.py`** — Implementation details

## Next Steps

1. **Try the examples** — Load `data/examples_training_format.jsonl` and inspect
2. **Generate your own** — Follow `docs/generation/README.md`
3. **Train a model** — Follow `docs/training/README.md`
4. **Optimize confidence** — Use SPO trainer for calibration
5. **Deploy** — Use trained model for reasoning tasks

---

**Status:** Production-ready ✅
**Repository:** https://github.com/thistleknot/spo-reasoning-training-regimen
