# SPO Training Regimen

Unified infrastructure for **synthetic reasoning dataset generation** + **SPO training**.

**Goal:** Generate synthetic reasoning examples → Train model → Optimize confidence via SPO

## What This Does

This repository combines two complementary processes:

1. **Synthetic Reasoning Dataset Generation** (`synthetic_generator.py`)
   - Takes raw quotes as input
   - Extracts implicit reasoning: non-entailed premises, entailed premises, syllogism
   - Outputs training-ready JSONL format

2. **SPO Training Infrastructure** (`spo_trainer.py`)
   - Implements Soft Policy Optimization reward signal
   - Learns accurate confidence calibration
   - Evaluates reasoning quality (triplet correctness, syllogism coherence)

3. **End-to-End Pipeline** (`pipeline.py`)
   - Orchestrates: generation → validation → training → evaluation

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Input Data
Create `data/sample_quotes.txt` with one quote per line:
```
"By the pricking of my thumbs, Something wicked this way comes."
"Here is a lesson in creative writing. First rule: Do not use semicolons."
```

### 3. Run Pipeline
```bash
python -m src.pipeline
```

This will:
- Generate synthetic reasoning examples from quotes
- Split into train/val sets
- Export JSONL format ready for QLoRA training
- Generate validation report

### 4. Train with QLoRA
Use the generated `data/train.jsonl` with your QLoRA setup.

### 5. Fine-tune with SPO (Optional)
```python
from src.spo_trainer import SPOTrainer, SPOEvaluator

trainer = SPOTrainer(
    model=model,
    tokenizer=tokenizer,
    evaluation_fn=SPOEvaluator.composite_score,
)

# Training step
metrics = trainer.training_step(batch, ground_truth)
```

## Core Components

### Synthetic Dataset Generation
**File:** `src/synthetic_generator.py`

Generates structured reasoning examples from quotes:
```python
from src.synthetic_generator import SyntheticReasoningGenerator

gen = SyntheticReasoningGenerator()
examples = gen.generate_from_quotes(quotes, llm_generate_fn=your_llm)
gen.export_to_jsonl("data/train.jsonl")
```

Outputs triplet-based reasoning with evidence tags (observed/inferred).

### Training Format Configuration
**File:** `src/training_config.py`

Parametrize training pipeline:
- **PremiseOrdering**: PEDAGOGICAL | LOGICAL | ENTAILED_ONLY
- **EntityNormalization**: Synset collapse, predicate equivalence
- **GraphTraversal**: Confidence thresholds, path depth
- **JudgeConfig**: LLM judge for post-hoc synthesis

### Graph Ontology
**File:** `src/graph_ontology.py`

Triplet storage and retrieval:
- Triplet: `subject | predicate (tag, confidence) | object`
- Entity normalization via synset collapse
- Multi-hop path traversal
- Fact extraction (entailed vs non-entailed)

### SPO Training & Evaluation
**File:** `src/spo_trainer.py`

Optimize confidence calibration:
- **SPOTrainer**: Reward high-confidence correct outputs
- **SPOEvaluator**: Score triplet correctness and syllogism quality
- **Loss function**: `-E[reward * log p(output)]`

### End-to-End Pipeline
**File:** `src/pipeline.py`

Orchestrates full workflow with configurable paths.

## Architecture Insight: Confidence as SPO Signal

**Core Insight:** Confidence is NOT a training label—it emerges during inference and is optimized via SPO.

### How It Works

1. **Training Phase** (synthetic_generator.py + training_config.py)
   - Model learns to generate triplets with evidence tags (observed/inferred)
   - Training data contains NO confidence scores
   - Focus: structural correctness

2. **SPO Phase** (spo_trainer.py)
   - Model generates output with confidence scores extracted from triplets
   - SPO reward: `correctness × confidence`
   - High-confidence correct outputs receive high reward
   - Low-confidence or incorrect outputs receive low reward
   - Result: Model learns to assign accurate confidence

3. **Downstream Use**
   - For retrieval: confidence guides which triplets to use in graph
   - For reasoning: confidence signals uncertainty in the model's reasoning
   - For calibration: SPO ensures confidence matches actual accuracy

### Why This Works

- Separates concerns: training (structure) vs SPO (calibration)
- Prevents confidence score leakage into training (only emergent)
- Enables contrastive learning (non-entailed premises first)
- Improves model interpretability (confidence is justifiable)

See `ARCHITECTURE_CONFIGURABLE_FORMAT.md` for full design details.

## Data Format

### Input (Quote)
```
"By the pricking of my thumbs, Something wicked this way comes."
```

### Generated Reasoning Example
```python
ReasoningExample(
    quote="By the pricking of my thumbs...",
    non_entailed_premises=[
        TripletItem(subject="thumbs", relation="are", object_="numb", tag="inferred")
    ],
    entailed_premises=[
        TripletItem(subject="something", relation="is", object_="wicked", tag="inferred"),
        TripletItem(subject="something", relation="is", object_="coming", tag="inferred")
    ],
    syllogism="When one feels a premonition, something bad approaches."
)
```

### Training JSONL Output (Pedagogical Order)
```json
{
  "input_text": "\"By the pricking of my thumbs, Something wicked this way comes.\"",
  "output_text": "Non-Entailed Premises:\n  thumbs | are (inferred, confidence=0.5) | numb\n\nEntailed Premises:\n  something | is (inferred, confidence=0.8) | wicked\n  something | is (inferred, confidence=0.8) | coming\n\nThroughline:\n  When one feels a premonition, something bad approaches."
}
```

**Key:** Non-Entailed → Entailed → Throughline (pedagogical, for contrastive learning)

## Configuration Examples

### Minimal Config (Data Generation Only)
```python
from src.pipeline import Pipeline, PipelineConfig
from pathlib import Path

config = PipelineConfig(
    generate_dataset=True,
    quotes_path=Path("data/sample_quotes.txt"),
)
pipeline = Pipeline(config)
pipeline.run()
```

### Full Training Config
```python
config = PipelineConfig(
    generate_dataset=True,
    quotes_path=Path("data/quotes.txt"),
    model_name="Qwen/Qwen3-0.6B",
    batch_size=2,
    num_epochs=3,
    use_spo=False,  # Set True for SPO fine-tuning
)
```

### SPO-Enabled Config
```python
config = PipelineConfig(
    # ... same as above
    use_spo=True,
    spo_beta=0.1,  # KL divergence penalty
    spo_epochs=1,
)
```

## Project Structure

```
src/
  synthetic_generator.py       # Quote → ReasoningExample (triplets + syllogism)
  training_config.py           # Configurable format system
  graph_ontology.py            # Triplet storage and fact traversal
  spo_trainer.py               # SPO reward and evaluation
  pipeline.py                  # End-to-end orchestration
  data.py                      # [Reference] Data loading for training

data/
  sample_quotes.txt            # Example input quotes
  train.jsonl                  # Generated training data (pedagogical format)
  validation.jsonl             # Validation split
  synthetic_dataset.jsonl      # Generated synthetic examples

docs/
  QUICKSTART.md                            # Getting started
  FORMAT_SPECIFICATION.md                  # Format details
  FORMAT_EXAMPLES_GENERATION_VS_TRAINING.md # Format differences
  ARCHITECTURE_CONFIGURABLE_FORMAT.md      # Design principles
  PROJECT_STRUCTURE.md                     # File guide
```

## Next Steps

1. **Extend synthetic generation** — Integrate your LLM for better premise extraction
2. **Train on your data** — Use generated JSONL with your QLoRA setup
3. **Implement SPO** — Optimize confidence calibration on downstream tasks
4. **Deploy inference** — Use trained model for reasoning tasks

## Key Design Principles

1. **Confidence is emergent** — Model learns to generate triplets, SPO optimizes confidence
2. **Evidence tags are features** — Observed vs inferred are training inputs
3. **Pedagogical order improves training** — Non-Entailed → Entailed → Throughline teaches contrastive reasoning
4. **Syllogism is optional** — Can be LLM-judged or model-generated
5. **Premise ordering is configurable** — Switch between PEDAGOGICAL, LOGICAL, ENTAILED_ONLY

## References

- **Soft Policy Optimization:** Optimize model confidence via downstream task rewards
- **Contrastive Learning:** Non-entailed premises teach negative inference
- **Graph Ontology:** Triplet format enables fact retrieval and path finding
- **Pedagogical Format:** Improves convergence and model interpretability
