# Data Directory Index

Complete guide to example data and training artifacts.

## Files

### Example Inputs
- **`sample_quotes.txt`** — 10 classic quotes ready for processing
  - Format: One quote per line
  - Use with: `Pipeline.load_quotes_from_file()`

### Example Outputs (Seeing is Believing)
- **`SEEING_IS_BELIEVING_EXAMPLES.md`** — 5 complete end-to-end examples
  - Shows input quotes and full model output
  - Formatted with pedagogical order (Non-Entailed → Entailed → Throughline)
  - Includes confidence scores and evidence tags
  - Demonstrates reasoning quality and format

- **`examples_training_format.jsonl`** — Same 5 examples in JSONL format
  - Ready for training or fine-tuning
  - Format: `{"input_text": "...", "output_text": "..."}`
  - Compatible with transformers and QLoRA

### Production Training Data
- **`train_clean_for_model_967.jsonl`** — 967 cleaned training records
  - Verified mojibake-free
  - Pedagogical order (Non-Entailed → Entailed → Throughline)
  - Real training dataset used in prior sessions

## How to Use

### Generate Your Own Examples

```python
from src.synthetic_generator import SyntheticReasoningGenerator
from pathlib import Path

# Option 1: Simple generation
gen = SyntheticReasoningGenerator()
quotes = [
    "Your quote here",
    "Another quote",
]
examples = gen.generate_from_quotes(quotes)
gen.export_to_jsonl("data/my_dataset.jsonl")

# Option 2: Full pipeline
from src.pipeline import Pipeline, PipelineConfig

config = PipelineConfig(
    generate_dataset=True,
    quotes_path=Path("data/sample_quotes.txt"),
)
pipeline = Pipeline(config)
pipeline.run()
# Outputs: data/train.jsonl, data/validation.jsonl
```

### Train with Generated Data

```python
from datasets import load_dataset

# Load example dataset
dataset = load_dataset("json", data_files="data/examples_training_format.jsonl")

# Use with transformers + QLoRA
# See ../README.md for training setup
```

### Understand the Format

Each example has:

1. **Input**: Quote (just the text)
   ```
   "By the pricking of my thumbs, Something wicked this way comes."
   ```

2. **Output**: Structured reasoning (pedagogical order)
   ```
   Non-Entailed Premises:
     [facts mentioned but not supporting main inference]
   
   Entailed Premises:
     [facts that support the throughline]
   
   Throughline:
     [synthesized reasoning/conclusion]
   ```

3. **Evidence Tags**: `(observed|inferred, confidence=X)`
   - `observed` = explicit in text (typically 1.0)
   - `inferred` = derived from text (0.3-0.95)
   - Confidence = model's uncertainty estimate

## Data Format Details

### Pedagogical Order (Why This Way?)

Non-Entailed first because:
1. **Teaches negative inference** — Model learns what's irrelevant
2. **Contrastive learning** — Discriminate true from false facts
3. **Better convergence** — Explicit negatives improve training

### Evidence Tags as Training Features

- NOT confidence scores in training data
- Model learns to distinguish observed vs inferred
- Confidence emerges at inference time
- SPO optimizes confidence via downstream rewards

### Triplet Format

Each premise is: `subject | relation (tag, confidence=X) | object`

Example:
```
something | is (inferred, confidence=0.85) | wicked
```

This is reversible and enables:
- Graph storage (subject → relation → object)
- Synset collapse (entity normalization)
- Path finding (multi-hop reasoning)
- Fact retrieval (match patterns)

## Statistics

| File | Records | Format | Purpose |
|------|---------|--------|---------|
| `sample_quotes.txt` | 10 | Plain text | Quick testing |
| `examples_training_format.jsonl` | 5 | JSONL | Demonstration |
| `train_clean_for_model_967.jsonl` | 967 | JSONL | Production training |

## See Also

- **`../README.md`** — Usage guide and architecture
- **`../docs/FORMAT_SPECIFICATION.md`** — Complete format spec
- **`../docs/ARCHITECTURE_CONFIGURABLE_FORMAT.md`** — Design principles
- **`../src/synthetic_generator.py`** — Generation implementation

## Next Steps

1. **Try the examples** — Load and inspect `examples_training_format.jsonl`
2. **Generate your own** — Use `sample_quotes.txt` with pipeline
3. **Train a model** — Feed generated JSONL to QLoRA
4. **Optimize confidence** — Use SPOTrainer for calibration
