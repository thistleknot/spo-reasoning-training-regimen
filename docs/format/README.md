# Data Format Specification

Complete specification for the SPO reasoning training format.

## Overview

The training format encodes semantic reasoning as structured triplets (subject-relation-object) with evidence tags and confidence scores. This enables:
- **Negative inference training** (what's NOT entailed improves discrimination)
- **Syllogistic reasoning** (how premises lead to conclusions)
- **Confidence calibration** (via SPO optimization)
- **Graph storage** (triplets are naturally graph-compatible)

## Format Versions

### Pedagogical Order (RECOMMENDED for training)

**Purpose:** Teaches the model what's relevant (contrastive learning)

```
[NON-ENTAILED]
subject | relation (evidence_tag, confidence=X) | object
subject | relation (evidence_tag, confidence=Y) | object
...

[ENTAILED]
subject | relation (evidence_tag, confidence=X) | object
subject | relation (evidence_tag, confidence=Y) | object
...

[CONCLUSION]
The syllogistic conclusion that maximizes entailed premises.
```

**Example:**
```
[NON-ENTAILED]
thumbs | are (observed, confidence=1.0) | sore
premonition | comes (inferred, confidence=0.4) | from fate alone

[ENTAILED]
something | is (inferred, confidence=0.85) | wicked
approaching | indicates (inferred, confidence=0.9) | danger
premonition | signals (observed, confidence=1.0) | something wicked

[CONCLUSION]
When one feels a premonition through physical sensation, something wicked or dangerous approaches.
```

**Why this order?**
- Non-entailed premises first teaches negative inference (what to exclude)
- Entailed premises teach supporting evidence (what to include)
- Better convergence than random order
- More interpretable outputs

### Logical Order (For inference output)

**Purpose:** Output in logical sequence (throughline → supporting → rejected)

```
[THROUGHLINE]
The core abductive hypothesis...

[ENTAILED]
subject | relation (evidence_tag, confidence=X) | object
...

[NON-ENTAILED]
subject | relation (evidence_tag, confidence=Y) | object
...
```

### Entailed Only (LEGACY)

```
[ENTAILED]
subject | relation (evidence_tag, confidence=X) | object
...

[CONCLUSION]
The hypothesis.
```

## Evidence Tags

Each triplet includes an evidence tag:

- **observed** (confidence = 1.0)
  - Explicit in the source text
  - Example: `thumbs | are (observed, confidence=1.0) | pricking`

- **inferred** (confidence ∈ [0.3, 0.9])
  - Derived from context, not explicit
  - Example: `something | is (inferred, confidence=0.85) | wicked`

**Confidence Score Semantics:**
- 1.0 = Certain (observed in text)
- 0.9 = Very likely (strong inference)
- 0.7-0.8 = Probable (reasonable inference)
- 0.5-0.6 = Weak (speculative but plausible)
- 0.3-0.4 = Unlikely (included as negative example)

## Triplet Structure

```
subject | relation (evidence_tag, confidence=X.XX) | object
```

### Subject & Object
- Any nominal entity: person, place, thing, property, event
- Normalized for synset collapse (entity linking)
- Example: "something", "premonition", "danger", "thumbs"

### Relation
- Predicate connecting subject to object
- Typically verb or verb phrase
- Example: "is", "causes", "indicates", "pricking"

### Tags
- **evidence_tag** = "observed" or "inferred"
- **confidence** = Float from 0.0 to 1.0

## Full Example: Training Record

```json
{
  "input_text": "By the pricking of my thumbs, Something wicked this way comes.",
  "output_text": "[NON-ENTAILED]\nthumbsare (observed, confidence=1.0) | sore\npremonition | comes (inferred, confidence=0.3) | from fate\n\n[ENTAILED]\nsomething | is (inferred, confidence=0.85) | wicked\napproaching | signals (inferred, confidence=0.9) | danger\npremonition | indicates (observed, confidence=1.0) | something wicked\n\n[CONCLUSION]\nWhen one feels a premonition through physical sensation, something wicked or dangerous approaches."
}
```

## Serialization

### JSONL Format (for training)
```
{"input_text": "quote", "output_text": "[NON-ENTAILED]..."}
{"input_text": "quote2", "output_text": "[NON-ENTAILED]..."}
```

### Python Objects
```python
from pydantic import BaseModel
from typing import List, Set

class TripletItem(BaseModel):
    subject: str
    relation: str
    object: str
    evidence_tag: str  # "observed" or "inferred"
    confidence: float  # 0.0-1.0

class ReasoningExample(BaseModel):
    quote: str
    non_entailed_premises: Set[TripletItem]
    entailed_premises: Set[TripletItem]
    syllogism: str
```

## Configuration Options

The pipeline supports three format variants via `TrainingFormat` enum:

```python
class TrainingFormat(Enum):
    PEDAGOGICAL = "pedagogical"      # Non-Entailed → Entailed → Conclusion
    LOGICAL = "logical"              # Conclusion → Entailed → Non-Entailed
    ENTAILED_ONLY = "entailed_only"  # Entailed only (no negatives)
```

**Selection guidance:**
- **PEDAGOGICAL** (default): Best for training convergence
- **LOGICAL**: Better matches human reading order
- **ENTAILED_ONLY**: Faster training, slightly lower quality

## Confidence NOT in Training Data

**Critical Design Decision:**
- Training data contains NO confidence scores
- Confidence emerges during inference
- SPO then optimizes confidence calibration

**Why?**
- Prevents confidence overfitting to training labels
- Enables better transfer to new domains
- Natural uncertainty estimation

## Deterministic Ordering

To ensure reproducible training:

```python
# Hard-code sort key before writing
records.sort(key=lambda r: (
    r.avg_char_pos,       # Position in original text
    r.sentence_index,     # Sentence number
    r.triplet_index       # Triplet order
))
```

Average character position computed on original pre-cleaning text:
```python
avg_char_pos = original.find(sentence) + len(sentence) / 2
```

## Validation

All training records must pass:

```python
from pydantic import ValidationError

try:
    example = ReasoningExample(
        quote=quote_text,
        non_entailed_premises=non_entailed,
        entailed_premises=entailed,
        syllogism=conclusion
    )
except ValidationError as e:
    print(f"Invalid record: {e}")
    # Discard and continue
```

Invalid records are DISCARDED, never written with defaults.

## Token Length Filtering

When records vary materially in token length:

```python
import numpy as np
from scipy import stats

# Tokenize
tokens_per_record = [len(tokenizer.encode(r)) for r in records]

# Log-normalize and Box-Cox transform
log_lengths = np.log1p(tokens_per_record)
transformed, lambda_param = stats.boxcox(log_lengths)

# Compute robust center
median_transformed = np.median(transformed)
mad = stats.median_abs_deviation(transformed)

# Discard outliers: median ± 2*MAD
keep_idx = np.abs(transformed - median_transformed) <= 2 * mad
filtered = [r for i, r in enumerate(records) if keep_idx[i]]
```

Result: Removes extremely long/short records while preserving distribution.

## Integration with Training

In `src/serialize_training_format.py`:

```python
from training_config import TrainingFormat

formatter = TrainingFormat.PEDAGOGICAL  # Select format

# Serialize
records_serialized = formatter.serialize_batch(reasoning_examples)

# Export to JSONL
with open("train.jsonl", "w") as f:
    for record in records_serialized:
        f.write(json.dumps(record) + "\n")
```

In training script:

```python
# Load JSONL
dataset = load_dataset("json", data_files="train.jsonl", split="train")

# Data is ready for QLoRA training
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=lambda examples: {
        "text": examples["output_text"]
    },
    # ... other args
)
```

## Troubleshooting

**Issue: Model outputs wrong format**
- Check confidence scores are 0.0-1.0 (not 0-100)
- Verify evidence_tags are lowercase ("observed" not "Observed")
- Use constrained decoding if format violations persist

**Issue: Token length causes OOM**
- Apply token length filtering (see above)
- Reduce batch size
- Use gradient accumulation

**Issue: Confidence not calibrated**
- Train with PEDAGOGICAL order first
- Then run SPO optimization phase
- Verify reward computation: correctness × confidence

## References

- Original format spec: `src/training_config.py`
- Serialization: `src/serialize_training_format.py`
- Data pipeline: `src/synthetic_generator.py`
- SPO optimization: `src/spo_trainer.py`
- Examples: `data/SEEING_IS_BELIEVING_EXAMPLES.md`
