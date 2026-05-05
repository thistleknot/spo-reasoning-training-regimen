# Format Examples: Generation vs Training

This document shows the **exact same data** in two different orderings depending on the use case.

---

## Example 1: "By the pricking of my thumbs..."

### GENERATION FORMAT (LLM Output / Inference)
**What the model produces when asked to generate reasoning**

```
Throughline:
  When one feels a premonition or intuitive sense, something bad is approaching.

Entailed Premises:
  - something | is (inferred, confidence=0.75) | wicked
  - something | is (inferred, confidence=0.8) | coming

Non-Entailed Premises:
  - thumbs | are (observed, confidence=1.0) | pricking
```

### TRAINING FORMAT (Model Input / Fine-tuning)
**What goes into the model DURING training**

```
"By the pricking of my thumbs, Something wicked this way comes."

Non-Entailed Premises:
  - thumbs | are (observed) | pricking

Entailed Premises:
  - something | is (inferred) | wicked
  - something | is (inferred) | coming

Throughline:
  When one feels a premonition or intuitive sense, something bad is approaching.
```

---

## Example 2: "Be yourself..."

### GENERATION FORMAT
```
Throughline:
  One should embrace their own uniqueness and not attempt to imitate others.

Entailed Premises:
  - people | are (observed, confidence=1.0) | unique individuals
  - copying others | is (observed, confidence=1.0) | redundant
  - authenticity | is (observed, confidence=1.0) | the only way to be oneself

Non-Entailed Premises:
  - social conformity | is (observed, confidence=1.0) | undesirable
  - everyone else | is (observed, confidence=1.0) | already taken
```

### TRAINING FORMAT
```
"Be yourself; everyone else is already taken."

Non-Entailed Premises:
  - social conformity | is (observed) | undesirable
  - everyone else | is (observed) | already taken

Entailed Premises:
  - people | are (observed) | unique individuals
  - copying others | is (observed) | redundant
  - authenticity | is (observed) | the only way to be oneself

Throughline:
  One should embrace their own uniqueness and not attempt to imitate others.
```

---

## Key Differences

### GENERATION FORMAT (Logical Order)
- **First**: Throughline (conclusion)
- **Second**: Entailed Premises (supporting evidence)
- **Third**: Non-Entailed Premises (non-supporting evidence)
- **Use case**: When the model is generating/inferring (human-readable, logical flow)
- **Order rationale**: Humans think: "Here's my conclusion, here's why, here's what I rejected"

### TRAINING FORMAT (Pedagogical Order)
- **First**: Non-Entailed Premises (false/misleading)
- **Second**: Entailed Premises (true/supporting)
- **Third**: Throughline (conclusion)
- **Use case**: When the model is being trained/fine-tuned
- **Order rationale**: Model learns: "Here's what's false, here's what's true, here's the conclusion" (contrastive learning)

---

## Field Names (Non-Negotiable)

- `Throughline` (not "Conclusion", not "Syllogism")
- `Entailed Premises` (not "Entailed Facts", not "True Premises")
- `Non-Entailed Premises` (canonical label for negative examples)

---

## Empty Sections

If a section has no entries, use explicit `N/A`:

```
Non-Entailed Premises:
  N/A

Entailed Premises:
  - fact1 | relation | object1
  - fact2 | relation | object2

Throughline:
  Some conclusion.
```

Never leave sections blank or use other variations like "none", "empty", or "null".

---

## Confidence Scores (Generation-Time Only)

Synthetic generation can retain confidence scores and tags for audit:

```
subject | relation (tag, confidence=X) | object
```

Training rows drop the numeric score and keep the tag:

Example:
```
something | is (inferred) | wicked
people | are (observed) | unique individuals
```

---

## Training Data Format (What's Actually Used)

The repo's production dataset uses **TRAINING FORMAT** with confidence-free targets:

```
File: data/train_clean_for_model_967.jsonl

Each line: {"input_text": "...", "output_text": "..."}

input_text = explicit instruction prompt over the quote
output_text = Non-Entailed + Entailed + Throughline
```

---

## Generation/Inference Format (What Model Produces)

When the trained model is asked to generate reasoning for a NEW quote, the base target remains confidence-free and any numeric scoring is applied later by a judge or calibrator:

```
Input:  prompted quote instruction wrapped in the model chat template
Output: [Training target order: Non-Entailed → Entailed → Throughline]
```

The model is trained to produce the pedagogical ordering it saw during
fine-tuning, while generation/inference still uses the same chat-turn surface as
training.

---

## Implementation in Code

### Training Pipeline
```python
# 1. Preprocess from hybrid to structured dicts
from src.preprocess_training_data import preprocess_training_dataset
preprocess_training_dataset("hybrid_data.jsonl", "structured.jsonl")

# 2. Serialize to training format (Non-Entailed first)
from src.serialize_training_format import convert_preprocessed_to_training
convert_preprocessed_to_training("structured.jsonl", "train_data.jsonl")

# 3. Train the model with train_data.jsonl
```

### Generation/Inference
```python
# After training, the model will produce Generation Format:
# Input: "Quote here"
# Output:
#   Throughline: ...
#   Entailed Premises: ...
#   Non-Entailed Premises: ...
```

---

## Why Two Orderings?

**Contrastive Learning**: Training with Non-Entailed FIRST teaches the model to:
1. Recognize what's false/misleading
2. Distinguish it from what's true
3. Apply reasoning to reach the throughline

This produces better negative inference capability than other ordering strategies.

**Human Readability**: Generation output uses Throughline first because humans understand conclusions before supporting details.

---

## Summary

- **Same data**, two orderings
- **Training**: Non-Entailed → Entailed → Throughline (pedagogical)
- **Generation**: Throughline → Entailed → Non-Entailed (logical)
- **Field name**: Throughline (locked)
- **Always preserve in training**: Evidence tags
- **Keep numeric confidence downstream**: add it later if needed
- **Always explicit**: Use N/A for empty sections
