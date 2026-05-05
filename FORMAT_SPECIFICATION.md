# Format Specification: Hybrid YAML/Markdown for Semantic Reasoning

**Status**: CANONICAL PRODUCTION STANDARD  
**Version**: 2.0 (Dual Ordering)  
**Date**: 2026-05-03  
**Author**: Josep Hua  
**Rationale**: Clean, readable, LLM-friendly format that balances human readability with model expectations. Supports two orderings: Generation (logical order) and Training (pedagogical order).

---

## Core Specification

### Data Order: Generation vs Training

The **same data** is presented in two different orders depending on context:

#### Generation Format (LLM Output / Inference)
**Logical reasoning order**: Conclusion first, then supporting premises

```
Throughline:
  When one feels a premonition or intuitive sense, something bad is approaching.

Entailed Premises:
  - something | is (inferred, confidence=0.75) | wicked
  - something | is (inferred, confidence=0.8) | coming

Non-Entailed Premises:
  - thumbs | are (observed, confidence=1.0) | pricking
```

#### Training Format (Model Input / Training Data)
**Pedagogical order**: Negative inference first (false premises), then true premises, then conclusion

```
Non-Entailed Premises:
  - thumbs | are (observed, confidence=1.0) | pricking

Entailed Premises:
  - something | is (inferred, confidence=0.75) | wicked
  - something | is (inferred, confidence=0.8) | coming

Throughline:
  When one feels a premonition or intuitive sense, something bad is approaching.
```

### Input Format (What Goes Into the Model)

```
"Quote text or statement here"

Non-Entailed Premises:
  - premise_1 | relation (tag, confidence=X) | object
  - premise_2 | relation (tag, confidence=Y) | object

Entailed Premises:
  - premise_3 | relation (tag, confidence=Z) | object
  - premise_4 | relation (tag, confidence=W) | object

Throughline:
  Text of reasoning or N/A if none
```

**Why this order?**: Model first sees false premises (negative inference context), then learns true premises, then generates throughline. This teaches the model to distinguish what's false before what's true.

### Output Format (What the Model Generates)

When asked to generate reasoning, the output order is **Generation Format** (logical order):

```
Throughline:
  Text of conclusion or N/A

Entailed Premises:
  - premise | relation (confidence: X) | object
  - premise | relation (confidence: Y) | object

Non-Entailed Premises:
  - false_premise | relation (confidence: X) | object
```

**Components**:
- **Throughline**: The reasoned conclusion (renamed from "Conclusion")
- **Entailed Premises**: Markdown heading, bullet list of true premises
- **Non-Entailed Premises**: Markdown heading, bullet list of false premises
- **Formatting**: 2-space indent, bullet markers, confidence scores

---

## Design Principles

### 1. Two Orderings for Different Stages
- **Generation (LLM Output)**: Throughline → Entailed → Non-Entailed (logical order)
- **Training (Model Input)**: Non-Entailed → Entailed → Throughline (pedagogical order)
- **Why**: Matches human reasoning flow in generation, but teaches negative inference during training

### 2. Field Naming
- `Throughline` (not "Conclusion", not "Syllogism")
- `Entailed Premises` (not "Entailed Facts", not "True Premises")
- `Non-Entailed Premises` (canonical label for negative examples)

### 3. Markdown Over Brackets
- ❌ DO NOT use: `[ENTAILED]` / `[NON-ENTAILED]` / `[THROUGHLINE]`
- ✅ DO use: `Entailed Premises:` / `Non-Entailed Premises:` / `Throughline:`
- **Why**: Cleaner, more familiar to LLMs, human-readable structure

### 4. Bullet Lists Over Raw Text
- ❌ DO NOT write: `entailed_triplet_1\nentailed_triplet_2\n...`
- ✅ DO use: `  - triplet_1\n  - triplet_2\n...`
- **Why**: Explicit structure, token-efficient (2-3 lines per premise), clear boundaries

### 5. Explicit N/A for Empty Sections
- ❌ DO NOT leave empty: `Non-Entailed Premises:\n\nThroughline:`
- ✅ DO write: `Non-Entailed Premises:\nN/A`
- **Why**: Prevents hallucinations, keeps things explicit, clear model expectations

### 6. Preserve Confidence Scores
- ❌ DO NOT drop: `premise | relation | object` (loses confidence)
- ✅ DO keep: `premise | relation (confidence: 0.95) | object`
- **Why**: Transparency, traceability, signal for model about certainty levels

### 7. Contrastive Learning: Order Matters
- **Non-Entailed Premises MUST be first in training format** (pedagogical ordering)
- **Why**: Model learns to distinguish true from false before generating
- **Effect**: Negative inference capability (reasoning about what's false)
- **Note**: Generation output can reorder to logical order for human readability

---

## Consistency Rules

### Indentation
```
  - (exactly 2 spaces, then hyphen, then space)
```

### Confidence Format
- Input: `(tag, confidence=0.95)` where tag ∈ {observed, inferred}
- Output: `(confidence: 0.95)` (tag optional if obvious from context)

### Empty Sections
```
N/A  (exactly this string, no variations like "none" or "N/A" or "n/a")
```

### Quote Markers
```
"quote in double quotes"  (preserve original quoting from source)
```

### Spacing
- Between sections: blank line
- Between entries: newline only (no blank line between bullets)

---

## Examples

### Example 1: Training Format (Non-Entailed First)

Training input to model:
```
"By the pricking of my thumbs, Something wicked this way comes."

Non-Entailed Premises:
  - thumbs | are (observed, confidence=1.0) | pricking

Entailed Premises:
  - something | is (inferred, confidence=0.75) | wicked
  - something | is (inferred, confidence=0.8) | coming

Throughline:
  When one feels a premonition or intuitive sense, something bad is approaching.
```

Expected model output (Generation Format - logical order):
```
Throughline:
  When one feels a premonition or intuitive sense, something bad is approaching.

Entailed Premises:
  - something | is (inferred, confidence=0.75) | wicked
  - something | is (inferred, confidence=0.8) | coming

Non-Entailed Premises:
  - thumbs | are (observed, confidence=1.0) | pricking
```

### Example 2: Be Yourself (Training Format)

Training input:
```
"Be yourself; everyone else is already taken."

Non-Entailed Premises:
  - social conformity | is (observed, confidence=1.0) | undesirable
  - everyone else | is (observed, confidence=1.0) | already taken

Entailed Premises:
  - people | are (observed, confidence=1.0) | unique individuals
  - copying others | is (observed, confidence=1.0) | redundant
  - authenticity | is (observed, confidence=1.0) | the only way to be oneself

Throughline:
  One should embrace their own uniqueness and not attempt to imitate others.
```

### Example 3: Empty Non-Entailed Section

Training input:
```
"The only way to do great work is to love what you do."

Non-Entailed Premises:
  N/A

Entailed Premises:
  - great_work | is_enabled_by (observed, confidence=1.0) | loving_your_work
  - passion | is_inferred_to_be (inferred, confidence=0.8) | necessary_for_excellence

Throughline:
  Passion and love are necessary conditions for producing great work.
```

### Example 4: Empty Throughline

Training input:
```
"Silence is so freaking loud"

Non-Entailed Premises:
  - Volume | is (observed, confidence=1.0) | measured in decibels
  - Sound | is (observed, confidence=1.0) | is an acoustic wave

Entailed Premises:
  - Loudness | is (inferred, confidence=0.5) | is a perceptual quality
  - Perception | is (inferred, confidence=0.5) | is subjective

Throughline:
  N/A
```

---

## Implementation

### Translation Functions
- `src/format_translator.py::plaintext_to_hybrid_yaml()` — Convert plaintext triplets to this format
- `src/format_translator.py::hybrid_yaml_to_plaintext()` — Reverse conversion
- `src/reformat_contrastive_hybrid.py` — Batch dataset transformation

### Production Datasets
- `data/train_contrastive_hybrid_clean_967.jsonl` — **USE THIS** (967 records in canonical format)
- `data/train_contrastive_clean_967.jsonl` — Legacy plaintext format (reference only)

### When Generating New Data
```bash
python src/reformat_contrastive_hybrid.py \
  --input upstream_data.jsonl \
  --output output_data.jsonl \
  --format hybrid
```

---

## Design Rationale

### Why This Format?

**For Humans**:
- ✅ Markdown headings are familiar and easy to scan
- ✅ Bullet lists provide clear visual structure
- ✅ Confidence scores add transparency
- ✅ Explicit N/A prevents confusion about missing sections

**For LLMs**:
- ✅ Consistent delimiters (`**heading**:` format)
- ✅ Structured entries (2-3 lines per premise)
- ✅ Token-efficient (not verbose like raw plaintext)
- ✅ Explicit boundaries (model knows exactly what to expect)
- ✅ Negative inference signal (contrastive context in input)

**For Robustness**:
- ✅ N/A avoids hallucinated entries
- ✅ Confidence scores enable uncertainty quantification
- ✅ Tag information (observed/inferred) provides reasoning signal
- ✅ Bidirectional translation preserves all information

---

## Non-Negotiable Rules

1. **Markdown over brackets**: Always use `**Heading:**` format
2. **2-space bullet indent**: Exactly `  - ` (2 spaces, hyphen, space)
3. **Explicit N/A**: Never leave sections empty or use placeholders
4. **Confidence preserved**: Always include confidence scores
5. **Contrastive in input**: NOT_ENTAILED context goes to model input, not output
6. **Triplet structure**: `subject | relation (tag, confidence=X) | object`

---

## Regression Testing

Before deploying changes:
1. Verify format against these rules
2. Check examples parse correctly
3. Test bidirectional translation (plaintext → hybrid → plaintext)
4. Validate N/A is explicit (not empty strings)
5. Confirm confidence scores are preserved

---

## Version History

| Date | Version | Changes | Status |
|------|---------|---------|--------|
| 2026-05-03 | 1.0 | Initial specification, MVP format finalized | CANONICAL |

---

## Summary

This is the **canonical production format** for semantic reasoning training data. It balances:
- **Readability** (Markdown structure, bullet lists)
- **Clarity** (explicit N/A, confidence scores)
- **Efficiency** (2-3 lines per premise, token-lean)
- **Model compatibility** (consistent delimiters, predictable structure)

Use this format for all new training data. Do not regress to plaintext `[BRACKET]` format.

**Last updated**: 2026-05-03  
**Maintained by**: Josep Hua  
**Status**: LOCKED (no changes without explicit user approval)
