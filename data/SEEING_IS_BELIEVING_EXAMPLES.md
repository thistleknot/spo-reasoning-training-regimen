# Seeing is Believing: Example Outputs

Complete end-to-end examples showing model inputs and outputs in training format (pedagogical order).

## Example 1: Premonition and Danger

Input:
```
"By the pricking of my thumbs, Something wicked this way comes."
```

Completion:
```
Non-Entailed Premises:
  thumbs | are (observed, confidence=1.0) | pricking
  thumbs | are (inferred, confidence=0.3) | numb

Entailed Premises:
  something | is (inferred, confidence=0.85) | wicked
  something | is (inferred, confidence=0.88) | coming
  premonition | indicates (inferred, confidence=0.9) | danger

Throughline:
  When one feels a premonition through physical sensation (pricking thumbs), something wicked or dangerous approaches.
```

---

## Example 2: Creative Writing and Semicolons

Input:
```
"Here is a lesson in creative writing. First rule: Do not use semicolons. They are transvestite hermaphrodites representing absolutely nothing."
```

Completion:
```
Non-Entailed Premises:
  semicolons | represent (inferred, confidence=0.2) | hermaphrodites
  writing | uses (inferred, confidence=0.4) | transvestite concepts

Entailed Premises:
  semicolons | are (observed, confidence=1.0) | punctuation
  semicolons | should be (inferred, confidence=0.8) | avoided in creative writing
  clear writing | requires (inferred, confidence=0.75) | simpler punctuation

Throughline:
  Effective creative writing avoids complex punctuation like semicolons in favor of simpler, clearer syntax.
```

---

## Example 3: Social Hierarchy

Input:
```
"All animals are equal, but some animals are more equal than others."
```

Completion:
```
Non-Entailed Premises:
  equality | is (inferred, confidence=0.3) | absolute

Entailed Premises:
  animals | are (observed, confidence=1.0) | equal (in principle)
  some animals | are (inferred, confidence=0.95) | more privileged
  hierarchy | exists (inferred, confidence=0.9) | despite stated equality
  power | determines (inferred, confidence=0.85) | actual equality

Throughline:
  Stated equality may mask actual hierarchies where those in power enjoy privileges not shared by others.
```

---

## Example 4: Historical Context

Input:
```
"It was the best of times, it was the worst of times."
```

Completion:
```
Non-Entailed Premises:
  times | are (inferred, confidence=0.2) | neutral

Entailed Premises:
  period | contains (inferred, confidence=0.8) | contradictions
  times | are (observed, confidence=1.0) | best AND worst simultaneously
  perspective | affects (inferred, confidence=0.85) | perception of era
  change | brings (inferred, confidence=0.9) | both loss and opportunity

Throughline:
  Historical periods contain simultaneous progress and suffering; whether an era is "best" or "worst" depends on one's position and perspective.
```

---

## Example 5: Narrative Identity

Input:
```
"Whether I shall turn out to be the hero of my own life depends on these pages."
```

Completion:
```
Non-Entailed Premises:
  hero | is (inferred, confidence=0.4) | predetermined

Entailed Premises:
  protagonist | will (observed, confidence=1.0) | be revealed through narrative
  identity | emerges (inferred, confidence=0.9) | through action and choice
  heroism | depends (inferred, confidence=0.85) | on decisions made
  self | is (inferred, confidence=0.8) | constructed through lived experience

Throughline:
  A person becomes the hero of their own story through the choices they make and the narrative they create by living.
```

---

## Format Notes

### Pedagogical Order (Training Format)
1. **Non-Entailed Premises** — Facts mentioned but not supporting the main inference
   - Help model learn negative inference
   - Teach what to NOT use for reasoning
   
2. **Entailed Premises** — Facts that support the throughline
   - Core reasoning elements
   - Facts the model learns to weight heavily
   
3. **Throughline** — Synthesized reasoning
   - The conclusion or reasoning path
   - What the model generates from premises

### Evidence Tags
- **observed**: Explicit in the text (confidence typically 1.0)
- **inferred**: Derived from the text (confidence 0.3-0.95)

### Confidence Scores
- Model assigns these during inference
- SPO training optimizes: reward = correctness × confidence
- Higher confidence on correct inferences → calibrated model

---

## Data Format: JSONL

These examples serialize to training JSONL like:

```json
{
  "input_text": "\"By the pricking of my thumbs, Something wicked this way comes.\"",
  "output_text": "Non-Entailed Premises:\n  thumbs | are (observed, confidence=1.0) | pricking\n  thumbs | are (inferred, confidence=0.3) | numb\n\nEntailed Premises:\n  something | is (inferred, confidence=0.85) | wicked\n  something | is (inferred, confidence=0.88) | coming\n  premonition | indicates (inferred, confidence=0.9) | danger\n\nThroughline:\n  When one feels a premonition through physical sensation (pricking thumbs), something wicked or dangerous approaches."
}
```

---

## How to Generate More Examples

```python
from src.synthetic_generator import SyntheticReasoningGenerator

# Load quotes
quotes = [
    "By the pricking of my thumbs, Something wicked this way comes.",
    "Here is a lesson in creative writing...",
]

# Generate examples
gen = SyntheticReasoningGenerator()
examples = gen.generate_from_quotes(quotes, llm_generate_fn=your_llm)

# Export to JSONL for training
gen.export_to_jsonl("data/train.jsonl")
```

---

## Key Insights from Examples

1. **Negative inference matters** — Non-entailed premises teach what's irrelevant
2. **Confidence calibration** — Higher confidence on "more certain" inferences
3. **Syllogism synthesis** — Throughline connects premises into coherent reasoning
4. **Evidence tags guide learning** — Model learns observed vs inferred distinction
5. **Pedagogical order helps convergence** — Non-entailed first teaches discrimination

These examples demonstrate the complete pipeline: **Quote → Triplets → Syllogism**.
