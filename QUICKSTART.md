# Installation & Quick Start

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 4.30+
- peft (for LoRA)
- pydantic

## Install

```bash
pip install torch transformers peft pydantic
```

## Quick Start

### 1. Load Configuration
```python
from src.training_config import PipelineConfig, TRAINING_ONLY_CONFIG

config = TRAINING_ONLY_CONFIG
print(config.training_format.premise_ordering)  # PEDAGOGICAL
```

### 2. Preprocess Training Data
```python
from src.preprocess_training_data import preprocess_training_record
import json

# Load raw data
with open("data/train_clean_for_model_967.jsonl") as f:
    records = [json.loads(line) for line in f]

# Preprocess
cleaned = [preprocess_training_record(r) for r in records]
```

### 3. Serialize to Training Format
```python
from src.serialize_training_format import serialize_training_record

# Apply pedagogical ordering
for record in cleaned:
    serialized = serialize_training_record(
        record,
        ordering="pedagogical",  # Non-Entailed → Entailed → Throughline
    )
    print(serialized.output_text)
```

### 4. Build Graph Ontology
```python
from src.graph_ontology import build_ontology_from_triplets, Triplet

# After inference: collect triplets
triplets = [
    Triplet("danger", "leads to", "peril", confidence=0.9),
    Triplet("peril", "causes", "fear", confidence=0.8),
]

# Build graph
ontology = build_ontology_from_triplets(triplets)

# Query facts
facts = ontology.traverse_path("danger", max_depth=2, confidence_threshold=0.7)
print(facts)
```

### 5. Generate Inference
```python
from src.infer_formatted import load_model_and_tokenizer, infer

# Load model
model, tokenizer = load_model_and_tokenizer(
    base_model="Qwen/Qwen3-0.6B",
    adapter_path="path/to/adapter",
)

# Generate
output = infer(
    model, tokenizer,
    quote="By the pricking of my thumbs, Something wicked this way comes.",
    max_tokens=200,
)
print(output)
```

## Data Format

### Input (What Model Sees)
```
[Quote]

Non-Entailed Premises:
- entity | relation (tag, confidence) | object
- ...

Entailed Premises:
- entity | relation (tag, confidence) | object
- ...
```

### Output (Pedagogical Format)
```
Non-Entailed Premises:
- entity | relation (tag, confidence) | object

Entailed Premises:
- entity | relation (tag, confidence) | object

Throughline:
[Reasoning text or triplets]
```

## Configuration Options

### TrainingFormat
- `syllogism_source`: NONE | GROUND_TRUTH | LLM_JUDGE
- `premise_ordering`: PEDAGOGICAL | LOGICAL | ENTAILED_ONLY
- `include_non_entailed`: bool
- `include_entailed`: bool
- `include_evidence_tags`: bool

### EntityNormalization
- `enable_synset_collapse`: bool
- `synset_map`: Dict[str, str] — entity → canonical
- `predicate_equivalence`: Dict[str, Set[str]] — canonical → variants

### GraphTraversal
- `confidence_threshold`: float (0.0-1.0)
- `max_path_depth`: int
- `include_non_entailed_in_traversal`: bool

### JudgeConfig
- `enable_judge`: bool
- `judge_model`: str (e.g., "gpt-4")
- `judge_temperature`: float
- `judge_max_tokens`: int

## Examples

See ARCHITECTURE_CONFIGURABLE_FORMAT.md for:
- Use case walkthroughs
- Configuration examples
- SPO integration roadmap

See FORMAT_SPECIFICATION.md for:
- Detailed format specification
- Design rationale
- Edge case handling

See FORMAT_EXAMPLES_GENERATION_VS_TRAINING.md for:
- Side-by-side examples
- Both orderings shown
- Confidence score usage

---

**Next:** Integrate into your training pipeline or downstream tasks.
