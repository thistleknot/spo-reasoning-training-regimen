# SPO Training Regimen

Standalone training infrastructure for semantic reasoning models with configurable format and graph ontology support.

## Core Components

### Training Format Configuration
`src/training_config.py` — Parametrize training pipeline:
- **SyllogismSource**: NONE | GROUND_TRUTH | LLM_JUDGE
- **PremiseOrdering**: PEDAGOGICAL | LOGICAL | ENTAILED_ONLY
- **EntityNormalization**: Synset collapse, predicate equivalence
- **GraphTraversal**: Confidence thresholds, path depth
- **JudgeConfig**: LLM judge for post-hoc synthesis

### Data Preprocessing
`src/preprocess_training_data.py` — Clean and parse training data
- Remove markdown artifacts
- Parse hybrid format (header markers)
- Normalize entities and predicates
- Handle N/A fields

### Training Format Serialization
`src/serialize_training_format.py` — Apply configurable ordering
- Pedagogical format: Non-Entailed → Entailed → Throughline
- Logical format: Throughline → Entailed → Non-Entailed
- Evidence tag preservation (observed/inferred)
- No confidence scores in training data (emergent at inference)

### Graph Ontology
`src/graph_ontology.py` — Triplet storage and retrieval
- Triplet: subject | predicate (tag, confidence) | object
- Entity normalization via synset collapse
- Predicate equivalence classes
- Multi-hop path traversal
- Fact extraction (entailed vs non-entailed)

### Inference
`src/infer_formatted.py` — Production inference API
- Load model and adapter
- Generate with format preservation
- Output triplets with model confidence scores

## Architecture

### The Confidence Insight
**Confidence is NOT a training label—it's emergent model output.**

```
Training Phase:
  Input:  Quote + premises
  Output: Structured triplets (no confidence in data)
  Model learns: To generate correct triplets

Inference Phase:
  Model generates: Triplets WITH confidence (emergent)
  Example: confidence=0.8 on "something | is | wicked"

SPO Optimization:
  Reward signal: Downstream task correctness
  Goal: Model learns to assign accurate confidence
```

### Pedagogical Format (Default)
Why Non-Entailed comes first:
1. **Negative inference first** — Model sees false premises
2. **Contrastive learning** — Learns to discriminate true from false
3. **Better convergence** — Explicit negative examples improve training

Order:
```
Non-Entailed Premises:    (false/irrelevant facts)
Entailed Premises:        (true facts that support conclusion)
Throughline:              (synthesized reasoning)
```

## Usage

### Basic Training Config
```python
from src.training_config import PipelineConfig, TRAINING_ONLY_CONFIG

config = TRAINING_ONLY_CONFIG
config.training_format.premise_ordering  # PEDAGOGICAL
config.training_format.include_evidence_tags  # True
```

### Data Preprocessing
```python
from src.preprocess_training_data import preprocess_training_record

record = preprocess_training_record(raw_data)
# Returns: dict with input_text, output_text, metadata
```

### Build Graph Ontology
```python
from src.graph_ontology import build_ontology_from_triplets, Triplet

triplets = [
    Triplet("danger", "leads to", "peril", confidence=0.9),
    Triplet("peril", "causes", "fear", confidence=0.8),
]

ontology = build_ontology_from_triplets(triplets)
facts = ontology.traverse_path("danger", max_depth=2)
```

## Configuration Examples

### Training Only (No Confidence Baking)
```python
config = PipelineConfig(
    training_format=TrainingFormat(
        premise_ordering=PremiseOrdering.PEDAGOGICAL,
        include_evidence_tags=True,
    ),
    judge_config=JudgeConfig(enable_judge=False),
)
```

### Inference with LLM Judge
```python
config = INFERENCE_WITH_JUDGE_CONFIG
# Model generates triplets → Judge synthesizes syllogism
```

### Graph Retrieval (SPO Downstream)
```python
config = GRAPH_RETRIEVAL_CONFIG
# Synset collapse, high confidence threshold, path finding
```

## Files

- `src/training_config.py` (240 lines) — Configuration system
- `src/graph_ontology.py` (330 lines) — Triplet storage and traversal
- `src/preprocess_training_data.py` (223 lines) — Data cleaning
- `src/serialize_training_format.py` (129 lines) — Format serialization
- `src/infer_formatted.py` (226 lines) — Inference wrapper
- `data/train_clean_for_model_967.jsonl` — 967 sample records
- `ARCHITECTURE_CONFIGURABLE_FORMAT.md` — Design principles
- `FORMAT_SPECIFICATION.md` — v2.0 format specification
- `FORMAT_EXAMPLES_GENERATION_VS_TRAINING.md` — Side-by-side examples

## Key Design Principles

1. **Confidence is emergent** — Model learns, SPO optimizes
2. **Evidence tags are features** — Observed vs inferred are training inputs
3. **Syllogism is optional** — Can be LLM-judged at inference
4. **Synset collapse is retrieval-time** — Not in training, only graph queries
5. **Premise ordering is configurable** — Pedagogical by default

## Next Steps

1. Integrate into training pipeline (apply config to data serialization)
2. Build SPO reward function (use model confidence as signal)
3. Implement LLM judge (call GPT-4 for optional syllogism synthesis)
4. Graph retrieval queries (fact matching for downstream tasks)
5. Extend synset maps (domain-specific entity normalization)

---

**Status:** Ready for integration into downstream training and inference pipelines.
