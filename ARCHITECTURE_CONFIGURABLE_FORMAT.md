# Configurable Training Format & Confidence Architecture

## Core Insight: Confidence is SPO Signal, Not Training Feature

**The model IS trained to generate triplets. Confidence is what it learns to output at inference time—not a label you bake into training.**

Flow:
```
Training:
  Input:  [Quote] + [Non-Entailed Premises] + [Entailed Premises]
  Output: [Non-Entailed] + [Entailed] + [Throughline]
  (No confidence scores in training data)

Inference:
  Model generates: triplets WITH confidence scores (emergent property)
  Example: "something | is (inferred) | wicked" → model outputs confidence=0.8

SPO Optimization:
  Reward/penalize based on: downstream task performance
  Goal: model learns to assign accurate confidence to its own outputs
```

## Configurability

### TrainingFormat
Controls what goes into the training pipeline:

```python
TrainingFormat(
    # Syllogism: model generates, LLM judge synthesizes (not in training)
    syllogism_source=SyllogismSource.NONE,
    include_syllogism_in_training=False,
    
    # Premise ordering: pedagogical (false first) vs logical (true first)
    premise_ordering=PremiseOrdering.PEDAGOGICAL,
    
    # What to include in training data
    include_non_entailed=True,  # false premises for negative inference
    include_entailed=True,      # true premises to learn
    
    # Evidence tags are FEATURES (observed vs inferred)
    # NOT confidence—that's model output
    include_evidence_tags=True,
)
```

### EntityNormalization
Synset collapse for graph retrieval:

```python
EntityNormalization(
    enable_synset_collapse=True,
    synset_map={
        "danger": "peril",
        "bad": "evil",
        "wicked": "evil",
    },
    predicate_equivalence={
        "is": {"equals", "represents", "means"},
        "causes": {"leads to", "produces", "creates"},
    }
)
```

### GraphTraversal
Controls fact retrieval from the learned triplets:

```python
GraphTraversal(
    # Threshold: only use facts with confidence >= this in traversal
    confidence_threshold=0.7,
    
    # Max hops for path finding (subject → pred → obj → ...)
    max_path_depth=3,
    
    # Include negative facts in traversal?
    include_non_entailed_in_traversal=False,
)
```

### JudgeConfig
LLM judge for post-hoc syllogism synthesis:

```python
JudgeConfig(
    enable_judge=True,
    judge_model="gpt-4",
    judge_temperature=0.3,  # low = deterministic synthesis
    judge_max_tokens=200,
)
```

## Use Cases

### 1. Pure Training (No Confidence Baking)
```python
config = PipelineConfig(
    training_format=TrainingFormat(
        premise_ordering=PremiseOrdering.PEDAGOGICAL,
        include_evidence_tags=True,
    ),
    judge_config=JudgeConfig(enable_judge=False),
)
```

Output format:
```
Non-Entailed Premises:
- thumbs | are (observed) | pricking

Entailed Premises:
- something | is (inferred) | wicked

Throughline:
[model generates]
```

### 2. Inference with LLM Judge
```python
config = PipelineConfig(
    training_format=TrainingFormat(
        syllogism_source=SyllogismSource.LLM_JUDGE,
    ),
    judge_config=JudgeConfig(enable_judge=True),
)
```

At inference:
1. Model generates triplets WITH confidence
2. LLM judge receives triplets + original quote
3. Judge synthesizes syllogism (optional)

### 3. Graph Retrieval (SPO Downstream)
```python
config = PipelineConfig(
    training_format=TrainingFormat(
        include_non_entailed=False,  # Only high-confidence facts
    ),
    entity_normalization=EntityNormalization(enable_synset_collapse=True),
    graph_traversal=GraphTraversal(
        confidence_threshold=0.7,
        max_path_depth=2,
    ),
)
```

Usage:
```python
# Build ontology from inference outputs
ontology = build_ontology_from_triplets(
    model_triplets,
    synset_map=config.entity_normalization.synset_map,
)

# Query: find facts connected to "danger"
paths = ontology.traverse_path("danger", max_depth=2)
# Returns: entities and their connecting triplets
```

## Key Design Principles

1. **Confidence is emergent:** Model learns to output it, SPO optimizes it
2. **Evidence tags are features:** Observed vs inferred—model learns to predict
3. **Syllogism is optional:** Can be ground truth in training or LLM-judged at inference
4. **Synset collapse is retrieval-time:** Not in training—only for graph queries
5. **Premise ordering is pedagogical:** Non-entailed first teaches negative inference

## Files

- `src/training_config.py` — Configurable pipeline (this file)
- `src/graph_ontology.py` — Triplet storage, synset collapse, path finding
- `src/serialize_training_format.py` — Apply config to training data serialization
- `src/infer_formatted.py` — Generate inference, let model output confidence

## Example: Building & Querying

```python
from src.training_config import PipelineConfig, GRAPH_RETRIEVAL_CONFIG
from src.graph_ontology import build_ontology_from_triplets, Triplet

# Load config
config = GRAPH_RETRIEVAL_CONFIG

# After inference: collect triplets the model generated
triplets = [
    Triplet("danger", "leads to", "peril", confidence=0.9, evidence_tag="inferred"),
    Triplet("peril", "causes", "fear", confidence=0.8, evidence_tag="inferred"),
    Triplet("thumbs", "are", "pricking", confidence=1.0, evidence_tag="observed"),
]

# Build ontology with synset collapse
ontology = build_ontology_from_triplets(
    triplets,
    synset_map=config.entity_normalization.synset_map,
)

# Query: traverse from danger with high confidence threshold
facts = ontology.traverse_path(
    "danger",
    max_depth=config.graph_traversal.max_path_depth,
    confidence_threshold=config.graph_traversal.confidence_threshold,
)
# Result: connected entities and their triplets, synsets normalized
```

## Future: SPO Integration

SPO (Soft Policy Optimization) uses model-generated confidence as reward signal:

```python
# After inference:
model_output = "something | is (inferred) | wicked"  # + confidence=0.8

# Downstream evaluation:
correctness_signal = evaluate(model_output, context)

# SPO reward:
reward = correctness_signal * confidence
# → model learns to assign high confidence only when correct
```
