# Architecture & Design Principles

High-level architecture and the reasoning behind key design decisions.

## The Core Insight: Confidence is Emergent

**Problem:** Traditional approaches fail at confidence calibration
- If you train models with confidence labels, they overfit to those labels
- Confidence doesn't transfer to new domains
- Models become overconfident or underconfident on out-of-distribution data

**Solution:** Separate training from optimization
1. Training phase: Model learns STRUCTURE (triplets, evidence tags)
2. Inference phase: Confidence EMERGES naturally
3. Optimization phase: SPO calibrates that emergent confidence

**Why this works:**
- No training-time leakage of confidence information
- Confidence emerges from learned patterns
- SPO can optimize what the model actually produces
- Better generalization to new tasks

## Three-Phase Workflow

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: GENERATE                                       │
│ Quote → LLM → ReasoningExample                         │
│                                                         │
│ Inputs: Quotes (any text)                              │
│ Output: Structured reasoning (triplets + syllogism)    │
│ Evidence tags: observed/inferred (NO confidence yet)   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 2: TRAIN                                          │
│ ReasoningExample → QLoRA → Trained Model               │
│                                                         │
│ Input: Training records (JSONL format, pedagogical)    │
│ Loss: Next-token prediction on structured output       │
│ Output: Fine-tuned model (adapter weights)             │
│ Learn: What's entailed vs non-entailed reasoning       │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 3: OPTIMIZE (SPO)                                 │
│ Trained Model → Confidence Calibration                 │
│                                                         │
│ Model generates: Triplets WITH confidence (emergent)   │
│ Reward: correctness × confidence                       │
│ Loss: Policy gradient to maximize reward               │
│ Learn: Accurate confidence calibration                 │
└─────────────────────────────────────────────────────────┘
```

## Module Architecture

```
src/
├── synthetic_generator.py
│   ├── TripletItem — Single evidence-tagged premise
│   ├── ReasoningExample — Complete reasoning (triplets + syllogism)
│   └── SyntheticReasoningGenerator — Quote → ReasoningExample
│
├── spo_trainer.py
│   ├── SPOReward — Composite score (correctness × confidence)
│   ├── SPOTrainer — Training loop for confidence optimization
│   └── SPOEvaluator — Validation and test evaluation
│
├── pipeline.py
│   ├── PipelineConfig — Centralized configuration
│   └── Pipeline — Orchestrate: generate → validate → train → eval
│
├── training_config.py
│   └── TrainingFormat — Enum for format variants (PEDAGOGICAL, LOGICAL, ENTAILED_ONLY)
│
├── graph_ontology.py
│   └── TripletGraph — Triplet storage with entity normalization
│
├── serialize_training_format.py
│   └── Deterministic serialization with token distribution filtering
│
├── preprocess_training_data.py
│   └── Data cleaning (mojibake handling, normalization)
│
└── infer_formatted.py
    └── Inference wrapper for trained models
```

## Key Design Decisions

### 1. Pedagogical Order for Training Data

**Order:** Non-Entailed → Entailed → Conclusion

```
[NON-ENTAILED]    ← Teaches negative inference first
subject | rel | obj

[ENTAILED]        ← Then teach supporting evidence
subject | rel | obj

[CONCLUSION]      ← Finally the hypothesis
```

**Why not logical order?**
- Logical order: Conclusion → Entailed → Non-Entailed
- Logical order requires foreknowledge to parse
- Pedagogical order teaches discrimination progressively
- Pedagogical converges faster in practice
- Matches human learning: negative examples → positive examples

### 2. Evidence Tags, Not Confidence, in Training

**In training data:**
```
subject | relation (observed, confidence=1.0) | object
subject | relation (inferred, confidence=0.75) | object
```

**WRONG approach:**
```
subject | relation | object | confidence=0.75  ← Don't do this!
```

**Why tags?**
- Tags are PROPERTIES of premises (observed vs inferred)
- Confidence is model OUTPUT (not training input)
- Model learns: "observed" → confidence=1.0, "inferred" → calibrated lower
- No overfitting because confidence isn't labeled in training

### 3. Triplets as Structured Output

**Why triplets?**
- Naturally representable as graphs (subject-relation-object)
- Compatible with knowledge bases
- Enables entity linking (synset collapse)
- Multi-hop reasoning support
- Inspectable output (not opaque like free-form text)

**Why not just generate free-form reasoning?**
- Free-form is hard to parse and validate
- Triplets are verifiable: each has well-defined parts
- Graph structure enables downstream retrieval
- Synset collapse prevents duplicate entities

### 4. SPO Optimization for Confidence

**Why not RLHF?**
```
RLHF: Reward model predicts "good" vs "bad" outputs
      Prone to reward hacking
      Requires expensive human labeling

SPO: Reward = correctness × confidence
     Simple signal
     No labeling needed (LLM judges correctness)
     Confidence is model's own output
```

**SPO Advantage:**
- Simple, interpretable reward
- Emerges naturally from triplet generation
- Model controls confidence level (no external judge)
- Calibration is observable and testable

### 5. Format Configurability

```python
class TrainingFormat(Enum):
    PEDAGOGICAL = "pedagogical"      # Non-Entailed → Entailed → Conclusion
    LOGICAL = "logical"              # Conclusion → Entailed → Non-Entailed
    ENTAILED_ONLY = "entailed_only"  # Entailed only (no negatives)
```

**Why configurable?**
- Different domains may need different orderings
- Allows A/B testing
- Supports gradual rollout
- Future formats without code changes

### 6. Pydantic for Validation

**Every generated example validated:**
```python
try:
    example = ReasoningExample(
        quote=text,
        non_entailed_premises=non_ent,
        entailed_premises=ent,
        syllogism=conclusion
    )
except ValidationError:
    discard_record()  # Invalid, skip it
```

**Why strict validation?**
- Prevents garbage data in training
- Detects LLM hallucinations
- Schema-safe JSONL export
- Downstream training won't fail on malformed data

## Data Flow

```
Raw Quotes (any text)
         ↓
   [GENERATE]
   LLM: Extract structured reasoning
   Output: ReasoningExample (triplets + tags)
         ↓
   [VALIDATE]
   Pydantic checks structure
   Invalid → discard
   Valid → continue
         ↓
   [SERIALIZE]
   Format: Pedagogical order
   Export: JSONL (deterministic ordering)
   Train data ready
         ↓
   [TRAIN]
   Model: QLoRA fine-tuning
   Input: Quotes
   Output: Triplets with evidence tags
         ↓
   [INFER]
   Model generates: Triplets + confidence
   Confidence: Emergent (not from training)
         ↓
   [EVALUATE]
   Score: correctness × confidence
   Select: High-confidence correct outputs
         ↓
   [OPTIMIZE (SPO)]
   Reward: Model predictions with high confidence + correctness
   Loss: Policy gradient to maximize reward
   Result: Calibrated confidence
```

## Deployment Architecture

```
┌──────────────────────────────────────────┐
│ User Application                         │
└──────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│ Inference Interface (infer_formatted.py) │
├──────────────────────────────────────────┤
│ - Load model (local or HF Hub)           │
│ - Handle quantization (4-bit, 8-bit)     │
│ - Batch processing                       │
│ - Parse triplet output                   │
└──────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│ Trained Model (QLoRA adapter + base)     │
├──────────────────────────────────────────┤
│ - Generates structured reasoning         │
│ - Evidence tags (observed/inferred)      │
│ - Confidence scores (emergent)           │
└──────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────┐
│ Graph Ontology (Entity Normalization)    │
├──────────────────────────────────────────┤
│ - Synset collapse (duplicate entities)   │
│ - Graph storage (Neo4j, local, etc.)     │
│ - Multi-hop retrieval                    │
└──────────────────────────────────────────┘
```

## Confidence Extraction

During inference, confidence emerges from triplet generation:

```python
import re

output = "[ENTAILED]\nsomething | is (inferred, confidence=0.85) | wicked"

# Extract confidence using regex
pattern = r'confidence=([0-9.]+)'
match = re.search(pattern, output)
if match:
    confidence = float(match.group(1))  # 0.85
    # Use in reward calculation
```

## Scalability Considerations

### Generation (Phase 1)
- LLM API call: ~0.5-2s per quote
- Batch with request pools for speed
- Can generate millions with budget

### Training (Phase 2)
- QLoRA: ~4-8GB VRAM per model
- 967 examples → ~30 min training on single GPU
- Larger datasets require distributed training (DDP)

### Inference (Phase 3)
- Quantization: 4-bit reduces memory 4x
- Batch inference: Process 32-64 quotes at once
- Latency: ~50-200ms per quote (depending on model)

## Error Handling

### Generation Errors
- Invalid JSON from LLM → Discard
- Missing fields → Discard
- Out-of-memory → Retry with smaller batch

### Training Errors
- NaN loss → Reduce learning rate
- OOM → Reduce batch size or enable gradient checkpointing
- Divergence → Check data distribution

### Inference Errors
- Model not found → Fall back to HF Hub auto-download
- Quantization error → Use different precision
- Memory error → Enable CPU offload

## Testing Strategy

1. **Unit tests** — Individual modules (synthetic_generator, spo_trainer)
2. **Integration tests** — End-to-end pipeline on small dataset
3. **Evaluation tests** — Inference quality on holdout set
4. **Regression tests** — Confidence calibration across datasets

## Future Enhancements

- [ ] Constrained decoding (enforce output format)
- [ ] Multi-hop reasoning (chains of triplets)
- [ ] Entity linking (Wikidata synsets)
- [ ] Retrieval-augmented generation (RAG integration)
- [ ] Distributed training (DDP, Deepspeed)
- [ ] Continuous learning (online updates)
- [ ] Explainability (why confidence is high/low)

## References

- Triplet format: `docs/format/README.md`
- Training guide: `docs/training/README.md`
- Inference guide: `docs/inference/README.md`
- Generation guide: `docs/generation/README.md`
- Code: `src/*.py`
