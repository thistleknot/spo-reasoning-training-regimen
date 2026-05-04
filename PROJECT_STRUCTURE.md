# SPO Training Regimen — Project Structure

```
spo-training-regimen/
├── src/
│   ├── training_config.py          # Configurable pipeline (240 lines)
│   ├── graph_ontology.py           # Triplet storage & traversal (330 lines)
│   ├── preprocess_training_data.py # Data cleaning (223 lines)
│   ├── serialize_training_format.py # Format serialization (129 lines)
│   └── infer_formatted.py          # Inference wrapper (226 lines)
│
├── data/
│   └── train_clean_for_model_967.jsonl  # 967 sample records
│
├── README.md                        # Project overview
├── QUICKSTART.md                    # Installation & quick start
├── ARCHITECTURE_CONFIGURABLE_FORMAT.md  # Design principles
├── FORMAT_SPECIFICATION.md          # v2.0 specification
├── FORMAT_EXAMPLES_GENERATION_VS_TRAINING.md  # Examples
│
└── .gitignore
```

## What's Included

### Core Modules (1,148 lines total)

**training_config.py** — Configuration system
- TrainingFormat: premise ordering, syllogism source, evidence tags
- EntityNormalization: synset collapse, predicate equivalence
- GraphTraversal: confidence thresholds, path depth
- JudgeConfig: LLM judge settings
- Pre-configured examples (TRAINING_ONLY, INFERENCE_WITH_JUDGE, GRAPH_RETRIEVAL)

**graph_ontology.py** — Triplet-based fact storage
- Triplet: subject | predicate (tag, confidence) | object
- GraphOntology: entity normalization, synset collapse
- Path finding, fact extraction, multi-hop traversal
- Factory function: build_ontology_from_triplets()

**preprocess_training_data.py** — Data cleaning
- Parse hybrid format (header markers, markdown removal)
- Extract sections (non-entailed, entailed, throughline)
- Normalize fields, handle N/A values
- Preserve confidence scores and evidence tags

**serialize_training_format.py** — Format serialization
- Apply configurable premise ordering
- Pedagogical (Non-Entailed first) vs Logical (Throughline first)
- Deterministic ordering for reproducibility
- JSONL output with input_text/output_text pairs

**infer_formatted.py** — Inference API
- Load base model + LoRA adapter
- Generate with format preservation
- Extract triplets from output
- Support for confidence score extraction

### Documentation

**README.md** — Project overview
- Component descriptions
- Architecture insights (confidence as SPO signal)
- Configuration examples
- Usage walkthrough

**QUICKSTART.md** — Getting started
- Installation requirements
- Step-by-step examples
- Data format specification
- Configuration reference

**ARCHITECTURE_CONFIGURABLE_FORMAT.md** — Design deep dive
- Core insight: confidence is emergent, not training label
- Use cases with code examples
- Configuration schema
- SPO integration roadmap

**FORMAT_SPECIFICATION.md** — Format v2.0 locked spec
- Dual ordering rationale (pedagogical vs logical)
- Triplet structure specification
- Confidence score usage
- Design rationale and edge cases

**FORMAT_EXAMPLES_GENERATION_VS_TRAINING.md** — Side-by-side examples
- Generation format (human-readable)
- Training format (pedagogical)
- Both orderings shown
- Confidence score examples

### Sample Data

**train_clean_for_model_967.jsonl** — 967 preprocessed records
- Cleaned of UTF-8 mojibake
- Verified no encoding corruption
- Pedagogical format serialization
- Ready for model training

---

## Key Design

### Separation of Concerns

1. **Training Format** — What data goes into training
   - `training_config.py` defines what to include/exclude
   - `serialize_training_format.py` applies the config

2. **Graph Ontology** — What data retrieval looks like
   - `graph_ontology.py` builds normalized fact graph
   - `training_config.py` parametrizes traversal

3. **Data Preprocessing** — How to clean data
   - `preprocess_training_data.py` handles parsing
   - Works with any format that's parseable

4. **Inference** — How to generate
   - `infer_formatted.py` is a simple wrapper
   - Model outputs confidence (emergent property)

### The Confidence Insight

**NOT in training:**
- Training data has no confidence scores
- Evidence tags (observed/inferred) are features
- Model learns to generate triplets

**Emergent at inference:**
- Model outputs triplets WITH confidence
- Confidence reflects model's own uncertainty
- SPO optimizes it based on downstream performance

---

## Next Steps

1. **Integrate into training:** Use TrainingFormat config in your trainer
2. **Build SPO reward:** Use model confidence as signal
3. **LLM judge:** Call GPT-4 for optional syllogism synthesis
4. **Graph queries:** Retrieve facts for downstream tasks
5. **Domain synsets:** Add entity normalization rules

---

**Status:** Standalone, ready for production integration.
