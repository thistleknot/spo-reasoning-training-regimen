# SPO Reasoning Training Regimen

Build quote -> structured reasoning dataset -> QLoRA adapter -> optional downstream confidence calibration.

This repo packages the full workflow for turning raw quotes into structured reasoning examples, training a model to emit that reasoning in a pedagogical order, and optionally calibrating confidence with Soft Policy Optimization (SPO). The root README is the front door; the deeper mechanics live in the docs folders linked below.

## What you get

- Synthetic reasoning dataset generation from quotes
- A format contract that separates generation order from training order
- Three complementary training regimens built from the same synthetic base
- QLoRA training guidance for small-to-mid-size reasoning models
- Optional downstream judging or calibration after the base reasoning model is trained
- "Seeing is believing" example artifacts under `data/`

## Workflow at a glance

```mermaid
flowchart LR
    A[Quotes] --> B[Generation<br/>LLM or templates]
    B --> C[Validated reasoning examples]
    C --> D[Training JSONL<br/>pedagogical order]
    D --> E[QLoRA adapter]
    E --> F[Inference on new quotes]
    F --> G[SPO calibration<br/>optional]
```

## Fast paths

| If you want to... | Start here |
|---|---|
| Understand the whole workflow fast | `docs/quickstart/README.md` |
| Configure an LLM for dataset generation | `docs/generation/README.md` |
| Train with QLoRA | `docs/training/README.md` |
| Configure model inference | `docs/inference/README.md` |
| Understand the exact format contract | `docs/format/README.md` |
| See finished examples before touching code | `data/SEEING_IS_BELIEVING_EXAMPLES.md` |

## Quick start

### Install

```bash
git clone https://github.com/thistleknot/spo-reasoning-training-regimen.git
cd spo-reasoning-training-regimen
pip install -r requirements.txt
```

### Generate a starter dataset

```python
from src.synthetic_generator import SyntheticReasoningGenerator

quotes = [
    "By the pricking of my thumbs, Something wicked this way comes.",
    "Call me Ishmael.",
]

gen = SyntheticReasoningGenerator()
examples = gen.generate_from_quotes(quotes)
gen.export_to_jsonl("data/my_dataset.jsonl")
```

That path creates template-backed examples immediately. If you want model-backed generation with OpenAI, Ollama, or Hugging Face, go straight to `docs/generation/README.md`.

### Then choose your next step

1. Inspect example outputs in `data/SEEING_IS_BELIEVING_EXAMPLES.md`
2. Learn the format contract in `docs/format/README.md`
3. Train a model via `docs/training/README.md`
4. Configure inference via `docs/inference/README.md`
5. Add SPO calibration if you want confidence to mean something downstream

## Synthetic generation -> training example

The same quote moves through a few different shapes before it becomes a training row.

### 1. Synthetic generation scaffold

```text
Input:
"By the pricking of my thumbs, Something wicked this way comes."

Completion:
Throughline:
  When one feels a premonition or intuitive sense, something bad is approaching.

Entailed Premises:
  - something | is (inferred, confidence=0.75) | wicked
  - something | is (inferred, confidence=0.8) | coming

Non-Entailed Premises:
  - thumbs | are (observed, confidence=1.0) | pricking
```

### 2. Preprocessed structured record

The preprocessing step strips hybrid markdown wrappers, extracts named sections, normalizes missing values, and turns the scaffold into a clean structured record:

```json
{
  "quote": "By the pricking of my thumbs, Something wicked this way comes.",
  "entailed_premises": [
    "something | is (inferred, confidence=0.75) | wicked",
    "something | is (inferred, confidence=0.8) | coming"
  ],
  "non_entailed_premises": [
    "thumbs | are (observed, confidence=1.0) | pricking"
  ],
  "syllogism": "When one feels a premonition or intuitive sense, something bad is approaching."
}
```

### 3. Training row actually fed to the model

The serializer then converts that structured record into the pedagogical training format used for QLoRA. The base regimen now uses an explicit task prompt plus chat-formatted supervision so the instruct model sees the same contract at train and inference time. Evidence tags stay; numeric confidence is stripped so the base model learns premises and throughline text rather than frozen scores:

```text
Given this quote, extract the implicit reasoning.

Quote: "By the pricking of my thumbs, Something wicked this way comes."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | relation (tag) | object
- tag: "observed" for explicit facts, "inferred" for derived facts

Response:

Non-Entailed Premises:
thumbs | are (observed) | pricking

Entailed Premises:
something | is (inferred) | wicked
something | is (inferred) | coming

Throughline:
When one feels a premonition or intuitive sense, something bad is approaching.
```

## Training regimen families

The repo now supports three adjacent supervised tasks over the same synthetic source data:

| Regimen | Input | Output | Numeric confidence |
|---|---|---|---|
| Base reasoning | Prompted quote instruction | Non-entailed + entailed premises + throughline | Stripped |
| Facts with confidence | Quote | Non-entailed + entailed premises | Preserved |
| Syllogism with confidence | Quote + confidence-bearing facts | Throughline + aggregate confidence | Preserved |

The first regimen teaches the reasoning structure cleanly. The other two turn the original synthetic numerics into follow-on tasks instead of freezing them into the base target.

You can build the two follow-on datasets with:

```bash
python -m src.build_training_regimens \
  --input path/to/confidence_rich_source.jsonl \
  --output data/train_facts_with_confidence.jsonl \
  --regimen facts_with_confidence

python -m src.build_training_regimens \
  --input path/to/confidence_rich_source.jsonl \
  --output data/train_syllogism_with_confidence.jsonl \
  --regimen syllogism_with_confidence
```

The staged curriculum for mixing those regimens now lives in `src/training_strategy.py`. It encodes:

1. base warm start
2. multi-task mixture with the base task dominant
3. optional later score refinement once better judge labels exist

You can materialize the default strategy JSON with:

```bash
python -m src.training_strategy --output training_strategy.json
```

The downstream evaluation harness now lives in `src/evaluate_regimens.py`. It scores whether confidence is useful rather than whether it merely matches synthetic numbers:

```bash
python -m src.evaluate_regimens \
  --input eval/scored_holdout.jsonl \
  --acceptance-threshold 0.7
```

That JSONL should contain at least:

```json
{"quote":"...", "predicted_confidence":0.82, "syllogism_quality":0.91}
```

where `syllogism_quality` is your downstream judge or rubric score on a normalized 0-1 scale.

If you need to rebuild the canonical corpora from recoverable upstream artifacts, use:

```bash
python -m src.rebuild_training_corpora \
  --confidence-source /tmp/gen-qwen3-qlora/output/train_preprocessed_structured_967.jsonl \
  --conclusion-source /tmp/triplet-abductive-native-full-20250501/output/train.section-format.backup.jsonl
```

If you want to run the ablation matrix directly, use:

```bash
python -m src.run_ablation_matrix \
  --output-dir output/ablations_run \
  --holdout-fraction 0.1 \
  --max-holdout-records 32 \
  --experiment base-only
```

That run now emits:

- per-experiment `results.json`
- `ablation_summary.json`
- `holdout_examples.md` with side-by-side sampled holdout outputs for each ablation
- live per-experiment and per-stage progress lines during training/eval

If you want to run the repo's SPO fine-tuning stage on a confidence-bearing dataset, use:

```bash
python -m src.run_spo_training \
  --adapter-path output/ablations_chatfix_baseonly/base-only/adapter \
  --dataset-path data/train_facts_with_confidence_967.jsonl \
  --output-dir output/spo_chatfix_facts \
  --evaluation-metric triplet \
  --num-epochs 1
```

That run writes a new adapter plus `spo_summary.json` with per-step loss/reward history.

**SPO reward design notes:**

- Rewards are pre-computed from the training data itself — no generation inside the training loop.
  Each sample gets a quality weight based on: unique-premises ratio, mean predicate specificity,
  and subject diversity (outputs where every line starts with `the speaker` are down-weighted).
- `evaluate_triplet_correctness` hard-zeros any output where more than half the triplet lines are
  duplicates, and deducts for self-referential triplets (e.g. `subject | is | is subject`).
  This prevents the SPO loop from reinforcing repetitive looping behaviour.
- Generation uses `repetition_penalty=1.3` and `no_repeat_ngram_size=4` throughout. Without these,
  greedy decoding on an under-trained model produces exact-line repetition indefinitely.

## Why the format matters

The core design choice is that the repo uses different orders for different stages of the pipeline.

| Stage | Order | Why |
|---|---|---|
| Generation | Throughline -> Entailed -> Non-Entailed | Natural reasoning flow for the generating LLM |
| Training | Non-Entailed -> Entailed -> Throughline | Negative inference first; better contrastive signal |
| Inference | Non-Entailed -> Entailed -> Throughline | The model tends to emit what it was taught |

This matters because the training target is not just "state the answer." It teaches the model what does **not** support the conclusion before teaching what does. That is the job of **non-entailed premises**.

For the full specification, examples, and exact `Input` / `Completion` layout, read `docs/format/README.md`.

## How the data gets cleaned before training

The repo's filtering/cleaning path is conservative and explicit rather than magic:

1. **Parse the hybrid record** via `src/preprocess_training_data.py`
   - Pull the quote out of `input_text`
   - Extract `Entailed Premises`, `Non-Entailed Premises`, and `Conclusion` / `Syllogism`
2. **Normalize section values**
   - Empty or explicit `N/A` sections become `None` in the structured record
   - Missing markdown wrappers are handled by the section extractor when possible
3. **Preserve reasoning signal**
    - Triplets stay intact
    - Evidence tags are preserved
    - Numeric confidence can remain in structured synthetic records for audit, but it is stripped from training rows so downstream judges can assign scores later
    - The cleaned corpus in `data/train_clean_for_model_967.jsonl` reflects the confidence-free training target
4. **Drop malformed rows at preprocessing time**
   - If a line fails JSON parsing or record conversion, it increments the preprocessing error count and is not written to the cleaned output
5. **Serialize survivors into training format**
   - `src/serialize_training_format.py` writes the final pedagogical order:
     `Non-Entailed Premises -> Entailed Premises -> Throughline`
   - Empty sections are written back out as explicit `N/A`

In other words, the repo is not training directly on whatever the generator spit out. It parses, normalizes, preserves the reasoning structure, and only then serializes clean training rows without static numeric confidence labels.

## Model configuration: where it lives

One of the prior pain points was making generation-model and inference-model setup too easy to miss. The split is now:

| Need | Where to go |
|---|---|
| Configure an LLM to generate synthetic data | `docs/generation/README.md` |
| Load a fine-tuned adapter or set inference parameters | `docs/inference/README.md` |
| Configure QLoRA training knobs | `docs/training/README.md` |
| Understand why confidence should stay downstream of the base training target | `docs/architecture/README.md` |

## Repo map

```text
spo-reasoning-training-regimen/
├── src/
│   ├── synthetic_generator.py
│   ├── build_training_regimens.py
│   ├── training_strategy.py
│   ├── evaluate_regimens.py
│   ├── spo_trainer.py
│   ├── pipeline.py
│   ├── training_config.py
│   ├── graph_ontology.py
│   └── ...
├── docs/
│   ├── generation/README.md
│   ├── training/README.md
│   ├── inference/README.md
│   ├── format/README.md
│   ├── architecture/README.md
│   └── quickstart/README.md
└── data/
    ├── sample_quotes.txt
    ├── examples_training_format.jsonl
    ├── examples_facts_with_confidence.jsonl
    ├── examples_syllogism_with_confidence.jsonl
    ├── SEEING_IS_BELIEVING_EXAMPLES.md
    ├── train_clean_for_model_967.jsonl
    ├── train_facts_with_confidence_967.jsonl
    └── train_syllogism_with_confidence_967.jsonl
```

## Seeing-is-believing artifacts

If you want to inspect the output shape before generating anything, start here:

| Artifact | Why it matters |
|---|---|
| `data/SEEING_IS_BELIEVING_EXAMPLES.md` | Human-readable examples of quote -> structured reasoning |
| `data/examples_training_format.jsonl` | The same examples in training-ready JSONL |
| `data/examples_facts_with_confidence.jsonl` | Example follow-on task for premise scoring |
| `data/examples_syllogism_with_confidence.jsonl` | Example follow-on task for throughline scoring |
| `data/sample_quotes.txt` | Easy starter input set |
| `data/train_clean_for_model_967.jsonl` | Larger cleaned training corpus used in prior work |
| `data/train_facts_with_confidence_967.jsonl` | Follow-on fact-confidence regimen |
| `data/train_syllogism_with_confidence_967.jsonl` | Follow-on syllogism-confidence regimen |

## Documentation guide

| Goal | Read |
|---|---|
| Generate datasets from quotes | `docs/generation/README.md` |
| Get running quickly | `docs/quickstart/README.md` |
| Configure models for inference | `docs/inference/README.md` |
| Train with QLoRA | `docs/training/README.md` |
| Understand data format and ordering | `docs/format/README.md` |
| Understand architecture and rationale | `docs/architecture/README.md` |
| Inspect example data and artifacts | `data/README.md` |

## The three phases in one paragraph each

### Phase 1: Generate

Turn raw quotes into structured reasoning examples. This can be done with a hosted LLM, a local model, or plain templates if you want to hand-fill the outputs.

### Phase 2: Train

Take the validated JSONL and fine-tune a base model so it learns the pedagogical reasoning format: non-entailed premises first, then entailed premises, then the throughline.

### Phase 3: Optimize

Apply downstream judging or calibration if you want numeric confidence later. The base model should first learn to emit the right premises and throughline; any scores can be assigned post hoc by a judge or calibration layer.

## Hardware by phase

| Phase | GPU | VRAM | Notes |
|---|---|---|---|
| Generation | Optional | Depends on model | Hosted APIs or local models both work |
| Training | Recommended | 8GB+ | QLoRA keeps smaller models practical |
| SPO | Recommended | 8GB+ | Useful when confidence calibration matters |

## Practical next steps

1. Read `data/SEEING_IS_BELIEVING_EXAMPLES.md` if you want output intuition first
2. Use `docs/generation/README.md` if you need help wiring up OpenAI, Ollama, or Hugging Face
3. Use `docs/format/README.md` if you need the exact prompt and serialization contract
4. Use `docs/training/README.md` when you are ready to fine-tune
5. Use `docs/inference/README.md` when you are ready to load adapters and run new quotes

---

## Lessons Learned

Three compounding bugs caused the initial SPO training to produce worse output than the base adapter. Each one is subtle and worth documenting because they are easy to repeat in any RL-from-feedback setup.

### 1. Greedy decoding without repetition controls loops forever

`model.generate()` with `do_sample=False` and no `repetition_penalty` will lock onto any high-probability token sequence and repeat it indefinitely. The model is not broken — it is being perfectly greedy. A single repeated line scores just as well on format metrics as a unique one, so the problem is invisible to offline evaluation.

**Fix:** Add `repetition_penalty=1.3` and `no_repeat_ngram_size=4` to every `generate()` call used in evaluation or inference. These two parameters eliminate exact-line repetition without degrading structured output format.

### 2. Format-only reward metrics actively reinforce repetition

The original SPO reward checked only the *presence* of `|...|...|` delimiters, a `confidence=` value, and an `observed`/`inferred` tag. A line repeated eight times passes all three checks eight times and earns a reward of 1.0. The reward function was measuring compliance with a template, not quality of reasoning.

**Fix:** Add a uniqueness gate before any format check. If `unique_lines / total_lines < 0.5`, return 0.0 immediately. Also detect tautological triplets where the subject appears verbatim in the object field (e.g. `the speaker | is | is the speaker`) and deduct proportionally. Reward functions must penalise the failure modes they are meant to prevent, not just reward the happy path.

### 3. Scoring gold outputs against gold ground truths yields uniform reward

This is the most insidious bug. When `compute_step()` evaluates gold `output_texts` against gold `ground_truths`, every clean sample scores ~1.0. All training examples receive equal weight. SPO becomes plain SFT with extra steps and no signal. The training loss may look fine; avg_reward hovering at a flat ~0.8 is the only diagnostic clue.

**Fix:** Pre-score each training sample *before* the training loop using data-quality signals that are independent of format compliance: unique-premises ratio, mean predicate specificity, and subject dominance ratio. Embed these scores as `precomputed_reward` in the training batch and use them in `compute_step()` instead of live evaluation. This separates reward differentiation (done offline at dataset construction time) from reward evaluation (which is inherently circular when gold data is both input and ground truth).

### 4. SPO-as-weighted-SFT cannot fix pre-trained abbreviation habits


After 5 epochs on 967 records, the adapter still outputs `Non-Entailed Prems:` and `Entailed Prims:` instead of the full header names from the training corpus. Every training record has the correct full names; SPO is applied on top; the model ignores the correction anyway.

The reason is architectural: SPO as implemented here is **offline weighted SFT**. `compute_step()` upweights loss on high-quality gold tokens and downweights loss on lower-quality ones. It never generates a bad output at training time and penalises it. The base model's abbreviation tendency — reinforced by Qwen pre-training — wins because it is never directly penalised in the gradient signal.

**Fix (future):** Use online RL (GRPO or PPO). Generate a candidate output, score it with `evaluate_triplet_correctness`, compute a reward signal, and backpropagate through the actual generated tokens. This is the only training loop that can penalise a garbled header that the model chose to produce.

**Short-term mitigation:** Constrained decoding (prefix forcing) or a few-shot prefix in the inference prompt that starts the output with the correct headers (`Non-Entailed Premises:\n`) can force correct headers at generation time without retraining.

### 5. For small models, balance the training data across (regimen × length) strata

Single-regimen training on the facts format produced clean, full-length headers throughout. Once training was extended to a second regimen sharing the same semantic domain (logical entailment reasoning), headers degraded to abbreviations (`Entailed Prims:`, `Non-Entailed Prems:`). The original single-regimen training was fine; the problem was introduced entirely by combining regimens.

**Mechanism:** Both regimens use nearly identical header vocabulary ("Entailed Premises:", "Non-Entailed Premises:"). When trained jointly, the gradients from each regimen are co-aligned but not identical — they point at the same token neighborhood with slightly different downstream structure expectations. The model converges to a weighted average over that neighborhood, and the base model's pre-training bias toward abbreviated forms (which were never fully suppressed, just outvoted by the single-regimen signal) re-emerges as the dominant pattern. A secondary amplifier is length imbalance: if one regimen's training records are systematically longer, those records dominate the gradient signal for the shared header tokens, reinforcing that regimen's abbreviation habits.

**Why text prompts cannot solve this:** A text tag like `[FACTS]` occupies the same embedding space as all other vocabulary. For a 0.6B parameter model, learning a clean conditional `[FACTS] → full header names` is infeasible when the tag is semantically adjacent to the content tokens and the regimens share near-identical output vocabulary. This is scale-dependent: larger models (7B+) can learn such conditionals from text prompts alone; small models cannot.

**Why special tokens are also wrong:** Special tokens require downstream users to inject a model-internal routing token they have no natural reason to know about. This creates a hidden contract — the model silently degrades for any user who doesn't know the required prefix. It's bad API design disguised as a training fix.

**The correct fix — stratified sampling across (regimen × length) strata:**
The problem is that the combined training mix is dominated by whichever regimen happens to have more or longer records. Fix this by ensuring every (regimen, prompt-length bucket) cell contributes equally to each training epoch.

The `sample_mixture()` function in `src/run_ablation_matrix.py` now implements this with two-level balancing:

1. **Between regimens** — the caller supplies explicit mixing weights (e.g., 60% base / 25% facts / 15% syllogism). These weights are respected as-is.
2. **Within each regimen** — records are bucketed by their prompt length using quantile cuts (so buckets are always equally populated regardless of the actual length distribution). An equal quota is drawn from each bucket, preventing a regimen's long-prompt records from drowning out its short-prompt examples.

```python
# src/run_ablation_matrix.py — sample_mixture()
#
# Within each regimen, length quantiles are computed on that regimen's own
# records, so bucket boundaries adapt to the distribution rather than
# using a fixed character-count threshold.
# Bucket quota is regimen_quota // n_active_buckets — equal weight per bucket.
```

No special tokens. No separate adapters. No hidden prompt contract. 2× adapter storage cost becomes zero.

**Remaining failure mode if the base adapter already has abbreviation bias baked in:** SPO (offline weighted SFT) cannot correct abbreviation habits — it only reweights gold tokens and never generates bad headers to penalise them. If the starting checkpoint was itself trained on a misbalanced multi-regimen mix, the abbreviation pattern will survive SPO. The fix is to apply stratified sampling at the **base training stage**, not just the fine-tuning stage.

**Short-term mitigation (no retraining):** Constrained decoding — use a `LogitsProcessor` to hard-constrain the first K tokens to the correct header sequence. The model generates the right headers without retraining or special tokens in the prompt.

### 6. Evaluation scripts must use the same prompt format as training

Post-training gate verification showed BASE REASONING producing `NO_HEADER` on every sample despite training successfully. The model scored avg_header_score=0.125 on the initial gate run. Increasing `max_new_tokens` from 192 to 512 made no difference. The actual cause: the verification script passed `record["input_text"]` directly to the tokenizer instead of wrapping it in the chat template via `build_generation_prompt(tokenizer, text)`.

**Why it matters:** An instruct model trained with the Qwen chat format (`<|im_start|>user...<|im_end|><|im_start|>assistant`) expects to see that exact structure at inference time. When given raw text continuation input, the model never enters its "assistant turn" generation mode and produces either nothing or instruction echo rather than the expected structured output. The model was not broken — the evaluation harness was using the wrong prompt format.

**Fix:**
```python
from src.chat_format import build_generation_prompt, strip_response_preamble

# Training used:  build_training_conversation(tokenizer, input_text, output_text)
# Inference must: build_generation_prompt(tokenizer, input_text)
chat_prompt = build_generation_prompt(tokenizer, record["input_text"])
# Then strip <think>...</think> from output before scoring:
output = strip_response_preamble(decoded_output)
score = contract.header_score(output)
```

**After fix:** FACTS avg_header_score=1.000, BASE REASONING avg_header_score=0.917, overall=0.958. Gate: PASS.

**Rule of thumb:** Any script that evaluates a fine-tuned instruct model must mirror the exact tokenizer call chain used during training. If training used `apply_chat_template`, evaluation must too. A mismatch is silent — there is no error; the model simply produces irrelevant output. The symptom (NO_HEADER, generic text, instruction echo) is easily misdiagnosed as a training problem or a token-budget problem when the root cause is purely a format mismatch in the evaluation harness.

### Takeaway

For any RL-from-feedback training loop: (1) generation controls must prevent degenerate outputs before rewards are ever computed, (2) reward functions must explicitly penalise known failure modes rather than only rewarding the ideal case, (3) reward signals computed from gold data are always suspect — check that the distribution of rewards across your training set actually varies before assuming SPO is doing anything useful, (4) offline preference optimisation cannot correct a habit the model has never been penalised for producing — if the failure mode is a specific generated token sequence, only online generation-and-penalise RL can reliably fix it, (5) for small models, multi-regimen training on semantically overlapping formats requires stratified sampling across (regimen × prompt-length) strata — without it, whichever regimen dominates the data mix will corrupt the shared header vocabulary for the other regimens, and (6) post-training evaluation scripts must mirror the exact prompt-format pipeline used during training — a chat-template mismatch produces NO_HEADER silently and is trivially diagnosed as a training bug or token-budget problem when the real cause is a one-line harness error.

---

**Status:** Production-ready  
**Repository:** https://github.com/thistleknot/spo-reasoning-training-regimen
