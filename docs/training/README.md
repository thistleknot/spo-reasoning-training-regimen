# Training Setup

How to train models with QLoRA on the generated datasets.

## Overview

Training transforms structured reasoning data into a fine-tuned model:

```
Generated Dataset (JSONL)
    ↓
Prepare Data (Tokenization, batching)
    ↓
Configure QLoRA (LoRA rank, alpha, learning rate)
    ↓
Train Model (Fine-tune with adapter)
    ↓
Save Adapter (Merged or separate)
    ↓
Use for Inference
```

## Basic QLoRA Training

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# 1. Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# 2. Configure LoRA
lora_config = LoraConfig(
    r=32,                          # LoRA rank
    lora_alpha=64,                 # LoRA alpha
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# 3. Load dataset
dataset = load_dataset("json", data_files="data/train.jsonl")
train_dataset = dataset["train"].train_test_split(0.9)["train"]
val_dataset = dataset["train"].train_test_split(0.9)["test"]

# 4. Tokenize
def preprocess(examples):
    return tokenizer(
        examples["output_text"],
        truncation=True,
        max_length=512,
    )

train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)

# 5. Configure training
training_args = TrainingArguments(
    output_dir="./output",
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    optim="paged_adamw_8bit",
)

# 6. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# 7. Save adapter
model.save_pretrained("./adapter")
tokenizer.save_pretrained("./adapter")
```

## Data Preparation

This repo now has three adjacent dataset families:

1. `train_clean_for_model_967.jsonl` — base reasoning target with no numeric confidence
2. `train_facts_with_confidence_967.jsonl` — quote -> premises with confidence
3. `train_syllogism_with_confidence_967.jsonl` — quote + confidence-bearing facts -> throughline + confidence

Use the first regimen to teach reasoning structure. Use the other two as follow-on multi-task or staged fine-tuning datasets.

### Build the Follow-On Regimens

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

The first builder keeps premise-level confidence. The second uses those confidence-bearing facts as input and emits `Throughline` plus an aggregate `Confidence` target.

## Staged Multi-Regimen Curriculum

The repo now has a first-class curriculum object in `src/training_strategy.py`. It makes the research conclusion explicit instead of leaving it as README advice.

### Default stages

1. **Base warm start**  
   Train on `train_clean_for_model_967.jsonl` first so the adapter learns quote -> premises + throughline without frozen numeric targets.
2. **Multi-task mix**  
   Add the two confidence-bearing regimens with the base task still dominant.
3. **Optional score refinement**  
   Replace bootstrap synthetic scores later with judge labels, preference data, or GRPO if that becomes available.

### Default initial mixture

| Regimen | Weight |
|---|---:|
| Base reasoning | 0.60 |
| Facts with confidence | 0.25 |
| Syllogism with confidence | 0.15 |

These are ablation starting points, not immutable truths.

### Write the default strategy config

```bash
python -m src.training_strategy --output training_strategy.json
```

You can then point your higher-level pipeline config at that file through `PipelineConfig.training_strategy_path`.

## Downstream Evaluation Contract

The right question is not "did the model copy the synthetic score?" The right question is "did the confidence layer help us keep better syllogisms and reject worse ones?"

The repo now includes `src/evaluate_regimens.py` for scored-holdout evaluation.

### Expected scored holdout format

Each JSONL row should contain at least:

```json
{"quote":"...", "predicted_confidence":0.82, "syllogism_quality":0.91}
```

Supported aliases:

- `predicted_confidence` or `confidence`
- `syllogism_quality` or `judge_score`

### Metrics

- Pearson and Spearman correlation between confidence and syllogism quality
- AUROC for pass/fail acceptance
- Brier score
- Expected Calibration Error (ECE)
- Risk-coverage curve for abstention or filtering

### Run the evaluation harness

```bash
python -m src.evaluate_regimens \
  --input eval/scored_holdout.jsonl \
  --acceptance-threshold 0.7
```

Set the threshold to the minimum judge score you consider acceptable for a production syllogism.

## Repairing and Rebuilding Canonical Corpora

If the canonical `data/*.jsonl` corpora drift into a broken state, the repo now has a repair path in `src/rebuild_training_corpora.py`.

That script merges:

1. a confidence-bearing structured source for premises
2. a conclusion-bearing legacy backup for throughlines

and rewrites:

- `data/train_structured_967.jsonl`
- `data/train_clean_for_model_967.jsonl`
- `data/train_facts_with_confidence_967.jsonl`
- `data/train_syllogism_with_confidence_967.jsonl`

Example:

```bash
python -m src.rebuild_training_corpora \
  --confidence-source /tmp/gen-qwen3-qlora/output/train_preprocessed_structured_967.jsonl \
  --conclusion-source /tmp/triplet-abductive-native-full-20250501/output/train.section-format.backup.jsonl
```

## Running the Ablation Matrix

The repo now includes `src/run_ablation_matrix.py` to execute:

1. `base-only`
2. `base-plus-facts`
3. `base-plus-facts-plus-syllogism`

against the default curriculum.

Example:

```bash
python -m src.run_ablation_matrix \
  --output-dir output/ablations_run \
  --holdout-fraction 0.1 \
  --max-holdout-records 32
```

This writes per-experiment `results.json` files plus a top-level `ablation_summary.json`.
It also writes `holdout_examples.md`, which shows the sampled holdout prompts,
expected outputs, and each ablation's generated output side by side.

### Format Data for Training

```python
from datasets import load_dataset, DatasetDict
import json

# Load JSONL
dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train",
)

# Split train/val
splits = dataset.train_test_split(test_size=0.1)
train_dataset = splits["train"]
val_dataset = splits["test"]

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

def tokenize_function(examples):
    return tokenizer(
        examples["output_text"],  # Model trains to predict output
        truncation=True,
        max_length=512,
        padding="max_length",
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Remove non-tensor columns
train_dataset = train_dataset.remove_columns(["input_text", "output_text"])
val_dataset = val_dataset.remove_columns(["input_text", "output_text"])

print(f"Train: {len(train_dataset)} examples")
print(f"Val: {len(val_dataset)} examples")
```

### Custom Data Collator

```python
from transformers import DataCollatorForLanguageModeling

# Built-in data collator (for language modeling)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Not masked language modeling
    return_tensors="pt",
)

# Or custom
def custom_data_collator(batch):
    input_ids = torch.stack([ex["input_ids"] for ex in batch])
    attention_mask = torch.stack([ex["attention_mask"] for ex in batch])
    labels = input_ids.clone()
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
```

## QLoRA Configuration Options

### Conservative (Low rank, slow training)
```python
lora_config = LoraConfig(
    r=8,                    # Lower rank = fewer parameters
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
```

### Balanced (Recommended)
```python
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
```

### Aggressive (High rank, more parameters)
```python
lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
)
```

## Training Hyperparameters

### Learning Rate
```python
# Typical range: 1e-5 to 5e-4
# QLoRA often uses higher LR than full fine-tuning
learning_rates = {
    "small_model": 2e-4,      # Qwen 0.6B
    "medium_model": 1e-4,     # Llama 7B
    "large_model": 5e-5,      # Llama 13B
}
```

### Batch Size
```python
# Typical: 2-8 per device
# With gradient accumulation: 2 * 4 = 8 effective batch
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch = 8
)
```

### Number of Epochs
```python
# Typical: 1-5 epochs
# Depends on dataset size
# Smaller datasets: more epochs (5-10)
# Larger datasets: fewer epochs (1-3)
```

## Training with SPO (Optional)

After base training, you can add a downstream scorer or calibration pass. The base `output_text` target should stay confidence-free:

```text
Non-Entailed Premises:
  thumbs | are (observed) | pricking

Entailed Premises:
  something | is (inferred) | wicked

Throughline:
  When one feels a premonition or intuitive sense, something bad is approaching.
```

If you still want SPO-style optimization, do it after the base model is already emitting premises + throughline reliably:

```python
from src.spo_trainer import SPOTrainer, SPOEvaluator

# Create SPO trainer
spo_trainer = SPOTrainer(
    model=model,
    tokenizer=tokenizer,
    evaluation_fn=SPOEvaluator.composite_score,
    beta=0.1,  # KL penalty
)

# SPO training loop
for epoch in range(spo_epochs):
    for batch in spo_dataloader:
        metrics = spo_trainer.training_step(batch, ground_truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

See `../../src/spo_trainer.py` for full SPO implementation.

## Saving and Loading

### Save Adapter Only
```python
# Save LoRA adapter (small, ~10MB)
model.save_pretrained("./adapter")
tokenizer.save_pretrained("./adapter")

# Load later
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = PeftModel.from_pretrained(base_model, "./adapter")
```

### Merge Adapter into Model
```python
# Merge adapter into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")

# Load merged model
model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

## Distributed Training

### Multi-GPU
```python
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    # Trainer auto-detects multiple GPUs
    # Use CUDA_VISIBLE_DEVICES="0,1" to select GPUs
)
```

### Using accelerate
```bash
accelerate config  # Interactive setup
accelerate launch train.py
```

## Monitoring Training

### With Weights & Biases
```python
import wandb

wandb.init(project="spo-reasoning")

training_args = TrainingArguments(
    report_to=["wandb"],
    run_name="qwen-spo-v1",
    # ... other args
)
```

### With TensorBoard
```python
training_args = TrainingArguments(
    report_to=["tensorboard"],
    # ... other args
)

# View: tensorboard --logdir ./output
```

## Troubleshooting

### OOM During Training
```python
# Reduce batch size
per_device_train_batch_size=1

# Increase gradient accumulation
gradient_accumulation_steps=8  # More steps, fewer GPU memory

# Use 4-bit quantization (instead of 8-bit)
load_in_4bit=True

# Reduce max_length
max_length=256  # Instead of 512
```

### Loss Not Decreasing
```python
# Check learning rate (try 5e-4, 2e-4, 1e-4)
learning_rate=2e-4

# Check data (print samples)
for batch in train_loader:
    print(tokenizer.decode(batch["input_ids"][0]))
    break

# Longer training
num_train_epochs=5
```

### CUDA Issues
```bash
# Check GPU
nvidia-smi

# Clear cache
torch.cuda.empty_cache()

# Use CPU (very slow)
device_map="cpu"
```

## Performance Benchmarks

| Model | Batch | Epochs | Duration | VRAM |
|-------|-------|--------|----------|------|
| Qwen 0.6B | 2 | 3 | ~30 min | 8GB |
| Qwen 1B | 2 | 3 | ~60 min | 12GB |
| Llama 7B | 1 | 3 | ~2 hours | 16GB |

## See Also

- `../generation/README.md` — Dataset generation
- `../inference/README.md` — Model inference configuration
- `../../data/examples_training_format.jsonl` — Example training data
- `../../src/spo_trainer.py` — SPO implementation
