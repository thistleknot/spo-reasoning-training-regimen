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

After base training, optimize confidence with SPO:

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
