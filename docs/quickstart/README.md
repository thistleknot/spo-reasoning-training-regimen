# Quick Start Guide

Get up and running in 5 minutes with working examples.

## Prerequisites

```bash
git clone https://github.com/thistleknot/spo-reasoning-training-regimen.git
cd spo-reasoning-training-regimen
pip install -r requirements.txt
```

## 1. Generate Reasoning Examples (2 min)

### Generate from Sample Quotes

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from synthetic_generator import SyntheticReasoningGenerator

# Create generator
gen = SyntheticReasoningGenerator()

# Load sample quotes
with open('data/sample_quotes.txt') as f:
    quotes = [line.strip() for line in f if line.strip()]

# Generate examples
examples = gen.generate_from_quotes(quotes)
print(f"Generated {len(examples)} examples")

# Export to JSONL (ready for training)
gen.export_to_jsonl('my_training_data.jsonl')
print("Exported to my_training_data.jsonl")

# Show first example
print("\nFirst example:")
print(f"Quote: {examples[0].quote[:80]}...")
print(f"Entailed premises: {len(examples[0].entailed_premises)}")
print(f"Non-entailed premises: {len(examples[0].non_entailed_premises)}")

EOF
```

**Output:**
```
Generated 10 examples
Exported to my_training_data.jsonl
First example:
Quote: By the pricking of my thumbs, Something wicked this way comes...
Entailed premises: 4
Non-entailed premises: 3
```

### Generate from Your Own Quotes

```bash
cat > my_quotes.txt << 'EOF'
It was a bright cold day in April, and the clocks were striking thirteen.
Call me Ishmael.
It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife.
EOF

python3 << 'PYEOF'
import sys
sys.path.insert(0, 'src')
from synthetic_generator import SyntheticReasoningGenerator

gen = SyntheticReasoningGenerator()

with open('my_quotes.txt') as f:
    quotes = [line.strip() for line in f if line.strip()]

examples = gen.generate_from_quotes(quotes)
gen.export_to_jsonl('my_data.jsonl')

print(f"Generated {len(examples)} examples in my_data.jsonl")

PYEOF
```

## 2. Train a Model (10 min setup, 30+ min training)

### Option A: Using QLoRA (Recommended)

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
import json

# Configuration
model_name = "Qwen/Qwen2-0.5B"  # Small model for demo
output_dir = "qwen_reasoning_adapter"

# Quantization config (4-bit for efficiency)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

# Training config
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_steps=50,
    learning_rate=2e-4,
)

# Training function for formatting
def formatting_func(examples):
    return {"text": examples["output_text"]}

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=load_dataset(
        "json",
        data_files="my_training_data.jsonl",
        split="train"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
    tokenizer=tokenizer,
)

# Train
print("Starting training...")
trainer.train()
print(f"Done! Adapter saved to {output_dir}")

EOF
```

### Option B: Quick Demo (No Real Training)

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from synthetic_generator import SyntheticReasoningGenerator

gen = SyntheticReasoningGenerator()

# Show what training data looks like
examples = gen.generate_from_quotes(["By the pricking of my thumbs..."])

print("Training Record Example:")
print("=" * 60)
print(examples[0].syllogism)
print()
print("This becomes training input/output:")
print("-" * 60)

# Serialize to training format
gen.export_to_jsonl('demo.jsonl')
import json
with open('demo.jsonl') as f:
    record = json.loads(f.readline())
    print(f"Input: {record['input_text']}")
    print()
    print(f"Output:\n{record['output_text']}")

EOF
```

## 3. Run Inference (3 min)

### Inference with Original Model

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

quote = "By the pricking of my thumbs, Something wicked this way comes."

# Inference
inputs = tokenizer(f"Quote: {quote}\nReasoning:", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)

reasoning = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated reasoning:")
print(reasoning)

EOF
```

### Inference with Fine-tuned Model

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
model_name = "Qwen/Qwen2-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load LoRA adapter
adapter_path = "qwen_reasoning_adapter"  # Path to trained adapter
model = PeftModel.from_pretrained(model, adapter_path)

# Run inference
quote = "By the pricking of my thumbs, Something wicked this way comes."
inputs = tokenizer(quote, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)

output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Fine-tuned model output:")
print(output)

EOF
```

## 4. Evaluate Confidence (2 min)

### Extract and Score Confidence

```bash
python3 << 'EOF'
import re
import sys
sys.path.insert(0, 'src')

# Simulated model output
model_output = """
[ENTAILED]
something | is (inferred, confidence=0.85) | wicked
approaching | signals (inferred, confidence=0.92) | danger

[CONCLUSION]
When one feels a premonition, something wicked approaches.
"""

# Extract confidence scores
pattern = r'confidence=([0-9.]+)'
scores = [float(m.group(1)) for m in re.finditer(pattern, model_output)]

print(f"Extracted {len(scores)} confidence scores: {scores}")
print(f"Average confidence: {sum(scores) / len(scores):.2f}")
print(f"Min: {min(scores):.2f}, Max: {max(scores):.2f}")

EOF
```

## 5. View Examples

### See "Seeing is Believing" Artifacts

```bash
cat data/SEEING_IS_BELIEVING_EXAMPLES.md
```

Shows 5 complete examples with:
- Input quote
- Generated triplets (non-entailed + entailed)
- Syllogistic conclusion
- Confidence scores

### View Training Data

```bash
head -1 data/examples_training_format.jsonl | python -m json.tool
```

Shows the JSONL format used for training.

## 6. Full Pipeline

### End-to-End Workflow

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from synthetic_generator import SyntheticReasoningGenerator
from training_config import TrainingFormat

# Step 1: Generate
print("Step 1: Generating reasoning examples...")
gen = SyntheticReasoningGenerator()
quotes = ["By the pricking of my thumbs, Something wicked this way comes."]
examples = gen.generate_from_quotes(quotes)
print(f"  ✓ Generated {len(examples)} examples")

# Step 2: Serialize
print("\nStep 2: Serializing to training format...")
gen.export_to_jsonl('pipeline_demo.jsonl')
print(f"  ✓ Exported to pipeline_demo.jsonl")

# Step 3: Show what training would receive
print("\nStep 3: Training input preview...")
import json
with open('pipeline_demo.jsonl') as f:
    record = json.loads(f.readline())
    
print("Training will receive:")
print(f"  Input (quote): {record['input_text']}")
print(f"  Output (structure):\n    {record['output_text'][:200]}...")

# Step 4: Inference preview
print("\nStep 4: After training, model generates triplets with confidence...")
sample_output = """[ENTAILED]
something | is (inferred, confidence=0.85) | wicked
approaching | signals (inferred, confidence=0.92) | danger

[CONCLUSION]
When one feels a premonition, something wicked approaches."""
print(sample_output)

# Step 5: Evaluation
print("\nStep 5: SPO optimizes for: correctness × confidence")
import re
scores = [float(m.group(1)) for m in re.finditer(r'confidence=([0-9.]+)', sample_output)]
avg_conf = sum(scores) / len(scores)
print(f"  Average confidence: {avg_conf:.2f}")
print(f"  Reward (if 100% correct): {1.0 * avg_conf:.2f}")

EOF
```

## Paths Forward

### Just Want to Generate?
```bash
→ Follow section 1 only
→ Use: src/synthetic_generator.py
→ Output: JSONL ready for your pipeline
```

### Want to Train?
```bash
→ Follow sections 1 + 2
→ Use: QLoRA (recommended) or Hugging Face Trainer
→ Output: Fine-tuned model (adapter weights)
```

### Want Full Pipeline?
```bash
→ Follow all sections 1-5
→ Use: Generation → Training → Inference → Evaluation
→ Output: Trained, calibrated model ready for deployment
```

### Want More Control?
```bash
→ See docs/generation/README.md for model options
→ See docs/training/README.md for hyperparameter tuning
→ See docs/inference/README.md for deployment
→ See docs/format/README.md for format details
```

## Troubleshooting

**"Module not found" error**
```bash
→ Make sure src/ is in Python path
→ Add: sys.path.insert(0, 'src')
```

**CUDA out of memory**
```bash
→ Reduce batch_size in TrainingArguments
→ Enable gradient_checkpointing=True
→ Use 4-bit quantization (bnb_4bit_use_double_quant=True)
```

**Model downloads slowly**
```bash
→ Set environment: HF_HOME=/path/to/cache
→ Use: huggingface-cli download command
→ Or download manually from huggingface.co
```

**Generated reasoning looks wrong**
```bash
→ Use a larger model (0.5B → 7B)
→ Provide more diverse quotes
→ Adjust LLM temperature: 0.7 (default) → 0.5 (more consistent)
```

## Next Steps

- [ ] Train on your own dataset (section 2)
- [ ] Deploy to production (docs/inference/README.md)
- [ ] Optimize confidence with SPO (src/spo_trainer.py)
- [ ] Integrate into your application

## Resources

- Full docs: See README.md for complete guide
- Format spec: docs/format/README.md
- Architecture: docs/architecture/README.md
- Code: src/*.py
- Examples: data/SEEING_IS_BELIEVING_EXAMPLES.md
