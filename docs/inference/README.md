# Inference & Model Configuration

How to configure models for inference to generate synthetic datasets.

## Overview

Inference transforms a trained model into a reasoning generator:

```
Trained Model (QLoRA adapter)
    ↓
Configure Inference (Temperature, max_length, etc.)
    ↓
Load Model & Tokenizer
    ↓
Generate from Quote
    ↓
Parse Output → ReasoningExample
```

## Model Configuration for Inference

### Inference Parameters

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Load model and tokenizer
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure generation
generation_config = GenerationConfig(
    max_length=1024,              # Max output tokens
    temperature=0.7,              # Creativity (0.0=deterministic, 1.0=random)
    top_p=0.9,                    # Nucleus sampling
    top_k=50,                     # Top-k sampling
    do_sample=True,               # Enable sampling (not beam search)
    num_beams=1,                  # 1 = sampling, >1 = beam search
    early_stopping=True,
    repetition_penalty=1.1,       # Penalize repetition
    pad_token_id=tokenizer.eos_token_id,
)

model.generation_config = generation_config
```

### Loading Fine-tuned Models (with QLoRA Adapter)

```python
from peft import AutoPeftModelForCausalLM

# Load base model
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load with QLoRA adapter
model = AutoPeftModelForCausalLM.from_pretrained(
    "path/to/adapter",  # Directory where adapter was saved
    device_map="auto",
    torch_dtype="auto",
)

# Merge adapter into model (optional, for inference)
model = model.merge_and_unload()
```

## Generation Functions for SyntheticReasoningGenerator

### Basic Generation Function

```python
def generate_reasoning(prompt: str, model, tokenizer, max_length: int = 1024) -> str:
    """Generate reasoning from prompt using model."""
    
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    # Decode output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt from output
    response = response[len(prompt):].strip()
    
    return response
```

### With Error Handling

```python
def generate_reasoning_safe(
    prompt: str,
    model,
    tokenizer,
    max_retries: int = 3,
) -> str:
    """Generate reasoning with fallback to template on failure."""
    
    for attempt in range(max_retries):
        try:
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=1024,
                    temperature=0.7,
                    timeout=30,  # 30 second timeout
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Validate response has expected format
            if "Entailed" in response or "Throughline" in response:
                return response[len(prompt):].strip()
            
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                return "[Template: Manual reasoning needed]"
    
    return "[Template: Manual reasoning needed]"
```

### With Batching for Efficiency

```python
def generate_batch_reasoning(
    prompts: list,
    model,
    tokenizer,
    batch_size: int = 4,
) -> list:
    """Generate reasoning for multiple prompts efficiently."""
    
    responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=1024,
                temperature=0.7,
                do_sample=True,
            )
        
        # Decode batch
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)
            response = response[len(batch_prompts[j]):].strip()
            responses.append(response)
    
    return responses
```

## Complete Inference Pipeline

```python
from src.synthetic_generator import SyntheticReasoningGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

# 1. Load model
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Configure generation
generation_config = GenerationConfig(
    max_length=1024,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
model.generation_config = generation_config

# 3. Define generation function
def generate_fn(prompt: str) -> str:
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=1024, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 4. Create generator and generate
gen = SyntheticReasoningGenerator()
quotes = [
    "By the pricking of my thumbs, Something wicked this way comes.",
    "All animals are equal, but some are more equal than others.",
]

examples = gen.generate_from_quotes(quotes, llm_generate_fn=generate_fn)

# 5. Export
gen.export_to_jsonl("data/generated.jsonl")
```

## Quantization for Memory Efficiency

### 4-bit Quantization (bitsandbytes)

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import torch

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Use normally
def generate_fn(prompt: str) -> str:
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_length=1024, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 8-bit Quantization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    load_in_8bit=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
```

## Using Different Models

### Smaller (Fast, Lower Memory)
```python
models = [
    "Qwen/Qwen2-0.5B",
    "gpt2",
    "distilbert-base-uncased",
    "EleutherAI/pythia-70m",
]
```

### Medium (Balanced)
```python
models = [
    "Qwen/Qwen3-1B",
    "meta-llama/Llama-2-7b",
    "mistralai/Mistral-7B",
]
```

### Larger (Better Quality, More Memory)
```python
models = [
    "Qwen/Qwen3-0.6B-Instruct",
    "meta-llama/Llama-2-13b",
    "mistralai/Mistral-7B-Instruct",
]
```

## Device Management

### GPU
```python
import torch

# Auto (recommended)
device_map = "auto"

# Specific GPU
device_map = {"": torch.device("cuda:0")}

# Multi-GPU
device_map = "auto"  # Splits model across GPUs
```

### CPU Offloading
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="./offload",  # Cache intermediate computations
)
```

## Prompting for Better Inference

### Structured Prompt
```python
SYSTEM_PROMPT = """You are an expert at extracting structured reasoning from quotes.

For each quote, provide:
1. Non-Entailed Premises: Facts mentioned but not core to reasoning
2. Entailed Premises: Facts that support the main insight
3. Throughline: The core reasoning thread

Format each premise as: subject | relation (observed/inferred, confidence=X) | object"""

def generate_with_system_prompt(quote: str, model, tokenizer) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nQuote: {quote}\n\nResponse:"
    # Generate as normal
    ...
```

### Few-Shot Prompting
```python
EXAMPLES = """
Example 1:
Quote: "By the pricking of my thumbs, Something wicked this way comes."

Non-Entailed Premises:
  thumbs | are (observed, confidence=1.0) | pricking

Entailed Premises:
  something | is (inferred, confidence=0.85) | wicked

Throughline: Physical sensations signal approaching danger.
"""

def generate_with_examples(quote: str, model, tokenizer) -> str:
    prompt = f"{EXAMPLES}\n\nQuote: {quote}\n\nResponse:"
    # Generate as normal
    ...
```

## Troubleshooting

### OOM (Out of Memory)
```python
# Use smaller model
model_name = "Qwen/Qwen2-0.5B"

# Or use quantization
bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)

# Or reduce batch size
batch_size = 1

# Or use cpu offloading
device_map = {"": torch.device("cpu")}
```

### Slow Generation
```python
# Use smaller model
# Use quantization
# Reduce max_length
# Use num_beams=1 (no beam search)
# Enable flash attention (if supported)
```

### Poor Output Quality
```python
# Adjust temperature (0.7 good, try 0.5-0.9)
# Use top_p sampling (nucleus sampling)
# Use few-shot prompting
# Fine-tune model on your data
```

## See Also

- `../generation/README.md` — Dataset generation workflow
- `../training/README.md` — QLoRA training to fine-tune models
- `../../src/synthetic_generator.py` — Generation implementation
- `../../src/spo_trainer.py` — SPO training for confidence calibration
