# Synthetic Dataset Generation

How to generate reasoning examples from quotes using configured LLM models.

## Overview

The generation pipeline transforms raw quotes into structured reasoning examples:

```
Quote (text)
    ↓
Model Inference (LLM generates premises + syllogism)
    ↓
Validation (Check format compliance)
    ↓
Serialization (Export to JSONL)
    ↓
Training Data (ready for QLoRA)
```

## Step 1: Configure Your Model

You need an LLM to generate the reasoning. Options:

### Option A: OpenAI API

```python
from src.synthetic_generator import SyntheticReasoningGenerator
from openai import OpenAI

client = OpenAI(api_key="your-key")

def generate_with_openai(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content

gen = SyntheticReasoningGenerator()
examples = gen.generate_from_quotes(quotes, llm_generate_fn=generate_with_openai)
```

### Option B: Local Qwen Model (via Ollama)

```python
import ollama

def generate_with_qwen(prompt: str) -> str:
    response = ollama.generate(
        model="qwen2",  # or qwen:latest
        prompt=prompt,
        stream=False,
    )
    return response['response']

gen = SyntheticReasoningGenerator()
examples = gen.generate_from_quotes(quotes, llm_generate_fn=generate_with_qwen)
```

### Option C: HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1B"  # or your preferred model
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_with_hf(prompt: str) -> str:
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs,
        max_length=1024,
        temperature=0.7,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

gen = SyntheticReasoningGenerator()
examples = gen.generate_from_quotes(quotes, llm_generate_fn=generate_with_hf)
```

### Option D: Anthropic Claude

```python
import anthropic

client = anthropic.Anthropic(api_key="your-key")

def generate_with_claude(prompt: str) -> str:
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

gen = SyntheticReasoningGenerator()
examples = gen.generate_from_quotes(quotes, llm_generate_fn=generate_with_claude)
```

### Option E: No LLM (Template Mode)

```python
gen = SyntheticReasoningGenerator()
examples = gen.generate_from_quotes(quotes)  # Creates templates to fill manually
gen.export_to_json("data/template_examples.json")
# Then manually edit each example with reasoning
```

## Step 2: Prepare Quotes

Quotes can come from various sources:

```python
from pathlib import Path
from src.pipeline import Pipeline, PipelineConfig

# Option A: From text file (one quote per line)
quotes = Pipeline.load_quotes_from_file(Path("data/sample_quotes.txt"))

# Option B: From JSON array
import json
with open("quotes.json") as f:
    data = json.load(f)
    quotes = data["quotes"]  # or just data if it's a list

# Option C: From JSONL
quotes = []
with open("quotes.jsonl") as f:
    for line in f:
        record = json.loads(line)
        quotes.append(record["quote"])

# Option D: Programmatically
quotes = [
    "By the pricking of my thumbs, Something wicked this way comes.",
    "Here is a lesson in creative writing...",
    # ...
]
```

## Step 3: Generate Examples

```python
from src.synthetic_generator import SyntheticReasoningGenerator

gen = SyntheticReasoningGenerator()

# Generate with LLM
examples = gen.generate_from_quotes(
    quotes,
    llm_generate_fn=generate_with_openai,  # or your function
)

# Or template-based
examples = gen.generate_from_quotes(quotes)  # No LLM needed
```

## Step 4: Export to Training Format

```python
# Export to JSONL (for training)
gen.export_to_jsonl("data/train.jsonl")

# Export to JSON (for inspection)
gen.export_to_json("data/examples.json")

# Get statistics
stats = gen.stats()
print(f"Generated {stats['total_examples']} examples")
print(f"Total entailed premises: {stats['total_entailed_premises']}")
```

## Full Pipeline Example

```python
from src.pipeline import Pipeline, PipelineConfig
from pathlib import Path

# Configure pipeline
config = PipelineConfig(
    generate_dataset=True,
    quotes_path=Path("data/my_quotes.txt"),
    model_name="Qwen/Qwen3-0.6B",
    batch_size=2,
    num_epochs=3,
)

# Run end-to-end
pipeline = Pipeline(config)
pipeline.run()

# Outputs:
# - data/train.jsonl
# - data/validation.jsonl
# - output/VALIDATION_REPORT.md
```

## Format Specification

The LLM receives this prompt:

```
Given this quote, extract the implicit reasoning:

Quote: "{quote}"

Generate a response with:
1. Non-Entailed Premises (facts mentioned but not supporting the main inference)
2. Entailed Premises (facts that lead to the conclusion)
3. Syllogism (the core reasoning thread)

Format each premise as: subject | relation (tag, confidence=X) | object
- tag: "observed" for explicit facts, "inferred" for derived
- confidence: 1.0 for observed, 0.5-0.9 for inferred

Response:
```

**Expected output format:**
```
Non-Entailed Premises:
  subject1 | relation1 (observed, confidence=1.0) | object1

Entailed Premises:
  subject2 | relation2 (inferred, confidence=0.8) | object2

Syllogism:
  The synthesized reasoning thread
```

See `../format/README.md` for complete format specification.

## Troubleshooting

### LLM Not Generating Valid JSON

Use structured output formats:

```python
# With OpenAI
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    response_format={"type": "json_object"},
)

# With other models, post-process the output
import json
def parse_response(response):
    try:
        return json.loads(response)
    except:
        # Try to extract JSON from response
        import re
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None
```

### Empty or Invalid Premises

Validate each example:

```python
from pydantic import ValidationError
from src.synthetic_generator import ReasoningExample

try:
    example = ReasoningExample.model_validate(raw_example)
except ValidationError as e:
    print(f"Invalid example: {e}")
    # Log and skip
```

### Memory Issues with Large Datasets

Process in batches:

```python
batch_size = 100
for i in range(0, len(quotes), batch_size):
    batch_quotes = quotes[i:i+batch_size]
    batch_examples = gen.generate_from_quotes(
        batch_quotes,
        llm_generate_fn=generate_fn,
    )
    gen.export_to_jsonl(f"data/batch_{i}.jsonl")
```

## Prompt Engineering Tips

### For Better Reasoning

```
Given this quote, extract the key implicit reasoning.

Quote: "{quote}"

Think step-by-step:
1. What facts are explicitly stated?
2. What can we infer from these facts?
3. What is the core reasoning thread?

Format response as:
[Non-Entailed Premises]
[Entailed Premises]
[Syllogism]
```

### For Better Confidence Calibration

```
For each premise, assign confidence:
- 1.0 if directly stated in the quote
- 0.8-0.9 if strongly implied
- 0.5-0.7 if weakly inferred
- 0.2-0.4 if speculative

Higher confidence = more certain the premise is correct.
```

### For Pedagogical Order

```
Order your response:
1. First: Non-entailed premises (red herrings, distractions)
2. Then: Entailed premises (supporting facts)
3. Finally: Syllogism (reasoning synthesis)

This order helps the model learn by discrimination.
```

## See Also

- `../format/README.md` — Complete format specification
- `../inference/README.md` — Model inference configuration
- `../training/README.md` — QLoRA training setup
- `../../data/SEEING_IS_BELIEVING_EXAMPLES.md` — Example outputs
