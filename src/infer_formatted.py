"""Inference wrapper for semantic reasoning model with formatted output.

Loads trained QLoRA adapter and generates reasoning in GENERATION FORMAT.
Handles input parsing, generation, and output formatting.
"""

import json
import re
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def load_model_and_tokenizer(adapter_path: str):
    """Load fine-tuned model and tokenizer.
    
    Args:
        adapter_path: Path to QLoRA adapter directory
        
    Returns:
        (model, tokenizer) tuple
    """
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return model, tokenizer


def extract_structured_output(raw_text: str) -> dict:
    """Parse raw model output into structured sections.
    
    Args:
        raw_text: Model-generated text
        
    Returns:
        dict with sections: throughline, entailed, non_entailed
    """
    sections = {
        "throughline": None,
        "entailed": [],
        "non_entailed": []
    }
    
    # Parse throughline
    throughline_match = re.search(
        r"(?:Throughline|Conclusion):\s*\n\s*(.+?)(?=\n\n(?:Entailed|Non-Entailed)|$)",
        raw_text,
        re.DOTALL
    )
    if throughline_match:
        sections["throughline"] = throughline_match.group(1).strip()
    
    # Parse entailed premises
    entailed_match = re.search(
        r"Entailed\s+Premises:\s*\n((?:(?:\s*-\s*.+\n)*)+)",
        raw_text
    )
    if entailed_match:
        premises = entailed_match.group(1)
        for line in premises.split("\n"):
            line = line.strip()
            if line and line.startswith("-"):
                sections["entailed"].append(line[1:].strip())
    
    # Parse non-entailed premises
    non_entailed_match = re.search(
        r"Non-Entailed\s+Premises:\s*\n((?:(?:\s*-\s*.+\n)*)+)",
        raw_text
    )
    if non_entailed_match:
        premises = non_entailed_match.group(1)
        for line in premises.split("\n"):
            line = line.strip()
            if line and line.startswith("-"):
                sections["non_entailed"].append(line[1:].strip())
    
    return sections


def format_generation_output(structured: dict) -> str:
    """Format structured output in GENERATION FORMAT.
    
    GENERATION FORMAT uses logical order:
    1. Throughline (the conclusion)
    2. Entailed Premises (what supports the conclusion)
    3. Non-Entailed Premises (what was present but not used)
    
    Args:
        structured: dict with throughline, entailed, non_entailed keys
        
    Returns:
        Formatted string in GENERATION FORMAT
    """
    output = []
    
    # Throughline
    if structured["throughline"]:
        output.append("Throughline:")
        output.append(f"  {structured['throughline']}")
        output.append("")
    
    # Entailed Premises
    output.append("Entailed Premises:")
    if structured["entailed"]:
        for premise in structured["entailed"]:
            output.append(f"  - {premise}")
    else:
        output.append("  - N/A")
    output.append("")
    
    # Non-Entailed Premises
    output.append("Non-Entailed Premises:")
    if structured["non_entailed"]:
        for premise in structured["non_entailed"]:
            output.append(f"  - {premise}")
    else:
        output.append("  - N/A")
    
    return "\n".join(output)


def infer(
    quote: str,
    model,
    tokenizer,
    not_entailed_premises: list = None,
    entailed_premises: list = None,
    max_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.9
) -> dict:
    """Generate reasoning for a quote.
    
    Args:
        quote: The quote to reason about
        model: Loaded model
        tokenizer: Loaded tokenizer
        not_entailed_premises: Optional list of false premises for contrastive context
        entailed_premises: Optional list of true premises
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        dict with keys:
            - raw_output: Raw model generation
            - structured: Parsed sections
            - formatted: GENERATION FORMAT output
    """
    # Build training-format input (what model was trained on)
    input_text = quote + "\n\n"
    
    # Add non-entailed premises if provided
    if not_entailed_premises:
        input_text += "Non-Entailed Premises:\n"
        for premise in not_entailed_premises:
            input_text += f"  - {premise}\n"
    else:
        input_text += "Non-Entailed Premises:\n  - N/A\n"
    
    # Add entailed premises if provided
    if entailed_premises:
        input_text += "\nEntailed Premises:\n"
        for premise in entailed_premises:
            input_text += f"  - {premise}\n"
    else:
        input_text += "\nEntailed Premises:\n  - N/A\n"
    
    input_text += "\nThroughline:\n  N/A"
    
    # Generate
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True
    )
    
    # Decode
    raw_output = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    # Parse and format
    structured = extract_structured_output(raw_output)
    formatted = format_generation_output(structured)
    
    return {
        "quote": quote,
        "raw_output": raw_output,
        "structured": structured,
        "formatted": formatted
    }


if __name__ == "__main__":
    # Example usage
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        "/tmp/gen-qwen3-qlora/output/training_clean_dataset/best_adapter/"
    )
    
    print("\nRunning inference example...")
    result = infer(
        quote='"The greatest glory in living lies not in never falling, but in rising every time we fall."',
        model=model,
        tokenizer=tokenizer,
        not_entailed_premises=[
            "failure | is (observed, confidence=1.0) | permanent",
            "struggle | is (observed, confidence=1.0) | shameful"
        ]
    )
    
    print("\n" + "="*80)
    print("FORMATTED GENERATION OUTPUT (Logical Order)")
    print("="*80)
    print(result["formatted"])
    
    print("\n" + "="*80)
    print("RAW MODEL OUTPUT")
    print("="*80)
    print(result["raw_output"])
