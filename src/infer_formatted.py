"""Inference wrapper for semantic reasoning model with formatted output.

Loads trained QLoRA adapter and generates reasoning in GENERATION FORMAT.
Handles input parsing, generation, and output formatting.
"""

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from .chat_format import build_generation_prompt, strip_response_preamble
from .preprocess_training_data import extract_section, parse_triplet_list
from .serialize_training_format import build_base_reasoning_prompt


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
    throughline = extract_section(raw_text, "Throughline")
    if not throughline:
        throughline = extract_section(raw_text, "Conclusion")
    if not throughline:
        throughline = extract_section(raw_text, "Syllogism")

    entailed = parse_triplet_list(extract_section(raw_text, "Entailed Premises")) or []
    non_entailed = (
        parse_triplet_list(extract_section(raw_text, "Non-Entailed Premises")) or []
    )

    sections = {
        "throughline": throughline or None,
        "entailed": entailed,
        "non_entailed": non_entailed,
    }
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
        not_entailed_premises: Retained for backward compatibility but ignored.
            The canonical base-reasoning adapter is trained on the base prompt.
        entailed_premises: Retained for backward compatibility but ignored.
            Premise-conditioned prompting belongs to the follow-on regimens.
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        dict with keys:
            - raw_output: Raw model generation
            - structured: Parsed sections
            - formatted: GENERATION FORMAT output
    """
    input_text = build_base_reasoning_prompt(quote)

    do_sample = temperature > 0
    prompt = build_generation_prompt(tokenizer, input_text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Decode
    raw_output = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    raw_output = strip_response_preamble(raw_output)
    
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
    )
    
    print("\n" + "="*80)
    print("FORMATTED GENERATION OUTPUT (Logical Order)")
    print("="*80)
    print(result["formatted"])
    
    print("\n" + "="*80)
    print("RAW MODEL OUTPUT")
    print("="*80)
    print(result["raw_output"])
