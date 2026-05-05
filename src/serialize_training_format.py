"""
Adapter: Convert preprocessed structured data to training format.

Takes clean structured dicts (quote, entailed_premises, non_entailed_premises,
syllogism) and serializes them to input_text/output_text format suitable for training.

This format is CLEAN (no markdown markers, only data).
"""

import json
from typing import Optional


def triplets_to_text(triplets: Optional[list[str]]) -> str:
    """Convert list of triplets to text format for training.
    
    Returns clean text without markdown markers.
    """
    if not triplets:
        return "N/A"
    
    lines = []
    for triplet in triplets:
        lines.append(triplet)
    
    return "\n".join(lines)


def serialize_training_record(structured_record: dict) -> dict:
    """Convert structured record to input_text/output_text format for training.
    
    Training format uses pedagogical ordering:
    - Non-Entailed Premises FIRST (teaches negative inference)
    - Entailed Premises second (teaches positive inference)
    - Throughline last (the conclusion)
    
    Args:
        structured_record: {quote, entailed_premises, non_entailed_premises,
                           throughline}
    
    Returns:
        {input_text, output_text} for trainer
    """
    quote = structured_record.get("quote", "")
    non_entailed = structured_record.get("non_entailed_premises")
    entailed = structured_record.get("entailed_premises")
    throughline = structured_record.get("syllogism")  # Use syllogism key for now, renamed to throughline in output
    
    # INPUT: Quote + Non-Entailed + Entailed + Throughline
    # Pedagogical order: false premises first, true premises second, conclusion last
    # No markdown markers, clean format for model
    input_lines = [
        f'"{quote}"',
        "",
        "Non-Entailed Premises:",
        triplets_to_text(non_entailed),
        "",
        "Entailed Premises:",
        triplets_to_text(entailed),
        "",
        "Throughline:",
        throughline or "N/A",
    ]
    input_text = "\n".join(input_lines)
    
    # For training, output is same as input (the model is trained to reproduce this format)
    output_text = input_text
    
    return {
        "input_text": input_text,
        "output_text": output_text,
    }


def convert_preprocessed_to_training(input_file: str, output_file: str) -> dict:
    """Convert preprocessed structured JSONL to training format JSONL.
    
    Args:
        input_file: Preprocessed structured JSONL
        output_file: Training format JSONL
    
    Returns:
        Statistics dict
    """
    stats = {
        "total": 0,
        "converted": 0,
        "errors": 0,
    }
    
    with open(input_file) as infile, open(output_file, "w") as outfile:
        for line in infile:
            try:
                structured = json.loads(line)
                stats["total"] += 1
                
                # Convert to training format
                training_record = serialize_training_record(structured)
                
                outfile.write(json.dumps(training_record) + "\n")
                stats["converted"] += 1
                
                if stats["converted"] % 200 == 0:
                    print(f"Converted {stats['converted']}...")
            
            except Exception as e:
                stats["errors"] += 1
                print(f"Error converting record: {e}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert preprocessed data to training format"
    )
    parser.add_argument("--input", required=True, help="Preprocessed structured JSONL")
    parser.add_argument("--output", required=True, help="Training format JSONL")
    
    args = parser.parse_args()
    
    stats = convert_preprocessed_to_training(args.input, args.output)
    
    print("\n=== CONVERSION STATISTICS ===")
    print(f"Total: {stats['total']}")
    print(f"Converted: {stats['converted']}")
    print(f"Errors: {stats['errors']}")
