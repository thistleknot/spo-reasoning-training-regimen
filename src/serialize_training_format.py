"""
Adapter: Convert preprocessed structured data to training format.

Takes clean structured dicts (quote, entailed_premises, non_entailed_premises,
syllogism) and serializes them to input_text/output_text format suitable for
training. Numeric confidence is stripped by default so the training target
teaches premise structure and throughline text, while downstream judges or
calibrators remain free to assign scores later.
"""

import json
import re
from typing import Optional


CONFIDENCE_ANNOTATION_RE = re.compile(
    r"\((observed|inferred)\s*,\s*confidence\s*[:=]\s*[^)]+\)"
)


def normalize_quote_text(quote: str) -> str:
    """Normalize a quote so the prompt surface adds exactly one quote wrapper."""
    return str(quote).strip().strip('"').strip("“").strip("”").strip()


def build_base_reasoning_prompt(quote: str) -> str:
    """Build the canonical base-regimen prompt surface.

    Preconditions:
        `quote` is the raw quote text from the structured corpus.
    Failure modes:
        Returns a best-effort prompt even when the quote is empty.
    """
    normalized_quote = normalize_quote_text(quote)
    return "\n".join(
        [
            "Given this quote, extract the implicit reasoning.",
            "",
            f'Quote: "{normalized_quote}"',
            "",
            "Generate a response with:",
            "1. Non-Entailed Premises",
            "2. Entailed Premises",
            "3. Throughline",
            "",
            "Format each premise as: subject | relation (tag) | object",
            '- tag: "observed" for explicit facts, "inferred" for derived facts',
            "",
            "Response:",
        ]
    )


def strip_confidence_annotation(triplet: str) -> str:
    """Remove numeric confidence while preserving the evidence tag.

    Preconditions:
        triplet follows the repo's serialized premise shape when confidence is
        present: ``subject | relation (tag, confidence=X) | object``.
    Failure modes:
        If no confidence annotation is present, the original triplet is returned
        unchanged.
    """
    return CONFIDENCE_ANNOTATION_RE.sub(r"(\1)", triplet)


def triplets_to_text(
    triplets: Optional[list[str]],
    include_confidence: bool = False,
) -> str:
    """Convert a list of triplets to text format for training."""
    if not triplets:
        return "N/A"

    lines = []
    for triplet in triplets:
        if include_confidence:
            lines.append(triplet)
        else:
            lines.append(strip_confidence_annotation(triplet))

    return "\n".join(lines)


def serialize_training_record(
    structured_record: dict,
    include_confidence: bool = False,
) -> dict:
    """Convert structured record to input_text/output_text format for training.

    Training format uses pedagogical ordering:
    - Non-Entailed Premises FIRST (teaches negative inference)
    - Entailed Premises second (teaches positive inference)
    - Throughline last (the conclusion)

    Args:
        structured_record: {quote, entailed_premises, non_entailed_premises,
                           throughline}
        include_confidence: Preserve numeric confidence in serialized triplets.

    Returns:
        {input_text, output_text} for trainer
    """
    quote = structured_record.get("quote", "")
    non_entailed = structured_record.get("non_entailed_premises")
    entailed = structured_record.get("entailed_premises")
    throughline = structured_record.get("syllogism")  # Use syllogism key for now, renamed to throughline in output
    
    input_text = build_base_reasoning_prompt(quote)

    output_lines = [
        "Non-Entailed Premises:",
        triplets_to_text(non_entailed, include_confidence=include_confidence),
        "",
        "Entailed Premises:",
        triplets_to_text(entailed, include_confidence=include_confidence),
        "",
        "Throughline:",
        throughline or "N/A",
    ]
    output_text = "\n".join(output_lines)
    
    return {
        "input_text": input_text,
        "output_text": output_text,
    }


def convert_preprocessed_to_training(
    input_file: str,
    output_file: str,
    include_confidence: bool = False,
) -> dict:
    """Convert preprocessed structured JSONL to training format JSONL.

    Args:
        input_file: Preprocessed structured JSONL
        output_file: Training format JSONL
        include_confidence: Preserve numeric confidence in training rows.

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
                training_record = serialize_training_record(
                    structured,
                    include_confidence=include_confidence,
                )
                
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
    parser.add_argument(
        "--include-confidence",
        action="store_true",
        help="Keep numeric confidence annotations in the serialized training rows",
    )
    
    args = parser.parse_args()
    
    stats = convert_preprocessed_to_training(
        args.input,
        args.output,
        include_confidence=args.include_confidence,
    )
    
    print("\n=== CONVERSION STATISTICS ===")
    print(f"Total: {stats['total']}")
    print(f"Converted: {stats['converted']}")
    print(f"Errors: {stats['errors']}")
