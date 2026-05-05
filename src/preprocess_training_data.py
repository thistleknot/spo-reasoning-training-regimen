"""
Training data preprocessor: Parse hybrid YAML/Markdown format to structured dicts.

Converts readable hybrid format (with ** markers and text headers) to clean
structured dictionaries suitable for model training.

Input (serialized hybrid format):
  Quote: "..."
  Entailed Premises: [list of triplets]
  Non-Entailed Premises: [list of triplets]
  Syllogism: "text or N/A"

Output (structured training format):
  {
    "quote": "...",
    "entailed_premises": ["triplet1", ...] or None,
    "non_entailed_premises": ["triplet1", ...] or None,
    "syllogism": "text" or None
  }
"""

import json
from typing import Optional


def parse_triplet_list(text: str) -> Optional[list[str]]:
    """Parse bullet list of triplets from text.
    
    Input: "  - triplet1\n  - triplet2\n  - triplet3"
    Output: ["triplet1", "triplet2", "triplet3"]
    
    Returns None if text is "N/A" or empty.
    """
    if not text or text.strip() == "N/A":
        return None
    
    triplets = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            triplet = line[2:].strip()
            if triplet:
                triplets.append(triplet)
    
    return triplets if triplets else None


def extract_section(text: str, header: str) -> str:
    """Extract section content between header and next header or end."""
    # Find header (with markdown markers)
    full_header = f"**{header}**"
    header_start = text.find(full_header)
    if header_start < 0:
        # Try without trailing **
        full_header = f"**{header}"
        header_start = text.find(full_header)
    if header_start < 0:
        return ""
    
    # Start after header line (find newline after header)
    content_start = text.find("\n", header_start)
    if content_start < 0:
        return ""
    content_start += 1
    
    # Find next header (starts with **)
    next_header = len(text)
    for pos in range(content_start, len(text) - 1):
        if text[pos:pos+2] == "**":
            next_header = pos
            break
    
    return text[content_start:next_header].strip()


def extract_quote(text: str) -> str:
    """Extract the raw quote prefix before the first structured section."""
    header_start = text.find("**")
    quote_text = text[:header_start] if header_start > 0 else text
    return quote_text.strip().strip('"').strip()


def preprocess_training_record(record: dict) -> dict:
    """Convert hybrid format record to structured training dict.
    
    Args:
        record: {"input_text": "...", "output_text": "..."}
    
    Returns:
        Structured dict ready for model training
    """
    input_text = record.get("input_text", "")
    output_text = record.get("output_text", "")
    
    quote = extract_quote(input_text)
    
    # Extract entailed_premises (from output)
    entailed_section = extract_section(output_text, "Entailed Premises")
    entailed_premises = parse_triplet_list(entailed_section)
    
    # Extract non_entailed_premises (from output)
    non_entailed_section = extract_section(output_text, "Non-Entailed Premises")
    non_entailed_premises = parse_triplet_list(non_entailed_section)
    
    # Extract syllogism (from output) - was called "Conclusion"
    syllogism_section = extract_section(output_text, "Conclusion")
    if not syllogism_section:
        # Try Syllogism
        syllogism_section = extract_section(output_text, "Syllogism")
    
    syllogism = None
    if syllogism_section and syllogism_section.strip() not in ("", "N/A"):
        syllogism = syllogism_section.strip()
    
    return {
        "quote": quote,
        "entailed_premises": entailed_premises,
        "non_entailed_premises": non_entailed_premises,
        "syllogism": syllogism
    }


def preprocess_training_dataset(input_file: str, output_file: str) -> dict:
    """Transform hybrid format JSONL to structured training JSONL.
    
    Args:
        input_file: Hybrid format JSONL
        output_file: Structured training JSONL
    
    Returns:
        Statistics dict with processing results
    """
    stats = {
        "total": 0,
        "processed": 0,
        "with_entailed": 0,
        "with_non_entailed": 0,
        "with_syllogism": 0,
        "errors": 0
    }
    
    with open(input_file) as infile, open(output_file, "w") as outfile:
        for line in infile:
            try:
                record = json.loads(line)
                stats["total"] += 1
                
                # Preprocess
                structured = preprocess_training_record(record)
                
                # Track stats
                if structured["entailed_premises"]:
                    stats["with_entailed"] += 1
                if structured["non_entailed_premises"]:
                    stats["with_non_entailed"] += 1
                if structured["syllogism"]:
                    stats["with_syllogism"] += 1
                
                # Write
                outfile.write(json.dumps(structured) + "\n")
                stats["processed"] += 1
                
                if stats["processed"] % 100 == 0:
                    print(f"Processed {stats['processed']}...")
            
            except Exception as e:
                stats["errors"] += 1
                print(f"Error processing record: {e}")
    
    return stats


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Preprocess hybrid format training data to structured dicts"
    )
    parser.add_argument("--input", required=True, help="Hybrid format JSONL")
    parser.add_argument("--output", required=True, help="Structured training JSONL")
    
    args = parser.parse_args()
    
    stats = preprocess_training_dataset(args.input, args.output)
    
    print("\n=== PREPROCESSING STATISTICS ===")
    print(f"Total records: {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"  - With entailed premises: {stats['with_entailed']}")
    print(f"  - With non-entailed premises: {stats['with_non_entailed']}")
    print(f"  - With syllogism: {stats['with_syllogism']}")
    print(f"Errors: {stats['errors']}")
