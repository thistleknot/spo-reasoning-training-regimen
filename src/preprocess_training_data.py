"""
Training data preprocessor: Parse readable reasoning records to structured dicts.

Converts hybrid YAML/Markdown and older prompt-shaped JSONL records to clean
structured dictionaries suitable for model training. Legacy candidate scaffolds
are stripped during quote extraction rather than preserved in the canonical
schema.

Input (serialized hybrid or legacy prompt format):
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
import re
from typing import Optional


LEGACY_CANDIDATE_MARKERS = (
    "**Candidate NOT_ENTAILED",
    "Candidate NOT_ENTAILED (for negative inference):",
    "[Candidate NOT_ENTAILED (for negative inference)]",
    "[Candidate NOT_ENTAILED premises]",
)

LEGACY_PROMPT_TAIL_MARKERS = (
    "Generate semantic triplets:",
    "Now generate the full analysis:",
)


def parse_triplet_list(text: str) -> Optional[list[str]]:
    """Parse a bullet list of triplets.

    Returns None when the section is empty or explicitly marked as N/A.
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
    """Extract section content between a markdown header and the next header."""
    full_header = f"**{header}**"
    header_start = text.find(full_header)
    if header_start < 0:
        full_header = f"**{header}"
        header_start = text.find(full_header)
    if header_start < 0:
        return ""

    content_start = text.find("\n", header_start)
    if content_start < 0:
        return ""
    content_start += 1

    next_header = len(text)
    for pos in range(content_start, len(text) - 1):
        if text[pos : pos + 2] == "**":
            next_header = pos
            break

    return text[content_start:next_header].strip()


def _split_at_first_marker(text: str, markers: tuple[str, ...]) -> str:
    """Truncate text at the earliest marker occurrence, if present."""
    indices = [idx for marker in markers if (idx := text.find(marker)) >= 0]
    if not indices:
        return text
    return text[: min(indices)]


def extract_quote(text: str) -> str:
    """Extract the raw quote while discarding legacy prompt scaffolding.

    Preconditions:
        text is the original input_text value from a reasoning record.
    Failure modes:
        Returns the best-effort trimmed remainder if the legacy prompt shape is
        malformed or only partially present.
    """
    quote_text = text.strip()

    if "Quote:" in quote_text:
        quote_text = quote_text.rsplit("Quote:", 1)[-1].strip()

    quote_text = _split_at_first_marker(quote_text, LEGACY_CANDIDATE_MARKERS)
    quote_text = _split_at_first_marker(quote_text, LEGACY_PROMPT_TAIL_MARKERS)
    quote_text = _split_at_first_marker(quote_text, ("**",))

    quote_text = re.sub(r"^Quote:\s*", "", quote_text).strip()
    quote_text = quote_text.strip('"').strip("“").strip("”").strip()
    return quote_text


def sanitize_legacy_record(record: dict) -> dict:
    """Remove legacy candidate scaffolding from a JSONL record.

    Preconditions:
        record is a decoded JSON object from the synthetic/training pipeline.
    Guarantees:
        candidate_not_entailed is removed if present and input_text is reduced
        to the quote-only prompt shape.
    """
    cleaned = dict(record)
    cleaned.pop("candidate_not_entailed", None)

    input_text = cleaned.get("input_text")
    if isinstance(input_text, str):
        cleaned["input_text"] = extract_quote(input_text)

    return cleaned


def preprocess_training_record(record: dict) -> dict:
    """Convert a reasoning-format record to the canonical structured schema."""
    input_text = record.get("input_text", "")
    output_text = record.get("output_text", "")

    quote = extract_quote(input_text)

    entailed_section = extract_section(output_text, "Entailed Premises")
    entailed_premises = parse_triplet_list(entailed_section)

    non_entailed_section = extract_section(output_text, "Non-Entailed Premises")
    non_entailed_premises = parse_triplet_list(non_entailed_section)

    syllogism_section = extract_section(output_text, "Conclusion")
    if not syllogism_section:
        syllogism_section = extract_section(output_text, "Syllogism")

    syllogism = None
    if syllogism_section and syllogism_section.strip() not in ("", "N/A"):
        syllogism = syllogism_section.strip()

    return {
        "quote": quote,
        "entailed_premises": entailed_premises,
        "non_entailed_premises": non_entailed_premises,
        "syllogism": syllogism,
    }


def sanitize_jsonl_dataset(input_file: str, output_file: str) -> dict:
    """Rewrite JSONL records without legacy candidate scaffolding.

    This preserves the original record shape where possible, but removes the
    candidate_not_entailed key and strips prompt-level candidate blocks from
    input_text.
    """
    stats = {
        "total": 0,
        "sanitized": 0,
        "errors": 0,
    }

    with open(input_file) as infile, open(output_file, "w") as outfile:
        for line in infile:
            try:
                record = json.loads(line)
                stats["total"] += 1

                cleaned = sanitize_legacy_record(record)
                outfile.write(json.dumps(cleaned) + "\n")
                stats["sanitized"] += 1

            except Exception as e:
                stats["errors"] += 1
                print(f"Error sanitizing record: {e}")

    return stats


def preprocess_training_dataset(input_file: str, output_file: str) -> dict:
    """Transform reasoning JSONL into canonical structured training JSONL."""
    stats = {
        "total": 0,
        "processed": 0,
        "with_entailed": 0,
        "with_non_entailed": 0,
        "with_syllogism": 0,
        "errors": 0,
    }

    with open(input_file) as infile, open(output_file, "w") as outfile:
        for line in infile:
            try:
                record = json.loads(line)
                stats["total"] += 1

                structured = preprocess_training_record(record)

                if structured["entailed_premises"]:
                    stats["with_entailed"] += 1
                if structured["non_entailed_premises"]:
                    stats["with_non_entailed"] += 1
                if structured["syllogism"]:
                    stats["with_syllogism"] += 1

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

    parser = argparse.ArgumentParser(
        description="Preprocess or sanitize reasoning training data"
    )
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument(
        "--mode",
        choices=("preprocess", "sanitize"),
        default="preprocess",
        help="preprocess=emit structured schema, sanitize=clean legacy prompt scaffolding",
    )

    args = parser.parse_args()

    if args.mode == "sanitize":
        stats = sanitize_jsonl_dataset(args.input, args.output)
        print("\n=== SANITIZATION STATISTICS ===")
        print(f"Total records: {stats['total']}")
        print(f"Sanitized: {stats['sanitized']}")
        print(f"Errors: {stats['errors']}")
    else:
        stats = preprocess_training_dataset(args.input, args.output)
        print("\n=== PREPROCESSING STATISTICS ===")
        print(f"Total records: {stats['total']}")
        print(f"Successfully processed: {stats['processed']}")
        print(f"  - With entailed premises: {stats['with_entailed']}")
        print(f"  - With non-entailed premises: {stats['with_non_entailed']}")
        print(f"  - With syllogism: {stats['with_syllogism']}")
        print(f"Errors: {stats['errors']}")
