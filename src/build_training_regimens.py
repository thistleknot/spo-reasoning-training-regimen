"""
Build additional training regimens from structured or confidence-bearing records.

This module keeps the existing base regimen intact:
    quote -> premises + throughline (no numeric confidence)

It adds two follow-on supervised tasks that preserve numeric confidence where it
is useful:
    1. facts_with_confidence:
       quote -> non-entailed + entailed premises with confidence
    2. syllogism_with_confidence:
       quote + confidence-bearing facts -> throughline + aggregate confidence

Preconditions:
    Input JSONL rows must be either:
    - structured records with quote / entailed_premises / non_entailed_premises /
      syllogism, or
    - training-style rows with input_text / output_text that still preserve
      confidence-bearing premises.

Failure modes:
    Rows that cannot be normalized or serialized are skipped and counted.
"""

import json
import re
from enum import Enum
from typing import Optional

from .preprocess_training_data import preprocess_training_record
from .serialize_training_format import normalize_quote_text, triplets_to_text


CONFIDENCE_RE = re.compile(r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)")


class TrainingRegimen(str, Enum):
    """Supported follow-on training regimens."""

    FACTS_WITH_CONFIDENCE = "facts_with_confidence"
    SYLLOGISM_WITH_CONFIDENCE = "syllogism_with_confidence"


def extract_triplet_confidence(triplet: str) -> Optional[float]:
    """Extract a numeric confidence score from a serialized triplet."""
    match = CONFIDENCE_RE.search(triplet)
    if not match:
        return None
    return float(match.group(1))


def normalize_record(record: dict) -> dict:
    """Normalize a source record into the canonical structured schema."""
    if "quote" in record and "entailed_premises" in record:
        return {
            "quote": record.get("quote", ""),
            "entailed_premises": record.get("entailed_premises"),
            "non_entailed_premises": record.get("non_entailed_premises"),
            "syllogism": record.get("syllogism"),
        }

    if "input_text" in record and "output_text" in record:
        return preprocess_training_record(record)

    raise ValueError("Unsupported record shape")


def serialize_facts_with_confidence_record(structured_record: dict) -> dict:
    """Build a quote -> facts-with-confidence training record.

    Preserves the 3-section format from tier3 (Non-Entailed / Entailed /
    Throughline) so tier4 training doesn't force the model to unlearn a new
    section structure.  The only difference from tier3 output is that triplets
    carry (tag, confidence=N.N) annotations.
    """
    quote = normalize_quote_text(structured_record.get("quote", ""))
    non_entailed = structured_record.get("non_entailed_premises")
    entailed = structured_record.get("entailed_premises")
    syllogism = (structured_record.get("syllogism") or "").strip() or "N/A"

    input_text = f"""Given this quote, extract the implicit reasoning facts.

Quote: "{quote}"

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

Response:"""

    output_text = "\n".join(
        [
            "Non-Entailed Premises:",
            triplets_to_text(non_entailed),
            "",
            "Entailed Premises:",
            triplets_to_text(entailed),
            "",
            "Throughline:",
            f"  {syllogism}",
        ]
    )

    return {
        "input_text": input_text,
        "output_text": output_text,
    }


def aggregate_syllogism_confidence(entailed_premises: Optional[list[str]]) -> Optional[float]:
    """Compute a deterministic syllogism confidence from entailed premise scores.

    Normalises each premise before extracting the confidence value so that
    triplets that lack explicit confidence annotations receive the tag-appropriate
    default (observed→1.0, inferred→0.7) rather than contributing None.
    """
    if not entailed_premises:
        return None

    from src.serialize_training_format import normalize_triplet

    normalized = [normalize_triplet(t) for t in entailed_premises]
    confidences = [
        extract_triplet_confidence(n)
        for n in normalized
        if n is not None
    ]
    confidences = [value for value in confidences if value is not None]
    if not confidences:
        return None

    return round(sum(confidences) / len(confidences), 2)


def serialize_syllogism_with_confidence_record(structured_record: dict) -> dict:
    """Build a fact-conditioned syllogism-with-confidence training record."""
    quote = normalize_quote_text(structured_record.get("quote", ""))
    non_entailed = structured_record.get("non_entailed_premises")
    entailed = structured_record.get("entailed_premises")
    syllogism = structured_record.get("syllogism") or "N/A"
    confidence = aggregate_syllogism_confidence(entailed)
    confidence_text = f"{confidence:.2f}" if confidence is not None and syllogism != "N/A" else "N/A"

    input_text = "\n".join(
        [
            f'Given this quote and the extracted facts, score the argument throughline.',
            "",
            f'Quote: "{quote}"',
            "",
            "Non-Entailed Premises:",
            triplets_to_text(non_entailed),
            "",
            "Entailed Premises:",
            triplets_to_text(entailed),
            "",
            "Generate a response with:",
            "1. Throughline",
            "2. Confidence",
            "",
            "Response:",
        ]
    )

    output_text = "\n".join(
        [
            "Throughline:",
            f"  {syllogism}",
            "",
            "Confidence:",
            f"  {confidence_text}",
        ]
    )

    return {
        "input_text": input_text,
        "output_text": output_text,
    }


def serialize_regimen_record(structured_record: dict, regimen: TrainingRegimen) -> dict:
    """Serialize one structured record for the requested regimen."""
    if regimen == TrainingRegimen.FACTS_WITH_CONFIDENCE:
        return serialize_facts_with_confidence_record(structured_record)
    if regimen == TrainingRegimen.SYLLOGISM_WITH_CONFIDENCE:
        return serialize_syllogism_with_confidence_record(structured_record)
    raise ValueError(f"Unsupported regimen: {regimen}")


def _is_valid_for_regimen(structured: dict, regimen: TrainingRegimen) -> bool:
    """Return True iff the structured record is clean enough to train on.

    FACTS_WITH_CONFIDENCE: entailed premises must have at least one valid pipe
    triplet so the model never sees an empty Entailed Premises section.
    SYLLOGISM_WITH_CONFIDENCE: entailed confidence must resolve to a float (no N/A
    throughline or missing premises).
    """
    if regimen == TrainingRegimen.FACTS_WITH_CONFIDENCE:
        entailed = structured.get("entailed_premises")
        return triplets_to_text(entailed) != "N/A"
    if regimen == TrainingRegimen.SYLLOGISM_WITH_CONFIDENCE:
        entailed = structured.get("entailed_premises")
        syllogism = structured.get("syllogism") or "N/A"
        if syllogism == "N/A":
            return False
        return aggregate_syllogism_confidence(entailed) is not None
    return True


def build_training_regimen_dataset(
    input_file: str,
    output_file: str,
    regimen: TrainingRegimen,
    max_records: int = 200,
) -> dict:
    """Convert a JSONL source file into one of the follow-on regimens.

    Scans the full input file but stops after writing max_records clean rows.
    Records that fail quality gates for their regimen are skipped and counted.

    Preconditions:
        input_file is a JSONL file of structured or training-style records.
        max_records >= 1.
    Guarantee:
        output_file contains at most max_records rows, all passing the regimen
        quality gate (non-empty entailed premises for facts_with_confidence,
        non-N/A confidence for syllogism_with_confidence).
    """
    stats = {
        "total": 0,
        "converted": 0,
        "skipped_quality": 0,
        "errors": 0,
        "regimen": regimen.value,
    }

    with open(input_file) as infile, open(output_file, "w") as outfile:
        for line in infile:
            if stats["converted"] >= max_records:
                break
            try:
                record = json.loads(line)
                stats["total"] += 1
                structured = normalize_record(record)
                if not _is_valid_for_regimen(structured, regimen):
                    stats["skipped_quality"] += 1
                    continue
                regimen_record = serialize_regimen_record(structured, regimen)
                outfile.write(json.dumps(regimen_record) + "\n")
                stats["converted"] += 1
            except Exception as exc:
                stats["errors"] += 1
                print(f"Error converting record: {exc}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build follow-on confidence-bearing training regimens"
    )
    parser.add_argument("--input", required=True, help="Input JSONL source")
    parser.add_argument("--output", required=True, help="Output JSONL dataset")
    parser.add_argument(
        "--regimen",
        required=True,
        choices=[regimen.value for regimen in TrainingRegimen],
        help="Which follow-on regimen to build",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=200,
        help="Maximum number of quality-filtered records to write (default: 200)",
    )

    args = parser.parse_args()

    stats = build_training_regimen_dataset(
        args.input,
        args.output,
        TrainingRegimen(args.regimen),
        max_records=args.max_records,
    )

    print("\n=== REGIMEN BUILD STATISTICS ===")
    print(f"Regimen: {stats['regimen']}")
    print(f"Total: {stats['total']}")
    print(f"Converted: {stats['converted']}")
    print(f"Errors: {stats['errors']}")
