"""
Rebuild canonical training corpora from recoverable upstream artifacts.

This module repairs a broken state where:
1. the base training serializer wrote `input_text == output_text`
2. the canonical 967-row corpora lost throughline text

It merges confidence-bearing facts from a structured source with conclusion text
from a conclusion-bearing legacy backup, then rewrites the canonical datasets.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .build_training_regimens import (
    TrainingRegimen,
    build_training_regimen_dataset,
)
from .preprocess_training_data import preprocess_training_record
from .serialize_training_format import convert_preprocessed_to_training


@dataclass
class RebuildStats:
    """Statistics from rebuilding canonical corpora."""

    total_confidence_rows: int
    matched_conclusions: int
    missing_conclusions: int


def _normalize_quote(text: str) -> str:
    return text.strip().strip('"').strip("“").strip("”").strip()


def load_conclusions_by_quote(path: Path) -> Dict[str, str]:
    """Load a quote -> throughline mapping from a legacy conclusion-bearing JSONL."""
    conclusions = {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            structured = preprocess_training_record(record)
            quote = _normalize_quote(structured.get("quote", ""))
            syllogism = structured.get("syllogism")
            if quote and syllogism:
                conclusions[quote] = syllogism
    return conclusions


def rebuild_structured_dataset(
    confidence_source: Path,
    conclusion_source: Path,
    output_path: Path,
) -> RebuildStats:
    """Merge confidence-bearing facts with conclusion-bearing backup rows."""
    conclusions = load_conclusions_by_quote(conclusion_source)
    total = 0
    matched = 0
    missing = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with confidence_source.open() as infile, output_path.open("w") as outfile:
        for line in infile:
            record = json.loads(line)
            total += 1
            quote = _normalize_quote(record.get("quote", ""))
            syllogism = conclusions.get(quote)
            if syllogism:
                matched += 1
            else:
                missing += 1

            rebuilt = {
                "quote": record.get("quote", ""),
                "entailed_premises": record.get("entailed_premises"),
                "non_entailed_premises": record.get("non_entailed_premises"),
                "syllogism": syllogism,
            }
            outfile.write(json.dumps(rebuilt) + "\n")

    return RebuildStats(
        total_confidence_rows=total,
        matched_conclusions=matched,
        missing_conclusions=missing,
    )


def rebuild_canonical_corpora(
    confidence_source: Path,
    conclusion_source: Path,
    structured_output: Path,
    base_output: Path,
    facts_output: Path,
    syllogism_output: Path,
) -> RebuildStats:
    """Rebuild the canonical structured, base, and confidence-bearing corpora."""
    stats = rebuild_structured_dataset(
        confidence_source=confidence_source,
        conclusion_source=conclusion_source,
        output_path=structured_output,
    )

    if stats.matched_conclusions != stats.total_confidence_rows:
        raise ValueError(
            "Recovered conclusions do not cover every confidence-bearing row: "
            f"{stats.matched_conclusions}/{stats.total_confidence_rows}"
        )

    convert_preprocessed_to_training(
        input_file=str(structured_output),
        output_file=str(base_output),
        include_confidence=False,
    )
    build_training_regimen_dataset(
        input_file=str(structured_output),
        output_file=str(facts_output),
        regimen=TrainingRegimen.FACTS_WITH_CONFIDENCE,
    )
    build_training_regimen_dataset(
        input_file=str(structured_output),
        output_file=str(syllogism_output),
        regimen=TrainingRegimen.SYLLOGISM_WITH_CONFIDENCE,
    )
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebuild canonical training corpora from recoverable upstream artifacts"
    )
    parser.add_argument(
        "--confidence-source",
        required=True,
        help="Structured JSONL with confidence-bearing premises",
    )
    parser.add_argument(
        "--conclusion-source",
        required=True,
        help="Legacy JSONL with conclusion-bearing output_text",
    )
    parser.add_argument(
        "--structured-output",
        default="data/train_structured_967.jsonl",
        help="Merged structured output JSONL",
    )
    parser.add_argument(
        "--base-output",
        default="data/train_clean_for_model_967.jsonl",
        help="Canonical base training dataset",
    )
    parser.add_argument(
        "--facts-output",
        default="data/train_facts_with_confidence_967.jsonl",
        help="Canonical facts-with-confidence dataset",
    )
    parser.add_argument(
        "--syllogism-output",
        default="data/train_syllogism_with_confidence_967.jsonl",
        help="Canonical syllogism-with-confidence dataset",
    )
    args = parser.parse_args()

    stats = rebuild_canonical_corpora(
        confidence_source=Path(args.confidence_source),
        conclusion_source=Path(args.conclusion_source),
        structured_output=Path(args.structured_output),
        base_output=Path(args.base_output),
        facts_output=Path(args.facts_output),
        syllogism_output=Path(args.syllogism_output),
    )

    print(json.dumps(stats.__dict__, indent=2))
