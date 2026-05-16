"""
Adapter: Convert preprocessed structured data to training format.

Takes clean structured dicts (quote, entailed_premises, non_entailed_premises,
syllogism) and serializes them to input_text/output_text format suitable for
training. Annotations are normalized to canonical (tag, confidence=X) format
so the model learns to produce scorer-compatible evidence tags.
"""

import json
import re
from typing import Optional


CONFIDENCE_ANNOTATION_RE = re.compile(
    r"\((observed|inferred)\s*,\s*confidence\s*[:=]\s*[^)]+\)"
)

# Triplet normalization patterns
_BULLET_RE = re.compile(r'^\s*\*+\s*')
_QUOTED_FIELD_RE = re.compile(r'^["\u201c\u201d\u2018\u2019]+(.+?)["\u201c\u201d\u2018\u2019]+$')
_BAR_ANNOT_RE = re.compile(
    r'(observed|inferred)\s*\|+\s*confidence\s*=\s*([0-9]+(?:\.[0-9]+)?)',
    re.IGNORECASE,
)
_BARE_TAG_RE = re.compile(r'\|\s*(observed|inferred)\s*\|', re.IGNORECASE)
_COMBINED_TAG_RE = re.compile(r'observed\s*\|+\s*inferred|inferred\s*\|+\s*observed', re.IGNORECASE)
_TEMPLATE_RE = re.compile(r'<subject>|<relation>|<object>|<[a-z_]+>', re.IGNORECASE)

# Default confidence values by tag
_TAG_DEFAULTS = {"observed": "1.0", "inferred": "0.7"}


def _clean_field(field: str) -> str:
    """Strip surrounding quotes and leading bullet markers from a triplet field."""
    field = field.strip()
    m = _QUOTED_FIELD_RE.match(field)
    if m:
        field = m.group(1).strip()
    return field


def normalize_triplet(triplet: str) -> Optional[str]:
    """Normalize a raw triplet string to canonical format: ``subject | predicate (tag, confidence=X) | object``.

    Preconditions:
        triplet is one line of a premise section.
    Postconditions:
        Returns a normalized string ``subject | predicate (tag, confidence=X) | object``
        where predicate is verbatim text from the quote (Entailed Premises) or synthetic
        (Non-Entailed Premises), tag is exactly 'observed' or 'inferred', and confidence
        is a float in [0, 1].  Returns None if the triplet is malformed/template/degenerate.
    Failure modes:
        Returns None for template placeholders, lines without exactly 3 pipe-delimited
        fields, or lines where subject or object is empty after cleaning.
    """
    line = _BULLET_RE.sub('', triplet).strip()

    if _TEMPLATE_RE.search(line):
        return None

    if line.count('|') < 2:
        return None

    parts = [p.strip() for p in line.split('|', 2)]
    if len(parts) != 3:
        return None

    subject, relation, obj = parts
    subject = _clean_field(subject)
    obj = _clean_field(obj)

    if not subject or not obj:
        return None

    # Case 1: already has canonical (tag, confidence=X) in the relation — keep relation as-is.
    if CONFIDENCE_ANNOTATION_RE.search(relation):
        # Ensure the tag inside is valid (not combined); replace invalid combined tags.
        clean_rel = _COMBINED_TAG_RE.sub('inferred', relation)
        return f"{subject} | {clean_rel.strip()} | {obj}"

    # Case 2: bar format  inferred|confidence=0.8  →  build canonical annotation.
    bar_match = _BAR_ANNOT_RE.search(relation)
    if bar_match:
        raw_tag = bar_match.group(1).lower()
        raw_conf = bar_match.group(2)
        try:
            conf_f = float(raw_conf)
            conf = raw_conf if 0.0 <= conf_f <= 1.0 else _TAG_DEFAULTS[raw_tag]
        except ValueError:
            conf = _TAG_DEFAULTS[raw_tag]
        tag = "observed" if raw_tag == "observed" else "inferred"
        # Strip the bar annotation, append canonical parens annotation.
        semantic = _BAR_ANNOT_RE.sub('', relation).strip().rstrip('|').strip()
        annot = f"({tag}, confidence={conf})"
        relation_clean = f"{semantic} {annot}".strip() if semantic else annot
        return f"{subject} | {relation_clean} | {obj}"

    # Case 3: combined tag  observed|inferred  →  conservative inferred default.
    if _COMBINED_TAG_RE.search(relation):
        semantic = _COMBINED_TAG_RE.sub('', relation).strip()
        annot = "(inferred, confidence=0.7)"
        relation_clean = f"{semantic} {annot}".strip() if semantic else annot
        return f"{subject} | {relation_clean} | {obj}"

    # Case 4: bare tag  | observed |  or  | inferred |  (tag IS the whole relation field).
    bare_match = _BARE_TAG_RE.search(f'|{relation}|')
    if bare_match:
        raw_tag = bare_match.group(1).lower()
        tag = "observed" if raw_tag == "observed" else "inferred"
        annot = f"({tag}, confidence={_TAG_DEFAULTS[tag]})"
        # relation may be just the tag word or may have extra semantic content
        semantic = re.sub(r'\b(observed|inferred)\b', '', relation, flags=re.IGNORECASE).strip()
        relation_clean = f"{semantic} {annot}".strip() if semantic else annot
        return f"{subject} | {relation_clean} | {obj}"

    # Case 5: tag keyword found somewhere in relation (e.g. "implies (inferred)").
    tag_search = re.search(r'\b(observed|inferred)\b', relation, re.IGNORECASE)
    if tag_search:
        raw_tag = tag_search.group(1).lower()
        tag = "observed" if raw_tag == "observed" else "inferred"
        existing_conf = re.search(r'confidence\s*=\s*([0-9.]+)', relation)
        if existing_conf:
            conf = existing_conf.group(1)
            try:
                conf_f = float(conf)
                conf = conf if 0.0 <= conf_f <= 1.0 else _TAG_DEFAULTS[tag]
            except ValueError:
                conf = _TAG_DEFAULTS[tag]
        else:
            conf = _TAG_DEFAULTS[tag]
        # Remove bare tag word, strip old annotation if any, append canonical
        semantic = re.sub(r'\b(observed|inferred)\b', '', relation, flags=re.IGNORECASE).strip()
        semantic = re.sub(r'\(.*?\)', '', semantic).strip()
        annot = f"({tag}, confidence={conf})"
        relation_clean = f"{semantic} {annot}".strip() if semantic else annot
        return f"{subject} | {relation_clean} | {obj}"

    # Case 6: no tag found — semantic predicate only, default to inferred.
    annot = "(inferred, confidence=0.7)"
    relation_clean = f"{relation.strip()} {annot}".strip() if relation.strip() else annot
    return f"{subject} | {relation_clean} | {obj}"


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
            "Given this quote, extract the explicit and implicit reasoning facts.",
            "",
            f'Quote: "{normalized_quote}"',
            "",
            "Generate a response with:",
            "1. Non-Entailed Premises",
            "2. Entailed Premises",
            "3. Throughline",
            "",
            "Format each premise as: subject | predicate (observed, confidence=0.9) | object",
            '- tag: "observed" for explicit facts, "inferred" for derived facts',
            "- confidence: a decimal number in [0,1] — e.g. 1.0 for observed, 0.7 for inferred",
            "",
            "VERBATIM EXTRACTION RULE (Entailed Premises only):",
            "- Subject, predicate, and object MUST be exact, verbatim text copied from the quote above.",
            "- Do NOT paraphrase, summarize, or invent language for Entailed fields.",
            "- Parenthetical clarifications may be added AFTER verbatim text: verbatim text (clarification)",
            "- Invariant: strip all (...) from a triplet and the remaining text must be verbatim from the quote.",
            '  Example: "Don\'t be | satisfied with (inferred, confidence=0.7) | stories"',
            '  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"',
            "- Non-Entailed Premises and the Throughline may use your own words.",
            "",
            "IMPORTANT: The Entailed Premises section MUST contain at least one triplet.",
            "Never leave Entailed Premises empty.",
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
) -> str:
    """Normalize and join triplets to text for training output.

    Each triplet is normalized to canonical (tag, confidence=X) format.
    Invalid/template/degenerate triplets are dropped.  Returns 'N/A' only
    when no valid triplets remain.

    Preconditions:
        triplets is a list of raw triplet strings from the structured corpus.
    Guarantee:
        Returned lines match ``subject | (tag, confidence=X) | object`` format.
    """
    if not triplets:
        return "N/A"

    lines = []
    for triplet in triplets:
        normalized = normalize_triplet(triplet)
        if normalized:
            lines.append(normalized)

    return "\n".join(lines) if lines else "N/A"


def serialize_training_record(structured_record: dict) -> dict:
    """Convert structured record to input_text/output_text format for training.

    Training format uses pedagogical ordering:
    - Non-Entailed Premises FIRST (teaches negative inference)
    - Entailed Premises second (teaches positive inference)
    - Throughline last (the conclusion)

    Triplets are normalized to canonical (tag, confidence=X) annotation format
    so the model learns scorer-compatible output.

    Args:
        structured_record: {quote, entailed_premises, non_entailed_premises, syllogism}

    Returns:
        {input_text, output_text} for trainer
    """
    quote = structured_record.get("quote", "")
    non_entailed = structured_record.get("non_entailed_premises")
    entailed = structured_record.get("entailed_premises")
    throughline = structured_record.get("syllogism")

    input_text = build_base_reasoning_prompt(quote)

    output_lines = [
        "Non-Entailed Premises:",
        triplets_to_text(non_entailed),
        "",
        "Entailed Premises:",
        triplets_to_text(entailed),
        "",
        "Throughline:",
        throughline.strip() if throughline and throughline.strip() else "N/A",
    ]
    output_text = "\n".join(output_lines)

    return {
        "input_text": input_text,
        "output_text": output_text,
    }


def is_tautological(record: dict) -> bool:
    """Return True when entailed_premises == non_entailed_premises or entailed is empty.

    Tautological records provide no distinguishing training signal: the model
    cannot learn when a premise is load-bearing vs. contextual if both sections
    are identical.  Filtering them out forces the model to learn a meaningful
    distinction between the two sections.

    Require: record has optional 'entailed_premises' and 'non_entailed_premises' keys.
    Guarantee: returns bool; never raises.
    """
    e = set(record.get("entailed_premises") or [])
    ne = set(record.get("non_entailed_premises") or [])
    return (not e) or (e == ne)


def is_bad_record(record: dict) -> bool:
    """Return True when a record should be excluded from training.

    Filters:
    - Empty entailed_premises (model would learn to output N/A conclusions)
    - Template placeholders in any triplet (model would learn to echo `<subject>` etc.)
    - Missing or N/A-only throughline (trains the model to output N/A conclusions)
    - Repetitive triplets: any single triplet repeated 3+ times (degenerate data)

    Require: record follows {entailed_premises, non_entailed_premises, syllogism} schema.
    Guarantee: returns bool; never raises.
    """
    entailed = record.get("entailed_premises") or []
    if not entailed:
        return True

    all_trips = list(entailed) + (record.get("non_entailed_premises") or [])

    if any(_TEMPLATE_RE.search(t) for t in all_trips):
        return True

    syl = (record.get("syllogism") or "").strip()
    if not syl or syl.upper() in ("N/A", "NA", ""):
        return True

    counts: dict[str, int] = {}
    for t in all_trips:
        counts[t] = counts.get(t, 0) + 1
    if any(v >= 3 for v in counts.values()):
        return True

    return False


def convert_preprocessed_to_training(
    input_file: str,
    output_file: str,
    filter_tautological: bool = True,
) -> dict:
    """Convert preprocessed structured JSONL to training format JSONL.

    Applies all quality gates: tautological filter, template/N/A/repetitive filter,
    and triplet normalization (canonical annotation format).

    Args:
        input_file: Preprocessed structured JSONL
        output_file: Training format JSONL
        filter_tautological: Skip records where entailed == non_entailed or
            entailed is empty.  Default True.

    Returns:
        Statistics dict with keys: total, converted, skipped_tautological,
        skipped_bad, errors
    """
    stats = {
        "total": 0,
        "converted": 0,
        "skipped_tautological": 0,
        "skipped_bad": 0,
        "errors": 0,
    }

    with open(input_file) as infile, open(output_file, "w") as outfile:
        for line in infile:
            try:
                structured = json.loads(line)
                stats["total"] += 1

                if filter_tautological and is_tautological(structured):
                    stats["skipped_tautological"] += 1
                    continue

                if is_bad_record(structured):
                    stats["skipped_bad"] += 1
                    continue

                training_record = serialize_training_record(structured)

                # Post-normalization: skip if either section ended up empty after normalization
                if "N/A" in training_record["output_text"].split("Entailed Premises:\n")[1][:4]:
                    stats["skipped_bad"] += 1
                    continue

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
        "--no-filter-tautological",
        action="store_true",
        help="Keep records where entailed_premises == non_entailed_premises (default: filter them out)",
    )

    args = parser.parse_args()

    stats = convert_preprocessed_to_training(
        args.input,
        args.output,
        filter_tautological=not args.no_filter_tautological,
    )

    print("\n=== CONVERSION STATISTICS ===")
    print(f"Total: {stats['total']}")
    print(f"Converted: {stats['converted']}")
    print(f"Skipped (tautological): {stats['skipped_tautological']}")
    print(f"Skipped (bad record): {stats['skipped_bad']}")
    print(f"Errors: {stats['errors']}")
