"""
Bootstrap verbatim predicates into the training corpus.

Takes structured records (S | tag | O) and injects a verbatim predicate by
locating the span between subject and object in the source quote.

Result: S | verbatim_predicate (tag, confidence=X) | O

Only Entailed Premises get verbatim predicate injection — the verbatim rule
applies there.  Non-Entailed Premises are left as-is (scorer does not require
verbatim predicates for non-entailed).

Preconditions:
    Input JSONL has records with: quote, entailed_premises, non_entailed_premises, syllogism.
    Each premise string is pipe-delimited with 2-3 fields.

Guarantee:
    Output JSONL preserves all input fields; entailed_premises are rewritten where
    a verbatim predicate can be extracted.  Records where no extraction succeeds are
    written unchanged (the serializer downstream handles bare-tag predicates).

Failure modes:
    Records with malformed structure are written unchanged.
    Extraction returns None if subject or object not found in quote, or the
    span between them is < MIN_PREDICATE_LEN or > MAX_PREDICATE_LEN characters.
"""

import json
import re
import sys
import argparse
from pathlib import Path

MIN_PREDICATE_LEN = 2
MAX_PREDICATE_LEN = 40

# Copula/auxiliary verbs used as fallback predicate when span extraction fails.
# Ordered longest-first so "is not" matches before "is".
_COPULA_RE = re.compile(
    r'\b(is not|are not|was not|were not|has not|have not|had not'
    r'|is|are|was|were|has|have|had|do not|does not|did not|do|does|did'
    r'|can|could|should|would|will|may|might|must|shall)\b',
    re.IGNORECASE,
)

# Pattern to detect bare tag: middle field has ONLY a tag word (no predicate text)
_BARE_TAG_FIELD_RE = re.compile(r'^\s*(observed|inferred)\s*$', re.IGNORECASE)
# Already has canonical annotation (predicate text present before or after tag annotation)
_CANONICAL_ANNOT_RE = re.compile(
    r'\(\s*(observed|inferred)\s*,\s*confidence\s*=\s*[0-9.]+\s*\)',
    re.IGNORECASE,
)
# Strip leading/trailing punctuation/whitespace from extracted predicate span
_STRIP_BOUNDS_RE = re.compile(r'^[\s,;:.!?\"\'"\u201c\u201d\u2018\u2019]+|[\s,;:.!?\"\'"\u201c\u201d\u2018\u2019]+$')
# Strip surrounding quotes/asterisks/bullets from S/O fields before quote search
_FIELD_STRIP_RE = re.compile(r'^[\s*""\u201c\u201d\'\']+|[\s*""\u201c\u201d\'\']+$')


def _normalize_field(text: str) -> str:
    """Strip bullets, asterisks, and smart-quotes from a subject/object field."""
    return _FIELD_STRIP_RE.sub('', text).strip()


def _copula_fallback(subject: str, quote: str) -> str | None:
    """Find the first copula verb in the quote sentence containing the subject.

    Preconditions: subject is a normalized (stripped) field value.
    Guarantee: returns the matched copula text verbatim from quote, or None.
    Failure modes: returns None if subject not found in quote or no copula present.
    """
    q_lower = quote.lower()
    s = subject.lower()
    s_pos = q_lower.find(s)
    if s_pos == -1:
        return None
    # Search for copula starting from the end of the subject
    after_s = s_pos + len(s)
    m = _COPULA_RE.search(quote, after_s)
    if m and m.start() - after_s <= 5:  # copula must be within 5 chars of subject end
        return m.group(0)
    return None


def _extract_predicate_from_quote(subject: str, obj: str, quote: str) -> str | None:
    """Extract verbatim predicate text between subject and object in the quote.

    Tries three strategies in order:
    1. Forward span: text between end-of-subject and start-of-object.
    2. Reversed span: when object appears before subject (text between end-of-object
       and start-of-subject).
    3. Copula fallback: first copula verb appearing after the subject in the quote.

    Preconditions:
        subject and object are verbatim spans from quote (case-insensitive).
        Leading/trailing punctuation is stripped from subject/object before search.
    Guarantee:
        Returns verbatim text from the original quote that can serve as a predicate.
        Returns None if all strategies fail or the extracted span is unusable.
    Failure modes:
        Returns None when subject == object (tautological triplet), when neither
        appears in the quote, or when the extracted text is purely punctuation.
    """
    q = str(quote)
    q_lower = q.lower()
    s = _normalize_field(subject).lower()
    o = _normalize_field(obj).lower()
    if not s or not o:
        return None
    # Tautological triplet — no predicate can be meaningfully extracted
    if s == o:
        return None

    # Strategy 1: forward span (S … O order in quote)
    s_pos = q_lower.find(s)
    if s_pos != -1:
        after_s = s_pos + len(s)
        o_pos = q_lower.find(o, after_s)
        if o_pos != -1:
            between = _STRIP_BOUNDS_RE.sub('', q[after_s:o_pos])
            if MIN_PREDICATE_LEN <= len(between) <= MAX_PREDICATE_LEN and re.search(r'[a-zA-Z]', between):
                return between

    # Strategy 2: reversed span (O … S order in quote)
    o_pos_rev = q_lower.find(o)
    if o_pos_rev != -1:
        after_o = o_pos_rev + len(o)
        s_pos_rev = q_lower.find(s, after_o)
        if s_pos_rev != -1:
            between = _STRIP_BOUNDS_RE.sub('', q[after_o:s_pos_rev])
            if MIN_PREDICATE_LEN <= len(between) <= MAX_PREDICATE_LEN and re.search(r'[a-zA-Z]', between):
                return between

    # Strategy 3: copula fallback (find first verb after subject in quote)
    if s_pos != -1:
        return _copula_fallback(s, q)

    return None


def _inject_predicate_into_triplet(triplet: str, quote: str) -> str | None:
    """Inject a verbatim predicate into a bare-tag triplet.

    Returns None for tautological triplets (subject == object) so callers can
    filter them out.  Returns the triplet unchanged if extraction fails.

    Input:  'room without books | inferred | body without a soul'
    Output: 'room without books | is like a inferred | body without a soul'
    (Downstream normalize_triplet() will canonicalize the relation field.)
    """
    parts = [p.strip() for p in triplet.split('|')]
    if len(parts) != 3:
        return triplet

    subject, relation, obj = parts

    # Filter tautological triplets before any further work
    if _normalize_field(subject).lower() == _normalize_field(obj).lower():
        return None  # discard

    # Skip if relation already has a canonical annotation (predicate present)
    if _CANONICAL_ANNOT_RE.search(relation):
        return triplet

    # Only inject if relation is a bare tag word
    if not _BARE_TAG_FIELD_RE.match(relation):
        return triplet

    predicate = _extract_predicate_from_quote(subject, obj, quote)
    if predicate is None:
        return triplet

    # Prepend predicate to the existing bare-tag relation field.
    # normalize_triplet() downstream handles: "is like a inferred" → "is like a (inferred, confidence=0.7)"
    new_relation = f"{predicate} {relation.strip()}"
    return f"{subject} | {new_relation} | {obj}"


def inject_predicates(record: dict) -> dict:
    """Rewrite entailed_premises in a structured record to include verbatim predicates.

    Preconditions:
        record has 'quote', 'entailed_premises' keys.
    Guarantee:
        Returns a new dict; entailed_premises entries are rewritten where possible.
        Tautological triplets (subject == object) are dropped.
        Non-entailed premises are left unchanged.
    """
    quote = str(record.get('quote', ''))
    entailed = record.get('entailed_premises', [])
    new_entailed = []
    for t in entailed:
        if not isinstance(t, str):
            continue
        result = _inject_predicate_into_triplet(t, quote)
        if result is not None:  # None means tautological — drop it
            new_entailed.append(result)
    return {**record, 'entailed_premises': new_entailed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input', default='data/train_structured_verbatim.jsonl',
        help='Structured JSONL with quote + entailed/non-entailed premises'
    )
    parser.add_argument(
        '--output', default='data/train_structured_verbatim_v11.jsonl',
        help='Output JSONL with verbatim predicates injected'
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    total = 0
    modified = 0
    predicates_added = 0

    with open(in_path) as fin, open(out_path, 'w') as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            new_record = inject_predicates(record)
            total += 1

            orig_entailed = record.get('entailed_premises', [])
            new_entailed = new_record.get('entailed_premises', [])
            added = sum(1 for a, b in zip(orig_entailed, new_entailed) if a != b)
            predicates_added += added
            if added > 0:
                modified += 1

            fout.write(json.dumps(new_record) + '\n')

    print(f"Total records   : {total}")
    print(f"Records modified: {modified} ({100*modified/max(1,total):.0f}%)")
    print(f"Predicates added: {predicates_added}")
    print(f"Written         : {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
