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

# Pattern to detect bare tag: middle field has ONLY a tag word (no predicate text)
_BARE_TAG_FIELD_RE = re.compile(r'^\s*(observed|inferred)\s*$', re.IGNORECASE)
# Already has canonical annotation (predicate text present before or after tag annotation)
_CANONICAL_ANNOT_RE = re.compile(
    r'\(\s*(observed|inferred)\s*,\s*confidence\s*=\s*[0-9.]+\s*\)',
    re.IGNORECASE,
)
# Strip leading/trailing punctuation/whitespace from extracted predicate span
_STRIP_BOUNDS_RE = re.compile(r'^[\s,;:.!?\"\'""'']+|[\s,;:.!?\"\'""'']+$')


def _extract_predicate_from_quote(subject: str, obj: str, quote: str) -> str | None:
    """Extract verbatim predicate text between subject and object in the quote.

    Preconditions:
        subject and object are verbatim spans from quote (case-insensitive).
    Guarantee:
        Returns the verbatim text (from original quote) between the end of
        subject and the start of object.  Returns None if either span is
        not found, order is reversed, or the extracted text fails length gates.
    Failure modes:
        Returns None if subject/object overlap, span is empty, or is purely
        punctuation/whitespace.
    """
    q = str(quote)
    q_lower = q.lower()
    s = subject.lower().strip()
    o = obj.lower().strip()
    if not s or not o:
        return None

    s_pos = q_lower.find(s)
    if s_pos == -1:
        return None
    after_s = s_pos + len(s)

    o_pos = q_lower.find(o, after_s)
    if o_pos == -1:
        return None

    # verbatim span from the original cased quote
    between = q[after_s:o_pos]
    between = _STRIP_BOUNDS_RE.sub('', between)

    if len(between) < MIN_PREDICATE_LEN or len(between) > MAX_PREDICATE_LEN:
        return None

    # Must contain at least one letter
    if not re.search(r'[a-zA-Z]', between):
        return None

    return between


def _inject_predicate_into_triplet(triplet: str, quote: str) -> str:
    """Inject a verbatim predicate into a bare-tag triplet.

    If the middle field already has a predicate (not a bare tag), returns
    the triplet unchanged.  If extraction fails, returns unchanged.

    Input:  'room without books | inferred | body without a soul'
    Output: 'room without books | is like a inferred | body without a soul'
    (Downstream normalize_triplet() will canonicalize the relation field.)
    """
    parts = [p.strip() for p in triplet.split('|')]
    if len(parts) != 3:
        return triplet

    subject, relation, obj = parts

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
        Non-entailed premises are left unchanged.
    """
    quote = str(record.get('quote', ''))
    entailed = record.get('entailed_premises', [])
    new_entailed = [
        _inject_predicate_into_triplet(t, quote) if isinstance(t, str) else t
        for t in entailed
    ]
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
