"""Extractive span locator for SPO triplet components.

Purpose
-------
Given a source text (the original quote) and a list of triplet dicts produced by
an LLM, locate each triplet component (subject, predicate, object_) as a
``(start, end)`` character-span into the source string.  The spans are *extractive*:
they point at verbatim substrings so downstream faithfulness checks can verify the
LLM did not paraphrase.

Preconditions
-------------
- ``source`` must be a non-empty plain-text string.
- Each triplet dict must have string values under the keys ``subject``,
  ``predicate``, and ``object_``.  Missing or empty components are silently skipped.

Failure modes
-------------
- ``find_span`` returns ``None`` when fuzzy coverage of the fragment in the source
  is below 0.5 (i.e. fewer than half the fragment characters can be matched).
- ``extract_spans`` returns an empty list when no component spans exceed the
  coverage threshold.
- No exceptions are raised for empty inputs; callers receive empty / ``None``
  results and should handle them accordingly.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def find_span(source: str, fragment: str) -> tuple[int, int] | None:
    """Return the tightest ``(start, end)`` char span in *source* that best covers *fragment*.

    Strategy (in order):
    1. Exact case-insensitive substring match – ``O(n)`` and always preferred.
    2. ``difflib.SequenceMatcher`` fuzzy match at character level – returns the
       union of all matching blocks projected onto *source*.
    3. Word-level token alignment fallback – useful when individual words are
       present but not consecutively.

    Returns ``None`` when the best achievable coverage is below 0.5.
    """
    if not source or not fragment or not fragment.strip():
        return None

    frag_stripped = fragment.strip()
    source_lower = source.lower()
    frag_lower = frag_stripped.lower()

    # --- 1. Exact substring match (case-insensitive) ---
    idx = source_lower.find(frag_lower)
    if idx != -1:
        return (idx, idx + len(frag_lower))

    # --- 2. SequenceMatcher fuzzy matching at char level ---
    sm = SequenceMatcher(None, frag_lower, source_lower, autojunk=False)
    blocks = [(i, j, n) for i, j, n in sm.get_matching_blocks() if n > 0]

    if blocks:
        covered = sum(n for _, _, n in blocks)
        coverage = covered / len(frag_lower)
        if coverage >= 0.5:
            span_start = min(j for _, j, _ in blocks)
            span_end = max(j + n for _, j, n in blocks)
            return (span_start, span_end)

    # --- 3. Word-level token alignment fallback ---
    return _word_level_span(source, frag_stripped)


def _word_level_span(source: str, fragment: str) -> tuple[int, int] | None:
    """Fallback: locate fragment words inside source and return their bounding span."""
    frag_tokens = set(re.split(r'\s+', fragment.lower().strip()))
    frag_tokens.discard("")

    if not frag_tokens:
        return None

    # Build list of (start, end, word) for each whitespace-delimited token in source
    source_matches = [
        (m.start(), m.end(), m.group().lower())
        for m in re.finditer(r'\S+', source)
    ]

    matched: list[tuple[int, int]] = [
        (start, end)
        for start, end, word in source_matches
        if word in frag_tokens
    ]

    if not matched:
        return None

    # Coverage: fraction of distinct fragment tokens found in source
    found_words = {
        word
        for _, _, word in source_matches
        if word in frag_tokens
    }
    coverage = len(found_words) / len(frag_tokens)
    if coverage < 0.5:
        return None

    return (min(s for s, _ in matched), max(e for _, e in matched))


def extract_spans(source: str, triplets: list[dict]) -> list[tuple[int, int]]:
    """Locate every triplet component in *source* and return deduplicated sorted spans.

    Parameters
    ----------
    source:
        The original quote / reference text.
    triplets:
        List of dicts each containing ``"subject"``, ``"predicate"``, and
        ``"object_"`` string values.  Unknown keys are silently skipped.

    Returns
    -------
    Sorted, deduplicated list of ``(start, end)`` character spans (spans may overlap).
    """
    seen: set[tuple[int, int]] = set()
    spans: list[tuple[int, int]] = []

    for triplet in triplets:
        for key in ("subject", "predicate", "object_"):
            component = triplet.get(key, "")
            if not component:
                continue
            span = find_span(source, str(component))
            if span is not None and span not in seen:
                seen.add(span)
                spans.append(span)

    spans.sort()
    return spans


def spans_to_surface(source: str, spans: list[tuple[int, int]]) -> str:
    """Reconstruct the surface form of extracted spans for display.

    Each span is extracted from *source* as a verbatim substring; spans are
    joined with a single space.  Spans that fall outside the bounds of *source*
    are silently ignored.
    """
    parts: list[str] = []
    for start, end in spans:
        if 0 <= start < end <= len(source):
            text = source[start:end].strip()
            if text:
                parts.append(text)
    return " ".join(parts)
