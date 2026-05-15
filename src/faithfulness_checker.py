"""Faithfulness checker for extractive span representations.

Purpose
-------
Given a list of ``(start, end)`` character spans extracted from a source text,
compute how faithfully those spans represent the source.  Three complementary
signals are computed:

1. **token_overlap_ratio** – fraction of whitespace tokens in the span text that
   appear verbatim in the source token vocabulary.  Penalises paraphrase or
   hallucinated content.
2. **order_preserved** – True when span start positions are monotonically
   non-decreasing (i.e. the extraction respects document order).
3. **char_coverage** – fraction of unique character positions in the source that
   are covered by at least one span.

Preconditions
-------------
- ``source`` is a non-empty plain-text string.
- ``spans`` is a list of ``(start, end)`` integer tuples with ``0 <= start < end
  <= len(source)``.  Spans outside this range are silently skipped.

Failure modes
-------------
- Empty ``spans`` list → all numeric outputs are 0.0 and ``is_faithful`` is
  False (order_preserved is True by vacuous truth).
- Empty ``source`` → char_coverage is defined as 0.0 to avoid division by zero.
- No NLTK / spacy dependency; only stdlib ``re`` is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TokenSpan:
    """A character-delimited span annotated with its surface text."""

    start_char: int
    end_char: int
    text: str


def tokenize_simple(text: str) -> list[str]:
    """Whitespace-split, lowercase, strip punctuation tokeniser (no external deps).

    Parameters
    ----------
    text:
        Any plain-text string.

    Returns
    -------
    List of lowercase tokens with leading/trailing punctuation removed.  Empty
    tokens produced by splitting are excluded.
    """
    tokens: list[str] = []
    for raw in text.lower().split():
        # Strip leading and trailing punctuation characters
        token = raw.strip(".,;:!?\"'()[]{}\\/-—_")
        if token:
            tokens.append(token)
    return tokens


def check_faithfulness(source: str, spans: list[tuple[int, int]]) -> dict:
    """Compute faithfulness metrics for a set of extractive spans.

    Parameters
    ----------
    source:
        The reference text the spans were extracted from.
    spans:
        Sorted list of ``(start, end)`` character spans into *source*.

    Returns
    -------
    Dict with keys:

    ``token_overlap_ratio``
        Fraction of span tokens (with repetition) that appear verbatim in the
        source token set.  Range [0, 1].
    ``order_preserved``
        True if span start positions are monotonically non-decreasing.
    ``char_coverage``
        Fraction of unique source character positions covered by at least one span.
        Range [0, 1].
    ``is_faithful``
        True when token_overlap_ratio >= 0.80 AND order_preserved is True.
    """
    if not spans:
        return {
            "token_overlap_ratio": 0.0,
            "order_preserved": True,
            "char_coverage": 0.0,
            "is_faithful": False,
        }

    # Clamp spans to valid source range
    valid_spans = [
        (s, e) for s, e in spans if 0 <= s < e <= len(source)
    ]

    # --- token_overlap_ratio ---
    source_token_set: set[str] = set(tokenize_simple(source))
    span_tokens: list[str] = []
    for s, e in valid_spans:
        span_tokens.extend(tokenize_simple(source[s:e]))

    if span_tokens:
        matching = sum(1 for t in span_tokens if t in source_token_set)
        token_overlap_ratio = matching / len(span_tokens)
    else:
        token_overlap_ratio = 0.0

    # --- order_preserved ---
    # Spans are expected to be already sorted, but we check the raw input order.
    order_preserved = all(
        spans[i][0] <= spans[i + 1][0] for i in range(len(spans) - 1)
    )

    # --- char_coverage ---
    covered_chars: set[int] = set()
    for s, e in valid_spans:
        covered_chars.update(range(s, e))
    char_coverage = len(covered_chars) / max(len(source), 1)

    is_faithful = token_overlap_ratio >= 0.80 and order_preserved

    return {
        "token_overlap_ratio": round(token_overlap_ratio, 6),
        "order_preserved": order_preserved,
        "char_coverage": round(char_coverage, 6),
        "is_faithful": is_faithful,
    }


def majority_token_check(generated: str, source: str) -> dict:
    """Check whether a free-form generated string has majority tokens from source.

    Uses a token-level LCS (longest common subsequence) limited to the first 512
    tokens of each sequence for speed.

    Parameters
    ----------
    generated:
        The free-form output string (e.g. a paraphrased triplet or model output).
    source:
        The original reference text.

    Returns
    -------
    Dict with keys:

    ``lcs_ratio``
        ``lcs_length / max(len(gen_tokens), 1)`` — proportion of generated tokens
        explained by the source.
    ``passes``
        True when ``lcs_ratio >= 0.6``.
    """
    TOKEN_CAP = 512

    gen_tokens = tokenize_simple(generated)[:TOKEN_CAP]
    src_tokens = tokenize_simple(source)[:TOKEN_CAP]

    lcs_len = _lcs_length(gen_tokens, src_tokens)
    lcs_ratio = lcs_len / max(len(gen_tokens), 1)

    return {
        "lcs_ratio": round(lcs_ratio, 6),
        "passes": lcs_ratio >= 0.6,
    }


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Standard O(m·n) DP longest-common-subsequence length, space-optimised to O(n)."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0

    # Two-row rolling DP
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]
