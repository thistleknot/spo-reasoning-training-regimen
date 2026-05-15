"""Full span-scoring pipeline combining faithfulness, BERT coverage, and LLM judge.

Purpose
-------
Orchestrates three complementary quality signals for extractive span outputs:

1. **Faithfulness** (``faithfulness_checker``) – token overlap and monotonic order
   checks that require no external model.
2. **BERT semantic coverage** (``sentence_transformers``, lazy import) – cosine
   similarity between the source embedding and the concatenated span embedding,
   measuring how well the extracted spans cover the source semantics.
3. **LLM judge** (caller-supplied callable) – prompted evaluation that identifies
   semantically missed facts and produces a 0-10 coverage score.

Preconditions
-------------
- ``source`` is a non-empty plain-text string.
- ``spans`` is a list of ``(start, end)`` character spans into *source*.
- ``llm_fn``, when provided, must be a ``str → str`` callable that accepts a
  plain-text prompt and returns a plain-text response.

Failure modes
-------------
- If ``sentence_transformers`` is not importable, ``bert_coverage_score`` returns
  the sentinel value ``-1.0`` (not a real score); callers and ``score_spans``
  detect this and renormalise weights accordingly.
- If the LLM response cannot be parsed, ``llm_judge_score`` returns a neutral
  score of 0.5 with an empty missed-items list and the raw response for debugging.
- ``score_spans`` never raises; internal errors in sub-calls propagate their own
  sentinel/fallback values through the composite.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .faithfulness_checker import check_faithfulness
from .span_extractor import spans_to_surface


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SpanScoreResult:
    """Holds the full scoring breakdown for one set of extractive spans."""

    spans: list[tuple[int, int]]
    faithfulness: dict                  # from faithfulness_checker.check_faithfulness
    bert_coverage: float                # cosine similarity in [0, 1], or -1.0 if unavailable
    llm_judge_score: float              # 0-1 normalised, or 0.5 if llm_fn not provided
    missed_items: list[str]             # items the LLM judge flagged as missed
    composite_score: float              # weighted blend of the three signals


# ---------------------------------------------------------------------------
# BERT coverage
# ---------------------------------------------------------------------------

def bert_coverage_score(
    source: str,
    spans: list[tuple[int, int]],
    model_name: str = "all-MiniLM-L6-v2",
) -> float:
    """Compute cosine similarity between source embedding and concatenated span text.

    Uses ``sentence_transformers.SentenceTransformer`` with a lazy import so the
    module can be imported even when the library is absent.

    Returns
    -------
    Float in ``[0, 1]``, or ``-1.0`` when ``sentence_transformers`` is not installed
    (sentinel indicating "unavailable", not a quality failure).
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
    except ImportError:
        return -1.0

    span_texts = [source[s:e] for s, e in spans if 0 <= s < e <= len(source)]
    if not span_texts:
        return 0.0

    combined = " ".join(span_texts)

    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode([source, combined], convert_to_numpy=True)
    except Exception:
        return -1.0

    src_emb, span_emb = embeddings[0], embeddings[1]
    dot = float(sum(float(a) * float(b) for a, b in zip(src_emb, span_emb)))
    norm_src = float(sum(float(x) ** 2 for x in src_emb) ** 0.5)
    norm_span = float(sum(float(x) ** 2 for x in span_emb) ** 0.5)
    denom = norm_src * norm_span
    if denom < 1e-10:
        return 0.0
    sim = dot / denom
    return float(max(0.0, min(1.0, sim)))


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def llm_judge_score(
    source: str,
    spans: list[tuple[int, int]],
    llm_fn: Callable[[str], str],
) -> dict:
    """Prompt an LLM judge to evaluate span coverage and identify missed facts.

    Parameters
    ----------
    source:
        The original reference text.
    spans:
        Extractive character spans into *source*.
    llm_fn:
        A ``str → str`` callable.  The prompt is passed in; the raw text response
        is expected back.

    Returns
    -------
    Dict with keys:

    ``score``
        Float in ``[0, 1]`` (the raw 0-10 rating divided by 10).
    ``missed_items``
        List of strings describing facts the judge flagged as not covered.
    ``raw_response``
        The full, unmodified LLM response for debugging.

    On any parse failure the dict contains ``score=0.5`` and ``missed_items=[]``.
    """
    span_texts = [source[s:e] for s, e in spans if 0 <= s < e <= len(source)]
    spans_display = "\n".join(
        f"  [{i + 1}] \"{t}\"" for i, t in enumerate(span_texts)
    ) if span_texts else "  (no spans extracted)"

    prompt = (
        "You are a rigorous fact-coverage evaluator.\n\n"
        "## Original text\n"
        f'"{source}"\n\n'
        "## Extracted spans\n"
        f"{spans_display}\n\n"
        "## Your task\n"
        "1. Rate how completely the extracted spans cover all semantically important "
        "facts in the original text.  Give a score from 0 to 10 in the format "
        "\"Score: X/10\".\n"
        "2. List any semantically important facts from the original text that are "
        "NOT covered by the extracted spans.  Use a bullet list prefixed with "
        "\"Missed:\" on its own line, one item per line starting with \"- \".  "
        "If nothing is missed write \"Missed: none\".\n"
    )

    response: str = llm_fn(prompt)

    # --- Parse score ---
    score_match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", response)
    if score_match:
        raw_score = float(score_match.group(1))
        score = float(max(0.0, min(1.0, raw_score / 10.0)))
    else:
        # Fallback: look for a bare integer 0-10
        bare_match = re.search(r"\bscore[:\s]+(\d+)\b", response, re.IGNORECASE)
        if bare_match:
            raw_score = float(bare_match.group(1))
            score = float(max(0.0, min(1.0, raw_score / 10.0)))
        else:
            return {"score": 0.5, "missed_items": [], "raw_response": response}

    # --- Parse missed items ---
    missed_items: list[str] = []
    missed_section = re.search(
        r"Missed\s*:\s*(.*?)(?:\n\n|\Z)", response, re.DOTALL | re.IGNORECASE
    )
    if missed_section:
        block = missed_section.group(1).strip()
        if block.lower() not in ("none", ""):
            for line in block.splitlines():
                item = re.sub(r"^[\-\*\d.)\s]+", "", line).strip()
                if item:
                    missed_items.append(item)

    return {
        "score": score,
        "missed_items": missed_items,
        "raw_response": response,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def score_spans(
    source: str,
    spans: list[tuple[int, int]],
    llm_fn: Optional[Callable[[str], str]] = None,
) -> SpanScoreResult:
    """Orchestrate faithfulness, BERT coverage, and optional LLM judge scoring.

    Weight schedule
    ---------------
    Normal (all three signals available):
        ``composite = 0.4 * faith_ratio + 0.3 * bert + 0.3 * llm``
    BERT unavailable (``bert_coverage == -1.0``):
        Rescale to ``0.5 * faith_ratio + 0.5 * llm``
    LLM not provided (``llm_fn is None``):
        Renormalise to ``(0.4/0.7) * faith_ratio + (0.3/0.7) * bert``
    Both BERT unavailable AND no LLM:
        ``composite = faith_ratio``

    Parameters
    ----------
    source:
        The original reference text.
    spans:
        Extractive character spans into *source*.
    llm_fn:
        Optional ``str → str`` callable for the LLM judge step.

    Returns
    -------
    :class:`SpanScoreResult` containing all sub-scores and the composite blend.
    """
    # 1. Faithfulness (always available)
    faithfulness = check_faithfulness(source, spans)
    faith_ratio = faithfulness["token_overlap_ratio"]

    # 2. BERT coverage (lazy, may return -1.0 sentinel)
    bert_cov = bert_coverage_score(source, spans)

    # 3. LLM judge (only when callable provided)
    judge_score_val: float
    missed_items: list[str]
    if llm_fn is not None:
        judge_result = llm_judge_score(source, spans, llm_fn)
        judge_score_val = judge_result["score"]
        missed_items = judge_result["missed_items"]
    else:
        judge_score_val = 0.5  # neutral sentinel — not used in blend
        missed_items = []

    # 4. Composite blend
    bert_available = bert_cov >= 0.0
    llm_available = llm_fn is not None

    if bert_available and llm_available:
        composite = 0.4 * faith_ratio + 0.3 * bert_cov + 0.3 * judge_score_val
    elif bert_available and not llm_available:
        # Renormalise weights 0.4 + 0.3 → 4/7, 3/7
        composite = (4.0 / 7.0) * faith_ratio + (3.0 / 7.0) * bert_cov
    elif not bert_available and llm_available:
        composite = 0.5 * faith_ratio + 0.5 * judge_score_val
    else:
        # Only faithfulness available
        composite = faith_ratio

    return SpanScoreResult(
        spans=spans,
        faithfulness=faithfulness,
        bert_coverage=bert_cov,
        llm_judge_score=judge_score_val,
        missed_items=missed_items,
        composite_score=round(float(composite), 6),
    )
