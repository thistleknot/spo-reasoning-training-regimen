"""Build a curated SFT corpus from best-of-N generated completions.

Reads grpo_generated.jsonl (output of generate_grpo_data.py), which has
per-quote completions with reward scores, and produces a clean SFT JSONL
that keeps only the highest-scoring valid completion(s) per quote.

Filtering rules (non-negotiable schema):
    1. Completion must parse into all three sections:
       Non-Entailed Premises / Entailed Premises / Throughline (Conclusion)
    2. Entailed section must contain >= min_entailed pipe-triplets.
    3. Non-entailed section must contain >= min_non_entailed pipe-triplets.
    4. Throughline must be non-empty (any printable text).
    5. Reward must be > min_reward (default 0.0 — excludes hard zeros).

Selection (when top_k > 1 — greedy diversity):
    Keep top-1 by effective reward, then greedily add completions that are
    structurally distinct (different fingerprint bucket) from already-selected
    ones, up to top_k.  Structural fingerprint = (n_entailed_bucket,
    n_non_entailed_bucket, conclusion_len_bucket, first_entailed_subject).
    This implements DEITA's diversity axis: quality filtered first, then
    diverse patterns preferred over near-duplicate high-reward completions.

Reward augmentation (groundedness bonus):
    Effective reward = raw_reward + diversity_alpha * groundedness_score
    groundedness_score = fraction of quote content-words (4+ chars) that
    appear in the completion's triplet subjects/objects.  Rewards completions
    that extract quote-specific entities rather than generic "speaker | is | X"
    templates.  Only used for ranking/selection; stored reward is raw.

Output schema per row:
    {quote, input_text, output_text, reward, rank, groundedness}
    output_text is the raw model completion (verbatim, already in SFT format).

Usage:
    python build_sft_corpus.py \\
        --input  data/grpo_generated.jsonl \\
        --output data/train_best_of_n.jsonl \\
        [--top-k 3] [--min-reward 0.0] [--diversity-alpha 0.15]
        [--min-entailed 1] [--min-non-entailed 1]
"""

import argparse
import json
import re
from pathlib import Path

from src.serialize_training_format import build_base_reasoning_prompt

# Stop-words excluded from quote-grounded overlap scoring
_STOPWORDS = frozenset(
    "the a an and or but in on at to of for is are was were be been being "
    "have has had do does did will would could should may might this that "
    "with from by it its not no so if as all you i we they he she what when "
    "how who which there here than then just can into up out about".split()
)

# ------------------------------------------------------------------
# Section parser (mirrors frozen_judge._extract_sections)
# ------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"(?:Non-Entailed Premises:|Non-Entailed:)(.*?)"
    r"(?:Entailed Premises:|Entailed:)(.*?)"
    r"(?:Throughline:|Conclusion:)(.*?)$",
    re.DOTALL | re.IGNORECASE,
)
_TRIPLET_RE = re.compile(r"([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)")


def _extract_sections(text: str) -> tuple[list[str], list[str], str]:
    """Return (non_entailed_triplets, entailed_triplets, conclusion_text)."""
    match = _SECTION_RE.search(text)
    if not match:
        return [], [], ""

    def _triplets(block: str) -> list[str]:
        return [
            line.strip()
            for line in block.splitlines()
            if _TRIPLET_RE.search(line)
        ]

    non_entailed = _triplets(match.group(1))
    entailed = _triplets(match.group(2))
    conclusion = match.group(3).strip()
    return non_entailed, entailed, conclusion


# ------------------------------------------------------------------
# Groundedness scoring (Orca insight: ground extraction in source text)
# ------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _content_words(text: str) -> set[str]:
    """Lowercase content words (4+ chars, not stopwords) from text."""
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if len(t) >= 4 and t.lower() not in _STOPWORDS
    }


def score_groundedness(quote: str, non_ent: list[str], ent: list[str], conclusion: str) -> float:
    """Fraction of quote content-words appearing in triplet subjects/objects.

    Rewards completions that extract quote-specific entities rather than
    generic templates ("speaker | is | person").  Range [0.0, 1.0].

    Require: quote is non-empty; triplet lists may be empty.
    Guarantee: returns 0.0 when quote has no content words or no triplets.
    """
    quote_words = _content_words(quote)
    if not quote_words:
        return 0.0

    # Collect all subject and object text from triplets
    triplet_text = " ".join(non_ent + ent + [conclusion])
    triplet_words = _content_words(triplet_text)

    overlap = quote_words & triplet_words
    return len(overlap) / len(quote_words)


# ------------------------------------------------------------------
# Structural fingerprint for diversity-aware selection (DEITA diversity axis)
# ------------------------------------------------------------------

def _structural_fingerprint(
    non_ent: list[str], ent: list[str], conclusion: str
) -> tuple:
    """Return a bucket tuple used to detect near-duplicate structures.

    Two completions with the same fingerprint are considered structurally
    redundant — the greedy diversity selector will prefer distinct fingerprints
    when filling top_k slots beyond the first.

    Buckets:
        n_entailed_bucket:     0=1, 1=2, 2=3+
        n_non_entailed_bucket: 0=1, 1=2, 2=3+
        conclusion_len_bucket: 0=short(<10w), 1=medium(10-20w), 2=long(20+w)
        first_ent_subject:     lowercased first word of first entailed triplet
                               subject, or "" if none.
    """
    def _bucket(n: int) -> int:
        if n <= 1:
            return 0
        if n == 2:
            return 1
        return 2

    conclusion_words = len(conclusion.split())
    if conclusion_words < 10:
        conc_bucket = 0
    elif conclusion_words <= 20:
        conc_bucket = 1
    else:
        conc_bucket = 2

    first_subj = ""
    if ent:
        parts = ent[0].split("|")
        if parts:
            first_subj = parts[0].strip().lower().split()[0] if parts[0].strip() else ""

    return (_bucket(len(ent)), _bucket(len(non_ent)), conc_bucket, first_subj)


def greedy_diverse_select(
    candidates: list[tuple[float, float, str]],
    top_k: int,
) -> list[tuple[float, float, str]]:
    """Select up to top_k completions with greedy diversity.

    Args:
        candidates: list of (effective_reward, raw_reward, completion) sorted
                    descending by effective_reward.
        top_k: maximum completions to select.

    Returns a list of up to top_k (effective_reward, raw_reward, completion)
    tuples.  The first slot is always filled by the top candidate.
    Subsequent slots prefer candidates with fingerprints not yet seen.
    Fallback: if all remaining candidates share seen fingerprints, pick the
    next-best by effective reward (no slot is wasted).
    """
    if not candidates:
        return []

    selected: list[tuple[float, float, str]] = []
    seen_fingerprints: set[tuple] = set()

    # Pre-compute fingerprints
    fingered = []
    for eff_rew, raw_rew, comp in candidates:
        ne, ent, conc = _extract_sections(comp)
        fp = _structural_fingerprint(ne, ent, conc)
        fingered.append((eff_rew, raw_rew, comp, fp))

    remaining = list(fingered)

    while remaining and len(selected) < top_k:
        # Prefer first candidate with unseen fingerprint; fallback to best remaining
        chosen_idx = None
        for i, (eff, raw, comp, fp) in enumerate(remaining):
            if fp not in seen_fingerprints:
                chosen_idx = i
                break
        if chosen_idx is None:
            chosen_idx = 0  # all fingerprints seen, just take best remaining

        eff, raw, comp, fp = remaining.pop(chosen_idx)
        selected.append((eff, raw, comp))
        seen_fingerprints.add(fp)

    return selected


# ------------------------------------------------------------------
# Schema validation
# ------------------------------------------------------------------

def validate_completion(
    completion: str,
    min_entailed: int = 1,
    min_non_entailed: int = 1,
) -> bool:
    """Non-negotiable structural gate.

    Require: completion is a non-empty string.
    Guarantee: True only when all three sections are present and populated.
    """
    if not completion or not completion.strip():
        return False
    non_ent, ent, conclusion = _extract_sections(completion)
    if len(ent) < min_entailed:
        return False
    if len(non_ent) < min_non_entailed:
        return False
    if not conclusion:
        return False
    return True


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def build_sft_corpus(
    input_path: Path,
    output_path: Path,
    top_k: int = 3,
    min_reward: float = 0.0,
    min_entailed: int = 1,
    min_non_entailed: int = 1,
    diversity_alpha: float = 0.15,
) -> dict:
    """Read generated JSONL, filter, select best-of-N, write SFT JSONL.

    Selection uses greedy diversity when top_k > 1: the first slot goes to
    the top effective-reward candidate; subsequent slots prefer candidates
    with unseen structural fingerprints (DEITA diversity axis).

    Effective reward = raw_reward + diversity_alpha * groundedness_score,
    where groundedness_score is the fraction of quote content-words found
    in the completion's triplet text.  Only used for ranking; stored reward
    is the raw score.

    Returns a stats dict for reporting.
    """
    n_quotes_in = 0
    n_completions_in = 0
    n_failed_schema = 0
    n_failed_reward = 0
    n_quotes_out = 0
    n_rows_written = 0
    total_groundedness = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            quote = row.get("quote", "")
            completions: list[str] = row.get("completions", [])
            rewards: list[float] = row.get("rewards", [])

            if not quote or not completions:
                continue

            n_quotes_in += 1

            # Pair completions with rewards; pad rewards with 0.0 if mismatched
            if len(rewards) < len(completions):
                rewards = list(rewards) + [0.0] * (len(completions) - len(rewards))

            n_completions_in += len(completions)

            # Apply hard filters and compute effective reward
            candidates: list[tuple[float, float, str]] = []  # (eff_rew, raw_rew, comp)
            for comp, raw_rew in zip(completions, rewards):
                if raw_rew <= min_reward:
                    n_failed_reward += 1
                    continue
                if not validate_completion(comp, min_entailed, min_non_entailed):
                    n_failed_schema += 1
                    continue
                ne, ent, conc = _extract_sections(comp)
                gnd = score_groundedness(quote, ne, ent, conc)
                eff_rew = raw_rew + diversity_alpha * gnd
                candidates.append((eff_rew, raw_rew, comp))

            if not candidates:
                continue

            # Sort descending by effective reward before diversity selection
            candidates.sort(key=lambda x: x[0], reverse=True)
            selected = greedy_diverse_select(candidates, top_k)

            n_quotes_out += 1
            input_text = build_base_reasoning_prompt(quote)

            for rank, (eff_rew, raw_rew, comp) in enumerate(selected, start=1):
                ne, ent, conc = _extract_sections(comp)
                gnd = score_groundedness(quote, ne, ent, conc)
                total_groundedness += gnd
                out_row = {
                    "quote": quote,
                    "input_text": input_text,
                    "output_text": comp,
                    "reward": round(raw_rew, 6),
                    "rank": rank,
                    "groundedness": round(gnd, 4),
                }
                fout.write(json.dumps(out_row) + "\n")
                n_rows_written += 1

    return {
        "quotes_in": n_quotes_in,
        "completions_in": n_completions_in,
        "failed_schema": n_failed_schema,
        "failed_reward": n_failed_reward,
        "quotes_out": n_quotes_out,
        "rows_written": n_rows_written,
        "coverage_pct": round(100 * n_quotes_out / max(n_quotes_in, 1), 1),
        "avg_groundedness": round(
            total_groundedness / max(n_rows_written, 1), 4
        ),
    }


def main():
    repo = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(repo / "data" / "grpo_generated.jsonl"),
        help="grpo_generated.jsonl from generate_grpo_data.py",
    )
    parser.add_argument(
        "--output",
        default=str(repo / "data" / "train_best_of_n.jsonl"),
        help="Output SFT JSONL path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Best-of-N: keep top K completions per quote with diversity (default 3)",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=0.0,
        help="Hard minimum reward threshold (exclusive, default 0.0 drops zeros)",
    )
    parser.add_argument(
        "--diversity-alpha",
        type=float,
        default=0.15,
        help="Weight for groundedness bonus in effective reward (default 0.15)",
    )
    parser.add_argument(
        "--min-entailed",
        type=int,
        default=1,
        help="Min entailed triplets required (default 1)",
    )
    parser.add_argument(
        "--min-non-entailed",
        type=int,
        default=1,
        help="Min non-entailed triplets required (default 1)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}")
        raise SystemExit(1)

    print(f"Reading: {input_path}")
    print(f"  top_k={args.top_k}  diversity_alpha={args.diversity_alpha}  "
          f"min_reward={args.min_reward}")
    stats = build_sft_corpus(
        input_path,
        output_path,
        top_k=args.top_k,
        min_reward=args.min_reward,
        min_entailed=args.min_entailed,
        min_non_entailed=args.min_non_entailed,
        diversity_alpha=args.diversity_alpha,
    )

    print(f"\nCorpus build complete → {output_path}")
    print(f"  Quotes in:          {stats['quotes_in']}")
    print(f"  Completions in:     {stats['completions_in']}")
    print(f"  Failed schema:      {stats['failed_schema']}")
    print(f"  Failed min_reward:  {stats['failed_reward']}")
    print(f"  Quotes surviving:   {stats['quotes_out']} ({stats['coverage_pct']}%)")
    print(f"  Rows written:       {stats['rows_written']}")
    print(f"  Avg groundedness:   {stats['avg_groundedness']:.4f}")

    if stats['coverage_pct'] < 50:
        print("\nWARNING: < 50% quote coverage. Consider lowering --min-reward or "
              "increasing --group-size in generate_grpo_data.py.")


if __name__ == "__main__":
    main()
