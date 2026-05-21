"""Build a curated SFT corpus from best-of-N generated completions.

Reads grpo_generated.jsonl (output of generate_grpo_data.py), which has
per-quote completions with reward scores, and produces a clean SFT JSONL
that keeps only the highest-scoring valid completion per quote.

Filtering rules (non-negotiable schema):
    1. Completion must parse into all three sections:
       Non-Entailed Premises / Entailed Premises / Throughline (Conclusion)
    2. Entailed section must contain >= min_entailed pipe-triplets.
    3. Non-entailed section must contain >= min_non_entailed pipe-triplets.
    4. Throughline must be non-empty (any printable text).
    5. Reward must be > min_reward (default 0.0 — excludes hard zeros).

Selection:
    Per quote, keep top top_k completions by reward (default 1 = best only).

Output schema per row:
    {quote, input_text, output_text, reward, rank}
    output_text is the raw model completion (verbatim, already in SFT format).

Usage:
    python build_sft_corpus.py \\
        --input  data/grpo_generated.jsonl \\
        --output data/train_best_of_n.jsonl \\
        [--top-k 1] [--min-reward 0.0]
        [--min-entailed 1] [--min-non-entailed 1]
"""

import argparse
import json
import re
from pathlib import Path

from src.serialize_training_format import build_base_reasoning_prompt

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
    top_k: int = 1,
    min_reward: float = 0.0,
    min_entailed: int = 1,
    min_non_entailed: int = 1,
) -> dict:
    """Read generated JSONL, filter, select best-of-N, write SFT JSONL.

    Returns a stats dict for reporting.
    """
    n_quotes_in = 0
    n_completions_in = 0
    n_failed_schema = 0
    n_failed_reward = 0
    n_quotes_out = 0
    n_rows_written = 0

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

            # Apply filters
            candidates = []
            for comp, rew in zip(completions, rewards):
                if rew <= min_reward:
                    n_failed_reward += 1
                    continue
                if not validate_completion(comp, min_entailed, min_non_entailed):
                    n_failed_schema += 1
                    continue
                candidates.append((rew, comp))

            if not candidates:
                continue

            # Sort descending by reward, keep top_k
            candidates.sort(key=lambda x: x[0], reverse=True)
            selected = candidates[:top_k]

            n_quotes_out += 1
            input_text = build_base_reasoning_prompt(quote)

            for rank, (rew, comp) in enumerate(selected, start=1):
                out_row = {
                    "quote": quote,
                    "input_text": input_text,
                    "output_text": comp,
                    "reward": round(rew, 6),
                    "rank": rank,
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
        default=1,
        help="Best-of-N: keep top K completions per quote (default 1)",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=0.0,
        help="Hard minimum reward threshold (exclusive, default 0.0 drops zeros)",
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
    stats = build_sft_corpus(
        input_path,
        output_path,
        top_k=args.top_k,
        min_reward=args.min_reward,
        min_entailed=args.min_entailed,
        min_non_entailed=args.min_non_entailed,
    )

    print(f"\nCorpus build complete → {output_path}")
    print(f"  Quotes in:        {stats['quotes_in']}")
    print(f"  Completions in:   {stats['completions_in']}")
    print(f"  Failed schema:    {stats['failed_schema']}")
    print(f"  Failed min_reward:{stats['failed_reward']}")
    print(f"  Quotes surviving: {stats['quotes_out']} ({stats['coverage_pct']}%)")
    print(f"  Rows written:     {stats['rows_written']}")

    if stats['coverage_pct'] < 50:
        print("\nWARNING: < 50% quote coverage. Consider lowering --min-reward or "
              "increasing --group-size in generate_grpo_data.py.")


if __name__ == "__main__":
    main()
