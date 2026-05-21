"""Export and merge quote corpora into a single training JSONL.

Sources merged (in priority order):
1. data/gen_verbatim_checkpoint.db — 2507 quotes with one structured response each
2. Any additional --extra-jsonl files

Deduplication: normalised quote text (strip outer whitespace + quotes).
Output: data/train_full_corpus.jsonl — {quote, entailed_premises,
        non_entailed_premises, syllogism} one row per unique quote.

Usage:
    python prep_full_corpus.py [--output data/train_full_corpus.jsonl]
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path


def _norm(q: str) -> str:
    return q.strip().strip('"').strip("'").strip()


def load_verbatim_db(db_path: Path) -> list[dict]:
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT result FROM done").fetchall()
    con.close()
    records = []
    for (r,) in rows:
        try:
            records.append(json.loads(r))
        except json.JSONDecodeError:
            pass
    return records


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def merge_corpora(
    verbatim_db: Path,
    extra_jsonls: list[Path],
) -> list[dict]:
    """Merge and deduplicate records. verbatim_db rows take priority."""
    seen: dict[str, dict] = {}

    def add(rec: dict):
        if not rec:
            return
        q = rec.get("quote", "")
        if not q or not q.strip():
            return
        key = _norm(q)
        if key not in seen:
            seen[key] = rec

    for rec in load_verbatim_db(verbatim_db):
        add(rec)

    for path in extra_jsonls:
        for rec in load_jsonl(path):
            add(rec)

    return list(seen.values())


def main():
    repo = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbatim-db",
        default=str(repo / "data" / "gen_verbatim_checkpoint.db"),
        help="Path to gen_verbatim_checkpoint.db",
    )
    parser.add_argument(
        "--extra-jsonl",
        nargs="*",
        default=[],
        help="Additional JSONL files to merge (e.g. train_structured_967.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=str(repo / "data" / "train_full_corpus.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    verbatim_db = Path(args.verbatim_db)
    extra_jsonls = [Path(p) for p in (args.extra_jsonl or [])]
    output_path = Path(args.output)

    if not verbatim_db.exists():
        print(f"ERROR: verbatim DB not found: {verbatim_db}")
        raise SystemExit(1)

    records = merge_corpora(verbatim_db, extra_jsonls)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(records)} unique quotes → {output_path}")

    # Quick stats
    has_entailed = sum(1 for r in records if r.get("entailed_premises"))
    has_non_entailed = sum(1 for r in records if r.get("non_entailed_premises"))
    has_syllogism = sum(1 for r in records if r.get("syllogism"))
    print(
        f"  entailed_premises present: {has_entailed}/{len(records)}"
        f"  | non_entailed: {has_non_entailed}/{len(records)}"
        f"  | syllogism: {has_syllogism}/{len(records)}"
    )


if __name__ == "__main__":
    main()
