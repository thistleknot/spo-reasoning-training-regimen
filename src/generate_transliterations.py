"""Generate transliteration triplets for each verbatim entailed premise in the corpus.

For each verbatim entailed triplet (e.g., "The unexamined life | is not worth
(observed, confidence=1.0) | living"), generates a plain-English paraphrase
triplet in the same S|P(tag,conf)|O format, wrapped in parentheses for the
training target:
    (A life without self-reflection | has no (inferred, confidence=0.9) | value)

Reads:  data/train_structured_verbatim_v12.jsonl (or --input)
Writes: data/train_structured_verbatim_v13.jsonl (or --output)

Progress is checkpointed to sqlite so the run can be safely interrupted and
resumed.

Usage:
    python -m src.generate_transliterations \\
        --input  data/train_structured_verbatim_v12.jsonl \\
        --output data/train_structured_verbatim_v13.jsonl \\
        --checkpoint data/gen_translit_checkpoint.db \\
        --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
DEFAULT_INPUT = "data/train_structured_verbatim_v12.jsonl"
DEFAULT_OUTPUT = "data/train_structured_verbatim_v13.jsonl"
DEFAULT_CHECKPOINT = "data/gen_translit_checkpoint.db"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_FEW_SHOT = """\
Given a verbatim triplet extracted from a quote, write a transliteration — a \
plain-English paraphrase triplet using the same S | P (tag, confidence=N.N) | O format.

Verbatim:        The unexamined life | is not worth (observed, confidence=1.0) | living
Transliteration: A life without self-reflection | has no (inferred, confidence=0.9) | value

Verbatim:        Injustice anywhere | is (observed, confidence=1.0) | a threat to justice everywhere
Transliteration: Local injustice | spreads to (inferred, confidence=0.9) | all of justice

Verbatim:        Success | usually comes to (observed, confidence=1.0) | those who are too busy to be looking for it
Transliteration: Success | finds (inferred, confidence=0.9) | those who focus on work rather than seeking it

Verbatim:        Courage | is (observed, confidence=1.0) | resistance to fear
Transliteration: Courage | means (inferred, confidence=0.9) | facing rather than eliminating fear
"""

_SYSTEM = (
    "You are a triplet-paraphrase assistant. "
    "Given a verbatim triplet in S | P (tag, confidence=N.N) | O format, "
    "rephrase it as a plain-English triplet with the same structure and semantics. "
    "Keep the same tag (observed/inferred) and a similar confidence value. "
    "Output ONLY the transliteration triplet — no labels, no extra lines."
)

# Matches a valid S|P(tag,conf)|O triplet line (with or without surrounding parens)
_TRIPLET_RE = re.compile(r"[^|]+\|[^|]+\|[^|]+")


def build_translit_prompt(verbatim_triplet: str) -> str:
    """Build the user-turn prompt for a single verbatim triplet."""
    return f"{_FEW_SHOT}\nVerbatim:        {verbatim_triplet.strip()}\nTransliteration:"


def _clean_translit_output(raw: str) -> Optional[str]:
    """Extract and validate the transliteration triplet from model output.

    Strips <think> blocks, takes the first line that looks like a triplet.
    Returns None when no valid triplet can be found.
    """
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    for line in text.splitlines():
        line = line.strip().strip("()")
        if _TRIPLET_RE.search(line) and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3 and all(parts):
                return f"({' | '.join(parts)})"
    return None


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def open_checkpoint(path: str) -> sqlite3.Connection:
    """Open (or create) the sqlite checkpoint for resumable generation.

    Stores one row per verbatim triplet: the key is the triplet text itself.
    """
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS done "
        "(triplet TEXT PRIMARY KEY, result TEXT)"
    )
    conn.commit()
    return conn


def cached(conn: sqlite3.Connection, triplet: str) -> Optional[str]:
    row = conn.execute("SELECT result FROM done WHERE triplet=?", (triplet,)).fetchone()
    return row[0] if row else None


def save(conn: sqlite3.Connection, triplet: str, result: Optional[str]) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO done (triplet, result) VALUES (?, ?)",
        (triplet, result),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    """Load model in NF4 for efficient generation."""
    print(f"Loading {model_name} in NF4…")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_batch(
    tokenizer,
    model,
    triplets: list[str],
    max_new_tokens: int = 128,
) -> list[Optional[str]]:
    """Generate one transliteration per verbatim triplet.

    Returns a list parallel to `triplets`; None entries mark generation failures.
    """
    messages_batch = [
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": build_translit_prompt(t)},
        ]
        for t in triplets
    ]
    input_texts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        for msgs in messages_batch
    ]

    enc = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    results: list[Optional[str]] = []
    for i, ids in enumerate(out):
        prompt_len = enc["input_ids"].shape[1]
        new_ids = ids[prompt_len:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True)
        results.append(_clean_translit_output(raw))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    records = [json.loads(l) for l in Path(args.input).open()]
    conn = open_checkpoint(args.checkpoint)

    # Collect all unique verbatim entailed triplets needing generation
    all_triplets: list[str] = []
    seen: set[str] = set()
    for r in records:
        for t in r.get("entailed_premises") or []:
            if t and t not in seen and cached(conn, t) is None:
                all_triplets.append(t)
                seen.add(t)

    if all_triplets:
        tokenizer, model = load_model(args.model)
        batch_size = args.batch_size
        for start in range(0, len(all_triplets), batch_size):
            batch = all_triplets[start:start + batch_size]
            results = generate_batch(tokenizer, model, batch, args.max_new_tokens)
            for triplet, result in zip(batch, results):
                save(conn, triplet, result)
            done = start + len(batch)
            print(f"  Generated {done}/{len(all_triplets)} transliterations", flush=True)
    else:
        print("All transliterations already cached.")

    # Write enriched output records
    out_path = Path(args.output)
    written = 0
    with out_path.open("w") as fh:
        for r in records:
            entailed = r.get("entailed_premises") or []
            translits = [cached(conn, t) for t in entailed]
            r["entailed_transliterations"] = translits
            fh.write(json.dumps(r) + "\n")
            written += 1

    print(f"Wrote {written} records to {out_path}")


if __name__ == "__main__":
    main()
