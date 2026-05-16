"""Batch-generate verbatim SPO training corpus from the english_quotes dataset.

Uses Qwen/Qwen3.5-0.8B in NF4 (BitsAndBytes) with the verbatim extraction
prompt.  Progress is checkpointed to a sqlite DB so the run can be resumed.

Usage:
    python -m src.generate_verbatim_corpus \
        --quotes-path /home/user/root_cache/.cache/huggingface/hub/datasets--Abirate--english_quotes/snapshots/7b544c4920a8be268b48b403c188acf0a462051b/quotes.jsonl \
        --output data/train_structured_verbatim.jsonl \
        --checkpoint data/gen_verbatim_checkpoint.db \
        --batch-size 8 \
        --max-new-tokens 512

Preconditions:
    bitsandbytes, transformers>=5.0, torch available in mamba-venv.
Failure modes:
    Malformed model outputs are logged and skipped (not written).
    Run can be safely interrupted and resumed via sqlite checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


QUOTES_PATH = (
    "/home/user/root_cache/.cache/huggingface/hub/"
    "datasets--Abirate--english_quotes/snapshots/"
    "7b544c4920a8be268b48b403c188acf0a462051b/quotes.jsonl"
)
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
DEFAULT_OUTPUT = "data/train_structured_verbatim.jsonl"
DEFAULT_CHECKPOINT = "data/gen_verbatim_checkpoint.db"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = """\
EXAMPLES — the predicate MUST be a verb phrase from the quote, never just a tag word:

Quote: "The unexamined life is not worth living."
Non-Entailed Premises:
reflection | leads to (inferred, confidence=0.8) | self-knowledge
Entailed Premises:
The unexamined life | is not worth (observed, confidence=1.0) | living
Throughline: A life without self-examination lacks meaning.

Quote: "Success usually comes to those who are too busy to be looking for it."
Non-Entailed Premises:
obsession with success | prevents (inferred, confidence=0.75) | achievement
Entailed Premises:
Success | usually comes to (observed, confidence=1.0) | those who are too busy to be looking for it
Throughline: Focused action produces success more than deliberate pursuit.

RULES: exactly 2 pipes per triplet (3 fields). Predicate = verb phrase + (tag, confidence=N.N). Never bare (inferred) or (observed) alone as predicate."""


def build_prompt(quote: str) -> str:
    """Return the user-turn text for the verbatim-extraction task.

    Preconditions: quote is a non-empty string.
    Guarantee: returned prompt includes few-shot examples showing verb-phrase predicates.
    Failure modes: empty quote yields a degenerate prompt but will not raise.
    """
    q = quote.strip().strip('"').strip('\u201c').strip('\u201d').strip()
    return (
        f'{_FEW_SHOT_EXAMPLES}\n'
        f'Now extract for this quote:\n\n'
        f'Quote: "{q}"\n\n'
        f'Non-Entailed Premises:\n'
    )


SYSTEM_PROMPT = (
    "You are a reasoning-extraction assistant. "
    "Given a quote, extract implicit premises and a conclusion in structured triplet format. "
    "Each triplet has exactly 3 pipe-separated fields: subject | verb_phrase (tag, confidence=N.N) | object. "
    "The predicate (middle field) MUST be a verb phrase — never just a bare tag word like 'inferred' or 'observed'. "
    "For Entailed Premises, subject, verb phrase, and object must be verbatim words from the quote. "
    "Non-Entailed Premises may use your own words. "
    "Always output exactly three labelled sections."
)


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r'(?:Non-Entailed Premises?|Non Entailed Premises?)[\s:]*\n(.*?)'
    r'(?:Entailed Premises?[\s:]*\n(.*?))'
    r'(?:(?:Throughline|Syllogism)[\s:]*\n(.*))',
    re.DOTALL | re.IGNORECASE,
)

# Exactly 2 pipes → 3 fields (S|P|O). Lines with more pipes are malformed and rejected.
_TRIPLET_LINE_RE = re.compile(r'^[^|]+\|[^|]+\|[^|]+$')


def _parse_section_lines(text: str) -> list[str]:
    """Extract non-empty triplet lines from a section blob."""
    lines = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if line and _TRIPLET_LINE_RE.match(line) and line.lower() not in ("n/a", "none"):
            lines.append(line)
    return lines


def parse_output(quote: str, text: str) -> Optional[dict]:
    """Parse model output into structured dict.

    Returns None when the output cannot be parsed into a valid record.
    """
    m = _SECTION_RE.search(text)
    if not m:
        # Fallback: try to split on section headers manually
        non_ent = _extract_section_fallback(text, "Non-Entailed")
        ent = _extract_section_fallback(text, "Entailed")
        throughline = _extract_throughline(text)
    else:
        non_ent = _parse_section_lines(m.group(1) or "")
        ent = _parse_section_lines(m.group(2) or "")
        throughline = (m.group(3) or "").strip().splitlines()[0].strip() if m.group(3) else ""

    if not ent and not non_ent:
        # Last-resort: no section headers found — collect all valid triplet lines as entailed.
        # This handles models that output triplets verbatim without section scaffolding.
        all_triplets = _parse_section_lines(text)
        if not all_triplets:
            return None
        ent = all_triplets

    return {
        "quote": quote,
        "entailed_premises": ent,
        "non_entailed_premises": non_ent,
        "syllogism": throughline or "",
    }


def _extract_section_fallback(text: str, header_keyword: str) -> list[str]:
    """Grab triplet lines between a header keyword and the next blank/header."""
    lines = text.splitlines()
    capturing = False
    result = []
    for line in lines:
        if re.search(header_keyword, line, re.IGNORECASE) and ":" in line:
            capturing = True
            # In case the first triplet is inline with the header
            inline = line.split(":", 1)[-1].strip()
            if inline and _TRIPLET_LINE_RE.match(inline):
                result.append(inline)
            continue
        if capturing:
            stripped = line.strip()
            if not stripped:
                continue
            # Stop at next section header
            if re.match(r'^(Entailed|Non-Entailed|Throughline|Syllogism)', stripped, re.IGNORECASE) and ":" in stripped:
                break
            if _TRIPLET_LINE_RE.match(stripped):
                result.append(stripped)
    return result


def _extract_throughline(text: str) -> str:
    m = re.search(r'(?:Throughline|Syllogism)[\s:]*\n(.+)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def open_checkpoint(path: str) -> sqlite3.Connection:
    """Open (or create) the sqlite checkpoint database.

    WAL journal mode reduces lock contention; synchronous=NORMAL is safe for
    crash recovery and faster than FULL.
    """
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS done (quote TEXT PRIMARY KEY, result TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def already_done(conn: sqlite3.Connection, quote: str) -> Optional[dict]:
    row = conn.execute("SELECT result FROM done WHERE quote=?", (quote,)).fetchone()
    return json.loads(row[0]) if row else None


def mark_done(conn: sqlite3.Connection, quote: str, record: Optional[dict]) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO done (quote, result) VALUES (?, ?)",
        (quote, json.dumps(record)),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model_nf4(model_name: str):
    """Load model in NF4 (BitsAndBytes 4-bit) for fast CPU-light generation.

    Sets padding_side='left' so batch generation with a causal model is correct.
    """
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
    quotes: list[str],
    max_new_tokens: int = 512,
) -> list[str]:
    """Generate one response per quote.  Returns raw text strings.

    Uses apply_chat_template with a system prompt so the model operates in
    its native instruction-following format.  Any <think>...</think> block is
    stripped before returning.
    """
    prompts = [build_prompt(q) for q in quotes]

    messages_batch = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p},
        ]
        for p in prompts
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
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = enc["input_ids"].shape[1]
    results = []
    for seq in out:
        new_tokens = seq[input_len:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        results.append(raw)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate verbatim SPO corpus")
    parser.add_argument("--quotes-path", default=QUOTES_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="Process only N quotes (0=all)")
    args = parser.parse_args()

    # Load quotes
    quotes_raw = []
    with open(args.quotes_path) as f:
        for line in f:
            d = json.loads(line)
            q = d.get("quote", "").strip()
            if q:
                quotes_raw.append(q)

    if args.limit:
        quotes_raw = quotes_raw[: args.limit]

    print(f"Total quotes: {len(quotes_raw)}")

    # Checkpoint
    conn = open_checkpoint(args.checkpoint)
    done_count = conn.execute("SELECT COUNT(*) FROM done").fetchone()[0]
    print(f"Already done: {done_count}")

    # Remaining
    pending = [q for q in quotes_raw if not already_done(conn, q)]
    print(f"Pending: {len(pending)}")

    if not pending:
        print("Nothing to do — all quotes already processed.")
        conn.close()
        _write_output(conn, quotes_raw, args.output)
        return

    tokenizer, model = load_model_nf4(MODEL_NAME)

    bs = args.batch_size
    skipped = 0
    written = 0

    for batch_start in range(0, len(pending), bs):
        batch = pending[batch_start : batch_start + bs]
        try:
            outputs = generate_batch(tokenizer, model, batch, args.max_new_tokens)
        except Exception as exc:
            print(f"[BATCH ERROR @ {batch_start}] {exc} — marking batch as null and continuing", flush=True)
            for quote in batch:
                mark_done(conn, quote, None)
            skipped += len(batch)
            continue

        for quote, raw_output in zip(batch, outputs):
            record = parse_output(quote, raw_output)
            mark_done(conn, quote, record)
            if record:
                written += 1
            else:
                skipped += 1

        done_total = done_count + batch_start + len(batch)
        pct = 100 * done_total / len(quotes_raw)
        print(
            f"[{done_total}/{len(quotes_raw)} {pct:.1f}%] "
            f"written={written} skipped={skipped}",
            flush=True,
        )

    conn.close()
    print("Generation done. Writing output JSONL…")
    conn2 = open_checkpoint(args.checkpoint)
    _write_output(conn2, quotes_raw, args.output)
    conn2.close()


def _write_output(conn: sqlite3.Connection, quotes: list[str], path: str) -> None:
    """Write all successfully parsed records to JSONL."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for q in quotes:
            row = already_done(conn, q)
            if row:
                f.write(json.dumps(row) + "\n")
                n += 1
    print(f"Wrote {n} records → {out_path}")


if __name__ == "__main__":
    main()
