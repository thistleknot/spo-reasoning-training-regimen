"""Batch-generate verbatim SPO training corpus from the english_quotes dataset.

Uses Qwen/Qwen3.5-0.8B in NF4 (BitsAndBytes) with the verbatim extraction
prompt.  Progress is checkpointed to a sqlite DB so the run can be resumed.

Each generated record passes a quality gate before being accepted:
  - at least 1 entailed premise
  - non-empty throughline
  - ALL triplets carry a (tag, confidence=N.N) annotation (100% threshold)
  - NO bare-tag predicates: every predicate must be a verb phrase, never just
    "(inferred)" or "(observed)" with no text before it
  - Completeness: total premise count >= number of sentences in the quote
    so every claim in the quote is represented by at least one premise
On first failure a retry is issued with sampling (temperature=0.7, new seed).
On second failure the quote is skipped — no NULL is written to the checkpoint.
Generation stops once --target-records clean records are produced.

Emojibake (UTF-8 decoded as latin-1, e.g. â€™ for ') is fixed with ftfy
before quotes are passed to the model and before records are stored.

Usage:
    python -m src.generate_verbatim_corpus \
        --quotes-path /home/user/root_cache/.cache/huggingface/hub/datasets--Abirate--english_quotes/snapshots/7b544c4920a8be268b48b403c188acf0a462051b/quotes.jsonl \
        --output data/train_structured_verbatim.jsonl \
        --checkpoint data/gen_verbatim_checkpoint.db \
        --target-records 200 \
        --batch-size 8 \
        --max-new-tokens 512

Preconditions:
    bitsandbytes, transformers>=5.0, torch available in mamba-venv.
Failure modes:
    Records that fail quality gate after two attempts are skipped entirely.
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

import ftfy
import nltk
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
               emojibake is fixed before inserting into the prompt.
    Failure modes: empty quote yields a degenerate prompt but will not raise.
    """
    q = ftfy.fix_text(quote).strip().strip('"').strip('\u201c').strip('\u201d').strip()
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
# Quality gate
# ---------------------------------------------------------------------------

_TAG_CONF_RE = re.compile(
    r"\(\s*(observed|inferred)\s*,\s*confidence\s*=\s*[0-9]", re.IGNORECASE
)
_PIPE_LINE_RE = re.compile(r"^[^|]+\|[^|]+\|[^|]+$")
# Predicate field starts immediately with a tag annotation: no verb phrase before '('
_BARE_PRED_RE = re.compile(r"^\s*\(?\s*(observed|inferred)\b", re.IGNORECASE)


def validate_record(record: Optional[dict]) -> bool:
    """Return True only when the generated record meets the quality bar.

    Requirements:
        - at least 1 entailed premise
        - non-empty throughline (not empty string or bare whitespace)
        - ALL pipe-bearing triplets carry a (tag, confidence=N.N) annotation (100%)
        - NO bare-tag predicates: predicate must be a verb phrase, never just
          "(inferred)" or "(observed)" with no text before the opening paren
        - Completeness: total premise count >= number of sentences in the quote,
          ensuring every claim in a multi-sentence quote is extracted

    The confidence requirement ensures the teacher model produced usable
    annotation signal before we strip confidence for base_reasoning training.
    """
    if not record:
        return False
    entailed = record.get("entailed_premises") or []
    if not entailed:
        return False
    syllogism = (record.get("syllogism") or "").strip()
    if not syllogism:
        return False

    non_entailed = list(record.get("non_entailed_premises") or [])
    all_premises = list(entailed) + non_entailed
    pipe_lines = [p for p in all_premises if isinstance(p, str) and _PIPE_LINE_RE.match(p.strip())]
    if not pipe_lines:
        return False

    # All triplets must carry a (tag, confidence=N.N) annotation
    tagged = sum(1 for p in pipe_lines if _TAG_CONF_RE.search(p))
    if tagged < len(pipe_lines):
        return False

    # Reject bare-tag predicates: predicate field must contain a verb phrase
    for p in pipe_lines:
        parts = [f.strip() for f in p.split("|")]
        if len(parts) == 3 and _BARE_PRED_RE.match(parts[1]):
            return False

    # Completeness: total premises must cover every sentence in the quote
    quote_text = ftfy.fix_text(record.get("quote", ""))
    try:
        n_sents = len(nltk.sent_tokenize(quote_text))
    except Exception:
        n_sents = 1
    if len(all_premises) < n_sents:
        return False

    return True


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
    if not row:
        return None
    parsed = json.loads(row[0])
    # Treat legacy NULL checkpoints as not-done so they get retried
    return parsed if parsed else None


def mark_done(conn: sqlite3.Connection, quote: str, record: dict) -> None:
    """Write a verified clean record to the checkpoint.

    Preconditions: record has passed validate_record() — never call with None.
    """
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
    do_sample: bool = False,
    temperature: float = 0.7,
    seed: int = 42,
) -> list[str]:
    """Generate one response per quote.  Returns raw text strings.

    Uses apply_chat_template with a system prompt so the model operates in
    its native instruction-following format.  Any <think>...</think> block is
    stripped before returning.

    Args:
        do_sample: False = greedy (deterministic, fast); True = sampling (retry path).
        temperature: sampling temperature; only used when do_sample=True.
        seed: torch manual seed for reproducible sampling on retry.
    """
    if do_sample:
        torch.manual_seed(seed)

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

    gen_kwargs: dict = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None)

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)

    input_len = enc["input_ids"].shape[1]
    results = []
    for seq in out:
        new_tokens = seq[input_len:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        results.append(raw)
    return results


def _generate_single(
    tokenizer,
    model,
    quote: str,
    max_new_tokens: int,
    do_sample: bool,
    seed: int,
) -> Optional[dict]:
    """Generate and parse one record.  Returns None on any exception."""
    try:
        outputs = generate_batch(
            tokenizer, model, [quote],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample, seed=seed,
        )
        return parse_output(quote, outputs[0])
    except Exception as exc:
        print(f"  [gen error] {exc}", flush=True)
        return None

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
    parser.add_argument(
        "--target-records", type=int, default=200,
        help="Stop after N clean records are produced (0 = unlimited)",
    )
    args = parser.parse_args()

    # Load quotes — apply emojibake fix at load time
    quotes_raw = []
    with open(args.quotes_path) as f:
        for line in f:
            d = json.loads(line)
            q = ftfy.fix_text(d.get("quote", "").strip())
            if q:
                quotes_raw.append(q)

    print(f"Total quotes available: {len(quotes_raw)}")

    conn = open_checkpoint(args.checkpoint)
    already_written = conn.execute("SELECT COUNT(*) FROM done").fetchone()[0]
    print(f"Already checkpointed: {already_written} clean records")

    target = args.target_records if args.target_records > 0 else len(quotes_raw)
    if already_written >= target:
        print(f"Target {target} already reached — writing output.")
        _write_output(conn, target, args.output)
        conn.close()
        return

    # Pending: quotes not yet in checkpoint
    done_quotes: set[str] = {
        row[0] for row in conn.execute("SELECT quote FROM done").fetchall()
    }
    pending = [q for q in quotes_raw if q not in done_quotes]
    print(f"Pending quotes: {len(pending)}")

    if not pending:
        print("No pending quotes — writing output from checkpoint.")
        _write_output(conn, target, args.output)
        conn.close()
        return

    tokenizer, model = load_model_nf4(MODEL_NAME)

    bs = args.batch_size
    written = already_written
    skipped = 0
    retry_count = 0
    quote_idx = 0

    while written < target and quote_idx < len(pending):
        # Greedy batch pass
        batch = pending[quote_idx : quote_idx + bs]
        quote_idx += len(batch)

        try:
            outputs = generate_batch(
                tokenizer, model, batch, args.max_new_tokens,
                do_sample=False,
            )
        except Exception as exc:
            print(f"[BATCH ERROR] {exc} — skipping batch", flush=True)
            skipped += len(batch)
            continue

        retry_queue: list[str] = []
        for quote, raw_output in zip(batch, outputs):
            record = parse_output(quote, raw_output)
            if validate_record(record):
                mark_done(conn, quote, record)
                written += 1
            else:
                retry_queue.append(quote)

        # Retry failures individually with sampling
        for quote in retry_queue:
            if written >= target:
                break
            record = _generate_single(
                tokenizer, model, quote, args.max_new_tokens,
                do_sample=True, seed=retry_count,
            )
            retry_count += 1
            if validate_record(record):
                mark_done(conn, quote, record)
                written += 1
            else:
                # Two attempts exhausted — skip this quote entirely (no NULL written)
                skipped += 1
                print(f"  [skip] quality gate failed twice for: {quote[:60]!r}", flush=True)

        print(
            f"[progress] written={written}/{target}  skipped={skipped}  "
            f"quotes_seen={quote_idx}/{len(pending)}",
            flush=True,
        )

    conn.close()
    print("Generation done. Writing output JSONL…")
    conn2 = open_checkpoint(args.checkpoint)
    _write_output(conn2, target, args.output)
    conn2.close()


def _write_output(conn: sqlite3.Connection, limit: int, path: str) -> None:
    """Write up to limit verified records from the checkpoint to JSONL."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = conn.execute("SELECT result FROM done LIMIT ?", (limit,)).fetchall()
    n = 0
    with open(out_path, "w") as f:
        for (result_json,) in rows:
            record = json.loads(result_json)
            if record:  # guard against any legacy NULL rows
                f.write(json.dumps(record) + "\n")
                n += 1
    print(f"Wrote {n} records → {out_path}")


if __name__ == "__main__":
    main()
