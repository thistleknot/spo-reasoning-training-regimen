"""Generate holdout inference examples from a trained SPO adapter.

Loads the adapter, runs greedy inference on holdout records from the
training dataset, and writes a readable markdown comparison file.

Usage:
    python -m src.gen_spo_holdout \
        --adapter-path output/spo_verbatim_3ep/adapter \
        --dataset-path data/train_facts_with_confidence_967.jsonl \
        --output output/spo_verbatim_3ep/holdout_examples.md \
        --n 20
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from .chat_format import build_generation_prompt, strip_response_preamble
from .spo_trainer import SPOEvaluator


def load_adapter(adapter_path: str):
    """Load SPO adapter and tokenizer.

    Precondition: adapter_path contains adapter_config.json without
    ``alora_invocation_tokens`` key (strip it first if present).
    """
    path = Path(adapter_path)
    cfg_path = path / "adapter_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if "alora_invocation_tokens" in cfg:
            del cfg["alora_invocation_tokens"]
            cfg_path.write_text(json.dumps(cfg, indent=2))

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 300) -> str:
    """Greedy decode one prompt.

    Precondition: model is on GPU in eval mode with use_cache=True.
    Guarantee: returns stripped decoded string (empty on generation error).
    """
    chat_prompt = build_generation_prompt(tokenizer, prompt)
    inputs = tokenizer(
        chat_prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            no_repeat_ngram_size=6,
        )
    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return strip_response_preamble(decoded)


def load_holdout_records(
    dataset_path: str,
    holdout_fraction: float = 0.1,
    seed: int = 42,
    n: int = 20,
) -> list[dict]:
    """Return up to ``n`` records from the holdout split.

    Uses the same deterministic split as run_spo_training: last
    ``holdout_fraction`` of records after seeded shuffle.
    """
    rng = random.Random(seed)
    records = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    rng.shuffle(records)
    split = int(len(records) * holdout_fraction)
    holdout = records[-split:] if split else records[-10:]
    rng.shuffle(holdout)
    return holdout[:n]


def _extract_quote(input_text: str) -> str:
    """Pull bare quote from an input_text prompt."""
    for prefix in ("Quote:", 'Quote: "', '"'):
        idx = input_text.find(prefix)
        if idx != -1:
            after = input_text[idx + len(prefix):]
            quote = after.split("\n")[0].strip().strip('"').strip("\u201c\u201d")
            if quote:
                return quote
    return input_text[:120]


def write_markdown(examples: list[dict], output_path: Path) -> None:
    """Write holdout examples to a readable markdown file.

    Each example block: input quote, gold output, model output, score.
    """
    lines = [
        "# SPO Holdout Inference Examples",
        "",
        f"**Adapter:** {examples[0].get('adapter_path', 'n/a')}",
        f"**Examples:** {len(examples)}",
        "",
        "---",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        score = ex.get("score", 0.0)
        lines += [
            f"## Example {i} — score: {score:.3f}",
            "",
            f"**Input quote:** {ex['quote']}",
            "",
            "**Gold output:**",
            "```",
            ex.get("gold", "").strip(),
            "```",
            "",
            "**Model output:**",
            "```",
            ex.get("output", "").strip(),
            "```",
            "",
            "---",
            "",
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def run(
    adapter_path: str,
    dataset_path: str,
    output: str,
    n: int = 20,
    max_new_tokens: int = 300,
    holdout_fraction: float = 0.1,
    seed: int = 42,
) -> Path:
    """Generate holdout examples and write markdown.

    Require: adapter_path is a valid PEFT adapter directory.
    Guarantee: writes markdown to output path; returns the path.
    """
    print(f"Loading adapter from {adapter_path} ...")
    model, tokenizer = load_adapter(adapter_path)

    print(f"Loading {n} holdout records from {dataset_path} ...")
    records = load_holdout_records(dataset_path, holdout_fraction, seed, n)

    examples = []
    for idx, record in enumerate(records, 1):
        prompt = record["input_text"]
        gold = record["output_text"]
        quote = _extract_quote(prompt)

        print(f"  [{idx}/{len(records)}] {quote[:60]} ...")
        output_text = generate(model, tokenizer, prompt, max_new_tokens)
        score = SPOEvaluator.evaluate_triplet_correctness(output_text)

        examples.append(
            {
                "adapter_path": adapter_path,
                "quote": quote,
                "gold": gold,
                "output": output_text,
                "score": score,
            }
        )

    output_path = Path(output)
    write_markdown(examples, output_path)
    avg_score = sum(e["score"] for e in examples) / len(examples)
    print(f"Wrote {len(examples)} examples → {output_path}  (avg_score={avg_score:.3f})")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate holdout inference examples from a trained SPO adapter"
    )
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument(
        "--dataset-path", default="data/train_facts_with_confidence_967.jsonl"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        adapter_path=args.adapter_path,
        dataset_path=args.dataset_path,
        output=args.output,
        n=args.n,
        max_new_tokens=args.max_new_tokens,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
