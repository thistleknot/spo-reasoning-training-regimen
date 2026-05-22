"""Generate before/after inference comparisons for SEEING_IS_BELIEVING.md.

Runs the same holdout quotes through the original adapter (before best-of-N
training) and the new adapter (after), then writes a markdown document showing
the actual model outputs side-by-side.

Usage:
    python gen_seeing_is_believing.py \
        --before output/full_run_qwen35_0.8b_3ep/base-plus-facts/adapter \
        --after  output/spo_best_of_n/adapter \
        --n-quotes 15 \
        --output  data/SEEING_IS_BELIEVING_EXAMPLES.md
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from src.chat_format import build_generation_prompt, strip_response_preamble
from src.run_ablation_matrix import split_indices
from src.serialize_training_format import build_base_reasoning_prompt

CORPUS = "data/train_best_of_n.jsonl"
HOLDOUT_FRACTION = 0.1
SEED = 42
MAX_NEW_TOKENS = 384


def load_model(adapter_path: str):
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        is_trainable=False,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def infer(model, tokenizer, quote: str) -> str:
    input_text = build_base_reasoning_prompt(quote)
    prompt = build_generation_prompt(tokenizer, input_text)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=6,
            use_cache=True,
        )
    decoded = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return strip_response_preamble(decoded).strip()


def pick_diverse_holdout(records: list, n: int, seed: int) -> list:
    """Pick n quotes spread across the holdout set by score diversity."""
    rng = random.Random(seed + 1)
    # Sort by reward descending, then sample uniformly across the range
    by_reward = sorted(records, key=lambda r: -r.get("reward", 0.0))
    step = max(1, len(by_reward) // n)
    candidates = [by_reward[i * step] for i in range(min(n, len(by_reward)))]
    rng.shuffle(candidates)
    return candidates[:n]


def fmt_output(text: str) -> str:
    """Indent output lines for markdown code block."""
    return "\n".join("    " + line for line in text.splitlines()) if text else "    (empty)"


def build_markdown(comparisons: list, before_path: str, after_path: str) -> str:
    lines = [
        "# Seeing Is Believing — Before / After Comparisons",
        "",
        "Real model outputs on holdout quotes (not seen during training).",
        "**Before**: original QLoRA adapter trained on the structured 967-quote corpus.",
        "**After**: SPO-fine-tuned adapter trained on 2366-quote best-of-N corpus",
        "  (top-3 greedy-diverse completions per quote, frozen-judge scored).",
        "",
        f"Before adapter: `{before_path}`  ",
        f"After adapter:  `{after_path}`  ",
        f"Holdout fraction: 10% of `data/train_best_of_n.jsonl` (seed=42)",
        "",
        "---",
        "",
    ]

    for i, c in enumerate(comparisons, 1):
        quote = c["quote"]
        reward = c.get("reward", "?")
        before = c["before"]
        after = c["after"]

        lines += [
            f"## Quote {i}",
            "",
            f"> {quote}",
            "",
            f"*(holdout reward: {reward:.3f})*" if isinstance(reward, float) else f"*(holdout reward: {reward})*",
            "",
            "### Before",
            "",
            "```",
            before,
            "```",
            "",
            "### After",
            "",
            "```",
            after,
            "```",
            "",
            "---",
            "",
        ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", default="output/full_run_qwen35_0.8b_3ep/base-plus-facts/adapter")
    parser.add_argument("--after",  default="output/spo_best_of_n/adapter")
    parser.add_argument("--n-quotes", type=int, default=15)
    parser.add_argument("--output",  default="data/SEEING_IS_BELIEVING_EXAMPLES.md")
    parser.add_argument("--corpus",  default=CORPUS)
    args = parser.parse_args()

    repo = Path(__file__).parent
    corpus_path = repo / args.corpus

    print(f"Loading corpus: {corpus_path}")
    records = [json.loads(l) for l in corpus_path.open()]

    _, holdout_idx = split_indices(len(records), HOLDOUT_FRACTION, SEED)
    holdout = [records[i] for i in holdout_idx]
    print(f"Holdout: {len(holdout)} records")

    sampled = pick_diverse_holdout(holdout, args.n_quotes, SEED)
    print(f"Selected {len(sampled)} diverse quotes")

    print(f"\nLoading BEFORE adapter: {args.before}")
    before_model, before_tok = load_model(args.before)

    print("Running inference (before)...")
    before_outputs = []
    for i, rec in enumerate(sampled, 1):
        out = infer(before_model, before_tok, rec["quote"])
        before_outputs.append(out)
        print(f"  [{i}/{len(sampled)}] done")

    del before_model, before_tok
    torch.cuda.empty_cache()

    print(f"\nLoading AFTER adapter: {args.after}")
    after_model, after_tok = load_model(args.after)

    print("Running inference (after)...")
    after_outputs = []
    for i, rec in enumerate(sampled, 1):
        out = infer(after_model, after_tok, rec["quote"])
        after_outputs.append(out)
        print(f"  [{i}/{len(sampled)}] done")

    del after_model, after_tok
    torch.cuda.empty_cache()

    comparisons = [
        {"quote": rec["quote"], "reward": rec.get("reward"), "before": b, "after": a}
        for rec, b, a in zip(sampled, before_outputs, after_outputs)
    ]

    md = build_markdown(comparisons, args.before, args.after)
    out_path = repo / args.output
    out_path.write_text(md)
    print(f"\nWritten: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
