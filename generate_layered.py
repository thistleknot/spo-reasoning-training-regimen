"""Layered batch generation — two-stage offline corpus expansion.

The pipeline runs three pure phases with checkpoints between them, separating
generation from scoring for better GPU utilization:

Stage 1 (Layer 1)
  Batch-generate K completions for ALL N quotes.
  Groups quotes into batches of --batch-size for efficient GPU use.
  Extracts the throughline from each completion.
  Writes: <output-dir>/layer1.jsonl
    {quote, prompt, completions[K], throughlines[K]}

Stage 2 (Layer 2)
  For each (quote, unique_throughline) pair, batch-generate M completions
  whose prompt constrains the conclusion to that specific throughline.
  Groups all pairs into batches of --batch-size.
  Writes: <output-dir>/layer2.jsonl
    {quote, throughline, completions[M]}

Score pass
  Group all K_unique × M completions per quote, score in bulk via
  judge.batch_score_completions, then write the final corpus.
  Writes: <output-path> (compatible with grpo_generated.jsonl format)
    {quote, prompt, completions[K*M], rewards[K*M], mean_reward, max_reward, all_zero}

Advantages over sequential generate_grpo_data.py:
  - Generation and scoring never interleave — no mode-switching overhead.
  - Large batches saturate the GPU during each pure phase.
  - Layer 1 is cheap (short outputs ~256 tokens); batch-size can be 16+.
  - Layer 2 conditions on fixed conclusions → guaranteed structural diversity.
  - Each stage is a resume checkpoint; re-runs skip completed stages.

Usage:
    python generate_layered.py \\
        --adapter-path output/full_run_qwen35_0.8b_3ep/base-plus-facts/adapter \\
        --dataset-path data/train_full_corpus.jsonl \\
        --output-path data/grpo_generated_layered.jsonl \\
        --layer1-completions 3 \\
        --layer2-completions 3 \\
        --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

from src.frozen_judge import FrozenJudge, _extract_sections
from src.run_ablation_matrix import load_jsonl
from src.serialize_training_format import build_base_reasoning_prompt, normalize_quote_text
from src.chat_format import build_generation_prompt, strip_response_preamble


# ---------------------------------------------------------------------------
# VRAM helpers (same logic as generate_grpo_data.py)
# ---------------------------------------------------------------------------

def _free_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, _ = torch.cuda.mem_get_info(0)
    return free_bytes / 1024 / 1024


def _device_map() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return "auto" if _free_vram_mb() >= 450 else "cpu"


def _bnb_config_if_cuda() -> dict:
    if _free_vram_mb() >= 450:
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        }
    return {"torch_dtype": torch.bfloat16}


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def build_conditioned_prompt(quote: str, throughline: str) -> str:
    """Build a Stage-2 prompt that constrains the generated conclusion.

    Identical to build_base_reasoning_prompt except it appends a single
    constraint line before 'Response:' instructing the model to match the
    given throughline.  The model has been trained on the unconstrained format;
    the constraint is a soft instruction-following cue, not a guarantee.

    Preconditions: throughline is a non-empty string from a prior generation.
    """
    normalized_quote = normalize_quote_text(quote)
    return "\n".join(
        [
            "Given this quote, extract the implicit reasoning.",
            "",
            f'Quote: "{normalized_quote}"',
            "",
            "Generate a response with:",
            "1. Non-Entailed Premises",
            "2. Entailed Premises",
            "3. Throughline",
            "",
            "Format each premise as: subject | relation (tag) | object",
            '- tag: "observed" for explicit facts, "inferred" for derived facts',
            "",
            f"The Throughline must be: {throughline}",
            "",
            "Response:",
        ]
    )


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def batch_sample_completions(
    model,
    tokenizer,
    prompts: list[str],
    num_sequences: int,
    max_new_tokens: int,
    max_length: int,
    batch_size: int = 8,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> list[list[str]]:
    """Generate num_sequences completions for each prompt, processed in batches.

    Preconditions:
        model is in eval mode.
        num_sequences >= 1; batch_size >= 1.
    Guarantee:
        Returns a list of len(prompts) sublists, each with num_sequences strings.
        Uses left-padding so generation appends consistently at the end of the
        padded input for all sequences in a batch.
    Failure modes:
        Returns empty sublists for prompts that fail to generate.
    """
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    all_results: list[list[str]] = []

    try:
        for chunk_start in range(0, len(prompts), batch_size):
            chunk = prompts[chunk_start : chunk_start + batch_size]
            B = len(chunk)

            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            padded_len = input_ids.shape[1]

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_sequences,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )

            # outputs: [B * num_sequences, padded_len + generated_len]
            # Order: [p0_s0, p0_s1, ..., p0_s(K-1), p1_s0, ...]
            for i in range(B):
                completions_i: list[str] = []
                for j in range(num_sequences):
                    seq = outputs[i * num_sequences + j]
                    generated_tokens = seq[padded_len:]
                    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    completions_i.append(strip_response_preamble(text))
                all_results.append(completions_i)

            del outputs, input_ids, attention_mask
            torch.cuda.empty_cache()

            print(
                f"  [gen] batch {chunk_start + B}/{len(prompts)} "
                f"({B} prompts × {num_sequences} seqs)"
            )
    finally:
        tokenizer.padding_side = orig_padding_side

    return all_results


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def run_stage1(
    model,
    tokenizer,
    quotes: list[str],
    prompts: list[str],
    k: int,
    max_new_tokens: int,
    max_length: int,
    batch_size: int,
    stage1_path: Path,
) -> list[dict]:
    """Generate K completions for all quotes and extract throughlines.

    Writes layer1.jsonl to stage1_path.  Each row:
        {quote, prompt, completions[K], throughlines[K]}
    throughlines[i] is the throughline extracted from completions[i], or ""
    when the completion is unparseable.

    Returns the list of dicts (same as what was written).
    """
    print(f"\n=== Stage 1: generating {k} completions × {len(quotes)} quotes ===")

    all_completions = batch_sample_completions(
        model, tokenizer, prompts,
        num_sequences=k,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        batch_size=batch_size,
    )

    rows: list[dict] = []
    for quote, prompt, completions in zip(quotes, prompts, all_completions):
        throughlines = [_extract_sections(c)[2] for c in completions]
        rows.append({
            "quote": quote,
            "prompt": prompt,
            "completions": completions,
            "throughlines": throughlines,
        })

    stage1_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stage1_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"Stage 1 done. Written to {stage1_path}")
    return rows


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def run_stage2(
    model,
    tokenizer,
    stage1_rows: list[dict],
    m: int,
    max_new_tokens: int,
    max_length: int,
    batch_size: int,
    stage2_path: Path,
) -> list[dict]:
    """For each (quote, unique_throughline) pair, generate M conditioned completions.

    Deduplicates throughlines within each quote group to avoid redundant pairs.
    Writes layer2.jsonl to stage2_path.  Each row:
        {quote, throughline, prompt_base, completions[M]}

    Returns the list of dicts.
    """
    print(f"\n=== Stage 2: generating {m} conditioned completions per (quote, throughline) ===")

    # Build all (quote, unique_throughline) pairs
    pairs: list[tuple[str, str, str]] = []  # (quote, prompt_base, throughline)
    for row in stage1_rows:
        quote = row["quote"]
        prompt_base = row["prompt"]
        seen: set[str] = set()
        for tl in row["throughlines"]:
            tl_clean = tl.strip()
            if tl_clean and tl_clean not in seen:
                seen.add(tl_clean)
                pairs.append((quote, prompt_base, tl_clean))

    total_pairs = len(pairs)
    print(f"  {total_pairs} unique (quote, throughline) pairs from {len(stage1_rows)} quotes")

    conditioned_prompts: list[str] = []
    for quote, prompt_base, throughline in pairs:
        raw_prompt = build_conditioned_prompt(quote, throughline)
        # Wrap in the same chat template as the base prompt
        cond_prompt = build_generation_prompt(tokenizer, raw_prompt)
        conditioned_prompts.append(cond_prompt)

    all_completions = batch_sample_completions(
        model, tokenizer, conditioned_prompts,
        num_sequences=m,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        batch_size=batch_size,
    )

    rows: list[dict] = []
    for (quote, prompt_base, throughline), completions in zip(pairs, all_completions):
        rows.append({
            "quote": quote,
            "throughline": throughline,
            "prompt_base": prompt_base,
            "completions": completions,
        })

    stage2_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stage2_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"Stage 2 done. Written to {stage2_path}")
    return rows


# ---------------------------------------------------------------------------
# Score pass
# ---------------------------------------------------------------------------

def run_score_pass(
    judge: FrozenJudge,
    stage2_rows: list[dict],
    stage1_rows: list[dict],
    output_path: Path,
) -> None:
    """Group all completions by quote and score in bulk.

    Groups together Stage-2 completions (all K_unique × M per quote) and
    calls judge.batch_score_completions once per quote.  Also merges in any
    Stage-1 completions so the final output includes all generated content.

    Writes one JSON line per quote to output_path — same schema as
    grpo_generated.jsonl, compatible with build_sft_corpus.py.
    """
    print(f"\n=== Score pass: scoring all completions grouped by quote ===")

    # Collect all completions per quote from Stage 2
    from collections import defaultdict
    quote_to_completions: dict[str, list[str]] = defaultdict(list)
    quote_to_prompt: dict[str, str] = {}

    for row in stage2_rows:
        q = row["quote"]
        quote_to_completions[q].extend(row["completions"])
        if q not in quote_to_prompt:
            quote_to_prompt[q] = row["prompt_base"]

    # Also include Stage-1 completions (which weren't conditioned)
    for row in stage1_rows:
        q = row["quote"]
        quote_to_completions[q].extend(row["completions"])
        if q not in quote_to_prompt:
            quote_to_prompt[q] = row["prompt"]

    all_quotes = list(quote_to_completions.keys())
    print(f"  Scoring {len(all_quotes)} quotes, "
          f"{sum(len(v) for v in quote_to_completions.values())} total completions")

    written = 0
    all_zero_count = 0
    with open(output_path, "w") as fh:
        for idx, quote in enumerate(all_quotes, start=1):
            completions = quote_to_completions[quote]
            rewards = judge.batch_score_completions(quote, completions)

            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            max_reward = max(rewards) if rewards else 0.0
            all_zero = max_reward == 0.0
            if all_zero:
                all_zero_count += 1

            record = {
                "quote": quote,
                "prompt": quote_to_prompt.get(quote, ""),
                "completions": completions,
                "rewards": rewards,
                "mean_reward": round(mean_reward, 6),
                "max_reward": round(max_reward, 6),
                "all_zero": all_zero,
            }
            fh.write(json.dumps(record) + "\n")
            written += 1

            if idx % 50 == 0 or idx == len(all_quotes):
                print(
                    f"  [{idx}/{len(all_quotes)}] mean_reward={mean_reward:.4f} "
                    f"all_zero_so_far={all_zero_count}"
                )

    print(f"Score pass done. Written {written} records to {output_path}. "
          f"All-zero fraction: {all_zero_count / max(written, 1):.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    output_path = Path(args.output_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_path = output_dir / "layer1.jsonl"
    stage2_path = output_dir / "layer2.jsonl"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    records = load_jsonl(Path(args.dataset_path))
    if args.max_records:
        records = records[: args.max_records]

    quotes = [
        (rec.get("quote") or rec.get("input_text", "")).strip()
        for rec in records
    ]
    quotes = [q for q in quotes if q]

    # Stage 1 (skip if checkpoint exists)
    if stage1_path.exists() and not args.force_stage1:
        print(f"Stage 1 checkpoint found at {stage1_path} — skipping generation.")
        stage1_rows = [json.loads(l) for l in open(stage1_path) if l.strip()]
    else:
        print(f"Loading model from {args.adapter_path} ...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.adapter_path,
            is_trainable=False,
            device_map=_device_map(),
            **_bnb_config_if_cuda(),
        )
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        prompts = [
            build_generation_prompt(tokenizer, build_base_reasoning_prompt(q))
            for q in quotes
        ]

        stage1_rows = run_stage1(
            model, tokenizer, quotes, prompts,
            k=args.layer1_completions,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            batch_size=args.batch_size,
            stage1_path=stage1_path,
        )

    # Stage 2 (skip if checkpoint exists)
    if stage2_path.exists() and not args.force_stage2:
        print(f"Stage 2 checkpoint found at {stage2_path} — skipping generation.")
        stage2_rows = [json.loads(l) for l in open(stage2_path) if l.strip()]
    else:
        # Reload model if we skipped Stage 1 and didn't load it yet
        if stage1_path.exists() and not args.force_stage1:
            print(f"Loading model from {args.adapter_path} ...")
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.adapter_path,
                is_trainable=False,
                device_map=_device_map(),
                **_bnb_config_if_cuda(),
            )
            tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()

        stage2_rows = run_stage2(
            model, tokenizer, stage1_rows,
            m=args.layer2_completions,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            batch_size=args.batch_size,
            stage2_path=stage2_path,
        )

    # Score pass — always loads judge fresh (model may have been freed)
    print(f"\nLoading judge from {args.adapter_path} ...")
    score_model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_path,
        is_trainable=False,
        device_map=_device_map(),
        **_bnb_config_if_cuda(),
    )
    score_tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    if score_tokenizer.pad_token is None:
        score_tokenizer.pad_token = score_tokenizer.eos_token
    score_model.eval()

    judge = FrozenJudge(adapter_path=args.adapter_path)
    judge.model = score_model
    judge.tokenizer = score_tokenizer

    run_score_pass(judge, stage2_rows, stage1_rows, output_path)

    summary = {
        "adapter_path": args.adapter_path,
        "dataset_path": args.dataset_path,
        "output_path": str(output_path),
        "stage1_path": str(stage1_path),
        "stage2_path": str(stage2_path),
        "layer1_completions": args.layer1_completions,
        "layer2_completions": args.layer2_completions,
        "batch_size": args.batch_size,
        "quotes": len(quotes),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nDone. Summary: {summary_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layered batch generation — two-stage corpus expansion")
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output-path", required=True, help="Final scored JSONL output")
    p.add_argument("--output-dir", default="data/layered_gen",
                   help="Directory for layer1.jsonl, layer2.jsonl checkpoints")
    p.add_argument("--layer1-completions", type=int, default=3,
                   help="K: completions per quote in Stage 1")
    p.add_argument("--layer2-completions", type=int, default=3,
                   help="M: conditioned completions per (quote, throughline) in Stage 2")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Quotes (or pairs) per generation batch")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-stage1", action="store_true",
                   help="Re-run Stage 1 even if layer1.jsonl exists")
    p.add_argument("--force-stage2", action="store_true",
                   help="Re-run Stage 2 even if layer2.jsonl exists")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
