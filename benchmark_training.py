"""
Benchmark offline GRPO training to find the fastest batch_size + dtype combo.

Usage:
    python benchmark_training.py \
        --adapter-path output/full_run_qwen35_0.8b_3ep/base-plus-facts/adapter \
        --data-path data/grpo_generated.jsonl \
        --steps 10

Reports: VRAM peak (MiB), mean step time (s), tokens/sec for each config.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ADAPTER_DEFAULT = "output/full_run_qwen35_0.8b_3ep/base-plus-facts/adapter"
DATA_DEFAULT = "data/grpo_generated.jsonl"


def load_records(path: str, n: int = 64) -> list:
    records = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("completions") and r.get("rewards"):
                    records.append(r)
            except Exception:
                pass
            if len(records) >= n:
                break
    return records


def vram_used_mib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024 ** 2


def vram_peak_mib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def tokenize_completion(tokenizer, prompt: str, completion: str, max_length: int = 512):
    full = prompt + completion
    enc = tokenizer(
        full,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    prompt_enc = tokenizer(prompt, return_tensors="pt")
    prompt_len = min(prompt_enc["input_ids"].shape[1], max_length)
    labels[:, :prompt_len] = -100
    labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels
    return enc


def run_benchmark_config(adapter_path: str, records: list, batch_size: int,
                          torch_dtype, steps: int = 10) -> dict:
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    dtype_name = "bf16" if torch_dtype == torch.bfloat16 else "fp32"
    print(f"\n--- batch_size={batch_size} dtype={dtype_name} ---", flush=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print("  loading model...", flush=True)
    t0 = time.time()
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        is_trainable=True,
        device_map="auto",
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_time = time.time() - t0
    vram_after_load = vram_used_mib()
    print(f"  loaded in {load_time:.1f}s, VRAM={vram_after_load:.0f}MiB", flush=True)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-4
    )

    device = next(model.parameters()).device
    step_times = []
    tokens_processed = []

    for step in range(steps):
        # pick a batch of records (cycle)
        batch_records = [records[i % len(records)] for i in range(step * batch_size, (step + 1) * batch_size)]

        t_step = time.time()
        total_tokens = 0
        batch_loss_scalar = 0.0

        for rec in batch_records:
            prompt = rec["prompt"]
            completions = rec["completions"]
            rewards = rec["rewards"]

            # group-relative advantage
            r_arr = torch.tensor(rewards, dtype=torch.float32)
            adv = ((r_arr - r_arr.mean()) / (r_arr.std() + 1e-8)).tolist()

            all_ids, all_mask, all_labels, all_adv = [], [], [], []
            for comp, a in zip(completions, adv):
                if not comp.strip():
                    continue
                enc = tokenize_completion(tokenizer, prompt, comp)
                all_ids.append(enc["input_ids"].to(device))
                all_mask.append(enc["attention_mask"].to(device))
                all_labels.append(enc["labels"].to(device))
                all_adv.append(a)

            if not all_ids:
                continue

            ids_b = torch.cat(all_ids, dim=0)
            mask_b = torch.cat(all_mask, dim=0)
            labels_b = torch.cat(all_labels, dim=0)
            total_tokens += ids_b.numel()

            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = model(input_ids=ids_b, attention_mask=mask_b)

            logits = outputs.logits
            del outputs  # free object; logits tensor still alive for backward
            shift_logits = logits[:, :-1, :].contiguous()
            del logits
            shift_labels = labels_b[:, 1:].contiguous()
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            token_loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())
            del shift_logits
            valid_mask = shift_labels.ne(-100)
            rec_loss = torch.tensor(0.0, device=device)
            for i, a in enumerate(all_adv):
                n_valid = valid_mask[i].sum().clamp_min(1)
                seq_lp = -(token_loss[i] * valid_mask[i]).sum() / n_valid
                rec_loss = rec_loss + (-a * seq_lp)
            rec_loss = rec_loss / len(all_adv)

            # backward per-quote: computation graph freed immediately after this line
            (rec_loss / batch_size).backward()
            batch_loss_scalar += rec_loss.item()
            del rec_loss, token_loss, ids_b, mask_b, labels_b, all_ids, all_mask, all_labels

        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        elapsed = time.time() - t_step
        step_times.append(elapsed)
        tokens_processed.append(total_tokens)
        print(f"  step {step+1}/{steps}: {elapsed:.2f}s  VRAM_peak={vram_peak_mib():.0f}MiB  loss={batch_loss_scalar:.4f}", flush=True)

    mean_step = sum(step_times[1:]) / max(len(step_times) - 1, 1)  # skip first (JIT warmup)
    mean_tokens = sum(tokens_processed[1:]) / max(len(tokens_processed) - 1, 1)
    tps = mean_tokens / mean_step if mean_step > 0 else 0.0
    vram_peak = vram_peak_mib()

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "batch_size": batch_size,
        "dtype": dtype_name,
        "mean_step_s": round(mean_step, 3),
        "tokens_per_sec": round(tps, 1),
        "vram_peak_mib": round(vram_peak, 0),
        "vram_load_mib": round(vram_after_load, 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", default=ADAPTER_DEFAULT)
    parser.add_argument("--data-path", default=DATA_DEFAULT)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-sizes", default="4,8,16,32")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — cannot benchmark.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MiB")

    records = load_records(args.data_path, n=128)
    print(f"Loaded {len(records)} records from {args.data_path}")

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    results = []
    # Only bf16 — fp32 is already proven to be the bug, no need to waste time on it
    for bs in batch_sizes:
        try:
            r = run_benchmark_config(
                args.adapter_path, records, bs, torch.bfloat16, steps=args.steps
            )
            results.append(r)
        except Exception as e:
            print(f"  FAILED at batch_size={bs} bf16: {type(e).__name__}: {str(e)[:300]}")
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"{'batch':>6}  {'dtype':>5}  {'step_s':>7}  {'tok/s':>8}  {'vram_peak':>10}  {'vram_load':>10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['batch_size']:>6}  {r['dtype']:>5}  {r['mean_step_s']:>7.3f}  "
            f"{r['tokens_per_sec']:>8.1f}  {r['vram_peak_mib']:>9.0f}M  {r['vram_load_mib']:>9.0f}M"
        )

    if results:
        best = max(results, key=lambda r: r["tokens_per_sec"])
        print(f"\nFASTEST: batch_size={best['batch_size']} dtype={best['dtype']} "
              f"→ {best['tokens_per_sec']:.1f} tok/s, {best['vram_peak_mib']:.0f}MiB peak VRAM")
        print(f"\nRecommended GRPOConfig change:")
        print(f"  batch_size: 4 → {best['batch_size']}")

    out_path = Path("benchmark_training_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()
