"""Offline GRPO data generation — phase 1 of the two-phase offline GRPO workflow.

Generates G completions per quote and scores them with a frozen judge, writing
the results to JSONL for later use by run_grpo_training.py --precomputed-data-path.
No gradient computation — inference only.

VRAM budget (for 0.8B model at 4-bit nf4 + double quantization):
  Same-judge mode (default; --judge-path == --adapter-path or judge omitted):
    Single model loaded once, used for both generation and scoring.
    Peak: ~460 MB (model weights + KV cache during generation).
  Shared-base mode (--shared-base):
    Base loaded once, policy + judge as named PEFT adapters.
    Peak: ~420 MB (4-bit base + 2× QLoRA adapter weight sets).
  Standalone mode (explicit --judge-path pointing to different checkpoint):
    Two separate model instances.
    Peak: ~800 MB — avoid on tight budgets.

Usage:
    # Same-judge mode (under 500 MB for 0.8B):
    python generate_grpo_data.py \\
        --adapter-path output/sft_adapter \\
        --dataset-path data/train_structured_967.jsonl \\
        --output-path data/grpo_generated.jsonl \\
        --group-size 8

    # Then train offline (policy only, no judge needed):
    python run_grpo_training.py \\
        --adapter-path output/sft_adapter \\
        --precomputed-data-path data/grpo_generated.jsonl \\
        --output-dir output/grpo_training
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

from src.frozen_judge import FrozenJudge
from src.run_ablation_matrix import load_jsonl
from src.serialize_training_format import build_base_reasoning_prompt
from src.chat_format import build_generation_prompt, strip_response_preamble


def _free_vram_mb() -> float:
    """Return free VRAM in MB across all processes, or 0 if CUDA is unavailable.

    Uses mem_get_info() which reports device-level free memory (not just the
    current process), unlike memory_reserved() which only tracks this process.
    """
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, _ = torch.cuda.mem_get_info(0)
    return free_bytes / 1024 / 1024


def _bnb_config_if_cuda() -> dict:
    """Return BitsAndBytesConfig kwargs when CUDA has enough headroom, else empty dict.

    4-bit quantization is CUDA-only. When running on CPU or when GPU VRAM is
    tight, we fall back to bfloat16 (~1.6 GB for 0.8B) which fits comfortably
    in system RAM.
    """
    # Need ~460 MB free for 4-bit 0.8B + KV cache during generation
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


def _device_map() -> str:
    """Return device_map value: 'auto' only when GPU has enough free VRAM, else 'cpu'."""
    if not torch.cuda.is_available():
        return "cpu"
    return "auto" if _free_vram_mb() >= 450 else "cpu"


def _load_done_quotes(output_path: Path) -> set:
    """Load quotes already written to the output file for --resume support."""
    done: set = set()
    if output_path.exists():
        with open(output_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["quote"])
                    except (KeyError, json.JSONDecodeError):
                        pass
    return done


def _sample_completions(
    model,
    tokenizer,
    prompt: str,
    group_size: int,
    max_new_tokens: int,
    max_length: int,
) -> list[str]:
    """Sample group_size completions from the model for a single prompt.

    Preconditions:
        model is in eval mode and on the correct device.
        prompt is a chat-format string.
    Guarantee:
        Returns exactly group_size strings; may be empty if generation fails.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    prompt_len = input_ids.shape[1]
    completions = []
    for seq in outputs:
        generated = seq[prompt_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        completions.append(strip_response_preamble(text))
    return completions


def generate_grpo_data(args: argparse.Namespace) -> None:
    """Main generation loop.

    Preconditions:
        args.adapter_path exists and contains a PEFT adapter.
        args.dataset_path exists and contains JSONL records with 'quote' or 'input_text'.
    Guarantee:
        Writes one JSON line per quote to args.output_path.
        Supports --resume: skips quotes already present in the output file.
        Writes a summary JSON alongside the output file on completion.
    """
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    judge_path = args.judge_path or args.adapter_path
    same_judge = judge_path == args.adapter_path

    # Resume: skip already-processed quotes
    done_quotes = _load_done_quotes(output_path) if args.resume else set()
    if done_quotes:
        print(f"  Resume: skipping {len(done_quotes)} already-generated quotes.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model(s)
    free_mb = _free_vram_mb()
    print(f"GPU free VRAM: {free_mb:.0f} MB — {'4-bit on GPU' if free_mb >= 450 else 'bfloat16 on CPU (GPU full)'}")

    if args.shared_base:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base_name = args.base_model_name
        if base_name is None:
            adapter_config = Path(args.adapter_path) / "adapter_config.json"
            base_name = json.loads(adapter_config.read_text()).get("base_model_name_or_path")
            if not base_name:
                raise ValueError(
                    "Cannot infer base_model_name from adapter_config.json; "
                    "pass --base-model-name explicitly."
                )

        print(f"Shared-base mode: loading {base_name} ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name, device_map=_device_map(), **_bnb_config_if_cuda()
        )
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"  Mounting policy adapter from {args.adapter_path} ...")
        policy_model = PeftModel.from_pretrained(
            base_model, args.adapter_path, adapter_name="policy", is_trainable=False
        )
        print(f"  Mounting judge adapter from {judge_path} ...")
        policy_model.load_adapter(judge_path, adapter_name="judge")
        policy_model.set_adapter("policy")
        policy_model.eval()

        judge = FrozenJudge.from_shared_model(
            model=policy_model,
            tokenizer=tokenizer,
            judge_adapter_name="judge",
            policy_adapter_name="policy",
            entailment_weight=args.judge_entailment_weight,
            non_entailment_weight=args.judge_non_entailment_weight,
            conclusion_weight=args.judge_conclusion_weight,
            confidence_samples=args.confidence_samples,
            confidence_weight=args.confidence_weight,
            confidence_temperature=args.confidence_temperature,
        )

    elif same_judge:
        # Single model used for both generation and scoring — lowest memory mode
        print(f"Same-judge mode: loading {args.adapter_path} ...")
        policy_model = AutoPeftModelForCausalLM.from_pretrained(
            args.adapter_path,
            is_trainable=False,
            device_map=_device_map(),
            **_bnb_config_if_cuda(),
        )
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        policy_model.eval()

        judge = FrozenJudge(
            adapter_path=args.adapter_path,
            entailment_weight=args.judge_entailment_weight,
            non_entailment_weight=args.judge_non_entailment_weight,
            conclusion_weight=args.judge_conclusion_weight,
            confidence_samples=args.confidence_samples,
            confidence_weight=args.confidence_weight,
            confidence_temperature=args.confidence_temperature,
        )
        # Reuse the already-loaded model to avoid a second load
        judge.model = policy_model
        judge.tokenizer = tokenizer

    else:
        # Standalone: separate policy and judge
        print(f"Standalone mode: loading policy from {args.adapter_path} ...")
        policy_model = AutoPeftModelForCausalLM.from_pretrained(
            args.adapter_path,
            is_trainable=False,
            device_map=_device_map(),
            **_bnb_config_if_cuda(),
        )
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        policy_model.eval()

        print(f"  Loading judge from {judge_path} ...")
        judge = FrozenJudge(
            adapter_path=judge_path,
            entailment_weight=args.judge_entailment_weight,
            non_entailment_weight=args.judge_non_entailment_weight,
            conclusion_weight=args.judge_conclusion_weight,
            confidence_samples=args.confidence_samples,
            confidence_weight=args.confidence_weight,
            confidence_temperature=args.confidence_temperature,
        )

    # Load dataset
    records = load_jsonl(Path(args.dataset_path))
    if args.max_records:
        records = records[: args.max_records]

    quotes = []
    prompts = []
    for rec in records:
        quote = rec.get("quote") or rec.get("input_text", "")
        if not quote or not quote.strip():
            continue
        if quote in done_quotes:
            continue
        prompt_text = build_base_reasoning_prompt(quote)
        prompt = build_generation_prompt(tokenizer, prompt_text)
        quotes.append(quote)
        prompts.append(prompt)

    print(f"Generating {args.group_size} completions for {len(quotes)} quotes → {output_path}")

    all_zero_count = 0
    written = 0

    with open(output_path, "a") as fh:
        for idx, (quote, prompt) in enumerate(zip(quotes, prompts), start=1):
            completions = _sample_completions(
                policy_model,
                tokenizer,
                prompt,
                args.group_size,
                args.max_new_tokens,
                args.max_length,
            )
            rewards = judge.batch_score_completions(quote, completions)

            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            max_reward = max(rewards) if rewards else 0.0
            all_zero = max_reward == 0.0
            if all_zero:
                all_zero_count += 1

            if args.min_reward is not None and mean_reward < args.min_reward:
                continue  # skip below-threshold records when filtering requested

            record = {
                "quote": quote,
                "prompt": prompt,
                "completions": completions,
                "rewards": rewards,
                "mean_reward": round(mean_reward, 6),
                "max_reward": round(max_reward, 6),
                "all_zero": all_zero,
            }
            fh.write(json.dumps(record) + "\n")
            written += 1

            if idx % 10 == 0 or idx == len(quotes):
                print(
                    f"  [{idx}/{len(quotes)}] mean_reward={mean_reward:.4f} "
                    f"all_zero_so_far={all_zero_count}"
                )

            # Free GPU memory after each quote to reduce peak footprint
            torch.cuda.empty_cache()

    summary = {
        "adapter_path": args.adapter_path,
        "judge_path": judge_path,
        "dataset_path": args.dataset_path,
        "output_path": str(output_path),
        "group_size": args.group_size,
        "quotes_processed": len(quotes),
        "records_written": written,
        "all_zero_count": all_zero_count,
        "all_zero_fraction": all_zero_count / max(len(quotes), 1),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"\nDone. Written {written}/{len(quotes)} records to {output_path}. "
        f"All-zero fraction: {summary['all_zero_fraction']:.1%}. "
        f"Summary: {summary_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 of offline GRPO: generate and score completions without gradients.\n"
            "Output JSONL is consumed by run_grpo_training.py --precomputed-data-path."
        )
    )
    parser.add_argument(
        "--adapter-path", required=True,
        help="Path to the PEFT adapter (policy model for generation).",
    )
    parser.add_argument(
        "--dataset-path",
        default="data/train_structured_967.jsonl",
        help="Input JSONL with 'quote' or 'input_text' fields.",
    )
    parser.add_argument(
        "--output-path",
        default="data/grpo_generated.jsonl",
        help="Output JSONL path for (quote, completions, rewards) records.",
    )
    parser.add_argument(
        "--judge-path", default=None,
        help=(
            "Path to judge adapter. Defaults to --adapter-path (same-judge mode, "
            "lowest VRAM — single model load)."
        ),
    )
    parser.add_argument("--group-size", type=int, default=8, help="Completions per quote (G).")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generation tokens.")
    parser.add_argument("--max-length", type=int, default=512, help="Max input tokenization length.")
    parser.add_argument("--max-records", type=int, default=None, help="Cap number of input quotes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip quotes already present in --output-path (append mode).",
    )
    parser.add_argument(
        "--min-reward", type=float, default=None,
        help="Only write records where mean_reward >= this threshold. "
             "Default: write all (including all-zero groups).",
    )
    # Judge reward weights
    parser.add_argument("--judge-entailment-weight", type=float, default=0.5)
    parser.add_argument("--judge-non-entailment-weight", type=float, default=0.2)
    parser.add_argument("--judge-conclusion-weight", type=float, default=0.3)
    # Confidence distribution
    parser.add_argument("--confidence-samples", type=int, default=0,
                        help="K confidence samples per completion (0 = disabled).")
    parser.add_argument("--confidence-weight", type=float, default=0.0)
    parser.add_argument("--confidence-temperature", type=float, default=0.7)
    # Memory mode
    parser.add_argument("--shared-base", action="store_true",
                        help="Use shared-base adapter mode (~420 MB for 0.8B).")
    parser.add_argument("--base-model-name", default=None,
                        help="Explicit HF model name; inferred from adapter_config.json if omitted.")

    args = parser.parse_args()
    generate_grpo_data(args)


if __name__ == "__main__":
    main()
