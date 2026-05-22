"""CLI entrypoint for GRPO fine-tuning of SPO extraction models.

Two operating modes:

  Live mode (default):
    python run_grpo_training.py \\
        --adapter-path output/sft_adapter \\
        --dataset-path data/train_structured_967.jsonl \\
        --output-dir output/grpo_training \\
        --group-size 8

  Offline mode (phase 2 of two-phase workflow; lower peak VRAM during training):
    # Phase 1: generate completions + rewards, inference only
    python generate_grpo_data.py \\
        --adapter-path output/sft_adapter \\
        --output-path data/grpo_generated.jsonl

    # Phase 2: train from precomputed data, no judge needed
    python run_grpo_training.py \\
        --adapter-path output/sft_adapter \\
        --precomputed-data-path data/grpo_generated.jsonl \\
        --output-dir output/grpo_training

The judge defaults to the same adapter as the policy (frozen copy).
Pass --judge-path to use a different checkpoint as the frozen judge.
"""

import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.grpo_trainer import GRPOConfig, run_grpo_training


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning: live generation + frozen-judge reward + patience termination"
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Starting PEFT adapter (SFT cold start).",
    )
    parser.add_argument(
        "--dataset-path",
        default="data/train_structured_967.jsonl",
        help="JSONL dataset with 'quote' or 'input_text' fields.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/grpo_training",
        help="Directory for checkpoints, history, and summary.",
    )
    parser.add_argument(
        "--judge-path",
        default=None,
        help="Frozen judge adapter path. Defaults to --adapter-path.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="G: completions sampled per quote per step.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop after this many epochs with no reward improvement.",
    )
    parser.add_argument(
        "--patience-delta",
        type=float,
        default=0.01,
        help="Minimum reward improvement to reset patience counter.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens generated per completion.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KL penalty weight.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.1,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Quotes per batch (each spawns group-size completions).",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--judge-entailment-weight",
        type=float,
        default=0.5,
        help="Reward weight for entailed-premise probes.",
    )
    parser.add_argument(
        "--judge-non-entailment-weight",
        type=float,
        default=0.2,
        help="Reward weight for non-entailed-premise probes.",
    )
    parser.add_argument(
        "--judge-conclusion-weight",
        type=float,
        default=0.3,
        help="Reward weight for conclusion coherence probe.",
    )
    parser.add_argument(
        "--confidence-samples",
        type=int,
        default=4,
        help=(
            "K: confidence rating samples per completion from the frozen judge. "
            "0 disables the confidence distribution signal. "
            "Total scoring work per quote = group-size × (probes + confidence-samples)."
        ),
    )
    parser.add_argument(
        "--confidence-weight",
        type=float,
        default=0.0,
        help=(
            "Budget for the confidence distribution signal (conf_mean × (1 - conf_std)). "
            "The structural weights (entailment + non_entailment + conclusion) are scaled "
            "to fill the remainder (1 - confidence-weight). Set > 0 to activate."
        ),
    )
    parser.add_argument(
        "--confidence-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for K confidence score generations.",
    )
    parser.add_argument(
        "--dead-quote-streak-threshold",
        type=int,
        default=2,
        help=(
            "Prune a quote after this many consecutive epochs where every completion "
            "in the group scores 0.0. Requires multi-epoch consensus before pruning."
        ),
    )
    parser.add_argument(
        "--precomputed-data-path",
        default=None,
        help=(
            "Path to precomputed JSONL from generate_grpo_data.py. "
            "When set, live generation and judge scoring are skipped entirely — "
            "the trainer reads (completions, rewards) from this file and only "
            "runs the policy gradient update. Lowest possible training VRAM: "
            "~460 MB peak for 0.8B at 4-bit (policy only, no judge loaded)."
        ),
    )
    parser.add_argument(
        "--shared-base",
        action="store_true",
        default=False,
        help=(
            "Load the base model once and mount policy + judge as two named PEFT adapters. "
            "Reduces peak VRAM from ~2× to ~1× model size. "
            "Required for sub-1B models targeting a 400MB budget."
        ),
    )
    parser.add_argument(
        "--base-model-name",
        default=None,
        help=(
            "HuggingFace model ID for the base model when --shared-base is set. "
            "E.g. 'Qwen/Qwen3-0.6B'. Defaults to the base model recorded in the adapter config."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    config = GRPOConfig(
        adapter_path=args.adapter_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        judge_path=args.judge_path,
        group_size=args.group_size,
        patience=args.patience,
        patience_delta=args.patience_delta,
        max_epochs=args.max_epochs,
        max_new_tokens=args.max_new_tokens,
        learning_rate=args.learning_rate,
        beta=args.beta,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_train_records=args.max_train_records,
        logging_steps=args.logging_steps,
        judge_entailment_weight=args.judge_entailment_weight,
        judge_non_entailment_weight=args.judge_non_entailment_weight,
        judge_conclusion_weight=args.judge_conclusion_weight,
        confidence_samples=args.confidence_samples,
        confidence_weight=args.confidence_weight,
        confidence_temperature=args.confidence_temperature,
        dead_quote_streak_threshold=args.dead_quote_streak_threshold,
        shared_base=args.shared_base,
        base_model_name=args.base_model_name,
        precomputed_data_path=args.precomputed_data_path,
    )

    result = run_grpo_training(config)
    print(json.dumps(result, indent=2))
