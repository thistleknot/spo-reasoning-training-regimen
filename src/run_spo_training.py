"""Runnable SPO fine-tuning entrypoint for confidence-bearing reasoning outputs."""

import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch
from peft import AutoPeftModelForCausalLM
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .run_ablation_matrix import (
    build_training_examples,
    collator,
    generate_completion,
    load_jsonl,
    split_indices,
    subset_records,
)
from .spo_trainer import PromptContract, SPOEvaluator, SPOTrainer, assert_output_quality


@dataclass
class SPOTrainingConfig:
    """Runtime configuration for a concrete SPO training run."""

    adapter_path: str
    dataset_path: str = "data/train_facts_with_confidence_967.jsonl"
    output_dir: str = "output/spo_training"
    model_name: str = "Qwen/Qwen3.5-0.8B"
    evaluation_metric: str = "triplet"
    holdout_fraction: float = 0.1
    seed: int = 42
    max_length: int = 512
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    beta: float = 0.1
    num_epochs: int = 1
    logging_steps: int = 10
    max_train_records: int | None = None
    max_holdout_records: int | None = None
    regression_gate_samples: int = 5
    regression_min_avg_score: float = 0.25
    regression_min_per_sample_score: float = 0.0
    regression_min_header_score: float = 0.5
    skip_regression_gate: bool = False


def maybe_limit(records: List[dict], limit: int | None) -> List[dict]:
    """Cap a record list when the caller requests a bounded run."""
    if limit is None or limit >= len(records):
        return records
    return records[:limit]


def load_spo_model_and_tokenizer(config: SPOTrainingConfig):
    """Load a trainable PEFT adapter and tokenizer for SPO fine-tuning."""
    model = AutoPeftModelForCausalLM.from_pretrained(
        config.adapter_path,
        is_trainable=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    return model, tokenizer


_TRIPLET_RE = re.compile(r"([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)")


def score_training_sample(record: dict) -> float:
    """Compute a data-quality weight for one SPO training sample.

    Require: record has an 'output_text' field containing triplet-formatted text.
    Guarantee: returns a float in [0, 1] where higher means better quality.
    Failure modes: returns 0.5 (neutral) if output_text is missing or unparseable.

    Quality signals (blended 50/50):
    - Format quality: evaluate_triplet_correctness on the gold output, which checks
      headers, pipe-format triplets, confidence annotations, and uniqueness. Examples
      with correct headers and well-formed triplets teach the model more reliably.
    - Diversity: unique premises ratio, predicate specificity, and subject diversity.
      High-diversity examples provide richer semantic coverage than degenerate ones.
    """
    text = record.get("output_text", "")
    if not text or not text.strip():
        return 0.5

    # Format quality on the gold output — rewards correct headers + pipe triplets
    prompt = record.get("input_text", "")
    contract = PromptContract.from_prompt(prompt) if prompt else None
    format_score = SPOEvaluator.evaluate_triplet_correctness(text, contract=contract)

    triplet_lines = [
        line.strip()
        for line in text.splitlines()
        if _TRIPLET_RE.search(line)
    ]
    if not triplet_lines:
        # No pipe triplets at all: rely purely on format score (likely 0)
        return round(format_score, 4)

    # Uniqueness ratio
    unique_ratio = len(set(triplet_lines)) / len(triplet_lines)

    # Mean predicate word-length across triplets
    predicate_lengths = []
    subject_tokens = []
    for line in triplet_lines:
        match = _TRIPLET_RE.search(line)
        if match:
            predicate = match.group(2).strip()
            predicate_lengths.append(len(predicate.split()))
            subject_tokens.append(match.group(1).strip().lower()[:20])

    mean_pred_len = statistics.mean(predicate_lengths) if predicate_lengths else 1.0
    # Normalise: predicates of length 1 (bare 'is') score 0, length >=5 score 1
    pred_score = min(max((mean_pred_len - 1) / 4.0, 0.0), 1.0)

    # Subject diversity: fraction of unique subjects
    subject_diversity = len(set(subject_tokens)) / max(len(subject_tokens), 1)
    # If one subject dominates (>60% of lines), down-weight
    if subject_tokens:
        most_common_count = max(subject_tokens.count(s) for s in set(subject_tokens))
        dominance = most_common_count / len(subject_tokens)
        subject_diversity *= (1.0 - max(dominance - 0.6, 0.0) / 0.4)

    diversity_score = 0.4 * unique_ratio + 0.3 * pred_score + 0.3 * subject_diversity

    # Blend: format quality (correct headers + triplet form) with diversity
    return round(0.5 * format_score + 0.5 * diversity_score, 4)


def score_dataset(records: List[dict]) -> List[float]:
    """Score every training record and normalise scores to [0, 1].

    Require: records is non-empty.
    Guarantee: returns a list of floats parallel to records, normalised so the
    maximum score in the dataset is 1.0.
    """
    raw_scores = [score_training_sample(r) for r in records]
    max_score = max(raw_scores) if raw_scores else 1.0
    if max_score <= 0:
        return [0.5] * len(records)
    return [s / max_score for s in raw_scores]


def build_spo_examples(records: List[dict], tokenizer, max_length: int) -> List[dict]:
    """Tokenize SPO training rows, retain output text, and embed pre-computed quality scores."""
    quality_scores = score_dataset(records)
    encoded_examples = build_training_examples(records, tokenizer, max_length)
    for feature, record, quality in zip(encoded_examples, records, quality_scores):
        feature["output_text"] = record["output_text"]
        feature["ground_truth"] = record["output_text"]
        feature["precomputed_reward"] = quality
    return encoded_examples


def spo_collator(tokenizer):
    """Batch collator that preserves output text metadata and pre-computed quality scores."""
    base_collator = collator(tokenizer)

    def _collate(features: List[dict]) -> Dict[str, object]:
        batch = base_collator(features)
        batch["output_texts"] = [feature["output_text"] for feature in features]
        batch["ground_truths"] = [feature["ground_truth"] for feature in features]
        batch["precomputed_rewards"] = torch.tensor(
            [feature["precomputed_reward"] for feature in features],
            dtype=torch.float32,
        )
        return batch

    return _collate


def resolve_evaluation_fn(metric_name: str) -> Callable[[str, str | None], float]:
    """Resolve the reward metric requested by the caller."""
    evaluators = {
        "triplet": SPOEvaluator.evaluate_triplet_correctness,
        "syllogism": SPOEvaluator.evaluate_syllogism_quality,
        "composite": SPOEvaluator.composite_score,
    }
    if metric_name not in evaluators:
        raise ValueError(
            f"Unsupported evaluation metric: {metric_name}. "
            f"Choose one of {sorted(evaluators)}."
        )
    return evaluators[metric_name]


def move_batch_to_model_device(batch: Dict[str, object], model) -> Dict[str, object]:
    """Move tensor fields in a batch to the active model device."""
    device = model.device
    moved = dict(batch)
    for key in ("input_ids", "attention_mask", "labels", "precomputed_rewards"):
        if key in moved and isinstance(moved[key], torch.Tensor):
            moved[key] = moved[key].to(device)
    return moved


def run_spo_training(config: SPOTrainingConfig) -> dict:
    """Run SPO fine-tuning and save the resulting adapter plus training history."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(Path(config.dataset_path))
    train_indices, holdout_indices = split_indices(
        len(records),
        config.holdout_fraction,
        config.seed,
    )
    train_records = maybe_limit(subset_records(records, train_indices), config.max_train_records)
    holdout_records = maybe_limit(subset_records(records, holdout_indices), config.max_holdout_records)

    model, tokenizer = load_spo_model_and_tokenizer(config)
    trainer = SPOTrainer(
        model=model,
        tokenizer=tokenizer,
        evaluation_fn=resolve_evaluation_fn(config.evaluation_metric),
        learning_rate=config.learning_rate,
        beta=config.beta,
    )

    train_examples = build_spo_examples(train_records, tokenizer, config.max_length)
    train_loader = DataLoader(
        train_examples,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=spo_collator(tokenizer),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )

    history = []
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(config.num_epochs):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            global_step += 1
            batch = move_batch_to_model_device(batch, model)
            loss, metrics = trainer.compute_step(
                batch,
                ground_truths=batch["ground_truths"],
            )
            (loss / config.gradient_accumulation_steps).backward()
            if (
                step % config.gradient_accumulation_steps == 0
                or step == len(train_loader)
            ):
                optimizer.step()
                optimizer.zero_grad()

            entry = {
                "epoch": epoch + 1,
                "step": global_step,
                **metrics,
            }
            history.append(entry)
            if (
                global_step == 1
                or global_step % config.logging_steps == 0
                or step == len(train_loader)
            ):
                print(
                    f"[epoch {epoch + 1}/{config.num_epochs}] "
                    f"step {step}/{len(train_loader)} "
                    f"loss={metrics['loss']:.4f} "
                    f"avg_reward={metrics['avg_reward']:.4f} "
                    f"avg_confidence={metrics['avg_confidence']:.4f}"
                )

    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")

    # Regression gate: generate actual model outputs on a sample of holdout prompts
    # and assert quality meets the floor. A failure here means SPO training degraded
    # the adapter below an acceptable quality threshold.
    regression_samples = holdout_records[: config.regression_gate_samples]
    if regression_samples and not config.skip_regression_gate:
        gate_prompts = [r["input_text"] for r in regression_samples]
        gate_outputs = [
            generate_completion(model, tokenizer, prompt, max_new_tokens=256)
            for prompt in gate_prompts
        ]
        gate_contracts = [PromptContract.from_prompt(p) for p in gate_prompts]
        gate_scores = [
            SPOEvaluator.evaluate_triplet_correctness(out, contract=c)
            for out, c in zip(gate_outputs, gate_contracts)
        ]
        gate_header_scores = [
            c.header_score(out) if c.expected_headers else 1.0
            for c, out in zip(gate_contracts, gate_outputs)
        ]
        gate_summary = {
            "samples": [
                {
                    "prompt_snippet": p[:80],
                    "output": o,
                    "score": s,
                    "header_score": hs,
                    "expected_headers": c.expected_headers,
                }
                for p, o, s, hs, c in zip(
                    gate_prompts, gate_outputs, gate_scores, gate_header_scores, gate_contracts
                )
            ],
            "avg_score": sum(gate_scores) / len(gate_scores),
            "avg_header_score": sum(gate_header_scores) / len(gate_header_scores),
            "min_avg_threshold": config.regression_min_avg_score,
            "min_per_sample_threshold": config.regression_min_per_sample_score,
            "min_header_threshold": config.regression_min_header_score,
        }
        (output_dir / "regression_gate.json").write_text(
            json.dumps(gate_summary, indent=2) + "\n"
        )
        assert_output_quality(
            gate_outputs,
            prompts=gate_prompts,
            min_avg_score=config.regression_min_avg_score,
            min_per_sample_score=config.regression_min_per_sample_score,
            min_header_score=config.regression_min_header_score,
        )

    holdout_rewards, holdout_metrics = trainer.evaluate_batch(
        inputs=[record["input_text"] for record in holdout_records],
        outputs=[record["output_text"] for record in holdout_records],
        ground_truths=[record["output_text"] for record in holdout_records],
    )
    summary = {
        "config": asdict(config),
        "train_records": len(train_records),
        "holdout_records": len(holdout_records),
        "history": history,
        "holdout_metrics": holdout_metrics,
        "holdout_reward_count": len(holdout_rewards),
        "adapter_path": str(output_dir / "adapter"),
    }
    (output_dir / "spo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SPO fine-tuning on a confidence-bearing dataset")
    parser.add_argument("--adapter-path", required=True, help="Starting PEFT adapter directory")
    parser.add_argument("--dataset-path", default="data/train_facts_with_confidence_967.jsonl")
    parser.add_argument("--output-dir", default="output/spo_training")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--evaluation-metric",
        default="triplet",
        choices=["triplet", "syllogism", "composite"],
        help="Reward function used for SPO weighting",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-holdout-records", type=int, default=None)
    parser.add_argument("--regression-gate-samples", type=int, default=5)
    parser.add_argument("--regression-min-avg-score", type=float, default=0.5)
    parser.add_argument("--regression-min-per-sample-score", type=float, default=0.3)
    parser.add_argument("--regression-min-header-score", type=float, default=0.5)
    parser.add_argument("--skip-regression-gate", action="store_true", default=False)
    args = parser.parse_args()

    result = run_spo_training(SPOTrainingConfig(**vars(args)))
    print(json.dumps(result, indent=2))
