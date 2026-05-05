"""
Run the confidence-regimen ablation matrix end to end.

This runner:
1. loads the rebuilt canonical corpora
2. trains the ablation variants defined by the staged curriculum
3. evaluates base reasoning quality on a holdout split
4. evaluates confidence utility when the experiment includes argument-level confidence

The evaluation is heuristic unless an external judge has already produced
`syllogism_quality` scores. That keeps the matrix executable in a local-only setup.
"""

import difflib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from .chat_format import (
    build_generation_prompt,
    build_training_conversation,
    strip_response_preamble,
)
from .evaluate_regimens import EvalRecord, evaluate_confidence_utility
from .preprocess_training_data import extract_section, parse_triplet_list
from .spo_trainer import SPOEvaluator
from .training_strategy import RegimenName, TrainingStrategy


SECTION_RE = re.compile(
    r"Throughline:\s*(.+?)(?:\n\s*Confidence:|\Z)",
    re.DOTALL,
)
CONFIDENCE_RE = re.compile(r"Confidence:\s*([0-9]*\.?[0-9]+)")


@dataclass
class AblationConfig:
    """Runtime configuration for the ablation matrix."""

    model_name: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "output/ablations"
    holdout_fraction: float = 0.1
    seed: int = 42
    max_length: int = 512
    max_new_tokens: int = 192
    batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warm_start_epochs: float = 1.0
    mix_epochs: float = 1.0
    logging_steps: int = 10
    max_train_records_per_regimen: int | None = None
    max_holdout_records: int | None = None
    experiment_names: List[str] | None = None


def load_jsonl(path: Path) -> List[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def split_indices(total: int, holdout_fraction: float, seed: int) -> tuple[List[int], List[int]]:
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    holdout_size = max(1, int(total * holdout_fraction))
    holdout = sorted(indices[:holdout_size])
    train = sorted(indices[holdout_size:])
    return train, holdout


def subset_records(records: Sequence[dict], indices: Sequence[int]) -> List[dict]:
    return [records[index] for index in indices]


def training_text(record: dict) -> tuple[str, str]:
    input_text = record["input_text"].strip()
    output_text = record["output_text"].strip()
    return input_text, output_text


def build_training_examples(
    records: Sequence[dict],
    tokenizer,
    max_length: int,
) -> List[dict]:
    examples = []
    for record in records:
        prompt_text, completion_text = training_text(record)
        full_text = build_training_conversation(tokenizer, prompt_text, completion_text)
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        prompt_length = len(
            tokenizer(
                build_generation_prompt(tokenizer, prompt_text),
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )["input_ids"]
        )
        labels = list(encoded["input_ids"])
        prompt_cutoff = min(prompt_length, len(labels))
        for index in range(prompt_cutoff):
            labels[index] = -100
        examples.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": labels,
            }
        )
    return examples


def pad_batch(features: List[dict], pad_token_id: int) -> dict:
    max_len = max(len(feature["input_ids"]) for feature in features)
    input_ids = []
    attention_masks = []
    labels = []
    for feature in features:
        pad_length = max_len - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_token_id] * pad_length)
        attention_masks.append(feature["attention_mask"] + [0] * pad_length)
        labels.append(feature["labels"] + [-100] * pad_length)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collator(tokenizer):
    def _collate(features: List[dict]) -> dict:
        return pad_batch(features, tokenizer.pad_token_id)

    return _collate


def sample_mixture(
    regimen_to_records: Dict[str, Sequence[dict]],
    weights: Dict[str, float],
    target_size: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    active_weights = {name: weight for name, weight in weights.items() if name in regimen_to_records}
    total_weight = sum(active_weights.values())
    if total_weight <= 0:
        raise ValueError("Mixture stage has no active regimens")

    mixture = []
    for regimen_name, weight in active_weights.items():
        records = list(regimen_to_records[regimen_name])
        if not records:
            continue
        rng.shuffle(records)
        sample_size = max(1, int(round((weight / total_weight) * target_size)))
        repeats = math.ceil(sample_size / len(records))
        pool = (records * repeats)[:sample_size]
        mixture.extend(pool)
    rng.shuffle(mixture)
    return mixture[:target_size]


def load_model_and_tokenizer(model_name: str):
    compute_dtype = torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def train_stage(
    model,
    tokenizer,
    stage_records: Sequence[dict],
    output_dir: Path,
    epochs: float,
    config: AblationConfig,
    stage_name: str,
):
    examples = build_training_examples(stage_records, tokenizer, config.max_length)
    dataset = Dataset.from_list(examples)
    args = TrainingArguments(
        output_dir=str(output_dir / stage_name),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        bf16=False,
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator(tokenizer),
    )
    trainer.train()


def extract_throughline(text: str) -> str:
    match = SECTION_RE.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def extract_confidence(text: str) -> float | None:
    match = CONFIDENCE_RE.search(text)
    if not match:
        return None
    return float(match.group(1))


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    chat_prompt = build_generation_prompt(tokenizer, prompt)
    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
        )
    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()
    return strip_response_preamble(decoded)


def throughline_quality(predicted: str, expected: str) -> float:
    heuristic = SPOEvaluator.evaluate_syllogism_quality(predicted or "N/A")
    similarity = difflib.SequenceMatcher(None, predicted.strip(), expected.strip()).ratio()
    return round((heuristic + similarity) / 2, 4)


def parse_reasoning_output(text: str) -> dict:
    """Extract the canonical base-reasoning sections from generated text."""
    throughline = extract_section(text, "Throughline")
    entailed = parse_triplet_list(extract_section(text, "Entailed Premises")) or []
    non_entailed = parse_triplet_list(extract_section(text, "Non-Entailed Premises")) or []
    return {
        "throughline": throughline.strip() if throughline else "",
        "entailed": entailed,
        "non_entailed": non_entailed,
    }


def normalize_triplet(triplet: str) -> str:
    """Canonicalize a triplet for set comparison."""
    return re.sub(r"\s+", " ", triplet.strip().lower())


def triplet_overlap_score(predicted: Sequence[str], expected: Sequence[str]) -> float:
    """Compute an F1-style overlap score between predicted and expected triplets."""
    predicted_set = {normalize_triplet(triplet) for triplet in predicted if triplet.strip()}
    expected_set = {normalize_triplet(triplet) for triplet in expected if triplet.strip()}
    if not expected_set:
        return 1.0 if not predicted_set else 0.0
    if not predicted_set:
        return 0.0

    matches = predicted_set & expected_set
    precision = len(matches) / len(predicted_set)
    recall = len(matches) / len(expected_set)
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)


def base_reasoning_quality(predicted_output: str, expected_output: str) -> float:
    required_headers = ("Non-Entailed Premises:", "Entailed Premises:", "Throughline:")
    if not all(header in predicted_output for header in required_headers):
        return 0.0

    predicted = parse_reasoning_output(predicted_output)
    expected = parse_reasoning_output(expected_output)
    if not predicted["throughline"]:
        return 0.0

    structure_score = sum(
        1.0 if predicted[section] else 0.0
        for section in ("non_entailed", "entailed", "throughline")
    ) / 3
    non_entailed_score = triplet_overlap_score(
        predicted["non_entailed"],
        expected["non_entailed"],
    )
    entailed_score = triplet_overlap_score(
        predicted["entailed"],
        expected["entailed"],
    )
    throughline_score = throughline_quality(
        predicted["throughline"],
        expected["throughline"],
    )
    return round(
        (structure_score + non_entailed_score + entailed_score + throughline_score) / 4,
        4,
    )


def run_quality_eval(model, tokenizer, holdout_records: Sequence[dict], config: AblationConfig) -> dict:
    scores = []
    samples = []
    for record in holdout_records:
        prompt, expected = training_text(record)
        generated = generate_completion(model, tokenizer, prompt, config.max_new_tokens)
        quality = base_reasoning_quality(generated, expected)
        scores.append(quality)
        if len(samples) < 3:
            samples.append(
                {
                    "prompt": prompt,
                    "generated": generated,
                    "expected": expected,
                    "quality": quality,
                }
            )
    return {
        "avg_quality": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "samples": samples,
    }


def run_confidence_eval(model, tokenizer, holdout_records: Sequence[dict], config: AblationConfig) -> dict | None:
    eval_records = []
    samples = []
    for record in holdout_records:
        prompt, expected = training_text(record)
        generated = generate_completion(model, tokenizer, prompt, config.max_new_tokens)
        predicted_throughline = extract_throughline(generated)
        expected_throughline = extract_throughline(expected)
        predicted_confidence = extract_confidence(generated)
        if predicted_confidence is None:
            continue
        quality = throughline_quality(predicted_throughline, expected_throughline)
        eval_records.append(
            EvalRecord(
                quote=prompt,
                predicted_confidence=predicted_confidence,
                syllogism_quality=quality,
            )
        )
        if len(samples) < 3:
            samples.append(
                {
                    "prompt": prompt,
                    "generated": generated,
                    "expected": expected,
                    "predicted_confidence": predicted_confidence,
                    "quality": quality,
                }
            )
    if not eval_records:
        return None
    report = evaluate_confidence_utility(eval_records)
    payload = report.to_dict()
    payload["samples"] = samples
    return payload


def maybe_limit(records: List[dict], limit: int | None) -> List[dict]:
    if limit is None or limit >= len(records):
        return records
    return records[:limit]


def _markdown_code_block(text: str) -> str:
    return f"```text\n{text.strip()}\n```"


def write_holdout_markdown(results: dict, output_path: Path) -> None:
    """Write a human-readable holdout comparison markdown for all ablations.

    Preconditions:
        `results` follows the `ablation_summary.json` schema produced by this
        module.
    Failure modes:
        Missing sample arrays degrade to shorter sections rather than raising.
    """
    experiments = results["experiments"]
    experiment_names = list(experiments.keys())
    quality_samples_by_experiment = {
        name: experiments[name]["quality_report"].get("samples", [])
        for name in experiment_names
    }
    quality_sample_count = min(
        (len(samples) for samples in quality_samples_by_experiment.values()),
        default=0,
    )

    lines = [
        "# Holdout Ablation Comparison",
        "",
        "This report compares the sampled holdout outputs from each trained ablation.",
        "",
        "## Quality summary",
        "",
        "| Ablation | Average quality |",
        "|---|---:|",
    ]
    for name in experiment_names:
        lines.append(
            f"| {name} | {experiments[name]['quality_report']['avg_quality']:.4f} |"
        )

    for index in range(quality_sample_count):
        reference = quality_samples_by_experiment[experiment_names[0]][index]
        lines.extend(
            [
                "",
                f"## Holdout example {index + 1}",
                "",
                "**Prompt**",
                "",
                _markdown_code_block(reference["prompt"]),
                "",
                "**Expected**",
                "",
                _markdown_code_block(reference["expected"]),
            ]
        )
        for name in experiment_names:
            sample = quality_samples_by_experiment[name][index]
            lines.extend(
                [
                    "",
                    f"### {name}",
                    "",
                    f"**Quality:** {sample['quality']:.4f}",
                    "",
                    _markdown_code_block(sample["generated"]),
                ]
            )

    confidence_experiment = next(
        (
            name
            for name in experiment_names
            if experiments[name].get("confidence_report") is not None
        ),
        None,
    )
    if confidence_experiment:
        confidence_report = experiments[confidence_experiment]["confidence_report"]
        lines.extend(
            [
                "",
                "## Confidence summary",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| count | {confidence_report['count']} |",
                f"| pearson | {confidence_report['pearson']:.4f} |",
                f"| spearman | {confidence_report['spearman']:.4f} |",
                f"| auroc | {confidence_report['auroc']:.4f} |",
                f"| brier | {confidence_report['brier']:.4f} |",
                f"| ece | {confidence_report['ece']:.4f} |",
            ]
        )
        for index, sample in enumerate(confidence_report.get("samples", []), start=1):
            lines.extend(
                [
                    "",
                    f"## Confidence holdout example {index}",
                    "",
                    "**Prompt**",
                    "",
                    _markdown_code_block(sample["prompt"]),
                    "",
                    f"**Predicted confidence:** {sample['predicted_confidence']:.4f}",
                    "",
                    f"**Heuristic quality:** {sample['quality']:.4f}",
                    "",
                    "**Generated**",
                    "",
                    _markdown_code_block(sample["generated"]),
                    "",
                    "**Expected**",
                    "",
                    _markdown_code_block(sample["expected"]),
                ]
            )

    output_path.write_text("\n".join(lines) + "\n")


def run_ablation_matrix(config: AblationConfig) -> dict:
    strategy = TrainingStrategy.default()
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    base_records = load_jsonl(Path("data/train_clean_for_model_967.jsonl"))
    facts_records = load_jsonl(Path("data/train_facts_with_confidence_967.jsonl"))
    syllogism_records = load_jsonl(Path("data/train_syllogism_with_confidence_967.jsonl"))

    train_indices, holdout_indices = split_indices(len(base_records), config.holdout_fraction, config.seed)
    train_sets = {
        RegimenName.BASE_REASONING.value: maybe_limit(subset_records(base_records, train_indices), config.max_train_records_per_regimen),
        RegimenName.FACTS_WITH_CONFIDENCE.value: maybe_limit(subset_records(facts_records, train_indices), config.max_train_records_per_regimen),
        RegimenName.SYLLOGISM_WITH_CONFIDENCE.value: maybe_limit(subset_records(syllogism_records, train_indices), config.max_train_records_per_regimen),
    }
    holdout_base = subset_records(base_records, holdout_indices)
    holdout_syllogism = subset_records(syllogism_records, holdout_indices)
    holdout_base = maybe_limit(holdout_base, config.max_holdout_records)
    holdout_syllogism = maybe_limit(holdout_syllogism, config.max_holdout_records)

    results = {
        "config": config.__dict__,
        "strategy": strategy.to_dict(),
        "experiments": {},
    }

    mix_stage = next(stage for stage in strategy.stages if stage.name == "multitask-mix")
    warm_stage = next(stage for stage in strategy.stages if stage.name == "base-warm-start")

    ablations = strategy.ablations
    if config.experiment_names:
        selected = set(config.experiment_names)
        ablations = [experiment for experiment in strategy.ablations if experiment.name in selected]
        if not ablations:
            raise ValueError(f"No ablations matched --experiment values: {sorted(selected)}")

    for experiment_index, experiment in enumerate(ablations, start=1):
        experiment_dir = output_root / experiment.name
        print(f"[{experiment_index}/{len(ablations)}] {experiment.name}: loading model")
        model, tokenizer = load_model_and_tokenizer(config.model_name)
        target_size = len(train_sets[RegimenName.BASE_REASONING.value])

        print(
            f"[{experiment_index}/{len(ablations)}] {experiment.name}: "
            f"warm start on {len(train_sets[RegimenName.BASE_REASONING.value])} records"
        )
        train_stage(
            model=model,
            tokenizer=tokenizer,
            stage_records=train_sets[RegimenName.BASE_REASONING.value],
            output_dir=experiment_dir,
            epochs=config.warm_start_epochs,
            config=config,
            stage_name=warm_stage.name,
        )

        if set(experiment.enabled_regimens) != {RegimenName.BASE_REASONING.value}:
            stage_records = sample_mixture(
                regimen_to_records={name: train_sets[name] for name in experiment.enabled_regimens},
                weights={name: mix_stage.regimen_weights[name] for name in experiment.enabled_regimens},
                target_size=target_size,
                seed=config.seed,
            )
            print(
                f"[{experiment_index}/{len(ablations)}] {experiment.name}: "
                f"mixture stage on {len(stage_records)} records"
            )
            train_stage(
                model=model,
                tokenizer=tokenizer,
                stage_records=stage_records,
                output_dir=experiment_dir,
                epochs=config.mix_epochs,
                config=config,
                stage_name=mix_stage.name,
            )

        print(
            f"[{experiment_index}/{len(ablations)}] {experiment.name}: "
            f"quality eval on {len(holdout_base)} holdout records"
        )
        quality_report = run_quality_eval(model, tokenizer, holdout_base, config)
        confidence_report = None
        if RegimenName.SYLLOGISM_WITH_CONFIDENCE.value in experiment.enabled_regimens:
            print(
                f"[{experiment_index}/{len(ablations)}] {experiment.name}: "
                f"confidence eval on {len(holdout_syllogism)} holdout records"
            )
            confidence_report = run_confidence_eval(model, tokenizer, holdout_syllogism, config)

        experiment_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(experiment_dir / "adapter")
        tokenizer.save_pretrained(experiment_dir / "adapter")

        results["experiments"][experiment.name] = {
            "enabled_regimens": experiment.enabled_regimens,
            "hypothesis": experiment.hypothesis,
            "quality_report": quality_report,
            "confidence_report": confidence_report,
        }
        with (experiment_dir / "results.json").open("w") as handle:
            json.dump(results["experiments"][experiment.name], handle, indent=2)

        del model
        torch.cuda.empty_cache()

    with (output_root / "ablation_summary.json").open("w") as handle:
        json.dump(results, handle, indent=2)
    write_holdout_markdown(results, output_root / "holdout_examples.md")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the confidence-regimen ablation matrix")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="output/ablations")
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warm-start-epochs", type=float, default=1.0)
    parser.add_argument("--mix-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-train-records-per-regimen", type=int, default=None)
    parser.add_argument("--max-holdout-records", type=int, default=None)
    parser.add_argument(
        "--experiment",
        dest="experiment_names",
        action="append",
        default=None,
        help="Limit the run to one or more ablation names",
    )
    args = parser.parse_args()

    result = run_ablation_matrix(
        AblationConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warm_start_epochs=args.warm_start_epochs,
            mix_epochs=args.mix_epochs,
            logging_steps=args.logging_steps,
            max_train_records_per_regimen=args.max_train_records_per_regimen,
            max_holdout_records=args.max_holdout_records,
            experiment_names=args.experiment_names,
        )
    )
    print(json.dumps(result, indent=2))
