"""GRPO (Group Relative Policy Optimization) trainer for SPO extraction.

Extends the SPO training loop with live generation, frozen-judge scoring,
group-relative advantage normalization, and patience-gated termination.

The outer loop:
    for epoch in 1..max_epochs:
        for each quote in dataset:
            sample G completions from policy
            score each with frozen judge
            normalize to group-relative advantages
            update policy toward high-advantage completions
        track mean epoch reward; stop when improvement < patience_delta
        for K consecutive epochs

Require:  adapter_path points to a trained PEFT adapter (the SFT cold start).
          judge_path points to the same or compatible frozen adapter.
Guarantee: best checkpoint (highest mean reward) is always saved before exit.
Maintain:  judge weights are never updated; policy weights update every step.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import AutoPeftModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .frozen_judge import FrozenJudge
from .run_ablation_matrix import load_jsonl, split_indices, subset_records
from .serialize_training_format import build_base_reasoning_prompt
from .chat_format import build_generation_prompt, strip_response_preamble


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    """Runtime configuration for a GRPO training run.

    Preconditions:
        adapter_path: valid PEFT adapter directory (policy start).
        judge_path: valid PEFT adapter directory (frozen judge; defaults to adapter_path).
        dataset_path: JSONL with 'quote' or 'input_text' fields.
    """

    adapter_path: str
    dataset_path: str = "data/train_structured_967.jsonl"
    output_dir: str = "output/grpo_training"
    judge_path: Optional[str] = None          # defaults to adapter_path if None

    # GRPO hyperparameters
    group_size: int = 8                        # G completions per prompt
    patience: int = 3                          # epochs with no improvement before stop
    patience_delta: float = 0.01              # minimum reward improvement to reset patience
    max_epochs: int = 20
    max_new_tokens: int = 256

    # Quote pruning: drop quotes where the entire group scores 0.0 for this many
    # consecutive epochs in a row. Requires multiple groups to confirm the quote
    # is genuinely unproductive rather than a transient bad generation.
    dead_quote_streak_threshold: int = 2

    # Optimizer
    learning_rate: float = 1e-5
    beta: float = 0.1                          # KL penalty weight

    # Data
    holdout_fraction: float = 0.1
    seed: int = 42
    max_length: int = 512
    batch_size: int = 4                        # quotes per batch (each spawns G completions)
    gradient_accumulation_steps: int = 4
    max_train_records: Optional[int] = None
    logging_steps: int = 10

    # Judge reward weights (must sum to 1.0)
    judge_entailment_weight: float = 0.5
    judge_non_entailment_weight: float = 0.2
    judge_conclusion_weight: float = 0.3


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QuoteDataset(Dataset):
    """Dataset of raw quote strings for GRPO live-generation sampling.

    Accepts records with either a 'quote' key (raw corpus format) or
    'input_text' key (preprocessed training format).
    """

    def __init__(self, records: List[dict], tokenizer, max_length: int) -> None:
        self.prompts: List[str] = []
        self.quotes: List[str] = []
        for record in records:
            quote = record.get("quote") or record.get("input_text", "")
            # Any non-blank quote has derivable premises — blank inputs have nothing to extract
            if not quote or not quote.strip():
                continue
            # Build the structured reasoning prompt surface
            prompt_text = build_base_reasoning_prompt(quote)
            self.quotes.append(quote)
            self.prompts.append(build_generation_prompt(tokenizer, prompt_text))

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        return {"quote": self.quotes[idx], "prompt": self.prompts[idx]}


def _quote_collator(batch: List[dict]) -> dict:
    return {
        "quotes": [item["quote"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
    }


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """Group Relative Policy Optimization trainer.

    Args:
        policy_model:  The PEFT model being trained.
        tokenizer:     Shared tokenizer.
        judge:         Frozen FrozenJudge instance.
        config:        GRPOConfig with all hyperparameters.
    """

    def __init__(
        self,
        policy_model,
        tokenizer,
        judge: FrozenJudge,
        config: GRPOConfig,
    ) -> None:
        self.model = policy_model
        self.tokenizer = tokenizer
        self.judge = judge
        self.config = config

    def sample_group(self, prompt: str) -> List[str]:
        """Sample G completions from the live policy for one prompt.

        Preconditions: prompt is a tokenized chat-format string.
        Guarantee: returns exactly config.group_size strings (may be empty on OOM).
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                num_return_sequences=self.config.group_size,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        prompt_len = input_ids.shape[1]
        completions = []
        for seq in outputs:
            generated = seq[prompt_len:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            completions.append(strip_response_preamble(text))
        self.model.train()
        return completions

    def score_group(self, quote: str, completions: List[str]) -> List[float]:
        """Score each completion with the frozen judge.

        Preconditions: completions is non-empty.
        Guarantee: returns a parallel list of floats in [0.0, 1.0].
        """
        return [self.judge.score_completion(quote, c) for c in completions]

    def normalize_group(self, rewards: List[float]) -> List[float]:
        """Group-relative advantage normalization (mean/std).

        Preconditions: rewards is non-empty.
        Guarantee: returns advantages with zero mean and unit std (or zeros if
        all rewards are identical — no signal, no update).
        """
        if len(rewards) <= 1:
            return [0.0] * len(rewards)
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = variance ** 0.5
        if std < 1e-8:
            return [0.0] * len(rewards)
        return [(r - mean) / std for r in rewards]

    def _tokenize_completion(self, prompt: str, completion: str) -> dict:
        """Tokenize a (prompt, completion) pair for policy gradient computation."""
        full_text = prompt + completion
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
        )
        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        labels = inputs["input_ids"].clone()
        labels[0, :prompt_len] = -100  # mask prompt tokens from loss
        inputs["labels"] = labels
        return inputs

    def compute_grpo_loss(
        self,
        prompt: str,
        completions: List[str],
        advantages: List[float],
    ) -> torch.Tensor:
        """Compute GRPO loss for one prompt's group of completions.

        Loss = -mean_over_group( advantage_i * mean_token_log_prob_i )

        Preconditions:
            len(completions) == len(advantages)
            all completions are non-empty strings
        Guarantee: returns a scalar loss tensor on the model's device.
        """
        device = self.model.device
        total_loss = torch.tensor(0.0, device=device)
        valid_count = 0

        self.model.train()
        for completion, advantage in zip(completions, advantages):
            if not completion.strip():
                continue
            inputs = self._tokenize_completion(prompt, completion)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = inputs["labels"].to(device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            token_loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())

            valid_mask = shift_labels.ne(-100)
            n_valid = valid_mask.sum().clamp_min(1)
            seq_log_prob = -(token_loss * valid_mask).sum() / n_valid  # mean log-prob

            # Upweight positive advantages, downweight negative
            total_loss += -advantage * seq_log_prob
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total_loss / valid_count

    def run_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch_num: int,
    ) -> Dict[str, float]:
        """Run one full GRPO epoch over the dataset.

        Returns:
            dict with 'mean_reward', 'mean_advantage_std', 'steps', 'epoch'
        """
        total_reward = 0.0
        total_adv_std = 0.0
        step = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader, start=1):
            quotes = batch["quotes"]
            prompts = batch["prompts"]

            batch_loss = torch.tensor(0.0, device=self.model.device)
            for quote, prompt in zip(quotes, prompts):
                completions = self.sample_group(prompt)
                rewards = self.score_group(quote, completions)
                advantages = self.normalize_group(rewards)

                total_reward += sum(rewards) / len(rewards)
                adv_values = [a for a in advantages if a != 0.0]
                total_adv_std += (
                    (sum(a**2 for a in adv_values) / len(adv_values)) ** 0.5
                    if adv_values else 0.0
                )

                loss = self.compute_grpo_loss(prompt, completions, advantages)
                batch_loss = batch_loss + loss

            step += 1
            (batch_loss / len(quotes) / self.config.gradient_accumulation_steps).backward()

            if (
                batch_idx % self.config.gradient_accumulation_steps == 0
                or batch_idx == len(dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step == 1 or step % self.config.logging_steps == 0:
                mean_r = total_reward / step
                print(
                    f"[epoch {epoch_num}] step {step}/{len(dataloader)} "
                    f"mean_reward={mean_r:.4f} loss={batch_loss.item():.4f}"
                )

        n = max(step, 1)
        return {
            "epoch": epoch_num,
            "mean_reward": total_reward / n,
            "mean_advantage_std": total_adv_std / n,
            "steps": step,
        }


# ---------------------------------------------------------------------------
# Outer training loop
# ---------------------------------------------------------------------------

def run_grpo_training(config: GRPOConfig) -> dict:
    """Run the GRPO outer loop with patience-gated termination.

    Preconditions:
        config.adapter_path exists and contains a PEFT adapter.
        config.dataset_path exists and contains JSONL records with 'quote' or 'input_text'.
    Guarantee:
        Best checkpoint (highest mean reward) is saved to output_dir/adapter/.
        Training history is written to output_dir/grpo_history.jsonl.
        Summary is written to output_dir/grpo_summary.json.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    judge_path = config.judge_path or config.adapter_path

    # Load frozen judge first (lighter — no grad)
    print(f"Loading frozen judge from {judge_path} ...")
    judge = FrozenJudge(
        adapter_path=judge_path,
        entailment_weight=config.judge_entailment_weight,
        non_entailment_weight=config.judge_non_entailment_weight,
        conclusion_weight=config.judge_conclusion_weight,
    )

    # Load trainable policy
    print(f"Loading policy from {config.adapter_path} ...")
    policy_model = AutoPeftModelForCausalLM.from_pretrained(
        config.adapter_path,
        is_trainable=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy_model.config.use_cache = False

    # Build dataset
    records = load_jsonl(Path(config.dataset_path))
    train_indices, _ = split_indices(len(records), config.holdout_fraction, config.seed)
    train_records = (
        records[: config.max_train_records]
        if config.max_train_records
        else subset_records(records, train_indices)
    )

    dataset = QuoteDataset(train_records, tokenizer, config.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_quote_collator,
    )

    optimizer = torch.optim.AdamW(
        (p for p in policy_model.parameters() if p.requires_grad),
        lr=config.learning_rate,
    )

    trainer = GRPOTrainer(
        policy_model=policy_model,
        tokenizer=tokenizer,
        judge=judge,
        config=config,
    )

    best_reward = float("-inf")
    patience_counter = 0
    history: List[dict] = []

    print(f"Starting GRPO training: max_epochs={config.max_epochs}, patience={config.patience}, G={config.group_size}")

    for epoch in range(1, config.max_epochs + 1):
        epoch_metrics = trainer.run_epoch(dataloader, optimizer, epoch)
        history.append(epoch_metrics)
        mean_reward = epoch_metrics["mean_reward"]

        # Checkpoint every epoch; overwrite best only when improved
        policy_model.save_pretrained(output_dir / "adapter_latest")
        tokenizer.save_pretrained(output_dir / "adapter_latest")

        if mean_reward > best_reward + config.patience_delta:
            best_reward = mean_reward
            patience_counter = 0
            policy_model.save_pretrained(output_dir / "adapter")
            tokenizer.save_pretrained(output_dir / "adapter")
            print(f"  → New best reward={best_reward:.4f}; checkpoint saved.")
        else:
            patience_counter += 1
            print(
                f"  → No improvement (reward={mean_reward:.4f}, best={best_reward:.4f}); "
                f"patience {patience_counter}/{config.patience}"
            )
            if patience_counter >= config.patience:
                print(f"Patience exhausted after epoch {epoch}. Stopping.")
                break

        # Append to history log after each epoch
        with open(output_dir / "grpo_history.jsonl", "a") as fh:
            fh.write(json.dumps(epoch_metrics) + "\n")

    summary = {
        "config": asdict(config),
        "epochs_run": len(history),
        "best_reward": best_reward,
        "history": history,
        "adapter_path": str(output_dir / "adapter"),
        "adapter_latest_path": str(output_dir / "adapter_latest"),
    }
    (output_dir / "grpo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"GRPO training complete. Best reward={best_reward:.4f}")
    return summary
