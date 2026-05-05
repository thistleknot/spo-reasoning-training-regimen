"""Soft Policy Optimization (SPO) training for reasoning models.

SPO uses model confidence as a signal: reward high-confidence correct outputs,
penalize low-confidence or incorrect outputs. This teaches the model to assign
accurate confidence to its own predictions.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch


@dataclass
class SPOReward:
    """Reward signal for SPO training."""
    correctness: float  # 0.0-1.0: is the output correct?
    confidence: float   # 0.0-1.0: model's confidence in output
    reward: float       # correctness * confidence

    def __init__(self, correctness: float, confidence: float):
        self.correctness = correctness
        self.confidence = confidence
        self.reward = correctness * confidence


class SPOTrainer:
    """Soft Policy Optimization trainer for reasoning models."""

    def __init__(
        self,
        model,
        tokenizer,
        evaluation_fn: Callable[[str, str], float],
        learning_rate: float = 2e-4,
        beta: float = 0.1,
    ):
        """Initialize SPO trainer.

        Args:
            model: The reasoning model to train
            tokenizer: Tokenizer for the model
            evaluation_fn: Function that scores output correctness (0.0-1.0)
            learning_rate: Learning rate for optimization
            beta: KL divergence penalty weight
        """
        self.model = model
        self.tokenizer = tokenizer
        self.evaluation_fn = evaluation_fn
        self.learning_rate = learning_rate
        self.beta = beta
        self.training_history = []

    def extract_confidence(self, output: str) -> float:
        """Extract confidence score from model output.

        Looks for either triplet confidence annotations (`confidence=0.8`) or an
        explicit `Confidence:` section in score-bearing outputs.
        """
        matches = re.findall(r"confidence\s*=\s*([0-9]*\.?[0-9]+)", output)
        if matches:
            # Return average confidence
            return sum(float(m) for m in matches) / len(matches)

        section_match = re.search(
            r"Confidence:\s*(?:\n\s*)?([0-9]*\.?[0-9]+)",
            output,
            flags=re.IGNORECASE,
        )
        if section_match:
            return float(section_match.group(1))

        return 0.5  # Default

    def evaluate_batch(
        self,
        inputs: list,
        outputs: list,
        ground_truths: Optional[list] = None,
    ) -> Tuple[list, Dict[str, float]]:
        """Evaluate a batch of outputs and compute rewards.

        Args:
            inputs: List of input prompts
            outputs: List of model outputs
            ground_truths: Optional list of ground truth outputs for comparison

        Returns:
            Tuple of (rewards, metrics)
        """
        if not outputs:
            return [], {
                "avg_correctness": 0.0,
                "avg_confidence": 0.0,
                "avg_reward": 0.0,
            }

        rewards = []
        metrics = {
            "avg_correctness": 0.0,
            "avg_confidence": 0.0,
            "avg_reward": 0.0,
        }

        for output, ground_truth in zip(outputs, ground_truths or [None] * len(outputs)):
            # Evaluate correctness
            correctness = self.evaluation_fn(output, ground_truth)

            # Extract model confidence
            confidence = self.extract_confidence(output)

            # Compute reward
            reward = SPOReward(correctness, confidence)
            rewards.append(reward)

            metrics["avg_correctness"] += correctness
            metrics["avg_confidence"] += confidence
            metrics["avg_reward"] += reward.reward

        # Normalize metrics
        n = len(outputs)
        metrics["avg_correctness"] /= n
        metrics["avg_confidence"] /= n
        metrics["avg_reward"] /= n

        return rewards, metrics

    def compute_loss(
        self,
        logits: torch.Tensor,
        rewards: list,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SPO loss.

        Loss = -E[reward * log p(output)]
        This upweights gradients for high-reward outputs.

        Args:
            logits: Model logits
            rewards: List of SPOReward objects
            labels: Ground truth token ids

        Returns:
            Loss tensor
        """
        if logits.ndim != 3:
            raise ValueError("logits must have shape [batch, seq_len, vocab_size].")
        if labels.ndim != 2:
            raise ValueError("labels must have shape [batch, seq_len].")
        if logits.size(0) != len(rewards):
            raise ValueError("reward count must match batch size.")

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        valid_mask = shift_labels.ne(-100)
        token_loss = token_loss * valid_mask
        tokens_per_sequence = valid_mask.sum(dim=1).clamp_min(1)
        sequence_loss = token_loss.sum(dim=1) / tokens_per_sequence

        # Weight by reward
        reward_weights = torch.tensor(
            [r.reward for r in rewards],
            device=logits.device,
            dtype=logits.dtype,
        ).clamp_min(0.0)

        # SPO loss: upweight high-reward sequences.
        weighted_loss = (sequence_loss * reward_weights).mean()

        return weighted_loss

    def compute_step(
        self,
        batch: Dict[str, Any],
        ground_truths: Optional[list[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute one SPO optimization step.

        When the batch includes pre-computed quality scores (`precomputed_rewards`),
        those are used directly as reward weights instead of calling evaluate_batch on
        gold outputs vs themselves, which would produce uniform rewards and degrade SPO
        to plain SFT.

        Args:
            batch: Dict with `input_ids`, `attention_mask`, `labels`, `output_texts`,
                and optionally `precomputed_rewards` (a float tensor of shape [batch]).
            ground_truths: Optional list of reference outputs for fallback evaluation.

        Returns:
            Tuple of `(loss, metrics)` for this step.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        output_texts = batch.get("output_texts", [])
        precomputed_rewards: Optional[torch.Tensor] = batch.get("precomputed_rewards")

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )

        if precomputed_rewards is not None and len(precomputed_rewards) > 0:
            # Use pre-scored data-quality weights directly — avoids gold-vs-gold evaluation.
            reward_values = precomputed_rewards.tolist()
            rewards = [SPOReward(correctness=r, confidence=r) for r in reward_values]
            metrics = {
                "avg_correctness": float(precomputed_rewards.mean()),
                "avg_confidence": float(precomputed_rewards.mean()),
                "avg_reward": float(precomputed_rewards.mean()),
            }
        else:
            # Fallback: evaluate gold outputs against ground_truths
            rewards, metrics = self.evaluate_batch(
                inputs=[],
                outputs=output_texts,
                ground_truths=ground_truths or batch.get("ground_truths"),
            )

        # Compute loss
        if rewards and output_texts:
            loss = self.compute_loss(
                outputs.logits,
                rewards,
                labels,
            )
            metrics["loss"] = loss.item()
        else:
            loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=input_ids.device)
            metrics["loss"] = loss.item()

        self.training_history.append(metrics.copy())
        return loss, metrics

    def training_step(
        self,
        batch: Dict[str, Any],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compatibility wrapper returning metrics for a single SPO step."""
        ground_truths = [ground_truth] if ground_truth is not None else None
        _, metrics = self.compute_step(batch, ground_truths=ground_truths)
        return metrics

    def save_history(self, path: str):
        """Save training history to JSON."""
        with open(path, "w") as f:
            json.dump(self.training_history, f, indent=2)

    def load_history(self, path: str):
        """Load training history from JSON."""
        with open(path) as f:
            self.training_history = json.load(f)


class SPOEvaluator:
    """Evaluate model outputs for SPO reward computation."""

    @staticmethod
    def evaluate_triplet_correctness(
        model_output: str,
        ground_truth: Optional[str] = None,
    ) -> float:
        """Evaluate correctness of triplet-based output.

        Scores based on:
        - Triplet format preservation (subject | relation | object) [0.4 weight]
        - Confidence score presence [0.2 weight]
        - Evidence tag correctness (observed vs inferred) [0.2 weight]
        - Uniqueness ratio: hard-zero if > half the triplet lines are duplicates [gate]
        - Non-self-referential content: deduct if tautological triplets dominate [0.1 weight]
        - Ground-truth overlap via SequenceMatcher if provided [0.1 bonus]

        Returns score 0.0-1.0
        """
        if not model_output or not model_output.strip():
            return 0.0

        triplet_pattern = re.compile(r"[^|]+\s*\|\s*[^|]+\s*\|\s*[^|]+")
        triplet_lines = [
            line.strip()
            for line in model_output.splitlines()
            if triplet_pattern.search(line)
        ]

        # Uniqueness gate: heavy penalty for repetitive outputs
        if triplet_lines:
            unique_ratio = len(set(triplet_lines)) / len(triplet_lines)
            if unique_ratio < 0.5:
                return 0.0

        score = 0.0

        # Format check
        if triplet_lines:
            score += 0.4

        # Confidence annotation
        if re.search(r"confidence=", model_output):
            score += 0.2

        # Evidence tags
        if re.search(r"\b(observed|inferred)\b", model_output):
            score += 0.2

        # Tautology penalty: subject appears as a substring of the object field,
        # e.g. "the speaker | is ... | is the speaker" or "x | is | x itself"
        def _is_tautological(line: str) -> bool:
            parts = [p.strip().lower() for p in line.split("|")]
            if len(parts) < 3:
                return False
            subject = parts[0].strip()
            # Strip parenthetical annotations from the object part before comparing
            obj = re.sub(r"\(.*?\)", "", parts[2]).strip()
            return len(subject) >= 3 and subject in obj

        if triplet_lines:
            taut_ratio = sum(1 for t in triplet_lines if _is_tautological(t)) / len(triplet_lines)
            # Add up to 0.1 when tautology rate is zero, subtract when high
            score += 0.1 * (1.0 - taut_ratio)

        # Ground-truth overlap bonus (0.1)
        if ground_truth and ground_truth.strip():
            import difflib
            overlap = difflib.SequenceMatcher(
                None, model_output.strip(), ground_truth.strip()
            ).ratio()
            score += 0.1 * overlap

        return min(score, 1.0)

    @staticmethod
    def evaluate_syllogism_quality(
        model_output: str,
        ground_truth: Optional[str] = None,
    ) -> float:
        """Evaluate quality of syllogism/reasoning text.

        Scores based on:
        - Presence of reasoning (not empty or just N/A)
        - Coherence (no obvious repetition or truncation)
        - Length (meaningful reasoning, not too short)

        Returns score 0.0-1.0
        """
        score = 0.0

        # Has content
        if model_output and model_output.strip() not in ["N/A", "n/a", ""]:
            score += 0.33

        # Not overly repetitive
        lines = model_output.split("\n")
        unique_lines = len(set(lines))
        if unique_lines / max(len(lines), 1) > 0.6:
            score += 0.33

        # Reasonable length (at least 10 chars, less than 1000)
        if 10 <= len(model_output) <= 1000:
            score += 0.34

        return score

    @staticmethod
    def composite_score(
        model_output: str,
        ground_truth: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Composite correctness score combining multiple evaluations.

        Args:
            model_output: Model's output string
            ground_truth: Optional ground truth for comparison
            weights: Dict of weights for each metric
                - "triplet": weight for triplet correctness
                - "syllogism": weight for syllogism quality

        Returns:
            Composite score 0.0-1.0
        """
        if weights is None:
            weights = {"triplet": 0.6, "syllogism": 0.4}

        triplet_score = SPOEvaluator.evaluate_triplet_correctness(
            model_output, ground_truth
        )
        syllogism_score = SPOEvaluator.evaluate_syllogism_quality(
            model_output, ground_truth
        )

        return (
            weights["triplet"] * triplet_score + weights["syllogism"] * syllogism_score
        )
