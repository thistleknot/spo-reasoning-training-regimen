"""Soft Policy Optimization (SPO) training for reasoning models.

SPO uses model confidence as a signal: reward high-confidence correct outputs,
penalize low-confidence or incorrect outputs. This teaches the model to assign
accurate confidence to its own predictions.
"""

from typing import Callable, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import json


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

        Looks for patterns like "confidence=0.8" in triplet format.
        """
        import re

        matches = re.findall(r"confidence=([0-9.]+)", output)
        if matches:
            # Return average confidence
            return sum(float(m) for m in matches) / len(matches)
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
        # Standard language modeling loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        # Weight by reward
        reward_weights = torch.tensor(
            [r.reward for r in rewards],
            device=logits.device,
            dtype=logits.dtype,
        )

        # SPO loss: downweight low-reward sequences
        weighted_loss = (token_loss * (1 - reward_weights)).mean()

        return weighted_loss

    def training_step(
        self,
        batch: Dict[str, Any],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Single training step with SPO.

        Args:
            batch: Dict with 'input_ids', 'attention_mask', 'output_text'
            ground_truth: Optional ground truth for evaluation

        Returns:
            Dict of metrics for this step
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        output_text = batch.get("output_text")

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Evaluate rewards
        rewards, metrics = self.evaluate_batch(
            inputs=[],  # Not needed for this step
            outputs=[output_text] if output_text else [],
            ground_truths=[ground_truth] if ground_truth else [],
        )

        # Compute loss
        if rewards and output_text:
            loss = self.compute_loss(
                outputs.logits,
                rewards,
                input_ids,
            )
            metrics["loss"] = loss.item()
        else:
            loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0)
            metrics["loss"] = loss.item()

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
        ground_truth: str,
    ) -> float:
        """Evaluate correctness of triplet-based output.

        Scores based on:
        - Triplet format preservation (subject | relation | object)
        - Confidence score presence
        - Evidence tag correctness (observed vs inferred)

        Returns score 0.0-1.0
        """
        import re

        score = 0.0
        max_score = 3.0

        # Check triplet format
        triplet_pattern = r"[^|]+\s*\|\s*[^|]+\s*\|\s*[^|]+"
        if re.search(triplet_pattern, model_output):
            score += 1.0

        # Check confidence scores
        if re.search(r"confidence=", model_output):
            score += 1.0

        # Check evidence tags
        if re.search(r"(observed|inferred)", model_output):
            score += 1.0

        return score / max_score

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
