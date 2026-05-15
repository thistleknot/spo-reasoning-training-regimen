"""Soft Policy Optimization (SPO) training for reasoning models.

SPO uses model confidence as a signal: reward high-confidence correct outputs,
penalize low-confidence or incorrect outputs. This teaches the model to assign
accurate confidence to its own predictions.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .span_extractor import find_span


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


@dataclass
class PromptContract:
    """Output format contract derived from a training or inference prompt.

    Parses the numbered section list from a prompt's 'Generate a response with:'
    block to determine which section headers the model is expected to produce.
    This decouples the quality gate from hardcoded header lists — changing the
    prompt automatically changes what the evaluator checks for.

    Require:  prompt contains a 'Generate a response with:' block with numbered items
    Guarantee: expected_headers is a list of header strings ending with ':'
    Maintain:  empty expected_headers means no header constraint (score=1.0 by default)
    """

    expected_headers: List[str] = field(default_factory=list)

    @classmethod
    def from_prompt(cls, prompt: str) -> "PromptContract":
        """Parse expected section headers from the numbered list in a prompt.

        Extracts items from a block like:
            Generate a response with:
            1. Non-Entailed Premises
            2. Entailed Premises
            3. Throughline

        Each item becomes a header string with ':' appended, matching the
        format the model is expected to output.
        """
        match = re.search(
            r"Generate a response with:\s*\n((?:\d+\.\s*.+\n?)+)",
            prompt,
        )
        if not match:
            return cls(expected_headers=[])
        section_block = match.group(1)
        headers = []
        for line in section_block.splitlines():
            m = re.match(r"\d+\.\s+(.+)", line.strip())
            if m:
                headers.append(m.group(1).strip() + ":")
        return cls(expected_headers=headers)

    def header_score(self, model_output: str) -> float:
        """Fraction of expected headers present in the output with exact match.

        A garbled or abbreviated header (e.g. 'Throughlin\\':') earns zero for
        that slot — the model attempted the header but got the spelling wrong,
        which is strictly worse than an absent optional section.

        Returns 1.0 when expected_headers is empty (no contract → no penalty).
        """
        if not self.expected_headers:
            return 1.0
        header_lines = {
            line.strip()
            for line in model_output.splitlines()
            if line.strip().endswith(":")
            and "|" not in line
            and len(line.strip()) <= 60
        }
        hits = sum(1 for h in self.expected_headers if h in header_lines)
        return hits / len(self.expected_headers)


def assert_output_quality(
    outputs: List[str],
    prompts: Optional[List[str]] = None,
    min_avg_score: float = 0.5,
    min_per_sample_score: float = 0.3,
    min_header_score: float = 0.5,
) -> None:
    """Regression gate: raises AssertionError if generated outputs fall below thresholds.

    Called after SPO training to verify the adapter produces outputs that satisfy
    the prompt contract. A failing gate means the RL training has regressed output
    quality below an acceptable floor.

    Header scoring is a *hard* separate gate — wrong or abbreviated section headers
    (e.g. 'Entailed Prims:' instead of 'Entailed Premises:') fail this check even
    when pipe structure and confidence annotations look fine. This prevents the
    0.35 header weight in evaluate_triplet_correctness from being compensated by
    other dimensions at a permissive avg threshold.

    Require:
        outputs is non-empty
        min_avg_score, min_per_sample_score, min_header_score are in [0, 1]
    Guarantee:
        Raises AssertionError with per-sample details on any quality failure
        Returns None (no exception) when all outputs pass all thresholds
    """
    if not outputs:
        raise AssertionError("assert_output_quality: outputs list is empty")

    scores = []
    header_scores = []
    per_sample_failures = []
    for i, output in enumerate(outputs):
        contract = PromptContract.from_prompt(prompts[i]) if prompts else None
        score = SPOEvaluator.evaluate_triplet_correctness(output, contract=contract)
        scores.append(score)
        hscore = contract.header_score(output) if contract and contract.expected_headers else 1.0
        header_scores.append(hscore)
        if score < min_per_sample_score:
            preview = output.replace("\n", " ")[:120]
            per_sample_failures.append((i, score, preview))

    avg_score = sum(scores) / len(scores)
    avg_header_score = sum(header_scores) / len(header_scores)

    if per_sample_failures:
        lines = [
            f"  sample {i}: score={s:.3f} — {preview!r}"
            for i, s, preview in per_sample_failures
        ]
        raise AssertionError(
            f"Output quality gate: {len(per_sample_failures)}/{len(outputs)} samples "
            f"below per-sample floor {min_per_sample_score:.2f}\n" + "\n".join(lines)
        )

    if avg_score < min_avg_score:
        raise AssertionError(
            f"Output quality gate: avg_score={avg_score:.3f} below "
            f"avg floor {min_avg_score:.2f} across {len(outputs)} samples"
        )

    if avg_header_score < min_header_score:
        bad = [
            f"  sample {i}: header_score={h:.2f} — expected {c.expected_headers}"
            for i, (h, c) in enumerate(
                zip(
                    header_scores,
                    [
                        PromptContract.from_prompt(prompts[j]) if prompts else PromptContract()
                        for j in range(len(outputs))
                    ],
                )
            )
            if h < min_header_score
        ]
        raise AssertionError(
            f"Header quality gate: avg_header_score={avg_header_score:.3f} below "
            f"header floor {min_header_score:.2f} — model is generating garbled or "
            f"abbreviated section headers\n" + "\n".join(bad)
        )


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
        Values outside [0, 1] are clamped so hallucinated values (1.5, -0.2) do
        not inflate the SPO reward signal.
        """
        matches = re.findall(r"confidence\s*=\s*([0-9]*\.?[0-9]+)", output)
        if matches:
            values = [max(0.0, min(1.0, float(m))) for m in matches]
            return sum(values) / len(values)

        section_match = re.search(
            r"Confidence:\s*(?:\n\s*)?([0-9]*\.?[0-9]+)",
            output,
            flags=re.IGNORECASE,
        )
        if section_match:
            return max(0.0, min(1.0, float(section_match.group(1))))

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

    # Canonical section headers required in the output, in order.
    # These must match the training data format exactly — any deviation is a format error.
    CANONICAL_HEADERS: list = [
        "Non-Entailed Premises:",
        "Entailed Premises:",
        "Throughline:",
    ]

    @staticmethod
    def _header_score(model_output: str) -> float:
        """Score section header correctness.

        Identifies all header-like lines (short, ends with ':', no pipe chars) and
        checks how many canonical headers are present with exact spelling.

        A garbled header (e.g. 'Throughlin':' or 'Entailed Prims:') counts as a
        header attempt that failed — it is treated as 0 credit for that slot while
        occupying the slot (i.e. the model tried but got it wrong, not that it was
        absent). This is strictly harsher than absence because absence is neutral
        (the section might be optional) whereas a misspelled header signals the model
        doesn't know the correct label.

        Returns fraction of canonical headers present with exact match [0.0-1.0].
        """
        canonical = SPOEvaluator.CANONICAL_HEADERS
        lines = model_output.splitlines()
        # A header-like line: stripped, ends with ':', no '|', length <= 60
        header_lines = {
            line.strip()
            for line in lines
            if line.strip().endswith(":")
            and "|" not in line
            and len(line.strip()) <= 60
        }
        hits = sum(1 for h in canonical if h in header_lines)
        return hits / len(canonical)

    @staticmethod
    def _extract_section_triplets(model_output: str, section_name: str) -> List[str]:
        """Return the triplet lines belonging to a named section in model_output.

        Scans line-by-line from the first occurrence of ``section_name:`` until
        the next header-like line (ends with ':', no '|', length ≤ 60) or EOF.

        Require: model_output is a non-empty string; section_name is the header
                 text without the trailing colon (e.g. "Entailed Premises").
        Guarantee: returns a list of stripped triplet lines (containing '|') from
                   that section only; empty list when the section is absent.
        """
        triplet_re = re.compile(r"[^|]+\s*\|\s*[^|]+\s*\|\s*[^|]+")
        lines = model_output.splitlines()
        in_section = False
        result: List[str] = []
        target_header = (section_name + ":").lower()
        for line in lines:
            stripped = line.strip()
            is_header = (
                stripped.endswith(":")
                and "|" not in stripped
                and len(stripped) <= 60
            )
            if is_header:
                if stripped.lower() == target_header:
                    in_section = True
                else:
                    in_section = False
                continue
            if in_section and triplet_re.search(stripped):
                result.append(stripped)
        return result

    @staticmethod
    def _entailed_verbatim_ratio(
        entailed_triplet_lines: List[str],
        source_quote: str,
    ) -> float:
        """Fraction of entailed triplet subject/object fields that are verbatim spans of source_quote.

        For each triplet line, the subject (field 0) and object (field 2) are
        extracted.  Parenthetical transliterations of the form ``term (clarification)``
        are stripped so only the base term is checked against source_quote via
        ``find_span()``.  A component counts as verbatim when ``find_span`` returns
        a non-None span (coverage ≥ 0.5 match, per span_extractor contract).

        Require:
            entailed_triplet_lines is a list of '|'-delimited triplet strings.
            source_quote is non-empty.
        Guarantee:
            Returns a float in [0, 1].  Returns 0.0 when the input list is empty
            or source_quote is blank.  Returns 1.0 when all checked components
            are verbatim.
        Failure modes:
            Single-token subjects/objects that happen to be common English words
            may match spuriously.  This is acceptable — false positives are benign
            (over-reward is bounded by the 0.05 weight).
        """
        if not entailed_triplet_lines or not source_quote.strip():
            return 0.0

        _paren_re = re.compile(r"\s*\(.*?\)\s*$")

        def _base_text(field: str) -> str:
            """Strip parenthetical transliteration and surrounding whitespace."""
            return _paren_re.sub("", field).strip()

        checked = 0
        verbatim = 0
        for line in entailed_triplet_lines:
            parts = line.split("|")
            if len(parts) < 3:
                continue
            for field_idx in (0, 2):  # subject and object only
                raw = parts[field_idx]
                base = _base_text(raw)
                if not base:
                    continue
                checked += 1
                if find_span(source_quote, base) is not None:
                    verbatim += 1

        return verbatim / checked if checked > 0 else 0.0

    @staticmethod
    def evaluate_triplet_correctness(
        model_output: str,
        ground_truth: Optional[str] = None,
        contract: Optional["PromptContract"] = None,
        source_quote: Optional[str] = None,
    ) -> float:
        """Evaluate correctness of triplet-based output.

        Scores based on:
        - Triplet format preservation (subject | relation | object) [0.20 weight]
        - Section header correctness against contract or CANONICAL_HEADERS [0.30 weight]
        - Confidence annotation validity [0.15 weight, proportional]:
          ratio of confidence=X values where X is numeric and in [0,1].
          Out-of-range (1.5, -0.2), comma-decimal (0,5), and tuple forms reduce score.
        - Evidence tag validity [0.15 weight, proportional]:
          ratio of annotation tags that are exactly 'observed' or 'inferred'.
          Misspellings (obsined), compound forms (observed/derived), and synonyms
          (inference, observable, obscured) all count as invalid.
        - Uniqueness ratio: hard-zero if > half the triplet lines are duplicates [gate]
        - Tautology penalty: subject appears in object field [0.04 weight]
        - Trivial-predicate penalty: bare copula ('is'/'are'/…) as sole predicate [0.04 weight]
        - Predicate-echo-in-object penalty: object starts with predicate token [0.04 weight]
        - Degenerate-subject penalty: empty or numeric-only subject (e.g. "1.") [0.03 weight]
        - Ground-truth overlap via SequenceMatcher if provided [0.05 bonus, capped at 1.0]
        - Entailed-section verbatim faithfulness when source_quote provided [0.15 penalty, floor 0.0]

        Section-targeted faithfulness gate:
        When ``source_quote`` is provided the Entailed Premises section is parsed
        separately.  Subject and object fields (after stripping any parenthetical
        transliterations) are checked against the source quote via ``find_span()``.
        The fraction of non-verbatim components subtracts up to 0.15 from the score (floored at 0.0).  Non-
        entailed premises and the throughline are deliberately excluded — they are
        synthesised context, not extractive claims.

        When a PromptContract is provided its expected_headers drive the header
        score. This prevents the evaluator drifting out of sync with the prompt.

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

        # Uniqueness gate: hard-zero if more than half of all triplet lines are duplicates
        if triplet_lines:
            unique_ratio = len(set(triplet_lines)) / len(triplet_lines)
            if unique_ratio < 0.5:
                return 0.0

        score = 0.0

        # Triplet format check [0.20]
        if triplet_lines:
            score += 0.20

        # Section header correctness [0.30] — use contract when provided, else CANONICAL_HEADERS
        if contract is not None:
            score += 0.30 * contract.header_score(model_output)
        else:
            score += 0.30 * SPOEvaluator._header_score(model_output)

        # Confidence annotation [0.15] — proportional: only values parseable as
        # float in [0, 1] count.  Comma decimals (0,5), ranges ((-0.5, 0.5)),
        # and out-of-range values (1.5, -0.2) are all invalid and reduce the score.
        _CONF_VAL_RE = re.compile(r"confidence\s*=\s*([^\s,)\n]+)")

        def _is_valid_confidence(v: str) -> bool:
            try:
                fv = float(v)
                return 0.0 <= fv <= 1.0
            except ValueError:
                return False

        conf_strs = _CONF_VAL_RE.findall(model_output)
        if conf_strs:
            valid_conf = sum(1 for v in conf_strs if _is_valid_confidence(v))
            score += 0.15 * (valid_conf / len(conf_strs))

        # Evidence tags [0.15] — proportional: tag must be exactly 'observed' or
        # 'inferred' (case-insensitive).  Compound forms (observed/derived),
        # misspellings (obsined, obsERVED), and synonyms (inference, observable)
        # all count as invalid and reduce the per-triplet score.
        _TAG_ANNOT_RE = re.compile(r"\(\s*([\w/]+)\s*,\s*confidence\s*=", re.IGNORECASE)
        _VALID_TAGS = {"observed", "inferred"}
        all_tags = [
            m.group(1).strip().lower()
            for line in triplet_lines
            for m in _TAG_ANNOT_RE.finditer(line)
        ]
        if all_tags:
            valid_tag_count = sum(1 for t in all_tags if t in _VALID_TAGS)
            score += 0.15 * (valid_tag_count / len(all_tags))

        # Tautology penalty: subject appears as a substring of the object field,
        # e.g. "the speaker | is ... | is the speaker" or "x | is | x itself" [0.04]
        def _is_tautological(line: str) -> bool:
            parts = [p.strip().lower() for p in line.split("|")]
            if len(parts) < 3:
                return False
            subject = parts[0].strip()
            obj = re.sub(r"\(.*?\)", "", parts[2]).strip()
            return len(subject) >= 3 and subject in obj

        # Trivial-predicate penalty: bare copula ('is', 'are', 'was' …) as the
        # entire predicate adds no semantic content and is a known failure mode
        # where the model copies prompt phrasing rather than extracting a relation. [0.04]
        _TRIVIAL_PREDICATES = {"is", "are", "was", "were", "has", "have", "had", "be"}

        def _is_trivial_predicate(line: str) -> bool:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                return False
            pred_clean = re.sub(r"\(.*?\)", "", parts[1]).strip().lower()
            return pred_clean in _TRIVIAL_PREDICATES

        # Predicate-echo-in-object penalty: object field starts with the same
        # token as the predicate (e.g. "x | is | is the correct action"), which
        # indicates a circular definition where the relation is repeated instead
        # of naming a distinct object. [0.04]
        def _object_echoes_predicate(line: str) -> bool:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                return False
            pred_words = re.sub(r"\(.*?\)", "", parts[1]).strip().lower().split()
            obj_words = re.sub(r"\(.*?\)", "", parts[2]).strip().lower().split()
            return bool(pred_words and obj_words and pred_words[0] == obj_words[0])

        # Degenerate-subject penalty: subject field is empty or a bare number
        # (e.g. "1. | is | ..."), which means the model failed to extract a
        # meaningful entity as the triplet subject. [0.03]
        def _has_degenerate_subject(line: str) -> bool:
            parts = [p.strip() for p in line.split("|")]
            if not parts:
                return True
            subject = re.sub(r"\(.*?\)", "", parts[0]).strip()
            return len(subject) == 0 or bool(re.match(r"^\d+\.?\s*$", subject))

        if triplet_lines:
            taut_ratio    = sum(1 for t in triplet_lines if _is_tautological(t))      / len(triplet_lines)
            trivial_ratio = sum(1 for t in triplet_lines if _is_trivial_predicate(t)) / len(triplet_lines)
            echo_ratio    = sum(1 for t in triplet_lines if _object_echoes_predicate(t)) / len(triplet_lines)
            degen_ratio   = sum(1 for t in triplet_lines if _has_degenerate_subject(t))  / len(triplet_lines)
            score += 0.04 * (1.0 - taut_ratio)
            score += 0.04 * (1.0 - trivial_ratio)
            score += 0.04 * (1.0 - echo_ratio)
            score += 0.03 * (1.0 - degen_ratio)

        # Ground-truth overlap bonus [0.05 max]
        if ground_truth and ground_truth.strip():
            import difflib
            overlap = difflib.SequenceMatcher(
                None, model_output.strip(), ground_truth.strip()
            ).ratio()
            score += 0.05 * overlap

        # Entailed-section verbatim faithfulness [0.15 penalty for non-verbatim].
        # Hard requirement: subject and object of EVERY Entailed premise must be
        # a verbatim span of the source quote (parenthetical transliterations are
        # stripped before checking).  non-verbatim_ratio=0 → no penalty;
        # non-verbatim_ratio=1 → -0.15.  Score floor is 0.0 (never negative).
        if source_quote and source_quote.strip():
            entailed_lines = SPOEvaluator._extract_section_triplets(
                model_output, "Entailed Premises"
            )
            verbatim_ratio = SPOEvaluator._entailed_verbatim_ratio(
                entailed_lines, source_quote
            )
            score -= 0.15 * (1.0 - verbatim_ratio)

        return max(0.0, min(score, 1.0))

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
        source_quote: Optional[str] = None,
    ) -> float:
        """Composite correctness score combining multiple evaluations.

        Args:
            model_output: Model's output string
            ground_truth: Optional ground truth for comparison
            weights: Dict of weights for each metric
                - "triplet": weight for triplet correctness
                - "syllogism": weight for syllogism quality
            source_quote: Optional original quote used to check verbatim faithfulness
                of entailed premises (passed through to evaluate_triplet_correctness).

        Returns:
            Composite score 0.0-1.0
        """
        if weights is None:
            weights = {"triplet": 0.6, "syllogism": 0.4}

        triplet_score = SPOEvaluator.evaluate_triplet_correctness(
            model_output, ground_truth, source_quote=source_quote
        )
        syllogism_score = SPOEvaluator.evaluate_syllogism_quality(
            model_output, ground_truth
        )

        return (
            weights["triplet"] * triplet_score + weights["syllogism"] * syllogism_score
        )
