"""Frozen judge for GRPO reward scoring.

Loads a PEFT adapter with frozen weights and evaluates SPO completions via
decomposed binary entailment probes.  Each probe compares the log-probability
of the 'Yes' token against 'No' — no generation needed, which keeps scoring fast.

Two construction modes:

  Standalone (default):
    FrozenJudge(adapter_path=...)
    Loads its own model instance.  Simple but doubles VRAM vs the policy model.

  Shared-base (memory-efficient):
    FrozenJudge.from_shared_model(model, tokenizer, adapter_name="judge",
                                  policy_adapter_name="policy")
    The caller loads the base model once and mounts both adapters by name.
    The judge temporarily switches to its adapter via set_adapter() for each
    probe, then restores the policy adapter — keeping peak VRAM near
    one-model-cost rather than two.

    Memory estimate for a 0.8B model at 4-bit:
      Base weights   ≈ 400 MB
      LoRA adapter A (policy, trainable)  ≈  10 MB
      LoRA adapter B (judge, frozen)      ≈  10 MB
      Total                               ≈ 420 MB

Require:  adapter_path (standalone) or model+adapter_name (shared) must reference
          a valid PEFT adapter for the same base architecture.
Guarantee: all scoring methods return floats in [0.0, 1.0]; judge weights are
           never updated (no_grad + frozen params enforced at construction).
Maintain:  in shared mode, the policy adapter is always restored after each probe.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import List, Optional

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


# Binary probe templates
_ENTAILMENT_TEMPLATE = (
    'Quote: "{quote}"\n'
    'Premise: "{premise}"\n'
    "Does this premise follow directly from the quote? Answer Yes or No."
)

_NON_ENTAILMENT_TEMPLATE = (
    'Quote: "{quote}"\n'
    'Premise: "{premise}"\n'
    "Is this premise absent from — or not directly stated in — the quote? Answer Yes or No."
)

_CONCLUSION_TEMPLATE = (
    "Premises:\n{premises}\n\n"
    'Conclusion: "{conclusion}"\n'
    "Does this conclusion follow logically from the premises above? Answer Yes or No."
)

# Confidence rating template — generates a float between 0.0 and 1.0.
# Sampled K times at temperature > 0 to produce a quality distribution.
# High mean + low std → judge consistently rates this extraction highly.
# High std → judge is uncertain → margin case, weak signal.
_CONFIDENCE_TEMPLATE = (
    'Quote: "{quote}"\n\n'
    "SPO Extraction:\n{completion}\n\n"
    "Rate the quality of this extraction. A perfect extraction has:\n"
    "- at least one entailed premise that directly restates a key claim from the quote\n"
    "- at least one non-entailed premise that identifies what is implied but unsaid\n"
    "- a conclusion that follows from those premises\n"
    "Output a single decimal number between 0.0 (completely wrong) and 1.0 (perfect). "
    "No explanation."
)
_CONF_FLOAT_RE = re.compile(r"\b(1\.0+|0\.\d+|0|1)\b")

# Regex for parsing SPO completion sections
_SECTION_RE = re.compile(
    r"(?:Non-Entailed Premises:|Non-Entailed:)(.*?)"
    r"(?:Entailed Premises:|Entailed:)(.*?)"
    r"(?:Throughline:|Conclusion:)(.*?)$",
    re.DOTALL | re.IGNORECASE,
)
_TRIPLET_RE = re.compile(r"([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)")


def _extract_sections(completion: str) -> tuple[list[str], list[str], str]:
    """Parse a completion into (non_entailed_premises, entailed_premises, conclusion).

    Returns empty lists / empty string on parse failure — caller handles gracefully.
    """
    match = _SECTION_RE.search(completion)
    if not match:
        return [], [], ""

    def _triplets_in(block: str) -> list[str]:
        return [
            line.strip()
            for line in block.splitlines()
            if _TRIPLET_RE.search(line)
        ]

    non_entailed = _triplets_in(match.group(1))
    entailed = _triplets_in(match.group(2))
    conclusion = match.group(3).strip()
    return non_entailed, entailed, conclusion


class FrozenJudge:
    """Frozen SPO model used as a reward judge for GRPO training.

    Construct via __init__ (standalone) or from_shared_model (shared-base).
    See module docstring for memory tradeoffs.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        adapter_path: str,
        entailment_weight: float = 0.5,
        non_entailment_weight: float = 0.2,
        conclusion_weight: float = 0.3,
        confidence_samples: int = 4,
        confidence_weight: float = 0.0,
        confidence_temperature: float = 0.7,
        device: Optional[str] = None,
    ) -> None:
        """Standalone mode: load a separate frozen model instance.

        Args:
            adapter_path:            Path to the PEFT adapter directory.
            confidence_samples:      K samples per completion for the quality
                                     distribution (0 disables conf scoring).
            confidence_weight:       Reward budget for the confidence signal.
                                     The structural weights (entailment + non_entailment
                                     + conclusion) are scaled to fill the remainder.
            confidence_temperature:  Sampling temperature for conf score generation.
            device:                  Explicit device string; auto-detects if None.
        """
        self.entailment_weight = entailment_weight
        self.non_entailment_weight = non_entailment_weight
        self.conclusion_weight = conclusion_weight
        self.confidence_samples = confidence_samples
        self.confidence_weight = confidence_weight
        self.confidence_temperature = confidence_temperature
        self._shared = False
        self._judge_adapter: Optional[str] = None
        self._policy_adapter: Optional[str] = None

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use 4-bit quantization only when CUDA has sufficient free VRAM;
        # otherwise fall back to bfloat16 on CPU (BnB 4-bit is CUDA-only).
        _device = device or ("auto" if torch.cuda.is_available() else "cpu")
        _extra: dict = {}
        if torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig as _BnB
                _free = (
                    torch.cuda.mem_get_info(0)[0]
                ) / 1024 / 1024
                if _free >= 450:
                    _extra["quantization_config"] = _BnB(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                else:
                    _extra["torch_dtype"] = torch.bfloat16
            except Exception:
                _extra["torch_dtype"] = torch.bfloat16
        else:
            _extra["torch_dtype"] = torch.bfloat16

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_path,
            is_trainable=False,
            device_map=_device,
            **_extra,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self._yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self._no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

    @classmethod
    def from_shared_model(
        cls,
        model,
        tokenizer,
        judge_adapter_name: str = "judge",
        policy_adapter_name: str = "policy",
        entailment_weight: float = 0.5,
        non_entailment_weight: float = 0.2,
        conclusion_weight: float = 0.3,
        confidence_samples: int = 4,
        confidence_weight: float = 0.0,
        confidence_temperature: float = 0.7,
    ) -> "FrozenJudge":
        """Shared-base mode: reuse a model that already has both adapters mounted.

        The judge adapter weights must already be loaded on `model` under
        `judge_adapter_name` with requires_grad=False.  The policy adapter is
        restored after every probe call.

        Preconditions:
            model.load_adapter(judge_path, adapter_name=judge_adapter_name) called.
            All params of judge_adapter_name have requires_grad=False.
        """
        instance = cls.__new__(cls)
        instance.entailment_weight = entailment_weight
        instance.non_entailment_weight = non_entailment_weight
        instance.conclusion_weight = conclusion_weight
        instance.confidence_samples = confidence_samples
        instance.confidence_weight = confidence_weight
        instance.confidence_temperature = confidence_temperature
        instance._shared = True
        instance._judge_adapter = judge_adapter_name
        instance._policy_adapter = policy_adapter_name
        instance.model = model
        instance.tokenizer = tokenizer
        instance._yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        instance._no_id = tokenizer.encode("No", add_special_tokens=False)[0]
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _as_judge(self):
        """Context manager: switch to judge adapter, restore policy on exit.

        In standalone mode this is a no-op (model is already the judge).
        """
        if self._shared:
            self.model.set_adapter(self._judge_adapter)
        try:
            yield
        finally:
            if self._shared:
                self.model.set_adapter(self._policy_adapter)

    def _binary_probe(self, prompt: str) -> float:
        """Return P(Yes) / (P(Yes) + P(No)) for the first generated token.

        Preconditions: prompt is a well-formed natural-language question ending
        with 'Answer Yes or No.'
        Guarantee: returns a float in [0.0, 1.0].
        In shared mode, the judge adapter is active only for the duration of
        this call; the policy adapter is restored on exit.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        with self._as_judge(), torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        next_token_logits = outputs.logits[0, -1, :]
        yes_logit = next_token_logits[self._yes_id].float()
        no_logit = next_token_logits[self._no_id].float()
        yes_prob = torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)[0]
        return float(yes_prob)

    def _batch_binary_probe(self, prompts: List[str], max_batch: int = 4) -> List[float]:
        """Return P(Yes) for each prompt via batched forward passes.

        Uses left-padding so the last real token position is always index -1,
        giving a consistent next-token prediction across variable-length inputs.
        Chunks into sub-batches of at most max_batch to cap VRAM usage.

        Key: only the yes/no logit pair is retained per sequence — the full
        [batch × seq × vocab] logit tensor is freed immediately to avoid
        multi-GB VRAM spikes from the large vocab (151k tokens).

        Preconditions: prompts is non-empty; each prompt ends with 'Answer Yes or No.'
        Guarantee: returns list of floats in [0.0, 1.0], same length as prompts.
        """
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        results: List[float] = []
        try:
            for chunk_start in range(0, len(prompts), max_batch):
                chunk = prompts[chunk_start : chunk_start + max_batch]
                encoded = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                input_ids = encoded["input_ids"].to(self.model.device)
                attention_mask = encoded["attention_mask"].to(self.model.device)

                with self._as_judge(), torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Extract only the two logits we need before freeing the full tensor.
                # outputs.logits is [batch, seq, vocab=151936] — keeping it alive
                # would consume batch × seq × vocab × dtype bytes of VRAM.
                yes_no = outputs.logits[:, -1, [self._yes_id, self._no_id]].float()
                del outputs  # free [batch, seq, vocab] immediately
                probs = torch.softmax(yes_no, dim=1)[:, 0]
                results.extend(probs.cpu().tolist())
        finally:
            self.tokenizer.padding_side = orig_padding_side
        return results

    def batch_score_completions(
        self, quote: str, completions: List[str], max_batch: int = 4
    ) -> List[float]:
        """Score all completions for a single quote in one batched judge call.

        Collects all entailment, non-entailment, and conclusion probes across
        every completion, runs them as a single batched forward pass, then
        aggregates results back to per-completion scores.

        This replaces N sequential calls to score_completion() with ~1-2 batched
        forward passes, giving a 10-20× speedup on the scoring step.

        Preconditions:
            quote is non-empty.
            completions is a non-empty list of decoded model outputs.
        Guarantee:
            Returns list of floats in [0.0, 1.0], same length as completions.
            Completions that fail to parse receive 0.0.
            confidence_samples support is intentionally omitted here (use
            score_completion for that path).
        """
        sections = [_extract_sections(c) for c in completions]

        # Separate valid (has entailed premises) from invalid
        valid_mask = [bool(entailed) for _, entailed, _ in sections]
        valid_sections = [s for s, ok in zip(sections, valid_mask) if ok]

        if not valid_sections:
            return [0.0] * len(completions)

        # Build all probe prompts in one flat list, tracking metadata
        prompts: List[str] = []
        probe_meta: List[tuple] = []  # (local_valid_idx, probe_kind)

        for local_idx, (non_entailed, entailed, conclusion) in enumerate(valid_sections):
            for p in entailed:
                prompts.append(_ENTAILMENT_TEMPLATE.format(
                    quote=quote.strip(), premise=p.strip()
                ))
                probe_meta.append((local_idx, "e"))

            for p in (non_entailed or []):
                prompts.append(_NON_ENTAILMENT_TEMPLATE.format(
                    quote=quote.strip(), premise=p.strip()
                ))
                probe_meta.append((local_idx, "ne"))

            if conclusion and entailed:
                premises_text = "\n".join(f"- {p}" for p in entailed)
                prompts.append(_CONCLUSION_TEMPLATE.format(
                    premises=premises_text, conclusion=conclusion.strip()
                ))
                probe_meta.append((local_idx, "c"))

        raw_scores = self._batch_binary_probe(prompts, max_batch=max_batch)

        # Aggregate per valid completion
        n_valid = len(valid_sections)
        e_acc: List[List[float]] = [[] for _ in range(n_valid)]
        ne_acc: List[List[float]] = [[] for _ in range(n_valid)]
        c_acc: List[List[float]] = [[] for _ in range(n_valid)]

        for score, (local_idx, kind) in zip(raw_scores, probe_meta):
            if kind == "e":
                e_acc[local_idx].append(score)
            elif kind == "ne":
                ne_acc[local_idx].append(score)
            else:
                c_acc[local_idx].append(score)

        valid_results: List[float] = []
        for local_idx in range(n_valid):
            entailment_score = sum(e_acc[local_idx]) / len(e_acc[local_idx]) if e_acc[local_idx] else 0.0
            non_entailment_score = sum(ne_acc[local_idx]) / len(ne_acc[local_idx]) if ne_acc[local_idx] else 0.5
            conclusion_score = c_acc[local_idx][0] if c_acc[local_idx] else 0.0

            structural_raw = (
                self.entailment_weight * entailment_score
                + self.non_entailment_weight * non_entailment_score
                + self.conclusion_weight * conclusion_score
            )
            structural_norm = self.entailment_weight + self.non_entailment_weight + self.conclusion_weight
            valid_results.append(min(max(structural_raw / structural_norm, 0.0), 1.0))

        # Re-interleave with 0.0 for invalid completions
        final: List[float] = []
        valid_iter = iter(valid_results)
        for is_valid in valid_mask:
            final.append(next(valid_iter) if is_valid else 0.0)
        return final

    def score_entailment(self, quote: str, premise: str) -> float:
        """P(premise follows from quote).

        Preconditions: quote and premise are non-empty strings.
        """
        prompt = _ENTAILMENT_TEMPLATE.format(quote=quote.strip(), premise=premise.strip())
        return self._binary_probe(prompt)

    def score_non_entailment(self, quote: str, premise: str) -> float:
        """P(premise is absent from / not stated in quote).

        Preconditions: quote and premise are non-empty strings.
        """
        prompt = _NON_ENTAILMENT_TEMPLATE.format(quote=quote.strip(), premise=premise.strip())
        return self._binary_probe(prompt)

    def score_conclusion_coherence(
        self, entailed_premises: List[str], conclusion: str
    ) -> float:
        """P(conclusion follows from entailed premises).

        Preconditions: entailed_premises is non-empty; conclusion is non-empty.
        Returns 0.5 (neutral) when either argument is empty.
        """
        if not entailed_premises or not conclusion.strip():
            return 0.0
        premises_text = "\n".join(f"- {p}" for p in entailed_premises)
        prompt = _CONCLUSION_TEMPLATE.format(
            premises=premises_text, conclusion=conclusion.strip()
        )
        return self._binary_probe(prompt)

    def sample_confidence(
        self, quote: str, completion: str
    ) -> tuple[float, float]:
        """Sample K confidence scores from the frozen judge for one completion.

        Generates `self.confidence_samples` text responses at temperature > 0,
        parses each as a float in [0.0, 1.0], and returns (mean, std).

        Interpretation:
            high mean + low std  → judge consistently rates this extraction well
            low mean             → poor extraction
            high std             → judge uncertain; completion is at the quality margin

        Samples that fail to parse a valid float are skipped.  If no samples
        parse, returns (0.0, 1.0) — unknown quality, maximum uncertainty.

        Guarantee: mean ∈ [0.0, 1.0], std ∈ [0.0, 1.0].
        """
        prompt = _CONFIDENCE_TEMPLATE.format(
            quote=quote.strip(), completion=completion.strip()
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        with self._as_judge(), torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=True,
                temperature=self.confidence_temperature,
                num_return_sequences=self.confidence_samples,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_len = input_ids.shape[1]
        scores: List[float] = []
        for seq in output_ids:
            text = self.tokenizer.decode(
                seq[prompt_len:], skip_special_tokens=True
            ).strip()
            m = _CONF_FLOAT_RE.search(text)
            if m:
                val = float(m.group(1))
                if 0.0 <= val <= 1.0:
                    scores.append(val)

        if not scores:
            return 0.0, 1.0  # no valid parse → unknown, max uncertainty

        mean = sum(scores) / len(scores)
        if len(scores) == 1:
            return mean, 0.0
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return mean, variance ** 0.5

    def score_completion(self, quote: str, completion: str) -> float:
        """Composite judge score for a full SPO completion.

        Structural component (always active):
            - entailment probe on each entailed premise (mean)
            - non-entailment probe on each non-entailed premise (mean)
            - conclusion coherence probe

        Confidence distribution component (active when confidence_samples > 0):
            - sample K confidence ratings at temperature > 0
            - signal = conf_mean * (1 - conf_std): rewards high mean, penalises
              uncertain/inconsistent assessments

        Weight budget:
            structural_scale = 1 - confidence_weight
            Each structural sub-score keeps its relative proportion within that budget.
            confidence_weight governs the conf-distribution slice.

        Returns 0.0 immediately when:
            - completion is unparseable (no sections)
            - no entailed premises found (any quote has derivable premises)
        Guarantee: return value is in [0.0, 1.0].
        """
        non_entailed, entailed, conclusion = _extract_sections(completion)

        if not entailed and not non_entailed and not conclusion:
            return 0.0

        # No entailed premises = failed extraction → hard zero
        if not entailed:
            return 0.0

        entailment_score = sum(
            self.score_entailment(quote, p) for p in entailed
        ) / len(entailed)

        non_entailment_score = 0.5
        if non_entailed:
            non_entailment_score = sum(
                self.score_non_entailment(quote, p) for p in non_entailed
            ) / len(non_entailed)

        conclusion_score = self.score_conclusion_coherence(entailed, conclusion)

        structural_raw = (
            self.entailment_weight * entailment_score
            + self.non_entailment_weight * non_entailment_score
            + self.conclusion_weight * conclusion_score
        )
        structural_norm = self.entailment_weight + self.non_entailment_weight + self.conclusion_weight
        structural_score = structural_raw / structural_norm  # normalise to [0,1]

        if self.confidence_samples <= 0 or self.confidence_weight <= 0.0:
            return min(max(structural_score, 0.0), 1.0)

        conf_mean, conf_std = self.sample_confidence(quote, completion)
        # High mean + low std is good; high std discounts the signal proportionally
        conf_signal = conf_mean * (1.0 - min(conf_std, 1.0))

        structural_budget = 1.0 - self.confidence_weight
        composite = structural_budget * structural_score + self.confidence_weight * conf_signal
        return min(max(composite, 0.0), 1.0)
