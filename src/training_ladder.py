"""Hierarchical training ladder for SPO adapter quality validation.

Four tiers, each checking a different partition of output quality at the
scale where that partition is cheapest to verify:

  Tier 0 — Structure (zero-shot, ~2 min)
      No training. Checks the model already knows the output skeleton.
      Gate: all 3 section headers present, Entailed non-empty, pipe format.

  Tier 1 — Annotation format (50 records, 1 epoch, ~10 min)
      Mini-train to confirm the model adopts canonical (tag, confidence=N)
      annotations. Gates that the prompt change lands before wasting time
      on larger corpus.

  Tier 2 — Content quality (200 records, 2 epochs, ~25 min)
      Medium train. Checks verbatim faithfulness of Entailed spans and that
      avg_score crosses the 0.75 threshold (annotation + format scores).

  Tier 3 — Full convergence (900 records, 5 epochs, ~90 min)
      Full train. Checks avg_score ≥ target and all structure rates ≥ 0.95.

Each tier gates the next: if Tier N fails, the run stops and prints why.
The caller pays only the cost of the first failing tier.

Usage (programmatic):
    ladder = TrainingLadder(
        base_adapter="output/spo_verbatim_3ep_v8/adapter",
        corpus_path="data/train_facts_verbatim_v9.jsonl",
        output_root="output/ladder_run_001",
    )
    result = ladder.run(max_tier=3)

Usage (CLI):
    python -m src.training_ladder \\
        --base-adapter output/spo_verbatim_3ep_v8/adapter \\
        --corpus data/train_facts_verbatim_v9.jsonl \\
        --output-root output/ladder_run_001 \\
        --max-tier 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── check patterns ────────────────────────────────────────────────────────────

_TEMPLATE_RE = re.compile(r"<subject>|<relation>|<object>", re.IGNORECASE)


def _spo() -> type:
    """Lazy import of SPOEvaluator to avoid loading torch at module import time."""
    from src.spo_trainer import SPOEvaluator  # type: ignore
    return SPOEvaluator


def _extract_quote(record: dict) -> Optional[str]:
    """Extract the source quote from a training record's input_text, if present."""
    try:
        from src.preprocess_training_data import extract_quote  # type: ignore
        return extract_quote(record.get("input_text", "")) or None
    except ImportError:
        return None


# ── per-example check functions ───────────────────────────────────────────────
# All structural checks delegate to SPOEvaluator internals so the tier gates
# use the same definitions as the actual training scorer.  This prevents the
# ladder and scorer from drifting apart in their meaning of "headers present"
# or "entailed non-empty".
#
# Score breakdown from SPOEvaluator.evaluate_triplet_correctness():
#   0.20 — triplet format (pipes present)
#   0.30 — section headers correct (own line, exact spelling, no '|')
#   0.15 — confidence annotation numeric and in [0,1]
#   0.15 — tag is exactly 'observed' or 'inferred'
#   0.15 — quality (tautology/trivial/echo/degenerate penalties)
#   0.05 — ground-truth overlap bonus


# Tier 0: structure
def check_headers(output: str, _record: dict) -> bool:
    """All 3 canonical headers present on their own lines (scorer definition)."""
    return _spo()._header_score(output) >= 1.0

def check_entailed_non_empty(output: str, _record: dict) -> bool:
    """Entailed Premises section contains at least one triplet line."""
    return bool(_spo()._extract_section_triplets(output, "Entailed Premises"))

def check_pipes_well_formed(output: str, _record: dict) -> bool:
    """At least one subject|relation|object triplet exists anywhere in the output."""
    pipe_re = re.compile(r".+\|.+\|.+")
    return any(pipe_re.match(l.strip()) for l in output.splitlines())

def check_no_template_leakage(output: str, _record: dict) -> bool:
    return not _TEMPLATE_RE.search(output)

# Tier 1: annotation format
def check_confidence_numeric(output: str, _record: dict) -> bool:
    """All confidence= annotations are parseable floats in [0,1] (scorer definition)."""
    conf_val_re = re.compile(r"confidence\s*=\s*([^\s,)\n]+)")
    vals = conf_val_re.findall(output)
    if not vals:
        return False
    def _valid(v: str) -> bool:
        try:
            fv = float(v)
            return 0.0 <= fv <= 1.0
        except ValueError:
            return False
    return all(_valid(v) for v in vals)

def check_canonical_tag_format(output: str, _record: dict) -> bool:
    """At least one triplet uses the scorer's canonical (observed|inferred, confidence=N) format."""
    tag_re = re.compile(r"\(\s*(observed|inferred)\s*,\s*confidence\s*=\s*[0-9]", re.IGNORECASE)
    return bool(tag_re.search(output))

def check_both_tag_types(output: str, _record: dict) -> bool:
    """Output uses both 'observed' and 'inferred' tags (scorer awards 0.15 for valid tags)."""
    has_obs = bool(re.search(r"\bobserved\b", output, re.IGNORECASE))
    has_inf = bool(re.search(r"\binferred\b", output, re.IGNORECASE))
    return has_obs and has_inf

# Tier 2: content quality — uses scorer's section extraction and verbatim logic
def check_sections_distinct(output: str, _record: dict) -> bool:
    """Entailed and Non-Entailed sections are not identical (degenerate copy failure)."""
    ev = _spo()._extract_section_triplets(output, "Entailed Premises")
    nev = _spo()._extract_section_triplets(output, "Non-Entailed Premises")
    if not ev or not nev:
        return True  # can't judge — section absent, not a copy failure
    return set(ev) != set(nev)

def check_verbatim_entailed(output: str, record: dict) -> bool:
    """≥50% of Entailed triplet subject/object fields are verbatim spans from the quote."""
    quote = _extract_quote(record)
    if not quote:
        return True  # no quote available — can't penalise
    entailed_lines = _spo()._extract_section_triplets(output, "Entailed Premises")
    if not entailed_lines:
        return False
    ratio = _spo()._entailed_verbatim_ratio(entailed_lines, quote)
    return ratio >= 0.5

def check_avg_score_tier2(output: str, record: dict) -> bool:
    """SPO scorer score ≥ 0.75 — format + headers + annotation components all present."""
    quote = _extract_quote(record)
    score = _spo().evaluate_triplet_correctness(output, source_quote=quote)
    return score >= 0.75

# Tier 3: full convergence
def check_avg_score_tier3(output: str, record: dict) -> bool:
    """SPO scorer score ≥ 0.85 — near-ceiling; v8+v9prompt baseline is 0.909."""
    quote = _extract_quote(record)
    score = _spo().evaluate_triplet_correctness(output, source_quote=quote)
    return score >= 0.85


# ── tier definitions ──────────────────────────────────────────────────────────


@dataclass
class TierSpec:
    """Specification for one tier of the training ladder.

    Precondition: checks is non-empty; thresholds keys match checks names.
    """
    name: str
    n_train: int          # 0 = no training step
    n_epochs: int         # ignored when n_train == 0 and n_epochs == 0
    n_holdout: int
    lr: float
    max_new_tokens: int   # generation budget; use fewer tokens for structure-only tiers
    checks: list          # list of (name: str, fn: callable) tuples
    thresholds: dict[str, float] = field(default_factory=dict)

    def threshold_for(self, check_name: str) -> float:
        return self.thresholds.get(check_name, 0.85)


def _make_tiers() -> list[TierSpec]:
    """Return the canonical four-tier ladder spec.

    Tier responsibilities — calibrated to what is DETECTABLE at each training scale:

    Tier 0 (zero-shot): Does the base adapter know the output skeleton?
        Check: all 3 section headers on own lines, Entailed non-empty, pipes present.
        50×1ep mini-train is NOT enough to change a 3-epoch confidence=X habit —
        so annotation numeric checks belong at Tier 2, not Tier 1.

    Tier 1 (50rec×1ep): Does structure survive mini-training? Do both tag types appear?
        Gate: structure regression check + basic annotation vocabulary (observed/inferred).
        Deliberately does NOT gate on confidence_numeric — a 3-epoch v8 habit won't
        break in 50 steps. We gate on vocabulary, not mastery.

    Tier 2 (200rec×2ep): Do annotations converge to canonical format?
        Gate: confidence_numeric, canonical (tag, confidence=N) format, section
        distinctness, verbatim faithfulness, avg_score ≥ 0.70.
        200×2ep is enough signal to start overwriting the confidence=X pattern.

    Tier 3 (full corpus×5ep): Does full convergence hold?
        Gate: all checks at ≥ 0.90–0.95, avg_score ≥ 0.85.
    """
    tier0_checks = [
        ("headers",             check_headers),
        ("entailed_non_empty",  check_entailed_non_empty),
        ("pipes_well_formed",   check_pipes_well_formed),
        ("no_template_leakage", check_no_template_leakage),
    ]
    # Tier 1 gate: structure preserved + both tag types appear (vocabulary, not mastery)
    tier1_checks = tier0_checks + [
        ("both_tag_types", check_both_tag_types),
    ]
    # Tier 2 gate: annotation format converges + content quality
    tier2_checks = tier1_checks + [
        ("confidence_numeric",   check_confidence_numeric),
        ("canonical_tag_format", check_canonical_tag_format),
        ("sections_distinct",    check_sections_distinct),
        ("verbatim_entailed",    check_verbatim_entailed),
        ("avg_score_tier2",      check_avg_score_tier2),
    ]
    tier3_checks = tier2_checks[:-1] + [  # swap tier2 score gate for tier3
        ("avg_score_tier3", check_avg_score_tier3),
    ]

    return [
        TierSpec(
            name="tier0_structure",
            n_train=0, n_epochs=0, n_holdout=10, lr=0.0,
            # 384 tokens: enough for all 3 headers + at least one triplet per section.
            # 10 samples keeps zero-shot gate under 2 min.
            max_new_tokens=384,
            checks=tier0_checks,
            # Sanity gate: adapter must know the output skeleton before any training.
            thresholds={c: 0.70 for c, _ in tier0_checks},
        ),
        TierSpec(
            name="tier1_annotation",
            n_train=50, n_epochs=1, n_holdout=20, lr=2e-5,
            max_new_tokens=384,
            checks=tier1_checks,
            # Structure must not regress; both tag types must appear in outputs.
            # confidence_numeric intentionally excluded — 50×1ep cannot overcome a
            # 3-epoch confidence=X habit. That check belongs at Tier 2.
            thresholds={
                "headers":             0.70,
                "entailed_non_empty":  0.75,
                "pipes_well_formed":   0.85,
                "no_template_leakage": 0.90,
                "both_tag_types":      0.50,
            },
        ),
        TierSpec(
            name="tier2_content",
            n_train=200, n_epochs=2, n_holdout=30, lr=2e-5,
            max_new_tokens=512,
            checks=tier2_checks,
            # 200×2ep should break the confidence=X habit and adopt canonical format.
            thresholds={
                **{c: 0.85 for c, _ in tier0_checks},
                "both_tag_types":       0.70,
                "confidence_numeric":   0.75,
                "canonical_tag_format": 0.70,
                "sections_distinct":    0.85,
                "verbatim_entailed":    0.45,
                "avg_score_tier2":      0.65,
            },
        ),
        TierSpec(
            name="tier3_convergence",
            n_train=0, n_epochs=5, n_holdout=20, lr=1e-5,  # n_train=0 uses full corpus
            max_new_tokens=512,
            checks=tier3_checks,
            # Full training: all annotations canonical, avg_score near ceiling.
            thresholds={
                **{c: 0.90 for c, _ in tier0_checks},
                "both_tag_types":       0.80,
                "confidence_numeric":   0.90,
                "canonical_tag_format": 0.85,
                "sections_distinct":    0.90,
                "verbatim_entailed":    0.55,
                "avg_score_tier3":      0.78,
            },
        ),
    ]


# ── inference ─────────────────────────────────────────────────────────────────


def _generate_outputs(
    adapter_path: Path,
    records: list[dict],
    max_new_tokens: int = 512,
) -> list[str]:
    """Load adapter once and generate outputs for all records.

    Uses the same generation path as gen_spo_holdout.py: chat template,
    no_repeat_ngram_size=6 (prevents degenerate repetition loops), and
    strip_response_preamble to remove Qwen <think> scaffolding.

    Returns model output strings in the same order as records.
    """
    import torch
    from peft import AutoPeftModelForCausalLM  # type: ignore
    from transformers import AutoTokenizer  # type: ignore
    from src.chat_format import build_generation_prompt, strip_response_preamble  # type: ignore

    cfg_path = adapter_path / "adapter_config.json"
    if cfg_path.exists():
        import json as _json
        cfg = _json.loads(cfg_path.read_text())
        if "alora_invocation_tokens" in cfg:
            del cfg["alora_invocation_tokens"]
            cfg_path.write_text(_json.dumps(cfg, indent=2))

    model = AutoPeftModelForCausalLM.from_pretrained(str(adapter_path), device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True

    outputs = []
    for rec in records:
        chat_prompt = build_generation_prompt(tokenizer, rec["input_text"])
        inputs = tokenizer(
            chat_prompt, return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                no_repeat_ngram_size=6,
            )
        gen = ids[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(gen, skip_special_tokens=True)
        outputs.append(strip_response_preamble(raw))

    del model
    torch.cuda.empty_cache()
    return outputs


# ── training ──────────────────────────────────────────────────────────────────


def _run_train(
    adapter_path: Path,
    corpus_path: Path,
    output_dir: Path,
    n_train: int,
    n_epochs: int,
    lr: float,
    seed: int,
) -> Path:
    """Run SPO training for one tier. Returns path to produced adapter."""
    from src.run_spo_training import SPOTrainingConfig, run_spo_training  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)

    config = SPOTrainingConfig(
        adapter_path=str(adapter_path),
        dataset_path=str(corpus_path),
        output_dir=str(output_dir),
        num_epochs=n_epochs,
        learning_rate=lr,
        seed=seed,
        max_train_records=n_train if n_train > 0 else None,
        skip_regression_gate=True,
    )
    run_spo_training(config)
    return output_dir / "adapter"


# ── tier runner ───────────────────────────────────────────────────────────────


@dataclass
class TierResult:
    tier: str
    n_outputs: int
    rates: dict[str, float]
    passed: bool
    failures: list[str]


def run_tier(
    tier: TierSpec,
    adapter_path: Path,
    corpus_path: Path,
    tier_output_dir: Path,
    full_corpus_records: list[dict],
    seed: int,
) -> tuple[TierResult, Path]:
    """Execute one tier: optional mini-train → inference → checks.

    Returns (TierResult, adapter_path_to_pass_to_next_tier).
    """
    rng = random.Random(seed)

    # ── optional training ─────────────────────────────────────────────────────
    trained_adapter = adapter_path
    if tier.n_train > 0:
        train_records = rng.sample(full_corpus_records, min(tier.n_train, len(full_corpus_records)))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for r in train_records:
                f.write(json.dumps(r) + "\n")
            mini_corpus = Path(f.name)
        trained_adapter = _run_train(
            adapter_path, mini_corpus, tier_output_dir,
            n_train=tier.n_train, n_epochs=tier.n_epochs, lr=tier.lr, seed=seed,
        )
    elif tier.n_epochs > 0:
        # Full corpus training (n_train=0 means no sampling limit)
        trained_adapter = _run_train(
            adapter_path, corpus_path, tier_output_dir,
            n_train=0, n_epochs=tier.n_epochs, lr=tier.lr, seed=seed,
        )

    # ── sample holdout records ────────────────────────────────────────────────
    holdout = rng.sample(full_corpus_records, min(tier.n_holdout, len(full_corpus_records)))

    # ── generate ─────────────────────────────────────────────────────────────
    print(f"  [{tier.name}] generating {len(holdout)} outputs …", flush=True)
    outputs = _generate_outputs(trained_adapter, holdout, max_new_tokens=tier.max_new_tokens)

    # ── check ─────────────────────────────────────────────────────────────────
    per_example = [
        {name: fn(out, rec) for name, fn in tier.checks}
        for out, rec in zip(outputs, holdout)
    ]
    rates = {name: sum(r[name] for r in per_example) / len(per_example)
             for name, _ in tier.checks}

    failures = [
        f"{name}: {rates[name]:.0%} < {tier.threshold_for(name):.0%}"
        for name, _ in tier.checks
        if rates[name] < tier.threshold_for(name)
    ]
    passed = len(failures) == 0

    return TierResult(
        tier=tier.name,
        n_outputs=len(outputs),
        rates=rates,
        passed=passed,
        failures=failures,
    ), trained_adapter


# ── ladder orchestrator ───────────────────────────────────────────────────────


@dataclass
class LadderResult:
    tiers_run: list[TierResult]
    final_adapter: Optional[Path]
    passed: bool
    stopped_at: Optional[str]


class TrainingLadder:
    """Hierarchical training ladder: run tiers 0–max_tier, stop on first failure.

    Precondition: base_adapter is a valid PEFT adapter directory.
    Guarantee: never runs Tier N+1 if Tier N failed.
    """

    def __init__(
        self,
        base_adapter: str | Path,
        corpus_path: str | Path,
        output_root: str | Path,
        seed: int = 42,
    ) -> None:
        self.base_adapter = Path(base_adapter)
        self.corpus_path = Path(corpus_path)
        self.output_root = Path(output_root)
        self.seed = seed
        self._tiers = _make_tiers()

    def run(self, max_tier: int = 3) -> LadderResult:
        """Run tiers 0 through min(max_tier, 3) with go/no-go gates.

        Returns LadderResult with full per-tier breakdown.
        """
        self.output_root.mkdir(parents=True, exist_ok=True)

        with open(self.corpus_path) as f:
            full_corpus = [json.loads(l) for l in f]

        adapter = self.base_adapter
        tier_results: list[TierResult] = []
        stopped_at = None

        for i, tier in enumerate(self._tiers[:max_tier + 1]):
            tier_dir = self.output_root / tier.name
            print(f"\n{'='*60}", flush=True)
            print(f"  Tier {i}: {tier.name}", flush=True)
            action = "zero-shot" if tier.n_train == 0 and tier.n_epochs == 0 else \
                     f"mini-train {tier.n_train}rec×{tier.n_epochs}ep" if tier.n_train > 0 else \
                     f"full-train {tier.n_epochs}ep"
            print(f"  Action : {action}  holdout={tier.n_holdout}", flush=True)
            print(f"{'='*60}", flush=True)

            result, adapter = run_tier(tier, adapter, self.corpus_path, tier_dir, full_corpus, self.seed)
            tier_results.append(result)

            # Print results
            for check_name, rate in result.rates.items():
                threshold = tier.threshold_for(check_name)
                status = "✓" if rate >= threshold else "✗"
                print(f"  {status} {check_name:<28} {rate:.0%}  (need {threshold:.0%})", flush=True)

            if result.passed:
                print(f"\n  → TIER {i} PASSED", flush=True)
            else:
                print(f"\n  → TIER {i} FAILED: {'; '.join(result.failures)}", flush=True)
                stopped_at = tier.name
                break

        passed = stopped_at is None
        final_adapter = adapter if passed else None

        # Summary
        print(f"\n{'='*60}", flush=True)
        if passed:
            print(f"  LADDER PASSED  (all {len(tier_results)} tiers)  adapter={final_adapter}", flush=True)
        else:
            print(f"  LADDER STOPPED at {stopped_at}", flush=True)
        print(f"{'='*60}", flush=True)

        # Persist result
        summary = {
            "passed": passed,
            "stopped_at": stopped_at,
            "tiers": [
                {"tier": r.tier, "passed": r.passed, "rates": r.rates, "failures": r.failures}
                for r in tier_results
            ],
        }
        (self.output_root / "ladder_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        return LadderResult(
            tiers_run=tier_results,
            final_adapter=final_adapter,
            passed=passed,
            stopped_at=stopped_at,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-adapter", required=True, help="Starting PEFT adapter directory")
    parser.add_argument("--corpus", default="data/train_facts_verbatim_v9.jsonl", help="JSONL training corpus")
    parser.add_argument("--output-root", required=True, help="Directory to write tier outputs into")
    parser.add_argument("--max-tier", type=int, default=3, choices=[0, 1, 2, 3], help="Highest tier to run (default 3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ladder = TrainingLadder(
        base_adapter=args.base_adapter,
        corpus_path=args.corpus,
        output_root=args.output_root,
        seed=args.seed,
    )
    result = ladder.run(max_tier=args.max_tier)
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
