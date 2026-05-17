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
    """At least one well-formed S|P|O triplet line exists in the output.

    Requires exactly 2 pipes (3 fields). Lines with more pipes are malformed
    (e.g. 'S | tag | confidence=0.0 | O' has 3 pipes and is rejected).
    """
    pipe_re = re.compile(r"^[^|]+\|[^|]+\|[^|]+$")
    return any(pipe_re.match(l.strip()) for l in output.splitlines())

def check_no_template_leakage(output: str, _record: dict) -> bool:
    return not _TEMPLATE_RE.search(output)

# Tier 1: annotation format

def _normalize_confidence_syntax(output: str) -> str:
    """Rewrite malformed confidence= syntax to confidence=N before format checks.

    Handles the full range of syntactic variants the v11 base model generates:
      confidence=<0.7>   or <0.7)  (angle-bracket with optional closing >)
      confidence="0.7"   or "-0.7) (quoted, possibly with missing close-quote)
      confidence=-0.7               (negative sign — abs value used)
      confidence=inferred, confidence=0  (duplicate tag — use trailing numeric value)
    """
    # angle-bracket variant: <N.N> or <N.N) — closing char optional
    output = re.sub(r'confidence\s*=\s*<([0-9]+\.?[0-9]*)[>)]?', r'confidence=\1', output)
    # quoted variant: "N.N" or "-N.N" with any closing delimiter
    output = re.sub(r'confidence\s*=\s*"-?([0-9]+\.?[0-9]*)[")>)]*',
                    r'confidence=\1', output)
    # negative value: -N.N → abs value
    output = re.sub(r'confidence\s*=\s*-([0-9]+\.?[0-9]*)', r'confidence=\1', output)
    # duplicate tag: confidence=WORD, confidence=N → keep numeric part
    output = re.sub(r'confidence\s*=\s*[A-Za-z]\w*\s*,\s*confidence\s*=\s*([0-9]+\.?[0-9]*)',
                    r'confidence=\1', output)
    return output

def check_confidence_numeric(output: str, _record: dict) -> bool:
    """≥50% of confidence= annotations are parseable floats in [0,1].

    Fractional check (not all-or-nothing): a sample with 1 valid and 1 invalid
    annotation still passes at 50%. This aligns with the scorer's partial-credit
    philosophy and handles mixed outputs that occur mid-training.

    Applies _normalize_confidence_syntax() first: confidence=<0.7> and
    confidence="0.7" are treated as confidence=0.7 for evaluation purposes.
    """
    output = _normalize_confidence_syntax(output)
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
    valid_count = sum(1 for v in vals if _valid(v))
    return valid_count / len(vals) >= 0.50

def check_canonical_tag_format(output: str, _record: dict) -> bool:
    """At least one triplet uses the scorer's canonical (observed|inferred, confidence=N) format.

    Applies _normalize_confidence_syntax() first so confidence=<0.7> counts as
    canonical (the value is correct; only the delimiter is wrong).
    Used by facts_with_confidence / syllogism_with_confidence regimens only.
    """
    output = _normalize_confidence_syntax(output)
    tag_re = re.compile(r"\(\s*(observed|inferred)\s*,\s*confidence\s*=\s*[0-9]", re.IGNORECASE)
    return bool(tag_re.search(output))


def check_clean_tag_format(output: str, _record: dict) -> bool:
    """At least one triplet uses the base-regimen tag-only annotation (observed) or (inferred).

    Base-reasoning training strips confidence scores so a judge can assign them
    independently.  This gate verifies the model learned that clean format —
    not that it retained the old (tag, confidence=N) shape.
    """
    tag_only_re = re.compile(r"\(\s*(observed|inferred)\s*\)", re.IGNORECASE)
    return bool(tag_only_re.search(output))

def check_tags_exclusive(output: str, _record: dict) -> bool:
    """No single triplet line carries both 'observed' and 'inferred' as annotation tags.

    A triplet annotation must use exactly one epistemics tag per line.
    Having both on the same line (e.g. '(observed, inferred, confidence=0.8)')
    is a format error regardless of how many triplets are in the output.

    Implementation: parses annotation parentheticals on each pipe-bearing line,
    strips the ``confidence=VALUE`` portion before searching for tag words.
    This avoids the false positive from ``confidence="inferred"`` where "inferred"
    is a value, not a semantic tag.
    """
    annot_re = re.compile(r'\(([^)]+)\)')
    conf_val_re = re.compile(r'confidence\s*=\s*\S+', re.IGNORECASE)
    tag_re = re.compile(r'\b(observed|inferred)\b', re.IGNORECASE)
    for line in output.splitlines():
        if "|" not in line:
            continue
        for m in annot_re.finditer(line):
            content = conf_val_re.sub("", m.group(1))
            tags = {t.lower() for t in tag_re.findall(content)}
            if "observed" in tags and "inferred" in tags:
                return False
    return True

# Tier 2: content quality — uses scorer's section extraction and verbatim logic
def check_sections_distinct(output: str, _record: dict) -> bool:
    """Entailed and Non-Entailed sections are not identical (degenerate copy failure)."""
    ev = _spo()._extract_section_triplets(output, "Entailed Premises")
    nev = _spo()._extract_section_triplets(output, "Non-Entailed Premises")
    if not ev or not nev:
        return True  # can't judge — section absent, not a copy failure
    return set(ev) != set(nev)

def check_verbatim_entailed(output: str, record: dict) -> bool:
    """≥50% of Entailed triplet S/P/O fields are verbatim spans from the quote."""
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
    """SPO scorer score ≥ 0.80 — strong convergence; calibrated to 900×5ep empirical baseline."""
    quote = _extract_quote(record)
    score = _spo().evaluate_triplet_correctness(output, source_quote=quote)
    return score >= 0.80


# transliteration: parenthetical paraphrase triplet after each entailed verbatim line
# Matches a transliteration line: (subject | predicate | object) where the
# predicate field may contain annotation parens like "(inferred)" or
# "(observed, confidence=0.9)".  Using [^|]+? for subject/predicate (no "|"
# allowed) and .+? for object, all non-greedy before \s*\).
_TRANSLIT_RE = re.compile(r"^\(\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*\)\s*$")


def _extract_entailed_block_lines(output: str) -> list[str]:
    """Return raw lines from the Entailed Premises section, including paren lines."""
    lines = output.splitlines()
    capturing = False
    result = []
    for line in lines:
        if re.match(r"Entailed Premises\s*:", line, re.IGNORECASE):
            capturing = True
            inline = line.split(":", 1)[-1].strip()
            if inline:
                result.append(inline)
            continue
        if capturing:
            if re.match(
                r"^(Non-Entailed Premises|Throughline|Syllogism)\s*:",
                line, re.IGNORECASE
            ):
                break
            if line.strip():
                result.append(line.strip())
    return result


def _record_has_transliteration(record: dict) -> bool:
    """True when the record's expected output_text contains at least one transliteration line."""
    for line in _extract_entailed_block_lines(record.get("output_text", "")):
        if _TRANSLIT_RE.match(line):
            return True
    return False


def check_transliteration_present(output: str, record: dict) -> bool | None:
    """Recall check: for records whose *expected* output contains transliteration lines,
    verify the model also produces at least one.

    Returns None (N/A) for records with no transliteration in their expected output so
    only transliteration-eligible records contribute to the pass rate.  This avoids
    diluting the score with the majority of plain-English records that should not produce
    transliteration at all.
    """
    if not _record_has_transliteration(record):
        return None  # N/A — plain-English record; skip denominator
    for line in _extract_entailed_block_lines(output):
        if _TRANSLIT_RE.match(line):
            return True
    return False


def check_transliteration_format(output: str, _record: dict) -> bool:
    """All transliteration lines that ARE present use (S | P (tag) | O) format.

    Accepts both tag-only ``(observed)`` and confidence-bearing
    ``(observed, confidence=N)`` annotations — the base-regimen strips
    confidence at serialization time, so either shape is valid at inference.
    Returns True vacuously when no transliteration lines are present so early
    tiers are not penalised.
    """
    tag_re = re.compile(r"\(\s*(observed|inferred)\s*[,)]", re.IGNORECASE)
    for line in _extract_entailed_block_lines(output):
        m = _TRANSLIT_RE.match(line)
        if not m:
            continue
        predicate_field = m.group(2)
        if not tag_re.search(predicate_field):
            return False
    return True


# ── regimen checks (facts_with_confidence / syllogism_with_confidence) ─────────


def check_facts_headers(output: str, _record: dict) -> bool:
    """Both facts-regimen headers present: 'Non-Entailed Premises:' and 'Entailed Premises:'.

    facts_with_confidence outputs do not include a Throughline section, so
    check_headers (which requires all 3 canonical headers) would always fail.
    This check validates the two factual-section headers only.
    """
    lines = {l.strip() for l in output.splitlines() if l.strip().endswith(":") and "|" not in l}
    return "Non-Entailed Premises:" in lines and "Entailed Premises:" in lines


def check_throughline_present(output: str, _record: dict) -> bool:
    """syllogism_with_confidence output contains 'Throughline:' with non-empty content.

    Checks that the Throughline header exists and the very next non-blank line
    has at least 10 characters (not a stub or template placeholder).
    """
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "Throughline:":
            for j in range(i + 1, min(i + 4, len(lines))):
                content = lines[j].strip()
                if content and len(content) >= 10:
                    return True
    return False


def check_syllogism_confidence_present(output: str, _record: dict) -> bool:
    """syllogism_with_confidence output contains 'Confidence:' followed by a valid float.

    The expected format is:
        Confidence:
          0.85
    Checks that the header exists and the next non-blank line is a parseable
    float in [0.0, 1.0].
    """
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "Confidence:":
            for j in range(i + 1, min(i + 4, len(lines))):
                content = lines[j].strip()
                if not content:
                    continue
                try:
                    val = float(content)
                    return 0.0 <= val <= 1.0
                except ValueError:
                    pass
    return False


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
    corpus_override: Optional[Path] = None  # per-tier corpus; overrides LadderRunner.corpus_path

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
    # Tier 1 gate: structure preserved + tag exclusivity (no line has both observed+inferred)
    tier1_checks = tier0_checks + [
        ("tags_exclusive", check_tags_exclusive),
    ]
    # Tier 2 gate: content quality + section structure.
    # clean_tag_format and transliteration checks deferred to tier3 — 200×2ep
    # consistently measures only ~30-40% on tag format even when verbatim/score are good.
    tier2_checks = tier1_checks + [
        ("sections_distinct",    check_sections_distinct),
        ("verbatim_entailed",    check_verbatim_entailed),
        ("avg_score_tier2",      check_avg_score_tier2),
    ]
    # Tier 3 adds tag format convergence.
    # transliteration_present removed: the feature is teacher-LLM noise, not a
    # learnable signal.  Only 13/173 "tl" records in the training corpus have
    # non-English input; the rest are English quotes where the teacher added
    # random paraphrase lines with no consistent input trigger.  103 unique tl
    # lines across 173 records confirms these are not a systematic pattern.
    # transliteration_format is retained as a soft format-only check (it is
    # vacuously True when no tl lines are present, so it does not block the tier).
    # confidence_numeric / canonical_tag_format removed: base_reasoning training
    # strips confidence by design — a judge assigns it independently post-training.
    # avg_score_tier3 also removed: SPOEvaluator.evaluate_triplet_correctness awards
    # 0.15 each for confidence annotation and tag-with-confidence format, both of
    # which are absent by design (max reachable score ~0.50 < any meaningful gate).
    tier3_checks = tier2_checks[:-1] + [  # drop tier2 score gate; tier3 has its own
        ("clean_tag_format",          check_clean_tag_format),
        ("transliteration_format",    check_transliteration_format),
    ]

    # Tier 4: facts_with_confidence regimen.
    # Teaches the model to produce (tag, confidence=N) annotations on top of the
    # tag-only base the previous three tiers established.  Uses a separate corpus
    # (confidence-bearing serialization of the same structured records).
    # check_headers is intentionally absent: facts_with_confidence has no Throughline
    # section, so the 3-header gate would always fail (2/3 = 0.67 < required 1.0).
    tier4_checks = [
        ("facts_headers",           check_facts_headers),
        ("pipes_well_formed",       check_pipes_well_formed),
        ("entailed_non_empty",      check_entailed_non_empty),
        ("confidence_numeric",      check_confidence_numeric),
        ("canonical_tag_format",    check_canonical_tag_format),
    ]

    # Tier 5: syllogism_with_confidence regimen.
    # Takes extracted facts as input and produces a throughline + confidence score.
    # Corpus is a separate serialization (give-me-the-throughline prompt format).
    tier5_checks = [
        ("throughline_present",           check_throughline_present),
        ("syllogism_confidence_present",  check_syllogism_confidence_present),
    ]

    return [
        TierSpec(
            name="tier0_structure",
            n_train=0, n_epochs=0, n_holdout=10, lr=0.0,
            max_new_tokens=384,
            checks=tier0_checks,
            thresholds={c: 0.70 for c, _ in tier0_checks},
        ),
        TierSpec(
            name="tier1_annotation",
            n_train=50, n_epochs=1, n_holdout=20, lr=2e-5,
            max_new_tokens=384,
            checks=tier1_checks,
            # Structure must not regress; no triplet line may carry both tag types.
            # confidence_numeric excluded — 50×1ep cannot overcome a 3-epoch confidence=X habit.
            thresholds={
                "headers":             0.70,
                "entailed_non_empty":  0.75,
                "pipes_well_formed":   0.85,
                "no_template_leakage": 0.90,
                "tags_exclusive":      0.70,  # mini-train instability; escalates tier2→0.85, tier3→0.90
            },
        ),
        TierSpec(
            name="tier2_content",
            n_train=200, n_epochs=2, n_holdout=30, lr=2e-5,
            max_new_tokens=512,
            checks=tier2_checks,
            # 200×2ep sufficient for content quality + section structure.
            # Tag-format and transliteration checks deferred to tier3 where 200×5ep
            # provides enough gradient steps for format convergence.
            thresholds={
                **{c: 0.85 for c, _ in tier0_checks},
                "headers":              0.80,  # 30-sample holdout noise; tier3 enforces 0.90
                "tags_exclusive":       0.85,  # measured 97% on 200×2ep v11
                "sections_distinct":    0.85,
                "verbatim_entailed":    0.45,
                "avg_score_tier2":      0.65,
            },
        ),
        TierSpec(
            name="tier3_convergence",
            n_train=200, n_epochs=5, n_holdout=50, lr=2e-5,
            max_new_tokens=512,
            checks=tier3_checks,
            # 200-record cap: format learnable from 200 examples; data/prompt is the
            # lever if it's not.  5 epochs ≈ 1000 gradient steps — sufficient for
            # convergence without a 2h run.
            # confidence_numeric / canonical_tag_format removed: base_reasoning training
            # now strips confidence by design so a judge can assign it independently.
            # clean_tag_format replaces them: checks that (observed)/(inferred) tags
            # appear without confidence numbers.
            # transliteration_present removed: teacher-LLM artifact (only 13/173 tl
            # records have non-English input; not a learnable conditional feature).
            # avg_score_tier3 removed: SPOEvaluator requires confidence=N.N which
            # base_reasoning strips by design; scorer max is ~0.50 with tag-only output.
            thresholds={
                **{c: 0.90 for c, _ in tier0_checks},
                "headers":                   0.85,
                "entailed_non_empty":        0.75,
                "tags_exclusive":            0.90,
                "clean_tag_format":          0.70,  # (observed)/(inferred) without confidence
                "sections_distinct":         0.90,
                "verbatim_entailed":         0.55,
                # transliteration_format: vacuously True when model produces no tl lines,
                # which is correct for the 160/173 English "tl" records.  When tl lines
                # are produced, this ensures (tag) format without confidence numbers.
                "transliteration_format":    0.40,
                # avg_score_tier3 removed: SPOEvaluator requires confidence=N.N which
                # base_reasoning strips by design; scorer max is ~0.50 with tag-only output.
            },
        ),
        TierSpec(
            name="tier4_facts_confidence",
            n_train=200, n_epochs=3, n_holdout=50, lr=2e-5,
            max_new_tokens=512,
            checks=tier4_checks,
            # 200×3ep is enough to convert tag-only annotations to (tag, confidence=N).
            # check_facts_headers validates the 2 factual-section headers (no Throughline).
            # check_canonical_tag_format threshold is conservative (0.65) because the model
            # may hedge with tag-only output during the first epoch before fully committing.
            thresholds={
                "facts_headers":           0.80,
                "pipes_well_formed":       0.85,
                "entailed_non_empty":      0.75,
                "confidence_numeric":      0.70,
                "canonical_tag_format":    0.65,
            },
            corpus_override=Path("data/train_facts_with_confidence_verbatim_v19.jsonl"),
        ),
        TierSpec(
            name="tier5_syllogism",
            n_train=200, n_epochs=3, n_holdout=50, lr=2e-5,
            max_new_tokens=256,  # throughline + confidence is short
            checks=tier5_checks,
            # 200×3ep for a short conditional generation task (given facts → score).
            thresholds={
                "throughline_present":           0.80,
                "syllogism_confidence_present":  0.75,
            },
            corpus_override=Path("data/train_syllogism_with_confidence_verbatim_v19.jsonl"),
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
class HoldoutSample:
    prompt: str
    expected: str
    generated: str
    score: float


@dataclass
class TierResult:
    tier: str
    n_outputs: int
    rates: dict[str, float]
    passed: bool
    failures: list[str]
    samples: list[HoldoutSample] = field(default_factory=list)


def _code_block(text: str) -> str:
    return f"```text\n{text.strip()}\n```"


def _write_tier_holdout_markdown(result: TierResult, path: Path) -> None:
    """Write prompt/expected/generated/score for every holdout sample in one tier."""
    avg_score = sum(s.score for s in result.samples) / len(result.samples) if result.samples else 0.0
    failure_checks = {f.split(":")[0] for f in result.failures}
    lines = [
        f"# Holdout examples — {result.tier}",
        "",
        f"**Pass:** {'✓' if result.passed else '✗'}  |  "
        f"**Avg score:** {avg_score:.4f}  |  "
        f"**N:** {result.n_outputs}",
        "",
        "## Check rates",
        "",
        "| Check | Rate | Status |",
        "|---|---:|:---:|",
    ]
    for check_name, rate in result.rates.items():
        status = "✗" if check_name in failure_checks else "✓"
        lines.append(f"| {check_name} | {rate:.0%} | {status} |")
    for i, s in enumerate(result.samples, start=1):
        lines.extend([
            "",
            f"## Example {i}  (score: {s.score:.4f})",
            "",
            "**Prompt**",
            "",
            _code_block(s.prompt),
            "",
            "**Expected**",
            "",
            _code_block(s.expected),
            "",
            "**Generated**",
            "",
            _code_block(s.generated),
        ])
    path.write_text("\n".join(lines) + "\n")


def _write_combined_holdout_markdown(tier_results: list[TierResult], path: Path) -> None:
    """Write a cross-tier summary with avg score progression and all per-tier examples."""
    lines = [
        "# Training Ladder — Holdout Examples",
        "",
        "## Score progression by tier",
        "",
        "| Tier | Avg score | Pass |",
        "|---|---:|:---:|",
    ]
    for r in tier_results:
        avg = sum(s.score for s in r.samples) / len(r.samples) if r.samples else 0.0
        lines.append(f"| {r.tier} | {avg:.4f} | {'✓' if r.passed else '✗'} |")
    for r in tier_results:
        avg = sum(s.score for s in r.samples) / len(r.samples) if r.samples else 0.0
        lines.extend([
            "",
            f"---",
            "",
            f"# {r.tier}  (avg score: {avg:.4f})",
        ])
        for i, s in enumerate(r.samples, start=1):
            lines.extend([
                "",
                f"## {r.tier} — Example {i}  (score: {s.score:.4f})",
                "",
                "**Prompt**",
                "",
                _code_block(s.prompt),
                "",
                "**Expected**",
                "",
                _code_block(s.expected),
                "",
                "**Generated**",
                "",
                _code_block(s.generated),
            ])
    path.write_text("\n".join(lines) + "\n")


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

    # ── stratified train/holdout split ────────────────────────────────────────
    # Partition by transliteration presence so both training AND holdout get tl
    # representation. Without stratification, all tl records are consumed by
    # training (take_tl = min(173, 200) = 173) and holdout gets ~0 by chance.
    with_tl = [r for r in full_corpus_records if _record_has_transliteration(r)]
    without_tl = [r for r in full_corpus_records if not _record_has_transliteration(r)]

    # Reserve a proportional share of tl records for holdout.
    n_total = max(len(full_corpus_records), 1)
    n_tl_holdout = max(5, len(with_tl) * tier.n_holdout // n_total)
    n_tl_holdout = min(n_tl_holdout, len(with_tl))

    rng.shuffle(with_tl)
    tl_holdout_pool = with_tl[:n_tl_holdout]
    tl_train_pool = with_tl[n_tl_holdout:]

    n_plain_holdout = max(0, tier.n_holdout - len(tl_holdout_pool))
    plain_holdout = rng.sample(without_tl, min(n_plain_holdout, len(without_tl)))
    holdout: list[dict] = tl_holdout_pool + plain_holdout
    rng.shuffle(holdout)

    # ── optional training ─────────────────────────────────────────────────────
    trained_adapter = adapter_path
    if tier.n_train > 0:
        # Training pool: remaining tl records + plain records not in holdout.
        plain_holdout_ids = {id(r) for r in plain_holdout}
        plain_train_pool = [r for r in without_tl if id(r) not in plain_holdout_ids]
        n = min(tier.n_train, len(full_corpus_records) - len(holdout))
        take_tl = min(len(tl_train_pool), n)
        take_plain = n - take_tl
        sampled_tl = rng.sample(tl_train_pool, take_tl)
        sampled_plain = rng.sample(plain_train_pool, min(take_plain, len(plain_train_pool)))
        train_records = sampled_tl + sampled_plain
        rng.shuffle(train_records)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for r in train_records:
                f.write(json.dumps(r) + "\n")
            mini_corpus = Path(f.name)
        trained_adapter = _run_train(
            adapter_path, mini_corpus, tier_output_dir,
            n_train=tier.n_train, n_epochs=tier.n_epochs, lr=tier.lr, seed=seed,
        )
    elif tier.n_epochs > 0:
        # n_train=0 with epochs>0: use full corpus (available for external callers;
        # no built-in tier uses this path — all tiers now have explicit n_train caps).
        trained_adapter = _run_train(
            adapter_path, corpus_path, tier_output_dir,
            n_train=0, n_epochs=tier.n_epochs, lr=tier.lr, seed=seed,
        )

    # ── generate ─────────────────────────────────────────────────────────────
    print(f"  [{tier.name}] generating {len(holdout)} outputs …", flush=True)
    outputs = _generate_outputs(trained_adapter, holdout, max_new_tokens=tier.max_new_tokens)

    # ── check ─────────────────────────────────────────────────────────────────
    per_example = [
        {name: fn(out, rec) for name, fn in tier.checks}
        for out, rec in zip(outputs, holdout)
    ]
    # None means N/A (check not applicable for this record); exclude from denominator.
    rates = {}
    for name, _ in tier.checks:
        eligible = [r[name] for r in per_example if r[name] is not None]
        rates[name] = sum(eligible) / len(eligible) if eligible else 1.0

    failures = [
        f"{name}: {rates[name]:.0%} < {tier.threshold_for(name):.0%}"
        for name, _ in tier.checks
        if rates[name] < tier.threshold_for(name)
    ]
    passed = len(failures) == 0

    # ── score + collect samples ───────────────────────────────────────────────
    spo = _spo()
    samples = [
        HoldoutSample(
            prompt=rec.get("input_text", ""),
            expected=rec.get("output_text", ""),
            generated=out,
            score=spo.evaluate_triplet_correctness(out, source_quote=_extract_quote(rec)),
        )
        for out, rec in zip(outputs, holdout)
    ]

    result = TierResult(
        tier=tier.name,
        n_outputs=len(outputs),
        rates=rates,
        passed=passed,
        failures=failures,
        samples=samples,
    )

    # Write per-tier holdout markdown
    tier_output_dir.mkdir(parents=True, exist_ok=True)
    _write_tier_holdout_markdown(result, tier_output_dir / "holdout_examples.md")

    return result, trained_adapter


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

    def run(self, max_tier: int = 3, start_tier: int = 0) -> LadderResult:
        """Run tiers start_tier through min(max_tier, 5) with go/no-go gates.

        When start_tier > 0, base_adapter is used directly as the starting
        adapter for that tier (tiers 0..start_tier-1 are skipped and not
        recorded in the result).

        Tiers 4–5 use their own corpus_override path (regimen-specific serialization).
        For these tiers the active corpus is loaded fresh from corpus_override so the
        holdout records match the regimen format.

        Returns LadderResult with full per-tier breakdown.
        """
        self.output_root.mkdir(parents=True, exist_ok=True)

        with open(self.corpus_path) as f:
            default_full_corpus = [json.loads(l) for l in f]

        adapter = self.base_adapter
        tier_results: list[TierResult] = []
        stopped_at = None

        for i, tier in enumerate(self._tiers[:max_tier + 1]):
            if i < start_tier:
                print(f"  Skipping tier {i}: {tier.name} (start_tier={start_tier})", flush=True)
                continue
            tier_dir = self.output_root / tier.name
            print(f"\n{'='*60}", flush=True)
            print(f"  Tier {i}: {tier.name}", flush=True)
            action = "zero-shot" if tier.n_train == 0 and tier.n_epochs == 0 else \
                     f"mini-train {tier.n_train}rec×{tier.n_epochs}ep" if tier.n_train > 0 else \
                     f"full-train {tier.n_epochs}ep"
            print(f"  Action : {action}  holdout={tier.n_holdout}", flush=True)
            print(f"{'='*60}", flush=True)

            if tier.corpus_override is not None:
                tier_corpus_path = tier.corpus_override
                with open(tier_corpus_path) as f:
                    tier_full_corpus = [json.loads(l) for l in f]
            else:
                tier_corpus_path = self.corpus_path
                tier_full_corpus = default_full_corpus

            result, adapter = run_tier(tier, adapter, tier_corpus_path, tier_dir, tier_full_corpus, self.seed)
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
                {"tier": r.tier, "passed": r.passed, "rates": r.rates, "failures": r.failures,
                 "avg_score": sum(s.score for s in r.samples) / len(r.samples) if r.samples else None}
                for r in tier_results
            ],
        }
        (self.output_root / "ladder_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        _write_combined_holdout_markdown(tier_results, self.output_root / "holdout_examples.md")

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
    parser.add_argument("--max-tier", type=int, default=3, choices=[0, 1, 2, 3, 4, 5], help="Highest tier to run (default 3)")
    parser.add_argument("--start-tier", type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help="First tier to run (default 0); use with --base-adapter pointing at a prior tier's output adapter")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ladder = TrainingLadder(
        base_adapter=args.base_adapter,
        corpus_path=args.corpus,
        output_root=args.output_root,
        seed=args.seed,
    )
    result = ladder.run(max_tier=args.max_tier, start_tier=args.start_tier)
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
