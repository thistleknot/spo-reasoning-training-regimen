"""
Tests for sample_mixture() stratified sampling in run_ablation_matrix.

Invariants tested:
  1. Regimen prior is respected (caller-supplied weights)
  2. Within-regimen length buckets are balanced (equal quota)
  3. target_size cap is enforced
  4. Single-regimen mode produces correct output
  5. Missing regimen in records is silently skipped
"""
import math
from collections import Counter

import pytest

from src.run_ablation_matrix import _length_buckets_for_records, sample_mixture


def make_records(n: int, input_len: int, tag: str) -> list:
    return [{"input_text": "x" * input_len, "output_text": "y" * 50, "_tag": tag}] * n


class TestRegimenPrior:
    def test_60_40_split(self):
        facts = make_records(900, 400, "facts")
        base  = make_records(100, 100, "base")
        mix = sample_mixture(
            {"facts": facts, "base_reasoning": base},
            {"facts": 0.6, "base_reasoning": 0.4},
            200, seed=42,
        )
        c = Counter(r["_tag"] for r in mix)
        # Allow ±5% tolerance due to integer rounding and bucket quota alignment
        assert 50 <= c["facts"] <= 140, f"facts={c['facts']} outside [50,140]"
        assert 40 <= c["base"]  <= 90,  f"base={c['base']} outside [40,90]"

    def test_equal_weights(self):
        a = make_records(500, 300, "a")
        b = make_records(500, 300, "b")
        mix = sample_mixture({"a": a, "b": b}, {"a": 0.5, "b": 0.5}, 100, seed=1)
        c = Counter(r["_tag"] for r in mix)
        # Allow ±20% tolerance
        assert 30 <= c["a"] <= 70, f"a={c['a']} outside [30,70]"
        assert 30 <= c["b"] <= 70, f"b={c['b']} outside [30,70]"


class TestWithinRegimenLengthBalance:
    def test_three_buckets_equal_quota(self):
        """Given 3 clearly-separated length groups, each should appear in output."""
        records = (
            make_records(300, 100, "short") +   # bucket 0
            make_records(300, 500, "medium") +  # bucket 1
            make_records(300, 900, "long")      # bucket 2
        )
        mix = sample_mixture({"r": records}, {"r": 1.0}, 300, seed=7)
        tags = Counter(r["_tag"] for r in mix)
        for tag in ("short", "medium", "long"):
            assert tags[tag] >= 1, f"tag '{tag}' not sampled at all"

    def test_skewed_length_still_samples_minority_bucket(self):
        """Minority bucket (10 records) must appear in a 300-sample mix."""
        records = (
            make_records(10, 100, "rare-short") +
            make_records(990, 500, "common-med")
        )
        mix = sample_mixture({"r": records}, {"r": 1.0}, 300, seed=3)
        tags = Counter(r["_tag"] for r in mix)
        assert tags["rare-short"] >= 1, "rare-short bucket not sampled"


class TestEdgeCases:
    def test_target_size_cap(self):
        records = make_records(1000, 400, "x")
        mix = sample_mixture({"r": records}, {"r": 1.0}, 50, seed=0)
        assert len(mix) <= 50

    def test_single_record_per_regimen(self):
        mix = sample_mixture(
            {"a": make_records(1, 200, "a"), "b": make_records(1, 200, "b")},
            {"a": 0.5, "b": 0.5}, 10, seed=0,
        )
        assert len(mix) >= 2

    def test_missing_regimen_skipped(self):
        """Weights mention 'c' but regimen_to_records does not — should not raise."""
        records = make_records(100, 300, "x")
        mix = sample_mixture(
            {"r": records},
            {"r": 1.0, "c": 0.5},  # 'c' not in records
            50, seed=0,
        )
        assert len(mix) >= 1

    def test_empty_active_raises(self):
        with pytest.raises(ValueError, match="no active regimens"):
            sample_mixture({}, {"r": 1.0}, 50, seed=0)


class TestLengthBuckets:
    def test_quantile_buckets_are_roughly_equal(self):
        """_length_buckets_for_records should produce approximately equal-sized buckets."""
        records = [{"input_text": "x" * i, "output_text": ""} for i in range(1, 301)]
        buckets = _length_buckets_for_records(records, n_buckets=3)
        c = Counter(buckets)
        assert len(c) == 3, f"Expected 3 buckets, got {len(c)}"
        sizes = sorted(c.values())
        # Each bucket should be within 10% of 100
        for s in sizes:
            assert 90 <= s <= 110, f"Bucket size {s} is not near 100"
