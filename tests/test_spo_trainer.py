"""Unit tests for SPO trainer reward parsing and weighting behavior."""

import unittest

import torch

from src.spo_trainer import (
    PromptContract,
    SPOEvaluator,
    SPOReward,
    SPOTrainer,
    assert_output_quality,
)


class SPOTrainerTests(unittest.TestCase):
    """Lock the runnable SPO trainer semantics."""

    def setUp(self) -> None:
        self.trainer = SPOTrainer(
            model=None,
            tokenizer=None,
            evaluation_fn=lambda output, ground_truth: 1.0,
        )

    def test_extract_confidence_averages_triplet_annotations(self) -> None:
        """Triplet outputs should use the average `confidence=` annotation."""
        output = "\n".join(
            [
                "Entailed Premises:",
                "foo | is (observed, confidence=1.0) | bar",
                "baz | is (inferred, confidence=0.5) | qux",
            ]
        )
        self.assertAlmostEqual(self.trainer.extract_confidence(output), 0.75)

    def test_extract_confidence_reads_confidence_section(self) -> None:
        """Score-bearing outputs should parse the explicit Confidence section."""
        output = "Throughline:\n  Example\n\nConfidence:\n  0.80"
        self.assertAlmostEqual(self.trainer.extract_confidence(output), 0.8)

    def test_compute_loss_upweights_high_reward_sequences(self) -> None:
        """Higher rewards should increase the contribution of higher-loss sequences."""
        logits = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 4.0], [0.0, 4.0]],
                [[0.0, 0.0], [4.0, 0.0], [4.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [
                [0, 1, 1],
                [0, 1, 1],
            ],
            dtype=torch.long,
        )

        low_loss_high_reward = self.trainer.compute_loss(
            logits,
            [SPOReward(1.0, 0.9), SPOReward(1.0, 0.1)],
            labels,
        )
        high_loss_high_reward = self.trainer.compute_loss(
            logits,
            [SPOReward(1.0, 0.1), SPOReward(1.0, 0.9)],
            labels,
        )

        self.assertLess(low_loss_high_reward.item(), high_loss_high_reward.item())


class SPOEvaluatorRepetitionTests(unittest.TestCase):
    """Lock the new repetition and tautology penalties in evaluate_triplet_correctness."""

    GOOD_OUTPUT = (
        "Non-Entailed Premises:\n"
        "silence | implies (inferred, confidence=0.8) | wisdom\n"
        "speech | reveals (observed, confidence=1.0) | foolishness\n\n"
        "Entailed Premises:\n"
        "remaining silent | avoids (inferred, confidence=0.7) | confirming foolishness\n"
        "talking | confirms (observed, confidence=1.0) | foolishness\n"
    )
    REPEATED_OUTPUT = "\n".join(
        ["majority | is (observed, confidence=1.0) | is a group of people"] * 8
    )
    SELF_REF_OUTPUT = (
        "Non-Entailed Premises:\n"
        "the speaker | is (observed, confidence=1.0) | is the speaker\n"
        "Entailed Premises:\n"
        "the speaker | is (inferred, confidence=0.5) | is speaking\n"
    )

    def test_good_output_scores_above_threshold(self) -> None:
        """A diverse, non-repetitive, non-tautological output should score > 0.6."""
        score = SPOEvaluator.evaluate_triplet_correctness(self.GOOD_OUTPUT)
        self.assertGreater(score, 0.6, f"Expected >0.6 for good output, got {score}")

    def test_repeated_output_scores_zero(self) -> None:
        """An output where every line is identical should receive a zero score."""
        score = SPOEvaluator.evaluate_triplet_correctness(self.REPEATED_OUTPUT)
        self.assertEqual(score, 0.0, f"Expected 0.0 for repeated output, got {score}")

    def test_self_referential_output_penalised(self) -> None:
        """An output with tautological triplets should score lower than the good output."""
        good_score = SPOEvaluator.evaluate_triplet_correctness(self.GOOD_OUTPUT)
        self_ref_score = SPOEvaluator.evaluate_triplet_correctness(self.SELF_REF_OUTPUT)
        self.assertGreater(good_score, self_ref_score)

    def test_ground_truth_overlap_boosts_score(self) -> None:
        """Providing a matching ground truth should yield a higher score than no ground truth."""
        score_with_gt = SPOEvaluator.evaluate_triplet_correctness(
            self.GOOD_OUTPUT, ground_truth=self.GOOD_OUTPUT
        )
        score_without_gt = SPOEvaluator.evaluate_triplet_correctness(self.GOOD_OUTPUT)
        self.assertGreaterEqual(score_with_gt, score_without_gt)


class SPOEvaluatorHeaderTests(unittest.TestCase):
    """Lock the section header quality gate in evaluate_triplet_correctness."""

    PERFECT_OUTPUT = (
        "Non-Entailed Premises:\n"
        "silence | implies (inferred, confidence=0.8) | wisdom\n\n"
        "Entailed Premises:\n"
        "talking | confirms (observed, confidence=1.0) | foolishness\n\n"
        "Throughline:\n"
        "remaining silent avoids confirming foolishness\n"
    )
    GARBLED_OUTPUT = (
        "Non-Entailed Prems:\n"
        "silence | implies (inferred, confidence=0.8) | wisdom\n\n"
        "Entailed Prims:\n"
        "talking | confirms (observed, confidence=1.0) | foolishness\n\n"
        "Throughlin':\n"
        "remaining silent avoids confirming foolishness\n"
    )
    TWO_OF_THREE_OUTPUT = (
        "Non-Entailed Premises:\n"
        "silence | implies (inferred, confidence=0.8) | wisdom\n\n"
        "Entailed Premises:\n"
        "talking | confirms (observed, confidence=1.0) | foolishness\n"
    )

    def test_all_canonical_headers_score_full(self) -> None:
        """Output with all three canonical headers should yield header_score=1.0."""
        self.assertAlmostEqual(SPOEvaluator._header_score(self.PERFECT_OUTPUT), 1.0)

    def test_garbled_headers_score_zero(self) -> None:
        """Output with garbled/abbreviated headers should yield header_score=0.0."""
        self.assertEqual(SPOEvaluator._header_score(self.GARBLED_OUTPUT), 0.0)

    def test_partial_headers_score_fractional(self) -> None:
        """Output with 2 of 3 canonical headers should yield header_score ~0.667."""
        score = SPOEvaluator._header_score(self.TWO_OF_THREE_OUTPUT)
        self.assertAlmostEqual(score, 2 / 3, places=2)

    def test_perfect_output_scores_one(self) -> None:
        """Output with all canonical headers, proper triplets, confidence, and tags scores 1.0."""
        score = SPOEvaluator.evaluate_triplet_correctness(self.PERFECT_OUTPUT)
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_garbled_headers_score_lower_than_canonical(self) -> None:
        """Same triplet content but garbled headers should score lower than canonical headers."""
        garbled = SPOEvaluator.evaluate_triplet_correctness(self.GARBLED_OUTPUT)
        canonical = SPOEvaluator.evaluate_triplet_correctness(self.PERFECT_OUTPUT)
        self.assertGreater(canonical, garbled)

    def test_no_headers_output_loses_header_weight(self) -> None:
        """An output with no section headers at all should lose the full 0.35 header weight."""
        no_headers = (
            "silence | implies (inferred, confidence=0.8) | wisdom\n"
            "talking | confirms (observed, confidence=1.0) | foolishness\n"
        )
        score = SPOEvaluator.evaluate_triplet_correctness(no_headers)
        # Max possible without any header credit: 0.25 + 0.15 + 0.15 + 0.1 = 0.65
        self.assertLessEqual(score, 0.65 + 1e-6)



    """Lock the pre-computed data-quality scorer used to differentiate SPO training rewards."""

    def _make_record(self, output_text: str) -> dict:
        return {"output_text": output_text}

    def test_diverse_output_scores_higher_than_repeated(self) -> None:
        """A record with diverse triplets should score higher than one with repeated triplets."""
        from src.run_spo_training import score_training_sample

        diverse = self._make_record(
            "silence | implies (inferred, confidence=0.8) | wisdom\n"
            "speech | reveals (observed, confidence=1.0) | foolishness\n"
            "remaining silent | avoids (inferred, confidence=0.7) | confirming foolishness\n"
        )
        repetitive = self._make_record(
            "\n".join(
                ["majority | is (observed, confidence=1.0) | is a group of people"] * 6
            )
        )
        self.assertGreater(
            score_training_sample(diverse), score_training_sample(repetitive)
        )

    def test_speaker_dominated_output_penalised(self) -> None:
        """An output where every triplet has 'the speaker' as subject should score lower."""
        from src.run_spo_training import score_training_sample

        speaker_dominated = self._make_record(
            "the speaker | is (observed, confidence=1.0) | a person\n"
            "the speaker | believes (inferred, confidence=0.8) | something\n"
            "the speaker | says (observed, confidence=1.0) | words\n"
            "the speaker | thinks (inferred, confidence=0.7) | thoughts\n"
        )
        varied_subjects = self._make_record(
            "silence | implies (inferred, confidence=0.8) | wisdom\n"
            "speech | reveals (observed, confidence=1.0) | foolishness\n"
            "the listener | receives (inferred, confidence=0.7) | judgment\n"
            "talking | removes (observed, confidence=1.0) | all doubt\n"
        )
        self.assertGreater(
            score_training_sample(varied_subjects),
            score_training_sample(speaker_dominated),
        )

    def test_score_dataset_normalises_to_max_one(self) -> None:
        """score_dataset should normalise so that the highest score equals 1.0."""
        from src.run_spo_training import score_dataset

        records = [
            self._make_record("a | b (observed, confidence=1.0) | c\n" * 4),
            self._make_record(
                "silence | implies (inferred, confidence=0.8) | wisdom\n"
                "speech | reveals (observed, confidence=1.0) | foolishness\n"
            ),
        ]
        scores = score_dataset(records)
        self.assertAlmostEqual(max(scores), 1.0)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))


class PromptContractTests(unittest.TestCase):
    """Lock PromptContract.from_prompt() parsing and header_score() logic."""

    TWO_SECTION_PROMPT = (
        'Given this quote, extract the implicit reasoning.\n\n'
        'Quote: "Be yourself."\n\n'
        'Generate a response with:\n'
        '1. Non-Entailed Premises\n'
        '2. Entailed Premises\n\n'
        'Format each premise as: subject | relation (tag) | object\n'
        'Response:'
    )
    THREE_SECTION_PROMPT = (
        'Given this quote, extract the implicit reasoning.\n\n'
        'Quote: "Be yourself."\n\n'
        'Generate a response with:\n'
        '1. Non-Entailed Premises\n'
        '2. Entailed Premises\n'
        '3. Throughline\n\n'
        'Response:'
    )

    def test_parses_two_section_prompt(self) -> None:
        contract = PromptContract.from_prompt(self.TWO_SECTION_PROMPT)
        self.assertEqual(contract.expected_headers, ["Non-Entailed Premises:", "Entailed Premises:"])

    def test_parses_three_section_prompt(self) -> None:
        contract = PromptContract.from_prompt(self.THREE_SECTION_PROMPT)
        self.assertEqual(
            contract.expected_headers,
            ["Non-Entailed Premises:", "Entailed Premises:", "Throughline:"],
        )

    def test_empty_contract_on_no_match(self) -> None:
        contract = PromptContract.from_prompt("No generate block here.")
        self.assertEqual(contract.expected_headers, [])
        self.assertEqual(contract.header_score("anything"), 1.0)

    def test_header_score_exact_match(self) -> None:
        contract = PromptContract.from_prompt(self.TWO_SECTION_PROMPT)
        output = "Non-Entailed Premises:\nfoo | is | bar\n\nEntailed Premises:\nbaz | is | qux\n"
        self.assertAlmostEqual(contract.header_score(output), 1.0)

    def test_header_score_garbled(self) -> None:
        contract = PromptContract.from_prompt(self.TWO_SECTION_PROMPT)
        output = "Non-Entailed Prems:\nfoo | is | bar\n\nEntailed Prims:\nbaz | is | qux\n"
        self.assertAlmostEqual(contract.header_score(output), 0.0)

    def test_evaluate_triplet_correctness_uses_contract(self) -> None:
        """Contract-aware evaluation scores higher than hardcoded CANONICAL_HEADERS
        when the prompt only specifies 2 sections (model produces exactly those 2)."""
        contract = PromptContract.from_prompt(self.TWO_SECTION_PROMPT)
        output = (
            "Non-Entailed Premises:\n"
            "silence | implies (inferred, confidence=0.8) | wisdom\n\n"
            "Entailed Premises:\n"
            "talking | confirms (observed, confidence=1.0) | foolishness\n"
        )
        score_with_contract = SPOEvaluator.evaluate_triplet_correctness(output, contract=contract)
        score_without_contract = SPOEvaluator.evaluate_triplet_correctness(output)
        # With 2-section contract the header score is 2/2=1.0; without it 2/3≈0.667
        self.assertGreater(score_with_contract, score_without_contract)


class AssertOutputQualityTests(unittest.TestCase):
    """Lock the post-training regression gate behavior."""

    GOOD_OUTPUT = (
        "Non-Entailed Premises:\n"
        "silence | implies (inferred, confidence=0.8) | wisdom\n\n"
        "Entailed Premises:\n"
        "talking | confirms (observed, confidence=1.0) | foolishness\n\n"
        "Throughline:\n"
        "remaining silent avoids confirming foolishness\n"
    )
    BAD_OUTPUT = "\n".join(
        ["majority | is (observed, confidence=1.0) | is a group of people"] * 8
    )

    def test_good_output_passes_gate(self) -> None:
        """A clean output should pass without raising."""
        assert_output_quality([self.GOOD_OUTPUT], min_avg_score=0.5, min_per_sample_score=0.3)

    def test_repeated_output_fails_gate(self) -> None:
        """An output that triggers the uniqueness hard-zero should fail the gate."""
        with self.assertRaises(AssertionError):
            assert_output_quality([self.BAD_OUTPUT], min_per_sample_score=0.3)

    def test_empty_outputs_raises(self) -> None:
        with self.assertRaises(AssertionError):
            assert_output_quality([])

    def test_gate_uses_prompt_contract(self) -> None:
        """When prompts are provided, the gate uses per-prompt contracts."""
        prompt = (
            'Given this quote.\n\nGenerate a response with:\n'
            '1. Non-Entailed Premises\n'
            '2. Entailed Premises\n\nResponse:'
        )
        # Output satisfies exactly those two sections — should pass
        output = (
            "Non-Entailed Premises:\n"
            "silence | implies (inferred, confidence=0.8) | wisdom\n\n"
            "Entailed Premises:\n"
            "talking | confirms (observed, confidence=1.0) | foolishness\n"
        )
        assert_output_quality([output], prompts=[prompt], min_avg_score=0.5, min_per_sample_score=0.3)

    def test_below_avg_threshold_fails(self) -> None:
        """An output that's individually OK but drags avg below threshold should fail."""
        mediocre = "some text without triplets or headers"
        with self.assertRaises(AssertionError):
            assert_output_quality([mediocre], min_avg_score=0.9, min_per_sample_score=0.0)


if __name__ == "__main__":
    unittest.main()

