"""Unit tests for SPO trainer reward parsing and weighting behavior."""

import unittest

import torch

from src.spo_trainer import SPOEvaluator, SPOReward, SPOTrainer


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


class SPODataQualityScorerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
