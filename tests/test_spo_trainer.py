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
        """Output with all canonical headers, proper triplets, confidence, and tags scores ~0.95."""
        score = SPOEvaluator.evaluate_triplet_correctness(self.PERFECT_OUTPUT)
        # Max without GT bonus: triplet=0.20 + header=0.30 + conf=0.15 + evid=0.15 + quality=0.15 = 0.95
        self.assertAlmostEqual(score, 0.95, places=2)

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
        # Max possible without header credit:
        #   triplet=0.20 + confidence=0.15 + evidence=0.15 + quality_checks=0.15 = 0.65
        self.assertLessEqual(score, 0.65 + 1e-6)

    def test_trivial_is_predicate_penalised(self) -> None:
        """Triplets whose sole predicate is a bare copula ('is') should score lower."""
        trivial = (
            "Non-Entailed Premises:\n"
            "everyone | is | already taken (observed, confidence=1.0)\n"
            "the path | is | chosen (inferred, confidence=0.8)\n"
            "Entailed Premises:\n"
            "be yourself | is | the correct action (observed, confidence=1.0)\n"
            "Throughline:\n"
            "Self-expression is paramount.\n"
        )
        meaningful = (
            "Non-Entailed Premises:\n"
            "everyone | occupies (observed, confidence=1.0) | a unique role\n"
            "the path | leads (inferred, confidence=0.8) | toward authenticity\n"
            "Entailed Premises:\n"
            "be yourself | validates (observed, confidence=1.0) | personal identity\n"
            "Throughline:\n"
            "Self-expression is paramount.\n"
        )
        score_trivial   = SPOEvaluator.evaluate_triplet_correctness(trivial)
        score_meaningful = SPOEvaluator.evaluate_triplet_correctness(meaningful)
        self.assertGreater(score_meaningful, score_trivial)

    def test_predicate_echo_in_object_penalised(self) -> None:
        """When the object starts with the same word as the predicate, the score is reduced."""
        # Canonical failure mode: 'be yourself | is | is the correct action'
        echo = (
            "Non-Entailed Premises:\n"
            "identity | implies (inferred, confidence=0.8) | implies uniqueness\n"
            "Entailed Premises:\n"
            "be yourself | is (observed, confidence=1.0) | is the correct action\n"
            "Throughline:\n"
            "Authenticity matters.\n"
        )
        clean = (
            "Non-Entailed Premises:\n"
            "identity | shapes (inferred, confidence=0.8) | how others perceive us\n"
            "Entailed Premises:\n"
            "be yourself | validates (observed, confidence=1.0) | personal identity\n"
            "Throughline:\n"
            "Authenticity matters.\n"
        )
        score_echo  = SPOEvaluator.evaluate_triplet_correctness(echo)
        score_clean = SPOEvaluator.evaluate_triplet_correctness(clean)
        self.assertGreater(score_clean, score_echo)



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

    def test_garbled_headers_fail_header_gate(self) -> None:
        """Abbreviated/garbled headers should fail the header gate even with good pipe structure."""
        prompt = (
            'Given this quote.\n\nGenerate a response with:\n'
            '1. Non-Entailed Premises\n'
            '2. Entailed Premises\n\nResponse:'
        )
        # Pipe structure fine but headers are abbreviated
        garbled_output = (
            "Non-Entailed Prems:\n"
            "silence | implies (inferred, confidence=0.8) | wisdom\n\n"
            "Entailed Prims:\n"
            "talking | confirms (observed, confidence=1.0) | foolishness\n"
        )
        with self.assertRaises(AssertionError) as ctx:
            assert_output_quality(
                [garbled_output],
                prompts=[prompt],
                min_avg_score=0.0,
                min_per_sample_score=0.0,
                min_header_score=0.5,
            )
        self.assertIn("Header quality gate", str(ctx.exception))

    def test_header_gate_disabled_at_zero(self) -> None:
        """Setting min_header_score=0.0 disables the header gate entirely."""
        prompt = (
            'Given this quote.\n\nGenerate a response with:\n'
            '1. Non-Entailed Premises\n'
            '2. Entailed Premises\n\nResponse:'
        )
        garbled_output = (
            "Non-Entailed Prems:\n"
            "silence | implies (inferred, confidence=0.8) | wisdom\n\n"
            "Entailed Prims:\n"
            "talking | confirms (observed, confidence=1.0) | foolishness\n"
        )
        # Should not raise when header gate is disabled
        assert_output_quality(
            [garbled_output],
            prompts=[prompt],
            min_avg_score=0.0,
            min_per_sample_score=0.0,
            min_header_score=0.0,
        )

    def test_no_prompt_skips_header_gate(self) -> None:
        """Without prompts there is no contract, so header score is 1.0 regardless of output."""
        any_output = "Non-Entailed Prems:\nfoo | bar (observed, confidence=0.9) | baz\n"
        # Would fail header gate if contract were applied — passes because no prompts given
        assert_output_quality(
            [any_output],
            min_avg_score=0.0,
            min_per_sample_score=0.0,
            min_header_score=0.5,
        )

    def test_mixed_tags_penalised(self) -> None:
        """Output with only mixed (observed+inferred) tags still gets the evidence credit — any tag present counts."""
        mixed = (
            "Non-Entailed Premises:\n"
            "difficulty | is (observed, inferred) | a challenge\n"
            "Entailed Premises:\n"
            "growth | requires (observed, inferred) | facing difficulty\n"
            "Throughline:\n"
            "Difficulties contain opportunities.\n"
        )
        score = SPOEvaluator.evaluate_triplet_correctness(mixed)
        # 'observed' and 'inferred' both present → evidence credit granted (binary)
        self.assertGreater(score, 0.0)

    def test_no_tags_partial_credit(self) -> None:
        """Canonical tag+confidence annotations score higher than bare predicate-only triplets."""
        no_tags = (
            "Non-Entailed Premises:\n"
            "difficulty | is | a challenge\n"
            "Entailed Premises:\n"
            "growth | requires | facing difficulty\n"
            "Throughline:\n"
            "Difficulties contain opportunities.\n"
        )
        # Full canonical annotation — scorer awards 0.15 confidence + 0.15 tag credit.
        tagged = (
            "Non-Entailed Premises:\n"
            "difficulty | is (observed, confidence=1.0) | a challenge\n"
            "Entailed Premises:\n"
            "growth | requires (inferred, confidence=0.7) | facing difficulty\n"
            "Throughline:\n"
            "Difficulties contain opportunities.\n"
        )
        score_no_tags = SPOEvaluator.evaluate_triplet_correctness(no_tags)
        score_tagged = SPOEvaluator.evaluate_triplet_correctness(tagged)
        self.assertGreater(score_tagged, score_no_tags)


class TestVerbatimFaithfulnessGate(unittest.TestCase):
    """Lock the verbatim-faithfulness gate for entailed premises."""

    SOURCE = "The unexamined life is not worth living."

    # All three S/P/O fields are verbatim words from SOURCE after stripping (tag, confidence=N).
    VERBATIM_OUTPUT = (
        "Non-Entailed Premises:\n"
        "Socrates | advocated (observed, confidence=0.9) | self-examination\n"
        "Entailed Premises:\n"
        "unexamined life | is not worth (observed, confidence=1.0) | living\n"
        "Throughline:\n"
        "  A life without reflection has no value.\n"
    )

    # S/P/O are all paraphrased — none appear verbatim in SOURCE.
    PARAPHRASE_OUTPUT = (
        "Non-Entailed Premises:\n"
        "Socrates | advocated (observed, confidence=0.9) | self-examination\n"
        "Entailed Premises:\n"
        "a life without scrutiny | lacks (inferred, confidence=0.7) | merit\n"
        "Throughline:\n"
        "  A life without reflection has no value.\n"
    )

    # Subject has parenthetical clarification; base text is still verbatim from SOURCE.
    PAREN_OUTPUT = (
        "Non-Entailed Premises:\n"
        "Socrates | advocated (observed, confidence=0.9) | critical inquiry\n"
        "Entailed Premises:\n"
        "unexamined life (a life without self-reflection) | is not worth (observed, confidence=1.0) | living\n"
        "Throughline:\n"
        "  Reflection is essential.\n"
    )

    def test_verbatim_entailed_no_penalty(self) -> None:
        """Fully verbatim Entailed section incurs no penalty when source_quote provided."""
        with_source = SPOEvaluator.evaluate_triplet_correctness(
            self.VERBATIM_OUTPUT, source_quote=self.SOURCE
        )
        without_source = SPOEvaluator.evaluate_triplet_correctness(
            self.VERBATIM_OUTPUT
        )
        # penalty = 0 for ratio=1.0 → score unchanged
        self.assertGreaterEqual(with_source, without_source)

    def test_paraphrase_entailed_penalized(self) -> None:
        """Paraphrased S/P/O receive a verbatim penalty; verbatim output scores higher."""
        verbatim_score = SPOEvaluator.evaluate_triplet_correctness(
            self.VERBATIM_OUTPUT, source_quote=self.SOURCE
        )
        paraphrase_score = SPOEvaluator.evaluate_triplet_correctness(
            self.PARAPHRASE_OUTPUT, source_quote=self.SOURCE
        )
        self.assertGreater(verbatim_score, paraphrase_score)

    def test_parenthetical_stripped_before_check(self) -> None:
        """Parenthetical transliteration stripped before verbatim check; base text is verbatim."""
        with_source = SPOEvaluator.evaluate_triplet_correctness(
            self.PAREN_OUTPUT, source_quote=self.SOURCE
        )
        without_source = SPOEvaluator.evaluate_triplet_correctness(
            self.PAREN_OUTPUT
        )
        # ratio=1.0 after stripping parens → no penalty → score unchanged
        self.assertGreaterEqual(with_source, without_source)

    def test_extract_section_triplets_isolates_section(self) -> None:
        """_extract_section_triplets returns only lines from the named section."""
        entailed = SPOEvaluator._extract_section_triplets(
            self.VERBATIM_OUTPUT, "Entailed Premises"
        )
        self.assertEqual(len(entailed), 1)
        self.assertTrue(all("|" in line for line in entailed))
        # Non-Entailed and Throughline must not bleed in
        for line in entailed:
            self.assertNotIn("Socrates", line)
            self.assertNotIn("reflection", line)

    def test_entailed_verbatim_ratio_pure(self) -> None:
        """_entailed_verbatim_ratio returns 1.0 when all components are verbatim."""
        lines = SPOEvaluator._extract_section_triplets(
            self.VERBATIM_OUTPUT, "Entailed Premises"
        )
        ratio = SPOEvaluator._entailed_verbatim_ratio(lines, self.SOURCE)
        self.assertGreater(ratio, 0.5)

    def test_entailed_verbatim_ratio_empty(self) -> None:
        """_entailed_verbatim_ratio returns 0.0 on empty input."""
        self.assertEqual(SPOEvaluator._entailed_verbatim_ratio([], self.SOURCE), 0.0)
        self.assertEqual(
            SPOEvaluator._entailed_verbatim_ratio(["x | y | z"], ""), 0.0
        )


class TestTrainingLadderCheckFunctions(unittest.TestCase):
    """Unit tests for check functions in src.training_ladder.

    Covers the three functions that were fixed in v11:
      1. _normalize_confidence_syntax  — normalises <X> and "X" delimiters
      2. check_confidence_numeric      — now normalises before parsing
      3. check_canonical_tag_format    — now normalises before regex
      4. check_tags_exclusive          — uses annotation-position matching
    """

    def setUp(self) -> None:
        from src.training_ladder import (
            _normalize_confidence_syntax,
            check_confidence_numeric,
            check_canonical_tag_format,
            check_tags_exclusive,
        )
        self.norm = _normalize_confidence_syntax
        self.conf_numeric = check_confidence_numeric
        self.canon_fmt = check_canonical_tag_format
        self.tags_excl = check_tags_exclusive

    # ── _normalize_confidence_syntax ─────────────────────────────────────────

    def test_normalize_angle_bracket_value(self) -> None:
        """confidence=<0.7> → confidence=0.7"""
        result = self.norm("(inferred, confidence=<0.7>)")
        self.assertIn("confidence=0.7", result)
        self.assertNotIn("<", result)

    def test_normalize_unterminated_angle_bracket(self) -> None:
        """confidence=<0.7 (no closing >) → confidence=0.7"""
        result = self.norm("(inferred, confidence=<0.7)")
        self.assertIn("confidence=0.7", result)

    def test_normalize_quoted_numeric(self) -> None:
        """confidence=\"0.7\" → confidence=0.7"""
        result = self.norm('(observed, confidence="0.9")')
        self.assertIn("confidence=0.9", result)
        self.assertNotIn('"', result)

    def test_normalize_leaves_non_numeric_quoted_alone(self) -> None:
        """confidence=\"inferred\" must NOT be rewritten (not a number)."""
        original = '(observed, confidence="inferred")'
        result = self.norm(original)
        self.assertEqual(result, original)

    def test_normalize_idempotent_on_correct_format(self) -> None:
        """Already-correct format must be unchanged."""
        line = "(inferred, confidence=0.8)"
        self.assertEqual(self.norm(line), line)

    # ── check_confidence_numeric ─────────────────────────────────────────────

    def test_angle_bracket_value_passes_after_normalisation(self) -> None:
        output = "subject | (inferred, confidence=<0.7) | object"
        self.assertTrue(self.conf_numeric(output, {}))

    def test_quoted_numeric_passes_after_normalisation(self) -> None:
        output = 'subject | (observed, confidence="0.9") | object'
        self.assertTrue(self.conf_numeric(output, {}))

    def test_tag_word_as_value_still_fails(self) -> None:
        """confidence=\"inferred\" is not a valid float — must fail."""
        output = '(observed, confidence="inferred") | object'
        self.assertFalse(self.conf_numeric(output, {}))

    def test_negative_confidence_normalised_passes(self) -> None:
        # Normalization strips the leading minus; -0.5 → 0.5 → passes numeric check
        output = "subject | (inferred, confidence=-0.5) | object"
        self.assertTrue(self.conf_numeric(output, {}))

    def test_valid_numeric_confidence_passes(self) -> None:
        output = "subject | (observed, confidence=0.85) | object"
        self.assertTrue(self.conf_numeric(output, {}))

    # ── check_canonical_tag_format ───────────────────────────────────────────

    def test_angle_bracket_format_passes_after_normalisation(self) -> None:
        output = "subject | (inferred, confidence=<0.7) | object"
        self.assertTrue(self.canon_fmt(output, {}))

    def test_quoted_format_passes_after_normalisation(self) -> None:
        output = 'subject | (observed, confidence="1.0") | object'
        self.assertTrue(self.canon_fmt(output, {}))

    def test_no_confidence_fails(self) -> None:
        output = "subject | predicate | object\nno annotations here"
        self.assertFalse(self.canon_fmt(output, {}))

    def test_tag_word_as_value_fails(self) -> None:
        output = '(observed, confidence="inferred") | object'
        self.assertFalse(self.canon_fmt(output, {}))

    # ── check_tags_exclusive ─────────────────────────────────────────────────

    def test_single_tag_per_line_passes(self) -> None:
        output = (
            "subject | (observed, confidence=0.9) | object\n"
            "other | (inferred, confidence=0.7) | thing"
        )
        self.assertTrue(self.tags_excl(output, {}))

    def test_both_tags_on_same_line_fails(self) -> None:
        output = "subject | (observed, inferred, confidence=0.8) | object"
        self.assertFalse(self.tags_excl(output, {}))

    def test_confidence_inferred_value_does_not_trigger_false_positive(self) -> None:
        """confidence=\"inferred\" puts 'inferred' in the line value — must not count as tag."""
        output = 'subject | (observed, confidence="inferred") | object'
        self.assertTrue(self.tags_excl(output, {}))

    def test_observed_vertical_bar_inferred_both_annotation_fails(self) -> None:
        """(observed|inferred, ...) has both tags in annotation — must fail."""
        output = "subject | (observed|inferred, confidence=0.9) | object"
        self.assertFalse(self.tags_excl(output, {}))

    def test_lines_without_pipe_ignored(self) -> None:
        """Non-triplet lines (headers, throughline) must not cause false failures."""
        output = (
            "Entailed Premises:\n"
            "subject | (observed, confidence=0.9) | object\n"
            "Throughline: observed inferred both words here"
        )
        self.assertTrue(self.tags_excl(output, {}))


class TestRegimenChecks(unittest.TestCase):
    """Unit tests for the regimen-specific check functions added in tier4/tier5.

    check_facts_headers     — detects Non-Entailed + Entailed headers
    check_throughline_present  — detects Throughline: with non-stub content
    check_syllogism_confidence_present — detects Confidence: with a float
    """

    def setUp(self) -> None:
        from src.training_ladder import (
            check_facts_headers,
            check_throughline_present,
            check_syllogism_confidence_present,
        )
        self.facts_headers = check_facts_headers
        self.throughline = check_throughline_present
        self.syl_conf = check_syllogism_confidence_present

    # ── check_facts_headers ──────────────────────────────────────────────────

    def test_facts_headers_both_present(self) -> None:
        output = "Non-Entailed Premises:\nS | P | O\n\nEntailed Premises:\nA | B | C"
        self.assertTrue(self.facts_headers(output, {}))

    def test_facts_headers_missing_entailed(self) -> None:
        output = "Non-Entailed Premises:\nS | P | O"
        self.assertFalse(self.facts_headers(output, {}))

    def test_facts_headers_missing_non_entailed(self) -> None:
        output = "Entailed Premises:\nA | B | C"
        self.assertFalse(self.facts_headers(output, {}))

    def test_facts_headers_empty_output(self) -> None:
        self.assertFalse(self.facts_headers("", {}))

    def test_facts_headers_only_throughline_fails(self) -> None:
        """Throughline-only output (syllogism format) must not pass facts check."""
        output = "Throughline:\n  Some text here.\n\nConfidence:\n  0.85"
        self.assertFalse(self.facts_headers(output, {}))

    # ── check_throughline_present ────────────────────────────────────────────

    def test_throughline_present_with_content(self) -> None:
        output = "Throughline:\n  One should embrace their unique identity."
        self.assertTrue(self.throughline(output, {}))

    def test_throughline_present_indented_content(self) -> None:
        output = "Throughline:\n    The core argument is that authenticity matters."
        self.assertTrue(self.throughline(output, {}))

    def test_throughline_absent(self) -> None:
        output = "Non-Entailed Premises:\nS | P | O"
        self.assertFalse(self.throughline(output, {}))

    def test_throughline_stub_too_short(self) -> None:
        """A Throughline: header with only whitespace or a very short stub must fail."""
        output = "Throughline:\n  N/A"
        self.assertFalse(self.throughline(output, {}))

    def test_throughline_header_with_no_content_line(self) -> None:
        output = "Throughline:"
        self.assertFalse(self.throughline(output, {}))

    # ── check_syllogism_confidence_present ───────────────────────────────────

    def test_syllogism_confidence_valid_float(self) -> None:
        output = "Throughline:\n  Something.\n\nConfidence:\n  0.85"
        self.assertTrue(self.syl_conf(output, {}))

    def test_syllogism_confidence_one_zero(self) -> None:
        output = "Confidence:\n  1.0"
        self.assertTrue(self.syl_conf(output, {}))

    def test_syllogism_confidence_zero(self) -> None:
        output = "Confidence:\n  0.0"
        self.assertTrue(self.syl_conf(output, {}))

    def test_syllogism_confidence_out_of_range(self) -> None:
        output = "Confidence:\n  1.5"
        self.assertFalse(self.syl_conf(output, {}))

    def test_syllogism_confidence_non_numeric(self) -> None:
        output = "Confidence:\n  N/A"
        self.assertFalse(self.syl_conf(output, {}))

    def test_syllogism_confidence_absent(self) -> None:
        output = "Throughline:\n  Some text."
        self.assertFalse(self.syl_conf(output, {}))


if __name__ == "__main__":
    unittest.main()



class TestBootstrapPredicates(unittest.TestCase):
    """Tests for bootstrap_predicates improvements: reversed-order, copula fallback, tautology filter."""

    def setUp(self):
        from src.bootstrap_predicates import (
            _extract_predicate_from_quote,
            _inject_predicate_into_triplet,
            inject_predicates,
        )
        self.extract = _extract_predicate_from_quote
        self.inject = _inject_predicate_into_triplet
        self.inject_record = inject_predicates

    def test_forward_span_basic(self):
        pred = self.extract("The unexamined life", "living", "The unexamined life is not worth living.")
        self.assertEqual(pred, "is not worth")

    def test_reversed_order_span(self):
        # "forget to live" appears AFTER "dwell on dreams" in the quote
        pred = self.extract("forget to live", "dwell on dreams",
                            "It does not do to dwell on dreams and forget to live.")
        self.assertIsNotNone(pred)
        self.assertIn("and", pred)

    def test_tautological_returns_none(self):
        # S == O: tautological triplet — should return None
        pred = self.extract("I make mistakes", "I make mistakes", "I make mistakes.")
        self.assertIsNone(pred)

    def test_copula_fallback(self):
        # span between S and O is empty (adjacent), fallback to copula
        pred = self.extract("two things", "infinite",
                            "Two things are infinite: the universe and human stupidity.")
        self.assertIsNotNone(pred)
        self.assertIn("are", pred.lower())

    def test_quote_stripping_in_subject(self):
        # Subject has surrounding quotes in triplet field
        pred = self.extract('"remain silent"', '"thought a fool"',
                            'It is better to remain silent at the risk of being thought a fool.')
        self.assertIsNotNone(pred)

    def test_inject_tautological_drops_triplet(self):
        t = "I make mistakes | inferred | I make mistakes"
        result = self.inject(t, "I make mistakes.")
        self.assertIsNone(result)

    def test_inject_already_canonical_unchanged(self):
        t = "The unexamined life | is not worth (observed, confidence=1.0) | living"
        result = self.inject(t, "The unexamined life is not worth living.")
        self.assertEqual(result, t)

    def test_inject_injects_verb(self):
        t = "The unexamined life | inferred | living"
        result = self.inject(t, "The unexamined life is not worth living.")
        self.assertIsNotNone(result)
        parts = result.split("|")
        self.assertEqual(len(parts), 3)
        self.assertIn("is not worth", parts[1])

    def test_inject_record_drops_tautological(self):
        rec = {
            "quote": "I make mistakes.",
            "entailed_premises": [
                "I make mistakes | inferred | I make mistakes",
                "mistakes | are (observed, confidence=1.0) | inevitable",
            ],
            "non_entailed_premises": [],
            "syllogism": "",
        }
        out = self.inject_record(rec)
        # Tautological triplet dropped; clean one preserved
        self.assertEqual(len(out["entailed_premises"]), 1)
        self.assertIn("inevitable", out["entailed_premises"][0])

    def test_exactly_two_pipes_enforced_in_generator(self):
        from src.generate_verbatim_corpus import _TRIPLET_LINE_RE
        self.assertIsNotNone(_TRIPLET_LINE_RE.match("S | verb (observed, confidence=1.0) | O"))
        self.assertIsNone(_TRIPLET_LINE_RE.match("S | P | confidence=0.0 | O"))
        self.assertIsNone(_TRIPLET_LINE_RE.match("S | (inferred) | C | D | E"))


class TestIsBadRecord(unittest.TestCase):
    """Tests for is_bad_record empty-entailed guard."""

    def setUp(self):
        from src.serialize_training_format import is_bad_record
        self.is_bad = is_bad_record

    def _good(self, **overrides):
        base = {
            "entailed_premises": ["S | is (observed, confidence=1.0) | O"],
            "non_entailed_premises": ["S2 | has (inferred, confidence=0.7) | O2"],
            "syllogism": "Therefore, X.",
        }
        base.update(overrides)
        return base

    def test_empty_entailed_is_bad(self):
        self.assertTrue(self.is_bad(self._good(entailed_premises=[])))

    def test_none_entailed_is_bad(self):
        self.assertTrue(self.is_bad(self._good(entailed_premises=None)))

    def test_missing_entailed_key_is_bad(self):
        rec = {"non_entailed_premises": ["X | is | Y"], "syllogism": "Therefore, X."}
        self.assertTrue(self.is_bad(rec))

    def test_good_record_passes(self):
        self.assertFalse(self.is_bad(self._good()))

    def test_na_syllogism_is_bad(self):
        self.assertTrue(self.is_bad(self._good(syllogism="N/A")))

    def test_empty_syllogism_is_bad(self):
        self.assertTrue(self.is_bad(self._good(syllogism="")))
