"""Prompt-contract tests for the three supervised reasoning regimens."""

import unittest

from src.build_training_regimens import (
    serialize_facts_with_confidence_record,
    serialize_syllogism_with_confidence_record,
)
from src.serialize_training_format import serialize_training_record


def make_structured_record() -> dict:
    """Create one deterministic structured sample for prompt-contract tests."""
    return {
        "quote": "“Be yourself; everyone else is already taken.”",
        "non_entailed_premises": [
            "social conformity | is (observed, confidence=1.0) | undesirable",
        ],
        "entailed_premises": [
            "people | are (observed, confidence=1.0) | unique individuals",
            "authenticity | is (inferred, confidence=0.5) | the only way to be oneself",
        ],
        "syllogism": (
            "One should embrace their own identity rather than imitating others."
        ),
    }


class SupervisionPromptContractTests(unittest.TestCase):
    """Lock the prompt instructions for each training regimen."""

    def test_base_reasoning_prompt_contract(self) -> None:
        """Base reasoning prompt uses tag-only format (no confidence); output strips confidence."""
        record = serialize_training_record(make_structured_record())

        self.assertEqual(
            record["input_text"],
            "\n".join(
                [
                    "Given this quote, extract the explicit and implicit reasoning facts.",
                    "",
                    'Quote: "Be yourself; everyone else is already taken."',
                    "",
                    "Generate a response with:",
                    "1. Non-Entailed Premises",
                    "2. Entailed Premises",
                    "3. Throughline",
                    "",
                    "Format each premise as: subject | predicate (observed) | object",
                    '- tag: "observed" for explicit facts, "inferred" for derived facts',
                    "",
                    "VERBATIM EXTRACTION RULE (Entailed Premises only):",
                    "- Subject, predicate, and object MUST be exact, verbatim text copied from the quote above.",
                    "- Do NOT paraphrase, summarize, or invent language for Entailed fields.",
                    "- After each verbatim triplet, add a transliteration on the NEXT LINE in parentheses,",
                    "  using the same S | P (tag) | O format but with plain-English paraphrase:",
                    "  verbatim:        The unexamined life | is not worth (observed) | living",
                    "  transliteration: (A life without self-reflection | has no (inferred) | value)",
                    "IMPORTANT: The Entailed Premises section MUST contain at least one triplet.",
                    "Never leave Entailed Premises empty.",
                    "",
                    "Response:",
                ]
            ),
        )
        self.assertEqual(
            record["output_text"],
            "\n".join(
                [
                    "Non-Entailed Premises:",
                    "social conformity | is (observed) | undesirable",
                    "",
                    "Entailed Premises:",
                    "people | are (observed) | unique individuals",
                    "authenticity | is (inferred) | the only way to be oneself",
                    "",
                    "Throughline:",
                    "One should embrace their own identity rather than imitating others.",
                ]
            ),
        )

    def test_facts_with_confidence_prompt_contract(self) -> None:
        """Facts-with-confidence should preserve the premise-extraction instructions."""
        record = serialize_facts_with_confidence_record(make_structured_record())

        self.assertEqual(
            record["input_text"],
            "\n".join(
                [
                    "Given this quote, extract the implicit reasoning facts.",
                    "",
                    'Quote: "Be yourself; everyone else is already taken."',
                    "",
                    "Generate a response with:",
                    "1. Non-Entailed Premises",
                    "2. Entailed Premises",
                    "",
                    "Format each premise as: subject | (tag, confidence=N) | object",
                    '- tag: "observed" for explicit facts, "inferred" for derived facts',
                    "- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred",
                    "",
                    "Response:",
                ]
            ),
        )
        self.assertEqual(
            record["output_text"],
            "\n".join(
                [
                    "Non-Entailed Premises:",
                    "social conformity | is (observed, confidence=1.0) | undesirable",
                    "",
                    "Entailed Premises:",
                    "people | are (observed, confidence=1.0) | unique individuals",
                    "authenticity | is (inferred, confidence=0.5) | the only way to be oneself",
                ]
            ),
        )

    def test_syllogism_with_confidence_prompt_contract(self) -> None:
        """Syllogism-with-confidence should preserve fact-conditioned scoring instructions."""
        record = serialize_syllogism_with_confidence_record(make_structured_record())

        self.assertEqual(
            record["input_text"],
            "\n".join(
                [
                    "Given this quote and the extracted facts, score the argument throughline.",
                    "",
                    'Quote: "Be yourself; everyone else is already taken."',
                    "",
                    "Non-Entailed Premises:",
                    "social conformity | is (observed, confidence=1.0) | undesirable",
                    "",
                    "Entailed Premises:",
                    "people | are (observed, confidence=1.0) | unique individuals",
                    "authenticity | is (inferred, confidence=0.5) | the only way to be oneself",
                    "",
                    "Generate a response with:",
                    "1. Throughline",
                    "2. Confidence",
                    "",
                    "Response:",
                ]
            ),
        )
        self.assertEqual(
            record["output_text"],
            "\n".join(
                [
                    "Throughline:",
                    "  One should embrace their own identity rather than imitating others.",
                    "",
                    "Confidence:",
                    "  0.75",
                ]
            ),
        )

    def test_transliteration_interleaving_in_output_text(self) -> None:
        """When entailed_transliterations is provided, confidence is stripped from both verbatim and paren lines."""
        rec = make_structured_record()
        rec["entailed_transliterations"] = [
            "(human beings | possess (inferred, confidence=0.9) | distinct identities)",
            "(being genuine | is (inferred, confidence=0.8) | the sole path to true selfhood)",
        ]
        record = serialize_training_record(rec)
        # confidence stripped from both verbatim triplet and its transliteration
        self.assertIn(
            "people | are (observed) | unique individuals\n"
            "(human beings | possess (inferred) | distinct identities)",
            record["output_text"],
        )
        self.assertIn(
            "authenticity | is (inferred) | the only way to be oneself\n"
            "(being genuine | is (inferred) | the sole path to true selfhood)",
            record["output_text"],
        )

    def test_transliteration_absent_when_not_provided(self) -> None:
        """When entailed_transliterations is absent, output_text is unchanged from v12 format."""
        record = serialize_training_record(make_structured_record())
        # No paren lines should appear in the Entailed block
        in_entailed = False
        for line in record["output_text"].splitlines():
            if line.startswith("Entailed Premises:"):
                in_entailed = True
                continue
            if line.startswith("Throughline:"):
                break
            if in_entailed and line.startswith("(") and "|" in line:
                self.fail(f"Unexpected transliteration line without transliterations: {line!r}")


if __name__ == "__main__":
    unittest.main()
