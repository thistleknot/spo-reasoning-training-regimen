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
        """Base reasoning uses explicit instruction prompt with confidence annotations."""
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
                    "Format each premise as: subject | predicate (observed, confidence=0.9) | object",
                    '- tag: "observed" for explicit facts, "inferred" for derived facts',
                    "- confidence: a decimal number in [0,1] — e.g. 1.0 for observed, 0.7 for inferred",
                    "",
                    "VERBATIM EXTRACTION RULE (Entailed Premises only):",
                    "- Subject, predicate, and object MUST be exact, verbatim text copied from the quote above.",
                    "- Do NOT paraphrase, summarize, or invent language for Entailed fields.",
                    "- Parenthetical clarifications may be added AFTER verbatim text: verbatim text (clarification)",
                    "- Invariant: strip all (...) from a triplet and the remaining text must be verbatim from the quote.",
                    '  Example: "Don\'t be | satisfied with (inferred, confidence=0.7) | stories"',
                    '  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"',
                    "- Non-Entailed Premises and the Throughline may use your own words.",
                    "",
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
                    "social conformity | is (observed, confidence=1.0) | undesirable",
                    "",
                    "Entailed Premises:",
                    "people | are (observed, confidence=1.0) | unique individuals",
                    "authenticity | is (inferred, confidence=0.5) | the only way to be oneself",
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


if __name__ == "__main__":
    unittest.main()
