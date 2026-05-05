"""Synthetic reasoning dataset generation for training.

Generates reasoning examples with structured triplets (entailed/non-entailed premises
and syllogisms) from quotes using LLM generation and validation.
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

from .serialize_training_format import build_base_reasoning_prompt


class TripletItem(BaseModel):
    """A single triplet in premise format."""
    subject: str
    relation: str
    object_: str = Field(alias="object")
    tag: str = "inferred"  # "observed" or "inferred"
    confidence: float = 0.5

    class Config:
        populate_by_name = True


class ReasoningExample(BaseModel):
    """A complete reasoning example with premises and conclusion."""
    quote: str
    non_entailed_premises: List[TripletItem] = Field(default_factory=list)
    entailed_premises: List[TripletItem] = Field(default_factory=list)
    syllogism: str  # The throughline/conclusion


@dataclass
class GenerationPrompt:
    """Template for generating reasoning examples."""

    @staticmethod
    def extract_reasoning(quote: str) -> str:
        """Generate prompt for extracting reasoning from quote."""
        return f"""Given this quote, extract the implicit reasoning:

Quote: "{quote}"

Generate a response with:
1. Non-Entailed Premises (facts mentioned but not supporting the main inference)
2. Entailed Premises (facts that lead to the conclusion)
3. Syllogism (the core reasoning thread)

Format each premise as: subject | relation (tag, confidence=X) | object
- tag: "observed" for explicit facts, "inferred" for derived
- confidence: 1.0 for observed, 0.5-0.9 for inferred

Response:
"""

    @staticmethod
    def validate_reasoning(quote: str, reasoning: str) -> str:
        """Generate prompt to validate reasoning quality."""
        return f"""Evaluate this reasoning extraction:

Quote: "{quote}"

Reasoning:
{reasoning}

Score on:
1. Accuracy (does reasoning match quote?)
2. Completeness (are key inferences captured?)
3. Clarity (is structure coherent?)

Return PASS or FAIL with explanation.
"""


class SyntheticReasoningGenerator:
    """Generate synthetic reasoning datasets."""

    def __init__(self, llm_client=None):
        """Initialize with optional LLM client for generation."""
        self.llm_client = llm_client
        self.examples: List[ReasoningExample] = []

    def add_example(self, example: ReasoningExample):
        """Add a reasoning example to the dataset."""
        self.examples.append(example)

    def generate_from_quotes(
        self,
        quotes: List[str],
        llm_generate_fn=None,
    ) -> List[ReasoningExample]:
        """Generate reasoning examples from quotes.

        Args:
            quotes: List of quotes to extract reasoning from
            llm_generate_fn: Function that takes prompt and returns response

        Returns:
            List of ReasoningExample objects
        """
        results = []

        for quote in quotes:
            if llm_generate_fn is None:
                # Without LLM, create template
                example = self._create_template_example(quote)
            else:
                # With LLM, generate
                prompt = GenerationPrompt.extract_reasoning(quote)
                response = llm_generate_fn(prompt)
                example = self._parse_response(quote, response)

            results.append(example)
            self.examples.append(example)

        return results

    def _create_template_example(self, quote: str) -> ReasoningExample:
        """Create a template example for manual filling."""
        return ReasoningExample(
            quote=quote,
            non_entailed_premises=[],
            entailed_premises=[],
            syllogism="[To be filled in]",
        )

    def _parse_response(self, quote: str, response: str) -> ReasoningExample:
        """Parse LLM response into ReasoningExample."""
        # This is a simplified parser - in practice, use structured output
        try:
            # Try to parse as JSON first
            data = json.loads(response)
            return ReasoningExample(
                quote=quote,
                non_entailed_premises=[
                    TripletItem(**p) for p in data.get("non_entailed_premises", [])
                ],
                entailed_premises=[
                    TripletItem(**p) for p in data.get("entailed_premises", [])
                ],
                syllogism=data.get("syllogism", ""),
            )
        except json.JSONDecodeError:
            # Fall back to template
            return self._create_template_example(quote)

    def _format_triplet(self, triplet: TripletItem, include_confidence: bool) -> str:
        """Serialize a triplet for training or inspection output."""
        if include_confidence:
            return (
                f"  {triplet.subject} | {triplet.relation} "
                f"({triplet.tag}, confidence={triplet.confidence}) | {triplet.object_}"
            )
        return f"  {triplet.subject} | {triplet.relation} ({triplet.tag}) | {triplet.object_}"

    def export_to_jsonl(self, path: str, include_confidence: bool = False):
        """Export examples to JSONL format for training.

        Numeric confidence is stripped by default so the training target remains
        text-first and downstream scoring stays permeable.
        """
        with open(path, "w") as f:
            for example in self.examples:
                input_text = build_base_reasoning_prompt(example.quote)

                # Output: pedagogical format (Non-Entailed → Entailed → Syllogism)
                non_entailed_str = "\n".join(
                    self._format_triplet(p, include_confidence=include_confidence)
                    for p in example.non_entailed_premises
                )
                entailed_str = "\n".join(
                    self._format_triplet(p, include_confidence=include_confidence)
                    for p in example.entailed_premises
                )

                output_text = f"""Non-Entailed Premises:
{non_entailed_str or "  N/A"}

Entailed Premises:
{entailed_str or "  N/A"}

Throughline:
  {example.syllogism}"""

                record = {
                    "input_text": input_text,
                    "output_text": output_text,
                }
                f.write(json.dumps(record) + "\n")

    def export_to_json(self, path: str):
        """Export examples to JSON format (for inspection/validation)."""
        data = [
            {
                "quote": ex.quote,
                "non_entailed_premises": [p.dict() for p in ex.non_entailed_premises],
                "entailed_premises": [p.dict() for p in ex.entailed_premises],
                "syllogism": ex.syllogism,
            }
            for ex in self.examples
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def stats(self) -> Dict[str, int]:
        """Return dataset statistics."""
        return {
            "total_examples": len(self.examples),
            "examples_with_non_entailed": sum(
                1 for ex in self.examples if ex.non_entailed_premises
            ),
            "examples_with_entailed": sum(
                1 for ex in self.examples if ex.entailed_premises
            ),
            "total_non_entailed_premises": sum(
                len(ex.non_entailed_premises) for ex in self.examples
            ),
            "total_entailed_premises": sum(
                len(ex.entailed_premises) for ex in self.examples
            ),
        }
