"""
Curriculum and evaluation strategy for multi-regimen reasoning training.

This module encodes the research-backed training approach:
1. Warm-start on the confidence-free base reasoning task
2. Add confidence-bearing regimens as auxiliary supervision
3. Judge success by downstream syllogism quality and confidence utility

The strategy is configuration-first so future runs can swap mixture weights,
stage order, or evaluation gates without editing trainer code.
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List


class RegimenName(str, Enum):
    """Supported supervised regimen families."""

    BASE_REASONING = "base_reasoning"
    FACTS_WITH_CONFIDENCE = "facts_with_confidence"
    SYLLOGISM_WITH_CONFIDENCE = "syllogism_with_confidence"


@dataclass
class CurriculumStage:
    """One stage in the staged training curriculum.

    Preconditions:
        regimen_weights contains only known regimen names and all weights are
        non-negative.
    Failure modes:
        `to_dict()` raises if the total weight is zero because such a stage would
        be impossible to run.
    """

    name: str
    description: str
    regimen_weights: Dict[str, float]
    objective: str
    exit_criteria: str

    def to_dict(self) -> Dict[str, Any]:
        total_weight = sum(self.regimen_weights.values())
        if total_weight <= 0:
            raise ValueError(f"Curriculum stage '{self.name}' has zero total weight")
        return asdict(self)


@dataclass
class AblationExperiment:
    """One ablation in the comparison matrix."""

    name: str
    enabled_regimens: List[str]
    hypothesis: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DownstreamEvaluationPlan:
    """Downstream evaluation contract for the training strategy."""

    primary_target: str
    judge_dimensions: List[str]
    confidence_metrics: List[str]
    selection_metric: str
    acceptance_threshold: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingStrategy:
    """Top-level multi-regimen training plan."""

    stages: List[CurriculumStage] = field(default_factory=list)
    ablations: List[AblationExperiment] = field(default_factory=list)
    evaluation: DownstreamEvaluationPlan | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "ablations": [ablation.to_dict() for ablation in self.ablations],
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def default(cls) -> "TrainingStrategy":
        """Return the research-backed default curriculum."""
        return cls(
            stages=[
                CurriculumStage(
                    name="base-warm-start",
                    description="Teach quote -> premises + throughline before introducing score-bearing tasks.",
                    regimen_weights={
                        RegimenName.BASE_REASONING.value: 1.0,
                    },
                    objective="Maximize base syllogism quality before any auxiliary confidence supervision.",
                    exit_criteria="Holdout syllogism quality plateaus or reaches the current acceptance threshold.",
                ),
                CurriculumStage(
                    name="multitask-mix",
                    description="Add confidence-bearing regimens while keeping the base task dominant.",
                    regimen_weights={
                        RegimenName.BASE_REASONING.value: 0.60,
                        RegimenName.FACTS_WITH_CONFIDENCE.value: 0.25,
                        RegimenName.SYLLOGISM_WITH_CONFIDENCE.value: 0.15,
                    },
                    objective="Improve selective syllogism quality without letting confidence imitation dominate.",
                    exit_criteria="Risk-coverage and syllogism quality improve over the base-only run.",
                ),
                CurriculumStage(
                    name="optional-score-refinement",
                    description="Later phase for replacing bootstrap confidence targets with judge or GRPO signals.",
                    regimen_weights={
                        RegimenName.FACTS_WITH_CONFIDENCE.value: 0.50,
                        RegimenName.SYLLOGISM_WITH_CONFIDENCE.value: 0.50,
                    },
                    objective="Refine score-bearing behaviors after the downstream evaluation harness exists.",
                    exit_criteria="Only run if judge-derived or preference-derived labels become available.",
                ),
            ],
            ablations=[
                AblationExperiment(
                    name="base-only",
                    enabled_regimens=[RegimenName.BASE_REASONING.value],
                    hypothesis="Confidence-free base training defines the trunk quality floor.",
                ),
                AblationExperiment(
                    name="base-plus-facts",
                    enabled_regimens=[
                        RegimenName.BASE_REASONING.value,
                        RegimenName.FACTS_WITH_CONFIDENCE.value,
                    ],
                    hypothesis="Premise-level confidence helps the model organize evidence before scoring the argument.",
                ),
                AblationExperiment(
                    name="base-plus-facts-plus-syllogism",
                    enabled_regimens=[
                        RegimenName.BASE_REASONING.value,
                        RegimenName.FACTS_WITH_CONFIDENCE.value,
                        RegimenName.SYLLOGISM_WITH_CONFIDENCE.value,
                    ],
                    hypothesis="Adding argument-level confidence improves selective syllogism quality, not just number copying.",
                ),
            ],
            evaluation=DownstreamEvaluationPlan(
                primary_target="syllogism_quality",
                judge_dimensions=[
                    "faithfulness_to_quote",
                    "entailed_coverage",
                    "non_entailed_exclusion",
                    "coherence_and_minimality",
                ],
                confidence_metrics=[
                    "spearman",
                    "pearson",
                    "auroc",
                    "brier",
                    "ece",
                    "risk_coverage",
                ],
                selection_metric="risk_coverage",
                acceptance_threshold=0.7,
            ),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Write the default multi-regimen training strategy to JSON"
    )
    parser.add_argument("--output", required=True, help="Where to save the strategy JSON")
    args = parser.parse_args()

    TrainingStrategy.default().to_json(args.output)
