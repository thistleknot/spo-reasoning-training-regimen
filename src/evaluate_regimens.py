"""
Downstream evaluation utilities for multi-regimen reasoning experiments.

The key question is not whether a model can reproduce synthetic confidence
numbers. It is whether confidence helps identify better syllogisms. This module
therefore centers evaluation on syllogism quality plus confidence utility.
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List


@dataclass
class EvalRecord:
    """One scored holdout prediction."""

    quote: str
    predicted_confidence: float
    syllogism_quality: float


@dataclass
class ConfidenceUtilityReport:
    """Aggregate confidence-utility metrics."""

    count: int
    acceptance_threshold: float
    pearson: float
    spearman: float
    auroc: float
    brier: float
    ece: float
    risk_coverage: List[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _pearson(xs: List[float], ys: List[float]) -> float:
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0 or y_var <= 0:
        return 0.0
    return numerator / ((x_var ** 0.5) * (y_var ** 0.5))


def _rank(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[start][1]:
            end += 1
        average_rank = (start + end) / 2 + 1
        for position in range(start, end + 1):
            ranks[indexed[position][0]] = average_rank
        start = end + 1
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> float:
    return _pearson(_rank(xs), _rank(ys))


def _brier(probabilities: List[float], labels: List[int]) -> float:
    return _mean((prob - label) ** 2 for prob, label in zip(probabilities, labels))


def _ece(probabilities: List[float], labels: List[int], bins: int = 10) -> float:
    bin_totals = [0] * bins
    bin_probs = [0.0] * bins
    bin_accs = [0.0] * bins

    for probability, label in zip(probabilities, labels):
        index = min(int(probability * bins), bins - 1)
        bin_totals[index] += 1
        bin_probs[index] += probability
        bin_accs[index] += label

    total = len(probabilities)
    error = 0.0
    for count, prob_sum, acc_sum in zip(bin_totals, bin_probs, bin_accs):
        if count == 0:
            continue
        avg_prob = prob_sum / count
        avg_acc = acc_sum / count
        error += (count / total) * abs(avg_prob - avg_acc)
    return error


def _auroc(probabilities: List[float], labels: List[int]) -> float:
    positives = [p for p, label in zip(probabilities, labels) if label == 1]
    negatives = [p for p, label in zip(probabilities, labels) if label == 0]
    if not positives or not negatives:
        return 0.0

    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _risk_coverage(probabilities: List[float], labels: List[int]) -> List[Dict[str, float]]:
    ranked = sorted(zip(probabilities, labels), key=lambda item: item[0], reverse=True)
    accepted = 0
    correct = 0
    total = len(ranked)
    curve = []

    for probability, label in ranked:
        accepted += 1
        correct += label
        coverage = accepted / total
        accuracy = correct / accepted
        risk = 1.0 - accuracy
        curve.append(
            {
                "coverage": round(coverage, 4),
                "risk": round(risk, 4),
                "threshold": round(probability, 4),
            }
        )
    return curve


def evaluate_confidence_utility(
    records: List[EvalRecord],
    acceptance_threshold: float = 0.7,
) -> ConfidenceUtilityReport:
    """Evaluate whether confidence helps select better syllogisms."""
    probabilities = [record.predicted_confidence for record in records]
    qualities = [record.syllogism_quality for record in records]
    labels = [1 if quality >= acceptance_threshold else 0 for quality in qualities]

    return ConfidenceUtilityReport(
        count=len(records),
        acceptance_threshold=acceptance_threshold,
        pearson=round(_pearson(probabilities, qualities), 4),
        spearman=round(_spearman(probabilities, qualities), 4),
        auroc=round(_auroc(probabilities, labels), 4),
        brier=round(_brier(probabilities, labels), 4),
        ece=round(_ece(probabilities, labels), 4),
        risk_coverage=_risk_coverage(probabilities, labels),
    )


def load_eval_records(path: str) -> List[EvalRecord]:
    """Load evaluation records from JSONL.

    Supported field aliases:
    - `predicted_confidence` or `confidence`
    - `syllogism_quality` or `judge_score`
    """
    records = []
    with open(path) as handle:
        for line in handle:
            payload = json.loads(line)
            records.append(
                EvalRecord(
                    quote=payload.get("quote", ""),
                    predicted_confidence=float(
                        payload.get("predicted_confidence", payload.get("confidence"))
                    ),
                    syllogism_quality=float(
                        payload.get("syllogism_quality", payload.get("judge_score"))
                    ),
                )
            )
    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate downstream confidence utility on a scored holdout JSONL"
    )
    parser.add_argument("--input", required=True, help="JSONL with confidence + quality scores")
    parser.add_argument(
        "--acceptance-threshold",
        type=float,
        default=0.7,
        help="Quality threshold for pass/fail acceptance labels",
    )
    args = parser.parse_args()

    report = evaluate_confidence_utility(
        load_eval_records(args.input),
        acceptance_threshold=args.acceptance_threshold,
    )
    print(json.dumps(report.to_dict(), indent=2))
