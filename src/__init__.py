"""SPO Training Regimen - Synthetic Dataset Generation + SPO Training Infrastructure."""

__version__ = "0.1.0"

from importlib import import_module

from .synthetic_generator import (
    SyntheticReasoningGenerator,
    ReasoningExample,
    TripletItem,
)

from .spo_trainer import (
    SPOTrainer,
    SPOEvaluator,
    SPOReward,
)

from .pipeline import (
    Pipeline,
    PipelineConfig,
)

from .build_training_regimens import (
    TrainingRegimen,
    build_training_regimen_dataset,
)

_LAZY_EXPORTS = {
    "RegimenName": (".training_strategy", "RegimenName"),
    "CurriculumStage": (".training_strategy", "CurriculumStage"),
    "AblationExperiment": (".training_strategy", "AblationExperiment"),
    "DownstreamEvaluationPlan": (".training_strategy", "DownstreamEvaluationPlan"),
    "TrainingStrategy": (".training_strategy", "TrainingStrategy"),
    "EvalRecord": (".evaluate_regimens", "EvalRecord"),
    "ConfidenceUtilityReport": (
        ".evaluate_regimens",
        "ConfidenceUtilityReport",
    ),
    "evaluate_confidence_utility": (
        ".evaluate_regimens",
        "evaluate_confidence_utility",
    ),
    "load_eval_records": (".evaluate_regimens", "load_eval_records"),
    "rebuild_canonical_corpora": (
        ".rebuild_training_corpora",
        "rebuild_canonical_corpora",
    ),
    "run_ablation_matrix": (".run_ablation_matrix", "run_ablation_matrix"),
    "AblationConfig": (".run_ablation_matrix", "AblationConfig"),
    "write_holdout_markdown": (".run_ablation_matrix", "write_holdout_markdown"),
    "render_holdout_markdown": (
        ".render_holdout_markdown",
        "render_holdout_markdown",
    ),
}


def __getattr__(name):
    """Load optional exports lazily so module CLIs can run without runpy warnings."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "SyntheticReasoningGenerator",
    "ReasoningExample",
    "TripletItem",
    "SPOTrainer",
    "SPOEvaluator",
    "SPOReward",
    "Pipeline",
    "PipelineConfig",
    "TrainingRegimen",
    "build_training_regimen_dataset",
    "RegimenName",
    "CurriculumStage",
    "AblationExperiment",
    "DownstreamEvaluationPlan",
    "TrainingStrategy",
    "EvalRecord",
    "ConfidenceUtilityReport",
    "evaluate_confidence_utility",
    "load_eval_records",
    "rebuild_canonical_corpora",
    "run_ablation_matrix",
    "AblationConfig",
    "write_holdout_markdown",
    "render_holdout_markdown",
]
