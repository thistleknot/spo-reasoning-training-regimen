"""SPO Training Regimen - Synthetic Dataset Generation + SPO Training Infrastructure."""

__version__ = "0.1.0"

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
]
