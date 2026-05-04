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

__all__ = [
    "SyntheticReasoningGenerator",
    "ReasoningExample",
    "TripletItem",
    "SPOTrainer",
    "SPOEvaluator",
    "SPOReward",
    "Pipeline",
    "PipelineConfig",
]
