"""Pipeline orchestrator for end-to-end reasoning model training.

Coordinates:
1. Synthetic dataset generation from quotes
2. Data cleaning and validation
3. QLoRA training with configurable formats
4. SPO reward-based fine-tuning
5. Inference and evaluation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict

from .synthetic_generator import SyntheticReasoningGenerator, ReasoningExample
from .spo_trainer import SPOTrainer, SPOEvaluator


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the entire training pipeline."""

    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    models_dir: Path = Path("models")

    # Dataset generation
    generate_dataset: bool = True
    quotes_path: Optional[Path] = None
    synthetic_output: Path = Path("data/synthetic_dataset.jsonl")

    # Training format
    use_contrastive_input: bool = True
    include_confidence: bool = False
    pedagogical_order: bool = True  # Non-Entailed → Entailed → Throughline

    # QLoRA training
    model_name: str = "Qwen/Qwen3.5-0.8B"
    lora_rank: int = 32
    lora_alpha: int = 64
    batch_size: int = 2
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3

    # SPO training
    use_spo: bool = False
    spo_beta: float = 0.1
    spo_epochs: int = 1

    # Evaluation
    eval_on_holdout: bool = True
    holdout_size: int = 10
    validation_report_path: Path = Path("output/VALIDATION_REPORT.md")

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        # Convert Path objects to strings
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
        return d


class Pipeline:
    """Orchestrates the complete training pipeline."""

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.setup_directories()

        # Initialize components
        self.generator = SyntheticReasoningGenerator()
        self.trainer = None  # Initialized when model is loaded

        logger.info("Pipeline initialized with config:")
        logger.info(json.dumps(config.to_dict(), indent=2))

    def setup_directories(self):
        """Create required directories."""
        for dir_path in [self.config.data_dir, self.config.output_dir, self.config.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ready: {dir_path}")

    def generate_synthetic_dataset(self, quotes: List[str]) -> List[ReasoningExample]:
        """Generate synthetic reasoning dataset from quotes.

        Args:
            quotes: List of quotes to process

        Returns:
            List of ReasoningExample objects
        """
        logger.info(f"Generating synthetic dataset from {len(quotes)} quotes...")

        examples = self.generator.generate_from_quotes(quotes)

        # Export to JSONL for training
        output_path = self.config.synthetic_output
        self.generator.export_to_jsonl(
            str(output_path),
            include_confidence=self.config.include_confidence,
        )
        logger.info(f"Exported {len(examples)} examples to {output_path}")

        # Show statistics
        stats = self.generator.stats()
        logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")

        return examples

    def load_quotes_from_file(self, path: Path) -> List[str]:
        """Load quotes from file (JSON array or JSONL).

        Args:
            path: Path to quotes file

        Returns:
            List of quote strings
        """
        quotes = []

        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
                quotes = data if isinstance(data, list) else data.get("quotes", [])

        elif path.suffix == ".jsonl":
            with open(path) as f:
                for line in f:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        quotes.append(record.get("quote", record.get("text", "")))
                    else:
                        quotes.append(record)

        elif path.suffix == ".txt":
            with open(path) as f:
                quotes = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(quotes)} quotes from {path}")
        return quotes

    def prepare_training_data(
        self,
        examples: List[ReasoningExample],
        split: float = 0.9,
    ) -> tuple:
        """Prepare training and validation splits.

        Args:
            examples: List of ReasoningExample objects
            split: Train/val split ratio

        Returns:
            Tuple of (train_examples, val_examples)
        """
        split_idx = int(len(examples) * split)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        logger.info(f"Split dataset: {len(train_examples)} train, {len(val_examples)} val")
        return train_examples, val_examples

    def export_for_training(
        self,
        examples: List[ReasoningExample],
        output_path: Path,
    ):
        """Export examples in training format (pedagogical order).

        Args:
            examples: List of ReasoningExample objects
            output_path: Where to save training data
        """
        # Reuse the generator's export method
        self.generator.examples = examples
        self.generator.export_to_jsonl(
            str(output_path),
            include_confidence=self.config.include_confidence,
        )
        logger.info(f"Exported training data to {output_path}")

    def generate_validation_report(
        self,
        holdout_examples: List[ReasoningExample],
        inferences: Optional[Dict] = None,
    ):
        """Generate markdown validation report.

        Args:
            holdout_examples: List of held-out examples
            inferences: Optional dict of model inferences by quote
        """
        report = "# Reasoning Model Validation Report\n\n"
        report += f"**Date Generated:** [auto]\n"
        report += f"**Examples Evaluated:** {len(holdout_examples)}\n\n"

        for i, example in enumerate(holdout_examples, 1):
            report += f"## Example {i}\n\n"
            report += f"Input:\n`{example.quote}`\n\n"

            if inferences and example.quote in inferences:
                report += f"Completion:\n```\n{inferences[example.quote]}\n```\n\n"
            else:
                # Generate expected output
                non_entailed_str = "\n".join(
                    self.generator._format_triplet(
                        p,
                        include_confidence=self.config.include_confidence,
                    )
                    for p in example.non_entailed_premises
                )
                entailed_str = "\n".join(
                    self.generator._format_triplet(
                        p,
                        include_confidence=self.config.include_confidence,
                    )
                    for p in example.entailed_premises
                )

                output_text = f"""Non-Entailed Premises:
{non_entailed_str or "  N/A"}

Entailed Premises:
{entailed_str or "  N/A"}

Throughline:
  {example.syllogism}"""

                report += f"Completion:\n```\n{output_text}\n```\n\n"

        # Save report
        report_path = self.config.validation_report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Generated validation report: {report_path}")

    def run(self):
        """Run complete pipeline."""
        logger.info("=" * 60)
        logger.info("REASONING MODEL TRAINING PIPELINE")
        logger.info("=" * 60)

        # Step 1: Generate or load dataset
        if self.config.generate_dataset and self.config.quotes_path:
            quotes = self.load_quotes_from_file(self.config.quotes_path)
            examples = self.generate_synthetic_dataset(quotes)
        else:
            logger.info("Skipping synthetic generation (use generate_dataset=True)")
            examples = []

        # Step 2: Prepare splits
        if examples:
            train_examples, val_examples = self.prepare_training_data(examples)

            # Step 3: Export for training
            train_path = self.config.data_dir / "train.jsonl"
            self.export_for_training(train_examples, train_path)

            val_path = self.config.data_dir / "validation.jsonl"
            self.export_for_training(val_examples, val_path)

            # Step 4: Generate validation report
            self.generate_validation_report(val_examples)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)


def main():
    """Entry point for pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create config
    config = PipelineConfig(
        generate_dataset=True,
        quotes_path=Path("data/sample_quotes.txt"),
        model_name="Qwen/Qwen3.5-0.8B",
        num_epochs=3,
        use_spo=False,  # Set to True for SPO fine-tuning
    )

    # Run pipeline
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
