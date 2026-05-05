"""Configurable training format and ontology settings.

Supports parametrization of:
- What fields go into training (syllogism, confidence, etc)
- Entity normalization (synset collapse)
- Graph traversal rules
- Judge-based synthesis vs training-based
"""

from dataclasses import dataclass, asdict
from typing import Optional, Set, Dict, Any
from enum import Enum
import json


class SyllogismSource(Enum):
    """Where syllogism comes from."""
    NONE = "none"  # Don't include in training
    GROUND_TRUTH = "ground_truth"  # Use provided syllogism
    LLM_JUDGE = "llm_judge"  # Generate post-hoc with LLM


class PremiseOrdering(Enum):
    """Order of premises in training data."""
    PEDAGOGICAL = "pedagogical"  # Non-Entailed → Entailed → Throughline
    LOGICAL = "logical"  # Throughline → Entailed → Non-Entailed
    ENTAILED_ONLY = "entailed_only"  # Only entailed premises


@dataclass
class TrainingFormat:
    """Configure what's included in training data."""
    
    # Syllogism handling (NOT in training, just inference)
    syllogism_source: SyllogismSource = SyllogismSource.NONE
    include_syllogism_in_training: bool = False
    
    # Premise structure (what we train ON)
    premise_ordering: PremiseOrdering = PremiseOrdering.PEDAGOGICAL
    include_non_entailed: bool = True
    include_entailed: bool = True
    
    # Evidence tags are FEATURES the model learns (observed vs inferred)
    # Numeric confidence stays post-hoc by default.
    include_evidence_tags: bool = True
    include_confidence_scores: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "syllogism_source": self.syllogism_source.value,
            "include_syllogism_in_training": self.include_syllogism_in_training,
            "premise_ordering": self.premise_ordering.value,
            "include_non_entailed": self.include_non_entailed,
            "include_entailed": self.include_entailed,
            "include_evidence_tags": self.include_evidence_tags,
            "include_confidence_scores": self.include_confidence_scores,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingFormat":
        """Deserialize from dict."""
        return cls(
            syllogism_source=SyllogismSource(d.get("syllogism_source", "none")),
            include_syllogism_in_training=d.get("include_syllogism_in_training", False),
            premise_ordering=PremiseOrdering(d.get("premise_ordering", "pedagogical")),
            include_non_entailed=d.get("include_non_entailed", True),
            include_entailed=d.get("include_entailed", True),
            include_evidence_tags=d.get("include_evidence_tags", True),
            include_confidence_scores=d.get("include_confidence_scores", False),
        )


@dataclass
class EntityNormalization:
    """Configure entity synset collapse and normalization."""
    
    # Enable synset collapsing
    enable_synset_collapse: bool = True
    
    # Synset mapping rules (entity → canonical form)
    synset_map: Dict[str, str] = None
    
    # Predicate equivalence classes (synonyms)
    predicate_equivalence: Dict[str, Set[str]] = None
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.synset_map is None:
            self.synset_map = {}
        if self.predicate_equivalence is None:
            self.predicate_equivalence = {
                "is": {"equals", "represents", "means"},
                "relates": {"connects", "links", "associates"},
                "causes": {"leads to", "produces", "creates"},
            }
    
    def normalize_entity(self, entity: str) -> str:
        """Apply synset collapse to entity."""
        if not self.enable_synset_collapse:
            return entity
        return self.synset_map.get(entity.lower(), entity)
    
    def get_predicate_canonical(self, predicate: str) -> str:
        """Get canonical form of predicate."""
        pred_lower = predicate.lower()
        for canonical, equivalents in self.predicate_equivalence.items():
            if pred_lower == canonical.lower() or pred_lower in {e.lower() for e in equivalents}:
                return canonical
        return predicate
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "enable_synset_collapse": self.enable_synset_collapse,
            "synset_map": self.synset_map,
            "predicate_equivalence": {k: list(v) for k, v in self.predicate_equivalence.items()},
        }


@dataclass
class GraphTraversal:
    """Configure graph traversal for fact retrieval."""
    
    # Confidence threshold for including fact in traversal
    confidence_threshold: float = 0.5
    
    # Maximum path depth (subject → pred → object → ...)
    max_path_depth: int = 3
    
    # Whether to follow both entailed and non-entailed
    include_non_entailed_in_traversal: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)


@dataclass
class JudgeConfig:
    """Configure LLM judge for post-hoc synthesis."""
    
    # Whether to use judge for syllogism synthesis
    enable_judge: bool = True
    
    # Judge model (can be same as base or different)
    judge_model: str = "gpt-4"
    
    # Judge temperature (0 = deterministic, 1 = creative)
    judge_temperature: float = 0.3
    
    # Maximum tokens for judge output
    judge_max_tokens: int = 200
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)


@dataclass
class PipelineConfig:
    """Top-level configuration for reasoning pipeline."""
    
    training_format: TrainingFormat
    entity_normalization: EntityNormalization
    graph_traversal: GraphTraversal
    judge_config: JudgeConfig
    training_strategy_path: Optional[str] = None
    
    def to_json(self, path: str):
        """Save to JSON file."""
        config_dict = {
            "training_format": self.training_format.to_dict(),
            "entity_normalization": self.entity_normalization.to_dict(),
            "graph_traversal": self.graph_traversal.to_dict(),
            "judge_config": self.judge_config.to_dict(),
            "training_strategy_path": self.training_strategy_path,
        }
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        return cls(
            training_format=TrainingFormat.from_dict(data.get("training_format", {})),
            entity_normalization=EntityNormalization(**data.get("entity_normalization", {})),
            graph_traversal=GraphTraversal(**data.get("graph_traversal", {})),
            judge_config=JudgeConfig(**data.get("judge_config", {})),
            training_strategy_path=data.get("training_strategy_path"),
        )
    
    @classmethod
    def default(cls) -> "PipelineConfig":
        """Create default configuration."""
        return cls(
            training_format=TrainingFormat(
                syllogism_source=SyllogismSource.NONE,
                premise_ordering=PremiseOrdering.PEDAGOGICAL,
                include_confidence_scores=False,
            ),
            entity_normalization=EntityNormalization(enable_synset_collapse=True),
            graph_traversal=GraphTraversal(confidence_threshold=0.5),
            judge_config=JudgeConfig(enable_judge=True),
            training_strategy_path=None,
        )


# Example configs for common use cases
TRAINING_ONLY_CONFIG = PipelineConfig(
    training_format=TrainingFormat(
        syllogism_source=SyllogismSource.NONE,
        include_syllogism_in_training=False,
        premise_ordering=PremiseOrdering.PEDAGOGICAL,
    ),
    entity_normalization=EntityNormalization(),
    graph_traversal=GraphTraversal(),
    judge_config=JudgeConfig(enable_judge=False),
    training_strategy_path=None,
)

INFERENCE_WITH_JUDGE_CONFIG = PipelineConfig(
    training_format=TrainingFormat(
        syllogism_source=SyllogismSource.LLM_JUDGE,
        include_syllogism_in_training=False,
    ),
    entity_normalization=EntityNormalization(),
    graph_traversal=GraphTraversal(),
    judge_config=JudgeConfig(enable_judge=True, judge_model="gpt-4"),
    training_strategy_path=None,
)

GRAPH_RETRIEVAL_CONFIG = PipelineConfig(
    training_format=TrainingFormat(
        syllogism_source=SyllogismSource.NONE,
        include_non_entailed=False,  # Only facts to retrieve
    ),
    entity_normalization=EntityNormalization(enable_synset_collapse=True),
    graph_traversal=GraphTraversal(confidence_threshold=0.7),
    judge_config=JudgeConfig(enable_judge=False),
    training_strategy_path=None,
)
