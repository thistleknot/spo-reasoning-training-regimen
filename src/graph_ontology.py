"""Graph ontology layer for entity normalization and triplet-based fact retrieval.

Handles:
- Synset collapse (entity normalization to canonical forms)
- Predicate equivalence (relation normalization)
- Graph node creation and traversal
- Subject-predicate-object (SPO) path extraction
"""

from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib


@dataclass
class Triplet:
    """A single subject-predicate-object triplet with metadata."""
    
    subject: str
    predicate: str
    object_: str
    evidence_tag: str = "inferred"  # "observed" or "inferred"
    confidence: float = 0.5
    source: str = "training"  # "training", "inference", "judge"
    
    def __hash__(self):
        """Hash by normalized form."""
        return hash((self.subject.lower(), self.predicate.lower(), self.object_.lower()))
    
    def __eq__(self, other):
        """Equality by normalized form."""
        if not isinstance(other, Triplet):
            return False
        return (
            self.subject.lower() == other.subject.lower()
            and self.predicate.lower() == other.predicate.lower()
            and self.object_.lower() == other.object_.lower()
        )
    
    def __repr__(self) -> str:
        return f"{self.subject} | {self.predicate} ({self.evidence_tag}, confidence={self.confidence}) | {self.object_}"
    
    def to_spo_tuple(self) -> Tuple[str, str, str]:
        """Return as (subject, predicate, object) tuple."""
        return (self.subject, self.predicate, self.object_)


@dataclass
class GraphOntology:
    """Knowledge graph with entity synset collapse and fact retrieval."""
    
    # Entity synset mapping (entity → canonical)
    synsets: Dict[str, str] = field(default_factory=dict)
    
    # Predicate equivalence (canonical → set of variants)
    predicate_variants: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    # Fact storage: (canonical_s, canonical_p, canonical_o) → Triplet
    facts: Dict[Tuple[str, str, str], Triplet] = field(default_factory=dict)
    
    # Index: subject → list of triplets
    subject_index: Dict[str, List[Triplet]] = field(default_factory=lambda: defaultdict(list))
    
    # Index: object → list of triplets
    object_index: Dict[str, List[Triplet]] = field(default_factory=lambda: defaultdict(list))
    
    def register_synset(self, entity: str, canonical: str):
        """Register entity synset mapping."""
        self.synsets[entity.lower()] = canonical.lower()
    
    def register_predicate_variant(self, canonical: str, variant: str):
        """Register predicate variant."""
        self.predicate_variants[canonical.lower()].add(variant.lower())
        self.predicate_variants[canonical.lower()].add(canonical.lower())
    
    def normalize_entity(self, entity: str) -> str:
        """Apply synset collapse to entity."""
        return self.synsets.get(entity.lower(), entity)
    
    def normalize_predicate(self, predicate: str) -> str:
        """Get canonical predicate form."""
        pred_lower = predicate.lower()
        for canonical, variants in self.predicate_variants.items():
            if pred_lower in variants:
                return canonical
        return predicate
    
    def add_triplet(self, triplet: Triplet):
        """Add triplet to graph."""
        # Normalize entities and predicates
        norm_s = self.normalize_entity(triplet.subject)
        norm_p = self.normalize_predicate(triplet.predicate)
        norm_o = self.normalize_entity(triplet.object_)
        
        # Store with normalized forms
        normalized_triplet = Triplet(
            subject=norm_s,
            predicate=norm_p,
            object_=norm_o,
            evidence_tag=triplet.evidence_tag,
            confidence=triplet.confidence,
            source=triplet.source,
        )
        
        key = (norm_s, norm_p, norm_o)
        if key not in self.facts:
            self.facts[key] = normalized_triplet
            self.subject_index[norm_s].append(normalized_triplet)
            self.object_index[norm_o].append(normalized_triplet)
    
    def add_triplets(self, triplets: List[Triplet]):
        """Add multiple triplets."""
        for triplet in triplets:
            self.add_triplet(triplet)
    
    def get_outgoing(self, subject: str, confidence_threshold: float = 0.0) -> List[Triplet]:
        """Get all facts where subject is the source.
        
        Returns facts ordered by confidence descending.
        """
        norm_s = self.normalize_entity(subject)
        facts = self.subject_index.get(norm_s, [])
        filtered = [f for f in facts if f.confidence >= confidence_threshold]
        return sorted(filtered, key=lambda f: f.confidence, reverse=True)
    
    def get_incoming(self, obj: str, confidence_threshold: float = 0.0) -> List[Triplet]:
        """Get all facts where obj is the target.
        
        Returns facts ordered by confidence descending.
        """
        norm_o = self.normalize_entity(obj)
        facts = self.object_index.get(norm_o, [])
        filtered = [f for f in facts if f.confidence >= confidence_threshold]
        return sorted(filtered, key=lambda f: f.confidence, reverse=True)
    
    def traverse_path(
        self,
        start_entity: str,
        max_depth: int = 2,
        confidence_threshold: float = 0.5,
        visited: Optional[Set[str]] = None,
    ) -> Dict[str, List[Triplet]]:
        """Traverse graph from start entity, collecting connected facts.
        
        Returns dict of {entity: [triplets connected to that entity]}
        """
        if visited is None:
            visited = set()
        
        visited.add(self.normalize_entity(start_entity))
        result = {start_entity: self.get_outgoing(start_entity, confidence_threshold)}
        
        if max_depth <= 0:
            return result
        
        # Traverse to neighboring entities
        for triplet in result.get(start_entity, []):
            neighbor = triplet.object_
            if neighbor not in visited:
                neighbor_paths = self.traverse_path(
                    neighbor,
                    max_depth - 1,
                    confidence_threshold,
                    visited,
                )
                result.update(neighbor_paths)
        
        return result
    
    def find_connecting_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 3,
        confidence_threshold: float = 0.5,
    ) -> List[List[Triplet]]:
        """Find all paths from source to target.
        
        Returns list of paths, where each path is a list of triplets.
        """
        paths = []
        
        def dfs(current: str, path: List[Triplet], visited: Set[str]):
            norm_current = self.normalize_entity(current)
            if norm_current in visited or len(path) >= max_depth:
                return
            
            visited.add(norm_current)
            
            if norm_current == self.normalize_entity(target):
                paths.append(path[:])
                visited.remove(norm_current)
                return
            
            for triplet in self.get_outgoing(current, confidence_threshold):
                path.append(triplet)
                dfs(triplet.object_, path, visited)
                path.pop()
            
            visited.remove(norm_current)
        
        dfs(source, [], set())
        return paths
    
    def extract_entailed_facts(self, confidence_threshold: float = 0.5) -> List[Triplet]:
        """Extract all high-confidence facts (entailed premises).
        
        Suitable for graph traversal and retrieval.
        """
        return [f for f in self.facts.values() if f.confidence >= confidence_threshold]
    
    def extract_non_entailed_facts(self, confidence_threshold: float = 0.5) -> List[Triplet]:
        """Extract low-confidence facts (non-entailed premises).
        
        Useful for negative inference and exclusion during retrieval.
        """
        return [f for f in self.facts.values() if f.confidence < confidence_threshold]
    
    def stats(self) -> Dict[str, int]:
        """Return graph statistics."""
        entailed = sum(1 for f in self.facts.values() if f.confidence >= 0.7)
        non_entailed = sum(1 for f in self.facts.values() if f.confidence < 0.7)
        
        return {
            "total_facts": len(self.facts),
            "entailed_facts": entailed,
            "non_entailed_facts": non_entailed,
            "unique_entities": len(set(f.subject for f in self.facts.values()) | set(f.object_ for f in self.facts.values())),
            "unique_predicates": len(set(f.predicate for f in self.facts.values())),
        }


def build_ontology_from_triplets(
    triplets: List[Triplet],
    synset_map: Optional[Dict[str, str]] = None,
    predicate_equivalence: Optional[Dict[str, Set[str]]] = None,
) -> GraphOntology:
    """Build graph ontology from triplet list.
    
    Args:
        triplets: List of Triplet objects
        synset_map: Entity → canonical entity mapping
        predicate_equivalence: Canonical predicate → variants mapping
    
    Returns:
        Configured GraphOntology
    """
    ontology = GraphOntology()
    
    # Register synsets
    if synset_map:
        for entity, canonical in synset_map.items():
            ontology.register_synset(entity, canonical)
    
    # Register predicate variants
    if predicate_equivalence:
        for canonical, variants in predicate_equivalence.items():
            for variant in variants:
                ontology.register_predicate_variant(canonical, variant)
    
    # Add all triplets
    ontology.add_triplets(triplets)
    
    return ontology
