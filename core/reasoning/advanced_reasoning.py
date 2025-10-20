"""
Week 10: Advanced Reasoning Module

Implements advanced cognitive capabilities:
- Analogical reasoning: Transfer knowledge by recognizing structural similarities
- Abstract pattern recognition: Identify domain-independent patterns
- Creative problem-solving: Generate novel solutions through recombination
- Cross-domain transfer: Apply solutions from one domain to another

This builds on:
- Week 5: Transfer learning
- Week 6: Causal reasoning
- Week 7: Meta-cognition
- Week 8: Strategic planning
- Week 9: Multi-agent coordination
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of abstract patterns"""
    SEQUENTIAL = "sequential"  # A -> B -> C
    CAUSAL = "causal"  # A causes B
    HIERARCHICAL = "hierarchical"  # A contains B, C
    CYCLICAL = "cyclical"  # A -> B -> C -> A
    FEEDBACK = "feedback"  # A <-> B bidirectional
    THRESHOLD = "threshold"  # X > threshold -> effect
    SATURATION = "saturation"  # Diminishing returns
    SYNERGY = "synergy"  # A + B > A + B individually


class ReasoningStrategy(Enum):
    """Problem-solving strategies"""
    ANALOGY = "analogy"  # Find similar problem
    DECOMPOSITION = "decomposition"  # Break into sub-problems
    ABSTRACTION = "abstraction"  # Remove details, find essence
    RECOMBINATION = "recombination"  # Combine existing solutions
    INVERSION = "inversion"  # Reverse the problem
    CONSTRAINT_RELAXATION = "constraint_relaxation"  # Remove constraints
    FIRST_PRINCIPLES = "first_principles"  # Reason from fundamentals


@dataclass
class Pattern:
    """Abstract pattern that transcends specific domains"""
    pattern_id: str
    pattern_type: PatternType
    structure: Dict[str, Any]  # Abstract structure
    instances: List[Dict[str, Any]] = field(default_factory=list)  # Concrete examples
    confidence: float = 0.0
    generality: float = 0.0  # How domain-independent
    discovered_at: datetime = field(default_factory=datetime.now)


@dataclass
class Analogy:
    """Mapping between source and target domains"""
    analogy_id: str
    source_domain: str
    target_domain: str
    source_structure: Dict[str, Any]
    target_structure: Dict[str, Any]
    mappings: Dict[str, str]  # source_element -> target_element
    similarity_score: float
    transferable_knowledge: List[str]


@dataclass
class Problem:
    """Problem to be solved"""
    problem_id: str
    description: str
    domain: str
    constraints: List[str]
    goal: str
    current_state: Dict[str, Any]
    target_state: Dict[str, Any]
    difficulty: int = 1  # 1-10


@dataclass
class Solution:
    """Solution to a problem"""
    solution_id: str
    problem_id: str
    strategy: ReasoningStrategy
    steps: List[str]
    expected_outcome: str
    confidence: float
    novelty: float  # How creative/unusual
    source_analogy: Optional[str] = None  # If derived from analogy


class StructureEncoder(nn.Module):
    """Neural network to encode problem structures into embeddings"""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, embedding_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode structure to embedding"""
        return self.encoder(x)


class PatternRecognizer(nn.Module):
    """Neural network to recognize abstract patterns"""

    def __init__(self, embedding_dim: int = 64, num_pattern_types: int = 8):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_pattern_types),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Classify pattern type from embedding"""
        return self.classifier(embedding)


class AdvancedReasoningEngine:
    """
    Advanced reasoning engine with analogical reasoning,
    pattern recognition, and creative problem-solving.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        device: str = "cpu",
    ):
        self.embedding_dim = embedding_dim
        self.device = device

        # Neural components
        self.structure_encoder = StructureEncoder(embedding_dim=embedding_dim).to(device)
        self.pattern_recognizer = PatternRecognizer(embedding_dim=embedding_dim).to(device)

        # Knowledge bases
        self.patterns: Dict[str, Pattern] = {}  # pattern_id -> Pattern
        self.problems: Dict[str, Problem] = {}  # problem_id -> Problem
        self.solutions: Dict[str, Solution] = {}  # solution_id -> Solution
        self.analogies: Dict[str, Analogy] = {}  # analogy_id -> Analogy

        # Domain knowledge structures
        self.domain_structures: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.stats = {
            "patterns_discovered": 0,
            "analogies_created": 0,
            "problems_solved": 0,
            "creative_solutions": 0,
            "successful_transfers": 0,
        }

        logger.info(f"Advanced reasoning engine initialized with {embedding_dim}D embeddings")

    # ========== Analogical Reasoning ==========

    def encode_structure(self, structure: Dict[str, Any]) -> np.ndarray:
        """Encode a structure (problem, domain, etc.) into embedding space"""
        # Convert structure to feature vector
        features = self._structure_to_features(structure)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

        # Encode
        self.structure_encoder.eval()
        with torch.no_grad():
            embedding = self.structure_encoder(features_tensor)

        return embedding.cpu().numpy()

    def _structure_to_features(self, structure: Dict[str, Any]) -> np.ndarray:
        """Convert structure dictionary to feature vector"""
        # Simple encoding: count different element types
        features = np.zeros(128)

        # Encode structure characteristics
        features[0] = len(structure.get("entities", []))
        features[1] = len(structure.get("relations", []))
        features[2] = len(structure.get("processes", []))
        features[3] = len(structure.get("constraints", []))
        features[4] = len(structure.get("goals", []))

        # Encode relation types
        relation_types = ["causes", "enables", "inhibits", "depends_on", "competes_with"]
        for i, rel_type in enumerate(relation_types):
            relations = structure.get("relations", [])
            count = sum(1 for r in relations if r.get("type") == rel_type)
            features[10 + i] = count

        # Encode complexity metrics
        features[20] = structure.get("complexity", 0.0)
        features[21] = structure.get("uncertainty", 0.0)
        features[22] = structure.get("dynamism", 0.0)

        # Add some noise to prevent exact duplicates
        features[30:60] = np.random.randn(30) * 0.1

        return features

    def find_analogous_domain(
        self,
        target_domain: str,
        target_structure: Dict[str, Any],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Find domains with similar structure to target domain"""
        if not self.domain_structures:
            logger.warning("No domain structures available for analogy")
            return []

        # Encode target structure
        target_embedding = self.encode_structure(target_structure)

        # Compute similarity with all known domains
        similarities = []
        for domain_name, domain_struct in self.domain_structures.items():
            if domain_name == target_domain:
                continue

            domain_embedding = self.encode_structure(domain_struct)
            similarity = self._compute_similarity(target_embedding, domain_embedding)
            similarities.append((domain_name, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        emb1_flat = emb1.flatten()
        emb2_flat = emb2.flatten()

        dot_product = np.dot(emb1_flat, emb2_flat)
        norm1 = np.linalg.norm(emb1_flat)
        norm2 = np.linalg.norm(emb2_flat)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    async def create_analogy(
        self,
        source_domain: str,
        target_domain: str,
        source_problem: Optional[Problem] = None,
    ) -> Optional[Analogy]:
        """Create analogy between source and target domains"""
        if source_domain not in self.domain_structures:
            logger.warning(f"Source domain {source_domain} not in knowledge base")
            return None

        if target_domain not in self.domain_structures:
            logger.warning(f"Target domain {target_domain} not in knowledge base")
            return None

        source_struct = self.domain_structures[source_domain]
        target_struct = self.domain_structures[target_domain]

        # Compute structural similarity
        source_emb = self.encode_structure(source_struct)
        target_emb = self.encode_structure(target_struct)
        similarity = self._compute_similarity(source_emb, target_emb)

        # Create mappings between elements
        mappings = self._create_element_mappings(source_struct, target_struct)

        # Identify transferable knowledge
        transferable = self._identify_transferable_knowledge(
            source_struct, target_struct, mappings
        )

        analogy = Analogy(
            analogy_id=f"analogy_{len(self.analogies) + 1}",
            source_domain=source_domain,
            target_domain=target_domain,
            source_structure=source_struct,
            target_structure=target_struct,
            mappings=mappings,
            similarity_score=similarity,
            transferable_knowledge=transferable,
        )

        self.analogies[analogy.analogy_id] = analogy
        self.stats["analogies_created"] += 1

        logger.info(
            f"Created analogy: {source_domain} -> {target_domain} "
            f"(similarity: {similarity:.3f}, {len(transferable)} transferable items)"
        )

        return analogy

    def _create_element_mappings(
        self,
        source_struct: Dict[str, Any],
        target_struct: Dict[str, Any],
    ) -> Dict[str, str]:
        """Create mappings between source and target elements"""
        mappings = {}

        # Map entities based on roles
        source_entities = source_struct.get("entities", [])
        target_entities = target_struct.get("entities", [])

        for s_entity in source_entities:
            s_role = s_entity.get("role", "")
            # Find target entity with similar role
            for t_entity in target_entities:
                t_role = t_entity.get("role", "")
                if self._roles_similar(s_role, t_role):
                    mappings[s_entity["name"]] = t_entity["name"]
                    break

        return mappings

    def _roles_similar(self, role1: str, role2: str) -> bool:
        """Check if two roles are similar"""
        # Simple keyword matching
        role1_lower = role1.lower()
        role2_lower = role2.lower()

        # Exact match
        if role1_lower == role2_lower:
            return True

        # Keyword overlap
        keywords1 = set(role1_lower.split())
        keywords2 = set(role2_lower.split())
        overlap = len(keywords1 & keywords2)

        return overlap > 0

    def _identify_transferable_knowledge(
        self,
        source_struct: Dict[str, Any],
        target_struct: Dict[str, Any],
        mappings: Dict[str, str],
    ) -> List[str]:
        """Identify knowledge that can be transferred via analogy"""
        transferable = []

        # Transfer causal relationships
        source_relations = source_struct.get("relations", [])
        for relation in source_relations:
            source_from = relation.get("from")
            source_to = relation.get("to")

            if source_from in mappings and source_to in mappings:
                target_from = mappings[source_from]
                target_to = mappings[source_to]
                rel_type = relation.get("type")

                knowledge = f"{target_from} {rel_type} {target_to} (by analogy)"
                transferable.append(knowledge)

        # Transfer strategies
        source_strategies = source_struct.get("successful_strategies", [])
        for strategy in source_strategies:
            transferable.append(f"Strategy: {strategy} (by analogy)")

        # Transfer constraints
        source_constraints = source_struct.get("constraints", [])
        for constraint in source_constraints[:2]:  # Transfer top 2 constraints
            transferable.append(f"Constraint: {constraint} (by analogy)")

        return transferable

    # ========== Pattern Recognition ==========

    async def discover_patterns(
        self,
        instances: List[Dict[str, Any]],
        min_confidence: float = 0.7,
    ) -> List[Pattern]:
        """Discover abstract patterns from concrete instances"""
        discovered = []

        # Encode all instances
        embeddings = [self.encode_structure(inst) for inst in instances]

        # Cluster similar instances
        clusters = self._cluster_instances(embeddings, instances)

        # Extract pattern from each cluster
        for cluster_instances in clusters:
            if len(cluster_instances) < 2:
                continue  # Need at least 2 instances to form pattern

            pattern = await self._extract_pattern(cluster_instances)

            if pattern and pattern.confidence >= min_confidence:
                self.patterns[pattern.pattern_id] = pattern
                self.stats["patterns_discovered"] += 1
                discovered.append(pattern)

                logger.info(
                    f"Discovered pattern {pattern.pattern_id}: {pattern.pattern_type.value} "
                    f"(confidence: {pattern.confidence:.3f}, instances: {len(pattern.instances)})"
                )

        return discovered

    def _cluster_instances(
        self,
        embeddings: List[np.ndarray],
        instances: List[Dict[str, Any]],
        similarity_threshold: float = 0.7,
    ) -> List[List[Dict[str, Any]]]:
        """Cluster similar instances together"""
        if not embeddings:
            return []

        clusters = []
        used = set()

        for i, emb1 in enumerate(embeddings):
            if i in used:
                continue

            cluster = [instances[i]]
            used.add(i)

            for j, emb2 in enumerate(embeddings):
                if j in used or j <= i:
                    continue

                similarity = self._compute_similarity(emb1, emb2)
                if similarity >= similarity_threshold:
                    cluster.append(instances[j])
                    used.add(j)

            if len(cluster) > 0:
                clusters.append(cluster)

        return clusters

    async def _extract_pattern(
        self,
        instances: List[Dict[str, Any]],
    ) -> Optional[Pattern]:
        """Extract abstract pattern from cluster of similar instances"""
        if not instances:
            return None

        # Analyze common structure
        common_structure = self._find_common_structure(instances)

        # Classify pattern type
        pattern_type = self._classify_pattern_type(common_structure)

        # Compute metrics
        confidence = len(instances) / (len(instances) + 1.0)  # Smoothed
        generality = self._compute_generality(instances)

        pattern = Pattern(
            pattern_id=f"pattern_{len(self.patterns) + 1}",
            pattern_type=pattern_type,
            structure=common_structure,
            instances=instances,
            confidence=confidence,
            generality=generality,
        )

        return pattern

    def _find_common_structure(
        self,
        instances: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Find common structure across instances"""
        if not instances:
            return {}

        # Count element occurrences
        entity_counts = {}
        relation_counts = {}

        for inst in instances:
            for entity in inst.get("entities", []):
                role = entity.get("role", "unknown")
                entity_counts[role] = entity_counts.get(role, 0) + 1

            for relation in inst.get("relations", []):
                rel_type = relation.get("type", "unknown")
                relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1

        # Extract common elements (appear in >50% of instances)
        threshold = len(instances) / 2

        common_entities = [
            {"role": role, "frequency": count / len(instances)}
            for role, count in entity_counts.items()
            if count > threshold
        ]

        common_relations = [
            {"type": rel_type, "frequency": count / len(instances)}
            for rel_type, count in relation_counts.items()
            if count > threshold
        ]

        return {
            "entities": common_entities,
            "relations": common_relations,
            "instance_count": len(instances),
        }

    def _classify_pattern_type(self, structure: Dict[str, Any]) -> PatternType:
        """Classify the type of pattern"""
        relations = structure.get("relations", [])

        if not relations:
            return PatternType.HIERARCHICAL

        # Check for specific pattern types
        relation_types = [r.get("type", "") for r in relations]

        if "causes" in relation_types:
            return PatternType.CAUSAL
        elif "cycles" in relation_types or "feedback" in relation_types:
            return PatternType.CYCLICAL
        elif "sequence" in relation_types:
            return PatternType.SEQUENTIAL
        elif "bidirectional" in relation_types:
            return PatternType.FEEDBACK
        else:
            return PatternType.HIERARCHICAL

    def _compute_generality(self, instances: List[Dict[str, Any]]) -> float:
        """Compute how domain-independent the pattern is"""
        # Count unique domains
        domains = set()
        for inst in instances:
            domain = inst.get("domain", "unknown")
            domains.add(domain)

        # More domains = more general
        generality = min(len(domains) / 3.0, 1.0)  # Normalize to [0, 1]

        return generality

    # ========== Creative Problem Solving ==========

    async def solve_problem(
        self,
        problem: Problem,
        strategies: Optional[List[ReasoningStrategy]] = None,
    ) -> List[Solution]:
        """Solve problem using multiple reasoning strategies"""
        if strategies is None:
            # Try all strategies
            strategies = list(ReasoningStrategy)

        solutions = []

        for strategy in strategies:
            solution = await self._apply_strategy(problem, strategy)
            if solution:
                solutions.append(solution)
                self.solutions[solution.solution_id] = solution

        # Sort by confidence * novelty (prefer confident creative solutions)
        solutions.sort(key=lambda s: s.confidence * (1 + s.novelty), reverse=True)

        if solutions:
            self.stats["problems_solved"] += 1
            best = solutions[0]
            logger.info(
                f"Solved problem {problem.problem_id} using {best.strategy.value} "
                f"(confidence: {best.confidence:.3f}, novelty: {best.novelty:.3f})"
            )

        return solutions

    async def _apply_strategy(
        self,
        problem: Problem,
        strategy: ReasoningStrategy,
    ) -> Optional[Solution]:
        """Apply specific reasoning strategy to problem"""

        if strategy == ReasoningStrategy.ANALOGY:
            return await self._solve_by_analogy(problem)

        elif strategy == ReasoningStrategy.DECOMPOSITION:
            return await self._solve_by_decomposition(problem)

        elif strategy == ReasoningStrategy.ABSTRACTION:
            return await self._solve_by_abstraction(problem)

        elif strategy == ReasoningStrategy.RECOMBINATION:
            return await self._solve_by_recombination(problem)

        elif strategy == ReasoningStrategy.INVERSION:
            return await self._solve_by_inversion(problem)

        elif strategy == ReasoningStrategy.FIRST_PRINCIPLES:
            return await self._solve_by_first_principles(problem)

        else:
            logger.warning(f"Strategy {strategy} not implemented")
            return None

    async def _solve_by_analogy(self, problem: Problem) -> Optional[Solution]:
        """Solve by finding analogous problem in different domain"""
        # Find analogous domains
        problem_structure = {
            "entities": [{"name": "problem", "role": "target"}],
            "relations": [],
            "goal": problem.goal,
            "complexity": problem.difficulty / 10.0,
        }

        similar_domains = self.find_analogous_domain(
            problem.domain, problem_structure, top_k=3
        )

        if not similar_domains:
            return None

        # Use most similar domain
        source_domain, similarity = similar_domains[0]

        # Create analogy
        analogy = await self.create_analogy(source_domain, problem.domain, problem)

        if not analogy:
            return None

        # Transfer solution
        steps = [
            f"Recognize similarity with {source_domain}",
            f"Apply analogical mapping (similarity: {similarity:.2f})",
        ]

        for knowledge in analogy.transferable_knowledge[:3]:
            steps.append(f"Transfer: {knowledge}")

        steps.append(f"Adapt solution to {problem.domain} context")

        solution = Solution(
            solution_id=f"sol_{len(self.solutions) + 1}",
            problem_id=problem.problem_id,
            strategy=ReasoningStrategy.ANALOGY,
            steps=steps,
            expected_outcome=f"Achieve {problem.goal} using {source_domain} approach",
            confidence=similarity * 0.8,  # Discount for transfer uncertainty
            novelty=0.6,  # Moderately novel (adapted from other domain)
            source_analogy=analogy.analogy_id,
        )

        return solution

    async def _solve_by_decomposition(self, problem: Problem) -> Optional[Solution]:
        """Solve by breaking into sub-problems"""
        steps = [
            "Identify independent sub-problems",
            f"Sub-problem 1: Analyze current state of {problem.domain}",
            f"Sub-problem 2: Identify gaps between current and target state",
            f"Sub-problem 3: Design interventions for each gap",
            f"Sub-problem 4: Sequence interventions optimally",
            "Integrate sub-solutions into complete solution",
        ]

        solution = Solution(
            solution_id=f"sol_{len(self.solutions) + 1}",
            problem_id=problem.problem_id,
            strategy=ReasoningStrategy.DECOMPOSITION,
            steps=steps,
            expected_outcome=f"Achieve {problem.goal} through systematic decomposition",
            confidence=0.75,
            novelty=0.3,  # Standard approach
        )

        return solution

    async def _solve_by_abstraction(self, problem: Problem) -> Optional[Solution]:
        """Solve by abstracting to essential elements"""
        steps = [
            "Remove domain-specific details",
            "Identify core problem structure",
            "Recognize pattern: This is essentially a optimization problem",
            "Apply general optimization principles",
            "Instantiate abstract solution in domain context",
        ]

        solution = Solution(
            solution_id=f"sol_{len(self.solutions) + 1}",
            problem_id=problem.problem_id,
            strategy=ReasoningStrategy.ABSTRACTION,
            steps=steps,
            expected_outcome=f"Achieve {problem.goal} through abstraction",
            confidence=0.70,
            novelty=0.5,
        )

        return solution

    async def _solve_by_recombination(self, problem: Problem) -> Optional[Solution]:
        """Solve by combining existing solutions in novel ways"""
        # Find past solutions from same domain
        past_solutions = [
            sol for sol in self.solutions.values()
            if self.problems.get(sol.problem_id, Problem("", "", problem.domain, [], "", {}, {})).domain == problem.domain
        ]

        if len(past_solutions) < 2:
            return None

        # Combine elements from multiple solutions
        steps = ["Identify relevant past solutions"]

        for i, past_sol in enumerate(past_solutions[:3], 1):
            steps.append(f"Extract technique {i} from {past_sol.solution_id}")

        steps.append("Synthesize novel combination of techniques")
        steps.append(f"Apply combined approach to achieve {problem.goal}")

        solution = Solution(
            solution_id=f"sol_{len(self.solutions) + 1}",
            problem_id=problem.problem_id,
            strategy=ReasoningStrategy.RECOMBINATION,
            steps=steps,
            expected_outcome=f"Achieve {problem.goal} through creative recombination",
            confidence=0.65,
            novelty=0.8,  # Highly novel!
        )

        self.stats["creative_solutions"] += 1

        return solution

    async def _solve_by_inversion(self, problem: Problem) -> Optional[Solution]:
        """Solve by inverting the problem"""
        steps = [
            f"Invert problem: Instead of {problem.goal}, consider opposite",
            "Identify what would prevent goal achievement",
            "Eliminate or mitigate those preventive factors",
            "By removing obstacles, enable goal achievement",
        ]

        solution = Solution(
            solution_id=f"sol_{len(self.solutions) + 1}",
            problem_id=problem.problem_id,
            strategy=ReasoningStrategy.INVERSION,
            steps=steps,
            expected_outcome=f"Achieve {problem.goal} through inversion",
            confidence=0.68,
            novelty=0.7,
        )

        return solution

    async def _solve_by_first_principles(self, problem: Problem) -> Optional[Solution]:
        """Solve by reasoning from fundamental principles"""
        steps = [
            "Identify fundamental truths about the domain",
            "Break problem down to basic elements",
            "Reason up from first principles",
            "Construct solution without relying on existing approaches",
            f"Apply first-principles solution to achieve {problem.goal}",
        ]

        solution = Solution(
            solution_id=f"sol_{len(self.solutions) + 1}",
            problem_id=problem.problem_id,
            strategy=ReasoningStrategy.FIRST_PRINCIPLES,
            steps=steps,
            expected_outcome=f"Achieve {problem.goal} through first-principles reasoning",
            confidence=0.80,
            novelty=0.6,
        )

        return solution

    # ========== Cross-Domain Transfer ==========

    async def transfer_knowledge(
        self,
        from_domain: str,
        to_domain: str,
        knowledge_items: List[str],
    ) -> Dict[str, Any]:
        """Transfer knowledge from one domain to another"""
        # Create analogy if not exists
        analogy = None
        for existing_analogy in self.analogies.values():
            if (existing_analogy.source_domain == from_domain and
                existing_analogy.target_domain == to_domain):
                analogy = existing_analogy
                break

        if not analogy:
            analogy = await self.create_analogy(from_domain, to_domain)

        if not analogy:
            logger.warning(f"Cannot create analogy: {from_domain} -> {to_domain}")
            return {"success": False, "transferred": []}

        # Transfer each knowledge item
        transferred = []
        for item in knowledge_items:
            # Map through analogy
            mapped_item = self._map_through_analogy(item, analogy)
            if mapped_item:
                transferred.append(mapped_item)

        if transferred:
            self.stats["successful_transfers"] += 1

        logger.info(
            f"Transferred {len(transferred)}/{len(knowledge_items)} items: "
            f"{from_domain} -> {to_domain}"
        )

        return {
            "success": len(transferred) > 0,
            "transferred": transferred,
            "analogy_id": analogy.analogy_id,
            "similarity": analogy.similarity_score,
        }

    def _map_through_analogy(self, knowledge: str, analogy: Analogy) -> Optional[str]:
        """Map knowledge item through analogy"""
        # Simple substitution through mappings
        mapped = knowledge
        for source_elem, target_elem in analogy.mappings.items():
            mapped = mapped.replace(source_elem, target_elem)

        # Only return if actually changed (successful mapping)
        if mapped != knowledge:
            return f"{mapped} (transferred from {analogy.source_domain})"

        return None

    # ========== Domain Management ==========

    def register_domain_structure(self, domain: str, structure: Dict[str, Any]):
        """Register structure of a domain for analogical reasoning"""
        self.domain_structures[domain] = structure
        logger.info(f"Registered domain structure: {domain}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get reasoning engine statistics"""
        return {
            **self.stats,
            "patterns": len(self.patterns),
            "analogies": len(self.analogies),
            "problems": len(self.problems),
            "solutions": len(self.solutions),
            "domains": len(self.domain_structures),
        }


# ========== Testing Functions ==========

async def test_advanced_reasoning():
    """Test advanced reasoning capabilities"""
    print("\n" + "="*60)
    print("Testing Advanced Reasoning Engine")
    print("="*60)

    engine = AdvancedReasoningEngine(embedding_dim=64)

    # Register domain structures for analogical reasoning
    print("\n1. Registering domain structures...")

    # Sewago (error management)
    engine.register_domain_structure("sewago", {
        "entities": [
            {"name": "errors", "role": "problem_source"},
            {"name": "monitoring", "role": "detector"},
            {"name": "fixes", "role": "solution"},
        ],
        "relations": [
            {"from": "errors", "to": "monitoring", "type": "detected_by"},
            {"from": "monitoring", "to": "fixes", "type": "triggers"},
            {"from": "fixes", "to": "errors", "type": "reduces"},
        ],
        "successful_strategies": ["proactive monitoring", "automated fixing"],
        "constraints": ["uptime requirements", "performance overhead"],
    })

    # Halobuzz (engagement management)
    engine.register_domain_structure("halobuzz", {
        "entities": [
            {"name": "low_engagement", "role": "problem_source"},
            {"name": "analytics", "role": "detector"},
            {"name": "campaigns", "role": "solution"},
        ],
        "relations": [
            {"from": "low_engagement", "to": "analytics", "type": "detected_by"},
            {"from": "analytics", "to": "campaigns", "type": "triggers"},
            {"from": "campaigns", "to": "low_engagement", "type": "reduces"},
        ],
        "successful_strategies": ["proactive monitoring", "automated optimization"],
        "constraints": ["budget limits", "audience saturation"],
    })

    # SolsniperPro (risk management)
    engine.register_domain_structure("solsniperpro", {
        "entities": [
            {"name": "risks", "role": "problem_source"},
            {"name": "analysis", "role": "detector"},
            {"name": "hedging", "role": "solution"},
        ],
        "relations": [
            {"from": "risks", "to": "analysis", "type": "detected_by"},
            {"from": "analysis", "to": "hedging", "type": "triggers"},
            {"from": "hedging", "to": "risks", "type": "reduces"},
        ],
        "successful_strategies": ["proactive monitoring", "automated hedging"],
        "constraints": ["capital requirements", "market volatility"],
    })

    print(f"Registered {len(engine.domain_structures)} domain structures")

    # Test analogical reasoning
    print("\n2. Testing analogical reasoning...")
    analogy = await engine.create_analogy("sewago", "halobuzz")

    if analogy:
        print(f"Created analogy: {analogy.source_domain} -> {analogy.target_domain}")
        print(f"Similarity score: {analogy.similarity_score:.3f}")
        print(f"Mappings: {len(analogy.mappings)}")
        for src, tgt in list(analogy.mappings.items())[:3]:
            print(f"  {src} -> {tgt}")
        print(f"Transferable knowledge: {len(analogy.transferable_knowledge)} items")
        for knowledge in analogy.transferable_knowledge[:3]:
            print(f"  - {knowledge}")

    # Test pattern discovery
    print("\n3. Testing pattern recognition...")

    # Create similar problem instances
    instances = [
        {
            "domain": "sewago",
            "entities": [{"name": "error", "role": "problem_source"}],
            "relations": [{"from": "error", "to": "fix", "type": "causes"}],
        },
        {
            "domain": "halobuzz",
            "entities": [{"name": "low_engagement", "role": "problem_source"}],
            "relations": [{"from": "low_engagement", "to": "campaign", "type": "causes"}],
        },
        {
            "domain": "solsniperpro",
            "entities": [{"name": "risk", "role": "problem_source"}],
            "relations": [{"from": "risk", "to": "hedge", "type": "causes"}],
        },
    ]

    patterns = await engine.discover_patterns(instances, min_confidence=0.5)
    print(f"Discovered {len(patterns)} patterns")
    for pattern in patterns:
        print(f"  Pattern {pattern.pattern_id}: {pattern.pattern_type.value}")
        print(f"    Confidence: {pattern.confidence:.3f}, Generality: {pattern.generality:.3f}")
        print(f"    Instances: {len(pattern.instances)}")

    # Test creative problem solving
    print("\n4. Testing creative problem solving...")

    problem = Problem(
        problem_id="prob_1",
        description="Reduce error rate in sewago",
        domain="sewago",
        constraints=["Must maintain 99.9% uptime", "Limited compute budget"],
        goal="Reduce error rate to <1%",
        current_state={"error_rate": 5.0},
        target_state={"error_rate": 0.8},
        difficulty=7,
    )

    engine.problems[problem.problem_id] = problem

    solutions = await engine.solve_problem(
        problem,
        strategies=[
            ReasoningStrategy.ANALOGY,
            ReasoningStrategy.DECOMPOSITION,
            ReasoningStrategy.ABSTRACTION,
        ],
    )

    print(f"Generated {len(solutions)} solutions")
    for i, sol in enumerate(solutions, 1):
        print(f"\nSolution {i} ({sol.strategy.value}):")
        print(f"  Confidence: {sol.confidence:.3f}, Novelty: {sol.novelty:.3f}")
        print(f"  Steps:")
        for step in sol.steps[:3]:
            print(f"    - {step}")
        if len(sol.steps) > 3:
            print(f"    ... and {len(sol.steps) - 3} more steps")

    # Test cross-domain transfer
    print("\n5. Testing cross-domain knowledge transfer...")

    knowledge_items = [
        "Proactive monitoring reduces problems",
        "Automated responses improve efficiency",
        "Early detection is critical",
    ]

    transfer_result = await engine.transfer_knowledge(
        "sewago", "solsniperpro", knowledge_items
    )

    print(f"Transfer success: {transfer_result['success']}")
    print(f"Transferred {len(transfer_result['transferred'])} items:")
    for item in transfer_result['transferred']:
        print(f"  - {item}")

    # Final statistics
    print("\n" + "="*60)
    print("Final Statistics")
    print("="*60)
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    return engine


if __name__ == "__main__":
    asyncio.run(test_advanced_reasoning())
