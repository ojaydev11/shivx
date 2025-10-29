"""
Neurosymbolic AI
Combines neural pattern recognition with symbolic logical reasoning
Key insight: AGI needs both sub-symbolic (neural) and symbolic (logic) reasoning
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from collections import defaultdict
import re

from .base import BaseAGIApproach
from ..schemas import AGIApproachType, AGIFitnessMetrics


class NeurosymbolicAI(BaseAGIApproach):
    """
    Hybrid neural-symbolic system
    Neural: Pattern recognition, embeddings
    Symbolic: Logic rules, reasoning chains
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(AGIApproachType.NEUROSYMBOLIC, config, **kwargs)

        # Symbolic knowledge base: facts and rules
        self.facts: Set[str] = set()
        self.rules: List[Tuple[Set[str], str]] = []  # (premises, conclusion)

        # Neural patterns: learned embeddings
        self.neural_patterns: Dict[str, np.ndarray] = {}
        self.embedding_dim = self.config.get("embedding_dim", 16)

        # Statistics
        self.logical_inferences = 0
        self.pattern_matches = 0

    def train(self, tasks: List[Dict[str, Any]]) -> None:
        """Learn both symbolic rules and neural patterns"""
        for task in tasks:
            task_type = task.get("type", "unknown")
            examples = task.get("examples", [])

            # Extract symbolic rules from examples
            for example in examples:
                # Learn facts
                if isinstance(example.get("input"), str):
                    self.facts.add(example["input"])

                # Learn rules: if A and B then C
                if "rule" in example:
                    premises_list = example["rule"].get("if", [])
                    # Convert to tuple (hashable) instead of set
                    premises = tuple(sorted(premises_list)) if premises_list else ()
                    conclusion = example["rule"].get("then", "")
                    self.rules.append((premises, conclusion))

                # Learn neural pattern
                pattern_key = str(example.get("input", ""))
                if pattern_key and pattern_key not in self.neural_patterns:
                    # Simple random embedding (in real system, use learned embeddings)
                    self.neural_patterns[pattern_key] = np.random.randn(self.embedding_dim)

            # Record learning pattern
            self.record_pattern(
                pattern_type="learning",
                context=f"task={task_type}",
                data={
                    "num_facts": len(self.facts),
                    "num_rules": len(self.rules),
                    "num_patterns": len(self.neural_patterns),
                },
                success_score=0.7,
                generalization_score=0.0,
            )

    def evaluate(self, test_tasks: List[Dict[str, Any]]) -> AGIFitnessMetrics:
        """Evaluate hybrid reasoning"""
        correct = 0
        total = 0

        symbolic_successes = []
        neural_successes = []

        for task in test_tasks:
            examples = task.get("examples", [])

            for example in examples:
                query = example.get("input", "")
                expected = example.get("output", "")

                # Try symbolic reasoning first
                symbolic_result = self._symbolic_reason(query)

                # Try neural pattern matching
                neural_result = self._neural_match(query)

                # Combine (symbolic has priority)
                result = symbolic_result if symbolic_result else neural_result

                if result == expected:
                    correct += 1
                    symbolic_successes.append(1 if symbolic_result else 0)
                    neural_successes.append(1 if neural_result else 0)
                else:
                    symbolic_successes.append(0)
                    neural_successes.append(0)

                total += 1

        accuracy = correct / max(total, 1)

        # How often does each component work?
        symbolic_rate = np.mean(symbolic_successes) if symbolic_successes else 0.0
        neural_rate = np.mean(neural_successes) if neural_successes else 0.0

        return AGIFitnessMetrics(
            overall_score=0.0,
            general_reasoning=accuracy,
            transfer_learning=0.0,
            causal_understanding=symbolic_rate,  # Logic captures causality
            abstraction=0.7,  # Symbols are abstract
            creativity=self._measure_creativity(),
            metacognition=0.8,  # Can explain symbolic reasoning
            sample_efficiency=len(self.rules) / max(len(self.neural_patterns), 1),
            robustness=(symbolic_rate + neural_rate) / 2,  # Dual system = robust
            interpretability=0.9,  # Symbolic rules fully interpretable
        )

    def transfer(self, new_domain: str, tasks: List[Dict[str, Any]]) -> float:
        """Test transfer learning"""
        # Symbolic rules transfer well!
        fitness_before = self.evaluate(tasks[:3])

        # Learn from few examples
        self.train(tasks[3:8])

        fitness_after = self.evaluate(tasks[8:])

        return max(0.0, fitness_after.general_reasoning - fitness_before.general_reasoning)

    def _symbolic_reason(self, query: str) -> Optional[str]:
        """Apply logical reasoning"""
        # Check if query is a known fact
        if query in self.facts:
            return query

        # Apply rules: forward chaining
        for premises, conclusion in self.rules:
            # Check if all premises are satisfied
            if all(p in self.facts or p in query for p in premises):
                self.logical_inferences += 1
                return conclusion

        return None

    def _neural_match(self, query: str) -> Optional[str]:
        """Pattern matching with embeddings"""
        if not self.neural_patterns:
            return None

        # Simple: find most similar pattern
        query_emb = np.random.randn(self.embedding_dim)  # In real system, encode query

        best_match = None
        best_similarity = -1.0

        for pattern, emb in self.neural_patterns.items():
            similarity = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern

        if best_similarity > 0.7:  # Threshold
            self.pattern_matches += 1
            return best_match

        return None

    def _measure_creativity(self) -> float:
        """Can it generate novel conclusions from rules?"""
        # Chain multiple rules to derive new facts
        novel_conclusions = set()

        for premises1, conclusion1 in self.rules:
            for premises2, conclusion2 in self.rules:
                # If conclusion1 is in premises2, we can chain
                if conclusion1 in premises2:
                    novel_conclusions.add(conclusion2)

        return min(1.0, len(novel_conclusions) / max(len(self.rules), 1))
