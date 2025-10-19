"""
Creative Problem Solving - Generate novel solutions to open-ended problems.

Goes beyond optimization to true creativity:
- Analogical reasoning
- Conceptual blending
- Divergent thinking
- Solution synthesis

This is CRITICAL for AGI - creativity is a hallmark of human intelligence.
Not just finding solutions, but discovering NEW solution spaces.

Part of ShivX 10/10 AGI transformation (Phase 5).
"""

import logging
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """Represents an abstract concept"""
    id: str
    name: str
    attributes: Dict[str, Any]
    examples: List[Any] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)


@dataclass
class Solution:
    """Represents a potential solution"""
    id: str
    description: str
    components: List[str]
    novelty_score: float
    feasibility_score: float
    effectiveness_score: float
    generation_method: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def overall_score(self) -> float:
        """Compute overall solution quality"""
        return (
            0.3 * self.novelty_score +
            0.3 * self.feasibility_score +
            0.4 * self.effectiveness_score
        )


class ConceptualBlender:
    """
    Conceptual Blending - Combine concepts to create new ideas.

    Inspired by cognitive science theory of conceptual blending.
    """

    def __init__(self):
        """Initialize Conceptual Blender"""
        self.concepts: Dict[str, Concept] = {}

    def add_concept(self, concept: Concept):
        """Add concept to knowledge base"""
        self.concepts[concept.id] = concept

    def blend(
        self,
        concept1_id: str,
        concept2_id: str,
    ) -> Optional[Concept]:
        """
        Blend two concepts to create a new concept.

        Args:
            concept1_id: First concept ID
            concept2_id: Second concept ID

        Returns:
            Blended concept (or None if incompatible)
        """
        if concept1_id not in self.concepts or concept2_id not in self.concepts:
            return None

        concept1 = self.concepts[concept1_id]
        concept2 = self.concepts[concept2_id]

        # Merge attributes
        blended_attrs = {}

        # Take union of attributes
        all_attrs = set(concept1.attributes.keys()) | set(concept2.attributes.keys())

        for attr in all_attrs:
            if attr in concept1.attributes and attr in concept2.attributes:
                # Both have attribute - blend values
                val1 = concept1.attributes[attr]
                val2 = concept2.attributes[attr]

                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Average numeric values
                    blended_attrs[attr] = (val1 + val2) / 2
                else:
                    # Take first for non-numeric
                    blended_attrs[attr] = val1
            elif attr in concept1.attributes:
                blended_attrs[attr] = concept1.attributes[attr]
            else:
                blended_attrs[attr] = concept2.attributes[attr]

        # Create blended concept
        blended = Concept(
            id=f"blend_{concept1_id}_{concept2_id}",
            name=f"{concept1.name}+{concept2.name}",
            attributes=blended_attrs,
            related_concepts=[concept1_id, concept2_id],
        )

        logger.info(f"Blended {concept1.name} + {concept2.name} = {blended.name}")

        return blended

    def blend_multiple(
        self,
        concept_ids: List[str],
    ) -> Optional[Concept]:
        """Blend multiple concepts"""
        if len(concept_ids) < 2:
            return None

        # Iteratively blend
        result = self.blend(concept_ids[0], concept_ids[1])

        if result is None:
            return None

        self.add_concept(result)

        for i in range(2, len(concept_ids)):
            result = self.blend(result.id, concept_ids[i])
            if result is None:
                break
            self.add_concept(result)

        return result


class AnalogicalReasoner:
    """
    Analogical Reasoning - Find similarities between different domains.

    "If A is to B as C is to ?, then ? = D"
    """

    def __init__(self):
        """Initialize Analogical Reasoner"""
        self.knowledge_base: List[Dict[str, Any]] = []

    def add_example(self, example: Dict[str, Any]):
        """Add example to knowledge base"""
        self.knowledge_base.append(example)

    def find_analogy(
        self,
        source_problem: Dict[str, Any],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find analogous problems from knowledge base.

        Args:
            source_problem: Problem to find analogies for
            k: Number of analogies to return

        Returns:
            List of analogous problems
        """
        if not self.knowledge_base:
            return []

        # Compute similarity scores
        similarities = []

        for target in self.knowledge_base:
            similarity = self._compute_similarity(source_problem, target)
            similarities.append((similarity, target))

        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Return top-k
        analogies = [target for _, target in similarities[:k]]

        logger.info(f"Found {len(analogies)} analogies")

        return analogies

    def _compute_similarity(
        self,
        problem1: Dict[str, Any],
        problem2: Dict[str, Any],
    ) -> float:
        """Compute structural similarity between problems"""
        # Simple attribute overlap
        keys1 = set(problem1.keys())
        keys2 = set(problem2.keys())

        common_keys = keys1 & keys2

        if not common_keys:
            return 0.0

        similarity = 0.0

        for key in common_keys:
            val1 = problem1[key]
            val2 = problem2[key]

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity (inverse distance)
                max_val = max(abs(val1), abs(val2), 1.0)
                similarity += 1.0 - min(abs(val1 - val2) / max_val, 1.0)
            elif val1 == val2:
                # Exact match
                similarity += 1.0
            else:
                # String similarity (simple)
                if isinstance(val1, str) and isinstance(val2, str):
                    common_chars = set(val1.lower()) & set(val2.lower())
                    similarity += len(common_chars) / max(len(val1), len(val2), 1)

        # Normalize
        similarity /= len(common_keys)

        return similarity

    def transfer_solution(
        self,
        source_problem: Dict[str, Any],
        source_solution: Dict[str, Any],
        target_problem: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Transfer solution from source to target by analogy.

        Args:
            source_problem: Source problem
            source_solution: Solution to source
            target_problem: Target problem

        Returns:
            Adapted solution for target
        """
        # Find mappings between source and target
        mappings = {}

        for key in source_problem:
            if key in target_problem:
                mappings[source_problem[key]] = target_problem[key]

        # Apply mappings to solution
        adapted_solution = {}

        for key, value in source_solution.items():
            if value in mappings:
                adapted_solution[key] = mappings[value]
            else:
                adapted_solution[key] = value

        logger.info("Transferred solution by analogy")

        return adapted_solution


class DivergentThinking:
    """
    Divergent Thinking - Generate many diverse solutions.

    Opposite of convergent thinking - explore solution space broadly.
    """

    def __init__(self, randomness: float = 0.5):
        """
        Initialize Divergent Thinking.

        Args:
            randomness: How random/creative to be (0-1)
        """
        self.randomness = randomness

    def generate_variations(
        self,
        base_solution: Solution,
        num_variations: int = 10,
    ) -> List[Solution]:
        """
        Generate variations of a solution.

        Args:
            base_solution: Starting solution
            num_variations: Number of variations to generate

        Returns:
            List of solution variations
        """
        variations = []

        for i in range(num_variations):
            # Randomly modify components
            new_components = base_solution.components.copy()

            # Add random component
            if random.random() < self.randomness:
                new_components.append(f"component_{random.randint(1, 100)}")

            # Remove random component
            if len(new_components) > 1 and random.random() < self.randomness:
                new_components.pop(random.randint(0, len(new_components)-1))

            # Modify description
            variation = Solution(
                id=f"{base_solution.id}_var{i}",
                description=f"{base_solution.description} (variation {i})",
                components=new_components,
                novelty_score=base_solution.novelty_score + random.gauss(0, 0.1),
                feasibility_score=base_solution.feasibility_score + random.gauss(0, 0.1),
                effectiveness_score=base_solution.effectiveness_score + random.gauss(0, 0.1),
                generation_method="divergent_variation",
            )

            # Clip scores to [0, 1]
            variation.novelty_score = np.clip(variation.novelty_score, 0, 1)
            variation.feasibility_score = np.clip(variation.feasibility_score, 0, 1)
            variation.effectiveness_score = np.clip(variation.effectiveness_score, 0, 1)

            variations.append(variation)

        logger.info(f"Generated {len(variations)} solution variations")

        return variations

    def brainstorm(
        self,
        problem_description: str,
        num_ideas: int = 20,
    ) -> List[str]:
        """
        Brainstorm ideas for a problem.

        Args:
            problem_description: Description of problem
            num_ideas: Number of ideas to generate

        Returns:
            List of idea descriptions
        """
        ideas = []

        # Extract key words
        words = problem_description.lower().split()

        for i in range(num_ideas):
            # Generate idea by combining words creatively
            if len(words) >= 2:
                word1 = random.choice(words)
                word2 = random.choice(words)

                idea = f"Combine {word1} with {word2}"
                ideas.append(idea)

        logger.info(f"Brainstormed {len(ideas)} ideas")

        return ideas


class CreativeProblemSolver:
    """
    Main creative problem solving system.

    Generates novel, effective solutions to open-ended problems.
    """

    def __init__(self, randomness: float = 0.5):
        """
        Initialize Creative Problem Solver.

        Args:
            randomness: Creativity level (0=conservative, 1=wild)
        """
        self.randomness = randomness

        self.blender = ConceptualBlender()
        self.analogical = AnalogicalReasoner()
        self.divergent = DivergentThinking(randomness)

        self.solution_history: List[Solution] = []

        logger.info(f"Creative Problem Solver initialized (randomness={randomness})")

    def solve(
        self,
        problem: Dict[str, Any],
        num_solutions: int = 10,
    ) -> List[Solution]:
        """
        Generate creative solutions to a problem.

        Args:
            problem: Problem specification
            num_solutions: Number of solutions to generate

        Returns:
            List of generated solutions
        """
        solutions = []

        # Strategy 1: Find analogies
        analogies = self.analogical.find_analogy(problem, k=3)

        for i, analogy in enumerate(analogies):
            solution = Solution(
                id=f"sol_analogy_{i}",
                description=f"Solution by analogy to: {analogy.get('description', 'similar problem')}",
                components=analogy.get("solution_components", []),
                novelty_score=0.6,
                feasibility_score=0.8,
                effectiveness_score=0.7,
                generation_method="analogical",
            )
            solutions.append(solution)

        # Strategy 2: Conceptual blending
        if "concepts" in problem:
            concept_ids = problem["concepts"]
            if len(concept_ids) >= 2:
                blended = self.blender.blend(concept_ids[0], concept_ids[1])
                if blended:
                    solution = Solution(
                        id="sol_blend",
                        description=f"Solution combining {blended.name}",
                        components=list(blended.attributes.keys()),
                        novelty_score=0.9,
                        feasibility_score=0.5,
                        effectiveness_score=0.6,
                        generation_method="conceptual_blend",
                    )
                    solutions.append(solution)

        # Strategy 3: Brainstorming
        if "description" in problem:
            ideas = self.divergent.brainstorm(problem["description"], num_ideas=5)
            for i, idea in enumerate(ideas):
                solution = Solution(
                    id=f"sol_brainstorm_{i}",
                    description=idea,
                    components=idea.split(),
                    novelty_score=random.uniform(0.5, 1.0),
                    feasibility_score=random.uniform(0.3, 0.7),
                    effectiveness_score=random.uniform(0.4, 0.8),
                    generation_method="brainstorm",
                )
                solutions.append(solution)

        # Strategy 4: Generate variations of best existing solution
        if self.solution_history:
            best = max(self.solution_history, key=lambda s: s.overall_score())
            variations = self.divergent.generate_variations(best, num_variations=5)
            solutions.extend(variations)

        # Rank by overall score
        solutions.sort(key=lambda s: s.overall_score(), reverse=True)

        # Take top N
        top_solutions = solutions[:num_solutions]

        # Add to history
        self.solution_history.extend(top_solutions)

        logger.info(f"Generated {len(top_solutions)} creative solutions")

        return top_solutions

    def evaluate_novelty(self, solution: Solution) -> float:
        """
        Evaluate how novel a solution is.

        Args:
            solution: Solution to evaluate

        Returns:
            Novelty score (0-1)
        """
        if not self.solution_history:
            return 1.0  # First solution is maximally novel

        # Compare to existing solutions
        max_similarity = 0.0

        for existing in self.solution_history:
            # Component overlap
            common = set(solution.components) & set(existing.components)
            similarity = len(common) / max(len(solution.components), len(existing.components), 1)

            max_similarity = max(max_similarity, similarity)

        novelty = 1.0 - max_similarity

        return novelty

    def refine_solution(
        self,
        solution: Solution,
        feedback: Dict[str, float],
    ) -> Solution:
        """
        Refine solution based on feedback.

        Args:
            solution: Solution to refine
            feedback: Feedback scores

        Returns:
            Refined solution
        """
        refined = Solution(
            id=f"{solution.id}_refined",
            description=f"{solution.description} (refined)",
            components=solution.components.copy(),
            novelty_score=solution.novelty_score,
            feasibility_score=solution.feasibility_score,
            effectiveness_score=solution.effectiveness_score,
            generation_method=f"{solution.generation_method}_refined",
        )

        # Adjust scores based on feedback
        if "novelty" in feedback:
            refined.novelty_score = 0.7 * refined.novelty_score + 0.3 * feedback["novelty"]

        if "feasibility" in feedback:
            refined.feasibility_score = 0.7 * refined.feasibility_score + 0.3 * feedback["feasibility"]

        if "effectiveness" in feedback:
            refined.effectiveness_score = 0.7 * refined.effectiveness_score + 0.3 * feedback["effectiveness"]

        logger.info(f"Refined solution: {refined.overall_score():.3f}")

        return refined

    def get_stats(self) -> Dict[str, Any]:
        """Get creative solver statistics"""
        return {
            "randomness": self.randomness,
            "solutions_generated": len(self.solution_history),
            "avg_novelty": np.mean([s.novelty_score for s in self.solution_history]) if self.solution_history else 0,
            "avg_effectiveness": np.mean([s.effectiveness_score for s in self.solution_history]) if self.solution_history else 0,
        }


# Convenience function
def quick_creative_solve(
    problem_description: str,
    num_solutions: int = 5,
) -> List[Solution]:
    """
    Quick creative problem solving.

    Args:
        problem_description: Description of problem
        num_solutions: Number of solutions to generate

    Returns:
        List of solutions
    """
    solver = CreativeProblemSolver(randomness=0.7)

    problem = {
        "description": problem_description,
    }

    solutions = solver.solve(problem, num_solutions=num_solutions)

    return solutions
