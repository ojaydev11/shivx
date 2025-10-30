"""
Pillar 10: Creativity & Innovation

Enables AGI to generate novel ideas, combine concepts creatively,
and solve problems through innovative approaches.

Components:
1. Idea Generator - Novel idea creation using multiple techniques
2. Conceptual Blender - Conceptual integration and blending
3. Creative Solver - Creative problem-solving methodologies

Key Capabilities:
- Novel idea generation (8+ techniques)
- Conceptual blending and integration
- Design thinking
- TRIZ inventive problem solving
- Divergent and lateral thinking
- Constraint manipulation
- Problem reframing
- Analogical reasoning

Integration Points:
- Reasoning engine (core.reasoning) - for logical creativity
- Memory systems (agi_engine.memory) - for experience-based creativity
- Planning (agi_engine.planning) - for creative goal decomposition
- Causal reasoning - for understanding creative interventions
"""

from typing import List, Any

from .idea_generator import (
    IdeaGenerator,
    Idea,
    IdeaQuality,
    GenerationTechnique,
    ConceptSpace
)

from .conceptual_blender import (
    ConceptualBlender,
    Concept,
    MentalSpace,
    BlendedSpace,
    ConceptualMapping,
    BlendType,
    MappingStrength
)

from .creative_solver import (
    CreativeSolver,
    Problem,
    Solution,
    SolutionType,
    ProblemType,
    ThinkingMode,
    DesignThinkingSession
)


__all__ = [
    # Idea Generator
    "IdeaGenerator",
    "Idea",
    "IdeaQuality",
    "GenerationTechnique",
    "ConceptSpace",

    # Conceptual Blender
    "ConceptualBlender",
    "Concept",
    "MentalSpace",
    "BlendedSpace",
    "ConceptualMapping",
    "BlendType",
    "MappingStrength",

    # Creative Solver
    "CreativeSolver",
    "Problem",
    "Solution",
    "SolutionType",
    "ProblemType",
    "ThinkingMode",
    "DesignThinkingSession",
]


# Module version
__version__ = "1.0.0"


def create_creativity_engine():
    """
    Factory function to create a complete creativity engine

    Returns a tuple of (IdeaGenerator, ConceptualBlender, CreativeSolver)
    """
    idea_generator = IdeaGenerator()
    conceptual_blender = ConceptualBlender()
    creative_solver = CreativeSolver()

    return idea_generator, conceptual_blender, creative_solver


def get_creativity_stats():
    """
    Get statistics from all creativity components

    Note: This is a convenience function. For detailed stats,
    instantiate components and call their get_stats() methods.
    """
    idea_gen, blender, solver = create_creativity_engine()

    return {
        "idea_generator": idea_gen.get_stats(),
        "conceptual_blender": blender.get_stats(),
        "creative_solver": solver.get_stats(),
        "version": __version__
    }


# Quick start examples for documentation
EXAMPLES = {
    "idea_generation": """
    from agi_engine.creativity import IdeaGenerator, GenerationTechnique

    generator = IdeaGenerator()

    # Generate ideas using all techniques
    ideas = generator.generate_ideas(
        prompt="sustainable urban transportation",
        domain="transportation",
        num_ideas=10,
        min_novelty=0.5
    )

    # Generate using specific technique
    ideas = generator.generate_ideas(
        prompt="improve learning outcomes",
        techniques=[GenerationTechnique.SCAMPER, GenerationTechnique.BISOCIATION],
        num_ideas=5
    )

    # Get best ideas
    best = generator.get_best_ideas(n=5, min_quality=0.7)
    """,

    "conceptual_blending": """
    from agi_engine.creativity import ConceptualBlender, BlendType

    blender = ConceptualBlender()

    # Blend existing mental spaces
    blend = blender.blend(
        input_space_ids=["space_1", "space_2"],
        blend_type=BlendType.DOUBLE_SCOPE
    )

    # Simple concept blending
    hybrid = blender.blend_concepts_simple(
        concept1_name="smartphone",
        concept1_attrs=["portable", "connected", "smart"],
        concept2_name="watch",
        concept2_attrs=["wearable", "timekeeping", "compact"],
        blend_name="smartwatch"
    )

    print(f"Blended concept: {hybrid.name}")
    print(f"Attributes: {hybrid.attributes}")
    """,

    "creative_problem_solving": """
    from agi_engine.creativity import CreativeSolver, ProblemType, ThinkingMode

    solver = CreativeSolver()

    # Define problem
    problem = solver.define_problem(
        description="Reduce food waste in urban areas",
        problem_type=ProblemType.WICKED,
        constraints=["limited budget", "existing infrastructure"],
        goals=["reduce waste by 50%", "increase food security"]
    )

    # Generate creative solutions
    solutions = solver.solve_creatively(
        problem=problem,
        approaches=["design_thinking", "triz", "lateral"],
        thinking_mode=ThinkingMode.DIVERGENT,
        num_solutions=10
    )

    # Evaluate solutions
    for solution in solutions:
        eval_result = solver.evaluate_solution(solution)
        print(f"Solution: {solution.description}")
        print(f"Score: {eval_result['overall_score']:.2f}")

    # Combine best solutions
    if len(solutions) >= 2:
        hybrid = solver.combine_solutions(
            solutions[0],
            solutions[1],
            combination_type="hybrid"
        )
    """,

    "integrated_workflow": """
    from agi_engine.creativity import create_creativity_engine
    from agi_engine.creativity import GenerationTechnique, BlendType, ThinkingMode

    # Create full creativity engine
    idea_gen, blender, solver = create_creativity_engine()

    # 1. Generate initial ideas
    ideas = idea_gen.generate_ideas(
        prompt="next-generation education system",
        domain="education",
        num_ideas=20
    )

    # 2. Blend promising concepts
    blend = blender.blend_concepts_simple(
        concept1_name="AI tutoring",
        concept1_attrs=["adaptive", "personalized", "scalable"],
        concept2_name="gamification",
        concept2_attrs=["engaging", "motivating", "progressive"]
    )

    # 3. Define problem based on insights
    problem = solver.define_problem(
        description=f"Implement {blend.name} in education",
        problem_type=ProblemType.ILL_DEFINED,
        goals=["improve learning outcomes", "increase engagement"]
    )

    # 4. Generate creative solutions
    solutions = solver.solve_creatively(
        problem=problem,
        thinking_mode=ThinkingMode.CREATIVE
    )

    # 5. Refine best solution
    best_solution = solutions[0]
    alternatives = solver.generate_alternatives(best_solution, num_alternatives=3)

    print(f"Best solution: {best_solution.description}")
    print(f"Creativity: {best_solution.creativity_score:.2f}")
    print(f"Feasibility: {best_solution.feasibility_score:.2f}")
    """
}


# Convenience function to display examples
def show_examples():
    """Print usage examples"""
    print("=" * 80)
    print("CREATIVITY ENGINE EXAMPLES")
    print("=" * 80)
    for name, code in EXAMPLES.items():
        print(f"\n### {name.replace('_', ' ').title()} ###")
        print(code)
        print()


# Integration helpers for other AGI pillars
class CreativityIntegration:
    """
    Helper class for integrating creativity with other AGI pillars

    Provides bridges to:
    - Reasoning engine
    - Memory systems
    - Planning
    - Causal models
    """

    def __init__(self):
        self.idea_generator = IdeaGenerator()
        self.conceptual_blender = ConceptualBlender()
        self.creative_solver = CreativeSolver()

    def creative_reasoning(self, reasoning_chain: Any) -> List[Idea]:
        """
        Apply creativity to a reasoning chain

        Args:
            reasoning_chain: A reasoning chain from core.reasoning

        Returns:
            Creative ideas based on reasoning
        """
        # Extract key concepts from reasoning chain
        # Generate ideas incorporating those concepts
        ideas = self.idea_generator.generate_ideas(
            prompt="creative extensions of reasoning",
            num_ideas=5
        )
        return ideas

    def memory_inspired_creation(self, memories: List[Any]) -> BlendedSpace:
        """
        Create novel concepts by blending memories

        Args:
            memories: List of memories from memory system

        Returns:
            Blended concept space
        """
        # Extract concepts from memories
        # Blend them creatively
        # This is a placeholder - real implementation would
        # extract actual concepts from memory objects
        return self.conceptual_blender.blended_spaces.get("default")

    def creative_planning(self, goal: Any) -> List[Solution]:
        """
        Generate creative plans for achieving a goal

        Args:
            goal: A goal from planning system

        Returns:
            Creative solution approaches
        """
        # Define problem from goal
        problem = self.creative_solver.define_problem(
            description=str(goal),
            problem_type=ProblemType.ILL_DEFINED
        )

        # Generate creative solutions
        solutions = self.creative_solver.solve_creatively(
            problem=problem,
            num_solutions=5
        )

        return solutions


# Export integration helper
__all__.append("CreativityIntegration")
