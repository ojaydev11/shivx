#!/usr/bin/env python3
"""
Comprehensive Demo of Creativity & Innovation Pillar

Demonstrates all key capabilities:
1. Novel idea generation (8 techniques)
2. Conceptual blending
3. Creative problem solving (6 approaches)
"""

from agi_engine.creativity import (
    IdeaGenerator,
    ConceptualBlender,
    CreativeSolver,
    GenerationTechnique,
    BlendType,
    ProblemType,
    ThinkingMode,
    create_creativity_engine
)


def demo_idea_generation():
    """Demonstrate idea generation capabilities"""
    print("=" * 80)
    print("DEMO 1: NOVEL IDEA GENERATION")
    print("=" * 80)

    generator = IdeaGenerator()

    # Generate ideas using all techniques
    print("\n1. Generating ideas for 'sustainable urban transportation'...")
    ideas = generator.generate_ideas(
        prompt="sustainable urban transportation",
        domain="transportation",
        num_ideas=10,
        min_novelty=0.5
    )

    print(f"\n✓ Generated {len(ideas)} ideas")
    for i, idea in enumerate(ideas[:5], 1):
        print(f"\n   Idea {i}: {idea.technique.value}")
        print(f"   {idea.description}")
        print(f"   Novelty: {idea.novelty_score:.2f} | Feasibility: {idea.feasibility_score:.2f} | Impact: {idea.impact_score:.2f}")

    # Generate using specific techniques
    print("\n\n2. Generating ideas using SCAMPER and Bisociation...")
    ideas = generator.generate_ideas(
        prompt="improve online education",
        techniques=[GenerationTechnique.SCAMPER, GenerationTechnique.BISOCIATION],
        num_ideas=4
    )

    for idea in ideas:
        print(f"\n   {idea.technique.value}: {idea.description[:100]}...")

    # Get statistics
    print("\n\n3. Generator Statistics:")
    stats = generator.get_stats()
    print(f"   Total ideas: {stats['total_ideas_generated']}")
    print(f"   Avg novelty: {stats['avg_novelty']:.2f}")
    print(f"   Avg feasibility: {stats['avg_feasibility']:.2f}")
    print(f"   Avg impact: {stats['avg_impact']:.2f}")
    print(f"   Techniques used: {list(stats['by_technique'].keys())}")


def demo_conceptual_blending():
    """Demonstrate conceptual blending capabilities"""
    print("\n\n" + "=" * 80)
    print("DEMO 2: CONCEPTUAL BLENDING")
    print("=" * 80)

    blender = ConceptualBlender()

    # Simple concept blending
    print("\n1. Blending 'smartphone' + 'wallet' = ?")
    hybrid1 = blender.blend_concepts_simple(
        concept1_name="smartphone",
        concept1_attrs=["portable", "connected", "smart", "multifunctional"],
        concept2_name="wallet",
        concept2_attrs=["payment", "identity", "secure", "essential"],
        blend_name="digital_wallet"
    )

    print(f"\n   Result: {hybrid1.name}")
    print(f"   Attributes: {', '.join(list(hybrid1.attributes)[:8])}")
    print(f"   Domain: {hybrid1.domain}")

    # Another blend
    print("\n\n2. Blending 'garden' + 'skyscraper' = ?")
    hybrid2 = blender.blend_concepts_simple(
        concept1_name="garden",
        concept1_attrs=["natural", "growing", "organic", "green", "peaceful"],
        concept2_name="skyscraper",
        concept2_attrs=["vertical", "urban", "tall", "structural", "dense"],
        blend_name="vertical_garden"
    )

    print(f"\n   Result: {hybrid2.name}")
    print(f"   Attributes: {', '.join(list(hybrid2.attributes))}")

    # Complex blending with mental spaces
    print("\n\n3. Complex blending: technology + biology")
    tech_space = blender.mental_spaces.get("space_1")  # Pre-created tech space
    bio_space = blender.mental_spaces.get("space_2")   # Pre-created bio space

    if tech_space and bio_space:
        blend = blender.blend(
            input_space_ids=[tech_space.space_id, bio_space.space_id],
            blend_type=BlendType.DOUBLE_SCOPE
        )

        print(f"\n   Blend ID: {blend.blend_id}")
        print(f"   Description: {blend.description}")
        print(f"   Creativity: {blend.creativity_score:.2f}")
        print(f"   Coherence: {blend.coherence_score:.2f}")
        print(f"   Novelty: {blend.novelty_score:.2f}")
        print(f"   Emergent properties: {', '.join(blend.emergent_structure[:3])}")

    # Statistics
    print("\n\n4. Blender Statistics:")
    stats = blender.get_stats()
    print(f"   Mental spaces: {stats['mental_spaces']}")
    print(f"   Blended spaces: {stats['blended_spaces']}")
    print(f"   Avg creativity: {stats['avg_creativity']:.2f}")
    print(f"   Avg coherence: {stats['avg_coherence']:.2f}")


def demo_creative_problem_solving():
    """Demonstrate creative problem solving capabilities"""
    print("\n\n" + "=" * 80)
    print("DEMO 3: CREATIVE PROBLEM SOLVING")
    print("=" * 80)

    solver = CreativeSolver()

    # Define a problem
    print("\n1. Defining problem: 'Reduce plastic waste in oceans'")
    problem = solver.define_problem(
        description="Reduce plastic waste in oceans",
        problem_type=ProblemType.WICKED,
        constraints=["limited budget", "international cooperation needed", "existing pollution"],
        goals=["reduce plastic by 80%", "prevent new pollution", "clean existing waste"]
    )

    print(f"   Problem ID: {problem.problem_id}")
    print(f"   Type: {problem.problem_type.value}")
    print(f"   Constraints: {', '.join(problem.constraints)}")
    print(f"   Goals: {', '.join(problem.goals)}")

    # Generate solutions using multiple approaches
    print("\n\n2. Generating creative solutions using 6 different approaches...")
    solutions = solver.solve_creatively(
        problem=problem,
        approaches=["design_thinking", "triz", "constraint_relaxation", "reframing", "analogical", "lateral"],
        thinking_mode=ThinkingMode.DIVERGENT,
        num_solutions=12
    )

    print(f"\n   ✓ Generated {len(solutions)} solutions\n")

    # Show top 5 solutions
    for i, solution in enumerate(solutions[:5], 1):
        print(f"\n   Solution {i} [{solution.approach}]:")
        print(f"   {solution.description[:150]}...")
        print(f"   Type: {solution.solution_type.value}")
        eval_result = solver.evaluate_solution(solution)
        print(f"   Overall Score: {eval_result['overall_score']:.2f}")
        print(f"   Creativity: {solution.creativity_score:.2f} | "
              f"Feasibility: {solution.feasibility_score:.2f} | "
              f"Impact: {solution.impact_score:.2f}")
        if solution.advantages:
            print(f"   Advantages: {', '.join(solution.advantages[:2])}")

    # Generate alternatives for best solution
    print("\n\n3. Generating alternatives for the best solution...")
    best_solution = solutions[0]
    alternatives = solver.generate_alternatives(best_solution, num_alternatives=3)

    for i, alt in enumerate(alternatives, 1):
        print(f"\n   Alternative {i}:")
        print(f"   {alt.description[:120]}...")

    # Combine solutions
    if len(solutions) >= 2:
        print("\n\n4. Combining two solutions into a hybrid...")
        hybrid = solver.combine_solutions(
            solutions[0],
            solutions[1],
            combination_type="hybrid"
        )
        print(f"\n   Hybrid solution:")
        print(f"   {hybrid.description[:150]}...")
        print(f"   Score: {hybrid.overall_score():.2f}")

    # Statistics
    print("\n\n5. Solver Statistics:")
    stats = solver.get_stats()
    print(f"   Problems defined: {stats['problems_defined']}")
    print(f"   Solutions generated: {stats['solutions_generated']}")
    print(f"   Approaches used: {list(stats['by_approach'].keys())[:5]}")
    print(f"   Avg creativity: {stats['avg_creativity']:.2f}")
    print(f"   Avg feasibility: {stats['avg_feasibility']:.2f}")


def demo_integrated_workflow():
    """Demonstrate integrated creativity workflow"""
    print("\n\n" + "=" * 80)
    print("DEMO 4: INTEGRATED CREATIVITY WORKFLOW")
    print("=" * 80)

    # Create full creativity engine
    idea_gen, blender, solver = create_creativity_engine()

    print("\n1. Generating ideas for 'future of work'...")
    ideas = idea_gen.generate_ideas(
        prompt="future of work",
        domain="social",
        num_ideas=5,
        techniques=[GenerationTechnique.BISOCIATION, GenerationTechnique.ANALOGICAL]
    )

    best_idea = ideas[0]
    print(f"\n   Best idea: {best_idea.description[:100]}...")
    print(f"   Technique: {best_idea.technique.value}")

    print("\n\n2. Blending concepts from the idea...")
    if len(best_idea.components) >= 2:
        blend = blender.blend_concepts_simple(
            concept1_name=best_idea.components[0],
            concept1_attrs=["innovative", "adaptive", "scalable"],
            concept2_name=best_idea.components[1],
            concept2_attrs=["efficient", "practical", "accessible"]
        )
        print(f"\n   Blended concept: {blend.name}")
        print(f"   Combined attributes: {', '.join(list(blend.attributes)[:6])}")

    print("\n\n3. Defining problem based on insights...")
    problem = solver.define_problem(
        description=f"Implement {best_idea.description[:50]}",
        problem_type=ProblemType.ILL_DEFINED,
        goals=["feasibility", "scalability", "impact"]
    )

    print("\n\n4. Generating creative implementation solutions...")
    solutions = solver.solve_creatively(
        problem=problem,
        approaches=["design_thinking", "triz"],
        thinking_mode=ThinkingMode.CREATIVE,
        num_solutions=5
    )

    print(f"\n   ✓ Generated {len(solutions)} implementation solutions")
    best_solution = solutions[0]
    print(f"\n   Best solution:")
    print(f"   {best_solution.description[:120]}...")
    print(f"   Score: {best_solution.overall_score():.2f}")

    print("\n\n5. Complete workflow summary:")
    print(f"   • Started with idea generation")
    print(f"   • Blended concepts creatively")
    print(f"   • Defined implementation problem")
    print(f"   • Generated creative solutions")
    print(f"   • Result: Comprehensive innovation pipeline")


def main():
    """Run all demonstrations"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "CREATIVITY ENGINE DEMONSTRATION" + " " * 27 + "█")
    print("█" + " " * 24 + "ShivX AGI - Pillar 10" + " " * 33 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    try:
        # Run all demos
        demo_idea_generation()
        demo_conceptual_blending()
        demo_creative_problem_solving()
        demo_integrated_workflow()

        print("\n\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("\n✓ All creativity components working successfully!")
        print("✓ Pillar 10: Creativity & Innovation is fully operational")

        print("\n\nKey Capabilities Demonstrated:")
        print("  1. Novel Idea Generation (8 techniques)")
        print("  2. Conceptual Blending & Integration")
        print("  3. Design Thinking Methodology")
        print("  4. TRIZ Inventive Problem Solving")
        print("  5. Divergent & Lateral Thinking")
        print("  6. Constraint Manipulation")
        print("  7. Problem Reframing")
        print("  8. Analogical Reasoning")
        print("  9. Integrated Creativity Workflow")

    except Exception as e:
        print(f"\n\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
