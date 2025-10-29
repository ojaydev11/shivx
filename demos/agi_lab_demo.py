#!/usr/bin/env python3
"""
AGI Lab Demo
Demonstrates parallel exploration of AGI approaches
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agi_lab import (
    ParallelExplorer,
    TaskGenerator,
    PatternRecorder,
)


def main():
    """Run AGI lab demo"""
    print("=" * 70)
    print("🧠 AGI LAB - PARALLEL AGI EXPLORATION")
    print("=" * 70)
    print()
    print("This demo runs 20 different AGI approaches in parallel:")
    print("  • World Model Learners (learn physics and causality)")
    print("  • Meta-Learners (learn to learn)")
    print("  • Causal Reasoners (understand cause and effect)")
    print()
    print("Each approach tries to solve diverse tasks.")
    print("The best performers are selected and combined (cross-pollination).")
    print("This simulates brain-like parallel exploration!")
    print()

    # Generate tasks
    print("📚 Generating tasks...")
    train_tasks = []
    train_tasks.extend(TaskGenerator.generate_world_model_tasks(num_tasks=20))
    train_tasks.extend(TaskGenerator.generate_meta_learning_tasks(num_tasks=20))
    train_tasks.extend(TaskGenerator.generate_causal_tasks(num_tasks=20))

    test_tasks = []
    test_tasks.extend(TaskGenerator.generate_world_model_tasks(num_tasks=10))
    test_tasks.extend(TaskGenerator.generate_meta_learning_tasks(num_tasks=10))
    test_tasks.extend(TaskGenerator.generate_causal_tasks(num_tasks=10))

    transfer_tasks = TaskGenerator.generate_transfer_tasks(num_tasks=15)

    print(f"   Train tasks: {len(train_tasks)}")
    print(f"   Test tasks: {len(test_tasks)}")
    print(f"   Transfer tasks: {len(transfer_tasks)}")
    print()

    # Create explorer
    explorer = ParallelExplorer(
        num_parallel=20,
        max_workers=10,
    )

    # Run exploration
    session = explorer.explore(
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        transfer_tasks=transfer_tasks,
        max_generations=3,  # Quick demo
    )

    # Show results
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print()

    best = explorer.get_best_approach()
    if best:
        print(f"🏆 Best Approach: {best.approach_type.value}")
        print(f"   Task Success Rate: {best.task_success_rate:.1%}")
        print(f"   Generalization: {best.generalization_score:.1%}")
        print(f"   Transfer Learning: {best.transfer_learning_score:.1%}")
        print(f"   Training Time: {best.training_time_sec:.2f}s")
        print(f"   Patterns Recorded: {len(best.patterns)}")
        print()

    # Analyze patterns
    print("🧬 Pattern Analysis:")
    recorder = PatternRecorder()

    from agi_lab.schemas import AGIApproachType

    for approach_type in [AGIApproachType.WORLD_MODEL, AGIApproachType.META_LEARNING, AGIApproachType.CAUSAL]:
        analysis = recorder.analyze_approach(approach_type)
        print(f"\n   {approach_type.value}:")
        print(f"      Total patterns: {analysis['total_patterns']}")
        print(f"      Avg success: {analysis['avg_success']:.2%}")
        print(f"      Avg generalization: {analysis['avg_generalization']:.2%}")
        print(f"      Max success: {analysis['max_success']:.2%}")

    print()
    print("=" * 70)
    print("✅ AGI Lab Demo Complete!")
    print()
    print("💡 Next steps to reach AGI:")
    print("   1. Add more approach types (neurosymbolic, active inference, etc.)")
    print("   2. Implement better cross-pollination (genetic algorithms)")
    print("   3. Add more sophisticated fitness metrics")
    print("   4. Scale to thousands of parallel experiments")
    print("   5. Test on increasingly complex tasks")
    print("   6. Implement recursive self-improvement")
    print()
    print("This is a training ground for AGI research! 🚀")
    print("=" * 70)


if __name__ == "__main__":
    main()
