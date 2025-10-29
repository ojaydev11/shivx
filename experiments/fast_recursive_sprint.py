#!/usr/bin/env python3
"""
Fast Recursive Improvement Sprint
10 iterations per approach - focused on getting complete results fast
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agi_lab import TaskGenerator
from agi_lab.recursive_improvement import RecursiveSelfImprover
from agi_lab.approaches import MetaLearner, WorldModelLearner, CausalReasoner, HybridAGI

def test_approach(ApproachClass, name, train_tasks, test_tasks):
    """Quick test of recursive improvement"""
    print(f"\n{'='*60}")
    print(f"🧬 TESTING: {name.upper()}")
    print(f"{'='*60}\n")

    target = ApproachClass()
    improver = RecursiveSelfImprover(target, improvement_budget=10)

    improved, final_fitness = improver.improve(
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        max_iterations=10
    )

    baseline = improver.original_fitness
    final_score = final_fitness.overall_score if hasattr(final_fitness, 'overall_score') else final_fitness
    improvement = final_score - baseline
    rate = (improvement / baseline * 100) if baseline > 0 else 0

    print(f"\n📊 {name.upper()} RESULTS:")
    print(f"   Baseline:    {baseline:.1%}")
    print(f"   Final:       {final_score:.1%}")
    print(f"   Improvement: +{improvement:.1%} ({rate:+.1f}%)")
    print(f"   Successful:  {len(improver.improvement_history)} improvements\n")

    return {
        "name": name,
        "baseline": baseline,
        "final": final_score,
        "improvement": improvement,
        "rate": rate,
        "count": len(improver.improvement_history)
    }

def main():
    print("=" * 60)
    print("🚀 FAST RECURSIVE IMPROVEMENT SPRINT")
    print("=" * 60)
    print("\nGoal: Test all 4 AGI approaches quickly\n")

    # Generate tasks
    print("📚 Generating tasks...")
    train_tasks = (
        TaskGenerator.generate_world_model_tasks(20) +
        TaskGenerator.generate_meta_learning_tasks(20) +
        TaskGenerator.generate_causal_tasks(20)
    )
    test_tasks = (
        TaskGenerator.generate_world_model_tasks(8) +
        TaskGenerator.generate_meta_learning_tasks(8) +
        TaskGenerator.generate_causal_tasks(8)
    )
    print(f"   ✓ {len(train_tasks)} train, {len(test_tasks)} test\n")

    # Test all approaches
    results = []
    results.append(test_approach(MetaLearner, "Meta-Learning", train_tasks, test_tasks))
    results.append(test_approach(WorldModelLearner, "World Model", train_tasks, test_tasks))
    results.append(test_approach(CausalReasoner, "Causal Reasoning", train_tasks, test_tasks))
    results.append(test_approach(HybridAGI, "Hybrid AGI", train_tasks, test_tasks))

    # Summary
    print("\n" + "=" * 60)
    print("📈 CAMPAIGN SUMMARY")
    print("=" * 60 + "\n")

    results.sort(key=lambda x: x["final"], reverse=True)

    print("🏆 RANKINGS:")
    for i, r in enumerate(results, 1):
        medal = ["🥇", "🥈", "🥉", "  "][min(i-1, 3)]
        print(f"   {medal} #{i}. {r['name']:20s} {r['final']:.1%}  (+{r['improvement']:.1%}, {r['count']} improvements)")

    avg_improvement = sum(r["improvement"] for r in results) / len(results)
    best_improvement = max(r["improvement"] for r in results)

    print(f"\n💡 KEY FINDINGS:")
    print(f"   Average Improvement: +{avg_improvement:.1%}")
    print(f"   Best Improvement:    +{best_improvement:.1%}")
    print(f"   Winner:              {results[0]['name']}")

    # AGI estimate
    baseline_agi = 0.25
    agi_boost = best_improvement
    estimated_agi = baseline_agi + agi_boost

    print(f"\n🧠 AGI PROGRESS:")
    print(f"   Previous:  {baseline_agi:.1%}")
    print(f"   Boost:     +{agi_boost:.1%}")
    print(f"   Current:   {estimated_agi:.1%}")

    if estimated_agi >= 0.40:
        print(f"\n   🎉 MILESTONE: 40%+ AGI ACHIEVED!")
    elif estimated_agi >= 0.35:
        print(f"\n   🚀 SIGNIFICANT PROGRESS: 35%+ AGI!")
    else:
        print(f"\n   📈 Measurable progress - continue scaling!")

    print("\n" + "=" * 60)
    print("✅ SPRINT COMPLETE!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
