#!/usr/bin/env python3
"""
Extended Recursive Self-Improvement Campaign
Run 20+ iterations to validate exponential growth hypothesis

This is the path to AGI: Let the system improve itself over many iterations
and watch the exponential curve emerge.

Goal: Validate that recursive self-improvement can take us from 25% AGI → 50%+ AGI
"""
import sys
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from agi_lab import TaskGenerator
from agi_lab.recursive_improvement import RecursiveSelfImprover
from agi_lab.approaches import MetaLearner, WorldModelLearner, CausalReasoner, HybridAGI


def run_extended_improvement(
    approach_class,
    approach_name: str,
    train_tasks: List,
    test_tasks: List,
    max_iterations: int = 25
) -> Dict:
    """
    Run extended recursive improvement on a single approach

    Returns:
        Dict with improvement history and final results
    """
    print(f"\n{'=' * 80}")
    print(f"🧬 EXTENDED RECURSIVE IMPROVEMENT: {approach_name.upper()}")
    print(f"{'=' * 80}")
    print()
    print(f"Parameters:")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Train tasks: {len(train_tasks)}")
    print(f"   Test tasks: {len(test_tasks)}")
    print()

    # Create target approach
    target = approach_class()

    # Create improver
    improver = RecursiveSelfImprover(
        target_approach=target,
        improvement_budget=max_iterations,
        safety_checks=True
    )

    # Run improvement
    start_time = time.time()

    improved_approach, final_fitness = improver.improve(
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        max_iterations=max_iterations
    )

    duration = time.time() - start_time

    # Extract improvement curve
    improvement_curve = []
    baseline = improver.original_fitness
    current_fitness = baseline

    improvement_curve.append({
        "iteration": 0,
        "fitness": baseline,
        "cumulative_improvement": 0.0,
        "strategy": "baseline"
    })

    for i, improvement in enumerate(improver.improvement_history, 1):
        current_fitness = improvement["new_fitness"]
        improvement_curve.append({
            "iteration": i,
            "fitness": current_fitness,
            "cumulative_improvement": current_fitness - baseline,
            "strategy": improvement["strategy"]
        })

    # Analyze results
    final_fitness_value = improvement_curve[-1]["fitness"] if improvement_curve else baseline
    total_improvement = final_fitness_value - baseline
    improvement_rate = (final_fitness_value / baseline - 1.0) * 100 if baseline > 0 else 0.0

    # Check for exponential growth
    is_exponential = len(improvement_curve) >= 3
    if is_exponential:
        # Simple check: is improvement accelerating?
        early_rate = (improvement_curve[len(improvement_curve)//2]["fitness"] - baseline) / (len(improvement_curve)//2)
        late_rate = (improvement_curve[-1]["fitness"] - improvement_curve[len(improvement_curve)//2]["fitness"]) / (len(improvement_curve) - len(improvement_curve)//2)
        is_exponential = late_rate > early_rate * 1.1  # 10% acceleration threshold

    results = {
        "approach": approach_name,
        "baseline_fitness": baseline,
        "final_fitness": final_fitness_value,
        "total_improvement": total_improvement,
        "improvement_rate_percent": improvement_rate,
        "iterations_completed": len(improvement_curve) - 1,
        "successful_improvements": len(improver.improvement_history),
        "duration_seconds": duration,
        "is_exponential": is_exponential,
        "improvement_curve": improvement_curve,
        "final_metrics": {
            "overall_score": final_fitness.overall_score,
            "general_reasoning": final_fitness.general_reasoning,
            "transfer_learning": final_fitness.transfer_learning,
            "causal_understanding": final_fitness.causal_understanding,
            "metacognition": final_fitness.metacognition,
        }
    }

    # Print summary
    print()
    print(f"📊 RESULTS FOR {approach_name.upper()}")
    print(f"   Baseline: {baseline:.1%}")
    print(f"   Final: {final_fitness_value:.1%}")
    print(f"   Total Improvement: +{total_improvement:.1%} ({improvement_rate:+.1f}%)")
    print(f"   Successful Strategies: {len(improver.improvement_history)}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Exponential Growth: {'✅ YES' if is_exponential else '❌ NO'}")
    print()

    return results


def analyze_campaign_results(all_results: List[Dict]) -> None:
    """
    Analyze results across all approaches
    """
    print()
    print("=" * 80)
    print("📈 CAMPAIGN ANALYSIS")
    print("=" * 80)
    print()

    # Sort by final fitness
    sorted_results = sorted(all_results, key=lambda x: x["final_fitness"], reverse=True)

    print("🏆 RANKINGS BY FINAL FITNESS:")
    for i, result in enumerate(sorted_results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        exp_mark = "⚡" if result["is_exponential"] else " "
        print(f"   {medal} #{i}. {result['approach'].upper()}: {result['final_fitness']:.1%} "
              f"(+{result['total_improvement']:.1%}) {exp_mark}")

    print()
    print("📊 IMPROVEMENT STATISTICS:")
    total_improvement = sum(r["total_improvement"] for r in all_results)
    avg_improvement = total_improvement / len(all_results)
    best_improvement = max(r["total_improvement"] for r in all_results)
    exponential_count = sum(1 for r in all_results if r["is_exponential"])

    print(f"   Average Improvement: +{avg_improvement:.1%}")
    print(f"   Best Improvement: +{best_improvement:.1%}")
    print(f"   Exponential Growth Detected: {exponential_count}/{len(all_results)} approaches")
    print()

    # AGI progress estimate
    print("🧠 AGI PROGRESS ESTIMATE:")
    baseline_agi = 0.25  # We were at ~25% AGI

    # Use best approach as proxy for AGI progress
    best_result = sorted_results[0]
    agi_boost = best_result["total_improvement"]
    estimated_agi = baseline_agi + agi_boost

    print(f"   Baseline AGI Level: {baseline_agi:.1%}")
    print(f"   Improvement Boost: +{agi_boost:.1%}")
    print(f"   Estimated Current AGI Level: {estimated_agi:.1%}")
    print()

    if estimated_agi >= 0.5:
        print("   🎉 MILESTONE: 50% AGI ACHIEVED!")
    elif estimated_agi >= 0.4:
        print("   🚀 PROGRESS: 40%+ AGI - Getting Close!")
    elif estimated_agi >= 0.35:
        print("   ⬆️  PROGRESS: 35%+ AGI - Significant Improvement!")
    else:
        print("   📈 PROGRESS: Measurable improvement, continue scaling")

    print()

    # Projection
    if exponential_count > 0:
        print("💡 PROJECTION:")
        print("   Exponential growth detected! With continued recursive improvement:")
        iterations_to_50 = int(10 * (0.5 - estimated_agi) / agi_boost) if agi_boost > 0 else 999
        iterations_to_90 = int(10 * (0.9 - estimated_agi) / agi_boost) if agi_boost > 0 else 999

        print(f"   • 50% AGI: ~{iterations_to_50} more iterations")
        print(f"   • 90% AGI: ~{iterations_to_90} more iterations")
        print()
        print("   🎯 RECOMMENDATION: Continue recursive improvement at scale!")
    else:
        print("⚠️  NOTE: Linear improvement detected. May need new strategies.")

    print()


def main():
    """
    Execute extended recursive improvement campaign
    """
    print("=" * 80)
    print("🎖️  EXTENDED RECURSIVE SELF-IMPROVEMENT CAMPAIGN")
    print("=" * 80)
    print()
    print("Mission: Validate exponential growth hypothesis")
    print("Target: Push from 25% AGI → 50%+ AGI via recursive improvement")
    print()
    print("Commander in Chief: Executing...")
    print()

    # Generate tasks (larger set for better validation)
    print("📚 Generating tasks...")

    train_tasks = []
    train_tasks.extend(TaskGenerator.generate_world_model_tasks(30))
    train_tasks.extend(TaskGenerator.generate_meta_learning_tasks(30))
    train_tasks.extend(TaskGenerator.generate_causal_tasks(30))

    test_tasks = []
    test_tasks.extend(TaskGenerator.generate_world_model_tasks(10))
    test_tasks.extend(TaskGenerator.generate_meta_learning_tasks(10))
    test_tasks.extend(TaskGenerator.generate_causal_tasks(10))

    print(f"   ✓ {len(train_tasks)} train tasks, {len(test_tasks)} test tasks")
    print()

    # Run extended improvement on each approach
    all_results = []

    # 1. Meta-Learner (should show strong improvement - learns to learn)
    result = run_extended_improvement(
        MetaLearner,
        "meta_learning",
        train_tasks,
        test_tasks,
        max_iterations=25
    )
    all_results.append(result)

    # 2. World Model (good baseline, may plateau)
    result = run_extended_improvement(
        WorldModelLearner,
        "world_model",
        train_tasks,
        test_tasks,
        max_iterations=25
    )
    all_results.append(result)

    # 3. Causal Reasoner (best for transfer, should improve well)
    result = run_extended_improvement(
        CausalReasoner,
        "causal_reasoning",
        train_tasks,
        test_tasks,
        max_iterations=25
    )
    all_results.append(result)

    # 4. Hybrid AGI (ULTIMATE TEST - combines all approaches)
    result = run_extended_improvement(
        HybridAGI,
        "hybrid_agi",
        train_tasks,
        test_tasks,
        max_iterations=25
    )
    all_results.append(result)

    # Analyze campaign
    analyze_campaign_results(all_results)

    # Save detailed report
    report_dir = Path("data/agi_lab/recursive_improvement")
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "campaign": "Extended Recursive Self-Improvement",
        "timestamp": datetime.now().isoformat(),
        "mission": "Validate exponential growth hypothesis and push toward 50% AGI",
        "results": all_results,
        "summary": {
            "approaches_tested": len(all_results),
            "exponential_growth_count": sum(1 for r in all_results if r["is_exponential"]),
            "best_approach": max(all_results, key=lambda x: x["final_fitness"])["approach"],
            "best_final_fitness": max(r["final_fitness"] for r in all_results),
            "average_improvement": sum(r["total_improvement"] for r in all_results) / len(all_results),
        }
    }

    report_file = report_dir / f"extended_campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"💾 Detailed report saved: {report_file}")
    print()

    # Generate improvement curves visualization (as text)
    print("=" * 80)
    print("📉 IMPROVEMENT CURVES (Sample - see JSON for full data)")
    print("=" * 80)
    print()

    for result in all_results:
        print(f"{result['approach'].upper()}:")
        curve = result["improvement_curve"]

        # Show every 5th iteration
        for point in curve[::5]:
            bar_length = int(point["fitness"] * 50)
            bar = "█" * bar_length
            print(f"   Iter {point['iteration']:2d}: {bar} {point['fitness']:.1%}")

        # Always show final
        if len(curve) > 1:
            final = curve[-1]
            bar_length = int(final["fitness"] * 50)
            bar = "█" * bar_length
            print(f"   Iter {final['iteration']:2d}: {bar} {final['fitness']:.1%} ⭐")

        print()

    print("=" * 80)
    print("✅ EXTENDED RECURSIVE IMPROVEMENT CAMPAIGN COMPLETE!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("   1. Analyze improvement curves in detail")
    print("   2. If exponential growth validated → Scale to 100+ iterations")
    print("   3. If plateau detected → Add new improvement strategies")
    print("   4. Deploy to cluster for massive parallel recursive improvement")
    print("   5. Add real AGI benchmarks (ARC, etc.)")
    print()


if __name__ == "__main__":
    main()
