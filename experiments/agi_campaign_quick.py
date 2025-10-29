#!/usr/bin/env python3
"""
Quick AGI Campaign - Fixed version with only working approaches
Tests the 3 fully functional approaches at scale
"""
import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from agi_lab import TaskGenerator, PatternRecorder, AGIApproachType
from agi_lab.parallel_explorer import ParallelExplorer
from agi_lab.approaches import WorldModelLearner, MetaLearner, CausalReasoner


def main():
    """Execute quick AGI campaign with working approaches"""
    print("=" * 80)
    print("🎖️  QUICK AGI RESEARCH CAMPAIGN")
    print("=" * 80)
    print()
    print("Testing 3 proven AGI approaches at scale")
    print()
    print("📍 Parameters:")
    print("   • Approaches: World Model, Meta-Learning, Causal Reasoning")
    print("   • Parallel: 30 (10 each)")
    print("   • Generations: 10")
    print("   • Tasks: 150 train, 50 test, 30 transfer")
    print()

    # Generate tasks
    print("📚 Generating tasks...")
    train_tasks = []
    train_tasks.extend(TaskGenerator.generate_world_model_tasks(50))
    train_tasks.extend(TaskGenerator.generate_meta_learning_tasks(50))
    train_tasks.extend(TaskGenerator.generate_causal_tasks(50))

    test_tasks = []
    test_tasks.extend(TaskGenerator.generate_world_model_tasks(15))
    test_tasks.extend(TaskGenerator.generate_meta_learning_tasks(20))
    test_tasks.extend(TaskGenerator.generate_causal_tasks(15))

    transfer_tasks = TaskGenerator.generate_transfer_tasks(30)

    print(f"   ✓ {len(train_tasks)} train, {len(test_tasks)} test, {len(transfer_tasks)} transfer")
    print()

    # Run exploration
    print("🧠 Running exploration...")
    explorer = ParallelExplorer(num_parallel=30, max_workers=10)

    session = explorer.explore(
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        transfer_tasks=transfer_tasks,
        max_generations=10,
    )

    # Analyze
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print()

    best = explorer.get_best_approach()
    if best:
        print(f"🏆 WINNER: {best.approach_type.value.upper()}")
        print(f"   Success: {best.task_success_rate:.1%}")
        print(f"   Transfer: {best.transfer_learning_score:.1%}")
        print(f"   Patterns: {len(best.patterns)}")
        print()

    # Rankings
    recorder = PatternRecorder()
    rankings = []

    for approach_type in [AGIApproachType.WORLD_MODEL, AGIApproachType.META_LEARNING, AGIApproachType.CAUSAL]:
        analysis = recorder.analyze_approach(approach_type)
        if analysis['total_patterns'] > 0:
            print(f"{approach_type.value.upper()}")
            print(f"   Success: {analysis['avg_success']:.1%}")
            print(f"   Patterns: {analysis['total_patterns']}")
            print()
            rankings.append((approach_type.value, analysis['avg_success']))

    rankings.sort(key=lambda x: x[1], reverse=True)

    print("🥇 RANKINGS:")
    for i, (name, score) in enumerate(rankings, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"   {medal} #{i}. {name.upper()}: {score:.1%}")

    print()
    print(f"📊 Fitness: {session.best_fitness:.3f}")
    print(f"📈 Experiments: {len(session.all_results)}")
    print(f"🧬 Generations: {session.current_generation + 1}")
    print()

    # Save report
    report_path = Path("data/agi_lab/research_reports")
    report_path.mkdir(parents=True, exist_ok=True)

    report = {
        "campaign": "Quick AGI Campaign",
        "timestamp": datetime.now().isoformat(),
        "best_approach": best.approach_type.value if best else None,
        "best_fitness": session.best_fitness,
        "rankings": [{"rank": i+1, "approach": n, "score": s} for i, (n, s) in enumerate(rankings)],
    }

    report_file = report_path / f"quick_campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"💾 Report: {report_file}")
    print()
    print("=" * 80)
    print("✅ CAMPAIGN COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
