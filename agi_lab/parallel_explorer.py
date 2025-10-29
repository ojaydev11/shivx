"""
Parallel AGI Explorer
Runs multiple AGI approaches simultaneously and selects best outcomes
Brain-inspired: try many possibilities in parallel, keep what works
"""
import concurrent.futures
import time
from typing import Any, Dict, List, Optional
import numpy as np
from pathlib import Path
import json

from .schemas import (
    AGIApproachType,
    ExperimentResult,
    ExplorationSession,
    CrossPollinationResult,
    AGIFitnessMetrics,
)
from .pattern_recorder import PatternRecorder
from .approaches import (
    BaseAGIApproach,
    WorldModelLearner,
    MetaLearner,
    CausalReasoner,
)


class ParallelExplorer:
    """
    Runs 10-20 AGI approaches in parallel
    Selects best performers
    Combines winning strategies (cross-pollination)
    """

    def __init__(
        self,
        num_parallel: int = 20,
        max_workers: int = 10,
        output_dir: str = "data/agi_lab/experiments"
    ):
        self.num_parallel = num_parallel
        self.max_workers = max_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.recorder = PatternRecorder()
        self.session: Optional[ExplorationSession] = None

    def explore(
        self,
        train_tasks: List[Dict[str, Any]],
        test_tasks: List[Dict[str, Any]],
        transfer_tasks: Optional[List[Dict[str, Any]]] = None,
        max_generations: int = 5
    ) -> ExplorationSession:
        """
        Run parallel exploration across multiple generations

        Args:
            train_tasks: Tasks for training
            test_tasks: Tasks for evaluation
            transfer_tasks: Tasks for transfer learning test
            max_generations: Number of evolutionary generations

        Returns:
            Exploration session with all results
        """
        print(f"🧠 Starting parallel AGI exploration")
        print(f"   Parallel approaches: {self.num_parallel}")
        print(f"   Generations: {max_generations}")
        print(f"   Train tasks: {len(train_tasks)}")
        print(f"   Test tasks: {len(test_tasks)}")

        self.session = ExplorationSession(
            num_parallel=self.num_parallel,
            max_generations=max_generations,
        )

        for generation in range(max_generations):
            print(f"\n🔬 Generation {generation + 1}/{max_generations}")

            # Generate approaches for this generation
            approaches = self._generate_approaches(generation)

            # Run all in parallel
            results = self._run_parallel(
                approaches,
                train_tasks,
                test_tasks,
                transfer_tasks
            )

            # Store results
            self.session.all_results.extend(results)
            self.session.current_generation = generation

            # Analyze and select best
            best_results = self._select_best(results, k=5)

            print(f"\n📊 Generation {generation + 1} Results:")
            for i, result in enumerate(best_results[:3], 1):
                fitness = self._compute_fitness(result)
                print(f"   {i}. {result.approach_type.value}: "
                      f"fitness={fitness.overall_score:.3f} "
                      f"reasoning={fitness.general_reasoning:.3f} "
                      f"transfer={fitness.transfer_learning:.3f}")

            # Check convergence
            if best_results:
                best_fitness = self._compute_fitness(best_results[0]).overall_score
                self.session.best_fitness = best_fitness

                if best_fitness > 0.85:  # Convergence threshold
                    print(f"\n✅ Convergence reached! Fitness: {best_fitness:.3f}")
                    self.session.convergence_reached = True
                    break

            # Cross-pollinate for next generation
            if generation < max_generations - 1:
                self._cross_pollinate(best_results)

        self.session.end_time = datetime.now()

        # Save session
        self._save_session()

        print(f"\n🎉 Exploration complete!")
        print(f"   Total experiments: {len(self.session.all_results)}")
        print(f"   Best fitness: {self.session.best_fitness:.3f}")

        return self.session

    def _generate_approaches(self, generation: int) -> List[BaseAGIApproach]:
        """Generate AGI approaches for this generation"""
        approaches = []

        # Base approaches (always include)
        base_configs = [
            (AGIApproachType.WORLD_MODEL, {}),
            (AGIApproachType.META_LEARNING, {"init_lr": 0.1}),
            (AGIApproachType.CAUSAL, {}),
        ]

        # Add variations
        for approach_type, base_config in base_configs:
            # Multiple configurations per approach
            for i in range(self.num_parallel // len(base_configs)):
                config = base_config.copy()

                # Random variations
                if approach_type == AGIApproachType.META_LEARNING:
                    config["init_lr"] = np.random.uniform(0.01, 0.5)
                    config["init_exploration"] = np.random.uniform(0.1, 0.9)

                # Create approach instance
                if approach_type == AGIApproachType.WORLD_MODEL:
                    approach = WorldModelLearner(config=config, pattern_recorder=self.recorder)
                elif approach_type == AGIApproachType.META_LEARNING:
                    approach = MetaLearner(config=config, pattern_recorder=self.recorder)
                elif approach_type == AGIApproachType.CAUSAL:
                    approach = CausalReasoner(config=config, pattern_recorder=self.recorder)
                else:
                    continue

                approaches.append(approach)

        return approaches[:self.num_parallel]

    def _run_parallel(
        self,
        approaches: List[BaseAGIApproach],
        train_tasks: List[Dict[str, Any]],
        test_tasks: List[Dict[str, Any]],
        transfer_tasks: Optional[List[Dict[str, Any]]]
    ) -> List[ExperimentResult]:
        """Run all approaches in parallel"""
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all experiments
            futures = []
            for approach in approaches:
                future = executor.submit(
                    approach.run_experiment,
                    train_tasks,
                    test_tasks,
                    transfer_tasks
                )
                futures.append((future, approach))

            # Collect results
            for future, approach in futures:
                try:
                    result = future.result(timeout=60)  # 60s timeout per experiment
                    results.append(result)
                    print(f"   ✓ {approach.approach_type.value}: "
                          f"success={result.task_success_rate:.2f}")
                except Exception as e:
                    print(f"   ✗ {approach.approach_type.value}: {e}")

        return results

    def _select_best(self, results: List[ExperimentResult], k: int = 5) -> List[ExperimentResult]:
        """Select top-k results by fitness"""
        # Compute fitness for each
        scored = []
        for result in results:
            fitness = self._compute_fitness(result)
            scored.append((fitness.overall_score, result))

        # Sort by fitness
        scored.sort(reverse=True, key=lambda x: x[0])

        return [result for _, result in scored[:k]]

    def _compute_fitness(self, result: ExperimentResult) -> AGIFitnessMetrics:
        """Compute AGI fitness metrics from experiment result"""
        fitness = AGIFitnessMetrics(
            overall_score=0.0,
            general_reasoning=result.task_success_rate,
            transfer_learning=result.transfer_learning_score,
            causal_understanding=0.5,  # Placeholder
            abstraction=result.generalization_score,
            creativity=result.novelty_score,
            metacognition=0.5,  # Placeholder
            sample_efficiency=result.efficiency,
            robustness=1.0 / (1.0 + result.training_time_sec / 10.0),
            interpretability=0.5,  # Placeholder
        )

        fitness.compute_overall()
        return fitness

    def _cross_pollinate(self, best_results: List[ExperimentResult]) -> None:
        """Combine strategies from best approaches"""
        if len(best_results) < 2:
            return

        print(f"\n🧬 Cross-pollinating best {len(best_results)} approaches...")

        # Extract best patterns from each
        all_best_patterns = []
        for result in best_results:
            # Get top patterns
            top_patterns = sorted(
                result.patterns,
                key=lambda p: p.success_score,
                reverse=True
            )[:10]
            all_best_patterns.extend(top_patterns)

        print(f"   Collected {len(all_best_patterns)} best patterns for next generation")

        # These patterns will influence next generation
        # (In practice, would modify approach configs based on patterns)

    def _save_session(self) -> None:
        """Save exploration session"""
        if not self.session:
            return

        session_file = self.output_dir / f"session_{self.session.session_id}.json"

        # Convert to JSON-serializable format
        data = {
            "session_id": self.session.session_id,
            "num_parallel": self.session.num_parallel,
            "max_generations": self.session.max_generations,
            "current_generation": self.session.current_generation,
            "best_fitness": self.session.best_fitness,
            "convergence_reached": self.session.convergence_reached,
            "num_results": len(self.session.all_results),
        }

        with open(session_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n💾 Session saved: {session_file}")

    def get_best_approach(self) -> Optional[ExperimentResult]:
        """Get single best approach from session"""
        if not self.session or not self.session.all_results:
            return None

        best = self._select_best(self.session.all_results, k=1)
        return best[0] if best else None


from datetime import datetime  # Import at module level for proper access
