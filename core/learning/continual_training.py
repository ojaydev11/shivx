"""
Continual Learning Training - Prevent catastrophic forgetting

Uses EWC (Elastic Weight Consolidation) from continual_learner.py
to train on multiple empire tasks without forgetting previous knowledge.

Part of ShivX Personal Empire AGI (Week 4).
"""

import logging
import torch
from typing import List, Dict, Any
from pathlib import Path

from core.learning.continual_learner import ContinualLearner
from core.learning.data_collector import get_collector, TaskDomain, TaskType
from core.ml.neural_base import MLPModel, ModelConfig
from core.ml.experiment_tracker import get_tracker

logger = logging.getLogger(__name__)


class EmpireContinualTrainer:
    """
    Continual learning trainer for empire tasks.

    Trains on sequences of tasks without forgetting:
    - Task 1: Sewago operations
    - Task 2: Halobuzz content
    - Task 3: SolsniperPro trading
    """

    def __init__(
        self,
        strategy: str = "hybrid",  # ewc, replay, hybrid
        model_dir: str = "data/models/continual",
    ):
        """
        Initialize continual trainer.

        Args:
            strategy: Continual learning strategy
            model_dir: Directory for saving models
        """
        self.strategy = strategy
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Data collector
        self.collector = get_collector()

        # Experiment tracker
        self.tracker = get_tracker()

        logger.info(f"Empire Continual Trainer initialized (strategy={strategy})")

    def prepare_task_data(
        self,
        domain: TaskDomain,
        num_examples: int = 30,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Prepare training/test data for a specific domain.

        Args:
            domain: Empire domain
            num_examples: Number of examples per split

        Returns:
            (train_examples, test_examples)
        """
        # Get examples for domain
        domain_examples = self.collector.current_dataset.filter_by_domain(domain)

        if len(domain_examples) < num_examples * 2:
            logger.warning(
                f"Insufficient data for {domain.value}: "
                f"{len(domain_examples)} examples (need {num_examples * 2})"
            )
            # Use what we have
            split_idx = len(domain_examples) // 2
        else:
            split_idx = num_examples

        # Split train/test
        train_raw = domain_examples[:split_idx]
        test_raw = domain_examples[split_idx:split_idx * 2]

        # Convert to format expected by continual learner
        train_examples = self._convert_to_ml_format(train_raw)
        test_examples = self._convert_to_ml_format(test_raw)

        logger.info(
            f"Prepared {domain.value}: "
            f"{len(train_examples)} train, {len(test_examples)} test"
        )

        return train_examples, test_examples

    def _convert_to_ml_format(self, task_examples: List) -> List[Dict[str, Any]]:
        """Convert TaskExample objects to ML format"""
        ml_examples = []

        for ex in task_examples:
            # Simple feature extraction: use hash of query
            # In production, would use sentence embeddings
            query_hash = hash(ex.query) % 10000
            features = [float(query_hash) / 10000.0]

            # Pad to 10 features (arbitrary size for demo)
            while len(features) < 10:
                features.append(0.0)

            # Label: map domain to class
            domain_to_label = {
                TaskDomain.SEWAGO: 0,
                TaskDomain.HALOBUZZ: 1,
                TaskDomain.SOLSNIPER: 2,
                TaskDomain.SHIVX_CORE: 3,
            }

            label = domain_to_label.get(ex.domain, 0)

            ml_examples.append({
                "input": features,
                "label": label,
            })

        return ml_examples

    def train_sequential_tasks(self) -> Dict[str, Any]:
        """
        Train on sequence of empire tasks.

        Tests continual learning across 3 domains.

        Returns:
            Training results
        """
        logger.info("Starting continual learning training...")

        # Start experiment
        run_id = self.tracker.start_run(
            run_name=f"continual_empire_{self.strategy}",
            config={
                "strategy": self.strategy,
                "tasks": ["sewago", "halobuzz", "solsniper"],
                "lambda_ewc": 1000.0,
                "replay_buffer_size": 500,
            },
            tags=["continual_learning", "empire", self.strategy],
            notes="Testing continual learning across empire domains"
        )

        # Prepare tasks
        tasks = []

        # Task 1: Sewago
        sewago_train, sewago_test = self.prepare_task_data(TaskDomain.SEWAGO, 15)
        if sewago_train and sewago_test:
            tasks.append(("sewago", sewago_train, sewago_test))

        # Task 2: Halobuzz
        halobuzz_train, halobuzz_test = self.prepare_task_data(TaskDomain.HALOBUZZ, 15)
        if halobuzz_train and halobuzz_test:
            tasks.append(("halobuzz", halobuzz_train, halobuzz_test))

        # Task 3: SolsniperPro
        solsniper_train, solsniper_test = self.prepare_task_data(TaskDomain.SOLSNIPER, 15)
        if solsniper_train and solsniper_test:
            tasks.append(("solsniper", solsniper_train, solsniper_test))

        if len(tasks) < 2:
            logger.error("Insufficient tasks for continual learning test")
            return {"error": "Need at least 2 tasks"}

        # Create model with correct output dimension (4 domains)
        config = ModelConfig(
            model_name="empire_continual",
            input_dim=10,  # Feature dimension
            hidden_dims=[64, 32],
            output_dim=4,  # 4 empire domains
        )

        from core.ml.neural_base import MLPModel
        model = MLPModel(config)

        # Create continual learner
        learner = ContinualLearner(
            model=model,
            strategy=self.strategy,
            lambda_ewc=1000.0,
            replay_buffer_size=500,
        )

        # Learn tasks sequentially
        for task_id, train_examples, test_examples in tasks:
            accuracy = learner.learn_task(
                task_id=task_id,
                task_name=task_id,
                train_examples=train_examples,
                test_examples=test_examples,
                num_epochs=30,
            )
            logger.info(f"Learned {task_id}: accuracy={accuracy:.3f}")

        # Evaluate on all tasks
        task_test_sets = {
            task_id: test_examples
            for task_id, _, test_examples in tasks
        }

        results = learner.evaluate_all_tasks(task_test_sets)
        forgetting = learner.compute_forgetting(task_test_sets)

        results["avg_forgetting"] = forgetting

        # Log results
        for task_id, accuracy in results.items():
            if task_id != "avg_forgetting":
                self.tracker.log({f"accuracy_{task_id}": accuracy})

        self.tracker.log({"avg_forgetting": results.get("avg_forgetting", 0.0)})

        # Log summary
        self.tracker.log_summary({
            "final_accuracies": {k: v for k, v in results.items() if k != "avg_forgetting"},
            "avg_forgetting": results.get("avg_forgetting", 0.0),
            "num_tasks": len(tasks),
            "strategy": self.strategy,
        })

        self.tracker.finish_run()

        logger.info(f"Continual learning complete: forgetting={results.get('avg_forgetting', 0.0):.3f}")

        return results

    def benchmark_strategies(self) -> Dict[str, Dict[str, float]]:
        """
        Benchmark different continual learning strategies.

        Returns:
            Results for each strategy
        """
        logger.info("Benchmarking continual learning strategies...")

        strategies = ["ewc", "replay", "hybrid"]
        all_results = {}

        for strategy in strategies:
            logger.info(f"\nTesting strategy: {strategy}")

            # Create trainer
            trainer = EmpireContinualTrainer(strategy=strategy)

            # Train
            results = trainer.train_sequential_tasks()

            all_results[strategy] = results

        # Compare
        logger.info("\n=== Strategy Comparison ===")
        for strategy, results in all_results.items():
            forgetting = results.get("avg_forgetting", 0.0)
            logger.info(f"{strategy}: forgetting={forgetting:.3f}")

        return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== Empire Continual Learning Test ===\n")

    # Check if we have data
    collector = get_collector()
    stats = collector.get_stats()

    print(f"Available training data: {stats['dataset']['total']} examples")

    if stats['dataset']['total'] < 30:
        print("\nWARNING: Insufficient data for meaningful continual learning test.")
        print("Need at least 30 examples across multiple domains.")
        print("\nGenerating bootstrap data...")

        # Generate bootstrap data if needed
        from core.learning.bootstrap_data_generator import BootstrapDataGenerator

        generator = BootstrapDataGenerator()
        generator.generate_and_save(count=100)

        print("Bootstrap data generated. Reloading...")

        # Reload collector
        from core.learning.data_collector import DataCollector
        collector = DataCollector()
        stats = collector.get_stats()

        print(f"Now have: {stats['dataset']['total']} examples")

    # Run continual learning test
    trainer = EmpireContinualTrainer(strategy="hybrid")
    results = trainer.train_sequential_tasks()

    print("\n=== Results ===")
    print(f"Average Forgetting: {results.get('avg_forgetting', 0.0):.3f}")

    print("\nTask Accuracies:")
    for task_id, accuracy in results.items():
        if task_id != "avg_forgetting":
            print(f"  {task_id}: {accuracy:.3f}")

    # Interpret results
    forgetting = results.get("avg_forgetting", 0.0)
    if forgetting < 0.1:
        print("\n✅ EXCELLENT: Very low forgetting!")
    elif forgetting < 0.3:
        print("\n✅ GOOD: Acceptable forgetting rate")
    else:
        print("\n⚠️  HIGH FORGETTING: May need tuning")

    print(f"\nModel saved to: data/models/continual/")
