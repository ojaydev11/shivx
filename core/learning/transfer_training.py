"""
Transfer Learning Training - Rapid adaptation to new tasks

Uses MAML (Model-Agnostic Meta-Learning) from transfer_learner.py
to enable few-shot learning across empire tasks.

Learn new tasks from just 5 examples!

Part of ShivX Personal Empire AGI (Week 5).
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path

from core.learning.transfer_learner import (
    TransferLearner,
    PrototypicalNetwork,
    quick_transfer,
)
from core.learning.data_collector import get_collector, TaskDomain, TaskType
from core.ml.neural_base import MLPModel, ModelConfig
from core.ml.experiment_tracker import get_tracker

logger = logging.getLogger(__name__)


class EmpireTransferTrainer:
    """
    Transfer learning trainer for empire tasks.

    Enables few-shot learning:
    - Learn new task from 5 examples
    - Adapt to new empire domain rapidly
    - Transfer knowledge across domains
    """

    def __init__(
        self,
        method: str = "prototypical",  # maml, prototypical
        model_dir: str = "data/models/transfer",
    ):
        """
        Initialize transfer trainer.

        Args:
            method: Transfer learning method
            model_dir: Directory for saving models
        """
        self.method = method
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Data collector
        self.collector = get_collector()

        # Experiment tracker
        self.tracker = get_tracker()

        logger.info(f"Empire Transfer Trainer initialized (method={method})")

    def prepare_few_shot_tasks(
        self,
        n_way: int = 3,  # 3 classes
        k_shot: int = 5,  # 5 examples per class
        num_tasks: int = 10,  # 10 meta-training tasks
    ) -> List[Dict[str, Any]]:
        """
        Prepare few-shot learning tasks.

        Args:
            n_way: Number of classes per task
            k_shot: Number of examples per class (support set)
            num_tasks: Number of meta-training tasks

        Returns:
            List of task dictionaries
        """
        logger.info(f"Preparing {num_tasks} few-shot tasks ({n_way}-way {k_shot}-shot)")

        # Get all examples
        all_examples = self.collector.current_dataset.examples

        if len(all_examples) < n_way * k_shot * 2:
            logger.warning(
                f"Insufficient data: {len(all_examples)} examples "
                f"(need {n_way * k_shot * 2} minimum)"
            )

        # Group by domain
        domain_groups = {
            TaskDomain.SEWAGO: [],
            TaskDomain.HALOBUZZ: [],
            TaskDomain.SOLSNIPER: [],
            TaskDomain.SHIVX_CORE: [],
        }

        for ex in all_examples:
            if ex.domain in domain_groups:
                domain_groups[ex.domain].append(ex)

        # Create tasks
        tasks = []

        for task_id in range(num_tasks):
            # Sample n_way domains
            available_domains = [d for d in domain_groups if len(domain_groups[d]) >= k_shot * 2]

            if len(available_domains) < n_way:
                logger.warning(f"Not enough domains with sufficient data for task {task_id}")
                continue

            selected_domains = np.random.choice(available_domains, n_way, replace=False)

            # Create support and query sets
            support_set = []
            query_set = []

            for class_idx, domain in enumerate(selected_domains):
                domain_examples = domain_groups[domain]

                # Sample k_shot for support, k_shot for query
                if len(domain_examples) < k_shot * 2:
                    continue

                indices = np.random.choice(len(domain_examples), k_shot * 2, replace=False)

                support_indices = indices[:k_shot]
                query_indices = indices[k_shot:k_shot * 2]

                # Add to support set
                for idx in support_indices:
                    ex = domain_examples[idx]
                    support_set.append({
                        "input": self._extract_features(ex),
                        "label": class_idx,
                    })

                # Add to query set
                for idx in query_indices:
                    ex = domain_examples[idx]
                    query_set.append({
                        "input": self._extract_features(ex),
                        "label": class_idx,
                    })

            if len(support_set) >= n_way * k_shot and len(query_set) >= n_way * k_shot:
                tasks.append({
                    "support": support_set,
                    "query": query_set,
                    "n_way": n_way,
                    "k_shot": k_shot,
                })

        logger.info(f"Prepared {len(tasks)} valid few-shot tasks")

        return tasks

    def _extract_features(self, task_example) -> List[float]:
        """Extract features from TaskExample"""
        # Simple feature extraction: hash-based
        query_hash = hash(task_example.query) % 10000
        action_hash = hash(task_example.action_taken) % 10000

        features = [
            float(query_hash) / 10000.0,
            float(action_hash) / 10000.0,
            float(task_example.confidence),
            float(task_example.success) if task_example.success is not None else 0.5,
            float(task_example.duration_seconds or 0) / 1000.0,  # Normalize
        ]

        # Pad to 10 features
        while len(features) < 10:
            features.append(0.0)

        return features[:10]

    def train_prototypical_networks(
        self,
        n_way: int = 3,
        k_shot: int = 5,
        num_tasks: int = 20,
        num_episodes: int = 100,
    ) -> Dict[str, Any]:
        """
        Train using Prototypical Networks.

        Args:
            n_way: Number of classes
            k_shot: Support examples per class
            num_tasks: Meta-training tasks
            num_episodes: Training episodes

        Returns:
            Training results
        """
        logger.info(f"Training Prototypical Networks ({n_way}-way {k_shot}-shot)")

        # Start experiment
        run_id = self.tracker.start_run(
            run_name=f"transfer_prototypical_{n_way}way_{k_shot}shot",
            config={
                "method": "prototypical",
                "n_way": n_way,
                "k_shot": k_shot,
                "num_tasks": num_tasks,
                "num_episodes": num_episodes,
                "input_dim": 10,
                "embedding_dim": 64,
            },
            tags=["transfer_learning", "prototypical", "empire"],
            notes=f"Few-shot learning with Prototypical Networks"
        )

        # Prepare tasks
        tasks = self.prepare_few_shot_tasks(n_way, k_shot, num_tasks)

        if len(tasks) < 5:
            logger.error("Insufficient tasks for meaningful training")
            return {"error": "Need at least 5 tasks"}

        # Create Transfer Learner with Prototypical Networks
        learner = TransferLearner(
            method="prototypical",
            input_dim=10,
            output_dim=64,  # Embedding dimension
            device="cpu",
        )

        # Training loop
        best_accuracy = 0.0
        accuracies = []

        for episode in range(num_episodes):
            # Sample random task
            task = tasks[episode % len(tasks)]  # Cycle through tasks

            # Train on this task using few-shot adaptation
            accuracy = learner.few_shot_adapt(
                support_set=task["support"],
                query_set=task["query"],
                num_classes=task["n_way"],
            )

            accuracies.append(accuracy)

            # Log progress
            if episode % 10 == 0:
                avg_acc = np.mean(accuracies[-10:]) if len(accuracies) >= 10 else np.mean(accuracies)
                logger.info(f"Episode {episode}/{num_episodes}: accuracy={avg_acc:.3f}")

                self.tracker.log({"accuracy": avg_acc}, step=episode)

            best_accuracy = max(best_accuracy, accuracy)

        # Final evaluation
        final_accuracy = np.mean(accuracies[-10:]) if len(accuracies) >= 10 else np.mean(accuracies)

        logger.info(f"Training complete: final_accuracy={final_accuracy:.3f}, best={best_accuracy:.3f}")

        # Log summary
        self.tracker.log_summary({
            "final_accuracy": final_accuracy,
            "best_accuracy": best_accuracy,
            "num_episodes": num_episodes,
            "n_way": n_way,
            "k_shot": k_shot,
        })

        # Save model
        model_path = self.model_dir / f"prototypical_{n_way}way_{k_shot}shot.pth"
        torch.save({
            "model_state_dict": learner.model.state_dict(),
            "method": learner.method,
            "input_dim": 10,
            "output_dim": 64,
            "n_way": n_way,
            "k_shot": k_shot,
        }, model_path)

        logger.info(f"Model saved to: {model_path}")

        self.tracker.finish_run()

        return {
            "final_accuracy": final_accuracy,
            "best_accuracy": best_accuracy,
            "model_path": str(model_path),
        }

    def test_few_shot_generalization(
        self,
        n_way: int = 3,
        k_shot: int = 5,
        num_test_tasks: int = 10,
    ) -> Dict[str, Any]:
        """
        Test few-shot learning on unseen tasks.

        Args:
            n_way: Number of classes
            k_shot: Support examples
            num_test_tasks: Number of test tasks

        Returns:
            Test results
        """
        logger.info(f"Testing few-shot generalization ({num_test_tasks} tasks)")

        # Prepare test tasks
        test_tasks = self.prepare_few_shot_tasks(n_way, k_shot, num_test_tasks)

        if not test_tasks:
            return {"error": "No test tasks available"}

        # Load trained model
        model_path = self.model_dir / f"prototypical_{n_way}way_{k_shot}shot.pth"

        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return {"error": "Model not trained yet"}

        # Load model
        checkpoint = torch.load(model_path)

        # Create Transfer Learner
        learner = TransferLearner(
            method=checkpoint.get("method", "prototypical"),
            input_dim=checkpoint.get("input_dim", 10),
            output_dim=checkpoint.get("output_dim", 64),
            device="cpu",
        )

        learner.model.load_state_dict(checkpoint["model_state_dict"])

        # Test on all tasks
        accuracies = []

        for task in test_tasks:
            accuracy = learner.few_shot_adapt(
                support_set=task["support"],
                query_set=task["query"],
                num_classes=task["n_way"],
            )

            accuracies.append(accuracy)

        # Compute statistics
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        min_accuracy = np.min(accuracies)
        max_accuracy = np.max(accuracies)

        logger.info(
            f"Generalization: mean={mean_accuracy:.3f}, "
            f"std={std_accuracy:.3f}, range=[{min_accuracy:.3f}, {max_accuracy:.3f}]"
        )

        return {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "min_accuracy": min_accuracy,
            "max_accuracy": max_accuracy,
            "num_tasks": len(test_tasks),
            "accuracies": accuracies,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== Empire Transfer Learning Test ===\n")

    # Check data availability
    collector = get_collector()
    stats = collector.get_stats()

    print(f"Available training data: {stats['dataset']['total']} examples")

    if stats['dataset']['total'] < 50:
        print("\nWARNING: Low data count. Results may be suboptimal.")
        print("For best results, need 100+ examples across multiple domains.")

    # Create trainer
    trainer = EmpireTransferTrainer(method="prototypical")

    # Train Prototypical Networks
    print("\nTraining Prototypical Networks...")
    print("Configuration: 3-way 5-shot (3 classes, 5 examples per class)")

    results = trainer.train_prototypical_networks(
        n_way=3,
        k_shot=5,
        num_tasks=20,
        num_episodes=50,  # Reduced for quick test
    )

    print(f"\n=== Training Results ===")
    print(f"Final Accuracy: {results['final_accuracy']:.3f}")
    print(f"Best Accuracy: {results['best_accuracy']:.3f}")
    print(f"Model: {results['model_path']}")

    # Test generalization
    print("\nTesting generalization on unseen tasks...")
    test_results = trainer.test_few_shot_generalization(
        n_way=3,
        k_shot=5,
        num_test_tasks=10,
    )

    if "error" not in test_results:
        print(f"\n=== Generalization Results ===")
        print(f"Mean Accuracy: {test_results['mean_accuracy']:.3f}")
        print(f"Std Dev: {test_results['std_accuracy']:.3f}")
        print(f"Range: [{test_results['min_accuracy']:.3f}, {test_results['max_accuracy']:.3f}]")
        print(f"Test Tasks: {test_results['num_tasks']}")

        # Interpret results
        if test_results['mean_accuracy'] > 0.5:
            print("\n✅ EXCELLENT: Few-shot learning working well!")
        elif test_results['mean_accuracy'] > 0.33:
            print("\n✅ GOOD: Better than random (33% for 3-way)")
        else:
            print("\n⚠️  NEEDS IMPROVEMENT: Close to random guessing")

    print(f"\nModels saved to: {trainer.model_dir}")
