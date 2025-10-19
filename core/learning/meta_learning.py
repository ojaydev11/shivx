"""
Meta-Learning System for Empire AGI
Week 16: Learning to Learn

Implements Model-Agnostic Meta-Learning (MAML) and few-shot learning
to enable rapid adaptation to new tasks with minimal data.

Key capabilities:
- MAML: Optimize for fast fine-tuning
- Few-shot learning: Learn from 1-5 examples
- Rapid adaptation: Quick task-specific tuning
- Meta-optimization: Learn optimal hyperparameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
import numpy as np
import copy
import time


class MetaLearningStrategy(Enum):
    """Meta-learning strategies"""
    MAML = "maml"  # Model-Agnostic Meta-Learning
    REPTILE = "reptile"  # Reptile (simpler alternative to MAML)
    FOMAML = "fomaml"  # First-Order MAML (faster)
    METASGD = "metasgd"  # Meta-SGD (learns learning rates)


class AdaptationStrategy(Enum):
    """How to adapt to new tasks"""
    FINE_TUNE = "fine_tune"  # Standard fine-tuning
    FEATURE_REUSE = "feature_reuse"  # Freeze features, train head
    FULL_RETRAIN = "full_retrain"  # Train from scratch
    PROBING = "probing"  # Linear probe only


@dataclass
class Task:
    """A meta-learning task"""
    task_id: str
    name: str
    support_set: List[Tuple[torch.Tensor, torch.Tensor]]  # Few examples for training
    query_set: List[Tuple[torch.Tensor, torch.Tensor]]  # Examples for evaluation
    num_classes: int
    metadata: Dict[str, Any]


@dataclass
class MetaTrainResult:
    """Result of meta-training"""
    meta_loss: float
    task_losses: List[float]
    task_accuracies: List[float]
    adaptation_steps: int
    meta_iteration: int


@dataclass
class AdaptationResult:
    """Result of adapting to a new task"""
    task_id: str
    initial_accuracy: float
    final_accuracy: float
    adaptation_steps: int
    adaptation_time: float
    converged: bool


@dataclass
class MetaOptimizationResult:
    """Result of meta-optimizing hyperparameters"""
    optimal_learning_rate: float
    optimal_num_steps: int
    optimal_batch_size: int
    validation_accuracy: float
    search_iterations: int


class MetaLearner:
    """
    Model-Agnostic Meta-Learning (MAML) implementation.

    MAML optimizes model parameters such that a small number of
    gradient steps on a new task will produce good performance.
    """

    def __init__(
        self,
        model: nn.Module,
        meta_lr: float = 0.001,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        strategy: MetaLearningStrategy = MetaLearningStrategy.MAML,
    ):
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.strategy = strategy

        # Meta-optimizer updates the model parameters
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)

        # Statistics
        self.meta_iterations = 0
        self.total_tasks_seen = 0
        self.adaptation_history: List[AdaptationResult] = []

    async def meta_train_step(
        self,
        tasks: List[Task],
    ) -> MetaTrainResult:
        """
        One step of meta-training across multiple tasks.

        For each task:
        1. Clone current model parameters
        2. Adapt on support set (inner loop)
        3. Evaluate on query set
        4. Compute meta-gradient
        5. Update meta-parameters (outer loop)
        """
        self.meta_optimizer.zero_grad()

        meta_loss = 0.0
        task_losses = []
        task_accuracies = []

        for task in tasks:
            # Inner loop: Adapt to this task
            adapted_model = self._adapt_to_task(task)

            # Evaluate on query set
            query_loss, query_acc = self._evaluate_on_query_set(
                adapted_model,
                task.query_set
            )

            # Accumulate meta-loss
            meta_loss += query_loss
            task_losses.append(query_loss.item())
            task_accuracies.append(query_acc)

            self.total_tasks_seen += 1

        # Average meta-loss
        meta_loss = meta_loss / len(tasks)

        # Outer loop: Update meta-parameters
        if self.strategy in [MetaLearningStrategy.MAML, MetaLearningStrategy.FOMAML]:
            meta_loss.backward()
            self.meta_optimizer.step()
        elif self.strategy == MetaLearningStrategy.REPTILE:
            # Reptile: Move toward adapted parameters
            self._reptile_update(tasks)

        self.meta_iterations += 1

        return MetaTrainResult(
            meta_loss=meta_loss.item(),
            task_losses=task_losses,
            task_accuracies=task_accuracies,
            adaptation_steps=self.num_inner_steps,
            meta_iteration=self.meta_iterations,
        )

    def _adapt_to_task(self, task: Task) -> nn.Module:
        """
        Adapt the model to a specific task using the support set.

        This is the "inner loop" of MAML.
        """
        # Clone model for task-specific adaptation
        adapted_model = copy.deepcopy(self.model)

        # Inner optimizer for adaptation
        inner_optimizer = optim.SGD(
            adapted_model.parameters(),
            lr=self.inner_lr
        )

        # Adapt for num_inner_steps
        for step in range(self.num_inner_steps):
            inner_optimizer.zero_grad()

            # Prepare batch from support set
            X_support = torch.stack([x for x, y in task.support_set])
            y_support = torch.stack([y for x, y in task.support_set])

            # Forward pass
            outputs = adapted_model(X_support)
            loss = F.cross_entropy(outputs, y_support)

            # Backward pass (create_graph=True for MAML second-order gradients)
            if self.strategy == MetaLearningStrategy.MAML:
                loss.backward(create_graph=True)
            else:  # FOMAML, REPTILE
                loss.backward()

            inner_optimizer.step()

        return adapted_model

    def _evaluate_on_query_set(
        self,
        model: nn.Module,
        query_set: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, float]:
        """
        Evaluate adapted model on query set.
        """
        X_query = torch.stack([x for x, y in query_set])
        y_query = torch.stack([y for x, y in query_set])

        # Forward pass
        outputs = model(X_query)
        loss = F.cross_entropy(outputs, y_query)

        # Accuracy
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_query).float().mean().item()

        return loss, accuracy

    def _reptile_update(self, tasks: List[Task]):
        """
        Reptile meta-learning update.

        Simpler than MAML: Just move meta-parameters toward
        adapted parameters.
        """
        # Collect adapted parameters from all tasks
        adapted_params = []
        for task in tasks:
            adapted_model = self._adapt_to_task(task)
            adapted_params.append([p.data.clone() for p in adapted_model.parameters()])

        # Average adapted parameters
        avg_adapted = []
        for i in range(len(list(self.model.parameters()))):
            avg_param = torch.stack([params[i] for params in adapted_params]).mean(dim=0)
            avg_adapted.append(avg_param)

        # Move meta-parameters toward average adapted parameters
        with torch.no_grad():
            for meta_param, avg_param in zip(self.model.parameters(), avg_adapted):
                meta_param.data.add_(
                    avg_param - meta_param.data,
                    alpha=self.meta_lr
                )

    async def adapt(
        self,
        task: Task,
        num_steps: Optional[int] = None,
    ) -> AdaptationResult:
        """
        Adapt meta-learned model to a new task.

        Uses the support set for adaptation, evaluates on query set.
        """
        start_time = time.time()

        if num_steps is None:
            num_steps = self.num_inner_steps

        # Evaluate initial performance
        initial_loss, initial_acc = self._evaluate_on_query_set(
            self.model,
            task.query_set
        )

        # Clone model for adaptation
        adapted_model = copy.deepcopy(self.model)
        optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        # Adapt
        for step in range(num_steps):
            optimizer.zero_grad()

            X_support = torch.stack([x for x, y in task.support_set])
            y_support = torch.stack([y for x, y in task.support_set])

            outputs = adapted_model(X_support)
            loss = F.cross_entropy(outputs, y_support)

            loss.backward()
            optimizer.step()

        # Evaluate final performance
        final_loss, final_acc = self._evaluate_on_query_set(
            adapted_model,
            task.query_set
        )

        adaptation_time = time.time() - start_time

        # Check convergence
        converged = final_acc > 0.8 or (final_acc - initial_acc) > 0.2

        result = AdaptationResult(
            task_id=task.task_id,
            initial_accuracy=initial_acc,
            final_accuracy=final_acc,
            adaptation_steps=num_steps,
            adaptation_time=adaptation_time,
            converged=converged,
        )

        self.adaptation_history.append(result)

        return result


class FewShotLearner:
    """
    Few-shot learning system.

    Learns from just a few examples (1-shot, 5-shot, etc.)
    using meta-learned initialization.
    """

    def __init__(
        self,
        meta_learner: MetaLearner,
        num_shots: int = 5,
        num_ways: int = 2,  # N-way classification
    ):
        self.meta_learner = meta_learner
        self.num_shots = num_shots
        self.num_ways = num_ways

        # Statistics
        self.few_shot_tasks_solved = 0
        self.average_adaptation_steps = 0.0

    async def learn_from_examples(
        self,
        examples: List[Tuple[torch.Tensor, torch.Tensor]],
        task_name: str = "few_shot_task",
    ) -> AdaptationResult:
        """
        Learn from just a few examples.

        Args:
            examples: List of (input, label) pairs
            task_name: Name of the task

        Returns:
            Adaptation result showing how quickly we adapted
        """
        # Split into support (train) and query (test)
        support_size = min(self.num_shots, len(examples) // 2)
        support_set = examples[:support_size]
        query_set = examples[support_size:]

        # Create task
        task = Task(
            task_id=f"few_shot_{self.few_shot_tasks_solved}",
            name=task_name,
            support_set=support_set,
            query_set=query_set,
            num_classes=self.num_ways,
            metadata={"num_shots": support_size},
        )

        # Adapt
        result = await self.meta_learner.adapt(task)

        self.few_shot_tasks_solved += 1
        self.average_adaptation_steps = (
            (self.average_adaptation_steps * (self.few_shot_tasks_solved - 1) +
             result.adaptation_steps) / self.few_shot_tasks_solved
        )

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get few-shot learning statistics"""
        if not self.meta_learner.adaptation_history:
            return {
                "tasks_solved": 0,
                "average_initial_accuracy": 0.0,
                "average_final_accuracy": 0.0,
                "average_improvement": 0.0,
                "average_adaptation_time": 0.0,
            }

        history = self.meta_learner.adaptation_history

        return {
            "tasks_solved": self.few_shot_tasks_solved,
            "average_initial_accuracy": np.mean([r.initial_accuracy for r in history]),
            "average_final_accuracy": np.mean([r.final_accuracy for r in history]),
            "average_improvement": np.mean([
                r.final_accuracy - r.initial_accuracy for r in history
            ]),
            "average_adaptation_time": np.mean([r.adaptation_time for r in history]),
            "convergence_rate": np.mean([r.converged for r in history]),
        }


class TaskSampler:
    """
    Generates meta-learning tasks for training.

    Each task is a mini-classification problem with support and query sets.
    """

    def __init__(
        self,
        num_ways: int = 2,
        num_shots: int = 5,
        num_queries: int = 10,
    ):
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries

    def sample_tasks(
        self,
        num_tasks: int,
        input_dim: int = 10,
    ) -> List[Task]:
        """
        Sample random classification tasks.

        For testing purposes, generates synthetic tasks.
        In production, would sample from real task distributions.
        """
        tasks = []

        for task_idx in range(num_tasks):
            # Generate random task parameters
            task_weights = torch.randn(input_dim, self.num_ways)
            task_bias = torch.randn(self.num_ways)

            # Generate support set (for training)
            support_set = []
            for _ in range(self.num_shots * self.num_ways):
                x = torch.randn(input_dim)
                logits = x @ task_weights + task_bias
                y = torch.argmax(logits)
                support_set.append((x, y))

            # Generate query set (for evaluation)
            query_set = []
            for _ in range(self.num_queries):
                x = torch.randn(input_dim)
                logits = x @ task_weights + task_bias
                y = torch.argmax(logits)
                query_set.append((x, y))

            task = Task(
                task_id=f"task_{task_idx}",
                name=f"Synthetic Task {task_idx}",
                support_set=support_set,
                query_set=query_set,
                num_classes=self.num_ways,
                metadata={"synthetic": True},
            )

            tasks.append(task)

        return tasks


class MetaOptimizer:
    """
    Meta-optimization: Learn optimal hyperparameters.

    Uses meta-learning to find the best learning rate, number of
    adaptation steps, etc. for fast adaptation.
    """

    def __init__(
        self,
        meta_learner: MetaLearner,
        task_sampler: TaskSampler,
    ):
        self.meta_learner = meta_learner
        self.task_sampler = task_sampler

    async def optimize_hyperparameters(
        self,
        search_iterations: int = 10,
        validation_tasks: int = 5,
    ) -> MetaOptimizationResult:
        """
        Search for optimal hyperparameters using meta-validation.

        Tries different hyperparameter settings and evaluates on
        validation tasks.
        """
        # Search space
        learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
        num_steps_options = [1, 3, 5, 10, 20]

        best_accuracy = 0.0
        best_lr = self.meta_learner.inner_lr
        best_steps = self.meta_learner.num_inner_steps

        for lr in learning_rates:
            for num_steps in num_steps_options:
                # Set hyperparameters
                self.meta_learner.inner_lr = lr
                self.meta_learner.num_inner_steps = num_steps

                # Evaluate on validation tasks
                val_tasks = self.task_sampler.sample_tasks(validation_tasks)
                accuracies = []

                for task in val_tasks:
                    result = await self.meta_learner.adapt(task)
                    accuracies.append(result.final_accuracy)

                avg_accuracy = np.mean(accuracies)

                # Update best
                if avg_accuracy > best_accuracy:
                    best_accuracy = avg_accuracy
                    best_lr = lr
                    best_steps = num_steps

        # Set optimal hyperparameters
        self.meta_learner.inner_lr = best_lr
        self.meta_learner.num_inner_steps = best_steps

        return MetaOptimizationResult(
            optimal_learning_rate=best_lr,
            optimal_num_steps=best_steps,
            optimal_batch_size=32,  # Fixed for now
            validation_accuracy=best_accuracy,
            search_iterations=len(learning_rates) * len(num_steps_options),
        )


class MetaLearningSystem:
    """
    Unified meta-learning system.

    Integrates meta-learning, few-shot learning, and meta-optimization.
    """

    def __init__(
        self,
        model: nn.Module,
        meta_lr: float = 0.001,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        num_shots: int = 5,
        num_ways: int = 2,
    ):
        self.meta_learner = MetaLearner(
            model=model,
            meta_lr=meta_lr,
            inner_lr=inner_lr,
            num_inner_steps=num_inner_steps,
        )

        self.few_shot_learner = FewShotLearner(
            meta_learner=self.meta_learner,
            num_shots=num_shots,
            num_ways=num_ways,
        )

        self.task_sampler = TaskSampler(
            num_ways=num_ways,
            num_shots=num_shots,
        )

        self.meta_optimizer = MetaOptimizer(
            meta_learner=self.meta_learner,
            task_sampler=self.task_sampler,
        )

    async def meta_train(
        self,
        num_iterations: int = 100,
        tasks_per_iteration: int = 4,
    ) -> List[MetaTrainResult]:
        """
        Meta-train the model to be good at fast adaptation.
        """
        results = []

        for iteration in range(num_iterations):
            # Sample tasks
            tasks = self.task_sampler.sample_tasks(tasks_per_iteration)

            # Meta-train on these tasks
            result = await self.meta_learner.meta_train_step(tasks)
            results.append(result)

        return results

    async def adapt_to_new_task(
        self,
        examples: List[Tuple[torch.Tensor, torch.Tensor]],
        task_name: str = "new_task",
    ) -> AdaptationResult:
        """
        Adapt to a completely new task with just a few examples.
        """
        return await self.few_shot_learner.learn_from_examples(
            examples,
            task_name=task_name,
        )

    async def optimize_hyperparameters(
        self,
        search_iterations: int = 10,
    ) -> MetaOptimizationResult:
        """
        Find optimal hyperparameters for fast adaptation.
        """
        return await self.meta_optimizer.optimize_hyperparameters(
            search_iterations=search_iterations,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        few_shot_stats = self.few_shot_learner.get_statistics()

        return {
            "meta_iterations": self.meta_learner.meta_iterations,
            "total_tasks_seen": self.meta_learner.total_tasks_seen,
            "few_shot_tasks_solved": self.few_shot_learner.few_shot_tasks_solved,
            "average_adaptation_steps": self.few_shot_learner.average_adaptation_steps,
            **few_shot_stats,
        }


# ============================================================
# Test Functions
# ============================================================

class SimpleMetaModel(nn.Module):
    """Simple model for meta-learning testing"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, num_classes: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)


async def test_meta_learning():
    """Test meta-learning system"""
    print("=" * 60)
    print("Testing Meta-Learning System")
    print("=" * 60)
    print()

    # Create model
    model = SimpleMetaModel(input_dim=10, hidden_dim=20, num_classes=2)

    # Create meta-learning system
    system = MetaLearningSystem(
        model=model,
        meta_lr=0.001,
        inner_lr=0.01,
        num_inner_steps=5,
        num_shots=5,
        num_ways=2,
    )

    # Test 1: Meta-training
    print("1. Testing meta-training...")
    train_results = await system.meta_train(
        num_iterations=20,
        tasks_per_iteration=4,
    )

    avg_meta_loss = np.mean([r.meta_loss for r in train_results])
    avg_task_acc = np.mean([np.mean(r.task_accuracies) for r in train_results])

    print(f"   Meta-training iterations: {len(train_results)}")
    print(f"   Average meta-loss: {avg_meta_loss:.4f}")
    print(f"   Average task accuracy: {avg_task_acc:.1%}")
    print()

    # Test 2: Few-shot learning
    print("2. Testing few-shot learning...")

    # Generate a new task with 5 examples
    task_weights = torch.randn(10, 2)
    task_bias = torch.randn(2)

    few_shot_examples = []
    for _ in range(10):
        x = torch.randn(10)
        logits = x @ task_weights + task_bias
        y = torch.argmax(logits)
        few_shot_examples.append((x, y))

    adapt_result = await system.adapt_to_new_task(
        examples=few_shot_examples,
        task_name="5-shot classification",
    )

    print(f"   Task: {adapt_result.task_id}")
    print(f"   Initial accuracy: {adapt_result.initial_accuracy:.1%}")
    print(f"   Final accuracy: {adapt_result.final_accuracy:.1%}")
    print(f"   Improvement: {adapt_result.final_accuracy - adapt_result.initial_accuracy:+.1%}")
    print(f"   Adaptation steps: {adapt_result.adaptation_steps}")
    print(f"   Adaptation time: {adapt_result.adaptation_time * 1000:.1f}ms")
    print(f"   Converged: {adapt_result.converged}")
    print()

    # Test 3: Meta-optimization
    print("3. Testing meta-optimization...")

    optimization_result = await system.optimize_hyperparameters(
        search_iterations=10,
    )

    print(f"   Optimal learning rate: {optimization_result.optimal_learning_rate}")
    print(f"   Optimal adaptation steps: {optimization_result.optimal_num_steps}")
    print(f"   Validation accuracy: {optimization_result.validation_accuracy:.1%}")
    print(f"   Search iterations: {optimization_result.search_iterations}")
    print()

    # Test 4: Rapid adaptation (after optimization)
    print("4. Testing rapid adaptation with optimized hyperparameters...")

    # Generate another new task
    rapid_examples = []
    for _ in range(10):
        x = torch.randn(10)
        logits = x @ torch.randn(10, 2) + torch.randn(2)
        y = torch.argmax(logits)
        rapid_examples.append((x, y))

    rapid_result = await system.adapt_to_new_task(
        examples=rapid_examples,
        task_name="rapid_adaptation_task",
    )

    print(f"   Initial accuracy: {rapid_result.initial_accuracy:.1%}")
    print(f"   Final accuracy: {rapid_result.final_accuracy:.1%}")
    print(f"   Improvement: {rapid_result.final_accuracy - rapid_result.initial_accuracy:+.1%}")
    print(f"   Adaptation time: {rapid_result.adaptation_time * 1000:.1f}ms")
    print()

    # System statistics
    print("=" * 60)
    print("System Statistics")
    print("=" * 60)

    stats = system.get_statistics()
    print(f"Meta-training iterations: {stats['meta_iterations']}")
    print(f"Total tasks seen: {stats['total_tasks_seen']}")
    print(f"Few-shot tasks solved: {stats['few_shot_tasks_solved']}")
    print(f"Average initial accuracy: {stats['average_initial_accuracy']:.1%}")
    print(f"Average final accuracy: {stats['average_final_accuracy']:.1%}")
    print(f"Average improvement: {stats['average_improvement']:.1%}")
    print(f"Average adaptation time: {stats['average_adaptation_time'] * 1000:.1f}ms")
    print(f"Convergence rate: {stats['convergence_rate']:.1%}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_meta_learning())
