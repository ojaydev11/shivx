"""
Continual Learning - Learn new tasks without forgetting old ones.

Addresses the "catastrophic forgetting" problem where neural networks
forget previously learned tasks when learning new ones.

Key techniques:
- Elastic Weight Consolidation (EWC): Protect important weights
- Progressive Neural Networks: Add capacity for new tasks
- Experience Replay: Rehearse old examples
- Memory-aware Synapses: Track weight importance

This is crucial for AGI - humans learn continuously throughout life
without forgetting everything they knew before.

Part of ShivX 7/10 AGI transformation (Phase 4).
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque

from core.ml.neural_base import BaseNeuralModel, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class TaskSnapshot:
    """Snapshot of model state for a task"""
    task_id: str
    task_name: str
    parameters: Dict[str, torch.Tensor]
    fisher_information: Dict[str, torch.Tensor]  # For EWC
    performance: float
    timestamp: datetime
    num_examples: int


@dataclass
class MemoryBuffer:
    """Buffer for experience replay"""
    max_size: int
    examples: deque = field(default_factory=deque)
    task_labels: deque = field(default_factory=deque)

    def add(self, example: Dict[str, Any], task_id: str):
        """Add example to buffer"""
        if len(self.examples) >= self.max_size:
            self.examples.popleft()
            self.task_labels.popleft()

        self.examples.append(example)
        self.task_labels.append(task_id)

    def sample(self, n: int) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Sample n examples from buffer"""
        if not self.examples:
            return [], []

        n = min(n, len(self.examples))
        indices = np.random.choice(len(self.examples), size=n, replace=False)

        sampled_examples = [self.examples[i] for i in indices]
        sampled_tasks = [self.task_labels[i] for i in indices]

        return sampled_examples, sampled_tasks

    def get_size(self) -> int:
        """Get current buffer size"""
        return len(self.examples)


class EWCRegularizer:
    """
    Elastic Weight Consolidation (EWC) regularizer.

    Protects important weights from large changes when learning new tasks.
    """

    def __init__(self, lambda_ewc: float = 1000.0):
        """
        Initialize EWC regularizer.

        Args:
            lambda_ewc: Regularization strength
        """
        self.lambda_ewc = lambda_ewc
        self.task_snapshots: Dict[str, TaskSnapshot] = {}

    def compute_fisher_information(
        self,
        model: nn.Module,
        data_loader: List[Dict[str, Any]],
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix diagonal.

        Approximates importance of each parameter for current task.

        Args:
            model: Neural network model
            data_loader: Training data
            device: Device to use

        Returns:
            Dictionary of Fisher information per parameter
        """
        model.eval()

        fisher = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Compute gradients for each example
        for example in data_loader:
            model.zero_grad()

            # Forward pass
            X = torch.FloatTensor([example["input"]]).to(device)
            y = torch.LongTensor([example["label"]]).to(device)

            logits = model(X)
            loss = F.cross_entropy(logits, y)

            # Backward pass
            loss.backward()

            # Accumulate squared gradients (Fisher approximation)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2

        # Average over examples
        n_examples = len(data_loader)
        for name in fisher:
            fisher[name] /= n_examples

        logger.info(f"Computed Fisher information for {len(fisher)} parameters")

        return fisher

    def save_task_snapshot(
        self,
        task_id: str,
        task_name: str,
        model: nn.Module,
        fisher: Dict[str, torch.Tensor],
        performance: float,
        num_examples: int,
    ):
        """Save snapshot of model state for a task"""
        snapshot = TaskSnapshot(
            task_id=task_id,
            task_name=task_name,
            parameters={
                name: param.clone().detach()
                for name, param in model.named_parameters()
            },
            fisher_information=fisher,
            performance=performance,
            timestamp=datetime.utcnow(),
            num_examples=num_examples,
        )

        self.task_snapshots[task_id] = snapshot
        logger.info(f"Saved snapshot for task: {task_name}")

    def compute_ewc_loss(
        self,
        model: nn.Module,
    ) -> torch.Tensor:
        """
        Compute EWC regularization loss.

        Penalizes changes to important parameters.

        Args:
            model: Current model

        Returns:
            EWC loss
        """
        if not self.task_snapshots:
            return torch.tensor(0.0)

        ewc_loss = torch.tensor(0.0)

        for task_id, snapshot in self.task_snapshots.items():
            for name, param in model.named_parameters():
                if name in snapshot.parameters and name in snapshot.fisher_information:
                    old_param = snapshot.parameters[name]
                    fisher = snapshot.fisher_information[name]

                    # Penalize squared difference weighted by Fisher information
                    ewc_loss += (fisher * (param - old_param) ** 2).sum()

        return self.lambda_ewc * ewc_loss


class ContinualLearner:
    """
    Main continual learning system.

    Enables learning new tasks without forgetting old ones.
    """

    def __init__(
        self,
        model: BaseNeuralModel,
        strategy: str = "ewc",  # "ewc", "replay", "hybrid"
        lambda_ewc: float = 1000.0,
        replay_buffer_size: int = 500,
        replay_batch_size: int = 32,
        device: str = "cpu",
    ):
        """
        Initialize Continual Learner.

        Args:
            model: Base neural model
            strategy: Continual learning strategy
            lambda_ewc: EWC regularization strength
            replay_buffer_size: Size of experience replay buffer
            replay_batch_size: Batch size for replay
            device: Device to use
        """
        self.model = model
        self.strategy = strategy
        self.device = device

        # EWC regularizer
        self.ewc = EWCRegularizer(lambda_ewc=lambda_ewc)

        # Experience replay buffer
        self.replay_buffer = MemoryBuffer(max_size=replay_buffer_size)
        self.replay_batch_size = replay_batch_size

        # Task tracking
        self.tasks_learned: List[str] = []
        self.task_performance: Dict[str, List[float]] = defaultdict(list)

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        logger.info(f"Continual Learner initialized: strategy={strategy}")

    def learn_task(
        self,
        task_id: str,
        task_name: str,
        train_examples: List[Dict[str, Any]],
        test_examples: List[Dict[str, Any]],
        num_epochs: int = 50,
    ) -> float:
        """
        Learn a new task with continual learning.

        Args:
            task_id: Task identifier
            task_name: Task name
            train_examples: Training examples
            test_examples: Test examples
            num_epochs: Training epochs

        Returns:
            Final accuracy on test set
        """
        logger.info(f"Learning task: {task_name} ({len(train_examples)} examples)")

        self.model.train()

        # Prepare data
        X_train = torch.FloatTensor([ex["input"] for ex in train_examples]).to(self.device)
        y_train = torch.LongTensor([ex["label"] for ex in train_examples]).to(self.device)

        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # Shuffle data
            perm = torch.randperm(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]

            # Mini-batch training
            batch_size = 32
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                self.optimizer.zero_grad()

                # Forward pass
                logits = self.model(batch_X)
                task_loss = F.cross_entropy(logits, batch_y)

                # Add regularization based on strategy
                total_loss = task_loss

                if self.strategy in ["ewc", "hybrid"]:
                    # Add EWC regularization
                    ewc_loss = self.ewc.compute_ewc_loss(self.model)
                    total_loss += ewc_loss

                if self.strategy in ["replay", "hybrid"]:
                    # Add experience replay
                    if self.replay_buffer.get_size() > 0:
                        replay_examples, _ = self.replay_buffer.sample(
                            self.replay_batch_size
                        )

                        if replay_examples:
                            X_replay = torch.FloatTensor(
                                [ex["input"] for ex in replay_examples]
                            ).to(self.device)
                            y_replay = torch.LongTensor(
                                [ex["label"] for ex in replay_examples]
                            ).to(self.device)

                            logits_replay = self.model(X_replay)
                            replay_loss = F.cross_entropy(logits_replay, y_replay)
                            total_loss += 0.5 * replay_loss

                # Backward pass
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()

            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Evaluate on test set
        accuracy = self.evaluate(test_examples)

        # Save task snapshot (for EWC)
        if self.strategy in ["ewc", "hybrid"]:
            fisher = self.ewc.compute_fisher_information(
                self.model,
                train_examples,
                self.device,
            )
            self.ewc.save_task_snapshot(
                task_id=task_id,
                task_name=task_name,
                model=self.model,
                fisher=fisher,
                performance=accuracy,
                num_examples=len(train_examples),
            )

        # Add examples to replay buffer
        if self.strategy in ["replay", "hybrid"]:
            for example in train_examples:
                self.replay_buffer.add(example, task_id)

        # Track task
        self.tasks_learned.append(task_id)
        self.task_performance[task_id].append(accuracy)

        logger.info(f"Learned task {task_name}: accuracy={accuracy:.3f}")

        return accuracy

    def evaluate(self, test_examples: List[Dict[str, Any]]) -> float:
        """Evaluate model on test examples"""
        if not test_examples:
            return 0.0

        self.model.eval()

        X_test = torch.FloatTensor([ex["input"] for ex in test_examples]).to(self.device)
        y_test = torch.LongTensor([ex["label"] for ex in test_examples]).to(self.device)

        with torch.no_grad():
            logits = self.model(X_test)
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == y_test).float().mean().item()

        return accuracy

    def evaluate_all_tasks(
        self,
        task_test_sets: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, float]:
        """
        Evaluate on all previously learned tasks.

        Used to measure catastrophic forgetting.

        Args:
            task_test_sets: Dictionary of task_id -> test_examples

        Returns:
            Dictionary of task_id -> accuracy
        """
        results = {}

        for task_id in self.tasks_learned:
            if task_id in task_test_sets:
                accuracy = self.evaluate(task_test_sets[task_id])
                results[task_id] = accuracy
                logger.info(f"Task {task_id}: accuracy={accuracy:.3f}")

        return results

    def compute_forgetting(
        self,
        task_test_sets: Dict[str, List[Dict[str, Any]]],
    ) -> float:
        """
        Compute average forgetting across all tasks.

        Forgetting = (peak_accuracy - current_accuracy)

        Args:
            task_test_sets: Dictionary of task_id -> test_examples

        Returns:
            Average forgetting
        """
        if not self.tasks_learned:
            return 0.0

        total_forgetting = 0.0
        num_tasks = 0

        current_performance = self.evaluate_all_tasks(task_test_sets)

        for task_id in self.tasks_learned[:-1]:  # Exclude current task
            if task_id in current_performance:
                peak_accuracy = max(self.task_performance[task_id])
                current_accuracy = current_performance[task_id]

                forgetting = peak_accuracy - current_accuracy
                total_forgetting += forgetting
                num_tasks += 1

        avg_forgetting = total_forgetting / num_tasks if num_tasks > 0 else 0.0

        logger.info(f"Average forgetting: {avg_forgetting:.3f}")

        return avg_forgetting

    def get_stats(self) -> Dict[str, Any]:
        """Get continual learning statistics"""
        return {
            "strategy": self.strategy,
            "num_tasks_learned": len(self.tasks_learned),
            "replay_buffer_size": self.replay_buffer.get_size(),
            "num_snapshots": len(self.ewc.task_snapshots),
            "tasks": self.tasks_learned,
        }


# Convenience function
def quick_continual_test(
    tasks: List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]],
    strategy: str = "hybrid",
) -> Dict[str, float]:
    """
    Quick continual learning test.

    Args:
        tasks: List of (task_id, train_examples, test_examples)
        strategy: Continual learning strategy

    Returns:
        Final accuracy on all tasks
    """
    from core.ml.neural_base import MLPModel

    # Assume all tasks have same input/output dimensions
    input_dim = len(tasks[0][1][0]["input"])
    output_dim = len(set(ex["label"] for ex in tasks[0][1]))

    config = ModelConfig(
        model_name="continual_test",
        input_dim=input_dim,
        hidden_dims=[128, 64],
        output_dim=output_dim,
    )

    model = MLPModel(config)
    learner = ContinualLearner(model, strategy=strategy)

    # Learn tasks sequentially
    for task_id, train_examples, test_examples in tasks:
        learner.learn_task(
            task_id=task_id,
            task_name=task_id,
            train_examples=train_examples,
            test_examples=test_examples,
            num_epochs=30,
        )

    # Evaluate on all tasks
    task_test_sets = {
        task_id: test_examples
        for task_id, _, test_examples in tasks
    }

    results = learner.evaluate_all_tasks(task_test_sets)
    forgetting = learner.compute_forgetting(task_test_sets)

    results["avg_forgetting"] = forgetting

    return results
