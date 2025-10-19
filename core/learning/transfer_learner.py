"""
Transfer Learning - Few-shot learning and cross-task skill transfer.

Enables ShivX to:
- Learn new tasks from very few examples (few-shot learning)
- Transfer skills learned in one domain to another
- Adapt quickly to new scenarios using meta-learning
- Build task-agnostic representations

This is a key component of AGI - the ability to generalize beyond training data.

Part of ShivX 6/10 AGI transformation (Phase 3).
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from core.ml.neural_base import BaseNeuralModel, ModelConfig

logger = logging.getLogger(__name__)

# Optional: learn2learn for meta-learning
try:
    import learn2learn as l2l
    L2L_AVAILABLE = True
except ImportError:
    L2L_AVAILABLE = False
    logger.warning("learn2learn not available. Install with: pip install learn2learn")


@dataclass
class Task:
    """Represents a learning task"""
    id: str
    name: str
    domain: str  # e.g., "code_generation", "bug_fixing", "documentation"
    examples: List[Dict[str, Any]]  # Training examples
    test_examples: List[Dict[str, Any]]  # Test examples
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransferResult:
    """Result of transfer learning"""
    source_task: str
    target_task: str
    source_performance: float
    target_performance_before: float
    target_performance_after: float
    improvement: float
    num_examples_used: int
    transfer_method: str


class MAMLModel(BaseNeuralModel):
    """
    Model-Agnostic Meta-Learning (MAML) network.

    Learns to learn - optimizes for fast adaptation to new tasks.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Feature extractor (shared across tasks)
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
        )

        # Task-specific head (will be adapted per task)
        self.task_head = nn.Linear(config.hidden_dims[1], config.output_dim)

        logger.info(f"MAML Model: {self.count_parameters()} params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.feature_extractor(x)
        output = self.task_head(features)
        return output

    def clone_parameters(self) -> Dict[str, torch.Tensor]:
        """Clone current parameters for inner loop adaptation"""
        return {name: param.clone().requires_grad_(True) for name, param in self.named_parameters()}

    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set parameters (for inner loop updates)"""
        for name, param in self.named_parameters():
            if name in params:
                param.data = params[name].data


class PrototypicalNetwork(BaseNeuralModel):
    """
    Prototypical Networks for few-shot learning.

    Creates "prototypes" for each class and classifies based on distance to prototype.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Embedding network
        self.embedding = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], config.output_dim),
        )

        logger.info(f"Prototypical Network: {self.count_parameters()} params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute embedding"""
        return self.embedding(x)

    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """
        Compute class prototypes (centroids).

        Args:
            support_embeddings: (n_support, embedding_dim)
            support_labels: (n_support,)
            num_classes: Number of classes

        Returns:
            prototypes: (num_classes, embedding_dim)
        """
        prototypes = []

        for c in range(num_classes):
            # Get embeddings for this class
            class_mask = (support_labels == c)
            class_embeddings = support_embeddings[class_mask]

            # Compute prototype (mean)
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes)

    def classify_by_distance(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify queries based on distance to prototypes.

        Args:
            query_embeddings: (n_query, embedding_dim)
            prototypes: (num_classes, embedding_dim)

        Returns:
            logits: (n_query, num_classes)
        """
        # Compute Euclidean distances
        # (n_query, num_classes)
        distances = torch.cdist(query_embeddings, prototypes)

        # Convert distances to logits (negative distance)
        logits = -distances

        return logits


class TransferLearner:
    """
    Main transfer learning system.

    Supports multiple transfer learning methods:
    - MAML (Model-Agnostic Meta-Learning)
    - Prototypical Networks
    - Fine-tuning with feature extraction
    """

    def __init__(
        self,
        method: str = "maml",  # "maml", "prototypical", "finetune"
        input_dim: int = 128,
        output_dim: int = 10,
        device: str = "cpu",
    ):
        """
        Initialize Transfer Learner.

        Args:
            method: Transfer learning method
            input_dim: Input dimension
            output_dim: Output dimension (number of classes)
            device: "cpu" or "cuda"
        """
        self.method = method
        self.device = device

        # Create model
        config = ModelConfig(
            model_name=f"{method}_transfer",
            input_dim=input_dim,
            hidden_dims=[256, 128],
            output_dim=output_dim,
            device=device,
        )

        if method == "maml":
            self.model = MAMLModel(config)
            self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            self.optimizer = self.meta_optimizer  # Alias for learn_task compatibility
            self.inner_lr = 0.01  # Learning rate for inner loop
            self.num_inner_steps = 5
        elif method == "prototypical":
            self.model = PrototypicalNetwork(config)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        elif method == "finetune":
            from core.ml.neural_base import MLPModel
            self.model = MLPModel(config)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Task registry
        self.tasks: Dict[str, Task] = {}

        # Transfer history
        self.transfer_history: List[TransferResult] = []

        # Domain embeddings (for similarity measurement)
        self.domain_embeddings: Dict[str, torch.Tensor] = {}

        logger.info(f"Transfer Learner initialized: {method}")

    def register_task(self, task: Task):
        """Register a task for transfer learning"""
        self.tasks[task.id] = task
        logger.info(f"Registered task: {task.name} ({len(task.examples)} examples)")

    def learn_task(
        self,
        task: Task,
        num_epochs: int = 100,
        batch_size: int = 32,
    ) -> float:
        """
        Learn a single task.

        Args:
            task: Task to learn
            num_epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Final accuracy
        """
        if not task.examples:
            logger.warning(f"No training examples for task: {task.name}")
            return 0.0

        self.model.train()

        # Prepare data
        X_train = torch.FloatTensor([ex["input"] for ex in task.examples]).to(self.device)
        y_train = torch.LongTensor([ex["label"] for ex in task.examples]).to(self.device)

        # Training loop
        for epoch in range(num_epochs):
            # Shuffle data
            perm = torch.randperm(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]

            epoch_loss = 0.0

            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = F.cross_entropy(logits, batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Evaluate
        accuracy = self.evaluate_task(task)

        logger.info(f"Learned task {task.name}: accuracy={accuracy:.3f}")

        return accuracy

    def evaluate_task(self, task: Task) -> float:
        """Evaluate on task test set"""
        if not task.test_examples:
            return 0.0

        self.model.eval()

        X_test = torch.FloatTensor([ex["input"] for ex in task.test_examples]).to(self.device)
        y_test = torch.LongTensor([ex["label"] for ex in task.test_examples]).to(self.device)

        with torch.no_grad():
            logits = self.model(X_test)
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == y_test).float().mean().item()

        return accuracy

    def few_shot_adapt(
        self,
        support_set: List[Dict[str, Any]],
        query_set: List[Dict[str, Any]],
        num_classes: int,
    ) -> float:
        """
        Adapt to a new task using few-shot learning.

        Args:
            support_set: Few examples to adapt from (k-shot)
            query_set: Examples to evaluate on
            num_classes: Number of classes in this task

        Returns:
            Accuracy on query set
        """
        if self.method == "prototypical":
            return self._prototypical_adapt(support_set, query_set, num_classes)
        elif self.method == "maml":
            return self._maml_adapt(support_set, query_set)
        else:
            return self._finetune_adapt(support_set, query_set)

    def _prototypical_adapt(
        self,
        support_set: List[Dict[str, Any]],
        query_set: List[Dict[str, Any]],
        num_classes: int,
    ) -> float:
        """Prototypical Networks few-shot adaptation"""
        self.model.eval()

        # Support set
        X_support = torch.FloatTensor([ex["input"] for ex in support_set]).to(self.device)
        y_support = torch.LongTensor([ex["label"] for ex in support_set]).to(self.device)

        # Query set
        X_query = torch.FloatTensor([ex["input"] for ex in query_set]).to(self.device)
        y_query = torch.LongTensor([ex["label"] for ex in query_set]).to(self.device)

        with torch.no_grad():
            # Compute embeddings
            support_embeddings = self.model(X_support)
            query_embeddings = self.model(X_query)

            # Compute prototypes
            prototypes = self.model.compute_prototypes(
                support_embeddings,
                y_support,
                num_classes,
            )

            # Classify queries
            logits = self.model.classify_by_distance(query_embeddings, prototypes)
            predictions = torch.argmax(logits, dim=-1)

            accuracy = (predictions == y_query).float().mean().item()

        return accuracy

    def _maml_adapt(
        self,
        support_set: List[Dict[str, Any]],
        query_set: List[Dict[str, Any]],
    ) -> float:
        """MAML few-shot adaptation"""
        # Clone current parameters
        adapted_params = self.model.clone_parameters()

        # Prepare support data
        X_support = torch.FloatTensor([ex["input"] for ex in support_set]).to(self.device)
        y_support = torch.LongTensor([ex["label"] for ex in support_set]).to(self.device)

        # Inner loop adaptation
        for _ in range(self.num_inner_steps):
            # Forward with current adapted params
            self.model.set_parameters(adapted_params)
            logits = self.model(X_support)
            loss = F.cross_entropy(logits, y_support)

            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True, allow_unused=True)

            # Update adapted params (skip if grad is None)
            adapted_params = {
                name: param - self.inner_lr * grad if grad is not None else param
                for (name, param), grad in zip(adapted_params.items(), grads)
            }

        # Evaluate on query set with adapted parameters
        self.model.set_parameters(adapted_params)
        self.model.eval()

        X_query = torch.FloatTensor([ex["input"] for ex in query_set]).to(self.device)
        y_query = torch.LongTensor([ex["label"] for ex in query_set]).to(self.device)

        with torch.no_grad():
            logits = self.model(X_query)
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == y_query).float().mean().item()

        return accuracy

    def _finetune_adapt(
        self,
        support_set: List[Dict[str, Any]],
        query_set: List[Dict[str, Any]],
        num_epochs: int = 10,
    ) -> float:
        """Simple fine-tuning adaptation"""
        # Fine-tune on support set
        X_support = torch.FloatTensor([ex["input"] for ex in support_set]).to(self.device)
        y_support = torch.LongTensor([ex["label"] for ex in support_set]).to(self.device)

        self.model.train()

        for _ in range(num_epochs):
            self.optimizer.zero_grad()
            logits = self.model(X_support)
            loss = F.cross_entropy(logits, y_support)
            loss.backward()
            self.optimizer.step()

        # Evaluate on query set
        return self.evaluate_task(Task(
            id="query",
            name="query",
            domain="",
            examples=[],
            test_examples=query_set,
        ))

    def transfer(
        self,
        source_task_id: str,
        target_task: Task,
        num_examples: int = 5,
    ) -> TransferResult:
        """
        Transfer knowledge from source task to target task.

        Args:
            source_task_id: ID of source task
            target_task: Target task to adapt to
            num_examples: Number of examples from target task to use

        Returns:
            TransferResult with performance metrics
        """
        if source_task_id not in self.tasks:
            raise ValueError(f"Unknown source task: {source_task_id}")

        source_task = self.tasks[source_task_id]

        # Evaluate source task performance
        source_performance = self.evaluate_task(source_task)

        # Evaluate target task before transfer (zero-shot)
        target_performance_before = self.evaluate_task(target_task) if target_task.test_examples else 0.0

        # Use few-shot adaptation
        support_set = target_task.examples[:num_examples]
        query_set = target_task.test_examples if target_task.test_examples else target_task.examples[num_examples:]

        if not query_set:
            logger.warning("No query set for evaluation")
            query_set = support_set

        num_classes = len(set(ex["label"] for ex in support_set))

        target_performance_after = self.few_shot_adapt(
            support_set=support_set,
            query_set=query_set,
            num_classes=num_classes,
        )

        improvement = target_performance_after - target_performance_before

        result = TransferResult(
            source_task=source_task_id,
            target_task=target_task.id,
            source_performance=source_performance,
            target_performance_before=target_performance_before,
            target_performance_after=target_performance_after,
            improvement=improvement,
            num_examples_used=num_examples,
            transfer_method=self.method,
        )

        self.transfer_history.append(result)

        logger.info(
            f"Transfer {source_task.name} → {target_task.name}: "
            f"{target_performance_before:.3f} → {target_performance_after:.3f} "
            f"(+{improvement:.3f})"
        )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get transfer learning statistics"""
        avg_improvement = np.mean([r.improvement for r in self.transfer_history]) if self.transfer_history else 0.0

        return {
            "method": self.method,
            "num_tasks": len(self.tasks),
            "num_transfers": len(self.transfer_history),
            "avg_improvement": avg_improvement,
            "model_params": self.model.count_parameters(),
        }


# Convenience function
def quick_transfer(
    source_examples: List[Dict[str, Any]],
    target_examples: List[Dict[str, Any]],
    method: str = "prototypical",
) -> float:
    """
    Quick transfer learning test.

    Args:
        source_examples: Source task examples
        target_examples: Target task examples (will split into support/query)
        method: Transfer method

    Returns:
        Accuracy on target task
    """
    learner = TransferLearner(method=method)

    # Learn source task
    source_task = Task(
        id="source",
        name="source",
        domain="source",
        examples=source_examples[:int(0.8*len(source_examples))],
        test_examples=source_examples[int(0.8*len(source_examples)):],
    )

    learner.register_task(source_task)
    learner.learn_task(source_task, num_epochs=50)

    # Transfer to target task
    target_task = Task(
        id="target",
        name="target",
        domain="target",
        examples=target_examples[:5],  # 5-shot
        test_examples=target_examples[5:],
    )

    result = learner.transfer(source_task.id, target_task, num_examples=5)

    return result.target_performance_after
