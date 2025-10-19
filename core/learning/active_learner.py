"""
Active Learning - Query for the most informative examples.

Instead of randomly sampling training data, actively select the examples
that will provide the most learning value. This dramatically reduces the
amount of labeled data needed.

Key strategies:
- Uncertainty Sampling: Query examples the model is uncertain about
- Query-by-Committee: Multiple models vote on uncertainty
- Expected Model Change: Query examples that will change model most
- Diversity Sampling: Ensure diverse coverage of input space

This is crucial for AGI - intelligent agents should actively seek
information that maximizes learning, not passively consume random data.

Part of ShivX 7/10 AGI transformation (Phase 4).
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from core.ml.neural_base import BaseNeuralModel, ModelConfig

logger = logging.getLogger(__name__)


class QueryStrategy(Enum):
    """Active learning query strategies"""
    UNCERTAINTY = "uncertainty"  # Query most uncertain examples
    MARGIN = "margin"  # Query examples with smallest margin
    ENTROPY = "entropy"  # Query highest entropy examples
    COMMITTEE = "committee"  # Query-by-committee
    EXPECTED_CHANGE = "expected_change"  # Expected model change
    DIVERSE = "diverse"  # Maximize diversity


@dataclass
class QueryResult:
    """Result of an active learning query"""
    indices: List[int]  # Indices of queried examples
    scores: List[float]  # Informativeness scores
    strategy: QueryStrategy
    num_queried: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class UncertaintySampler:
    """
    Uncertainty-based sampling strategies.

    Queries examples the model is most uncertain about.
    """

    def __init__(self, model: BaseNeuralModel, device: str = "cpu"):
        """
        Initialize Uncertainty Sampler.

        Args:
            model: Neural network model
            device: Device to use
        """
        self.model = model
        self.device = device

    def compute_uncertainty(
        self,
        X: torch.Tensor,
        method: str = "entropy",
    ) -> torch.Tensor:
        """
        Compute uncertainty for each example.

        Args:
            X: Input tensor (batch_size, input_dim)
            method: "entropy", "margin", or "least_confidence"

        Returns:
            Uncertainty scores (batch_size,)
        """
        self.model.eval()

        with torch.no_grad():
            logits = self.model(X)
            probs = F.softmax(logits, dim=-1)

            if method == "entropy":
                # Entropy: -sum(p * log(p))
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                return entropy

            elif method == "margin":
                # Margin: difference between top 2 predictions
                sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
                margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                # Return negative margin (higher uncertainty = smaller margin)
                return -margin

            elif method == "least_confidence":
                # Least confidence: 1 - max(p)
                max_probs, _ = torch.max(probs, dim=-1)
                return 1.0 - max_probs

            else:
                raise ValueError(f"Unknown uncertainty method: {method}")

    def query(
        self,
        unlabeled_pool: List[Dict[str, Any]],
        num_query: int = 10,
        method: str = "entropy",
    ) -> QueryResult:
        """
        Query most uncertain examples from unlabeled pool.

        Args:
            unlabeled_pool: Pool of unlabeled examples
            num_query: Number of examples to query
            method: Uncertainty estimation method

        Returns:
            QueryResult with selected indices
        """
        if not unlabeled_pool:
            return QueryResult(
                indices=[],
                scores=[],
                strategy=QueryStrategy.UNCERTAINTY,
                num_queried=0,
            )

        # Convert to tensor
        X = torch.FloatTensor([ex["input"] for ex in unlabeled_pool]).to(self.device)

        # Compute uncertainty
        uncertainty_scores = self.compute_uncertainty(X, method=method)

        # Select top-k most uncertain
        num_query = min(num_query, len(unlabeled_pool))
        top_k_scores, top_k_indices = torch.topk(uncertainty_scores, k=num_query)

        result = QueryResult(
            indices=top_k_indices.cpu().tolist(),
            scores=top_k_scores.cpu().tolist(),
            strategy=QueryStrategy.UNCERTAINTY,
            num_queried=num_query,
        )

        logger.info(f"Queried {num_query} examples (uncertainty: {method})")

        return result


class QueryByCommittee:
    """
    Query-by-Committee (QBC) strategy.

    Uses disagreement among multiple models to measure uncertainty.
    """

    def __init__(
        self,
        models: List[BaseNeuralModel],
        device: str = "cpu",
    ):
        """
        Initialize Query-by-Committee.

        Args:
            models: Committee of models
            device: Device to use
        """
        self.models = models
        self.device = device

    def compute_disagreement(
        self,
        X: torch.Tensor,
        method: str = "vote_entropy",
    ) -> torch.Tensor:
        """
        Compute disagreement among committee members.

        Args:
            X: Input tensor (batch_size, input_dim)
            method: "vote_entropy" or "kl_divergence"

        Returns:
            Disagreement scores (batch_size,)
        """
        # Get predictions from all models
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(X)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

        predictions = torch.stack(predictions)  # (num_models, batch_size, num_classes)

        if method == "vote_entropy":
            # Vote entropy: entropy of average predictions
            avg_probs = predictions.mean(dim=0)  # (batch_size, num_classes)
            entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum(dim=-1)
            return entropy

        elif method == "kl_divergence":
            # Average KL divergence from consensus
            avg_probs = predictions.mean(dim=0)
            kl_divs = []

            for model_probs in predictions:
                kl = F.kl_div(
                    torch.log(model_probs + 1e-10),
                    avg_probs,
                    reduction="none",
                ).sum(dim=-1)
                kl_divs.append(kl)

            avg_kl = torch.stack(kl_divs).mean(dim=0)
            return avg_kl

        else:
            raise ValueError(f"Unknown disagreement method: {method}")

    def query(
        self,
        unlabeled_pool: List[Dict[str, Any]],
        num_query: int = 10,
        method: str = "vote_entropy",
    ) -> QueryResult:
        """Query examples with highest disagreement"""
        if not unlabeled_pool:
            return QueryResult(
                indices=[],
                scores=[],
                strategy=QueryStrategy.COMMITTEE,
                num_queried=0,
            )

        X = torch.FloatTensor([ex["input"] for ex in unlabeled_pool]).to(self.device)

        disagreement_scores = self.compute_disagreement(X, method=method)

        num_query = min(num_query, len(unlabeled_pool))
        top_k_scores, top_k_indices = torch.topk(disagreement_scores, k=num_query)

        result = QueryResult(
            indices=top_k_indices.cpu().tolist(),
            scores=top_k_scores.cpu().tolist(),
            strategy=QueryStrategy.COMMITTEE,
            num_queried=num_query,
        )

        logger.info(f"Queried {num_query} examples (committee disagreement)")

        return result


class DiversitySampler:
    """
    Diversity-based sampling.

    Ensures queried examples cover diverse regions of input space.
    """

    def __init__(self, device: str = "cpu"):
        """Initialize Diversity Sampler"""
        self.device = device

    def compute_diversity_scores(
        self,
        X: torch.Tensor,
        already_labeled: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute diversity scores using k-means++ initialization.

        Args:
            X: Unlabeled examples (batch_size, input_dim)
            already_labeled: Already labeled examples (optional)

        Returns:
            Diversity scores (batch_size,)
        """
        # Distance to nearest labeled example
        if already_labeled is not None and len(already_labeled) > 0:
            # Compute pairwise distances
            distances = torch.cdist(X, already_labeled)  # (batch_size, num_labeled)
            min_distances, _ = torch.min(distances, dim=-1)
        else:
            # No labeled examples yet, all have equal diversity
            min_distances = torch.ones(len(X))

        return min_distances

    def query(
        self,
        unlabeled_pool: List[Dict[str, Any]],
        num_query: int = 10,
        already_labeled: Optional[List[Dict[str, Any]]] = None,
    ) -> QueryResult:
        """Query diverse examples"""
        if not unlabeled_pool:
            return QueryResult(
                indices=[],
                scores=[],
                strategy=QueryStrategy.DIVERSE,
                num_queried=0,
            )

        X_unlabeled = torch.FloatTensor([ex["input"] for ex in unlabeled_pool]).to(self.device)

        X_labeled = None
        if already_labeled:
            X_labeled = torch.FloatTensor([ex["input"] for ex in already_labeled]).to(self.device)

        # Iteratively select diverse examples
        selected_indices = []
        selected_X = []

        for _ in range(min(num_query, len(unlabeled_pool))):
            # Combine already labeled + already selected
            if X_labeled is not None or selected_X:
                combined = []
                if X_labeled is not None:
                    combined.append(X_labeled)
                if selected_X:
                    combined.append(torch.stack(selected_X))

                current_labeled = torch.cat(combined, dim=0)
            else:
                current_labeled = None

            # Compute diversity scores
            diversity_scores = self.compute_diversity_scores(X_unlabeled, current_labeled)

            # Select most diverse (farthest from labeled)
            max_score, max_idx = torch.max(diversity_scores, dim=0)

            selected_indices.append(max_idx.item())
            selected_X.append(X_unlabeled[max_idx])

            # Remove from pool (set distance to 0)
            diversity_scores[max_idx] = 0.0

        result = QueryResult(
            indices=selected_indices,
            scores=[1.0] * len(selected_indices),  # Placeholder scores
            strategy=QueryStrategy.DIVERSE,
            num_queried=len(selected_indices),
        )

        logger.info(f"Queried {len(selected_indices)} diverse examples")

        return result


class ActiveLearner:
    """
    Main active learning system.

    Intelligently queries informative examples to minimize labeling cost.
    """

    def __init__(
        self,
        model: BaseNeuralModel,
        strategy: QueryStrategy = QueryStrategy.UNCERTAINTY,
        device: str = "cpu",
    ):
        """
        Initialize Active Learner.

        Args:
            model: Base neural model
            strategy: Query strategy
            device: Device to use
        """
        self.model = model
        self.strategy = strategy
        self.device = device

        # Samplers
        self.uncertainty_sampler = UncertaintySampler(model, device)
        self.diversity_sampler = DiversitySampler(device)

        # Data pools
        self.labeled_pool: List[Dict[str, Any]] = []
        self.unlabeled_pool: List[Dict[str, Any]] = []

        # Query history
        self.query_history: List[QueryResult] = []

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        logger.info(f"Active Learner initialized: strategy={strategy.value}")

    def set_data(
        self,
        unlabeled_pool: List[Dict[str, Any]],
        initial_labeled: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Set data pools.

        Args:
            unlabeled_pool: Pool of unlabeled examples
            initial_labeled: Initial labeled examples (optional)
        """
        self.unlabeled_pool = unlabeled_pool.copy()
        self.labeled_pool = initial_labeled.copy() if initial_labeled else []

        logger.info(
            f"Data set: {len(self.labeled_pool)} labeled, "
            f"{len(self.unlabeled_pool)} unlabeled"
        )

    def query(self, num_query: int = 10) -> QueryResult:
        """
        Query most informative examples.

        Args:
            num_query: Number of examples to query

        Returns:
            QueryResult with selected examples
        """
        if not self.unlabeled_pool:
            logger.warning("No unlabeled examples to query")
            return QueryResult(
                indices=[],
                scores=[],
                strategy=self.strategy,
                num_queried=0,
            )

        # Query based on strategy
        if self.strategy == QueryStrategy.UNCERTAINTY:
            result = self.uncertainty_sampler.query(
                self.unlabeled_pool,
                num_query=num_query,
                method="entropy",
            )
        elif self.strategy == QueryStrategy.MARGIN:
            result = self.uncertainty_sampler.query(
                self.unlabeled_pool,
                num_query=num_query,
                method="margin",
            )
        elif self.strategy == QueryStrategy.ENTROPY:
            result = self.uncertainty_sampler.query(
                self.unlabeled_pool,
                num_query=num_query,
                method="entropy",
            )
        elif self.strategy == QueryStrategy.DIVERSE:
            result = self.diversity_sampler.query(
                self.unlabeled_pool,
                num_query=num_query,
                already_labeled=self.labeled_pool,
            )
        else:
            # Default to uncertainty
            result = self.uncertainty_sampler.query(
                self.unlabeled_pool,
                num_query=num_query,
            )

        self.query_history.append(result)

        logger.info(f"Query result: {result.num_queried} examples selected")

        return result

    def label_and_train(
        self,
        query_result: QueryResult,
        oracle_labels: Optional[List[int]] = None,
        num_epochs: int = 20,
    ) -> float:
        """
        Label queried examples and retrain model.

        Args:
            query_result: Result from query()
            oracle_labels: True labels (if available)
            num_epochs: Training epochs

        Returns:
            Training accuracy
        """
        # Move queried examples to labeled pool
        queried_examples = []
        for idx in sorted(query_result.indices, reverse=True):
            example = self.unlabeled_pool.pop(idx)

            # If oracle labels provided, use them
            if oracle_labels and len(oracle_labels) > len(queried_examples):
                example["label"] = oracle_labels[len(queried_examples)]

            queried_examples.append(example)
            self.labeled_pool.append(example)

        logger.info(
            f"Labeled {len(queried_examples)} examples. "
            f"Total labeled: {len(self.labeled_pool)}"
        )

        # Train on labeled pool
        accuracy = self.train(num_epochs=num_epochs)

        return accuracy

    def train(self, num_epochs: int = 20) -> float:
        """Train model on labeled pool"""
        if not self.labeled_pool:
            logger.warning("No labeled examples to train on")
            return 0.0

        self.model.train()

        X_train = torch.FloatTensor([ex["input"] for ex in self.labeled_pool]).to(self.device)
        y_train = torch.LongTensor([ex["label"] for ex in self.labeled_pool]).to(self.device)

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            logits = self.model(X_train)
            loss = F.cross_entropy(logits, y_train)

            loss.backward()
            self.optimizer.step()

        # Evaluate
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_train)
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == y_train).float().mean().item()

        logger.info(f"Trained on {len(self.labeled_pool)} examples: accuracy={accuracy:.3f}")

        return accuracy

    def evaluate(self, test_examples: List[Dict[str, Any]]) -> float:
        """Evaluate on test set"""
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

    def active_learning_loop(
        self,
        test_examples: List[Dict[str, Any]],
        num_iterations: int = 10,
        examples_per_iteration: int = 10,
        oracle_fn: Optional[Callable] = None,
    ) -> List[float]:
        """
        Run active learning loop.

        Args:
            test_examples: Test set for evaluation
            num_iterations: Number of query iterations
            examples_per_iteration: Examples to query per iteration
            oracle_fn: Function to label examples

        Returns:
            List of test accuracies per iteration
        """
        test_accuracies = []

        for iteration in range(num_iterations):
            logger.info(f"\n=== Active Learning Iteration {iteration + 1}/{num_iterations} ===")

            # Query examples
            query_result = self.query(num_query=examples_per_iteration)

            if query_result.num_queried == 0:
                logger.info("No more unlabeled examples")
                break

            # Get labels from oracle
            oracle_labels = None
            if oracle_fn:
                queried_examples = [self.unlabeled_pool[idx] for idx in query_result.indices]
                oracle_labels = [oracle_fn(ex) for ex in queried_examples]

            # Label and train
            self.label_and_train(query_result, oracle_labels=oracle_labels)

            # Evaluate
            test_accuracy = self.evaluate(test_examples)
            test_accuracies.append(test_accuracy)

            logger.info(
                f"Iteration {iteration + 1}: "
                f"Labeled={len(self.labeled_pool)}, "
                f"Test Accuracy={test_accuracy:.3f}"
            )

        return test_accuracies

    def get_stats(self) -> Dict[str, Any]:
        """Get active learning statistics"""
        return {
            "strategy": self.strategy.value,
            "num_labeled": len(self.labeled_pool),
            "num_unlabeled": len(self.unlabeled_pool),
            "num_queries": len(self.query_history),
            "total_queried": sum(q.num_queried for q in self.query_history),
        }


# Convenience function
def quick_active_learning_test(
    all_examples: List[Dict[str, Any]],
    test_examples: List[Dict[str, Any]],
    strategy: QueryStrategy = QueryStrategy.UNCERTAINTY,
    initial_labeled: int = 10,
    num_iterations: int = 5,
) -> List[float]:
    """
    Quick active learning test.

    Args:
        all_examples: All available examples (will split into labeled/unlabeled)
        test_examples: Test set
        strategy: Query strategy
        initial_labeled: Number of initial labeled examples
        num_iterations: Number of active learning iterations

    Returns:
        Test accuracies per iteration
    """
    from core.ml.neural_base import MLPModel

    input_dim = len(all_examples[0]["input"])
    output_dim = len(set(ex["label"] for ex in all_examples))

    config = ModelConfig(
        model_name="active_test",
        input_dim=input_dim,
        hidden_dims=[128, 64],
        output_dim=output_dim,
    )

    model = MLPModel(config)
    learner = ActiveLearner(model, strategy=strategy)

    # Split into initial labeled and unlabeled
    initial = all_examples[:initial_labeled]
    unlabeled = all_examples[initial_labeled:]

    learner.set_data(unlabeled_pool=unlabeled, initial_labeled=initial)

    # Run active learning loop
    test_accuracies = learner.active_learning_loop(
        test_examples=test_examples,
        num_iterations=num_iterations,
        examples_per_iteration=10,
        oracle_fn=lambda ex: ex["label"],  # True labels available
    )

    return test_accuracies
