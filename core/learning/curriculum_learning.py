"""
Curriculum Learning System for Empire AGI
Week 17: Easy-to-Hard Training

Implements curriculum learning strategies that train models progressively
from easy to hard examples, improving learning efficiency and final performance.

Key capabilities:
- Difficulty scoring: Automatically assess sample difficulty
- Curriculum strategies: Linear, exponential, step-wise progression
- Adaptive curriculum: Adjust based on learner performance
- Multi-task curriculum: Coordinate curriculum across multiple tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
import numpy as np
import time


class DifficultyMetric(Enum):
    """Metrics for assessing sample difficulty"""
    LOSS_BASED = "loss_based"  # Higher loss = harder
    CONFIDENCE_BASED = "confidence_based"  # Lower confidence = harder
    GRADIENT_BASED = "gradient_based"  # Higher gradient = harder
    ENSEMBLE_DISAGREEMENT = "ensemble_disagreement"  # More disagreement = harder
    CUSTOM = "custom"  # User-defined difficulty function


class CurriculumStrategy(Enum):
    """Curriculum progression strategies"""
    LINEAR = "linear"  # Uniform increase in difficulty
    EXPONENTIAL = "exponential"  # Slow start, fast later
    STEP_WISE = "step_wise"  # Discrete difficulty steps
    ADAPTIVE = "adaptive"  # Based on learner performance
    SELF_PACED = "self_paced"  # Learner chooses difficulty


class ScoringMode(Enum):
    """How to score sample difficulty"""
    PRETRAINED = "pretrained"  # Use pretrained model
    ENSEMBLE = "ensemble"  # Use ensemble of models
    HEURISTIC = "heuristic"  # Use domain heuristics
    LEARNED = "learned"  # Learn difficulty scorer


@dataclass
class Sample:
    """A training sample with difficulty score"""
    features: torch.Tensor
    label: torch.Tensor
    difficulty: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CurriculumPhase:
    """A phase in the curriculum"""
    phase_id: int
    min_difficulty: float
    max_difficulty: float
    num_epochs: int
    learning_rate: float
    samples: List[Sample]


@dataclass
class CurriculumProgress:
    """Progress through curriculum"""
    current_phase: int
    total_phases: int
    samples_seen: int
    current_difficulty: float
    performance_history: List[float]
    phase_completion: float


@dataclass
class CurriculumResult:
    """Result of curriculum training"""
    final_accuracy: float
    training_time: float
    phases_completed: int
    total_samples_seen: int
    convergence_epoch: int
    performance_curve: List[float]


class DifficultyScorer:
    """
    Assesses difficulty of training samples.

    Uses various metrics to score how difficult each sample is for learning.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        metric: DifficultyMetric = DifficultyMetric.LOSS_BASED,
        custom_scorer: Optional[Callable] = None,
    ):
        self.model = model
        self.metric = metric
        self.custom_scorer = custom_scorer

        # Statistics
        self.num_scored = 0
        self.difficulty_distribution = []

    def score_sample(
        self,
        sample: Sample,
    ) -> float:
        """
        Score the difficulty of a single sample.

        Returns a difficulty score in [0, 1] where:
        - 0 = very easy
        - 1 = very hard
        """
        if self.metric == DifficultyMetric.CUSTOM and self.custom_scorer:
            difficulty = self.custom_scorer(sample)
        elif self.metric == DifficultyMetric.LOSS_BASED:
            difficulty = self._score_by_loss(sample)
        elif self.metric == DifficultyMetric.CONFIDENCE_BASED:
            difficulty = self._score_by_confidence(sample)
        elif self.metric == DifficultyMetric.GRADIENT_BASED:
            difficulty = self._score_by_gradient(sample)
        else:
            # Default: random difficulty
            difficulty = np.random.random()

        self.num_scored += 1
        self.difficulty_distribution.append(difficulty)

        return float(difficulty)

    def score_dataset(
        self,
        samples: List[Sample],
    ) -> List[Sample]:
        """
        Score difficulty for all samples in dataset.
        """
        scored_samples = []

        for sample in samples:
            difficulty = self.score_sample(sample)
            sample.difficulty = difficulty
            scored_samples.append(sample)

        return scored_samples

    def _score_by_loss(self, sample: Sample) -> float:
        """
        Score difficulty based on model loss.

        Higher loss = harder sample.
        """
        if self.model is None:
            return np.random.random()

        self.model.eval()
        with torch.no_grad():
            output = self.model(sample.features.unsqueeze(0))
            loss = F.cross_entropy(output, sample.label.unsqueeze(0))

        # Normalize to [0, 1] using sigmoid
        difficulty = torch.sigmoid(loss - 1.0).item()

        return difficulty

    def _score_by_confidence(self, sample: Sample) -> float:
        """
        Score difficulty based on model confidence.

        Lower confidence = harder sample.
        """
        if self.model is None:
            return np.random.random()

        self.model.eval()
        with torch.no_grad():
            output = self.model(sample.features.unsqueeze(0))
            probs = F.softmax(output, dim=1)
            confidence = probs.max().item()

        # Invert confidence (low confidence = high difficulty)
        difficulty = 1.0 - confidence

        return difficulty

    def _score_by_gradient(self, sample: Sample) -> float:
        """
        Score difficulty based on gradient magnitude.

        Larger gradients = harder sample (more to learn).
        """
        if self.model is None:
            return np.random.random()

        self.model.eval()
        self.model.zero_grad()

        output = self.model(sample.features.unsqueeze(0))
        loss = F.cross_entropy(output, sample.label.unsqueeze(0))
        loss.backward()

        # Calculate total gradient magnitude
        total_grad = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad += param.grad.abs().sum().item()

        # Normalize using sigmoid
        difficulty = torch.sigmoid(torch.tensor(total_grad / 100.0)).item()

        return difficulty

    def get_statistics(self) -> Dict[str, Any]:
        """Get difficulty scoring statistics"""
        if not self.difficulty_distribution:
            return {
                "num_scored": 0,
                "mean_difficulty": 0.0,
                "std_difficulty": 0.0,
                "min_difficulty": 0.0,
                "max_difficulty": 0.0,
            }

        return {
            "num_scored": self.num_scored,
            "mean_difficulty": np.mean(self.difficulty_distribution),
            "std_difficulty": np.std(self.difficulty_distribution),
            "min_difficulty": np.min(self.difficulty_distribution),
            "max_difficulty": np.max(self.difficulty_distribution),
        }


class CurriculumGenerator:
    """
    Generates curriculum (sequence of training phases) from dataset.

    Orders samples from easy to hard and creates training phases.
    """

    def __init__(
        self,
        strategy: CurriculumStrategy = CurriculumStrategy.LINEAR,
        num_phases: int = 5,
        samples_per_phase: Optional[int] = None,
    ):
        self.strategy = strategy
        self.num_phases = num_phases
        self.samples_per_phase = samples_per_phase

    def generate_curriculum(
        self,
        samples: List[Sample],
        base_lr: float = 0.001,
    ) -> List[CurriculumPhase]:
        """
        Generate curriculum phases from scored samples.
        """
        # Sort samples by difficulty
        sorted_samples = sorted(samples, key=lambda s: s.difficulty)

        # Determine samples per phase
        if self.samples_per_phase is None:
            self.samples_per_phase = len(sorted_samples) // self.num_phases

        phases = []

        if self.strategy == CurriculumStrategy.LINEAR:
            phases = self._generate_linear_curriculum(sorted_samples, base_lr)
        elif self.strategy == CurriculumStrategy.EXPONENTIAL:
            phases = self._generate_exponential_curriculum(sorted_samples, base_lr)
        elif self.strategy == CurriculumStrategy.STEP_WISE:
            phases = self._generate_stepwise_curriculum(sorted_samples, base_lr)
        else:
            # Default: linear
            phases = self._generate_linear_curriculum(sorted_samples, base_lr)

        return phases

    def _generate_linear_curriculum(
        self,
        sorted_samples: List[Sample],
        base_lr: float,
    ) -> List[CurriculumPhase]:
        """
        Linear curriculum: Evenly spaced difficulty phases.
        """
        phases = []
        samples_per_phase = len(sorted_samples) // self.num_phases

        for phase_id in range(self.num_phases):
            start_idx = phase_id * samples_per_phase
            end_idx = (phase_id + 1) * samples_per_phase if phase_id < self.num_phases - 1 else len(sorted_samples)

            phase_samples = sorted_samples[start_idx:end_idx]

            if not phase_samples:
                continue

            min_diff = min(s.difficulty for s in phase_samples)
            max_diff = max(s.difficulty for s in phase_samples)

            # Learning rate decreases as difficulty increases
            lr = base_lr * (1.0 - (phase_id / self.num_phases) * 0.5)

            phase = CurriculumPhase(
                phase_id=phase_id,
                min_difficulty=min_diff,
                max_difficulty=max_diff,
                num_epochs=3,  # Fixed for now
                learning_rate=lr,
                samples=phase_samples,
            )

            phases.append(phase)

        return phases

    def _generate_exponential_curriculum(
        self,
        sorted_samples: List[Sample],
        base_lr: float,
    ) -> List[CurriculumPhase]:
        """
        Exponential curriculum: More samples in later (harder) phases.
        """
        phases = []
        total_samples = len(sorted_samples)

        # Exponential distribution of samples
        phase_sizes = []
        for phase_id in range(self.num_phases):
            # Exponential growth: phase 0 gets few samples, last phase gets many
            proportion = np.exp(phase_id / self.num_phases) / np.exp(1)
            size = int(total_samples * proportion / (self.num_phases * 0.5))
            phase_sizes.append(size)

        # Normalize to total_samples
        total = sum(phase_sizes)
        phase_sizes = [int(s * total_samples / total) for s in phase_sizes]

        current_idx = 0
        for phase_id, size in enumerate(phase_sizes):
            if current_idx >= total_samples:
                break

            end_idx = min(current_idx + size, total_samples)
            phase_samples = sorted_samples[current_idx:end_idx]

            if not phase_samples:
                break

            min_diff = min(s.difficulty for s in phase_samples)
            max_diff = max(s.difficulty for s in phase_samples)

            lr = base_lr * (1.0 - (phase_id / self.num_phases) * 0.5)

            phase = CurriculumPhase(
                phase_id=phase_id,
                min_difficulty=min_diff,
                max_difficulty=max_diff,
                num_epochs=2 + phase_id,  # More epochs for harder phases
                learning_rate=lr,
                samples=phase_samples,
            )

            phases.append(phase)
            current_idx = end_idx

        return phases

    def _generate_stepwise_curriculum(
        self,
        sorted_samples: List[Sample],
        base_lr: float,
    ) -> List[CurriculumPhase]:
        """
        Step-wise curriculum: Discrete difficulty bins.
        """
        # Create difficulty bins
        difficulty_bins = np.linspace(0, 1, self.num_phases + 1)

        phases = []

        for phase_id in range(self.num_phases):
            min_bin = difficulty_bins[phase_id]
            max_bin = difficulty_bins[phase_id + 1]

            # Select samples in this difficulty range
            phase_samples = [
                s for s in sorted_samples
                if min_bin <= s.difficulty < max_bin
            ]

            if not phase_samples:
                continue

            min_diff = min(s.difficulty for s in phase_samples)
            max_diff = max(s.difficulty for s in phase_samples)

            lr = base_lr * (1.0 - (phase_id / self.num_phases) * 0.5)

            phase = CurriculumPhase(
                phase_id=phase_id,
                min_difficulty=min_diff,
                max_difficulty=max_diff,
                num_epochs=3,
                learning_rate=lr,
                samples=phase_samples,
            )

            phases.append(phase)

        return phases


class AdaptiveCurriculumLearner:
    """
    Adaptive curriculum learning.

    Adjusts curriculum based on learner's performance - speeds up if doing well,
    slows down if struggling.
    """

    def __init__(
        self,
        model: nn.Module,
        curriculum: List[CurriculumPhase],
        patience: int = 3,
        performance_threshold: float = 0.7,
    ):
        self.model = model
        self.curriculum = curriculum
        self.patience = patience
        self.performance_threshold = performance_threshold

        # Current state
        self.current_phase_idx = 0
        self.epochs_in_phase = 0
        self.performance_history = []

        # Statistics
        self.total_samples_seen = 0
        self.phase_transitions = 0

    async def train(
        self,
        validation_samples: Optional[List[Sample]] = None,
    ) -> CurriculumResult:
        """
        Train through curriculum adaptively.
        """
        start_time = time.time()
        performance_curve = []

        while self.current_phase_idx < len(self.curriculum):
            phase = self.curriculum[self.current_phase_idx]

            # Train on this phase
            phase_accuracy = await self._train_phase(phase)
            performance_curve.append(phase_accuracy)

            # Check if we should advance to next phase
            should_advance = self._should_advance_phase(phase_accuracy)

            if should_advance:
                self.current_phase_idx += 1
                self.epochs_in_phase = 0
                self.phase_transitions += 1
            else:
                self.epochs_in_phase += 1

            # Check convergence
            if validation_samples and phase_accuracy > 0.95:
                break

        training_time = time.time() - start_time

        # Final evaluation
        final_accuracy = await self._evaluate(validation_samples) if validation_samples else phase_accuracy

        return CurriculumResult(
            final_accuracy=final_accuracy,
            training_time=training_time,
            phases_completed=self.current_phase_idx,
            total_samples_seen=self.total_samples_seen,
            convergence_epoch=self.epochs_in_phase,
            performance_curve=performance_curve,
        )

    async def _train_phase(
        self,
        phase: CurriculumPhase,
    ) -> float:
        """
        Train on a single curriculum phase.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=phase.learning_rate)

        # Train for phase.num_epochs
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(phase.num_epochs):
            for sample in phase.samples:
                optimizer.zero_grad()

                output = self.model(sample.features.unsqueeze(0))
                loss = F.cross_entropy(output, sample.label.unsqueeze(0))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Track accuracy
                pred = output.argmax(dim=1)
                correct += (pred == sample.label).sum().item()
                total += 1

                self.total_samples_seen += 1

        accuracy = correct / total if total > 0 else 0.0
        self.performance_history.append(accuracy)

        return accuracy

    def _should_advance_phase(
        self,
        current_accuracy: float,
    ) -> bool:
        """
        Decide whether to advance to next phase.

        Advances if:
        - Performance exceeds threshold
        - Or patience epochs elapsed with no improvement
        """
        # Check performance threshold
        if current_accuracy >= self.performance_threshold:
            return True

        # Check patience
        if self.epochs_in_phase >= self.patience:
            return True

        return False

    async def _evaluate(
        self,
        samples: List[Sample],
    ) -> float:
        """
        Evaluate model on samples.
        """
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for sample in samples:
                output = self.model(sample.features.unsqueeze(0))
                pred = output.argmax(dim=1)
                correct += (pred == sample.label).sum().item()
                total += 1

        accuracy = correct / total if total > 0 else 0.0

        return accuracy

    def get_progress(self) -> CurriculumProgress:
        """Get current curriculum progress"""
        if not self.curriculum:
            current_difficulty = 0.0
        else:
            phase = self.curriculum[self.current_phase_idx] if self.current_phase_idx < len(self.curriculum) else self.curriculum[-1]
            current_difficulty = phase.max_difficulty

        phase_completion = 0.0
        if self.current_phase_idx < len(self.curriculum):
            phase = self.curriculum[self.current_phase_idx]
            phase_completion = self.epochs_in_phase / phase.num_epochs

        return CurriculumProgress(
            current_phase=self.current_phase_idx,
            total_phases=len(self.curriculum),
            samples_seen=self.total_samples_seen,
            current_difficulty=current_difficulty,
            performance_history=self.performance_history,
            phase_completion=phase_completion,
        )


class CurriculumLearningSystem:
    """
    Unified curriculum learning system.

    Integrates difficulty scoring, curriculum generation, and adaptive training.
    """

    def __init__(
        self,
        model: nn.Module,
        difficulty_metric: DifficultyMetric = DifficultyMetric.LOSS_BASED,
        curriculum_strategy: CurriculumStrategy = CurriculumStrategy.LINEAR,
        num_phases: int = 5,
    ):
        self.model = model

        self.difficulty_scorer = DifficultyScorer(
            model=model,
            metric=difficulty_metric,
        )

        self.curriculum_generator = CurriculumGenerator(
            strategy=curriculum_strategy,
            num_phases=num_phases,
        )

        self.adaptive_learner = None  # Created during training

    async def train_with_curriculum(
        self,
        training_samples: List[Sample],
        validation_samples: Optional[List[Sample]] = None,
        base_lr: float = 0.001,
    ) -> CurriculumResult:
        """
        Complete curriculum learning pipeline:
        1. Score sample difficulties
        2. Generate curriculum
        3. Train adaptively through curriculum
        """
        # Step 1: Score difficulties
        scored_samples = self.difficulty_scorer.score_dataset(training_samples)

        # Step 2: Generate curriculum
        curriculum = self.curriculum_generator.generate_curriculum(
            scored_samples,
            base_lr=base_lr,
        )

        # Step 3: Adaptive training
        self.adaptive_learner = AdaptiveCurriculumLearner(
            model=self.model,
            curriculum=curriculum,
        )

        result = await self.adaptive_learner.train(validation_samples)

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        scorer_stats = self.difficulty_scorer.get_statistics()

        learner_stats = {}
        if self.adaptive_learner:
            progress = self.adaptive_learner.get_progress()
            learner_stats = {
                "current_phase": progress.current_phase,
                "total_phases": progress.total_phases,
                "samples_seen": progress.samples_seen,
                "current_difficulty": progress.current_difficulty,
                "phase_completion": progress.phase_completion,
            }

        return {
            **scorer_stats,
            **learner_stats,
        }


# ============================================================
# Test Functions
# ============================================================

class SimpleCurriculumModel(nn.Module):
    """Simple model for curriculum learning testing"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, num_classes: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)


async def test_curriculum_learning():
    """Test curriculum learning system"""
    print("=" * 60)
    print("Testing Curriculum Learning System")
    print("=" * 60)
    print()

    # Create model
    model = SimpleCurriculumModel(input_dim=10, hidden_dim=20, num_classes=2)

    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    num_samples = 200

    training_samples = []
    for i in range(num_samples):
        # Generate sample with varying difficulty
        if i < 100:
            # Easy samples: Clear separation
            x = torch.randn(10) + (torch.randn(1) * 2)  # High noise
            y = torch.tensor(0 if x.mean() < 0 else 1)
        else:
            # Hard samples: Ambiguous
            x = torch.randn(10) * 0.5  # Low noise, harder to classify
            y = torch.tensor(0 if x.sum() < 0 else 1)

        sample = Sample(features=x, label=y)
        training_samples.append(sample)

    # Validation set
    validation_samples = []
    for i in range(50):
        x = torch.randn(10)
        y = torch.tensor(0 if x.mean() < 0 else 1)
        sample = Sample(features=x, label=y)
        validation_samples.append(sample)

    print(f"   Training samples: {len(training_samples)}")
    print(f"   Validation samples: {len(validation_samples)}")
    print()

    # Test 1: Linear curriculum
    print("1. Testing linear curriculum...")
    system_linear = CurriculumLearningSystem(
        model=model,
        difficulty_metric=DifficultyMetric.LOSS_BASED,
        curriculum_strategy=CurriculumStrategy.LINEAR,
        num_phases=5,
    )

    result_linear = await system_linear.train_with_curriculum(
        training_samples=training_samples,
        validation_samples=validation_samples,
        base_lr=0.01,
    )

    print(f"   Final accuracy: {result_linear.final_accuracy:.1%}")
    print(f"   Training time: {result_linear.training_time:.2f}s")
    print(f"   Phases completed: {result_linear.phases_completed}")
    print(f"   Total samples seen: {result_linear.total_samples_seen}")
    print()

    # Test 2: Difficulty scoring
    print("2. Testing difficulty scoring...")
    stats = system_linear.get_statistics()
    print(f"   Samples scored: {stats['num_scored']}")
    print(f"   Mean difficulty: {stats['mean_difficulty']:.3f}")
    print(f"   Std difficulty: {stats['std_difficulty']:.3f}")
    print(f"   Min difficulty: {stats['min_difficulty']:.3f}")
    print(f"   Max difficulty: {stats['max_difficulty']:.3f}")
    print()

    # Test 3: Exponential curriculum
    print("3. Testing exponential curriculum...")
    model_exp = SimpleCurriculumModel(input_dim=10, hidden_dim=20, num_classes=2)
    system_exp = CurriculumLearningSystem(
        model=model_exp,
        difficulty_metric=DifficultyMetric.CONFIDENCE_BASED,
        curriculum_strategy=CurriculumStrategy.EXPONENTIAL,
        num_phases=5,
    )

    result_exp = await system_exp.train_with_curriculum(
        training_samples=training_samples,
        validation_samples=validation_samples,
        base_lr=0.01,
    )

    print(f"   Final accuracy: {result_exp.final_accuracy:.1%}")
    print(f"   Training time: {result_exp.training_time:.2f}s")
    print(f"   Phases completed: {result_exp.phases_completed}")
    print()

    # Test 4: Step-wise curriculum
    print("4. Testing step-wise curriculum...")
    model_step = SimpleCurriculumModel(input_dim=10, hidden_dim=20, num_classes=2)
    system_step = CurriculumLearningSystem(
        model=model_step,
        difficulty_metric=DifficultyMetric.LOSS_BASED,
        curriculum_strategy=CurriculumStrategy.STEP_WISE,
        num_phases=3,
    )

    result_step = await system_step.train_with_curriculum(
        training_samples=training_samples,
        validation_samples=validation_samples,
        base_lr=0.01,
    )

    print(f"   Final accuracy: {result_step.final_accuracy:.1%}")
    print(f"   Training time: {result_step.training_time:.2f}s")
    print(f"   Phases completed: {result_step.phases_completed}")
    print()

    # Summary
    print("=" * 60)
    print("Curriculum Comparison")
    print("=" * 60)
    print(f"Linear:      {result_linear.final_accuracy:.1%} accuracy, {result_linear.training_time:.2f}s")
    print(f"Exponential: {result_exp.final_accuracy:.1%} accuracy, {result_exp.training_time:.2f}s")
    print(f"Step-wise:   {result_step.final_accuracy:.1%} accuracy, {result_step.training_time:.2f}s")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_curriculum_learning())
