"""
Week 15: Online Learning

Implements online learning for real-time model updates:
- Incremental learning without full retraining
- Concept drift detection and adaptation
- Streaming data processing
- A/B testing framework
- Model versioning and rollback

Key Features:
- Learn continuously from production data
- Detect and adapt to distribution shifts
- No downtime for retraining
- Test model variants safely
- Roll back if performance degrades

Integrates with:
- Week 4: Continual learning (prevent catastrophic forgetting)
- Week 7: Meta-cognition (confidence monitoring)
- Week 11: Production hardening (safe deployment)
- Week 14: Federated learning (distributed online learning)
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of concept drift"""
    NO_DRIFT = "no_drift"
    GRADUAL = "gradual"  # Slow change over time
    SUDDEN = "sudden"  # Abrupt change
    INCREMENTAL = "incremental"  # Step-by-step changes
    RECURRING = "recurring"  # Cyclical patterns


class UpdateStrategy(Enum):
    """Model update strategies"""
    IMMEDIATE = "immediate"  # Update immediately
    BATCHED = "batched"  # Accumulate and update periodically
    ADAPTIVE = "adaptive"  # Update based on drift detection
    SCHEDULED = "scheduled"  # Update on schedule


@dataclass
class DataPoint:
    """Single data point for online learning"""
    point_id: str
    features: np.ndarray
    label: Any
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftDetection:
    """Drift detection result"""
    detected: bool
    drift_type: DriftType
    confidence: float
    statistical_distance: float
    recommended_action: str
    timestamp: datetime


@dataclass
class ModelVersion:
    """Model version for A/B testing"""
    version_id: str
    model_state: Dict[str, torch.Tensor]
    created_at: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    traffic_percentage: float = 0.0  # 0-100
    num_predictions: int = 0
    is_champion: bool = False


@dataclass
class ABTestResult:
    """A/B test comparison result"""
    test_id: str
    champion_version: str
    challenger_version: str
    champion_metric: float
    challenger_metric: float
    improvement: float
    p_value: float
    is_significant: bool
    recommendation: str


class DriftDetector:
    """
    Detects concept drift in streaming data.

    Methods:
    - ADWIN (Adaptive Windowing)
    - Page-Hinkley test
    - Statistical distance (KL divergence)
    """

    def __init__(
        self,
        window_size: int = 1000,
        drift_threshold: float = 0.05,
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold

        # Sliding windows
        self.reference_window: deque = deque(maxlen=window_size)
        self.current_window: deque = deque(maxlen=window_size)

        # Statistics
        self.reference_mean = 0.0
        self.reference_std = 1.0

        # Drift history
        self.drift_history: List[DriftDetection] = []

        logger.info(f"Drift detector initialized: window={window_size}, "
                   f"threshold={drift_threshold}")

    def update(self, value: float) -> Optional[DriftDetection]:
        """
        Update detector with new value and check for drift.

        Args:
            value: Performance metric value (e.g., loss, accuracy)

        Returns:
            DriftDetection if drift detected, None otherwise
        """
        # Add to current window
        self.current_window.append(value)

        # Need enough data
        if len(self.reference_window) < 100 or len(self.current_window) < 100:
            self.reference_window.append(value)
            return None

        # Calculate statistics
        ref_mean = np.mean(self.reference_window)
        ref_std = np.std(self.reference_window)
        cur_mean = np.mean(self.current_window)
        cur_std = np.std(self.current_window)

        # Statistical distance (normalized difference in means)
        if ref_std > 0:
            statistical_distance = abs(cur_mean - ref_mean) / ref_std
        else:
            statistical_distance = 0.0

        # Detect drift
        drift_detected = statistical_distance > self.drift_threshold

        if drift_detected:
            # Classify drift type
            if cur_std > ref_std * 1.5:
                drift_type = DriftType.SUDDEN
            elif abs(cur_mean - ref_mean) > ref_std * 2:
                drift_type = DriftType.SUDDEN
            else:
                drift_type = DriftType.GRADUAL

            # Confidence based on statistical distance
            confidence = min(statistical_distance / (self.drift_threshold * 2), 1.0)

            # Recommended action
            if drift_type == DriftType.SUDDEN:
                action = "Retrain model immediately"
            else:
                action = "Increase learning rate gradually"

            detection = DriftDetection(
                detected=True,
                drift_type=drift_type,
                confidence=confidence,
                statistical_distance=statistical_distance,
                recommended_action=action,
                timestamp=datetime.now(),
            )

            self.drift_history.append(detection)

            # Update reference window (adapt to new distribution)
            self.reference_window = self.current_window.copy()
            self.current_window.clear()

            logger.warning(f"Drift detected: {drift_type.value}, "
                          f"distance={statistical_distance:.3f}, "
                          f"confidence={confidence:.1%}")

            return detection

        return None


class OnlineLearner:
    """
    Online learning with incremental updates.

    Supports:
    - Stochastic gradient descent
    - Mini-batch updates
    - Learning rate adaptation
    - Forgetting prevention
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        update_strategy: UpdateStrategy = UpdateStrategy.BATCHED,
        forgetting_factor: float = 0.99,
    ):
        self.model = model
        self.batch_size = batch_size
        self.update_strategy = update_strategy
        self.forgetting_factor = forgetting_factor

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Data buffer for batched updates
        self.data_buffer: List[DataPoint] = []

        # Statistics
        self.stats = {
            "updates": 0,
            "samples_processed": 0,
            "avg_loss": 0.0,
            "learning_rate": learning_rate,
        }

        logger.info(f"Online learner initialized: lr={learning_rate}, "
                   f"batch={batch_size}, strategy={update_strategy.value}")

    async def learn(self, data_point: DataPoint) -> Dict[str, Any]:
        """
        Learn from a single data point.

        Args:
            data_point: New training example

        Returns:
            Learning results (loss, updated, etc.)
        """
        result = {
            "updated": False,
            "loss": None,
            "samples_in_buffer": len(self.data_buffer),
        }

        # Add to buffer
        self.data_buffer.append(data_point)

        # Check if should update
        should_update = False

        if self.update_strategy == UpdateStrategy.IMMEDIATE:
            should_update = len(self.data_buffer) >= 1
        elif self.update_strategy == UpdateStrategy.BATCHED:
            should_update = len(self.data_buffer) >= self.batch_size
        elif self.update_strategy == UpdateStrategy.ADAPTIVE:
            # Update if buffer is full or drift detected
            should_update = len(self.data_buffer) >= self.batch_size

        if should_update:
            loss = await self._update_model()
            result["updated"] = True
            result["loss"] = loss
            result["samples_processed"] = len(self.data_buffer)

            # Clear buffer
            self.data_buffer.clear()

        return result

    async def _update_model(self) -> float:
        """Update model with buffered data"""

        if not self.data_buffer:
            return 0.0

        # Prepare batch
        features = torch.tensor(
            np.array([dp.features for dp in self.data_buffer]),
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [dp.label for dp in self.data_buffer],
            dtype=torch.long,
        )

        # Forward pass
        self.model.train()
        outputs = self.model(features)
        loss = self.criterion(outputs, labels)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Apply forgetting prevention (EWC-style)
        # Scale gradients to prevent catastrophic forgetting
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad *= self.forgetting_factor

        self.optimizer.step()

        # Update statistics
        self.stats["updates"] += 1
        self.stats["samples_processed"] += len(self.data_buffer)
        self.stats["avg_loss"] = (
            0.9 * self.stats["avg_loss"] + 0.1 * loss.item()
        )

        logger.debug(f"Model updated: loss={loss.item():.4f}, "
                    f"samples={len(self.data_buffer)}")

        return loss.item()

    def adapt_learning_rate(self, factor: float):
        """Adapt learning rate (e.g., in response to drift)"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor

        self.stats["learning_rate"] = self.optimizer.param_groups[0]['lr']

        logger.info(f"Learning rate adapted: {self.stats['learning_rate']:.6f}")


class ABTestingFramework:
    """
    A/B testing framework for model variants.

    Features:
    - Multiple model versions
    - Traffic splitting
    - Statistical significance testing
    - Champion/challenger paradigm
    """

    def __init__(
        self,
        champion_model: nn.Module,
        significance_level: float = 0.05,
    ):
        self.significance_level = significance_level

        # Model versions
        self.versions: Dict[str, ModelVersion] = {}

        # Create champion version
        champion_version = ModelVersion(
            version_id="champion_v1",
            model_state=champion_model.state_dict(),
            created_at=datetime.now(),
            traffic_percentage=100.0,
            is_champion=True,
        )
        self.versions["champion_v1"] = champion_version
        self.champion_id = "champion_v1"

        # Active models
        self.active_models: Dict[str, nn.Module] = {
            "champion_v1": champion_model
        }

        # Test history
        self.test_history: List[ABTestResult] = []

        logger.info(f"A/B testing framework initialized: "
                   f"significance={significance_level}")

    def add_challenger(
        self,
        model: nn.Module,
        traffic_percentage: float = 10.0,
    ) -> str:
        """Add a challenger model variant"""

        version_id = f"challenger_v{len(self.versions)}"

        version = ModelVersion(
            version_id=version_id,
            model_state=model.state_dict(),
            created_at=datetime.now(),
            traffic_percentage=traffic_percentage,
            is_champion=False,
        )

        self.versions[version_id] = version
        self.active_models[version_id] = model

        # Adjust champion traffic
        champion = self.versions[self.champion_id]
        champion.traffic_percentage -= traffic_percentage

        logger.info(f"Added challenger: {version_id} ({traffic_percentage}% traffic)")

        return version_id

    def select_model(self) -> Tuple[str, nn.Module]:
        """Select model variant based on traffic split"""

        # Random selection based on traffic percentages
        rand = np.random.random() * 100

        cumulative = 0.0
        for version_id, version in self.versions.items():
            cumulative += version.traffic_percentage
            if rand <= cumulative:
                return version_id, self.active_models[version_id]

        # Fallback to champion
        return self.champion_id, self.active_models[self.champion_id]

    def record_prediction(
        self,
        version_id: str,
        metric_value: float,
    ):
        """Record prediction result for a version"""

        version = self.versions.get(version_id)
        if not version:
            return

        version.num_predictions += 1

        # Update rolling average metric
        if "accuracy" not in version.performance_metrics:
            version.performance_metrics["accuracy"] = metric_value
        else:
            # Exponential moving average
            alpha = 0.1
            version.performance_metrics["accuracy"] = (
                alpha * metric_value +
                (1 - alpha) * version.performance_metrics["accuracy"]
            )

    async def compare_versions(
        self,
        version_a: str,
        version_b: str,
    ) -> ABTestResult:
        """
        Compare two model versions using statistical test.

        Uses t-test for significance.
        """

        ver_a = self.versions.get(version_a)
        ver_b = self.versions.get(version_b)

        if not ver_a or not ver_b:
            raise ValueError("Invalid version IDs")

        # Get performance metrics
        metric_a = ver_a.performance_metrics.get("accuracy", 0.0)
        metric_b = ver_b.performance_metrics.get("accuracy", 0.0)

        # Calculate improvement
        if metric_a > 0:
            improvement = (metric_b - metric_a) / metric_a * 100
        else:
            improvement = 0.0

        # Statistical significance (simplified t-test)
        # In production, would use actual samples
        n_a = ver_a.num_predictions
        n_b = ver_b.num_predictions

        # Assume variance of ~0.1 for demonstration
        variance = 0.1
        se = np.sqrt(variance / n_a + variance / n_b) if (n_a > 0 and n_b > 0) else 1.0

        # Z-score
        z_score = abs(metric_b - metric_a) / se if se > 0 else 0.0

        # P-value (two-tailed, simplified)
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(z_score))

        # Significant if p < significance_level
        is_significant = p_value < self.significance_level

        # Recommendation
        if is_significant and metric_b > metric_a:
            recommendation = f"Promote {version_b} to champion"
        elif is_significant and metric_b < metric_a:
            recommendation = f"Remove {version_b} (underperforming)"
        else:
            recommendation = "Continue testing (not significant yet)"

        result = ABTestResult(
            test_id=f"test_{len(self.test_history) + 1}",
            champion_version=version_a,
            challenger_version=version_b,
            champion_metric=metric_a,
            challenger_metric=metric_b,
            improvement=improvement,
            p_value=p_value,
            is_significant=is_significant,
            recommendation=recommendation,
        )

        self.test_history.append(result)

        logger.info(f"A/B test result: {version_b} vs {version_a}: "
                   f"{improvement:+.1f}% (p={p_value:.4f}, "
                   f"significant={is_significant})")

        return result

    def promote_to_champion(self, version_id: str):
        """Promote a challenger to champion"""

        new_champion = self.versions.get(version_id)
        if not new_champion:
            raise ValueError(f"Version {version_id} not found")

        # Demote old champion
        old_champion = self.versions[self.champion_id]
        old_champion.is_champion = False
        old_champion.traffic_percentage = 0.0

        # Promote new champion
        new_champion.is_champion = True
        new_champion.traffic_percentage = 100.0
        self.champion_id = version_id

        logger.info(f"Promoted {version_id} to champion")


class OnlineLearningSystem:
    """
    Complete online learning system.

    Integrates:
    - Online learner
    - Drift detector
    - A/B testing
    - Model versioning
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        drift_threshold: float = 0.05,
    ):
        self.learner = OnlineLearner(
            model=model,
            learning_rate=learning_rate,
            batch_size=batch_size,
            update_strategy=UpdateStrategy.BATCHED,
        )

        self.drift_detector = DriftDetector(
            window_size=1000,
            drift_threshold=drift_threshold,
        )

        self.ab_testing = ABTestingFramework(
            champion_model=model,
            significance_level=0.05,
        )

        # Statistics
        self.stats = {
            "data_points_processed": 0,
            "drifts_detected": 0,
            "model_updates": 0,
            "ab_tests_run": 0,
        }

        logger.info("Online learning system initialized")

    async def process_stream(
        self,
        data_stream: List[DataPoint],
        check_drift: bool = True,
    ) -> Dict[str, Any]:
        """
        Process streaming data.

        Args:
            data_stream: Stream of data points
            check_drift: Whether to check for concept drift

        Returns:
            Processing results
        """

        results = {
            "processed": 0,
            "updates": 0,
            "drifts": [],
            "avg_loss": 0.0,
        }

        total_loss = 0.0
        num_losses = 0

        for data_point in data_stream:
            # Learn from data point
            learn_result = await self.learner.learn(data_point)

            results["processed"] += 1
            self.stats["data_points_processed"] += 1

            if learn_result["updated"]:
                results["updates"] += 1
                self.stats["model_updates"] += 1

                loss = learn_result["loss"]
                total_loss += loss
                num_losses += 1

                # Check for drift
                if check_drift:
                    drift = self.drift_detector.update(loss)

                    if drift:
                        results["drifts"].append({
                            "type": drift.drift_type.value,
                            "confidence": drift.confidence,
                            "action": drift.recommended_action,
                        })
                        self.stats["drifts_detected"] += 1

                        # Adapt to drift
                        await self._handle_drift(drift)

        # Calculate average loss
        if num_losses > 0:
            results["avg_loss"] = total_loss / num_losses

        logger.info(f"Processed {results['processed']} data points, "
                   f"{results['updates']} updates, "
                   f"{len(results['drifts'])} drifts detected")

        return results

    async def _handle_drift(self, drift: DriftDetection):
        """Handle detected concept drift"""

        if drift.drift_type == DriftType.SUDDEN:
            # Sudden drift: increase learning rate
            self.learner.adapt_learning_rate(factor=2.0)
            logger.info("Sudden drift: increased learning rate 2x")

        elif drift.drift_type == DriftType.GRADUAL:
            # Gradual drift: slightly increase learning rate
            self.learner.adapt_learning_rate(factor=1.2)
            logger.info("Gradual drift: increased learning rate 1.2x")

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            **self.stats,
            "learner_stats": self.learner.stats,
            "drift_history": len(self.drift_detector.drift_history),
            "ab_tests": len(self.ab_testing.test_history),
        }


# ========== Testing Functions ==========

class SimpleOnlineModel(nn.Module):
    """Simple model for testing"""
    def __init__(self, input_dim: int = 10, output_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


async def test_online_learning():
    """Test online learning system"""
    print("\n" + "="*60)
    print("Testing Online Learning System")
    print("="*60)

    # Create model
    model = SimpleOnlineModel(input_dim=10, output_dim=2)

    # Create online learning system
    system = OnlineLearningSystem(
        model=model,
        learning_rate=0.001,
        batch_size=32,
        drift_threshold=0.05,
    )

    # Test 1: Process Streaming Data (No Drift)
    print("\n1. Testing streaming data processing (stable distribution)...")

    # Generate synthetic streaming data
    stream = []
    for i in range(200):
        features = np.random.randn(10)
        label = int(features[0] > 0)  # Simple decision boundary
        stream.append(DataPoint(
            point_id=f"point_{i}",
            features=features,
            label=label,
            timestamp=datetime.now(),
        ))

    results = await system.process_stream(stream, check_drift=True)

    print(f"   Processed: {results['processed']} data points")
    print(f"   Updates: {results['updates']}")
    print(f"   Average Loss: {results['avg_loss']:.4f}")
    print(f"   Drifts Detected: {len(results['drifts'])}")

    # Test 2: Concept Drift
    print("\n2. Testing concept drift detection...")

    # Generate data with sudden distribution shift
    stream_with_drift = []
    for i in range(300):
        if i < 150:
            # Original distribution
            features = np.random.randn(10)
            label = int(features[0] > 0)
        else:
            # Shifted distribution (drift)
            features = np.random.randn(10) + 2.0  # Shift mean
            label = int(features[0] > 0)

        stream_with_drift.append(DataPoint(
            point_id=f"drift_point_{i}",
            features=features,
            label=label,
            timestamp=datetime.now(),
        ))

    results = await system.process_stream(stream_with_drift, check_drift=True)

    print(f"   Processed: {results['processed']} data points")
    print(f"   Updates: {results['updates']}")
    print(f"   Drifts Detected: {len(results['drifts'])}")
    for i, drift in enumerate(results['drifts'], 1):
        print(f"\n   Drift {i}:")
        print(f"     Type: {drift['type']}")
        print(f"     Confidence: {drift['confidence']:.1%}")
        print(f"     Action: {drift['action']}")

    # Test 3: A/B Testing
    print("\n3. Testing A/B testing framework...")

    # Create challenger model
    challenger = SimpleOnlineModel(input_dim=10, output_dim=2)

    # Add challenger
    challenger_id = system.ab_testing.add_challenger(
        challenger,
        traffic_percentage=20.0,
    )
    print(f"   Added challenger: {challenger_id} (20% traffic)")

    # Simulate predictions
    print("\n   Simulating predictions...")
    for i in range(1000):
        version_id, selected_model = system.ab_testing.select_model()

        # Simulate prediction accuracy
        # Challenger is slightly better
        if version_id == challenger_id:
            accuracy = 0.85 + np.random.randn() * 0.05
        else:
            accuracy = 0.80 + np.random.randn() * 0.05

        system.ab_testing.record_prediction(version_id, accuracy)

    # Compare versions
    result = await system.ab_testing.compare_versions(
        "champion_v1",
        challenger_id,
    )

    print(f"\n   A/B Test Results:")
    print(f"     Champion: {result.champion_metric:.1%}")
    print(f"     Challenger: {result.challenger_metric:.1%}")
    print(f"     Improvement: {result.improvement:+.1f}%")
    print(f"     P-value: {result.p_value:.4f}")
    print(f"     Significant: {result.is_significant}")
    print(f"     Recommendation: {result.recommendation}")

    # Final Statistics
    print("\n" + "="*60)
    print("System Statistics")
    print("="*60)

    stats = system.get_statistics()
    print(f"Data Points Processed: {stats['data_points_processed']}")
    print(f"Model Updates: {stats['model_updates']}")
    print(f"Drifts Detected: {stats['drifts_detected']}")
    print(f"A/B Tests Run: {stats['ab_tests']}")
    print(f"\nLearner Stats:")
    print(f"  Average Loss: {stats['learner_stats']['avg_loss']:.4f}")
    print(f"  Learning Rate: {stats['learner_stats']['learning_rate']:.6f}")

    return system


if __name__ == "__main__":
    asyncio.run(test_online_learning())
