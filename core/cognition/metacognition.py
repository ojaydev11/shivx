"""
Meta-Cognition Module - "Thinking about thinking"

Enables the AGI to:
- Monitor its own cognitive processes
- Calibrate confidence in predictions
- Quantify uncertainty ("I don't know")
- Detect when strategies are failing
- Adapt behavior based on performance

Key Capabilities:
- Self-awareness: Know what you know/don't know
- Confidence calibration: Accurate uncertainty estimates
- Strategy monitoring: Track effectiveness
- Error detection: Catch mistakes
- Adaptive learning: Adjust based on feedback

Part of ShivX Personal Empire AGI (Week 7).
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A prediction with confidence"""

    value: Any  # Predicted value
    confidence: float  # Confidence (0-1)
    uncertainty: float  # Epistemic uncertainty
    timestamp: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""

    accuracy: float
    calibration_error: float  # Expected Calibration Error
    brier_score: float  # Probability calibration
    num_predictions: int
    correct_predictions: int
    overconfident_rate: float  # % predictions overconfident
    underconfident_rate: float  # % predictions underconfident


@dataclass
class CognitiveState:
    """Current cognitive state of the system"""

    confidence_level: float  # Overall confidence
    uncertainty_level: float  # Overall uncertainty
    performance_trend: str  # improving, stable, declining
    active_strategies: List[str]
    recent_accuracy: float
    calibration_quality: str  # well_calibrated, overconfident, underconfident


class MetaCognitiveMonitor:
    """
    Monitor and regulate the AGI's cognitive processes.

    Tracks:
    - Prediction confidence and accuracy
    - Calibration (are high-confidence predictions actually correct?)
    - Uncertainty (epistemic vs. aleatoric)
    - Strategy effectiveness
    - Learning progress
    """

    def __init__(
        self,
        history_size: int = 1000,
        calibration_bins: int = 10,
    ):
        self.history_size = history_size
        self.calibration_bins = calibration_bins

        # Prediction history
        self.predictions: deque = deque(maxlen=history_size)
        self.outcomes: deque = deque(maxlen=history_size)

        # Strategy tracking
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.active_strategy: Optional[str] = None

        # Calibration tracking
        self.calibration_data: Dict[int, List[Tuple[float, bool]]] = {
            i: [] for i in range(calibration_bins)
        }

        # Performance metrics
        self.current_metrics: Optional[PerformanceMetrics] = None

        logger.info("Meta-Cognitive Monitor initialized")

    def make_prediction(
        self,
        value: Any,
        confidence: float,
        uncertainty: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Prediction:
        """
        Make a prediction with meta-cognitive awareness.

        Args:
            value: Predicted value
            confidence: Confidence in prediction (0-1)
            uncertainty: Epistemic uncertainty (0-1)
            context: Additional context

        Returns:
            Prediction object
        """
        if context is None:
            context = {}

        # Adjust confidence based on recent performance
        calibrated_confidence = self._calibrate_confidence(confidence)

        prediction = Prediction(
            value=value,
            confidence=calibrated_confidence,
            uncertainty=uncertainty,
            timestamp=datetime.now().isoformat(),
            context=context,
        )

        self.predictions.append(prediction)

        logger.debug(
            f"Prediction: {value} (conf={calibrated_confidence:.3f}, "
            f"unc={uncertainty:.3f})"
        )

        return prediction

    def record_outcome(
        self,
        predicted_value: Any,
        actual_value: Any,
        correct: bool,
    ):
        """
        Record outcome of a prediction for calibration.

        Args:
            predicted_value: What was predicted
            actual_value: What actually happened
            correct: Whether prediction was correct
        """
        self.outcomes.append({
            "predicted": predicted_value,
            "actual": actual_value,
            "correct": correct,
            "timestamp": datetime.now().isoformat(),
        })

        # Update calibration data
        if len(self.predictions) > 0:
            last_prediction = self.predictions[-1]
            confidence = last_prediction.confidence

            # Bin by confidence level
            bin_idx = min(
                int(confidence * self.calibration_bins),
                self.calibration_bins - 1
            )
            self.calibration_data[bin_idx].append((confidence, correct))

        # Update strategy performance
        if self.active_strategy is not None:
            success_rate = 1.0 if correct else 0.0
            self.strategy_performance[self.active_strategy].append(success_rate)

        logger.debug(f"Outcome recorded: correct={correct}")

    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """
        Calibrate confidence based on historical accuracy.

        If system is overconfident (high confidence, low accuracy),
        reduce confidence. If underconfident, increase it.

        Args:
            raw_confidence: Original confidence

        Returns:
            Calibrated confidence
        """
        if len(self.outcomes) < 10:
            # Not enough data yet
            return raw_confidence

        # Compute recent accuracy
        recent_outcomes = list(self.outcomes)[-50:]
        recent_accuracy = np.mean([o["correct"] for o in recent_outcomes])

        # Compute recent average confidence
        recent_predictions = list(self.predictions)[-50:]
        recent_confidence = np.mean([p.confidence for p in recent_predictions])

        # Calibration factor
        if recent_confidence > 0.1:
            calibration_factor = recent_accuracy / recent_confidence
        else:
            calibration_factor = 1.0

        # Apply calibration (with smoothing)
        alpha = 0.3  # Smoothing factor
        calibrated = alpha * (raw_confidence * calibration_factor) + (1 - alpha) * raw_confidence

        # Clamp to [0, 1]
        calibrated = np.clip(calibrated, 0.0, 1.0)

        return calibrated

    def compute_calibration_error(self) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE measures how well confidence matches accuracy.
        Lower is better (0 = perfectly calibrated).

        Returns:
            ECE score (0-1)
        """
        if not any(self.calibration_data.values()):
            return 0.0

        ece = 0.0
        total_samples = 0

        for bin_idx, bin_data in self.calibration_data.items():
            if not bin_data:
                continue

            # Average confidence in bin
            confidences = [conf for conf, _ in bin_data]
            avg_confidence = np.mean(confidences)

            # Average accuracy in bin
            corrects = [correct for _, correct in bin_data]
            avg_accuracy = np.mean(corrects)

            # Bin weight
            bin_size = len(bin_data)
            total_samples += bin_size

            # Contribution to ECE
            ece += bin_size * abs(avg_confidence - avg_accuracy)

        if total_samples > 0:
            ece /= total_samples

        return ece

    def compute_brier_score(self) -> float:
        """
        Compute Brier score for probability calibration.

        Brier score measures accuracy of probabilistic predictions.
        Lower is better (0 = perfect).

        Returns:
            Brier score (0-1)
        """
        if len(self.outcomes) == 0 or len(self.predictions) == 0:
            return 0.0

        # Match predictions to outcomes (assume same order)
        n = min(len(self.predictions), len(self.outcomes))

        brier = 0.0
        for i in range(-n, 0):  # Last n predictions
            pred_conf = self.predictions[i].confidence
            actual = 1.0 if self.outcomes[i]["correct"] else 0.0

            brier += (pred_conf - actual) ** 2

        brier /= n

        return brier

    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Compute comprehensive performance metrics.

        Returns:
            PerformanceMetrics object
        """
        if len(self.outcomes) == 0:
            return PerformanceMetrics(
                accuracy=0.0,
                calibration_error=0.0,
                brier_score=0.0,
                num_predictions=0,
                correct_predictions=0,
                overconfident_rate=0.0,
                underconfident_rate=0.0,
            )

        # Accuracy
        correct = sum(1 for o in self.outcomes if o["correct"])
        accuracy = correct / len(self.outcomes)

        # Calibration error
        ece = self.compute_calibration_error()

        # Brier score
        brier = self.compute_brier_score()

        # Over/underconfidence
        overconfident_count = 0
        underconfident_count = 0

        n = min(len(self.predictions), len(self.outcomes))
        for i in range(-n, 0):
            confidence = self.predictions[i].confidence
            correct = self.outcomes[i]["correct"]

            if confidence > 0.7 and not correct:
                overconfident_count += 1
            elif confidence < 0.3 and correct:
                underconfident_count += 1

        overconfident_rate = overconfident_count / n if n > 0 else 0.0
        underconfident_rate = underconfident_count / n if n > 0 else 0.0

        metrics = PerformanceMetrics(
            accuracy=accuracy,
            calibration_error=ece,
            brier_score=brier,
            num_predictions=len(self.outcomes),
            correct_predictions=correct,
            overconfident_rate=overconfident_rate,
            underconfident_rate=underconfident_rate,
        )

        self.current_metrics = metrics

        return metrics

    def get_cognitive_state(self) -> CognitiveState:
        """
        Get current cognitive state.

        Returns:
            CognitiveState object
        """
        metrics = self.get_performance_metrics()

        # Overall confidence (average of recent predictions)
        if self.predictions:
            recent_preds = list(self.predictions)[-20:]
            avg_confidence = np.mean([p.confidence for p in recent_preds])
        else:
            avg_confidence = 0.5

        # Overall uncertainty
        if self.predictions:
            recent_preds = list(self.predictions)[-20:]
            avg_uncertainty = np.mean([p.uncertainty for p in recent_preds])
        else:
            avg_uncertainty = 0.5

        # Performance trend
        if len(self.outcomes) >= 20:
            early_outcomes = list(self.outcomes)[-20:-10]
            late_outcomes = list(self.outcomes)[-10:]

            early_acc = np.mean([o["correct"] for o in early_outcomes])
            late_acc = np.mean([o["correct"] for o in late_outcomes])

            if late_acc > early_acc + 0.05:
                trend = "improving"
            elif late_acc < early_acc - 0.05:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        # Calibration quality
        ece = metrics.calibration_error
        if ece < 0.05:
            calibration = "well_calibrated"
        elif metrics.overconfident_rate > 0.3:
            calibration = "overconfident"
        elif metrics.underconfident_rate > 0.3:
            calibration = "underconfident"
        else:
            calibration = "moderate"

        state = CognitiveState(
            confidence_level=avg_confidence,
            uncertainty_level=avg_uncertainty,
            performance_trend=trend,
            active_strategies=list(self.strategy_performance.keys()),
            recent_accuracy=metrics.accuracy,
            calibration_quality=calibration,
        )

        return state

    def should_request_help(self) -> Tuple[bool, str]:
        """
        Determine if system should request human help.

        Returns:
            (should_request, reason)
        """
        state = self.get_cognitive_state()

        # High uncertainty
        if state.uncertainty_level > 0.8:
            return True, "High uncertainty - need expert guidance"

        # Declining performance
        if state.performance_trend == "declining" and state.recent_accuracy < 0.5:
            return True, "Performance declining - need assistance"

        # Poor calibration
        if state.calibration_quality == "overconfident" and state.recent_accuracy < 0.6:
            return True, "Overconfident with low accuracy - need recalibration"

        # All strategies failing
        if len(state.active_strategies) > 0:
            all_failing = all(
                np.mean(self.strategy_performance[s][-10:]) < 0.3
                for s in state.active_strategies
                if len(self.strategy_performance[s]) >= 10
            )
            if all_failing:
                return True, "All strategies failing - need new approach"

        return False, ""

    def select_strategy(
        self,
        available_strategies: List[str],
        task_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Select best strategy based on historical performance.

        Args:
            available_strategies: List of strategy names
            task_context: Context about the task

        Returns:
            Selected strategy name
        """
        if not available_strategies:
            raise ValueError("No strategies available")

        # If no performance data, explore (choose randomly)
        if not self.strategy_performance:
            strategy = np.random.choice(available_strategies)
            logger.info(f"Exploring: selected strategy '{strategy}' (no data)")
            return strategy

        # Compute average performance for each strategy
        strategy_scores = {}

        for strategy in available_strategies:
            if strategy in self.strategy_performance:
                recent_perf = self.strategy_performance[strategy][-20:]
                if recent_perf:
                    strategy_scores[strategy] = np.mean(recent_perf)
                else:
                    strategy_scores[strategy] = 0.0  # Untried
            else:
                strategy_scores[strategy] = 0.0  # Untried

        # Epsilon-greedy: 10% exploration
        if np.random.rand() < 0.1:
            strategy = np.random.choice(available_strategies)
            logger.info(f"Exploring: selected strategy '{strategy}'")
        else:
            # Exploit: choose best performing
            strategy = max(strategy_scores, key=strategy_scores.get)
            logger.info(
                f"Exploiting: selected strategy '{strategy}' "
                f"(score={strategy_scores[strategy]:.3f})"
            )

        self.active_strategy = strategy

        return strategy

    def explain_prediction(
        self,
        prediction: Prediction,
    ) -> str:
        """
        Generate explanation for a prediction.

        Args:
            prediction: Prediction to explain

        Returns:
            Natural language explanation
        """
        explanation = []

        explanation.append(f"Predicted: {prediction.value}")
        explanation.append(f"Confidence: {prediction.confidence:.1%}")

        # Confidence interpretation
        if prediction.confidence > 0.8:
            explanation.append("(Very confident)")
        elif prediction.confidence > 0.6:
            explanation.append("(Moderately confident)")
        elif prediction.confidence > 0.4:
            explanation.append("(Low confidence)")
        else:
            explanation.append("(Very uncertain)")

        # Uncertainty
        if prediction.uncertainty > 0.5:
            explanation.append(
                f"High uncertainty ({prediction.uncertainty:.1%}) "
                f"- limited information available"
            )

        # Calibration context
        if self.current_metrics:
            if self.current_metrics.calibration_error < 0.05:
                explanation.append(
                    "Past predictions have been well-calibrated "
                    f"({self.current_metrics.accuracy:.1%} accuracy)"
                )
            else:
                explanation.append(
                    f"Note: Recent accuracy is {self.current_metrics.accuracy:.1%}"
                )

        return " ".join(explanation)

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information for debugging.

        Returns:
            Dictionary of diagnostic metrics
        """
        metrics = self.get_performance_metrics()
        state = self.get_cognitive_state()

        return {
            "performance": {
                "accuracy": metrics.accuracy,
                "calibration_error": metrics.calibration_error,
                "brier_score": metrics.brier_score,
                "num_predictions": metrics.num_predictions,
            },
            "cognitive_state": {
                "confidence": state.confidence_level,
                "uncertainty": state.uncertainty_level,
                "trend": state.performance_trend,
                "calibration": state.calibration_quality,
            },
            "strategies": {
                name: {
                    "num_uses": len(perf),
                    "avg_performance": np.mean(perf) if perf else 0.0,
                }
                for name, perf in self.strategy_performance.items()
            },
        }


# Singleton instance
_metacog_monitor: Optional[MetaCognitiveMonitor] = None


def get_metacog_monitor() -> MetaCognitiveMonitor:
    """Get singleton meta-cognitive monitor"""
    global _metacog_monitor

    if _metacog_monitor is None:
        _metacog_monitor = MetaCognitiveMonitor()

    return _metacog_monitor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== Meta-Cognition Test ===\n")

    # Create monitor
    monitor = MetaCognitiveMonitor()

    print("Simulating predictions with varying confidence...\n")

    # Simulate predictions
    np.random.seed(42)

    for i in range(100):
        # Generate prediction
        true_value = np.random.choice([0, 1], p=[0.3, 0.7])

        # Simulate confidence (initially overconfident)
        if i < 50:
            # Overconfident phase
            raw_confidence = 0.8 if true_value == 1 else 0.6
        else:
            # Learning phase
            raw_confidence = 0.7 if true_value == 1 else 0.4

        # Add noise
        confidence = np.clip(raw_confidence + np.random.randn() * 0.1, 0, 1)
        uncertainty = np.random.uniform(0.1, 0.3)

        # Make prediction
        prediction = monitor.make_prediction(
            value=1 if confidence > 0.5 else 0,
            confidence=confidence,
            uncertainty=uncertainty,
        )

        # Record outcome
        predicted_value = prediction.value
        correct = (predicted_value == true_value)

        monitor.record_outcome(
            predicted_value=predicted_value,
            actual_value=true_value,
            correct=correct,
        )

    # Get performance metrics
    print("=== Performance Metrics ===")
    metrics = monitor.get_performance_metrics()
    print(f"Accuracy: {metrics.accuracy:.1%}")
    print(f"Calibration Error (ECE): {metrics.calibration_error:.3f}")
    print(f"Brier Score: {metrics.brier_score:.3f}")
    print(f"Predictions: {metrics.num_predictions}")
    print(f"Overconfident Rate: {metrics.overconfident_rate:.1%}")
    print(f"Underconfident Rate: {metrics.underconfident_rate:.1%}")

    # Get cognitive state
    print("\n=== Cognitive State ===")
    state = monitor.get_cognitive_state()
    print(f"Confidence Level: {state.confidence_level:.3f}")
    print(f"Uncertainty Level: {state.uncertainty_level:.3f}")
    print(f"Performance Trend: {state.performance_trend}")
    print(f"Recent Accuracy: {state.recent_accuracy:.1%}")
    print(f"Calibration Quality: {state.calibration_quality}")

    # Check if help needed
    print("\n=== Self-Awareness Check ===")
    need_help, reason = monitor.should_request_help()
    if need_help:
        print(f"Status: REQUESTING HELP")
        print(f"Reason: {reason}")
    else:
        print(f"Status: OPERATING AUTONOMOUSLY")

    # Test strategy selection
    print("\n=== Strategy Selection ===")
    strategies = ["strategy_a", "strategy_b", "strategy_c"]

    # Simulate strategy performance
    for j in range(30):
        selected = monitor.select_strategy(strategies)

        # Simulate outcome (strategy_b is best)
        if selected == "strategy_b":
            success = np.random.rand() < 0.8
        else:
            success = np.random.rand() < 0.4

        monitor.record_outcome(
            predicted_value=selected,
            actual_value="strategy_b" if success else "failed",
            correct=success,
        )

    # Final strategy selection
    final_strategy = monitor.select_strategy(strategies)
    print(f"\nFinal strategy selection: {final_strategy}")

    # Strategy performance summary
    print("\nStrategy Performance:")
    for strategy in strategies:
        if strategy in monitor.strategy_performance:
            perf = monitor.strategy_performance[strategy]
            print(f"  {strategy}: {np.mean(perf):.1%} ({len(perf)} uses)")

    # Test explanation
    print("\n=== Prediction Explanation ===")
    test_pred = monitor.make_prediction(
        value="increase_engagement",
        confidence=0.75,
        uncertainty=0.2,
    )
    explanation = monitor.explain_prediction(test_pred)
    print(explanation)

    print("\n=== Meta-Cognition Ready ===")
    print("The system can now:")
    print("- Monitor its own predictions")
    print("- Calibrate confidence accurately")
    print("- Detect when it needs help")
    print("- Select strategies adaptively")
    print("- Explain its reasoning")
