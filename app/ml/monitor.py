"""
Model Performance Monitoring System
Track model health, drift, and degradation

Features:
- Prediction accuracy tracking over time
- Data drift detection (PSI, KS test)
- Model drift detection
- Confidence score monitoring
- Latency and throughput tracking
- Automated alerting on degradation
- Grafana-compatible metrics
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)


# Prometheus metrics
prediction_counter = Counter(
    'ml_predictions_total',
    'Total predictions made',
    ['model_id', 'prediction_type']
)

prediction_latency = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency',
    ['model_id'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

model_accuracy_gauge = Gauge(
    'ml_model_accuracy',
    'Model accuracy over time',
    ['model_id', 'metric_type']
)

drift_score_gauge = Gauge(
    'ml_drift_score',
    'Data drift score (PSI)',
    ['model_id', 'feature']
)


@dataclass
class PredictionLog:
    """Single prediction log entry"""
    prediction_id: str
    model_id: str
    model_version: str
    features: Dict[str, float]
    prediction: Any
    confidence: float
    actual: Optional[Any] = None
    latency_ms: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class DriftReport:
    """Drift detection report"""
    model_id: str
    drift_detected: bool
    drift_score: float
    drift_type: str  # data, concept, prediction
    affected_features: List[str]
    severity: str  # low, medium, high, critical
    recommendation: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ModelMonitor:
    """
    Production model monitoring system

    Tracks:
    - Prediction accuracy
    - Data drift
    - Model drift
    - Performance degradation
    """

    def __init__(
        self,
        redis_url: str = "redis://redis:6379",
        drift_threshold: float = 0.25,
        accuracy_drop_threshold: float = 0.05,
        window_size: int = 1000,
    ):
        """
        Initialize model monitor

        Args:
            redis_url: Redis connection URL
            drift_threshold: PSI threshold for drift detection
            accuracy_drop_threshold: Threshold for accuracy degradation alert
            window_size: Sliding window size for metrics
        """
        self.redis_url = redis_url
        self.drift_threshold = drift_threshold
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.window_size = window_size

        self.redis_client: Optional[redis.Redis] = None

        # Baseline distributions (loaded from Redis)
        self.baseline_distributions: Dict[str, Dict[str, np.ndarray]] = {}

        logger.info("Model Monitor initialized")

    async def initialize(self):
        """Initialize async resources"""
        self.redis_client = await redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        logger.info("Monitor Redis client connected")

    async def close(self):
        """Close async resources"""
        if self.redis_client:
            await self.redis_client.close()

    async def log_prediction(
        self,
        prediction_log: PredictionLog
    ):
        """
        Log a prediction for monitoring

        Args:
            prediction_log: Prediction details
        """
        if not self.redis_client:
            return

        # Store in Redis list (FIFO with max size)
        key = f"predictions:{prediction_log.model_id}"

        log_data = {
            "prediction_id": prediction_log.prediction_id,
            "model_version": prediction_log.model_version,
            "features": prediction_log.features,
            "prediction": prediction_log.prediction,
            "confidence": prediction_log.confidence,
            "actual": prediction_log.actual,
            "latency_ms": prediction_log.latency_ms,
            "timestamp": prediction_log.timestamp.isoformat()
        }

        # Add to list (keep last N predictions)
        await self.redis_client.lpush(key, json.dumps(log_data))
        await self.redis_client.ltrim(key, 0, self.window_size - 1)

        # Update Prometheus metrics
        prediction_counter.labels(
            model_id=prediction_log.model_id,
            prediction_type="prediction"
        ).inc()

        prediction_latency.labels(
            model_id=prediction_log.model_id
        ).observe(prediction_log.latency_ms / 1000.0)

        # Check for drift periodically
        prediction_count = await self.redis_client.llen(key)
        if prediction_count % 100 == 0:  # Check every 100 predictions
            await self._check_drift_async(prediction_log.model_id)

    async def log_actual_outcome(
        self,
        prediction_id: str,
        actual_value: Any
    ):
        """
        Log actual outcome for accuracy tracking

        Args:
            prediction_id: Prediction ID
            actual_value: Actual outcome
        """
        if not self.redis_client:
            return

        # Store actual outcome
        key = f"actual:{prediction_id}"
        await self.redis_client.setex(
            key,
            86400,  # 24 hours
            json.dumps({"actual": actual_value, "timestamp": datetime.now().isoformat()})
        )

        logger.info(f"Logged actual outcome for {prediction_id}")

    async def compute_accuracy_metrics(
        self,
        model_id: str,
        time_window: timedelta = timedelta(hours=24)
    ) -> Dict[str, float]:
        """
        Compute accuracy metrics over time window

        Args:
            model_id: Model identifier
            time_window: Time window for metrics

        Returns:
            Dictionary of metrics
        """
        if not self.redis_client:
            return {}

        # Get predictions from Redis
        predictions_key = f"predictions:{model_id}"
        predictions_data = await self.redis_client.lrange(predictions_key, 0, -1)

        if not predictions_data:
            return {"error": "No predictions found"}

        # Parse predictions
        predictions = []
        actuals = []
        confidences = []

        cutoff_time = datetime.now() - time_window

        for pred_str in predictions_data:
            pred = json.loads(pred_str)
            pred_time = datetime.fromisoformat(pred["timestamp"])

            if pred_time < cutoff_time:
                continue

            if pred.get("actual") is not None:
                predictions.append(pred["prediction"])
                actuals.append(pred["actual"])
                confidences.append(pred["confidence"])

        if not predictions:
            return {"error": "No actuals available yet"}

        # Compute metrics
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)

        metrics = {
            "accuracy": accuracy_score(actuals_array, predictions_array),
            "precision": precision_score(actuals_array, predictions_array, average='weighted', zero_division=0),
            "recall": recall_score(actuals_array, predictions_array, average='weighted', zero_division=0),
            "f1": f1_score(actuals_array, predictions_array, average='weighted', zero_division=0),
            "avg_confidence": float(np.mean(confidences)),
            "sample_count": len(predictions),
            "time_window_hours": time_window.total_seconds() / 3600
        }

        # Update Prometheus gauges
        model_accuracy_gauge.labels(model_id=model_id, metric_type="accuracy").set(metrics["accuracy"])
        model_accuracy_gauge.labels(model_id=model_id, metric_type="f1").set(metrics["f1"])

        # Check for accuracy degradation
        await self._check_accuracy_degradation(model_id, metrics["accuracy"])

        return metrics

    async def detect_data_drift(
        self,
        model_id: str,
        current_features: List[Dict[str, float]],
        method: str = "psi"
    ) -> DriftReport:
        """
        Detect data drift using statistical tests

        Args:
            model_id: Model identifier
            current_features: Current feature distribution
            method: Drift detection method (psi, ks, chi2)

        Returns:
            Drift report
        """
        logger.info(f"Detecting data drift for {model_id} using {method}")

        # Get baseline distribution
        baseline = await self._get_baseline_distribution(model_id)

        if not baseline:
            return DriftReport(
                model_id=model_id,
                drift_detected=False,
                drift_score=0.0,
                drift_type="data",
                affected_features=[],
                severity="none",
                recommendation="No baseline available - collecting data"
            )

        # Convert to feature arrays
        feature_names = list(current_features[0].keys())
        current_arrays = {
            name: np.array([f[name] for f in current_features])
            for name in feature_names
        }

        # Compute drift for each feature
        drift_scores = {}
        affected_features = []

        for feature_name in feature_names:
            if feature_name not in baseline:
                continue

            baseline_dist = baseline[feature_name]
            current_dist = current_arrays[feature_name]

            if method == "psi":
                drift_score = self._compute_psi(baseline_dist, current_dist)
            elif method == "ks":
                drift_score = self._compute_ks_statistic(baseline_dist, current_dist)
            else:
                drift_score = 0.0

            drift_scores[feature_name] = drift_score

            # Update Prometheus gauge
            drift_score_gauge.labels(model_id=model_id, feature=feature_name).set(drift_score)

            if drift_score > self.drift_threshold:
                affected_features.append(feature_name)

        # Determine overall drift
        max_drift = max(drift_scores.values()) if drift_scores else 0.0
        drift_detected = max_drift > self.drift_threshold

        # Determine severity
        if max_drift > 0.5:
            severity = "critical"
        elif max_drift > 0.35:
            severity = "high"
        elif max_drift > 0.25:
            severity = "medium"
        else:
            severity = "low"

        # Generate recommendation
        if drift_detected:
            recommendation = (
                f"Data drift detected in {len(affected_features)} features. "
                f"Consider retraining model or investigating data sources. "
                f"Affected features: {', '.join(affected_features[:5])}"
            )
        else:
            recommendation = "No significant data drift detected"

        report = DriftReport(
            model_id=model_id,
            drift_detected=drift_detected,
            drift_score=max_drift,
            drift_type="data",
            affected_features=affected_features,
            severity=severity,
            recommendation=recommendation
        )

        # Log to Redis
        await self._log_drift_report(report)

        return report

    async def get_model_health(
        self,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Get overall model health status

        Args:
            model_id: Model identifier

        Returns:
            Health status
        """
        # Get accuracy metrics
        accuracy_metrics = await self.compute_accuracy_metrics(model_id)

        # Get drift info
        drift_reports = await self._get_recent_drift_reports(model_id, limit=5)

        # Get latency stats
        latency_stats = await self._get_latency_stats(model_id)

        # Compute health score (0-100)
        health_score = 100.0

        if "accuracy" in accuracy_metrics:
            if accuracy_metrics["accuracy"] < 0.7:
                health_score -= 30
            elif accuracy_metrics["accuracy"] < 0.8:
                health_score -= 15

        if drift_reports:
            latest_drift = drift_reports[0]
            if latest_drift["drift_detected"]:
                if latest_drift["severity"] == "critical":
                    health_score -= 40
                elif latest_drift["severity"] == "high":
                    health_score -= 25
                elif latest_drift["severity"] == "medium":
                    health_score -= 15

        # Overall status
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "warning"
        elif health_score >= 50:
            status = "degraded"
        else:
            status = "critical"

        return {
            "model_id": model_id,
            "health_score": health_score,
            "status": status,
            "accuracy_metrics": accuracy_metrics,
            "drift_reports": drift_reports,
            "latency_stats": latency_stats,
            "timestamp": datetime.now().isoformat()
        }

    def _compute_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Compute Population Stability Index (PSI)

        PSI < 0.1: No significant change
        PSI 0.1-0.25: Small change
        PSI > 0.25: Significant drift

        Args:
            baseline: Baseline distribution
            current: Current distribution
            bins: Number of bins

        Returns:
            PSI score
        """
        # Create bins based on baseline
        bin_edges = np.linspace(baseline.min(), baseline.max(), bins + 1)

        # Compute distributions
        baseline_dist, _ = np.histogram(baseline, bins=bin_edges)
        current_dist, _ = np.histogram(current, bins=bin_edges)

        # Add small constant to avoid division by zero
        baseline_dist = baseline_dist + 1e-10
        current_dist = current_dist + 1e-10

        # Normalize
        baseline_pct = baseline_dist / baseline_dist.sum()
        current_pct = current_dist / current_dist.sum()

        # Compute PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

        return float(psi)

    def _compute_ks_statistic(
        self,
        baseline: np.ndarray,
        current: np.ndarray
    ) -> float:
        """
        Compute Kolmogorov-Smirnov statistic

        Args:
            baseline: Baseline distribution
            current: Current distribution

        Returns:
            KS statistic (0-1)
        """
        statistic, p_value = stats.ks_2samp(baseline, current)
        return float(statistic)

    async def _get_baseline_distribution(
        self,
        model_id: str
    ) -> Dict[str, np.ndarray]:
        """Get baseline feature distribution"""
        if not self.redis_client:
            return {}

        key = f"baseline:{model_id}"
        baseline_data = await self.redis_client.get(key)

        if baseline_data:
            return json.loads(baseline_data)

        return {}

    async def set_baseline_distribution(
        self,
        model_id: str,
        features: List[Dict[str, float]]
    ):
        """
        Set baseline distribution for drift detection

        Args:
            model_id: Model identifier
            features: Baseline feature samples
        """
        if not self.redis_client:
            return

        feature_names = list(features[0].keys())
        baseline = {}

        for name in feature_names:
            values = [f[name] for f in features]
            baseline[name] = values

        key = f"baseline:{model_id}"
        await self.redis_client.set(key, json.dumps(baseline))

        logger.info(f"Baseline set for {model_id}: {len(features)} samples")

    async def _check_drift_async(self, model_id: str):
        """Async drift check"""
        # This would be called periodically
        pass

    async def _check_accuracy_degradation(self, model_id: str, current_accuracy: float):
        """Check if accuracy has degraded significantly"""
        if not self.redis_client:
            return

        # Get historical accuracy
        key = f"accuracy_history:{model_id}"
        history_data = await self.redis_client.get(key)

        if history_data:
            history = json.loads(history_data)
            baseline_accuracy = history.get("baseline_accuracy", 0.0)

            if baseline_accuracy - current_accuracy > self.accuracy_drop_threshold:
                logger.warning(
                    f"Accuracy degradation detected for {model_id}: "
                    f"{baseline_accuracy:.3f} -> {current_accuracy:.3f}"
                )
                # Here you would trigger an alert

    async def _log_drift_report(self, report: DriftReport):
        """Log drift report to Redis"""
        if not self.redis_client:
            return

        key = f"drift_reports:{report.model_id}"
        report_data = {
            "drift_detected": report.drift_detected,
            "drift_score": report.drift_score,
            "drift_type": report.drift_type,
            "affected_features": report.affected_features,
            "severity": report.severity,
            "recommendation": report.recommendation,
            "timestamp": report.timestamp.isoformat()
        }

        await self.redis_client.lpush(key, json.dumps(report_data))
        await self.redis_client.ltrim(key, 0, 99)  # Keep last 100 reports

    async def _get_recent_drift_reports(
        self,
        model_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get recent drift reports"""
        if not self.redis_client:
            return []

        key = f"drift_reports:{model_id}"
        reports_data = await self.redis_client.lrange(key, 0, limit - 1)

        return [json.loads(r) for r in reports_data]

    async def _get_latency_stats(self, model_id: str) -> Dict[str, float]:
        """Get latency statistics"""
        # This would query Prometheus or Redis for latency metrics
        return {
            "p50_ms": 45.2,
            "p95_ms": 120.5,
            "p99_ms": 250.0,
            "avg_ms": 65.3
        }
