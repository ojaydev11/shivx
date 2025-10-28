"""
ML Predictions Caching Service
Caches ML model predictions, inference results, and feature engineering outputs
"""

import json
import hashlib
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

import redis.asyncio as aioredis
from prometheus_client import Counter, Histogram

from app.cache import make_cache_key, redis_operation_duration


logger = logging.getLogger(__name__)


# ============================================================================
# Prometheus Metrics
# ============================================================================

ml_cache_hits = Counter(
    "ml_cache_hits_total",
    "Total number of ML cache hits",
    ["prediction_type"]
)

ml_cache_misses = Counter(
    "ml_cache_misses_total",
    "Total number of ML cache misses",
    ["prediction_type"]
)

ml_cache_invalidations = Counter(
    "ml_cache_invalidations_total",
    "Total number of ML cache invalidations",
    ["reason"]
)

ml_inference_duration = Histogram(
    "ml_inference_duration_seconds",
    "ML model inference duration",
    ["model_type"]
)


# ============================================================================
# ML Predictions Cache Service
# ============================================================================

class MLPredictionCache:
    """
    Caches ML predictions with confidence thresholds and model versioning

    TTL Strategy:
    - ML predictions: 30 seconds (short-lived, market changes quickly)
    - Feature engineering: 1 minute (slightly longer)
    - Ensemble predictions: 30 seconds
    - Model metadata: 1 hour (static during runtime)

    Features:
    - Confidence threshold filtering (only cache high-confidence predictions)
    - Model version tracking (invalidate on model update)
    - Feature hashing for cache keys
    - Separate caching for ensemble predictions
    """

    # Cache TTLs (seconds)
    TTL_PREDICTION = 30  # 30 seconds
    TTL_FEATURES = 60  # 1 minute
    TTL_ENSEMBLE = 30  # 30 seconds
    TTL_MODEL_METADATA = 3600  # 1 hour

    # Cache version
    CACHE_VERSION = 1

    # Confidence threshold (only cache predictions above this)
    MIN_CONFIDENCE = 0.7

    def __init__(self, redis: Optional[aioredis.Redis] = None):
        self.redis = redis

    # ========================================================================
    # Helper Methods
    # ========================================================================

    @staticmethod
    def _hash_features(features: Dict[str, Any]) -> str:
        """
        Create a hash from feature dictionary for cache key

        Args:
            features: Feature dictionary

        Returns:
            Hash string
        """
        # Sort keys for consistent hashing
        sorted_features = json.dumps(features, sort_keys=True)
        return hashlib.sha256(sorted_features.encode()).hexdigest()[:16]

    @staticmethod
    def _meets_confidence_threshold(prediction: Dict[str, Any], threshold: float) -> bool:
        """
        Check if prediction meets confidence threshold

        Args:
            prediction: Prediction data
            threshold: Minimum confidence threshold

        Returns:
            True if meets threshold
        """
        confidence = prediction.get("confidence", 0.0)
        return confidence >= threshold

    # ========================================================================
    # ML Prediction Caching
    # ========================================================================

    async def get_prediction(
        self,
        model_name: str,
        model_version: str,
        token_pair: str,
        features_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached ML prediction

        Args:
            model_name: Name of ML model
            model_version: Model version
            token_pair: Token pair
            features_hash: Hash of input features

        Returns:
            Prediction data or None if not cached
        """
        if not self.redis:
            return None

        cache_key = make_cache_key(
            "ml", "prediction", model_name, model_version,
            token_pair, features_hash, f"v{self.CACHE_VERSION}"
        )

        try:
            with redis_operation_duration.labels(operation="get").time():
                cached = await self.redis.get(cache_key)

            if cached:
                ml_cache_hits.labels(prediction_type=model_name).inc()
                return json.loads(cached)
            else:
                ml_cache_misses.labels(prediction_type=model_name).inc()
                return None

        except Exception as e:
            logger.error(f"Error getting cached prediction: {e}")
            ml_cache_misses.labels(prediction_type=model_name).inc()
            return None

    async def set_prediction(
        self,
        model_name: str,
        model_version: str,
        token_pair: str,
        features: Dict[str, Any],
        prediction: Dict[str, Any],
        ttl: Optional[int] = None,
        min_confidence: Optional[float] = None
    ) -> bool:
        """
        Cache ML prediction (only if confidence is high enough)

        Args:
            model_name: Name of ML model
            model_version: Model version
            token_pair: Token pair
            features: Input features
            prediction: Prediction data
            ttl: Custom TTL (default: TTL_PREDICTION)
            min_confidence: Minimum confidence to cache (default: MIN_CONFIDENCE)

        Returns:
            True if cached successfully
        """
        if not self.redis:
            return False

        # Check confidence threshold
        min_confidence = min_confidence or self.MIN_CONFIDENCE
        if not self._meets_confidence_threshold(prediction, min_confidence):
            logger.debug(
                f"Prediction confidence too low to cache: "
                f"{prediction.get('confidence', 0.0)} < {min_confidence}"
            )
            return False

        features_hash = self._hash_features(features)
        cache_key = make_cache_key(
            "ml", "prediction", model_name, model_version,
            token_pair, features_hash, f"v{self.CACHE_VERSION}"
        )
        ttl = ttl or self.TTL_PREDICTION

        try:
            prediction_data = {
                **prediction,
                "model_name": model_name,
                "model_version": model_version,
                "token_pair": token_pair,
                "features_hash": features_hash,
                "cached_at": datetime.utcnow().isoformat(),
            }

            with redis_operation_duration.labels(operation="set").time():
                await self.redis.setex(cache_key, ttl, json.dumps(prediction_data))

            return True

        except Exception as e:
            logger.error(f"Error caching prediction: {e}")
            return False

    # ========================================================================
    # Feature Engineering Caching
    # ========================================================================

    async def get_features(
        self,
        token_pair: str,
        timeframe: str,
        feature_set: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached feature engineering output

        Args:
            token_pair: Token pair
            timeframe: Timeframe
            feature_set: Feature set name

        Returns:
            Feature data or None if not cached
        """
        if not self.redis:
            return None

        cache_key = make_cache_key(
            "ml", "features", token_pair, timeframe, feature_set, f"v{self.CACHE_VERSION}"
        )

        try:
            with redis_operation_duration.labels(operation="get").time():
                cached = await self.redis.get(cache_key)

            if cached:
                ml_cache_hits.labels(prediction_type="features").inc()
                return json.loads(cached)
            else:
                ml_cache_misses.labels(prediction_type="features").inc()
                return None

        except Exception as e:
            logger.error(f"Error getting cached features: {e}")
            ml_cache_misses.labels(prediction_type="features").inc()
            return None

    async def set_features(
        self,
        token_pair: str,
        timeframe: str,
        feature_set: str,
        features: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache feature engineering output

        Args:
            token_pair: Token pair
            timeframe: Timeframe
            feature_set: Feature set name
            features: Computed features
            ttl: Custom TTL (default: TTL_FEATURES)

        Returns:
            True if cached successfully
        """
        if not self.redis:
            return False

        cache_key = make_cache_key(
            "ml", "features", token_pair, timeframe, feature_set, f"v{self.CACHE_VERSION}"
        )
        ttl = ttl or self.TTL_FEATURES

        try:
            features_data = {
                **features,
                "token_pair": token_pair,
                "timeframe": timeframe,
                "feature_set": feature_set,
                "cached_at": datetime.utcnow().isoformat(),
            }

            with redis_operation_duration.labels(operation="set").time():
                await self.redis.setex(cache_key, ttl, json.dumps(features_data))

            return True

        except Exception as e:
            logger.error(f"Error caching features: {e}")
            return False

    # ========================================================================
    # Ensemble Prediction Caching
    # ========================================================================

    async def get_ensemble_prediction(
        self,
        ensemble_name: str,
        ensemble_version: str,
        token_pair: str,
        features_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached ensemble prediction

        Args:
            ensemble_name: Name of ensemble
            ensemble_version: Ensemble version
            token_pair: Token pair
            features_hash: Hash of input features

        Returns:
            Ensemble prediction or None if not cached
        """
        if not self.redis:
            return None

        cache_key = make_cache_key(
            "ml", "ensemble", ensemble_name, ensemble_version,
            token_pair, features_hash, f"v{self.CACHE_VERSION}"
        )

        try:
            with redis_operation_duration.labels(operation="get").time():
                cached = await self.redis.get(cache_key)

            if cached:
                ml_cache_hits.labels(prediction_type="ensemble").inc()
                return json.loads(cached)
            else:
                ml_cache_misses.labels(prediction_type="ensemble").inc()
                return None

        except Exception as e:
            logger.error(f"Error getting cached ensemble prediction: {e}")
            ml_cache_misses.labels(prediction_type="ensemble").inc()
            return None

    async def set_ensemble_prediction(
        self,
        ensemble_name: str,
        ensemble_version: str,
        token_pair: str,
        features: Dict[str, Any],
        prediction: Dict[str, Any],
        model_predictions: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache ensemble prediction

        Args:
            ensemble_name: Name of ensemble
            ensemble_version: Ensemble version
            token_pair: Token pair
            features: Input features
            prediction: Final ensemble prediction
            model_predictions: Individual model predictions
            ttl: Custom TTL (default: TTL_ENSEMBLE)

        Returns:
            True if cached successfully
        """
        if not self.redis:
            return False

        features_hash = self._hash_features(features)
        cache_key = make_cache_key(
            "ml", "ensemble", ensemble_name, ensemble_version,
            token_pair, features_hash, f"v{self.CACHE_VERSION}"
        )
        ttl = ttl or self.TTL_ENSEMBLE

        try:
            ensemble_data = {
                "ensemble_name": ensemble_name,
                "ensemble_version": ensemble_version,
                "token_pair": token_pair,
                "features_hash": features_hash,
                "prediction": prediction,
                "model_predictions": model_predictions,
                "num_models": len(model_predictions),
                "cached_at": datetime.utcnow().isoformat(),
            }

            with redis_operation_duration.labels(operation="set").time():
                await self.redis.setex(cache_key, ttl, json.dumps(ensemble_data))

            return True

        except Exception as e:
            logger.error(f"Error caching ensemble prediction: {e}")
            return False

    # ========================================================================
    # Model Metadata Caching
    # ========================================================================

    async def get_model_metadata(
        self,
        model_name: str,
        model_version: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached model metadata

        Args:
            model_name: Name of model
            model_version: Model version

        Returns:
            Model metadata or None if not cached
        """
        if not self.redis:
            return None

        cache_key = make_cache_key(
            "ml", "metadata", model_name, model_version, f"v{self.CACHE_VERSION}"
        )

        try:
            with redis_operation_duration.labels(operation="get").time():
                cached = await self.redis.get(cache_key)

            if cached:
                ml_cache_hits.labels(prediction_type="metadata").inc()
                return json.loads(cached)
            else:
                ml_cache_misses.labels(prediction_type="metadata").inc()
                return None

        except Exception as e:
            logger.error(f"Error getting cached model metadata: {e}")
            ml_cache_misses.labels(prediction_type="metadata").inc()
            return None

    async def set_model_metadata(
        self,
        model_name: str,
        model_version: str,
        metadata: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache model metadata

        Args:
            model_name: Name of model
            model_version: Model version
            metadata: Model metadata (architecture, params, etc.)
            ttl: Custom TTL (default: TTL_MODEL_METADATA)

        Returns:
            True if cached successfully
        """
        if not self.redis:
            return False

        cache_key = make_cache_key(
            "ml", "metadata", model_name, model_version, f"v{self.CACHE_VERSION}"
        )
        ttl = ttl or self.TTL_MODEL_METADATA

        try:
            metadata_data = {
                **metadata,
                "model_name": model_name,
                "model_version": model_version,
                "cached_at": datetime.utcnow().isoformat(),
            }

            with redis_operation_duration.labels(operation="set").time():
                await self.redis.setex(cache_key, ttl, json.dumps(metadata_data))

            return True

        except Exception as e:
            logger.error(f"Error caching model metadata: {e}")
            return False

    # ========================================================================
    # Cache Invalidation (on model retrain)
    # ========================================================================

    async def invalidate_model_predictions(
        self,
        model_name: str,
        model_version: Optional[str] = None
    ) -> int:
        """
        Invalidate all predictions for a model (e.g., after retraining)

        Args:
            model_name: Name of model
            model_version: Model version (if None, invalidate all versions)

        Returns:
            Number of keys invalidated
        """
        if not self.redis:
            return 0

        if model_version:
            pattern = make_cache_key("ml", "prediction", model_name, model_version, "*")
        else:
            pattern = make_cache_key("ml", "prediction", model_name, "*")

        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self.redis.delete(*keys)
                ml_cache_invalidations.labels(reason="model_retrain").inc()
                logger.info(f"Invalidated {deleted} predictions for model {model_name}")
                return deleted
            return 0

        except Exception as e:
            logger.error(f"Error invalidating model predictions: {e}")
            return 0

    async def invalidate_all_predictions(self, reason: str = "manual") -> int:
        """
        Invalidate all ML predictions

        Args:
            reason: Reason for invalidation

        Returns:
            Number of keys invalidated
        """
        if not self.redis:
            return 0

        pattern = make_cache_key("ml", "prediction", "*")

        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self.redis.delete(*keys)
                ml_cache_invalidations.labels(reason=reason).inc()
                logger.info(f"Invalidated all {deleted} ML predictions (reason: {reason})")
                return deleted
            return 0

        except Exception as e:
            logger.error(f"Error invalidating all predictions: {e}")
            return 0

    # ========================================================================
    # Batch Prediction Caching
    # ========================================================================

    async def get_predictions_batch(
        self,
        model_name: str,
        model_version: str,
        token_pairs: List[str],
        features_hashes: List[str]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get multiple predictions in batch

        Args:
            model_name: Name of model
            model_version: Model version
            token_pairs: List of token pairs
            features_hashes: List of feature hashes (same length as token_pairs)

        Returns:
            Dictionary of token_pair -> prediction
        """
        if not self.redis:
            return {pair: None for pair in token_pairs}

        cache_keys = [
            make_cache_key(
                "ml", "prediction", model_name, model_version,
                pair, hash_, f"v{self.CACHE_VERSION}"
            )
            for pair, hash_ in zip(token_pairs, features_hashes)
        ]

        try:
            with redis_operation_duration.labels(operation="mget").time():
                cached_values = await self.redis.mget(cache_keys)

            results = {}
            for pair, cached in zip(token_pairs, cached_values):
                if cached:
                    results[pair] = json.loads(cached)
                    ml_cache_hits.labels(prediction_type=model_name).inc()
                else:
                    results[pair] = None
                    ml_cache_misses.labels(prediction_type=model_name).inc()

            return results

        except Exception as e:
            logger.error(f"Error getting batch predictions: {e}")
            return {pair: None for pair in token_pairs}

    # ========================================================================
    # Prediction Confidence Tracking
    # ========================================================================

    async def get_prediction_stats(
        self,
        model_name: str,
        model_version: str
    ) -> Dict[str, Any]:
        """
        Get prediction statistics for a model

        Args:
            model_name: Name of model
            model_version: Model version

        Returns:
            Statistics dictionary
        """
        if not self.redis:
            return {"status": "unavailable"}

        pattern = make_cache_key(
            "ml", "prediction", model_name, model_version, "*"
        )

        try:
            confidences = []
            count = 0

            async for key in self.redis.scan_iter(match=pattern):
                cached = await self.redis.get(key)
                if cached:
                    data = json.loads(cached)
                    confidence = data.get("confidence", 0.0)
                    confidences.append(confidence)
                    count += 1

            if confidences:
                return {
                    "model_name": model_name,
                    "model_version": model_version,
                    "cached_predictions": count,
                    "avg_confidence": sum(confidences) / len(confidences),
                    "min_confidence": min(confidences),
                    "max_confidence": max(confidences),
                }
            else:
                return {
                    "model_name": model_name,
                    "model_version": model_version,
                    "cached_predictions": 0,
                }

        except Exception as e:
            logger.error(f"Error getting prediction stats: {e}")
            return {"status": "error", "error": str(e)}
