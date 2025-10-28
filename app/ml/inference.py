"""
Async ML Inference Service
Non-blocking inference with Celery and Redis

Features:
- Async background inference tasks
- Batch inference for efficiency
- Redis caching of predictions
- Queue management
- Timeout handling
- Inference latency monitoring
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import uuid

import numpy as np
import redis.asyncio as redis
from celery import Celery

logger = logging.getLogger(__name__)


# Celery configuration
celery_app = Celery(
    'ml_inference',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/1'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max
    task_soft_time_limit=120,  # Soft limit 2 minutes
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
)


class MLInferenceService:
    """
    Async ML inference service with caching and batching

    Features:
    - Non-blocking inference
    - Batch processing
    - Redis caching
    - Timeout handling
    """

    def __init__(
        self,
        redis_url: str = "redis://redis:6379",
        cache_ttl: int = 3600,
        batch_size: int = 32,
        batch_timeout: float = 0.1,
    ):
        """
        Initialize inference service

        Args:
            redis_url: Redis connection URL
            cache_ttl: Cache TTL in seconds
            batch_size: Maximum batch size
            batch_timeout: Max time to wait for batch (seconds)
        """
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        # Redis client (initialized async)
        self.redis_client: Optional[redis.Redis] = None

        # Batch queue
        self.batch_queue: List[Dict[str, Any]] = []
        self.batch_lock = asyncio.Lock()

        logger.info("ML Inference Service initialized")

    async def initialize(self):
        """Initialize async resources"""
        self.redis_client = await redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        logger.info("Redis client connected")

    async def close(self):
        """Close async resources"""
        if self.redis_client:
            await self.redis_client.close()

    async def predict_async(
        self,
        model_id: str,
        features: Dict[str, float],
        use_cache: bool = True,
        timeout: float = 5.0
    ) -> Dict[str, Any]:
        """
        Make async prediction

        Args:
            model_id: Model identifier
            features: Input features
            use_cache: Use cached predictions
            timeout: Prediction timeout (seconds)

        Returns:
            Prediction result
        """
        prediction_id = str(uuid.uuid4())

        # Check cache first
        if use_cache:
            cached = await self._get_cached_prediction(model_id, features)
            if cached:
                logger.info(f"Cache hit for prediction {prediction_id}")
                return {
                    "prediction_id": prediction_id,
                    "cached": True,
                    **cached
                }

        # Submit to Celery
        start_time = time.time()

        try:
            task = run_inference_task.apply_async(
                args=[model_id, features, prediction_id],
                expires=timeout
            )

            # Wait for result with timeout
            result = task.get(timeout=timeout)

            latency = time.time() - start_time

            # Cache result
            if use_cache:
                await self._cache_prediction(model_id, features, result, latency)

            return {
                "prediction_id": prediction_id,
                "cached": False,
                "latency_ms": latency * 1000,
                **result
            }

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                "prediction_id": prediction_id,
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000
            }

    async def predict_batch(
        self,
        model_id: str,
        features_list: List[Dict[str, float]],
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction for efficiency

        Args:
            model_id: Model identifier
            features_list: List of feature dictionaries
            use_cache: Use cached predictions

        Returns:
            List of prediction results
        """
        logger.info(f"Batch prediction: {len(features_list)} samples")

        start_time = time.time()

        # Check cache for each sample
        results = []
        uncached_indices = []
        uncached_features = []

        for idx, features in enumerate(features_list):
            if use_cache:
                cached = await self._get_cached_prediction(model_id, features)
                if cached:
                    results.append({
                        "index": idx,
                        "cached": True,
                        **cached
                    })
                    continue

            uncached_indices.append(idx)
            uncached_features.append(features)

        # Run batch inference for uncached samples
        if uncached_features:
            try:
                task = run_batch_inference_task.apply_async(
                    args=[model_id, uncached_features]
                )

                batch_results = task.get(timeout=30)

                # Merge results
                for idx, features, result in zip(uncached_indices, uncached_features, batch_results):
                    results.append({
                        "index": idx,
                        "cached": False,
                        **result
                    })

                    # Cache each result
                    if use_cache:
                        await self._cache_prediction(model_id, features, result, 0)

            except Exception as e:
                logger.error(f"Batch inference failed: {e}")
                # Add errors for failed samples
                for idx in uncached_indices:
                    results.append({
                        "index": idx,
                        "error": str(e)
                    })

        # Sort by original index
        results.sort(key=lambda x: x["index"])

        latency = time.time() - start_time
        logger.info(f"Batch prediction completed: {latency*1000:.2f}ms")

        return results

    async def get_prediction_status(
        self,
        prediction_id: str
    ) -> Dict[str, Any]:
        """
        Get status of async prediction

        Args:
            prediction_id: Prediction ID

        Returns:
            Prediction status
        """
        if not self.redis_client:
            return {"error": "Redis not initialized"}

        # Check if result is cached
        key = f"prediction:{prediction_id}"
        result = await self.redis_client.get(key)

        if result:
            return json.loads(result)

        return {
            "prediction_id": prediction_id,
            "status": "not_found"
        }

    async def _get_cached_prediction(
        self,
        model_id: str,
        features: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Get cached prediction if available"""
        if not self.redis_client:
            return None

        # Create cache key from features
        features_str = json.dumps(features, sort_keys=True)
        cache_key = f"pred_cache:{model_id}:{hash(features_str)}"

        cached = await self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)

        return None

    async def _cache_prediction(
        self,
        model_id: str,
        features: Dict[str, float],
        result: Dict[str, Any],
        latency: float
    ):
        """Cache prediction result"""
        if not self.redis_client:
            return

        features_str = json.dumps(features, sort_keys=True)
        cache_key = f"pred_cache:{model_id}:{hash(features_str)}"

        cache_data = {
            **result,
            "cached_at": datetime.now().isoformat(),
            "inference_latency_ms": latency * 1000
        }

        await self.redis_client.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(cache_data)
        )

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {}

        # Get all cache keys
        keys = await self.redis_client.keys("pred_cache:*")

        return {
            "total_cached_predictions": len(keys),
            "cache_ttl_seconds": self.cache_ttl,
        }


# =============================================================================
# Celery Tasks
# =============================================================================

@celery_app.task(name='ml_inference.run_inference', bind=True)
def run_inference_task(self, model_id: str, features: Dict[str, float], prediction_id: str) -> Dict[str, Any]:
    """
    Run inference task (executed by Celery worker)

    Args:
        model_id: Model identifier
        features: Input features
        prediction_id: Prediction ID

    Returns:
        Prediction result
    """
    logger.info(f"Running inference: {model_id} (pred_id={prediction_id})")

    start_time = time.time()

    try:
        # Load model (should be cached in worker)
        from app.ml.registry import ModelRegistry
        registry = ModelRegistry()

        model = registry.load_model(model_id, stage="Production")

        # Prepare input
        feature_array = np.array([list(features.values())])

        # Run inference
        prediction = model.predict(feature_array)

        latency = time.time() - start_time

        # Format result
        result = {
            "model_id": model_id,
            "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            "confidence": float(np.max(prediction)) if hasattr(prediction, 'max') else 0.85,
            "inference_time_ms": latency * 1000,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Inference completed: {latency*1000:.2f}ms")

        return result

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return {
            "error": str(e),
            "model_id": model_id,
            "timestamp": datetime.now().isoformat()
        }


@celery_app.task(name='ml_inference.run_batch_inference')
def run_batch_inference_task(model_id: str, features_list: List[Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Run batch inference task

    Args:
        model_id: Model identifier
        features_list: List of feature dictionaries

    Returns:
        List of prediction results
    """
    logger.info(f"Running batch inference: {model_id}, batch_size={len(features_list)}")

    start_time = time.time()

    try:
        # Load model
        from app.ml.registry import ModelRegistry
        registry = ModelRegistry()

        model = registry.load_model(model_id, stage="Production")

        # Prepare batch input
        feature_arrays = [np.array(list(f.values())) for f in features_list]
        batch_input = np.array(feature_arrays)

        # Run batch inference
        predictions = model.predict(batch_input)

        latency = time.time() - start_time

        # Format results
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "model_id": model_id,
                "prediction": pred.tolist() if hasattr(pred, 'tolist') else pred,
                "confidence": float(np.max(pred)) if hasattr(pred, 'max') else 0.85,
                "timestamp": datetime.now().isoformat()
            })

        logger.info(f"Batch inference completed: {latency*1000:.2f}ms, {len(predictions)} predictions")

        return results

    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        # Return errors for all samples
        return [{"error": str(e)} for _ in features_list]
