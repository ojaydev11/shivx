"""
ML Inference Service Integration
Integrates async ML inference with FastAPI

This module bridges the FastAPI endpoints with the MLOps infrastructure
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from app.ml.inference import MLInferenceService as CoreInferenceService
from app.ml.monitor import ModelMonitor, PredictionLog
from app.ml.explainability import XAISystem

logger = logging.getLogger(__name__)


class MLInferenceService:
    """
    Production ML inference service for FastAPI integration

    Features:
    - Async prediction
    - Automatic monitoring
    - Explainability on demand
    """

    def __init__(self):
        """Initialize ML inference service"""
        self.core_service = CoreInferenceService()
        self.monitor = ModelMonitor()
        self.xai = XAISystem()

        self._initialized = False

    async def initialize(self):
        """Initialize async resources"""
        if not self._initialized:
            await self.core_service.initialize()
            await self.monitor.initialize()
            self._initialized = True
            logger.info("ML Inference Service initialized")

    async def close(self):
        """Close async resources"""
        if self._initialized:
            await self.core_service.close()
            await self.monitor.close()
            self._initialized = False

    async def predict(
        self,
        model_id: str,
        features: Dict[str, float],
        explain: bool = False,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make prediction with monitoring and optional explainability

        Args:
            model_id: Model identifier
            features: Input features
            explain: Include explainability
            user_id: User making request

        Returns:
            Prediction result with metadata
        """
        # Make prediction
        result = await self.core_service.predict_async(
            model_id=model_id,
            features=features,
            use_cache=True,
            timeout=5.0
        )

        # Log prediction for monitoring
        if "error" not in result:
            pred_log = PredictionLog(
                prediction_id=result["prediction_id"],
                model_id=model_id,
                model_version="latest",  # TODO: Get actual version
                features=features,
                prediction=result.get("prediction"),
                confidence=result.get("confidence", 0.0),
                latency_ms=result.get("latency_ms", 0.0)
            )

            await self.monitor.log_prediction(pred_log)

            # Add explainability if requested
            if explain:
                # TODO: Load actual model for explanation
                # For now, add placeholder
                result["explanation"] = {
                    "method": "lime",
                    "available": True,
                    "message": "Call /explainability/{prediction_id} for details"
                }

        return result

    async def get_model_health(self, model_id: str) -> Dict[str, Any]:
        """
        Get model health status

        Args:
            model_id: Model identifier

        Returns:
            Health status
        """
        return await self.monitor.get_model_health(model_id)


# Global instance (initialized in FastAPI startup)
_ml_service: Optional[MLInferenceService] = None


def get_ml_service() -> MLInferenceService:
    """Get ML inference service instance"""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLInferenceService()
    return _ml_service
