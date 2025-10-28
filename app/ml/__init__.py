"""
ShivX MLOps Module
Production-ready ML operations for trading models

Components:
- Model Registry: MLflow-based versioning
- Async Inference: Celery/Redis background tasks
- Monitoring: Drift detection, performance tracking
- Training: Automated retraining pipelines
- Serving: ONNX optimization, model warming
- Features: Feature store with versioning
- XAI: LIME/SHAP explainability
- Orchestration: End-to-end ML pipelines
"""

from app.ml.registry import ModelRegistry
from app.ml.inference import MLInferenceService
from app.ml.monitor import ModelMonitor
from app.ml.training import TrainingPipeline
from app.ml.serving import ModelServingOptimizer
from app.ml.features import FeatureStore
from app.ml.explainability import XAISystem

__all__ = [
    "ModelRegistry",
    "MLInferenceService",
    "ModelMonitor",
    "TrainingPipeline",
    "ModelServingOptimizer",
    "FeatureStore",
    "XAISystem",
]
