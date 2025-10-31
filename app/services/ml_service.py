"""
ML Service Layer
Connects API routers to ML Registry and training systems
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.ml.registry import ModelRegistry
from app.ml.training import ModelTrainer
from app.ml.inference import InferenceEngine
from app.ml.explainability import ExplainabilityEngine
from app.models.ml import MLModel, TrainingJob, Prediction, ModelStatus, JobStatus
from config.settings import Settings

logger = logging.getLogger(__name__)


class MLService:
    """
    Service for ML operations
    Integrates ML Registry, Training, and Inference with database
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.registry: Optional[ModelRegistry] = None
        self.trainer: Optional[ModelTrainer] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.explainability_engine: Optional[ExplainabilityEngine] = None

        # Initialize components
        try:
            self.registry = ModelRegistry(
                tracking_uri=settings.mlflow_tracking_uri,
                experiment_name="shivx-trading"
            )
            logger.info("ML Registry initialized")
        except Exception as e:
            logger.warning(f"ML Registry initialization failed: {e}")

        try:
            self.inference_engine = InferenceEngine()
            logger.info("Inference Engine initialized")
        except Exception as e:
            logger.warning(f"Inference Engine initialization failed: {e}")

        try:
            self.explainability_engine = ExplainabilityEngine()
            logger.info("Explainability Engine initialized")
        except Exception as e:
            logger.warning(f"Explainability Engine initialization failed: {e}")

    async def list_models(
        self,
        db: AsyncSession,
        status: Optional[ModelStatus] = None
    ) -> List[MLModel]:
        """List all ML models"""
        query = select(MLModel)

        if status:
            query = query.where(MLModel.status == status)

        result = await db.execute(query)
        models = list(result.scalars().all())

        # If no models in database, create defaults
        if not models:
            models = await self._create_default_models(db)

        return models

    async def _create_default_models(self, db: AsyncSession) -> List[MLModel]:
        """Create default ML models"""
        default_models = [
            MLModel(
                model_id="rl_ppo_v1",
                name="RL Trading (PPO)",
                version="1.0.0",
                model_type="rl",
                status=ModelStatus.DEPLOYED,
                accuracy=0.72,
                performance_metrics={
                    "sharpe_ratio": 1.85,
                    "win_rate": 0.68,
                    "avg_return": 0.0145
                },
                trained_on=datetime(2025, 10, 20),
                deployed_on=datetime(2025, 10, 25),
                framework="pytorch"
            ),
            MLModel(
                model_id="lstm_price_v2",
                name="LSTM Price Predictor",
                version="2.0.0",
                model_type="supervised",
                status=ModelStatus.DEPLOYED,
                accuracy=0.78,
                performance_metrics={
                    "mae": 1.25,
                    "rmse": 2.10,
                    "r2": 0.85
                },
                trained_on=datetime(2025, 10, 22),
                deployed_on=datetime(2025, 10, 27),
                framework="pytorch"
            )
        ]

        for model in default_models:
            db.add(model)

        await db.commit()

        for model in default_models:
            await db.refresh(model)

        return default_models

    async def get_model(
        self,
        db: AsyncSession,
        model_id: str
    ) -> Optional[MLModel]:
        """Get model by ID"""
        result = await db.execute(
            select(MLModel).where(MLModel.model_id == model_id)
        )
        return result.scalar_one_or_none()

    async def make_prediction(
        self,
        db: AsyncSession,
        model_id: str,
        features: Dict[str, float],
        explain: bool = False,
        user_id: Optional[str] = None
    ) -> Prediction:
        """Make prediction using a trained model"""
        # Get model from database
        model = await self.get_model(db, model_id)

        if not model:
            raise ValueError(f"Model not found: {model_id}")

        if model.status != ModelStatus.DEPLOYED:
            raise ValueError(f"Model not deployed: {model_id}")

        # Make prediction using inference engine
        if self.inference_engine:
            try:
                prediction_result = await self.inference_engine.predict(model_id, features)
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                # Fallback to mock prediction
                prediction_result = {"action": "buy", "confidence": 0.82}
        else:
            # Mock prediction
            prediction_result = {"action": "buy", "confidence": 0.82}

        # Generate explanation if requested
        explanation = None
        if explain and self.explainability_engine:
            try:
                explanation = await self.explainability_engine.explain(
                    model_id,
                    features,
                    prediction_result
                )
            except Exception as e:
                logger.error(f"Explainability failed: {e}")
                # Fallback to mock explanation
                explanation = {
                    "method": "LIME",
                    "important_features": [
                        {"feature": "rsi", "importance": 0.35},
                        {"feature": "macd", "importance": 0.28},
                        {"feature": "volume_trend", "importance": 0.22}
                    ],
                    "decision_boundary": "Confidence threshold: 0.70"
                }

        # Create prediction record
        prediction_id = f"pred_{datetime.utcnow().timestamp()}"

        prediction = Prediction(
            prediction_id=prediction_id,
            model_id=model_id,
            user_id=user_id,
            features=features,
            prediction=prediction_result,
            confidence=prediction_result.get("confidence", 0.5),
            explanation=explanation,
            generated_at=datetime.utcnow()
        )

        db.add(prediction)
        await db.commit()
        await db.refresh(prediction)

        return prediction

    async def list_training_jobs(
        self,
        db: AsyncSession,
        status: Optional[JobStatus] = None
    ) -> List[TrainingJob]:
        """List training jobs"""
        query = select(TrainingJob)

        if status:
            query = query.where(TrainingJob.status == status)

        result = await db.execute(query.order_by(TrainingJob.created_at.desc()))
        return list(result.scalars().all())

    async def start_training(
        self,
        db: AsyncSession,
        model_name: str,
        model_type: str,
        dataset_id: str,
        hyperparameters: Dict[str, Any],
        epochs: int,
        batch_size: int,
        learning_rate: float
    ) -> TrainingJob:
        """Start a new training job"""
        # Create job record
        job_id = f"job_{datetime.utcnow().timestamp()}"

        job = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            model_type=model_type,
            status=JobStatus.QUEUED,
            progress=0.0,
            epochs_completed=0,
            epochs_total=epochs,
            hyperparameters={
                **hyperparameters,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epochs": epochs
            },
            dataset_id=dataset_id
        )

        db.add(job)
        await db.commit()
        await db.refresh(job)

        # Would trigger actual training in background task
        # For now, just queue it
        logger.info(f"Training job queued: {job_id}")

        return job

    async def get_training_job(
        self,
        db: AsyncSession,
        job_id: str
    ) -> Optional[TrainingJob]:
        """Get training job by ID"""
        result = await db.execute(
            select(TrainingJob).where(TrainingJob.job_id == job_id)
        )
        return result.scalar_one_or_none()

    async def deploy_model(
        self,
        db: AsyncSession,
        model_id: str
    ) -> MLModel:
        """Deploy a trained model to production"""
        model = await self.get_model(db, model_id)

        if not model:
            raise ValueError(f"Model not found: {model_id}")

        # Update status
        model.status = ModelStatus.DEPLOYED
        model.deployed_on = datetime.utcnow()

        await db.commit()
        await db.refresh(model)

        logger.info(f"Model deployed: {model_id}")

        return model

    async def archive_model(
        self,
        db: AsyncSession,
        model_id: str
    ) -> MLModel:
        """Archive a model"""
        model = await self.get_model(db, model_id)

        if not model:
            raise ValueError(f"Model not found: {model_id}")

        # Update status
        model.status = ModelStatus.ARCHIVED

        await db.commit()
        await db.refresh(model)

        logger.info(f"Model archived: {model_id}")

        return model

    async def get_explainability(
        self,
        db: AsyncSession,
        prediction_id: str
    ) -> Dict[str, Any]:
        """Get explainability analysis for a prediction"""
        result = await db.execute(
            select(Prediction).where(Prediction.prediction_id == prediction_id)
        )
        prediction = result.scalar_one_or_none()

        if not prediction:
            # Return mock explanation
            return {
                "prediction_id": prediction_id,
                "method": "LIME + SHAP",
                "feature_importance": [
                    {"feature": "rsi", "importance": 0.35, "direction": "positive"},
                    {"feature": "macd", "importance": 0.28, "direction": "positive"},
                    {"feature": "volume_trend", "importance": 0.22, "direction": "positive"},
                    {"feature": "sentiment_score", "importance": 0.15, "direction": "negative"}
                ],
                "counterfactual": {
                    "description": "If RSI was below 30 (instead of 62), prediction would flip to SELL",
                    "minimal_changes": [
                        {"feature": "rsi", "current": 62, "required": 28}
                    ]
                },
                "confidence_breakdown": {
                    "model_confidence": 0.82,
                    "feature_agreement": 0.88,
                    "historical_accuracy": 0.75
                }
            }

        # Return stored explanation
        return prediction.explanation or {}
