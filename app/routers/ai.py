"""
AI/ML API Router
Endpoints for ML models, predictions, and training
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, require_permission, get_settings
from app.dependencies.auth import TokenData
from app.database import get_db
from app.services.ml_service import MLService
from app.models.ml import ModelStatus, JobStatus
from config.settings import Settings
from core.security.hardening import Permission


router = APIRouter(
    prefix="/api/ai",
    tags=["ai"],
)


# ============================================================================
# Pydantic Models
# ============================================================================

class ModelInfo(BaseModel):
    """ML model information"""
    model_id: str
    name: str
    version: str
    type: str  # rl, supervised, unsupervised
    status: str  # training, ready, deployed, archived
    accuracy: Optional[float] = None
    performance_metrics: Dict[str, float] = {}
    trained_on: Optional[datetime] = None
    deployed_on: Optional[datetime] = None


class PredictionRequest(BaseModel):
    """Prediction request"""
    model_id: str = Field(..., description="Model ID to use for prediction")
    features: Dict[str, float] = Field(..., description="Input features")
    explain: bool = Field(default=False, description="Include explainability")


class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction_id: str
    model_id: str
    prediction: Any
    confidence: float = Field(..., ge=0, le=1)
    explanation: Optional[Dict[str, Any]] = None
    generated_at: datetime


class TrainingJob(BaseModel):
    """ML training job"""
    job_id: str
    model_name: str
    model_type: str
    status: str  # queued, running, completed, failed
    progress: float = Field(..., ge=0, le=1)
    epochs_completed: int
    epochs_total: int
    loss: Optional[float] = None
    metrics: Dict[str, float] = {}
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TrainingConfig(BaseModel):
    """Training configuration"""
    model_name: str
    model_type: str = Field(..., description="rl, supervised, unsupervised")
    dataset_id: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    List all ML models

    Requires: READ permission

    Args:
        status: Filter by status (training, ready, deployed, archived)
    """
    service = MLService(settings)

    # Parse status
    model_status = ModelStatus(status) if status else None

    models = await service.list_models(db=db, status=model_status)

    return [
        ModelInfo(
            model_id=model.model_id,
            name=model.name,
            version=model.version,
            type=model.model_type,
            status=model.status.value,
            accuracy=model.accuracy,
            performance_metrics=model.performance_metrics or {},
            trained_on=model.trained_on,
            deployed_on=model.deployed_on
        )
        for model in models
    ]


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    Get model details

    Requires: READ permission
    """
    service = MLService(settings)
    model = await service.get_model(db=db, model_id=model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}"
        )

    return ModelInfo(
        model_id=model.model_id,
        name=model.name,
        version=model.version,
        type=model.model_type,
        status=model.status.value,
        accuracy=model.accuracy,
        performance_metrics=model.performance_metrics or {},
        trained_on=model.trained_on,
        deployed_on=model.deployed_on
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
    settings: Settings = Depends(get_settings)
):
    """
    Make a prediction using a trained model

    Requires: EXECUTE permission

    Args:
        request: Prediction request with model ID and features
    """
    service = MLService(settings)

    prediction = await service.make_prediction(
        db=db,
        model_id=request.model_id,
        features=request.features,
        explain=request.explain,
        user_id=current_user.username
    )

    return PredictionResponse(
        prediction_id=prediction.prediction_id,
        model_id=prediction.model_id,
        prediction=prediction.prediction,
        confidence=prediction.confidence,
        explanation=prediction.explanation,
        generated_at=prediction.generated_at
    )


@router.get("/training-jobs", response_model=List[TrainingJob])
async def list_training_jobs(
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    List training jobs

    Requires: READ permission

    Args:
        status: Filter by status (queued, running, completed, failed)
    """
    service = MLService(settings)

    # Parse status
    job_status = JobStatus(status) if status else None

    jobs = await service.list_training_jobs(db=db, status=job_status)

    return [
        TrainingJob(
            job_id=job.job_id,
            model_name=job.model_name,
            model_type=job.model_type,
            status=job.status.value,
            progress=job.progress,
            epochs_completed=job.epochs_completed,
            epochs_total=job.epochs_total,
            loss=job.loss,
            metrics=job.metrics or {},
            started_at=job.started_at
        )
        for job in jobs
    ]


@router.post("/train", response_model=TrainingJob)
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.ADMIN)),
    settings: Settings = Depends(get_settings)
):
    """
    Start a new training job

    Requires: ADMIN permission

    Args:
        config: Training configuration
        background_tasks: FastAPI background tasks
    """
    if not settings.feature_rl_trading and config.model_type == "rl":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="RL training is disabled"
        )

    service = MLService(settings)

    job = await service.start_training(
        db=db,
        model_name=config.model_name,
        model_type=config.model_type,
        dataset_id=config.dataset_id,
        hyperparameters=config.hyperparameters,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate
    )

    # Add training task to background
    # background_tasks.add_task(run_training, job.job_id, config)

    return TrainingJob(
        job_id=job.job_id,
        model_name=job.model_name,
        model_type=job.model_type,
        status=job.status.value,
        progress=job.progress,
        epochs_completed=job.epochs_completed,
        epochs_total=job.epochs_total,
        metrics=job.metrics or {}
    )


@router.get("/training-jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    Get training job details

    Requires: READ permission
    """
    service = MLService(settings)
    job = await service.get_training_job(db=db, job_id=job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job not found: {job_id}"
        )

    return TrainingJob(
        job_id=job.job_id,
        model_name=job.model_name,
        model_type=job.model_type,
        status=job.status.value,
        progress=job.progress,
        epochs_completed=job.epochs_completed,
        epochs_total=job.epochs_total,
        loss=job.loss,
        metrics=job.metrics or {},
        started_at=job.started_at
    )


@router.post("/models/{model_id}/deploy")
async def deploy_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.ADMIN)),
    settings: Settings = Depends(get_settings)
):
    """
    Deploy a trained model to production

    Requires: ADMIN permission
    """
    service = MLService(settings)
    model = await service.deploy_model(db=db, model_id=model_id)

    return {
        "model_id": model.model_id,
        "status": model.status.value,
        "deployed_at": model.deployed_on
    }


@router.post("/models/{model_id}/archive")
async def archive_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.ADMIN)),
    settings: Settings = Depends(get_settings)
):
    """
    Archive a model

    Requires: ADMIN permission
    """
    service = MLService(settings)
    model = await service.archive_model(db=db, model_id=model_id)

    return {
        "model_id": model.model_id,
        "status": model.status.value,
        "archived_at": datetime.utcnow()
    }


@router.get("/explainability/{prediction_id}")
async def get_explainability(
    prediction_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    Get explainability analysis for a prediction

    Requires: READ permission

    Uses LIME, SHAP, and attention visualization
    """
    service = MLService(settings)
    explanation = await service.get_explainability(db=db, prediction_id=prediction_id)

    return explanation


@router.get("/capabilities")
async def get_ai_capabilities(
    settings: Settings = Depends(get_settings)
):
    """
    Get enabled AI capabilities (public endpoint)

    No authentication required
    """
    return {
        "rl_trading": settings.feature_rl_trading,
        "sentiment_analysis": settings.feature_sentiment_analysis,
        "dex_arbitrage": settings.feature_dex_arbitrage,
        "metacognition": settings.feature_metacognition,
        "explainability": True,
        "available_models": [
            "RL Trading (PPO)",
            "LSTM Price Predictor",
            "Sentiment Analyzer",
            "Arbitrage Detector"
        ]
    }
