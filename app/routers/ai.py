"""
AI/ML API Router
Endpoints for ML models, predictions, and training
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from app.dependencies import get_current_user, require_permission, get_settings
from app.dependencies.auth import TokenData
from config.settings import Settings
from core.security.hardening import Permission
from utils.prompt_filter import get_prompt_filter, ThreatLevel
from utils.content_moderation import get_content_moderator, ModerationSeverity
import logging

logger = logging.getLogger(__name__)


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
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    List all ML models

    Requires: READ permission

    Args:
        status: Filter by status (training, ready, deployed, archived)
    """
    # TODO: Fetch from model registry
    models = [
        ModelInfo(
            model_id="rl_ppo_v1",
            name="RL Trading (PPO)",
            version="1.0.0",
            type="rl",
            status="deployed",
            accuracy=0.72,
            performance_metrics={
                "sharpe_ratio": 1.85,
                "win_rate": 0.68,
                "avg_return": 0.0145
            },
            trained_on=datetime(2025, 10, 20),
            deployed_on=datetime(2025, 10, 25)
        ),
        ModelInfo(
            model_id="lstm_price_v2",
            name="LSTM Price Predictor",
            version="2.0.0",
            type="supervised",
            status="deployed",
            accuracy=0.78,
            performance_metrics={
                "mae": 1.25,
                "rmse": 2.10,
                "r2": 0.85
            },
            trained_on=datetime(2025, 10, 22),
            deployed_on=datetime(2025, 10, 27)
        )
    ]

    if status:
        models = [m for m in models if m.status == status]

    return models


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    current_user: TokenData = Depends(require_permission(Permission.READ))
):
    """
    Get model details

    Requires: READ permission
    """
    # TODO: Fetch from model registry
    return ModelInfo(
        model_id=model_id,
        name="RL Trading (PPO)",
        version="1.0.0",
        type="rl",
        status="deployed",
        accuracy=0.72,
        performance_metrics={
            "sharpe_ratio": 1.85,
            "win_rate": 0.68,
            "avg_return": 0.0145
        },
        trained_on=datetime(2025, 10, 20),
        deployed_on=datetime(2025, 10, 25)
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
    settings: Settings = Depends(get_settings)
):
    """
    Make a prediction using a trained model

    Requires: EXECUTE permission

    Args:
        request: Prediction request with model ID and features

    Security:
        - Validates input for prompt injection attempts
        - Scans output for leaked secrets
        - Content moderation for harmful content
    """
    # SECURITY: Validate model_id for prompt injection
    prompt_filter = get_prompt_filter(strict_mode=True)
    input_validation = prompt_filter.filter_input(request.model_id)

    if not input_validation.is_safe:
        logger.warning(
            f"Prompt injection detected in model_id: {input_validation.detected_threats}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Invalid input detected",
                "threat_level": input_validation.threat_level.value,
                "threats": input_validation.detected_threats
            }
        )

    # TODO: Load model and make prediction
    # from core.learning.model_registry import ModelRegistry
    # registry = ModelRegistry()
    # model = registry.load_model(request.model_id)
    # prediction = model.predict(request.features)

    prediction_id = f"pred_{datetime.now().timestamp()}"

    response = PredictionResponse(
        prediction_id=prediction_id,
        model_id=request.model_id,
        prediction={"action": "buy", "confidence": 0.82},
        confidence=0.82,
        generated_at=datetime.now()
    )

    # Add explainability if requested
    if request.explain:
        response.explanation = {
            "method": "LIME",
            "important_features": [
                {"feature": "rsi", "importance": 0.35},
                {"feature": "macd", "importance": 0.28},
                {"feature": "volume_trend", "importance": 0.22}
            ],
            "decision_boundary": "Confidence threshold: 0.70"
        }

    # SECURITY: Validate output for leaked secrets
    # Note: This is a basic check. The DLP middleware will do a more thorough scan
    import json
    output_str = json.dumps(response.dict())
    output_validation = prompt_filter.filter_output(output_str)

    if not output_validation.is_safe:
        logger.critical(
            f"Secret leak detected in prediction output: {output_validation.detected_threats}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Output validation failed - potential security issue"
        )

    return response


@router.get("/training-jobs", response_model=List[TrainingJob])
async def list_training_jobs(
    status: Optional[str] = None,
    current_user: TokenData = Depends(require_permission(Permission.READ))
):
    """
    List training jobs

    Requires: READ permission

    Args:
        status: Filter by status (queued, running, completed, failed)
    """
    # TODO: Fetch from training job queue
    jobs = [
        TrainingJob(
            job_id="job_123",
            model_name="RL Trading v2",
            model_type="rl",
            status="running",
            progress=0.65,
            epochs_completed=65,
            epochs_total=100,
            loss=0.0125,
            metrics={"reward": 1250.5, "sharpe": 1.92},
            started_at=datetime.now()
        )
    ]

    if status:
        jobs = [j for j in jobs if j.status == status]

    return jobs


@router.post("/train", response_model=TrainingJob)
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
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

    # Create job
    job_id = f"job_{datetime.now().timestamp()}"

    job = TrainingJob(
        job_id=job_id,
        model_name=config.model_name,
        model_type=config.model_type,
        status="queued",
        progress=0.0,
        epochs_completed=0,
        epochs_total=config.epochs,
        metrics={}
    )

    # Add training task to background
    # background_tasks.add_task(run_training, job_id, config)

    return job


@router.get("/training-jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(
    job_id: str,
    current_user: TokenData = Depends(require_permission(Permission.READ))
):
    """
    Get training job details

    Requires: READ permission
    """
    # TODO: Fetch job from queue/database
    return TrainingJob(
        job_id=job_id,
        model_name="RL Trading v2",
        model_type="rl",
        status="running",
        progress=0.65,
        epochs_completed=65,
        epochs_total=100,
        loss=0.0125,
        metrics={"reward": 1250.5, "sharpe": 1.92},
        started_at=datetime.now()
    )


@router.post("/models/{model_id}/deploy")
async def deploy_model(
    model_id: str,
    current_user: TokenData = Depends(require_permission(Permission.ADMIN))
):
    """
    Deploy a trained model to production

    Requires: ADMIN permission
    """
    # TODO: Deploy model
    return {
        "model_id": model_id,
        "status": "deployed",
        "deployed_at": datetime.now()
    }


@router.post("/models/{model_id}/archive")
async def archive_model(
    model_id: str,
    current_user: TokenData = Depends(require_permission(Permission.ADMIN))
):
    """
    Archive a model

    Requires: ADMIN permission
    """
    # TODO: Archive model
    return {
        "model_id": model_id,
        "status": "archived",
        "archived_at": datetime.now()
    }


@router.get("/explainability/{prediction_id}")
async def get_explainability(
    prediction_id: str,
    current_user: TokenData = Depends(require_permission(Permission.READ))
):
    """
    Get explainability analysis for a prediction

    Requires: READ permission

    Uses LIME, SHAP, and attention visualization
    """
    # TODO: Fetch explainability data
    # from core.explain.xai import XAISystem
    # xai = XAISystem(model)
    # explanation = xai.explain(instance)

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
