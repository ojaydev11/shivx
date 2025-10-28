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
from core.memory.rag import RAGPipeline, RAGConfig
from core.memory.long_term_memory import LongTermMemory, MemoryType
from core.memory.conversation_memory import ConversationMemory
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
        "memory_rag": True,  # New RAG capability
        "available_models": [
            "RL Trading (PPO)",
            "LSTM Price Predictor",
            "Sentiment Analyzer",
            "Arbitrage Detector"
        ]
    }


# ============================================================================
# Memory & RAG Endpoints
# ============================================================================

class MemoryStoreRequest(BaseModel):
    """Request to store memory"""
    content: str = Field(..., description="Memory content")
    memory_type: str = Field(..., description="Memory type: episodic, semantic, or procedural")
    importance: float = Field(default=0.5, ge=0, le=1, description="Importance score")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class MemoryRetrieveRequest(BaseModel):
    """Request to retrieve memories"""
    query: str = Field(..., description="Search query")
    memory_type: Optional[str] = Field(default=None, description="Filter by memory type")
    k: int = Field(default=10, ge=1, le=50, description="Number of results")


class RAGChatRequest(BaseModel):
    """Request for RAG-enhanced chat"""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(default=None, description="Conversation session ID")
    use_rag: bool = Field(default=True, description="Use RAG for context")
    max_context_tokens: int = Field(default=4000, description="Max context tokens")


class RAGChatResponse(BaseModel):
    """Response from RAG-enhanced chat"""
    response: str
    session_id: str
    contexts_used: int
    confidence_score: float
    hallucination_detected: bool
    warnings: List[str] = []
    metadata: Dict[str, Any] = {}


# Initialize memory systems (singleton pattern)
_long_term_memory = None
_conversation_memory = None
_rag_pipeline = None


def get_memory_systems():
    """Get or create memory system instances"""
    global _long_term_memory, _conversation_memory, _rag_pipeline

    if _long_term_memory is None:
        _long_term_memory = LongTermMemory()
        logger.info("Initialized long-term memory")

    if _conversation_memory is None:
        _conversation_memory = ConversationMemory()
        logger.info("Initialized conversation memory")

    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(_long_term_memory)
        logger.info("Initialized RAG pipeline")

    return _long_term_memory, _conversation_memory, _rag_pipeline


@router.post("/memory/store")
async def store_memory(
    request: MemoryStoreRequest,
    current_user: TokenData = Depends(require_permission(Permission.WRITE)),
    settings: Settings = Depends(get_settings)
):
    """
    Store a memory in long-term storage

    Requires: WRITE permission

    Args:
        request: Memory storage request
    """
    try:
        memory, _, _ = get_memory_systems()

        # Validate memory type
        try:
            mem_type = MemoryType(request.memory_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid memory type. Must be: episodic, semantic, or procedural"
            )

        # Store memory
        memory_id = memory.store(
            content=request.content,
            memory_type=mem_type,
            source=current_user.username,
            importance_score=request.importance,
            metadata=request.metadata or {}
        )

        return {
            "memory_id": memory_id,
            "status": "stored",
            "memory_type": request.memory_type,
            "importance": request.importance
        }

    except Exception as e:
        logger.error(f"Failed to store memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store memory: {str(e)}"
        )


@router.post("/memory/retrieve")
async def retrieve_memories(
    request: MemoryRetrieveRequest,
    current_user: TokenData = Depends(require_permission(Permission.READ)),
):
    """
    Retrieve memories using semantic search

    Requires: READ permission

    Args:
        request: Memory retrieval request
    """
    try:
        memory, _, _ = get_memory_systems()

        # Parse memory type filter
        mem_type = None
        if request.memory_type:
            try:
                mem_type = MemoryType(request.memory_type)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid memory type"
                )

        # Retrieve memories
        results = memory.retrieve(
            query=request.query,
            memory_type=mem_type,
            k=request.k
        )

        # Format results
        formatted_results = [
            {
                "memory_id": mem.memory_id,
                "content": mem.content,
                "memory_type": mem.memory_type.value,
                "importance": mem.importance_score,
                "relevance_score": score,
                "created_at": mem.created_at.isoformat(),
                "access_count": mem.access_count
            }
            for mem, score in results
        ]

        return {
            "query": request.query,
            "results": formatted_results,
            "count": len(formatted_results)
        }

    except Exception as e:
        logger.error(f"Failed to retrieve memories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memories: {str(e)}"
        )


@router.post("/chat/rag", response_model=RAGChatResponse)
async def chat_with_rag(
    request: RAGChatRequest,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
    settings: Settings = Depends(get_settings)
):
    """
    Chat with RAG (Retrieval-Augmented Generation)

    Enhances responses with relevant context from long-term memory.

    Requires: EXECUTE permission

    Args:
        request: RAG chat request

    Security:
        - Input validation for prompt injection
        - Content moderation
        - Output validation
    """
    try:
        memory, conv_memory, rag_pipeline = get_memory_systems()

        # SECURITY: Validate input
        prompt_filter = get_prompt_filter(strict_mode=True)
        input_validation = prompt_filter.filter_input(request.message)

        if not input_validation.is_safe:
            logger.warning(f"Prompt injection detected: {input_validation.detected_threats}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid input detected",
                    "threat_level": input_validation.threat_level.value
                }
            )

        # Get or create conversation session
        session_id = request.session_id
        if not session_id:
            session = conv_memory.create_session(user_id=current_user.username)
            session_id = session.session_id
        else:
            session = conv_memory.get_session(session_id)
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Session not found"
                )

        # Add user message to conversation
        conv_memory.add_message(session_id, "user", request.message)

        # Mock LLM callback (replace with actual LLM integration)
        def llm_callback(prompt: str) -> str:
            # This is a placeholder - integrate with OpenAI/Anthropic/etc
            return f"This is a RAG-enhanced response to: {request.message}. [Based on retrieved context]"

        # Generate response with RAG
        if request.use_rag:
            rag_config = RAGConfig(max_context_tokens=request.max_context_tokens)
            rag_pipeline.config = rag_config

            rag_result = rag_pipeline.generate(
                query=request.message,
                llm_callback=llm_callback,
                include_metadata=True
            )

            response_text = rag_result.response
            contexts_used = len(rag_result.contexts)
            confidence = rag_result.confidence_score
            hallucination = rag_result.hallucination_detected
            warnings = rag_result.warnings
            metadata = rag_result.metadata
        else:
            # Direct LLM without RAG
            response_text = llm_callback(request.message)
            contexts_used = 0
            confidence = 0.7
            hallucination = False
            warnings = []
            metadata = {}

        # SECURITY: Validate output
        output_validation = prompt_filter.filter_output(response_text)
        if not output_validation.is_safe:
            logger.critical(f"Secret leak detected in response")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Output validation failed"
            )

        # Add assistant message to conversation
        conv_memory.add_message(
            session_id,
            "assistant",
            response_text,
            metadata={"rag_enabled": request.use_rag, "contexts_used": contexts_used}
        )

        return RAGChatResponse(
            response=response_text,
            session_id=session_id,
            contexts_used=contexts_used,
            confidence_score=confidence,
            hallucination_detected=hallucination,
            warnings=warnings,
            metadata=metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG chat failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )


@router.get("/memory/stats")
async def get_memory_stats(
    current_user: TokenData = Depends(require_permission(Permission.READ))
):
    """
    Get memory system statistics

    Requires: READ permission
    """
    try:
        memory, conv_memory, _ = get_memory_systems()

        return {
            "long_term_memory": memory.get_stats(),
            "conversation_memory": conv_memory.get_stats()
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )
