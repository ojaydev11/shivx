"""
ML Database Models
"""

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Float, Integer, DateTime, JSON, Enum as SQLEnum
from datetime import datetime
from typing import Optional, Dict, Any
import enum

from app.models.base import Base, TimestampMixin


class ModelStatus(str, enum.Enum):
    """Model status enum"""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class JobStatus(str, enum.Enum):
    """Training job status enum"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MLModel(Base, TimestampMixin):
    """ML model registry"""
    __tablename__ = "ml_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # rl, supervised, unsupervised

    status: Mapped[ModelStatus] = mapped_column(
        SQLEnum(ModelStatus),
        nullable=False,
        default=ModelStatus.TRAINING
    )

    # Performance metrics
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    performance_metrics: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON, nullable=True)

    # Timestamps
    trained_on: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    deployed_on: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # MLflow integration
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    mlflow_model_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Model artifacts
    artifacts_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Metadata
    framework: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # pytorch, tensorflow, sklearn
    author: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)


class TrainingJob(Base, TimestampMixin):
    """ML training job"""
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)

    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)

    status: Mapped[JobStatus] = mapped_column(
        SQLEnum(JobStatus),
        nullable=False,
        default=JobStatus.QUEUED
    )

    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    epochs_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    epochs_total: Mapped[int] = mapped_column(Integer, nullable=False)

    # Current metrics
    loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metrics: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON, nullable=True)

    # Configuration
    hyperparameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    dataset_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Timestamps
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Error information
    error_message: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # Result model ID
    result_model_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)


class Prediction(Base, TimestampMixin):
    """ML prediction record"""
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)

    model_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    # Input/Output
    features: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    prediction: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)

    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Explainability
    explanation: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Timestamp
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Feedback (for model improvement)
    actual_outcome: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    feedback_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
