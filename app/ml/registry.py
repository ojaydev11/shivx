"""
MLflow Model Registry
Implements production model versioning and lifecycle management

Features:
- Semantic versioning (major.minor.patch)
- Model promotion workflow (dev → staging → production)
- Model metadata tracking (author, metrics, artifacts)
- Experiment tracking with parameters
- Model lineage and reproducibility
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import json

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Production model registry with MLflow backend

    Implements:
    - Model versioning
    - Promotion workflows
    - Metadata tracking
    - Artifact management
    """

    def __init__(
        self,
        tracking_uri: str = "http://mlflow:5000",
        experiment_name: str = "shivx-trading"
    ):
        """
        Initialize model registry

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Default experiment name
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        # Configure MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Initialize client
        self.client = MlflowClient(tracking_uri=tracking_uri)

        logger.info(f"Model registry initialized: {tracking_uri}")

    def register_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        framework: str = "pytorch",
        metadata: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Register a new model version

        Args:
            model: The trained model object
            model_name: Model name (e.g., "rl_trading_ppo")
            model_type: Type (rl, supervised, unsupervised)
            framework: ML framework (pytorch, sklearn, tensorflow)
            metadata: Additional metadata (author, description, etc.)
            artifacts: Additional artifacts to log (plots, configs, etc.)

        Returns:
            Model version string
        """
        logger.info(f"Registering model: {model_name} ({model_type})")

        with mlflow.start_run() as run:
            # Log model based on framework
            if framework == "pytorch":
                mlflow.pytorch.log_model(model, "model", registered_model_name=model_name)
            elif framework == "sklearn":
                mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
            elif framework == "tensorflow":
                mlflow.tensorflow.log_model(model, "model", registered_model_name=model_name)
            else:
                raise ValueError(f"Unsupported framework: {framework}")

            # Log metadata as parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("framework", framework)
            mlflow.log_param("registered_at", datetime.now().isoformat())

            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (int, float, str, bool)):
                        mlflow.log_param(f"metadata_{key}", value)

            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, artifact_path=name)

            # Get version
            model_version = self._get_latest_version(model_name)

            logger.info(f"Model registered: {model_name} v{model_version}")

            return model_version

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log model metrics

        Args:
            metrics: Dictionary of metric name -> value
            step: Training step/epoch
        """
        for name, value in metrics.items():
            mlflow.log_metric(name, value, step=step)

    def log_training_experiment(
        self,
        model_name: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts_dir: Optional[str] = None
    ) -> str:
        """
        Log a complete training experiment

        Args:
            model_name: Model name
            hyperparameters: Training hyperparameters
            metrics: Final metrics
            artifacts_dir: Directory with training artifacts

        Returns:
            Run ID
        """
        with mlflow.start_run() as run:
            # Log hyperparameters
            for key, value in hyperparameters.items():
                mlflow.log_param(key, value)

            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Log artifacts
            if artifacts_dir and os.path.exists(artifacts_dir):
                mlflow.log_artifacts(artifacts_dir)

            run_id = run.info.run_id
            logger.info(f"Experiment logged: {model_name} (run_id={run_id})")

            return run_id

    def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ):
        """
        Promote model to a new stage

        Args:
            model_name: Model name
            version: Model version
            stage: Target stage (Staging, Production)
            archive_existing: Archive current production model
        """
        logger.info(f"Promoting {model_name} v{version} to {stage}")

        # Archive existing models in target stage
        if archive_existing and stage == "Production":
            self._archive_stage_models(model_name, "Production")

        # Transition to new stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )

        logger.info(f"Model promoted successfully")

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Any:
        """
        Load a model from registry

        Args:
            model_name: Model name
            version: Specific version (e.g., "3")
            stage: Stage to load from (Staging, Production)

        Returns:
            Loaded model
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            # Load latest version
            model_uri = f"models:/{model_name}/latest"

        logger.info(f"Loading model: {model_uri}")

        try:
            model = mlflow.pyfunc.load_model(model_uri)
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_model_info(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get model information and metadata

        Args:
            model_name: Model name
            version: Model version (None for latest)

        Returns:
            Model information dictionary
        """
        try:
            if version:
                model_version = self.client.get_model_version(model_name, version)
            else:
                versions = self.client.search_model_versions(f"name='{model_name}'")
                model_version = max(versions, key=lambda v: int(v.version))

            return {
                "name": model_name,
                "version": model_version.version,
                "stage": model_version.current_stage,
                "creation_timestamp": model_version.creation_timestamp,
                "last_updated_timestamp": model_version.last_updated_timestamp,
                "description": model_version.description,
                "run_id": model_version.run_id,
                "status": model_version.status,
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}

    def list_models(
        self,
        stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all registered models

        Args:
            stage: Filter by stage (Staging, Production, Archived)

        Returns:
            List of model information dictionaries
        """
        models = []

        for rm in self.client.search_registered_models():
            for mv in rm.latest_versions:
                if stage is None or mv.current_stage == stage:
                    models.append({
                        "name": rm.name,
                        "version": mv.version,
                        "stage": mv.current_stage,
                        "creation_timestamp": mv.creation_timestamp,
                    })

        return models

    def compare_models(
        self,
        model_name: str,
        version_a: str,
        version_b: str,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Compare two model versions

        Args:
            model_name: Model name
            version_a: First version
            version_b: Second version
            metrics: Metrics to compare

        Returns:
            Comparison results
        """
        logger.info(f"Comparing {model_name} v{version_a} vs v{version_b}")

        # Get runs for each version
        mv_a = self.client.get_model_version(model_name, version_a)
        mv_b = self.client.get_model_version(model_name, version_b)

        run_a = self.client.get_run(mv_a.run_id)
        run_b = self.client.get_run(mv_b.run_id)

        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "metrics": {},
            "winner": None
        }

        # Compare metrics
        total_diff = 0
        for metric in metrics:
            val_a = run_a.data.metrics.get(metric, 0)
            val_b = run_b.data.metrics.get(metric, 0)

            comparison["metrics"][metric] = {
                "version_a": val_a,
                "version_b": val_b,
                "difference": val_b - val_a,
                "percent_change": ((val_b - val_a) / val_a * 100) if val_a != 0 else 0
            }

            total_diff += val_b - val_a

        # Determine winner
        comparison["winner"] = version_b if total_diff > 0 else version_a

        return comparison

    def rollback_model(
        self,
        model_name: str,
        target_version: str
    ):
        """
        Rollback to a previous model version

        Args:
            model_name: Model name
            target_version: Version to rollback to
        """
        logger.warning(f"Rolling back {model_name} to v{target_version}")

        # Promote target version to production
        self.promote_model(
            model_name=model_name,
            version=target_version,
            stage="Production",
            archive_existing=True
        )

        logger.info("Rollback completed")

    def _get_latest_version(self, model_name: str) -> str:
        """Get latest version number for model"""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if versions:
                return str(max(int(v.version) for v in versions))
            return "1"
        except Exception:
            return "1"

    def _archive_stage_models(self, model_name: str, stage: str):
        """Archive all models in a stage"""
        versions = self.client.search_model_versions(
            f"name='{model_name}' AND current_stage='{stage}'"
        )

        for version in versions:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )

    def delete_model(self, model_name: str, version: str):
        """
        Delete a specific model version

        Args:
            model_name: Model name
            version: Version to delete
        """
        logger.warning(f"Deleting {model_name} v{version}")
        self.client.delete_model_version(model_name, version)
