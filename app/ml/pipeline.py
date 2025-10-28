"""
ML Pipeline Orchestration
End-to-end ML pipeline management

Pipeline stages:
1. Data Collection
2. Feature Engineering
3. Model Training
4. Model Evaluation
5. Model Deployment

Features:
- Pipeline versioning
- DAG execution
- Failure recovery
- Pipeline monitoring
- Scheduling integration
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Pipeline execution stages"""
    DATA_COLLECTION = "data_collection"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_DEPLOYMENT = "model_deployment"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineStep:
    """Single pipeline step"""
    name: str
    stage: PipelineStage
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout_seconds: int = 3600


@dataclass
class PipelineRun:
    """Pipeline execution run"""
    run_id: str
    pipeline_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"
    current_stage: PipelineStage = PipelineStage.DATA_COLLECTION
    stages_completed: List[str] = field(default_factory=list)
    stages_failed: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None


class MLPipeline:
    """
    ML Pipeline orchestrator

    Features:
    - DAG-based execution
    - Dependency management
    - Failure recovery
    - Artifact tracking
    """

    def __init__(
        self,
        pipeline_name: str,
        description: str = "",
        version: str = "1.0"
    ):
        """
        Initialize ML pipeline

        Args:
            pipeline_name: Pipeline name
            description: Pipeline description
            version: Pipeline version
        """
        self.pipeline_name = pipeline_name
        self.description = description
        self.version = version

        # Pipeline steps
        self.steps: List[PipelineStep] = []

        # Execution history
        self.runs: Dict[str, PipelineRun] = {}

        logger.info(f"ML Pipeline initialized: {pipeline_name} v{version}")

    def add_step(
        self,
        name: str,
        stage: PipelineStage,
        function: Callable,
        dependencies: Optional[List[str]] = None,
        retry_count: int = 3,
        timeout_seconds: int = 3600
    ):
        """
        Add step to pipeline

        Args:
            name: Step name
            stage: Pipeline stage
            function: Step function
            dependencies: List of step names this depends on
            retry_count: Number of retries on failure
            timeout_seconds: Step timeout
        """
        step = PipelineStep(
            name=name,
            stage=stage,
            function=function,
            dependencies=dependencies or [],
            retry_count=retry_count,
            timeout_seconds=timeout_seconds
        )

        self.steps.append(step)

        logger.info(f"Added step: {name} ({stage})")

    async def run(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> PipelineRun:
        """
        Execute pipeline

        Args:
            config: Pipeline configuration

        Returns:
            Pipeline run result
        """
        run_id = f"run_{datetime.now().timestamp()}"

        logger.info(f"Starting pipeline run: {run_id}")

        run = PipelineRun(
            run_id=run_id,
            pipeline_name=self.pipeline_name,
            started_at=datetime.now()
        )

        self.runs[run_id] = run

        try:
            # Execute pipeline stages in order
            context = {"config": config or {}}

            for stage in PipelineStage:
                if stage in [PipelineStage.COMPLETED, PipelineStage.FAILED]:
                    continue

                run.current_stage = stage
                logger.info(f"Executing stage: {stage}")

                # Get steps for this stage
                stage_steps = [s for s in self.steps if s.stage == stage]

                # Execute steps respecting dependencies
                for step in stage_steps:
                    if not await self._check_dependencies(step, run):
                        logger.warning(f"Skipping {step.name} - dependencies not met")
                        continue

                    # Execute step with retries
                    success = await self._execute_step(step, context, run)

                    if success:
                        run.stages_completed.append(step.name)
                    else:
                        run.stages_failed.append(step.name)
                        raise Exception(f"Step failed: {step.name}")

            # Pipeline completed successfully
            run.status = "completed"
            run.current_stage = PipelineStage.COMPLETED
            run.completed_at = datetime.now()

            logger.info(f"Pipeline run completed: {run_id}")

        except Exception as e:
            logger.error(f"Pipeline run failed: {e}")
            run.status = "failed"
            run.current_stage = PipelineStage.FAILED
            run.error_message = str(e)
            run.completed_at = datetime.now()

        return run

    async def _execute_step(
        self,
        step: PipelineStep,
        context: Dict[str, Any],
        run: PipelineRun
    ) -> bool:
        """Execute a pipeline step with retries"""
        logger.info(f"Executing step: {step.name}")

        for attempt in range(step.retry_count):
            try:
                # Execute step function
                result = await asyncio.wait_for(
                    self._run_step_function(step.function, context),
                    timeout=step.timeout_seconds
                )

                # Store result in context
                context[step.name] = result

                # Store artifacts in run
                if isinstance(result, dict):
                    run.artifacts[step.name] = result

                logger.info(f"Step completed: {step.name}")
                return True

            except asyncio.TimeoutError:
                logger.warning(f"Step timeout: {step.name} (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Step error: {step.name} - {e} (attempt {attempt + 1})")

            if attempt < step.retry_count - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        logger.error(f"Step failed after {step.retry_count} attempts: {step.name}")
        return False

    async def _run_step_function(
        self,
        function: Callable,
        context: Dict[str, Any]
    ) -> Any:
        """Run step function (sync or async)"""
        if asyncio.iscoroutinefunction(function):
            return await function(context)
        else:
            return function(context)

    async def _check_dependencies(
        self,
        step: PipelineStep,
        run: PipelineRun
    ) -> bool:
        """Check if step dependencies are met"""
        for dep in step.dependencies:
            if dep not in run.stages_completed:
                return False
        return True

    def get_run_status(self, run_id: str) -> Optional[PipelineRun]:
        """Get pipeline run status"""
        return self.runs.get(run_id)

    def list_runs(
        self,
        status: Optional[str] = None,
        limit: int = 10
    ) -> List[PipelineRun]:
        """
        List pipeline runs

        Args:
            status: Filter by status
            limit: Max runs to return

        Returns:
            List of pipeline runs
        """
        runs = list(self.runs.values())

        if status:
            runs = [r for r in runs if r.status == status]

        # Sort by start time (newest first)
        runs.sort(key=lambda r: r.started_at, reverse=True)

        return runs[:limit]

    def visualize_pipeline(self) -> str:
        """Generate pipeline DAG visualization"""
        lines = [f"Pipeline: {self.pipeline_name} v{self.version}"]
        lines.append("=" * 60)

        for stage in PipelineStage:
            if stage in [PipelineStage.COMPLETED, PipelineStage.FAILED]:
                continue

            stage_steps = [s for s in self.steps if s.stage == stage]
            if stage_steps:
                lines.append(f"\n{stage.value.upper()}:")
                for step in stage_steps:
                    deps = f" (depends on: {', '.join(step.dependencies)})" if step.dependencies else ""
                    lines.append(f"  - {step.name}{deps}")

        return "\n".join(lines)


# =============================================================================
# Example Pipeline Steps
# =============================================================================

async def data_collection_step(context: Dict[str, Any]) -> Dict[str, Any]:
    """Collect training data"""
    logger.info("Collecting training data...")

    # Simulate data collection
    await asyncio.sleep(1)

    return {
        "data_path": "/tmp/training_data.csv",
        "num_samples": 10000,
        "features": ["rsi", "macd", "volume_trend"]
    }


async def feature_engineering_step(context: Dict[str, Any]) -> Dict[str, Any]:
    """Engineer features"""
    logger.info("Engineering features...")

    data_info = context.get("data_collection")

    # Simulate feature engineering
    await asyncio.sleep(1)

    return {
        "features_computed": 15,
        "feature_names": data_info["features"] + ["feature_4", "feature_5"]
    }


async def model_training_step(context: Dict[str, Any]) -> Dict[str, Any]:
    """Train model"""
    logger.info("Training model...")

    # Simulate training
    await asyncio.sleep(2)

    return {
        "model_path": "/tmp/model.pt",
        "training_loss": 0.125,
        "validation_loss": 0.145,
        "epochs": 50
    }


async def model_evaluation_step(context: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate model"""
    logger.info("Evaluating model...")

    # Simulate evaluation
    await asyncio.sleep(1)

    return {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1_score": 0.85
    }


async def model_deployment_step(context: Dict[str, Any]) -> Dict[str, Any]:
    """Deploy model"""
    logger.info("Deploying model...")

    eval_results = context.get("model_evaluation")

    # Check if model meets deployment criteria
    if eval_results["accuracy"] >= 0.80:
        logger.info("Model meets deployment criteria")

        # Simulate deployment
        await asyncio.sleep(1)

        return {
            "deployed": True,
            "model_version": "1.2.0",
            "deployment_time": datetime.now().isoformat()
        }
    else:
        logger.warning("Model does not meet deployment criteria")
        return {
            "deployed": False,
            "reason": "Accuracy below threshold"
        }


# =============================================================================
# Example Pipeline Creation
# =============================================================================

def create_trading_model_pipeline() -> MLPipeline:
    """Create end-to-end trading model pipeline"""
    pipeline = MLPipeline(
        pipeline_name="trading_model_pipeline",
        description="End-to-end ML pipeline for trading models",
        version="1.0"
    )

    # Add pipeline steps
    pipeline.add_step(
        name="data_collection",
        stage=PipelineStage.DATA_COLLECTION,
        function=data_collection_step
    )

    pipeline.add_step(
        name="feature_engineering",
        stage=PipelineStage.FEATURE_ENGINEERING,
        function=feature_engineering_step,
        dependencies=["data_collection"]
    )

    pipeline.add_step(
        name="model_training",
        stage=PipelineStage.MODEL_TRAINING,
        function=model_training_step,
        dependencies=["feature_engineering"]
    )

    pipeline.add_step(
        name="model_evaluation",
        stage=PipelineStage.MODEL_EVALUATION,
        function=model_evaluation_step,
        dependencies=["model_training"]
    )

    pipeline.add_step(
        name="model_deployment",
        stage=PipelineStage.MODEL_DEPLOYMENT,
        function=model_deployment_step,
        dependencies=["model_evaluation"]
    )

    logger.info("Trading model pipeline created")

    return pipeline


# =============================================================================
# Example Usage
# =============================================================================

async def example_usage():
    """Example of running ML pipeline"""
    # Create pipeline
    pipeline = create_trading_model_pipeline()

    # Visualize
    print(pipeline.visualize_pipeline())
    print("\n")

    # Run pipeline
    run = await pipeline.run(config={"model_type": "rl_trading"})

    # Print results
    print(f"Pipeline run: {run.run_id}")
    print(f"Status: {run.status}")
    print(f"Duration: {(run.completed_at - run.started_at).total_seconds():.2f}s")
    print(f"Stages completed: {len(run.stages_completed)}")
    print(f"Artifacts: {list(run.artifacts.keys())}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
