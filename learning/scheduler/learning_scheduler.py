"""
Learning scheduler for continuous lifelong learning.

Decides when to train adapters based on idle time and performance metrics.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger

from learning.experience_buffer.buffer import ExperienceBuffer
from learning.online_adapter.adapter import OnlineAdapter


class LearningScheduler:
    """
    Scheduler for continuous learning sessions.

    Trains adapters during idle periods with anti-regression checks.
    """

    def __init__(
        self,
        adapter: OnlineAdapter,
        experience_buffer: ExperienceBuffer,
        mode: str = "idle",
        idle_threshold_seconds: int = 300,
        batch_size: int = 16,
        max_steps_per_session: int = 100,
        learning_rate: float = 0.0001,
    ):
        """
        Initialize learning scheduler.

        Args:
            adapter: Online adapter
            experience_buffer: Experience buffer
            mode: idle, scheduled, or continuous
            idle_threshold_seconds: Idle time before training
            batch_size: Training batch size
            max_steps_per_session: Max training steps per session
            learning_rate: Learning rate
        """
        self.adapter = adapter
        self.experience_buffer = experience_buffer
        self.mode = mode
        self.idle_threshold = timedelta(seconds=idle_threshold_seconds)
        self.batch_size = batch_size
        self.max_steps_per_session = max_steps_per_session
        self.learning_rate = learning_rate

        self.last_activity = datetime.utcnow()
        self.last_training = None
        self.training_history: List[Dict] = []

        logger.info(
            f"Learning scheduler initialized: mode={mode}, "
            f"idle_threshold={idle_threshold_seconds}s"
        )

    def record_activity(self) -> None:
        """Record user activity (resets idle timer)."""
        self.last_activity = datetime.utcnow()

    def is_idle(self) -> bool:
        """Check if system is idle."""
        elapsed = datetime.utcnow() - self.last_activity
        return elapsed >= self.idle_threshold

    def should_train(self) -> bool:
        """Determine if training should occur."""
        if len(self.experience_buffer) < self.batch_size:
            return False

        if self.mode == "idle":
            return self.is_idle()
        elif self.mode == "scheduled":
            # Train every hour (simplified)
            if self.last_training is None:
                return True
            elapsed = datetime.utcnow() - self.last_training
            return elapsed >= timedelta(hours=1)
        elif self.mode == "continuous":
            return True
        return False

    def train_session(
        self,
        adapter_id: Optional[str] = None,
        task_name: str = "general",
    ) -> Dict:
        """
        Run a training session.

        Args:
            adapter_id: Adapter to train (creates new if None)
            task_name: Task name for new adapter

        Returns:
            Training session report
        """
        logger.info(f"Starting training session for task: {task_name}")
        start_time = datetime.utcnow()

        # Create or select adapter
        if adapter_id is None:
            adapter_id = self.adapter.create_adapter(task_name)

        # Training loop
        total_loss = 0.0
        steps = 0

        for step in range(self.max_steps_per_session):
            # Sample batch
            experiences = self.experience_buffer.sample(self.batch_size)
            if not experiences:
                break

            # Prepare batch
            batch = {
                "inputs": [exp.input_data for exp in experiences],
                "targets": [exp.target for exp in experiences],
            }

            # Training step
            metrics = self.adapter.train_step(
                adapter_id=adapter_id,
                batch=batch,
                learning_rate=self.learning_rate,
            )

            total_loss += metrics["loss"]
            steps += 1

            # Early stopping if loss plateaus
            if steps > 10 and metrics["loss"] < 0.01:
                logger.info("Early stopping: loss converged")
                break

        # Session report
        duration = (datetime.utcnow() - start_time).total_seconds()
        avg_loss = total_loss / steps if steps > 0 else 0.0

        report = {
            "adapter_id": adapter_id,
            "task_name": task_name,
            "steps": steps,
            "avg_loss": avg_loss,
            "duration_seconds": duration,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.training_history.append(report)
        self.last_training = datetime.utcnow()

        logger.info(
            f"Training session complete: "
            f"steps={steps}, avg_loss={avg_loss:.4f}, "
            f"duration={duration:.1f}s"
        )

        return report

    def evaluate_adapter(
        self, adapter_id: str, eval_data: List[Dict]
    ) -> Dict:
        """
        Evaluate adapter against evaluation set.

        Args:
            adapter_id: Adapter to evaluate
            eval_data: Evaluation dataset

        Returns:
            Evaluation metrics
        """
        return self.adapter.evaluate(adapter_id, eval_data)

    def check_regression(
        self,
        adapter_id: str,
        golden_tasks: List[Dict],
        min_pass_rate: float = 0.95,
    ) -> bool:
        """
        Check if adapter causes regression on golden tasks.

        Args:
            adapter_id: Adapter to check
            golden_tasks: Golden task set
            min_pass_rate: Minimum pass rate

        Returns:
            True if no regression detected
        """
        logger.info(f"Checking regression for adapter {adapter_id}")

        metrics = self.evaluate_adapter(adapter_id, golden_tasks)
        pass_rate = metrics.get("accuracy", 0.0)

        if pass_rate >= min_pass_rate:
            logger.info(f"No regression: pass_rate={pass_rate:.3f}")
            return True
        else:
            logger.warning(f"Regression detected: pass_rate={pass_rate:.3f}")
            return False

    def promote_adapter(self, adapter_id: str) -> bool:
        """
        Promote adapter to production.

        Args:
            adapter_id: Adapter to promote

        Returns:
            True if promoted successfully
        """
        logger.info(f"Promoting adapter {adapter_id}")
        adapter_path = self.adapter.save_adapter(adapter_id)
        logger.info(f"Adapter promoted: {adapter_path}")
        return True

    def rollback_adapter(self, adapter_id: str) -> None:
        """
        Rollback adapter (remove from history).

        Args:
            adapter_id: Adapter to rollback
        """
        logger.warning(f"Rolling back adapter {adapter_id}")
        # In production, would restore previous version
        pass

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        if not self.training_history:
            return {"sessions": 0}

        return {
            "sessions": len(self.training_history),
            "total_steps": sum(s["steps"] for s in self.training_history),
            "avg_loss": (
                sum(s["avg_loss"] for s in self.training_history)
                / len(self.training_history)
            ),
            "last_training": (
                self.training_history[-1]["timestamp"]
                if self.training_history
                else None
            ),
        }
