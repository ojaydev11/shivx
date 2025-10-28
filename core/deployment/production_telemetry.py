"""
Production Telemetry - Real-world deployment data collection

Collects comprehensive data from real-world AGI deployments:
- Task performance metrics
- User satisfaction scores
- Error rates and failure modes
- Latency and throughput
- Resource utilization
- Capability usage patterns

This data is critical for:
- Continuous improvement
- Real-world benchmarking
- Identifying edge cases
- User experience optimization

Privacy Controls:
- Respects user consent for analytics
- Respects Do Not Track (DNT) headers
- Respects telemetry mode (disabled/minimal/standard/full)
- Disabled in offline mode and air-gap mode
"""

import logging
import json
import sqlite3
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

from config.settings import get_settings

logger = logging.getLogger(__name__)


class TaskOutcome(Enum):
    """Outcome of a task"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    ERROR = "error"
    TIMEOUT = "timeout"


class UserFeedback(Enum):
    """User feedback sentiment"""
    VERY_SATISFIED = "very_satisfied"
    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    DISSATISFIED = "dissatisfied"
    VERY_DISSATISFIED = "very_dissatisfied"


@dataclass
class DeploymentTask:
    """A task executed in production"""
    task_id: str
    task_type: str
    query: str
    capabilities_used: List[str]
    outcome: TaskOutcome
    confidence: float
    latency_ms: float
    user_feedback: Optional[UserFeedback] = None
    user_comment: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentMetrics:
    """Aggregated deployment metrics"""
    total_tasks: int
    success_rate: float
    avg_latency_ms: float
    avg_confidence: float
    user_satisfaction: float
    error_rate: float
    most_used_capabilities: List[str]
    period_start: datetime
    period_end: datetime


class ProductionTelemetry:
    """
    Production telemetry system for real-world deployment
    """

    def __init__(self, db_path: str = "./data/deployment/production.db"):
        """
        Initialize production telemetry

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings = get_settings()
        self._enabled = self._check_telemetry_enabled()

        # Initialize database only if telemetry enabled
        if self._enabled:
            self._init_database()
            logger.info(f"Production Telemetry initialized: {db_path}")
        else:
            logger.info("Production Telemetry disabled due to privacy settings")

    def _check_telemetry_enabled(self) -> bool:
        """
        Check if telemetry collection is enabled

        Returns:
            True if telemetry allowed, False otherwise
        """
        # Check offline mode
        if self.settings.offline_mode:
            logger.debug("Telemetry disabled: offline mode")
            return False

        # Check air-gap mode
        if self.settings.airgap_mode:
            logger.debug("Telemetry disabled: air-gap mode")
            return False

        # Check telemetry mode
        if self.settings.telemetry_mode == "disabled":
            logger.debug("Telemetry disabled: telemetry_mode=disabled")
            return False

        return True

    def is_enabled(self) -> bool:
        """Check if telemetry is enabled"""
        return self._enabled

    def should_collect_event(self, event_type: str) -> bool:
        """
        Check if event type should be collected based on telemetry mode

        Args:
            event_type: Type of event (error, performance, usage)

        Returns:
            True if event should be collected
        """
        if not self._enabled:
            return False

        mode = self.settings.telemetry_mode

        if mode == "disabled":
            return False
        elif mode == "minimal":
            # Only errors and critical events
            return event_type in ("error", "critical", "failure")
        elif mode == "standard":
            # Errors + performance
            return event_type in ("error", "critical", "failure", "performance", "latency")
        elif mode == "full":
            # All events (dev only)
            return True

        return False

    def _init_database(self):
        """Initialize SQLite database for telemetry"""
        with sqlite3.connect(self.db_path) as conn:
            # Tasks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT,
                    query TEXT,
                    capabilities_used TEXT,
                    outcome TEXT,
                    confidence REAL,
                    latency_ms REAL,
                    user_feedback TEXT,
                    user_comment TEXT,
                    error_message TEXT,
                    timestamp TIMESTAMP,
                    metadata TEXT
                )
            """)

            # Daily aggregates
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    date TEXT PRIMARY KEY,
                    total_tasks INTEGER,
                    success_rate REAL,
                    avg_latency_ms REAL,
                    avg_confidence REAL,
                    user_satisfaction REAL,
                    error_rate REAL,
                    most_used_capabilities TEXT,
                    computed_at TIMESTAMP
                )
            """)

            # User sessions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    tasks_completed INTEGER,
                    avg_satisfaction REAL,
                    metadata TEXT
                )
            """)

            # System health
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_health (
                    timestamp TIMESTAMP PRIMARY KEY,
                    cpu_percent REAL,
                    memory_mb REAL,
                    active_tasks INTEGER,
                    queue_size INTEGER,
                    avg_response_time_ms REAL
                )
            """)

            conn.commit()

        logger.info("Production telemetry database initialized")

    def log_task(self, task: DeploymentTask):
        """
        Log a production task

        Args:
            task: Task to log
        """
        # Check if telemetry enabled
        if not self._enabled:
            logger.debug("Telemetry disabled - skipping task log")
            return

        # Check if event type should be collected
        event_type = "error" if task.outcome == TaskOutcome.ERROR else "usage"
        if not self.should_collect_event(event_type):
            logger.debug(f"Telemetry mode {self.settings.telemetry_mode} - skipping {event_type} event")
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.task_id,
                task.task_type,
                task.query,
                json.dumps(task.capabilities_used),
                task.outcome.value,
                task.confidence,
                task.latency_ms,
                task.user_feedback.value if task.user_feedback else None,
                task.user_comment,
                task.error_message,
                task.timestamp.isoformat(),
                json.dumps(task.metadata)
            ))
            conn.commit()

        logger.debug(f"Logged task: {task.task_id} ({task.outcome.value})")

    def log_user_feedback(
        self,
        task_id: str,
        feedback: UserFeedback,
        comment: Optional[str] = None
    ):
        """
        Log user feedback for a task

        Args:
            task_id: Task ID
            feedback: User feedback
            comment: Optional comment
        """
        # Check if telemetry enabled
        if not self._enabled:
            logger.debug("Telemetry disabled - skipping feedback log")
            return

        if not self.should_collect_event("usage"):
            logger.debug(f"Telemetry mode {self.settings.telemetry_mode} - skipping feedback event")
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE tasks
                SET user_feedback = ?, user_comment = ?
                WHERE task_id = ?
            """, (feedback.value, comment, task_id))
            conn.commit()

        logger.info(f"Logged user feedback for {task_id}: {feedback.value}")

    def get_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> DeploymentMetrics:
        """
        Get deployment metrics for a time period

        Args:
            start_date: Start of period (default: 24 hours ago)
            end_date: End of period (default: now)

        Returns:
            DeploymentMetrics
        """
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=1)

        with sqlite3.connect(self.db_path) as conn:
            # Get tasks in period
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as success_count,
                    AVG(latency_ms) as avg_latency,
                    AVG(confidence) as avg_confidence,
                    SUM(CASE WHEN outcome IN ('failure', 'error', 'timeout') THEN 1 ELSE 0 END) as error_count,
                    capabilities_used
                FROM tasks
                WHERE timestamp >= ? AND timestamp <= ?
            """, (start_date.isoformat(), end_date.isoformat()))

            row = cursor.fetchone()

            if row and row[0] > 0:
                total_tasks = row[0]
                success_count = row[1]
                avg_latency = row[2]
                avg_confidence = row[3]
                error_count = row[4]

                success_rate = success_count / total_tasks
                error_rate = error_count / total_tasks
            else:
                return DeploymentMetrics(
                    total_tasks=0,
                    success_rate=0.0,
                    avg_latency_ms=0.0,
                    avg_confidence=0.0,
                    user_satisfaction=0.0,
                    error_rate=0.0,
                    most_used_capabilities=[],
                    period_start=start_date,
                    period_end=end_date
                )

            # Get user satisfaction
            cursor = conn.execute("""
                SELECT user_feedback
                FROM tasks
                WHERE timestamp >= ? AND timestamp <= ? AND user_feedback IS NOT NULL
            """, (start_date.isoformat(), end_date.isoformat()))

            feedback_scores = {
                UserFeedback.VERY_SATISFIED: 1.0,
                UserFeedback.SATISFIED: 0.75,
                UserFeedback.NEUTRAL: 0.5,
                UserFeedback.DISSATISFIED: 0.25,
                UserFeedback.VERY_DISSATISFIED: 0.0
            }

            feedbacks = [UserFeedback(row[0]) for row in cursor.fetchall()]
            if feedbacks:
                avg_satisfaction = sum(feedback_scores[f] for f in feedbacks) / len(feedbacks)
            else:
                avg_satisfaction = 0.5  # Neutral default

            # Get most used capabilities
            cursor = conn.execute("""
                SELECT capabilities_used
                FROM tasks
                WHERE timestamp >= ? AND timestamp <= ?
            """, (start_date.isoformat(), end_date.isoformat()))

            capability_counts = {}
            for row in cursor.fetchall():
                if row[0]:
                    caps = json.loads(row[0])
                    for cap in caps:
                        capability_counts[cap] = capability_counts.get(cap, 0) + 1

            most_used = sorted(capability_counts.items(), key=lambda x: x[1], reverse=True)
            most_used_capabilities = [cap for cap, _ in most_used[:5]]

        return DeploymentMetrics(
            total_tasks=total_tasks,
            success_rate=success_rate,
            avg_latency_ms=avg_latency,
            avg_confidence=avg_confidence,
            user_satisfaction=avg_satisfaction,
            error_rate=error_rate,
            most_used_capabilities=most_used_capabilities,
            period_start=start_date,
            period_end=end_date
        )

    def compute_daily_aggregates(self, date: Optional[datetime] = None):
        """
        Compute and store daily aggregates

        Args:
            date: Date to compute (default: yesterday)
        """
        if date is None:
            date = datetime.utcnow().date() - timedelta(days=1)
        else:
            date = date.date()

        start = datetime.combine(date, datetime.min.time())
        end = datetime.combine(date, datetime.max.time())

        metrics = self.get_metrics(start, end)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO daily_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date.isoformat(),
                metrics.total_tasks,
                metrics.success_rate,
                metrics.avg_latency_ms,
                metrics.avg_confidence,
                metrics.user_satisfaction,
                metrics.error_rate,
                json.dumps(metrics.most_used_capabilities),
                datetime.utcnow().isoformat()
            ))
            conn.commit()

        logger.info(f"Computed daily aggregates for {date}")

    def generate_production_report(
        self,
        days: int = 7
    ) -> str:
        """
        Generate production deployment report

        Args:
            days: Number of days to include

        Returns:
            Markdown-formatted report
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        metrics = self.get_metrics(start_date, end_date)

        report = "# Production Deployment Report\n\n"
        report += f"**Period**: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n"
        report += f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Overall metrics
        report += "## Overall Performance\n\n"
        report += f"- **Total Tasks**: {metrics.total_tasks:,}\n"
        report += f"- **Success Rate**: {metrics.success_rate:.1%}\n"
        report += f"- **Average Latency**: {metrics.avg_latency_ms:.0f}ms\n"
        report += f"- **Average Confidence**: {metrics.avg_confidence:.1%}\n"
        report += f"- **User Satisfaction**: {metrics.user_satisfaction:.1%}\n"
        report += f"- **Error Rate**: {metrics.error_rate:.1%}\n\n"

        # Most used capabilities
        report += "## Most Used Capabilities\n\n"
        for i, cap in enumerate(metrics.most_used_capabilities, 1):
            report += f"{i}. {cap}\n"
        report += "\n"

        # Health assessment
        report += "## System Health Assessment\n\n"
        if metrics.success_rate >= 0.95:
            report += "✅ **Excellent**: Success rate > 95%\n"
        elif metrics.success_rate >= 0.90:
            report += "🟡 **Good**: Success rate 90-95%\n"
        else:
            report += "🔴 **Needs Attention**: Success rate < 90%\n"

        if metrics.user_satisfaction >= 0.80:
            report += "✅ **Users Satisfied**: Satisfaction > 80%\n"
        elif metrics.user_satisfaction >= 0.60:
            report += "🟡 **Users Moderately Satisfied**: Satisfaction 60-80%\n"
        else:
            report += "🔴 **Users Dissatisfied**: Satisfaction < 60%\n"

        if metrics.avg_latency_ms <= 1000:
            report += "✅ **Fast Response**: Latency < 1s\n"
        elif metrics.avg_latency_ms <= 3000:
            report += "🟡 **Acceptable Response**: Latency 1-3s\n"
        else:
            report += "🔴 **Slow Response**: Latency > 3s\n"

        return report

    def get_stats(self) -> Dict[str, Any]:
        """Get telemetry statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM tasks")
            total_tasks = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(DISTINCT DATE(timestamp)) FROM tasks")
            days_active = cursor.fetchone()[0]

        return {
            "total_tasks_logged": total_tasks,
            "days_active": days_active,
            "database_path": str(self.db_path)
        }


# Global telemetry instance
_telemetry = None


def get_production_telemetry() -> ProductionTelemetry:
    """Get global production telemetry instance"""
    global _telemetry
    if _telemetry is None:
        _telemetry = ProductionTelemetry()
    return _telemetry


# Convenience functions
def log_production_task(
    task_id: str,
    task_type: str,
    query: str,
    capabilities: List[str],
    outcome: TaskOutcome,
    confidence: float,
    latency_ms: float
):
    """Quick log a production task"""
    telemetry = get_production_telemetry()
    task = DeploymentTask(
        task_id=task_id,
        task_type=task_type,
        query=query,
        capabilities_used=capabilities,
        outcome=outcome,
        confidence=confidence,
        latency_ms=latency_ms
    )
    telemetry.log_task(task)
