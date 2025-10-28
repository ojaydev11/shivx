"""
Resource Governor - Multi-Agent Orchestration Framework
========================================================

Enforces resource quotas and limits for agents with:
- Per-agent resource quotas (CPU, memory, API calls, tasks)
- Resource tracking and enforcement
- Quota exhaustion handling
- Dashboard integration for monitoring

Resource Types:
- CPU time (seconds)
- Memory (MB)
- Concurrent tasks
- API calls
- File operations
- Network bandwidth

Features:
- Real-time resource tracking
- Automatic quota enforcement
- Graceful degradation on quota exhaustion
- Prometheus metrics integration
- Audit logging
"""

import time
import logging
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
from uuid import uuid4
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources tracked"""
    CPU_TIME = "cpu_time"  # seconds
    MEMORY = "memory"  # MB
    CONCURRENT_TASKS = "concurrent_tasks"  # count
    API_CALLS = "api_calls"  # count
    FILE_OPERATIONS = "file_operations"  # count
    NETWORK_REQUESTS = "network_requests"  # count


class QuotaStatus(Enum):
    """Quota status"""
    OK = "ok"
    WARNING = "warning"  # >80% used
    CRITICAL = "critical"  # >95% used
    EXHAUSTED = "exhausted"  # 100% used


@dataclass
class ResourceQuota:
    """Resource quota definition"""
    agent_id: str
    resource_type: ResourceType
    limit: float  # Maximum allowed
    warning_threshold: float = 0.8  # Warn at 80%
    critical_threshold: float = 0.95  # Critical at 95%
    reset_period_sec: Optional[float] = None  # Auto-reset period (e.g., 3600 for hourly)


@dataclass
class ResourceUsage:
    """Current resource usage"""
    agent_id: str
    resource_type: ResourceType
    current: float
    limit: float
    percentage: float
    status: QuotaStatus
    timestamp: str
    reset_at: Optional[str] = None


@dataclass
class QuotaViolation:
    """Record of quota violation"""
    violation_id: str
    agent_id: str
    resource_type: ResourceType
    usage: float
    limit: float
    timestamp: str
    action_taken: str  # throttled, denied, alerted


class ResourceGovernor:
    """
    Enforces resource quotas and limits for agents.

    Features:
    - Per-agent resource tracking
    - Automatic quota enforcement
    - Quota reset/renewal
    - Violation logging
    """

    def __init__(self):
        """Initialize resource governor"""

        # Quota definitions
        self.quotas: Dict[str, Dict[ResourceType, ResourceQuota]] = defaultdict(dict)

        # Current usage tracking
        self.usage: Dict[str, Dict[ResourceType, float]] = defaultdict(lambda: defaultdict(float))

        # Quota reset times
        self.reset_times: Dict[str, Dict[ResourceType, datetime]] = defaultdict(dict)

        # Violations
        self.violations: List[QuotaViolation] = []

        # Statistics
        self.total_checks = 0
        self.total_violations = 0

        # Locks for thread safety
        self.lock = threading.Lock()

        # Process for CPU tracking
        self.agent_start_times: Dict[str, float] = {}

        logger.info("ResourceGovernor initialized")

    def set_quota(
        self,
        agent_id: str,
        resource_type: ResourceType,
        limit: float,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
        reset_period_sec: Optional[float] = None
    ):
        """
        Set resource quota for agent.

        Args:
            agent_id: Agent identifier
            resource_type: Type of resource
            limit: Maximum allowed value
            warning_threshold: Warning threshold (0.0-1.0)
            critical_threshold: Critical threshold (0.0-1.0)
            reset_period_sec: Auto-reset period in seconds (None for no reset)
        """
        quota = ResourceQuota(
            agent_id=agent_id,
            resource_type=resource_type,
            limit=limit,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            reset_period_sec=reset_period_sec
        )

        with self.lock:
            self.quotas[agent_id][resource_type] = quota

            # Set initial reset time if applicable
            if reset_period_sec:
                self.reset_times[agent_id][resource_type] = (
                    datetime.now() + timedelta(seconds=reset_period_sec)
                )

        logger.info(
            f"Quota set: {agent_id} - {resource_type.value} = {limit} "
            f"(reset: {reset_period_sec}s)" if reset_period_sec else ""
        )

    def track_usage(
        self,
        agent_id: str,
        resource_type: ResourceType,
        amount: float
    ) -> bool:
        """
        Track resource usage and enforce quota.

        Args:
            agent_id: Agent identifier
            resource_type: Type of resource used
            amount: Amount of resource consumed

        Returns:
            True if usage allowed, False if quota exceeded
        """
        with self.lock:
            # Check for quota reset
            self._check_reset(agent_id, resource_type)

            # Get quota
            if agent_id not in self.quotas or resource_type not in self.quotas[agent_id]:
                # No quota set - allow
                self.usage[agent_id][resource_type] += amount
                return True

            quota = self.quotas[agent_id][resource_type]

            # Check if usage would exceed quota
            new_usage = self.usage[agent_id][resource_type] + amount

            if new_usage > quota.limit:
                # Quota exceeded
                self._record_violation(
                    agent_id,
                    resource_type,
                    new_usage,
                    quota.limit,
                    "denied"
                )

                logger.warning(
                    f"Quota exceeded: {agent_id} - {resource_type.value} "
                    f"({new_usage:.2f}/{quota.limit:.2f})"
                )

                return False

            # Update usage
            self.usage[agent_id][resource_type] = new_usage
            self.total_checks += 1

            # Check thresholds
            percentage = new_usage / quota.limit

            if percentage >= quota.critical_threshold:
                logger.warning(
                    f"Critical quota usage: {agent_id} - {resource_type.value} "
                    f"({percentage:.1%})"
                )
            elif percentage >= quota.warning_threshold:
                logger.info(
                    f"High quota usage: {agent_id} - {resource_type.value} "
                    f"({percentage:.1%})"
                )

            return True

    def check_quota(self, agent_id: str, resource_type: ResourceType) -> ResourceUsage:
        """
        Check current quota usage.

        Args:
            agent_id: Agent identifier
            resource_type: Resource type to check

        Returns:
            ResourceUsage with current status
        """
        with self.lock:
            # Check for reset
            self._check_reset(agent_id, resource_type)

            current = self.usage[agent_id].get(resource_type, 0.0)

            # Get quota
            if agent_id in self.quotas and resource_type in self.quotas[agent_id]:
                quota = self.quotas[agent_id][resource_type]
                limit = quota.limit
                percentage = current / limit if limit > 0 else 0.0

                # Determine status
                if percentage >= 1.0:
                    status = QuotaStatus.EXHAUSTED
                elif percentage >= quota.critical_threshold:
                    status = QuotaStatus.CRITICAL
                elif percentage >= quota.warning_threshold:
                    status = QuotaStatus.WARNING
                else:
                    status = QuotaStatus.OK

                # Get reset time
                reset_at = None
                if agent_id in self.reset_times and resource_type in self.reset_times[agent_id]:
                    reset_at = self.reset_times[agent_id][resource_type].isoformat()

            else:
                # No quota set
                limit = float('inf')
                percentage = 0.0
                status = QuotaStatus.OK
                reset_at = None

            return ResourceUsage(
                agent_id=agent_id,
                resource_type=resource_type,
                current=current,
                limit=limit,
                percentage=percentage,
                status=status,
                timestamp=datetime.now().isoformat(),
                reset_at=reset_at
            )

    def reset_quota(self, agent_id: str, resource_type: Optional[ResourceType] = None):
        """
        Reset quota usage.

        Args:
            agent_id: Agent identifier
            resource_type: Resource type to reset (None for all)
        """
        with self.lock:
            if resource_type:
                # Reset specific resource
                self.usage[agent_id][resource_type] = 0.0

                # Update reset time
                if (agent_id in self.quotas and
                    resource_type in self.quotas[agent_id] and
                    self.quotas[agent_id][resource_type].reset_period_sec):

                    reset_period = self.quotas[agent_id][resource_type].reset_period_sec
                    self.reset_times[agent_id][resource_type] = (
                        datetime.now() + timedelta(seconds=reset_period)
                    )

                logger.info(f"Quota reset: {agent_id} - {resource_type.value}")

            else:
                # Reset all resources
                for rt in ResourceType:
                    self.usage[agent_id][rt] = 0.0

                    if (agent_id in self.quotas and
                        rt in self.quotas[agent_id] and
                        self.quotas[agent_id][rt].reset_period_sec):

                        reset_period = self.quotas[agent_id][rt].reset_period_sec
                        self.reset_times[agent_id][rt] = (
                            datetime.now() + timedelta(seconds=reset_period)
                        )

                logger.info(f"All quotas reset: {agent_id}")

    def _check_reset(self, agent_id: str, resource_type: ResourceType):
        """Check if quota should be auto-reset"""
        if (agent_id in self.reset_times and
            resource_type in self.reset_times[agent_id]):

            reset_time = self.reset_times[agent_id][resource_type]

            if datetime.now() >= reset_time:
                # Auto-reset quota
                self.reset_quota(agent_id, resource_type)
                logger.info(f"Auto-reset quota: {agent_id} - {resource_type.value}")

    def _record_violation(
        self,
        agent_id: str,
        resource_type: ResourceType,
        usage: float,
        limit: float,
        action: str
    ):
        """Record quota violation"""
        violation = QuotaViolation(
            violation_id=str(uuid4()),
            agent_id=agent_id,
            resource_type=resource_type,
            usage=usage,
            limit=limit,
            timestamp=datetime.now().isoformat(),
            action_taken=action
        )

        self.violations.append(violation)
        self.total_violations += 1

        # Log to audit chain
        try:
            from utils.audit_chain import append_jsonl

            log_entry = {
                "event_type": "quota_violation",
                "violation_id": violation.violation_id,
                "agent_id": agent_id,
                "resource_type": resource_type.value,
                "usage": usage,
                "limit": limit,
                "action_taken": action,
                "timestamp": violation.timestamp
            }

            append_jsonl("var/orchestration/quota_violations.ndjson", log_entry)

        except Exception as e:
            logger.error(f"Failed to log violation: {e}")

    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get comprehensive resource status for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Status dictionary with all resource usage
        """
        with self.lock:
            status = {
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat(),
                "resources": {}
            }

            # Get status for all resource types
            for resource_type in ResourceType:
                usage = self.check_quota(agent_id, resource_type)
                status["resources"][resource_type.value] = {
                    "current": usage.current,
                    "limit": usage.limit if usage.limit != float('inf') else None,
                    "percentage": usage.percentage,
                    "status": usage.status.value,
                    "reset_at": usage.reset_at
                }

            return status

    def get_all_agent_status(self) -> List[Dict[str, Any]]:
        """Get resource status for all agents"""
        with self.lock:
            agent_ids = set(self.quotas.keys()) | set(self.usage.keys())
            return [self.get_agent_status(agent_id) for agent_id in agent_ids]

    def get_violations(
        self,
        agent_id: Optional[str] = None,
        resource_type: Optional[ResourceType] = None,
        limit: Optional[int] = None
    ) -> List[QuotaViolation]:
        """
        Get quota violations.

        Args:
            agent_id: Filter by agent
            resource_type: Filter by resource type
            limit: Maximum number of results

        Returns:
            List of violations
        """
        violations = self.violations

        if agent_id:
            violations = [v for v in violations if v.agent_id == agent_id]

        if resource_type:
            violations = [v for v in violations if v.resource_type == resource_type]

        if limit:
            violations = violations[-limit:]

        return violations

    def get_stats(self) -> Dict[str, Any]:
        """Get resource governor statistics"""
        with self.lock:
            total_agents = len(set(self.quotas.keys()) | set(self.usage.keys()))

            # Count quotas by status
            status_counts = defaultdict(int)
            for agent_id in self.usage:
                for resource_type in ResourceType:
                    usage = self.check_quota(agent_id, resource_type)
                    status_counts[usage.status.value] += 1

            return {
                "total_agents": total_agents,
                "total_quotas": sum(len(q) for q in self.quotas.values()),
                "total_checks": self.total_checks,
                "total_violations": self.total_violations,
                "violation_rate": self.total_violations / max(self.total_checks, 1),
                "status_breakdown": dict(status_counts),
                "recent_violations": len([
                    v for v in self.violations
                    if datetime.fromisoformat(v.timestamp) > datetime.now() - timedelta(hours=1)
                ])
            }

    def export_metrics(self) -> Dict[str, Any]:
        """
        Export metrics for Prometheus integration.

        Returns:
            Metrics dictionary suitable for Prometheus
        """
        metrics = {
            "resource_usage": [],
            "quota_status": [],
            "violations": []
        }

        with self.lock:
            # Export usage metrics
            for agent_id in self.usage:
                for resource_type in ResourceType:
                    usage = self.check_quota(agent_id, resource_type)

                    metrics["resource_usage"].append({
                        "agent_id": agent_id,
                        "resource_type": resource_type.value,
                        "value": usage.current,
                        "limit": usage.limit if usage.limit != float('inf') else None
                    })

                    metrics["quota_status"].append({
                        "agent_id": agent_id,
                        "resource_type": resource_type.value,
                        "status": usage.status.value,
                        "percentage": usage.percentage
                    })

            # Export recent violations
            recent_violations = [
                v for v in self.violations
                if datetime.fromisoformat(v.timestamp) > datetime.now() - timedelta(hours=24)
            ]

            for violation in recent_violations:
                metrics["violations"].append({
                    "agent_id": violation.agent_id,
                    "resource_type": violation.resource_type.value,
                    "timestamp": violation.timestamp
                })

        return metrics


# Singleton instance
_resource_governor: Optional[ResourceGovernor] = None


def get_resource_governor() -> ResourceGovernor:
    """Get singleton resource governor instance"""
    global _resource_governor
    if _resource_governor is None:
        _resource_governor = ResourceGovernor()
    return _resource_governor
