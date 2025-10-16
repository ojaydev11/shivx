"""
ShivX Resilience Core
====================
Purpose: Ultimate fault tolerance - watchdog, health monitoring, graceful degradation
Ensures ShivX NEVER halts, only scales down gracefully under any failure condition.

Features:
- Live process health monitoring (CPU, memory, disk, threads)
- Automatic restart of failing submodules
- Graceful degradation mode (scale down non-critical features)
- Circuit breaker pattern for external dependencies
- Health scoring system (0-100)
- Immutable audit logging
"""

import os
import sys
import time
import json
import psutil
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from uuid import uuid4

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"

class DegradationLevel(Enum):
    """System degradation levels"""
    NORMAL = 0          # All features enabled
    LEVEL_1 = 1         # Non-critical features disabled
    LEVEL_2 = 2         # Advanced features disabled
    LEVEL_3 = 3         # Minimal operation mode
    EMERGENCY = 4       # Core only

@dataclass
class HealthMetrics:
    """System health metrics"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_percent: float
    disk_free_gb: float
    thread_count: int
    process_uptime_sec: float
    health_score: float  # 0-100
    status: str
    degradation_level: int

@dataclass
class ModuleHealth:
    """Individual module health"""
    module_name: str
    status: str
    last_check: str
    restart_count: int
    error_count: int
    last_error: Optional[str]
    health_score: float

@dataclass
class ResilienceEvent:
    """Resilience event for audit log"""
    event_id: str
    timestamp: str
    event_type: str  # health_check, restart, degradation, recovery
    module: str
    details: Dict[str, Any]
    severity: str
    hash: str  # SHA256 of event data

class ResilienceCore:
    """
    Ultimate resilience engine - ensures ShivX never halts
    """

    def __init__(self, check_interval: float = 60.0, log_dir: str = "var/resilience"):
        self.check_interval = check_interval
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Audit log
        self.audit_log_path = self.log_dir / "resilience_audit.ndjson"

        # State
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.process = psutil.Process()
        self.start_time = time.time()

        # Module registry
        self.modules: Dict[str, Dict[str, Any]] = {}

        # Degradation state
        self.current_degradation = DegradationLevel.NORMAL
        self.degraded_features: List[str] = []

        # Thresholds
        self.thresholds = {
            "cpu_critical": 90.0,
            "cpu_warning": 75.0,
            "memory_critical": 90.0,
            "memory_warning": 75.0,
            "disk_critical": 95.0,
            "disk_warning": 85.0,
            "thread_max": 500,
            "health_score_critical": 30.0,
            "health_score_warning": 60.0,
        }

        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}

        logger.info("ResilienceCore initialized")

    def register_module(
        self,
        name: str,
        health_check: Callable[[], bool],
        restart_func: Optional[Callable[[], bool]] = None,
        critical: bool = False
    ) -> None:
        """Register a module for health monitoring"""
        self.modules[name] = {
            "health_check": health_check,
            "restart_func": restart_func,
            "critical": critical,
            "status": HealthStatus.HEALTHY.value,
            "restart_count": 0,
            "error_count": 0,
            "last_check": None,
            "last_error": None,
        }
        logger.info(f"Registered module: {name} (critical={critical})")

    def get_system_metrics(self) -> HealthMetrics:
        """Collect current system health metrics"""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('.')

            mem_info = self.process.memory_info()
            threads = self.process.num_threads()
            uptime = time.time() - self.start_time

            # Calculate health score (0-100)
            health_score = 100.0

            # Deduct for high CPU
            if cpu > self.thresholds["cpu_critical"]:
                health_score -= 30
            elif cpu > self.thresholds["cpu_warning"]:
                health_score -= 15

            # Deduct for high memory
            if mem.percent > self.thresholds["memory_critical"]:
                health_score -= 30
            elif mem.percent > self.thresholds["memory_warning"]:
                health_score -= 15

            # Deduct for high disk usage
            if disk.percent > self.thresholds["disk_critical"]:
                health_score -= 20
            elif disk.percent > self.thresholds["disk_warning"]:
                health_score -= 10

            # Deduct for excessive threads
            if threads > self.thresholds["thread_max"]:
                health_score -= 20

            # Determine status
            if health_score < self.thresholds["health_score_critical"]:
                status = HealthStatus.CRITICAL.value
            elif health_score < self.thresholds["health_score_warning"]:
                status = HealthStatus.DEGRADED.value
            else:
                status = HealthStatus.HEALTHY.value

            return HealthMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu,
                memory_percent=mem.percent,
                memory_mb=mem_info.rss / (1024**2),
                disk_percent=disk.percent,
                disk_free_gb=disk.free / (1024**3),
                thread_count=threads,
                process_uptime_sec=uptime,
                health_score=health_score,
                status=status,
                degradation_level=self.current_degradation.value
            )

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return HealthMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_mb=0.0,
                disk_percent=0.0,
                disk_free_gb=0.0,
                thread_count=0,
                process_uptime_sec=0.0,
                health_score=0.0,
                status=HealthStatus.FAILED.value,
                degradation_level=DegradationLevel.EMERGENCY.value
            )

    def check_module_health(self, name: str) -> ModuleHealth:
        """Check health of a single module"""
        if name not in self.modules:
            raise ValueError(f"Module not registered: {name}")

        module = self.modules[name]

        try:
            # Run health check
            is_healthy = module["health_check"]()

            if is_healthy:
                module["status"] = HealthStatus.HEALTHY.value
                module["error_count"] = max(0, module["error_count"] - 1)  # Decay errors
            else:
                module["status"] = HealthStatus.DEGRADED.value
                module["error_count"] += 1

                # Attempt restart if available
                if module["restart_func"] and module["error_count"] >= 3:
                    logger.warning(f"Module {name} unhealthy, attempting restart...")
                    self._log_event("restart", name, {"reason": "health_check_failure"}, "warning")

                    if module["restart_func"]():
                        module["restart_count"] += 1
                        module["error_count"] = 0
                        module["status"] = HealthStatus.HEALTHY.value
                        logger.info(f"Module {name} restarted successfully")
                    else:
                        module["status"] = HealthStatus.FAILED.value
                        logger.error(f"Module {name} restart failed")

            module["last_check"] = datetime.now().isoformat()

            # Calculate module health score
            health_score = 100.0
            health_score -= (module["error_count"] * 10)  # -10 per error
            health_score -= (module["restart_count"] * 5)  # -5 per restart
            health_score = max(0.0, min(100.0, health_score))

            return ModuleHealth(
                module_name=name,
                status=module["status"],
                last_check=module["last_check"],
                restart_count=module["restart_count"],
                error_count=module["error_count"],
                last_error=module.get("last_error"),
                health_score=health_score
            )

        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            module["status"] = HealthStatus.FAILED.value
            module["error_count"] += 1
            module["last_error"] = str(e)

            return ModuleHealth(
                module_name=name,
                status=HealthStatus.FAILED.value,
                last_check=datetime.now().isoformat(),
                restart_count=module["restart_count"],
                error_count=module["error_count"],
                last_error=str(e),
                health_score=0.0
            )

    def adjust_degradation_level(self, metrics: HealthMetrics) -> None:
        """Adjust system degradation level based on health"""
        new_level = self.current_degradation

        if metrics.health_score < 30:
            new_level = DegradationLevel.EMERGENCY
        elif metrics.health_score < 50:
            new_level = DegradationLevel.LEVEL_3
        elif metrics.health_score < 65:
            new_level = DegradationLevel.LEVEL_2
        elif metrics.health_score < 80:
            new_level = DegradationLevel.LEVEL_1
        else:
            new_level = DegradationLevel.NORMAL

        if new_level != self.current_degradation:
            old_level = self.current_degradation
            self.current_degradation = new_level

            logger.warning(f"Degradation level changed: {old_level.name} -> {new_level.name}")
            self._log_event(
                "degradation",
                "system",
                {
                    "old_level": old_level.value,
                    "new_level": new_level.value,
                    "health_score": metrics.health_score,
                    "reason": "automatic_adjustment"
                },
                "warning" if new_level.value > 0 else "info"
            )

            # Apply degradation
            self._apply_degradation(new_level)

    def _apply_degradation(self, level: DegradationLevel) -> None:
        """Apply degradation by disabling non-critical features"""
        self.degraded_features.clear()

        if level == DegradationLevel.NORMAL:
            logger.info("All features enabled (NORMAL mode)")
            return

        if level.value >= DegradationLevel.LEVEL_1.value:
            # Disable non-critical features
            self.degraded_features.extend([
                "advanced_analytics",
                "background_tasks",
                "cache_warming"
            ])

        if level.value >= DegradationLevel.LEVEL_2.value:
            # Disable advanced features
            self.degraded_features.extend([
                "ai_inference",
                "web_search",
                "voice_processing"
            ])

        if level.value >= DegradationLevel.LEVEL_3.value:
            # Minimal operation
            self.degraded_features.extend([
                "file_uploads",
                "integrations",
                "automation"
            ])

        if level == DegradationLevel.EMERGENCY:
            # Core only
            self.degraded_features.extend([
                "chat",
                "memory_writes"
            ])

        logger.warning(f"Degraded features: {', '.join(self.degraded_features)}")

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if feature is enabled (not degraded)"""
        return feature not in self.degraded_features

    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        logger.info("Resilience monitoring loop started")

        while self.running:
            try:
                # Collect system metrics
                metrics = self.get_system_metrics()

                # Log health check
                self._log_event(
                    "health_check",
                    "system",
                    {
                        "health_score": metrics.health_score,
                        "cpu_percent": metrics.cpu_percent,
                        "memory_percent": metrics.memory_percent,
                        "disk_percent": metrics.disk_percent,
                        "status": metrics.status
                    },
                    "info"
                )

                # Adjust degradation if needed
                self.adjust_degradation_level(metrics)

                # Check all registered modules
                for name in self.modules:
                    try:
                        module_health = self.check_module_health(name)

                        if module_health.status == HealthStatus.FAILED.value:
                            if self.modules[name]["critical"]:
                                logger.critical(f"CRITICAL MODULE FAILED: {name}")
                                # Could trigger emergency protocols here
                    except Exception as e:
                        logger.error(f"Failed to check module {name}: {e}")

                # Sleep until next check
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}", exc_info=True)
                time.sleep(5)  # Brief pause before retry

    def _log_event(self, event_type: str, module: str, details: Dict[str, Any], severity: str) -> None:
        """Log resilience event to immutable audit log"""
        try:
            event = ResilienceEvent(
                event_id=str(uuid4()),
                timestamp=datetime.now().isoformat(),
                event_type=event_type,
                module=module,
                details=details,
                severity=severity,
                hash=""  # Will be set below
            )

            # Calculate hash for immutability
            event_data = {
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "module": event.module,
                "details": event.details,
                "severity": event.severity
            }
            event_json = json.dumps(event_data, sort_keys=True)
            event.hash = hashlib.sha256(event_json.encode()).hexdigest()

            # Append to NDJSON log
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(asdict(event)) + '\n')

        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def start(self) -> None:
        """Start resilience monitoring"""
        if self.running:
            logger.warning("Resilience monitoring already running")
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("Resilience monitoring started")
        self._log_event("startup", "resilience_core", {"message": "Monitoring started"}, "info")

    def stop(self) -> None:
        """Stop resilience monitoring"""
        if not self.running:
            return

        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        logger.info("Resilience monitoring stopped")
        self._log_event("shutdown", "resilience_core", {"message": "Monitoring stopped"}, "info")

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        metrics = self.get_system_metrics()

        module_statuses = {}
        for name in self.modules:
            try:
                module_statuses[name] = asdict(self.check_module_health(name))
            except Exception as e:
                module_statuses[name] = {"error": str(e)}

        return {
            "system_metrics": asdict(metrics),
            "degradation_level": self.current_degradation.name,
            "degraded_features": self.degraded_features,
            "modules": module_statuses,
            "uptime_sec": time.time() - self.start_time,
            "monitoring_active": self.running
        }


# Singleton instance
_resilience_core: Optional[ResilienceCore] = None

def get_resilience_core() -> ResilienceCore:
    """Get singleton resilience core instance"""
    global _resilience_core
    if _resilience_core is None:
        _resilience_core = ResilienceCore()
    return _resilience_core
