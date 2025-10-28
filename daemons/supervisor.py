"""
Supervisor daemon for managing all AGI background processes.

Coordinates:
- Memory daemon
- Learning daemon
- Reflection daemon
- Telemetry daemon
- Scheduler daemon
"""

import time
from datetime import datetime
from threading import Event, Thread
from typing import Dict, List, Optional

from loguru import logger


class DaemonStatus:
    """Status of a daemon."""

    def __init__(self, name: str):
        self.name = name
        self.running = False
        self.last_heartbeat: Optional[datetime] = None
        self.restart_count = 0
        self.error_count = 0


class Supervisor:
    """
    Supervisor for AGI daemons.

    Features:
    - Starts and stops all daemons
    - Health checks
    - Auto-restart on failure
    - Status monitoring
    """

    def __init__(
        self,
        restart_policy: str = "on-failure",
        max_restarts: int = 5,
        restart_interval_seconds: int = 60,
        health_check_interval: int = 30,
    ):
        """
        Initialize supervisor.

        Args:
            restart_policy: always, on-failure, or never
            max_restarts: Maximum restart attempts
            restart_interval_seconds: Seconds between restarts
            health_check_interval: Health check interval
        """
        self.restart_policy = restart_policy
        self.max_restarts = max_restarts
        self.restart_interval = restart_interval_seconds
        self.health_check_interval = health_check_interval

        self.daemons: Dict[str, DaemonStatus] = {}
        self.daemon_instances: Dict[str, Any] = {}

        self._stop_event = Event()
        self._thread: Optional[Thread] = None

        logger.info(
            f"Supervisor initialized: "
            f"restart_policy={restart_policy}, max_restarts={max_restarts}"
        )

    def register_daemon(self, name: str, daemon_instance: Any) -> None:
        """
        Register daemon with supervisor.

        Args:
            name: Daemon name
            daemon_instance: Daemon object with start/stop/is_running methods
        """
        self.daemons[name] = DaemonStatus(name)
        self.daemon_instances[name] = daemon_instance
        logger.info(f"Registered daemon: {name}")

    def start(self) -> None:
        """Start supervisor and all registered daemons."""
        logger.info("Starting supervisor...")

        # Start all daemons
        for name, instance in self.daemon_instances.items():
            self._start_daemon(name)

        # Start supervisor thread
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True, name="Supervisor")
        self._thread.start()

        logger.info("Supervisor started")

    def stop(self, timeout: float = 30.0) -> None:
        """Stop supervisor and all daemons."""
        logger.info("Stopping supervisor...")

        # Stop supervisor thread
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)

        # Stop all daemons
        for name, instance in self.daemon_instances.items():
            self._stop_daemon(name)

        logger.info("Supervisor stopped")

    def _run(self) -> None:
        """Main supervisor loop."""
        logger.info("Supervisor monitoring active")

        while not self._stop_event.is_set():
            try:
                # Health check all daemons
                for name in self.daemons:
                    self._health_check(name)

                # Sleep for interval
                self._stop_event.wait(timeout=self.health_check_interval)

            except Exception as e:
                logger.error(f"Supervisor error: {e}", exc_info=True)
                self._stop_event.wait(timeout=10)

    def _start_daemon(self, name: str) -> bool:
        """Start a daemon."""
        try:
            status = self.daemons[name]
            instance = self.daemon_instances[name]

            logger.info(f"Starting daemon: {name}")
            instance.start()

            status.running = True
            status.last_heartbeat = datetime.utcnow()

            logger.info(f"Daemon started: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to start daemon {name}: {e}", exc_info=True)
            status.error_count += 1
            return False

    def _stop_daemon(self, name: str) -> None:
        """Stop a daemon."""
        try:
            status = self.daemons[name]
            instance = self.daemon_instances[name]

            logger.info(f"Stopping daemon: {name}")
            instance.stop()

            status.running = False
            logger.info(f"Daemon stopped: {name}")

        except Exception as e:
            logger.error(f"Failed to stop daemon {name}: {e}", exc_info=True)

    def _health_check(self, name: str) -> None:
        """Health check a daemon."""
        status = self.daemons[name]
        instance = self.daemon_instances[name]

        # Check if running
        is_running = instance.is_running()

        if is_running:
            status.last_heartbeat = datetime.utcnow()
            status.running = True
        else:
            status.running = False

            # Attempt restart if policy allows
            if self.restart_policy == "on-failure":
                if status.restart_count < self.max_restarts:
                    logger.warning(f"Daemon {name} is down. Restarting...")
                    if self._start_daemon(name):
                        status.restart_count += 1
                    time.sleep(self.restart_interval)
                else:
                    logger.error(
                        f"Daemon {name} exceeded max restarts ({self.max_restarts})"
                    )

    def get_status(self) -> Dict[str, Any]:
        """Get status of all daemons."""
        daemon_statuses = {}
        for name, status in self.daemons.items():
            daemon_statuses[name] = {
                "running": status.running,
                "last_heartbeat": (
                    status.last_heartbeat.isoformat()
                    if status.last_heartbeat
                    else None
                ),
                "restart_count": status.restart_count,
                "error_count": status.error_count,
            }

        return {
            "supervisor_running": self._thread and self._thread.is_alive(),
            "daemons": daemon_statuses,
        }

    def health_check_all(self) -> Dict[str, bool]:
        """Quick health check of all daemons."""
        health = {}
        for name, instance in self.daemon_instances.items():
            health[name] = instance.is_running()
        return health
