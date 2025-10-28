"""
Memory daemon for background memory maintenance.

Runs autonomously to:
- Consolidate memories
- Auto-tag events
- Detect and prune hallucinations/anomalies
- Create backups/snapshots
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Thread
from typing import Optional

from loguru import logger

from memory.consolidation.consolidator import MemoryConsolidator
from memory.encoders.text_encoder import TextEncoder
from memory.graph_store.store import MemoryGraphStore


class MemoryDaemon:
    """
    Background daemon for memory maintenance.

    Runs in a separate thread, performing periodic consolidation
    and maintenance tasks.
    """

    def __init__(
        self,
        graph_store: MemoryGraphStore,
        text_encoder: TextEncoder,
        consolidation_interval_hours: int = 24,
        snapshot_interval_hours: int = 168,  # Weekly
    ):
        """
        Initialize memory daemon.

        Args:
            graph_store: Memory graph store
            text_encoder: Text encoder
            consolidation_interval_hours: Hours between consolidations
            snapshot_interval_hours: Hours between snapshots
        """
        self.graph_store = graph_store
        self.text_encoder = text_encoder
        self.consolidation_interval = timedelta(hours=consolidation_interval_hours)
        self.snapshot_interval = timedelta(hours=snapshot_interval_hours)

        self.consolidator = MemoryConsolidator(
            graph_store=graph_store,
            text_encoder=text_encoder,
        )

        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._last_consolidation: Optional[datetime] = None
        self._last_snapshot: Optional[datetime] = None

        logger.info(
            f"Memory daemon initialized "
            f"(consolidate every {consolidation_interval_hours}h, "
            f"snapshot every {snapshot_interval_hours}h)"
        )

    def start(self) -> None:
        """Start the daemon in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Memory daemon already running")
            return

        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True, name="MemoryDaemon")
        self._thread.start()
        logger.info("Memory daemon started")

    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the daemon.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        if not self._thread or not self._thread.is_alive():
            logger.warning("Memory daemon not running")
            return

        logger.info("Stopping memory daemon...")
        self._stop_event.set()
        self._thread.join(timeout=timeout)

        if self._thread.is_alive():
            logger.error("Memory daemon did not stop gracefully")
        else:
            logger.info("Memory daemon stopped")

    def _run(self) -> None:
        """Main daemon loop."""
        logger.info("Memory daemon running")

        # Initial delay
        time.sleep(60)  # Wait 1 minute before first run

        while not self._stop_event.is_set():
            try:
                # Check if consolidation is due
                if self._should_consolidate():
                    self._run_consolidation()

                # Check if snapshot is due
                if self._should_snapshot():
                    self._create_snapshot()

                # Auto-tag recent memories
                self._auto_tag_recent()

                # Sleep for a while (check every minute)
                self._stop_event.wait(timeout=60)

            except Exception as e:
                logger.error(f"Error in memory daemon: {e}", exc_info=True)
                # Continue running despite errors
                self._stop_event.wait(timeout=60)

    def _should_consolidate(self) -> bool:
        """Check if consolidation is due."""
        if self._last_consolidation is None:
            return True

        elapsed = datetime.utcnow() - self._last_consolidation
        return elapsed >= self.consolidation_interval

    def _run_consolidation(self) -> None:
        """Run memory consolidation."""
        logger.info("Running memory consolidation...")
        try:
            report = self.consolidator.consolidate()
            self._last_consolidation = datetime.utcnow()
            logger.info(
                f"Consolidation complete: "
                f"merged={report.nodes_merged}, "
                f"pruned={report.nodes_pruned}"
            )
        except Exception as e:
            logger.error(f"Consolidation failed: {e}", exc_info=True)

    def _should_snapshot(self) -> bool:
        """Check if snapshot is due."""
        if self._last_snapshot is None:
            # Create initial snapshot after 1 day
            return (
                self._last_consolidation is not None
                and (datetime.utcnow() - self._last_consolidation) > timedelta(days=1)
            )

        elapsed = datetime.utcnow() - self._last_snapshot
        return elapsed >= self.snapshot_interval

    def _create_snapshot(self) -> None:
        """Create memory graph snapshot."""
        logger.info("Creating memory snapshot...")
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            snapshot_dir = Path("./data/memory/snapshots")
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            snapshot_path = snapshot_dir / f"memory_snapshot_{timestamp}.json"
            self.graph_store.export_graph(str(snapshot_path))

            self._last_snapshot = datetime.utcnow()
            logger.info(f"Snapshot created: {snapshot_path}")

            # Keep only last 5 snapshots
            self._cleanup_old_snapshots(snapshot_dir, keep=5)

        except Exception as e:
            logger.error(f"Snapshot creation failed: {e}", exc_info=True)

    def _cleanup_old_snapshots(self, snapshot_dir: Path, keep: int = 5) -> None:
        """Remove old snapshots, keeping only the most recent."""
        snapshots = sorted(snapshot_dir.glob("memory_snapshot_*.json"))
        if len(snapshots) > keep:
            for old_snapshot in snapshots[:-keep]:
                old_snapshot.unlink()
                logger.debug(f"Removed old snapshot: {old_snapshot.name}")

    def _auto_tag_recent(self) -> None:
        """Auto-tag recent memories (placeholder)."""
        # This would use NER or other extraction to auto-tag
        # For now, it's a placeholder
        pass

    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._thread is not None and self._thread.is_alive()

    def get_status(self) -> dict:
        """Get daemon status."""
        return {
            "running": self.is_running(),
            "last_consolidation": (
                self._last_consolidation.isoformat()
                if self._last_consolidation
                else None
            ),
            "last_snapshot": (
                self._last_snapshot.isoformat() if self._last_snapshot else None
            ),
            "next_consolidation": (
                (self._last_consolidation + self.consolidation_interval).isoformat()
                if self._last_consolidation
                else "pending"
            ),
            "next_snapshot": (
                (self._last_snapshot + self.snapshot_interval).isoformat()
                if self._last_snapshot
                else "pending"
            ),
        }
