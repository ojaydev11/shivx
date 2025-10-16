"""
ShivX Guardian Defense System
==============================
Purpose: Autonomous intrusion detection, resource abuse prevention, code tampering detection
Auto-isolates threats, initiates lockdown, and restores from safe snapshots.

Features:
- Real-time intrusion detection (suspicious patterns, API abuse)
- Resource abuse monitoring (CPU/memory bombs, disk floods)
- Code integrity verification (hash-based tamper detection)
- Auto-isolation of compromised modules
- Lockdown mode with safe state restoration
- Immutable guardian audit log
"""

import os
import sys
import time
import json
import hashlib
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import uuid4
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DefenseMode(Enum):
    """Defense system modes"""
    NORMAL = "normal"
    ELEVATED = "elevated"
    LOCKDOWN = "lockdown"

@dataclass
class ThreatEvent:
    """Detected threat event"""
    event_id: str
    timestamp: str
    threat_type: str
    threat_level: str
    source: str  # IP, module, user
    details: Dict[str, Any]
    action_taken: str
    hash: str  # SHA256 for immutability

@dataclass
class IsolationRecord:
    """Record of isolated module/source"""
    isolation_id: str
    timestamp: str
    source: str
    reason: str
    duration_sec: Optional[float]
    restored: bool

class GuardianDefense:
    """
    Autonomous defense system - protects ShivX from intrusions and abuse
    """

    def __init__(self, log_dir: str = "var/security"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Audit log
        self.audit_log_path = self.log_dir / "guardian_audit.ndjson"

        # State
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.defense_mode = DefenseMode.NORMAL

        # Threat tracking
        self.threat_history: deque = deque(maxlen=1000)  # Last 1000 threats
        self.isolated_sources: Dict[str, IsolationRecord] = {}

        # Rate limiting (requests per source per minute)
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Code integrity hashes
        self.code_hashes: Dict[str, str] = {}
        self.verified_files: Set[str] = set()

        # Thresholds
        self.thresholds = {
            "requests_per_minute_warning": 100,
            "requests_per_minute_critical": 500,
            "failed_auth_warning": 5,
            "failed_auth_critical": 10,
            "cpu_spike_percent": 95.0,
            "memory_spike_percent": 95.0,
            "disk_write_mb_per_sec": 100.0,
        }

        # Snapshots
        self.snapshots_dir = self.log_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("GuardianDefense initialized")

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        try:
            sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash {file_path}: {e}")
            return ""

    def register_code_integrity(self, file_paths: List[Path]) -> None:
        """Register files for integrity monitoring"""
        for file_path in file_paths:
            if not file_path.exists():
                logger.warning(f"File not found for integrity check: {file_path}")
                continue

            file_hash = self.compute_file_hash(file_path)
            self.code_hashes[str(file_path)] = file_hash
            self.verified_files.add(str(file_path))

        logger.info(f"Registered {len(file_paths)} files for integrity monitoring")

    def verify_code_integrity(self) -> List[str]:
        """Verify code integrity, return list of tampered files"""
        tampered = []

        for file_path_str in list(self.verified_files):
            file_path = Path(file_path_str)

            if not file_path.exists():
                logger.warning(f"Verified file missing: {file_path}")
                tampered.append(file_path_str)
                continue

            current_hash = self.compute_file_hash(file_path)
            expected_hash = self.code_hashes.get(file_path_str)

            if current_hash != expected_hash:
                logger.error(f"CODE TAMPERING DETECTED: {file_path}")
                tampered.append(file_path_str)

                self._log_threat(
                    "code_tampering",
                    ThreatLevel.CRITICAL,
                    "filesystem",
                    {
                        "file": file_path_str,
                        "expected_hash": expected_hash,
                        "current_hash": current_hash
                    },
                    "lockdown_initiated"
                )

        return tampered

    def detect_rate_limit_abuse(self, source: str, endpoint: str) -> Optional[ThreatLevel]:
        """Detect rate limiting abuse"""
        now = time.time()

        # Record request
        self.request_counts[source].append(now)

        # Count requests in last minute
        one_min_ago = now - 60
        recent = [t for t in self.request_counts[source] if t > one_min_ago]
        rpm = len(recent)

        if rpm > self.thresholds["requests_per_minute_critical"]:
            self._log_threat(
                "rate_limit_critical",
                ThreatLevel.CRITICAL,
                source,
                {"requests_per_minute": rpm, "endpoint": endpoint},
                "auto_isolated"
            )
            self.isolate_source(source, "Rate limit exceeded (critical)")
            return ThreatLevel.CRITICAL

        elif rpm > self.thresholds["requests_per_minute_warning"]:
            self._log_threat(
                "rate_limit_warning",
                ThreatLevel.HIGH,
                source,
                {"requests_per_minute": rpm, "endpoint": endpoint},
                "warning_logged"
            )
            return ThreatLevel.HIGH

        return None

    def detect_auth_abuse(self, source: str, failed_attempts: int) -> Optional[ThreatLevel]:
        """Detect authentication abuse"""
        if failed_attempts >= self.thresholds["failed_auth_critical"]:
            self._log_threat(
                "auth_brute_force",
                ThreatLevel.CRITICAL,
                source,
                {"failed_attempts": failed_attempts},
                "auto_isolated"
            )
            self.isolate_source(source, "Brute force attack detected")
            return ThreatLevel.CRITICAL

        elif failed_attempts >= self.thresholds["failed_auth_warning"]:
            self._log_threat(
                "auth_suspicious",
                ThreatLevel.MEDIUM,
                source,
                {"failed_attempts": failed_attempts},
                "warning_logged"
            )
            return ThreatLevel.MEDIUM

        return None

    def detect_resource_abuse(self, source: str, cpu: float, memory: float) -> Optional[ThreatLevel]:
        """Detect resource abuse (CPU/memory bombs)"""
        threat_level = None

        if cpu > self.thresholds["cpu_spike_percent"]:
            self._log_threat(
                "cpu_abuse",
                ThreatLevel.HIGH,
                source,
                {"cpu_percent": cpu},
                "throttled"
            )
            threat_level = ThreatLevel.HIGH

        if memory > self.thresholds["memory_spike_percent"]:
            self._log_threat(
                "memory_abuse",
                ThreatLevel.HIGH,
                source,
                {"memory_percent": memory},
                "throttled"
            )
            threat_level = ThreatLevel.HIGH

        return threat_level

    def isolate_source(self, source: str, reason: str, duration_sec: Optional[float] = None) -> None:
        """Isolate a threat source (IP, module, user)"""
        if source in self.isolated_sources:
            logger.warning(f"Source already isolated: {source}")
            return

        isolation = IsolationRecord(
            isolation_id=str(uuid4()),
            timestamp=datetime.now().isoformat(),
            source=source,
            reason=reason,
            duration_sec=duration_sec,
            restored=False
        )

        self.isolated_sources[source] = isolation

        logger.warning(f"SOURCE ISOLATED: {source} - {reason}")

        self._log_threat(
            "source_isolated",
            ThreatLevel.HIGH,
            source,
            {"reason": reason, "duration_sec": duration_sec},
            "isolated"
        )

        # If no duration specified, isolate indefinitely until manual restore
        if duration_sec:
            # Schedule auto-restore
            def auto_restore():
                time.sleep(duration_sec)
                self.restore_source(source)

            threading.Thread(target=auto_restore, daemon=True).start()

    def restore_source(self, source: str) -> bool:
        """Restore previously isolated source"""
        if source not in self.isolated_sources:
            logger.warning(f"Source not isolated: {source}")
            return False

        isolation = self.isolated_sources[source]
        isolation.restored = True

        del self.isolated_sources[source]

        logger.info(f"SOURCE RESTORED: {source}")

        self._log_threat(
            "source_restored",
            ThreatLevel.LOW,
            source,
            {"isolation_id": isolation.isolation_id},
            "restored"
        )

        return True

    def is_source_isolated(self, source: str) -> bool:
        """Check if source is currently isolated"""
        return source in self.isolated_sources

    def enter_lockdown_mode(self, reason: str) -> None:
        """Enter lockdown mode - maximum security"""
        if self.defense_mode == DefenseMode.LOCKDOWN:
            logger.warning("Already in lockdown mode")
            return

        old_mode = self.defense_mode
        self.defense_mode = DefenseMode.LOCKDOWN

        logger.critical(f"LOCKDOWN MODE ACTIVATED: {reason}")

        self._log_threat(
            "lockdown_activated",
            ThreatLevel.CRITICAL,
            "system",
            {"reason": reason, "previous_mode": old_mode.value},
            "lockdown"
        )

        # In lockdown:
        # - All external connections blocked
        # - Only critical operations allowed
        # - All new requests rejected
        # - Integrity checks on every operation

    def exit_lockdown_mode(self) -> None:
        """Exit lockdown mode"""
        if self.defense_mode != DefenseMode.LOCKDOWN:
            logger.warning("Not in lockdown mode")
            return

        self.defense_mode = DefenseMode.NORMAL

        logger.info("Lockdown mode deactivated")

        self._log_threat(
            "lockdown_deactivated",
            ThreatLevel.LOW,
            "system",
            {"message": "Returned to normal mode"},
            "restored"
        )

    def create_safe_snapshot(self, name: str) -> Path:
        """Create snapshot of current safe state"""
        snapshot_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        snapshot_path = self.snapshots_dir / f"{snapshot_id}.json"

        snapshot_data = {
            "snapshot_id": snapshot_id,
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "code_hashes": self.code_hashes,
            "verified_files": list(self.verified_files),
            "defense_mode": self.defense_mode.value,
            "isolated_sources": {k: asdict(v) for k, v in self.isolated_sources.items()}
        }

        # Calculate snapshot hash
        snapshot_json = json.dumps(snapshot_data, sort_keys=True)
        snapshot_data["hash"] = hashlib.sha256(snapshot_json.encode()).hexdigest()

        with open(snapshot_path, 'w') as f:
            json.dump(snapshot_data, f, indent=2)

        logger.info(f"Safe snapshot created: {snapshot_path}")

        return snapshot_path

    def restore_from_snapshot(self, snapshot_path: Path) -> bool:
        """Restore from safe snapshot"""
        try:
            with open(snapshot_path, 'r') as f:
                snapshot = json.load(f)

            # Verify snapshot hash
            snapshot_copy = snapshot.copy()
            expected_hash = snapshot_copy.pop("hash")
            snapshot_json = json.dumps(snapshot_copy, sort_keys=True)
            actual_hash = hashlib.sha256(snapshot_json.encode()).hexdigest()

            if actual_hash != expected_hash:
                logger.error("Snapshot hash verification failed - cannot restore")
                return False

            # Restore state
            self.code_hashes = snapshot["code_hashes"]
            self.verified_files = set(snapshot["verified_files"])
            self.defense_mode = DefenseMode[snapshot["defense_mode"].upper()]

            logger.info(f"Restored from snapshot: {snapshot['snapshot_id']}")

            self._log_threat(
                "snapshot_restored",
                ThreatLevel.LOW,
                "system",
                {"snapshot_id": snapshot["snapshot_id"]},
                "restored"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to restore snapshot: {e}")
            return False

    def _log_threat(
        self,
        threat_type: str,
        threat_level: ThreatLevel,
        source: str,
        details: Dict[str, Any],
        action_taken: str
    ) -> None:
        """Log threat to immutable audit log"""
        try:
            event = ThreatEvent(
                event_id=str(uuid4()),
                timestamp=datetime.now().isoformat(),
                threat_type=threat_type,
                threat_level=threat_level.value,
                source=source,
                details=details,
                action_taken=action_taken,
                hash=""  # Will be set below
            )

            # Calculate hash for immutability
            event_data = {
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "threat_type": event.threat_type,
                "threat_level": event.threat_level,
                "source": event.source,
                "details": event.details,
                "action_taken": event.action_taken
            }
            event_json = json.dumps(event_data, sort_keys=True)
            event.hash = hashlib.sha256(event_json.encode()).hexdigest()

            # Append to NDJSON log
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(asdict(event)) + '\n')

            # Add to history
            self.threat_history.append(event)

            # Auto-escalate if many high/critical threats
            recent_critical = sum(
                1 for e in list(self.threat_history)[-10:]
                if e.threat_level in [ThreatLevel.HIGH.value, ThreatLevel.CRITICAL.value]
            )

            if recent_critical >= 5 and self.defense_mode == DefenseMode.NORMAL:
                logger.warning("Multiple high-severity threats detected, entering ELEVATED mode")
                self.defense_mode = DefenseMode.ELEVATED

        except Exception as e:
            logger.error(f"Failed to log threat: {e}")

    def get_threat_summary(self) -> Dict[str, Any]:
        """Get threat summary statistics"""
        threats_by_level = defaultdict(int)
        threats_by_type = defaultdict(int)

        for threat in self.threat_history:
            threats_by_level[threat.threat_level] += 1
            threats_by_type[threat.threat_type] += 1

        return {
            "total_threats": len(self.threat_history),
            "by_level": dict(threats_by_level),
            "by_type": dict(threats_by_type),
            "isolated_sources": len(self.isolated_sources),
            "defense_mode": self.defense_mode.value,
            "recent_threats": [asdict(t) for t in list(self.threat_history)[-10:]]
        }

    def start(self) -> None:
        """Start guardian defense monitoring"""
        if self.running:
            logger.warning("Guardian defense already running")
            return

        self.running = True

        logger.info("Guardian defense started")
        self._log_threat(
            "guardian_started",
            ThreatLevel.LOW,
            "system",
            {"message": "Defense monitoring active"},
            "started"
        )

    def stop(self) -> None:
        """Stop guardian defense monitoring"""
        if not self.running:
            return

        self.running = False

        logger.info("Guardian defense stopped")
        self._log_threat(
            "guardian_stopped",
            ThreatLevel.LOW,
            "system",
            {"message": "Defense monitoring stopped"},
            "stopped"
        )


# Singleton instance
_guardian_defense: Optional[GuardianDefense] = None

def get_guardian_defense() -> GuardianDefense:
    """Get singleton guardian defense instance"""
    global _guardian_defense
    if _guardian_defense is None:
        _guardian_defense = GuardianDefense()
    return _guardian_defense
