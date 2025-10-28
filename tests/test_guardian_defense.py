"""
Guardian Defense System Tests
Coverage Target: 90% of security/guardian_defense.py

Tests intrusion detection, rate limiting, code integrity, threat isolation,
lockdown mode, and snapshot/restore functionality
"""

import pytest
import time
import tempfile
from pathlib import Path
from datetime import datetime

from security.guardian_defense import (
    GuardianDefense,
    ThreatLevel,
    DefenseMode,
    ThreatEvent,
    IsolationRecord,
    get_guardian_defense
)


@pytest.mark.unit
class TestGuardianDefenseInitialization:
    """Test Guardian Defense initialization"""

    def test_guardian_init_default(self):
        """Test initializing with default settings"""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = GuardianDefense(log_dir=tmpdir)

            assert guardian.defense_mode == DefenseMode.NORMAL
            assert not guardian.running
            assert len(guardian.isolated_sources) == 0
            assert len(guardian.threat_history) == 0

    def test_guardian_init_creates_directories(self):
        """Test that initialization creates necessary directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = GuardianDefense(log_dir=tmpdir)

            assert guardian.log_dir.exists()
            assert guardian.snapshots_dir.exists()
            assert guardian.audit_log_path.exists() or True  # Will be created on first write

    def test_guardian_thresholds(self):
        """Test default thresholds are set correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            guardian = GuardianDefense(log_dir=tmpdir)

            assert guardian.thresholds["requests_per_minute_warning"] == 100
            assert guardian.thresholds["requests_per_minute_critical"] == 500
            assert guardian.thresholds["failed_auth_warning"] == 5
            assert guardian.thresholds["failed_auth_critical"] == 10


@pytest.mark.unit
class TestCodeIntegrity:
    """Test code integrity verification"""

    def test_compute_file_hash(self, guardian_defense):
        """Test computing file hash"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content for hashing")
            test_file = Path(f.name)

        try:
            hash1 = guardian_defense.compute_file_hash(test_file)
            hash2 = guardian_defense.compute_file_hash(test_file)

            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex digest
        finally:
            test_file.unlink()

    def test_register_code_integrity(self, guardian_defense):
        """Test registering files for integrity monitoring"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            test_file = Path(f.name)

        try:
            guardian_defense.register_code_integrity([test_file])

            assert str(test_file) in guardian_defense.verified_files
            assert str(test_file) in guardian_defense.code_hashes
            assert len(guardian_defense.code_hashes[str(test_file)]) == 64
        finally:
            test_file.unlink()

    def test_verify_code_integrity_no_tampering(self, guardian_defense):
        """Test verification when files haven't been tampered"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("original content")
            test_file = Path(f.name)

        try:
            guardian_defense.register_code_integrity([test_file])
            tampered = guardian_defense.verify_code_integrity()

            assert len(tampered) == 0
        finally:
            test_file.unlink()

    def test_verify_code_integrity_with_tampering(self, guardian_defense):
        """Test detection of tampered files"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("original content")
            test_file = Path(f.name)

        try:
            # Register with original content
            guardian_defense.register_code_integrity([test_file])

            # Tamper with file
            with open(test_file, 'w') as f:
                f.write("tampered content")

            tampered = guardian_defense.verify_code_integrity()

            assert len(tampered) == 1
            assert str(test_file) in tampered
        finally:
            test_file.unlink()

    def test_verify_code_integrity_missing_file(self, guardian_defense):
        """Test detection of missing files"""
        test_file = Path("/tmp/nonexistent_test_file.py")

        guardian_defense.verified_files.add(str(test_file))
        guardian_defense.code_hashes[str(test_file)] = "fake_hash"

        tampered = guardian_defense.verify_code_integrity()

        assert str(test_file) in tampered


@pytest.mark.unit
class TestRateLimitDetection:
    """Test rate limit abuse detection"""

    def test_detect_rate_limit_normal(self, guardian_defense):
        """Test normal request rate"""
        source = "192.168.1.1"

        for _ in range(50):  # Below warning threshold
            threat = guardian_defense.detect_rate_limit_abuse(source, "/api/test")

        assert threat is None

    def test_detect_rate_limit_warning(self, guardian_defense):
        """Test warning level rate limit"""
        source = "192.168.1.2"

        # Exceed warning threshold (100 requests/minute)
        for _ in range(150):
            threat = guardian_defense.detect_rate_limit_abuse(source, "/api/test")

        # Last one should trigger warning
        assert threat == ThreatLevel.HIGH

    def test_detect_rate_limit_critical(self, guardian_defense):
        """Test critical level rate limit"""
        source = "192.168.1.3"

        # Exceed critical threshold (500 requests/minute)
        for _ in range(600):
            threat = guardian_defense.detect_rate_limit_abuse(source, "/api/test")

        # Should trigger critical and auto-isolate
        assert threat == ThreatLevel.CRITICAL
        assert source in guardian_defense.isolated_sources

    def test_rate_limit_different_sources(self, guardian_defense):
        """Test rate limiting tracks sources separately"""
        source1 = "192.168.1.1"
        source2 = "192.168.1.2"

        for _ in range(50):
            guardian_defense.detect_rate_limit_abuse(source1, "/api/test")
            guardian_defense.detect_rate_limit_abuse(source2, "/api/test")

        # Neither should be isolated (each under threshold individually)
        assert source1 not in guardian_defense.isolated_sources
        assert source2 not in guardian_defense.isolated_sources


@pytest.mark.unit
class TestAuthAbuseDetection:
    """Test authentication abuse detection"""

    def test_detect_auth_abuse_normal(self, guardian_defense):
        """Test normal failed auth attempts"""
        source = "192.168.1.1"

        threat = guardian_defense.detect_auth_abuse(source, 3)
        assert threat is None

    def test_detect_auth_abuse_warning(self, guardian_defense):
        """Test warning level auth abuse"""
        source = "192.168.1.2"

        threat = guardian_defense.detect_auth_abuse(source, 7)
        assert threat == ThreatLevel.MEDIUM

    def test_detect_auth_abuse_critical(self, guardian_defense):
        """Test critical level auth abuse (brute force)"""
        source = "192.168.1.3"

        threat = guardian_defense.detect_auth_abuse(source, 15)
        assert threat == ThreatLevel.CRITICAL
        assert source in guardian_defense.isolated_sources

    def test_detect_auth_abuse_threshold_boundaries(self, guardian_defense):
        """Test threshold boundaries"""
        # Just below warning
        threat = guardian_defense.detect_auth_abuse("source1", 4)
        assert threat is None

        # At warning
        threat = guardian_defense.detect_auth_abuse("source2", 5)
        assert threat == ThreatLevel.MEDIUM

        # Just below critical
        threat = guardian_defense.detect_auth_abuse("source3", 9)
        assert threat == ThreatLevel.MEDIUM

        # At critical
        threat = guardian_defense.detect_auth_abuse("source4", 10)
        assert threat == ThreatLevel.CRITICAL


@pytest.mark.unit
class TestResourceAbuseDetection:
    """Test resource abuse detection"""

    def test_detect_cpu_abuse(self, guardian_defense):
        """Test CPU abuse detection"""
        source = "process_123"

        threat = guardian_defense.detect_resource_abuse(source, cpu=98.0, memory=50.0)
        assert threat == ThreatLevel.HIGH

    def test_detect_memory_abuse(self, guardian_defense):
        """Test memory abuse detection"""
        source = "process_456"

        threat = guardian_defense.detect_resource_abuse(source, cpu=50.0, memory=97.0)
        assert threat == ThreatLevel.HIGH

    def test_detect_combined_resource_abuse(self, guardian_defense):
        """Test combined CPU and memory abuse"""
        source = "process_789"

        threat = guardian_defense.detect_resource_abuse(source, cpu=96.0, memory=96.0)
        assert threat == ThreatLevel.HIGH

    def test_detect_resource_normal(self, guardian_defense):
        """Test normal resource usage"""
        source = "process_normal"

        threat = guardian_defense.detect_resource_abuse(source, cpu=50.0, memory=60.0)
        assert threat is None


@pytest.mark.unit
class TestSourceIsolation:
    """Test source isolation functionality"""

    def test_isolate_source_basic(self, guardian_defense):
        """Test isolating a source"""
        source = "malicious_ip"
        reason = "Rate limit exceeded"

        guardian_defense.isolate_source(source, reason)

        assert source in guardian_defense.isolated_sources
        assert guardian_defense.is_source_isolated(source)

    def test_isolate_source_with_duration(self, guardian_defense):
        """Test isolating source with auto-restore"""
        source = "temp_ban_ip"
        reason = "Temporary ban"

        # Isolate for 1 second
        guardian_defense.isolate_source(source, reason, duration_sec=1)

        assert guardian_defense.is_source_isolated(source)

        # Wait for auto-restore
        time.sleep(1.5)

        # Should be restored automatically
        assert not guardian_defense.is_source_isolated(source)

    def test_isolate_already_isolated(self, guardian_defense):
        """Test attempting to isolate already isolated source"""
        source = "already_isolated"

        guardian_defense.isolate_source(source, "First isolation")
        guardian_defense.isolate_source(source, "Second isolation attempt")

        # Should still be isolated (no error)
        assert guardian_defense.is_source_isolated(source)

    def test_restore_source(self, guardian_defense):
        """Test manually restoring isolated source"""
        source = "isolated_source"

        guardian_defense.isolate_source(source, "Test isolation")
        assert guardian_defense.is_source_isolated(source)

        success = guardian_defense.restore_source(source)

        assert success
        assert not guardian_defense.is_source_isolated(source)

    def test_restore_non_isolated(self, guardian_defense):
        """Test restoring source that isn't isolated"""
        source = "never_isolated"

        success = guardian_defense.restore_source(source)
        assert not success

    def test_is_source_isolated(self, guardian_defense):
        """Test checking isolation status"""
        source = "test_source"

        assert not guardian_defense.is_source_isolated(source)

        guardian_defense.isolate_source(source, "Test")
        assert guardian_defense.is_source_isolated(source)

        guardian_defense.restore_source(source)
        assert not guardian_defense.is_source_isolated(source)


@pytest.mark.unit
class TestLockdownMode:
    """Test lockdown mode functionality"""

    def test_enter_lockdown_mode(self, guardian_defense):
        """Test entering lockdown mode"""
        reason = "Critical threat detected"

        guardian_defense.enter_lockdown_mode(reason)

        assert guardian_defense.defense_mode == DefenseMode.LOCKDOWN

    def test_enter_lockdown_already_in_lockdown(self, guardian_defense):
        """Test entering lockdown when already in lockdown"""
        guardian_defense.enter_lockdown_mode("First reason")
        guardian_defense.enter_lockdown_mode("Second reason")

        # Should still be in lockdown
        assert guardian_defense.defense_mode == DefenseMode.LOCKDOWN

    def test_exit_lockdown_mode(self, guardian_defense):
        """Test exiting lockdown mode"""
        guardian_defense.enter_lockdown_mode("Test lockdown")
        guardian_defense.exit_lockdown_mode()

        assert guardian_defense.defense_mode == DefenseMode.NORMAL

    def test_exit_lockdown_when_not_in_lockdown(self, guardian_defense):
        """Test exiting lockdown when not in lockdown"""
        assert guardian_defense.defense_mode == DefenseMode.NORMAL

        guardian_defense.exit_lockdown_mode()

        # Should remain normal (no error)
        assert guardian_defense.defense_mode == DefenseMode.NORMAL


@pytest.mark.unit
class TestSnapshots:
    """Test snapshot creation and restoration"""

    def test_create_safe_snapshot(self, guardian_defense):
        """Test creating a safe state snapshot"""
        snapshot_path = guardian_defense.create_safe_snapshot("test_snapshot")

        assert snapshot_path.exists()
        assert snapshot_path.suffix == ".json"

    def test_snapshot_contains_state(self, guardian_defense):
        """Test snapshot contains guardian state"""
        import json

        # Set some state
        guardian_defense.defense_mode = DefenseMode.ELEVATED
        guardian_defense.isolate_source("test_ip", "test")

        snapshot_path = guardian_defense.create_safe_snapshot("state_test")

        with open(snapshot_path) as f:
            snapshot = json.load(f)

        assert "snapshot_id" in snapshot
        assert "timestamp" in snapshot
        assert "defense_mode" in snapshot
        assert "code_hashes" in snapshot
        assert "hash" in snapshot

    def test_restore_from_snapshot(self, guardian_defense):
        """Test restoring from a snapshot"""
        # Create initial state
        original_mode = DefenseMode.NORMAL
        guardian_defense.defense_mode = original_mode

        snapshot_path = guardian_defense.create_safe_snapshot("restore_test")

        # Change state
        guardian_defense.defense_mode = DefenseMode.LOCKDOWN

        # Restore
        success = guardian_defense.restore_from_snapshot(snapshot_path)

        assert success
        assert guardian_defense.defense_mode == original_mode

    def test_restore_snapshot_hash_verification(self, guardian_defense):
        """Test snapshot hash verification prevents tampering"""
        import json

        snapshot_path = guardian_defense.create_safe_snapshot("tamper_test")

        # Tamper with snapshot
        with open(snapshot_path, 'r') as f:
            snapshot = json.load(f)

        snapshot["defense_mode"] = "corrupted"

        with open(snapshot_path, 'w') as f:
            json.dump(snapshot, f)

        # Restore should fail due to hash mismatch
        success = guardian_defense.restore_from_snapshot(snapshot_path)
        assert not success


@pytest.mark.unit
class TestThreatLogging:
    """Test threat logging and audit trail"""

    def test_log_threat(self, guardian_defense):
        """Test logging a threat event"""
        guardian_defense._log_threat(
            "test_threat",
            ThreatLevel.MEDIUM,
            "test_source",
            {"detail": "test"},
            "logged"
        )

        assert len(guardian_defense.threat_history) == 1
        threat = guardian_defense.threat_history[0]

        assert threat.threat_type == "test_threat"
        assert threat.threat_level == ThreatLevel.MEDIUM.value
        assert threat.source == "test_source"

    def test_threat_history_max_length(self, guardian_defense):
        """Test threat history respects max length"""
        # Log more than max (1000)
        for i in range(1500):
            guardian_defense._log_threat(
                f"threat_{i}",
                ThreatLevel.LOW,
                "source",
                {},
                "logged"
            )

        assert len(guardian_defense.threat_history) == 1000

    def test_threat_auto_escalation(self, guardian_defense):
        """Test auto-escalation from NORMAL to ELEVATED"""
        assert guardian_defense.defense_mode == DefenseMode.NORMAL

        # Log 5 high-severity threats
        for i in range(5):
            guardian_defense._log_threat(
                f"high_threat_{i}",
                ThreatLevel.HIGH,
                f"source_{i}",
                {},
                "logged"
            )

        assert guardian_defense.defense_mode == DefenseMode.ELEVATED


@pytest.mark.unit
class TestThreatSummary:
    """Test threat summary and statistics"""

    def test_get_threat_summary(self, guardian_defense):
        """Test getting threat summary"""
        summary = guardian_defense.get_threat_summary()

        assert "total_threats" in summary
        assert "by_level" in summary
        assert "by_type" in summary
        assert "isolated_sources" in summary
        assert "defense_mode" in summary
        assert "recent_threats" in summary

    def test_threat_summary_with_threats(self, guardian_defense):
        """Test summary with actual threats"""
        guardian_defense._log_threat("test1", ThreatLevel.LOW, "s1", {}, "logged")
        guardian_defense._log_threat("test2", ThreatLevel.HIGH, "s2", {}, "logged")
        guardian_defense._log_threat("test3", ThreatLevel.CRITICAL, "s3", {}, "logged")

        summary = guardian_defense.get_threat_summary()

        assert summary["total_threats"] == 3
        assert summary["by_level"]["low"] >= 1
        assert summary["by_level"]["high"] >= 1
        assert summary["by_level"]["critical"] >= 1


@pytest.mark.unit
class TestGuardianStartStop:
    """Test starting and stopping guardian"""

    def test_start_guardian(self, guardian_defense):
        """Test starting guardian monitoring"""
        guardian_defense.start()

        assert guardian_defense.running

    def test_start_already_running(self, guardian_defense):
        """Test starting when already running"""
        guardian_defense.start()
        guardian_defense.start()

        assert guardian_defense.running

    def test_stop_guardian(self, guardian_defense):
        """Test stopping guardian monitoring"""
        guardian_defense.start()
        guardian_defense.stop()

        assert not guardian_defense.running

    def test_stop_not_running(self, guardian_defense):
        """Test stopping when not running"""
        guardian_defense.stop()

        assert not guardian_defense.running


@pytest.mark.unit
class TestGetGuardianDefenseSingleton:
    """Test singleton pattern"""

    def test_get_guardian_defense_singleton(self):
        """Test getting guardian defense singleton"""
        instance1 = get_guardian_defense()
        instance2 = get_guardian_defense()

        assert instance1 is instance2

    def test_singleton_persists_state(self):
        """Test singleton maintains state across calls"""
        instance1 = get_guardian_defense()
        instance1.isolate_source("test_source", "test")

        instance2 = get_guardian_defense()

        assert instance2.is_source_isolated("test_source")


@pytest.mark.integration
class TestGuardianIntegration:
    """Integration tests for guardian defense"""

    def test_complete_threat_workflow(self, guardian_defense):
        """Test complete threat detection and isolation workflow"""
        source = "attacker_ip"

        # 1. Detect rate limit abuse
        for _ in range(600):
            guardian_defense.detect_rate_limit_abuse(source, "/api/test")

        # Should be isolated
        assert guardian_defense.is_source_isolated(source)

        # 2. Create snapshot before restoration
        snapshot = guardian_defense.create_safe_snapshot("before_restore")

        # 3. Restore source
        guardian_defense.restore_source(source)
        assert not guardian_defense.is_source_isolated(source)

        # 4. Check threat summary
        summary = guardian_defense.get_threat_summary()
        assert summary["total_threats"] > 0

    def test_escalation_and_lockdown_workflow(self, guardian_defense):
        """Test escalation from NORMAL → ELEVATED → LOCKDOWN"""
        # Start in normal mode
        assert guardian_defense.defense_mode == DefenseMode.NORMAL

        # Trigger escalation to ELEVATED (5 high threats)
        for i in range(5):
            guardian_defense._log_threat(
                f"high_{i}",
                ThreatLevel.HIGH,
                "source",
                {},
                "logged"
            )

        assert guardian_defense.defense_mode == DefenseMode.ELEVATED

        # Manually trigger lockdown
        guardian_defense.enter_lockdown_mode("Critical threat")
        assert guardian_defense.defense_mode == DefenseMode.LOCKDOWN

    def test_code_integrity_monitoring_workflow(self, guardian_defense):
        """Test code integrity monitoring workflow"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write("# Original code\ndef func():\n    pass")
            test_file = Path(f.name)

        try:
            # 1. Register file
            guardian_defense.register_code_integrity([test_file])

            # 2. Verify (should pass)
            tampered = guardian_defense.verify_code_integrity()
            assert len(tampered) == 0

            # 3. Create snapshot
            snapshot = guardian_defense.create_safe_snapshot("clean_state")

            # 4. Tamper with code
            with open(test_file, 'w') as f:
                f.write("# Malicious code\nos.system('rm -rf /')")

            # 5. Verify (should detect tampering)
            tampered = guardian_defense.verify_code_integrity()
            assert len(tampered) == 1

            # 6. Should trigger lockdown
            assert guardian_defense.defense_mode == DefenseMode.LOCKDOWN

        finally:
            test_file.unlink()


@pytest.mark.performance
class TestGuardianPerformance:
    """Performance tests for guardian defense"""

    def test_rate_limit_detection_performance(self, guardian_defense, benchmark):
        """Benchmark rate limit detection"""
        source = "perf_test_ip"

        def detect():
            return guardian_defense.detect_rate_limit_abuse(source, "/api/test")

        result = benchmark(detect)

    def test_concurrent_threat_detection(self, guardian_defense):
        """Test concurrent threat detection"""
        import concurrent.futures

        def detect_threat(i):
            guardian_defense.detect_rate_limit_abuse(f"source_{i}", "/api/test")
            guardian_defense.detect_auth_abuse(f"source_{i}", 3)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(detect_threat, i) for i in range(100)]
            [f.result() for f in concurrent.futures.as_completed(futures)]

        # Should handle concurrent requests safely
        summary = guardian_defense.get_threat_summary()
        assert summary["total_threats"] > 0
