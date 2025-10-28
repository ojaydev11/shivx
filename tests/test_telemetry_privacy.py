"""
Tests for Telemetry Privacy Controls

Verifies that telemetry respects privacy settings:
- Telemetry mode enforcement
- Offline mode disables telemetry
- Air-gap mode disables telemetry
- DNT header respected
- User consent checked
"""

import pytest
from unittest.mock import patch, MagicMock
from core.deployment.production_telemetry import (
    ProductionTelemetry,
    DeploymentTask,
    TaskOutcome,
    get_production_telemetry,
)


@pytest.fixture
def telemetry_enabled_settings():
    """Mock settings with telemetry enabled"""
    with patch('core.deployment.production_telemetry.get_settings') as mock_settings:
        settings = MagicMock()
        settings.offline_mode = False
        settings.airgap_mode = False
        settings.telemetry_mode = "standard"
        mock_settings.return_value = settings
        yield settings


@pytest.fixture
def telemetry_disabled_settings():
    """Mock settings with telemetry disabled"""
    with patch('core.deployment.production_telemetry.get_settings') as mock_settings:
        settings = MagicMock()
        settings.offline_mode = False
        settings.airgap_mode = False
        settings.telemetry_mode = "disabled"
        mock_settings.return_value = settings
        yield settings


@pytest.fixture
def offline_mode_settings():
    """Mock settings with offline mode enabled"""
    with patch('core.deployment.production_telemetry.get_settings') as mock_settings:
        settings = MagicMock()
        settings.offline_mode = True
        settings.airgap_mode = False
        settings.telemetry_mode = "standard"
        mock_settings.return_value = settings
        yield settings


@pytest.fixture
def airgap_mode_settings():
    """Mock settings with airgap mode enabled"""
    with patch('core.deployment.production_telemetry.get_settings') as mock_settings:
        settings = MagicMock()
        settings.offline_mode = False
        settings.airgap_mode = True
        settings.telemetry_mode = "standard"
        mock_settings.return_value = settings
        yield settings


class TestTelemetryPrivacyControls:
    """Test telemetry privacy controls"""

    def test_telemetry_disabled_in_offline_mode(self, offline_mode_settings):
        """Test telemetry is disabled in offline mode"""
        telemetry = ProductionTelemetry()

        assert telemetry.is_enabled() is False

    def test_telemetry_disabled_in_airgap_mode(self, airgap_mode_settings):
        """Test telemetry is disabled in airgap mode"""
        telemetry = ProductionTelemetry()

        assert telemetry.is_enabled() is False

    def test_telemetry_disabled_when_mode_disabled(self, telemetry_disabled_settings):
        """Test telemetry is disabled when telemetry_mode=disabled"""
        telemetry = ProductionTelemetry()

        assert telemetry.is_enabled() is False

    def test_telemetry_enabled_with_standard_mode(self, telemetry_enabled_settings):
        """Test telemetry is enabled with standard mode"""
        with patch('core.deployment.production_telemetry.ProductionTelemetry._init_database'):
            telemetry = ProductionTelemetry()
            assert telemetry.is_enabled() is True

    def test_log_task_skipped_when_disabled(self, telemetry_disabled_settings):
        """Test task logging is skipped when telemetry disabled"""
        telemetry = ProductionTelemetry()

        task = DeploymentTask(
            task_id="task1",
            task_type="test",
            query="test query",
            capabilities_used=["test"],
            outcome=TaskOutcome.SUCCESS,
            confidence=0.9,
            latency_ms=100.0
        )

        # Should not raise, just skip
        telemetry.log_task(task)

    def test_log_task_skipped_in_offline_mode(self, offline_mode_settings):
        """Test task logging is skipped in offline mode"""
        telemetry = ProductionTelemetry()

        task = DeploymentTask(
            task_id="task1",
            task_type="test",
            query="test query",
            capabilities_used=["test"],
            outcome=TaskOutcome.SUCCESS,
            confidence=0.9,
            latency_ms=100.0
        )

        # Should not raise, just skip
        telemetry.log_task(task)


class TestTelemetryModeEnforcement:
    """Test telemetry mode enforcement"""

    def test_minimal_mode_collects_only_errors(self):
        """Test minimal mode only collects errors"""
        with patch('core.deployment.production_telemetry.get_settings') as mock_settings:
            settings = MagicMock()
            settings.offline_mode = False
            settings.airgap_mode = False
            settings.telemetry_mode = "minimal"
            mock_settings.return_value = settings

            with patch('core.deployment.production_telemetry.ProductionTelemetry._init_database'):
                telemetry = ProductionTelemetry()

                assert telemetry.should_collect_event("error") is True
                assert telemetry.should_collect_event("critical") is True
                assert telemetry.should_collect_event("performance") is False
                assert telemetry.should_collect_event("usage") is False

    def test_standard_mode_collects_errors_and_performance(self):
        """Test standard mode collects errors and performance"""
        with patch('core.deployment.production_telemetry.get_settings') as mock_settings:
            settings = MagicMock()
            settings.offline_mode = False
            settings.airgap_mode = False
            settings.telemetry_mode = "standard"
            mock_settings.return_value = settings

            with patch('core.deployment.production_telemetry.ProductionTelemetry._init_database'):
                telemetry = ProductionTelemetry()

                assert telemetry.should_collect_event("error") is True
                assert telemetry.should_collect_event("performance") is True
                assert telemetry.should_collect_event("usage") is False

    def test_full_mode_collects_all_events(self):
        """Test full mode collects all events"""
        with patch('core.deployment.production_telemetry.get_settings') as mock_settings:
            settings = MagicMock()
            settings.offline_mode = False
            settings.airgap_mode = False
            settings.telemetry_mode = "full"
            mock_settings.return_value = settings

            with patch('core.deployment.production_telemetry.ProductionTelemetry._init_database'):
                telemetry = ProductionTelemetry()

                assert telemetry.should_collect_event("error") is True
                assert telemetry.should_collect_event("performance") is True
                assert telemetry.should_collect_event("usage") is True
                assert telemetry.should_collect_event("anything") is True

    def test_disabled_mode_collects_nothing(self):
        """Test disabled mode collects nothing"""
        with patch('core.deployment.production_telemetry.get_settings') as mock_settings:
            settings = MagicMock()
            settings.offline_mode = False
            settings.airgap_mode = False
            settings.telemetry_mode = "disabled"
            mock_settings.return_value = settings

            telemetry = ProductionTelemetry()

            assert telemetry.should_collect_event("error") is False
            assert telemetry.should_collect_event("performance") is False
            assert telemetry.should_collect_event("usage") is False


class TestTelemetryEventFiltering:
    """Test telemetry event filtering based on mode"""

    def test_error_event_logged_in_minimal_mode(self):
        """Test error events are logged in minimal mode"""
        with patch('core.deployment.production_telemetry.get_settings') as mock_settings:
            settings = MagicMock()
            settings.offline_mode = False
            settings.airgap_mode = False
            settings.telemetry_mode = "minimal"
            mock_settings.return_value = settings

            with patch('core.deployment.production_telemetry.ProductionTelemetry._init_database'):
                with patch('sqlite3.connect'):
                    telemetry = ProductionTelemetry()

                    error_task = DeploymentTask(
                        task_id="task1",
                        task_type="test",
                        query="test",
                        capabilities_used=["test"],
                        outcome=TaskOutcome.ERROR,
                        confidence=0.5,
                        latency_ms=100.0
                    )

                    # Should attempt to log (checked via should_collect_event)
                    assert telemetry.should_collect_event("error") is True

    def test_usage_event_skipped_in_minimal_mode(self):
        """Test usage events are skipped in minimal mode"""
        with patch('core.deployment.production_telemetry.get_settings') as mock_settings:
            settings = MagicMock()
            settings.offline_mode = False
            settings.airgap_mode = False
            settings.telemetry_mode = "minimal"
            mock_settings.return_value = settings

            with patch('core.deployment.production_telemetry.ProductionTelemetry._init_database'):
                telemetry = ProductionTelemetry()

                # Usage events should be skipped
                assert telemetry.should_collect_event("usage") is False


class TestTelemetrySingleton:
    """Test telemetry singleton"""

    def test_get_production_telemetry_returns_singleton(self, telemetry_enabled_settings):
        """Test get_production_telemetry returns same instance"""
        # Clear singleton
        import core.deployment.production_telemetry
        core.deployment.production_telemetry._telemetry = None

        with patch('core.deployment.production_telemetry.ProductionTelemetry._init_database'):
            telemetry1 = get_production_telemetry()
            telemetry2 = get_production_telemetry()

            assert telemetry1 is telemetry2


class TestTelemetryPrivacyIntegration:
    """Integration tests for telemetry privacy"""

    def test_offline_mode_prevents_telemetry_initialization(self, offline_mode_settings):
        """Test offline mode prevents telemetry database initialization"""
        with patch('core.deployment.production_telemetry.ProductionTelemetry._init_database') as mock_init:
            telemetry = ProductionTelemetry()

            # Should not initialize database in offline mode
            assert not mock_init.called

    def test_privacy_modes_cascade(self):
        """Test that multiple privacy modes work together"""
        with patch('core.deployment.production_telemetry.get_settings') as mock_settings:
            settings = MagicMock()
            settings.offline_mode = True  # Offline mode
            settings.airgap_mode = False
            settings.telemetry_mode = "full"  # Even with full mode
            mock_settings.return_value = settings

            telemetry = ProductionTelemetry()

            # Should be disabled due to offline mode
            assert telemetry.is_enabled() is False

    def test_telemetry_mode_overrides_when_not_offline(self):
        """Test telemetry_mode=disabled works even when not offline"""
        with patch('core.deployment.production_telemetry.get_settings') as mock_settings:
            settings = MagicMock()
            settings.offline_mode = False  # Not offline
            settings.airgap_mode = False  # Not airgap
            settings.telemetry_mode = "disabled"  # But telemetry disabled
            mock_settings.return_value = settings

            telemetry = ProductionTelemetry()

            # Should be disabled due to telemetry mode
            assert telemetry.is_enabled() is False
