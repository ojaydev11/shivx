"""
Tests for Air-Gap Mode Implementation

Verifies maximum network isolation:
- Network interface detection
- External interface blocking
- Startup verification
- Violation detection
"""

import pytest
from unittest.mock import patch, MagicMock
from core.privacy.airgap import (
    NetworkInterface,
    NetworkMonitor,
    AirGapMode,
    AirGapViolation,
    get_airgap_mode,
    check_network_isolation,
)


@pytest.fixture
def airgap_settings():
    """Mock settings with airgap mode enabled"""
    with patch('core.privacy.airgap.get_settings') as mock_settings:
        mock_settings.return_value.airgap_mode = True
        yield mock_settings


@pytest.fixture
def normal_settings():
    """Mock settings with airgap mode disabled"""
    with patch('core.privacy.airgap.get_settings') as mock_settings:
        mock_settings.return_value.airgap_mode = False
        yield mock_settings


class TestNetworkInterface:
    """Test NetworkInterface class"""

    def test_is_loopback_lo(self):
        """Test loopback interface detection"""
        iface = NetworkInterface("lo", "up", ["127.0.0.1"])
        assert iface.is_loopback() is True

    def test_is_loopback_loop(self):
        """Test loop interface detection"""
        iface = NetworkInterface("loop0", "up", ["127.0.0.1"])
        assert iface.is_loopback() is True

    def test_is_not_loopback(self):
        """Test non-loopback interface"""
        iface = NetworkInterface("eth0", "up", ["192.168.1.100"])
        assert iface.is_loopback() is False

    def test_is_active_up(self):
        """Test active interface detection"""
        iface = NetworkInterface("eth0", "up", ["192.168.1.100"])
        assert iface.is_active() is True

    def test_is_active_running(self):
        """Test active interface detection (running status)"""
        iface = NetworkInterface("eth0", "running", ["192.168.1.100"])
        assert iface.is_active() is True

    def test_is_not_active_down(self):
        """Test inactive interface detection"""
        iface = NetworkInterface("eth0", "down", [])
        assert iface.is_active() is False


class TestNetworkMonitor:
    """Test NetworkMonitor class"""

    @patch('core.privacy.airgap.netifaces')
    def test_get_network_interfaces_with_netifaces(self, mock_netifaces):
        """Test getting interfaces with netifaces library"""
        mock_netifaces.interfaces.return_value = ["lo", "eth0"]
        mock_netifaces.AF_INET = 2
        mock_netifaces.AF_INET6 = 10

        mock_netifaces.ifaddresses.side_effect = [
            {2: [{"addr": "127.0.0.1"}]},  # lo
            {2: [{"addr": "192.168.1.100"}]},  # eth0
        ]

        monitor = NetworkMonitor()
        interfaces = monitor.get_network_interfaces()

        assert len(interfaces) == 2
        assert any(i.name == "lo" for i in interfaces)
        assert any(i.name == "eth0" for i in interfaces)

    def test_get_active_external_interfaces(self, airgap_settings):
        """Test getting active external interfaces"""
        monitor = NetworkMonitor()

        # Mock interfaces
        with patch.object(monitor, 'get_network_interfaces') as mock_get:
            mock_get.return_value = [
                NetworkInterface("lo", "up", ["127.0.0.1"]),
                NetworkInterface("eth0", "up", ["192.168.1.100"]),
                NetworkInterface("wlan0", "down", []),
            ]

            external = monitor.get_active_external_interfaces()

            # Should only include eth0 (active and non-loopback)
            assert len(external) == 1
            assert external[0].name == "eth0"

    def test_log_connection_attempt(self, airgap_settings):
        """Test logging connection attempts"""
        monitor = NetworkMonitor()

        monitor.log_connection_attempt("example.com", 443, "tcp")

        assert len(monitor.connection_attempts) == 1
        assert monitor.connection_attempts[0]["target"] == "example.com"
        assert monitor.connection_attempts[0]["port"] == 443

    def test_get_connection_attempts(self, airgap_settings):
        """Test retrieving connection attempts"""
        monitor = NetworkMonitor()

        monitor.log_connection_attempt("example.com", 443)
        monitor.log_connection_attempt("google.com", 80)

        attempts = monitor.get_connection_attempts()

        assert len(attempts) == 2


class TestAirGapMode:
    """Test AirGapMode class"""

    def test_is_enabled_when_airgap(self, airgap_settings):
        """Test is_enabled returns True when airgap mode on"""
        airgap = AirGapMode()
        assert airgap.is_enabled() is True

    def test_is_enabled_when_normal(self, normal_settings):
        """Test is_enabled returns False when airgap mode off"""
        airgap = AirGapMode()
        assert airgap.is_enabled() is False

    def test_verify_isolation_when_disabled(self, normal_settings):
        """Test verification when airgap mode disabled"""
        airgap = AirGapMode()
        result = airgap.verify_isolation()

        assert result["airgap_mode"] is False
        assert result["verified"] is False

    def test_verify_isolation_with_external_interfaces(self, airgap_settings):
        """Test verification fails with external interfaces"""
        airgap = AirGapMode()

        # Mock external interfaces
        with patch.object(airgap.monitor, 'get_active_external_interfaces') as mock_get:
            mock_get.return_value = [
                NetworkInterface("eth0", "up", ["192.168.1.100"])
            ]

            with pytest.raises(AirGapViolation, match="active external"):
                airgap.verify_isolation(fail_on_violation=True)

    def test_verify_isolation_without_failure(self, airgap_settings):
        """Test verification without raising exception"""
        airgap = AirGapMode()

        # Mock external interfaces
        with patch.object(airgap.monitor, 'get_active_external_interfaces') as mock_get:
            mock_get.return_value = [
                NetworkInterface("eth0", "up", ["192.168.1.100"])
            ]

            result = airgap.verify_isolation(fail_on_violation=False)

            assert result["airgap_mode"] is True
            assert result["verified"] is False
            assert result["violations"] == 1

    def test_verify_isolation_success(self, airgap_settings):
        """Test successful verification with no external interfaces"""
        airgap = AirGapMode()

        # Mock no external interfaces
        with patch.object(airgap.monitor, 'get_active_external_interfaces') as mock_get:
            mock_get.return_value = []

            result = airgap.verify_isolation()

            assert result["airgap_mode"] is True
            assert result["verified"] is True
            assert result["violations"] == 0

    def test_verify_on_startup_success(self, airgap_settings):
        """Test startup verification succeeds"""
        airgap = AirGapMode()

        # Mock no external interfaces
        with patch.object(airgap.monitor, 'get_active_external_interfaces') as mock_get:
            mock_get.return_value = []

            # Should not raise
            airgap.verify_on_startup()

    def test_verify_on_startup_failure(self, airgap_settings):
        """Test startup verification fails and raises"""
        airgap = AirGapMode()

        # Mock external interfaces
        with patch.object(airgap.monitor, 'get_active_external_interfaces') as mock_get:
            mock_get.return_value = [
                NetworkInterface("eth0", "up", ["192.168.1.100"])
            ]

            with pytest.raises(AirGapViolation):
                airgap.verify_on_startup()

    def test_get_status_when_enabled(self, airgap_settings):
        """Test status when airgap mode enabled"""
        airgap = AirGapMode()

        # Mock no external interfaces
        with patch.object(airgap.monitor, 'get_active_external_interfaces') as mock_get:
            mock_get.return_value = []

            status = airgap.get_status()

            assert status["airgap_mode"] is True
            assert status["status"] == "isolated"
            assert status["verified"] is True

    def test_get_status_when_disabled(self, normal_settings):
        """Test status when airgap mode disabled"""
        airgap = AirGapMode()
        status = airgap.get_status()

        assert status["airgap_mode"] is False
        assert status["status"] == "disabled"


class TestGlobalFunctions:
    """Test module-level functions"""

    def test_get_airgap_mode_singleton(self):
        """Test that get_airgap_mode returns singleton"""
        airgap1 = get_airgap_mode()
        airgap2 = get_airgap_mode()

        assert airgap1 is airgap2

    def test_check_network_isolation_when_enabled(self, airgap_settings):
        """Test check_network_isolation when enabled"""
        # Clear singleton
        import core.privacy.airgap
        core.privacy.airgap._airgap_mode = None

        # Mock no external interfaces
        with patch('core.privacy.airgap.NetworkMonitor.get_active_external_interfaces') as mock_get:
            mock_get.return_value = []

            result = check_network_isolation()

            assert result["airgap_mode"] is True
            assert result["verified"] is True

    def test_check_network_isolation_when_disabled(self, normal_settings):
        """Test check_network_isolation when disabled"""
        # Clear singleton
        import core.privacy.airgap
        core.privacy.airgap._airgap_mode = None

        result = check_network_isolation()

        assert result["airgap_mode"] is False
