"""
Tests for Offline Mode Implementation

Verifies that offline mode:
- Blocks external network requests
- Allows localhost requests
- Tracks blocked requests
- Reports status correctly
- Degrades features appropriately
"""

import pytest
from unittest.mock import patch, MagicMock
from core.privacy.offline import (
    NetworkBlocker,
    NetworkBlockedError,
    OfflineMode,
    get_offline_mode,
    check_url_allowed,
    is_offline_mode,
)


@pytest.fixture
def offline_settings():
    """Mock settings with offline mode enabled"""
    with patch('core.privacy.offline.get_settings') as mock_settings:
        mock_settings.return_value.offline_mode = True
        yield mock_settings


@pytest.fixture
def online_settings():
    """Mock settings with offline mode disabled"""
    with patch('core.privacy.offline.get_settings') as mock_settings:
        mock_settings.return_value.offline_mode = False
        yield mock_settings


class TestNetworkBlocker:
    """Test NetworkBlocker class"""

    def test_is_allowed_host_localhost(self):
        """Test that localhost is allowed"""
        blocker = NetworkBlocker()
        assert blocker.is_allowed_host("localhost") is True
        assert blocker.is_allowed_host("127.0.0.1") is True
        assert blocker.is_allowed_host("::1") is True
        assert blocker.is_allowed_host("0.0.0.0") is True

    def test_is_allowed_host_with_port(self):
        """Test that localhost with port is allowed"""
        blocker = NetworkBlocker()
        assert blocker.is_allowed_host("localhost:8000") is True
        assert blocker.is_allowed_host("127.0.0.1:3000") is True

    def test_is_allowed_host_loopback_range(self):
        """Test that loopback IP range is allowed"""
        blocker = NetworkBlocker()
        assert blocker.is_allowed_host("127.0.0.1") is True
        assert blocker.is_allowed_host("127.0.0.255") is True
        assert blocker.is_allowed_host("127.100.200.50") is True

    def test_is_allowed_host_external(self):
        """Test that external hosts are not allowed"""
        blocker = NetworkBlocker()
        assert blocker.is_allowed_host("google.com") is False
        assert blocker.is_allowed_host("192.168.1.1") is False
        assert blocker.is_allowed_host("8.8.8.8") is False
        assert blocker.is_allowed_host("api.example.com") is False

    def test_check_url_localhost(self, offline_settings):
        """Test that localhost URLs pass when offline"""
        blocker = NetworkBlocker()
        assert blocker.check_url("http://localhost:8000/api") is True
        assert blocker.check_url("http://127.0.0.1:3000/test") is True

    def test_check_url_external_blocked(self, offline_settings):
        """Test that external URLs are blocked when offline"""
        blocker = NetworkBlocker()

        with pytest.raises(NetworkBlockedError) as exc_info:
            blocker.check_url("https://api.example.com/data")

        assert "Offline mode" in str(exc_info.value)
        assert "example.com" in str(exc_info.value)

    def test_check_url_when_online(self, online_settings):
        """Test that all URLs pass when online"""
        blocker = NetworkBlocker()
        assert blocker.check_url("https://api.example.com/data") is True
        assert blocker.check_url("https://google.com") is True

    def test_blocked_count_increments(self, offline_settings):
        """Test that blocked count increments"""
        blocker = NetworkBlocker()

        assert blocker.blocked_count == 0

        try:
            blocker.check_url("https://example.com")
        except NetworkBlockedError:
            pass

        assert blocker.blocked_count == 1

        try:
            blocker.check_url("https://google.com")
        except NetworkBlockedError:
            pass

        assert blocker.blocked_count == 2

    def test_invalid_url_blocked(self, offline_settings):
        """Test that invalid URLs are blocked on error"""
        blocker = NetworkBlocker()

        with pytest.raises(NetworkBlockedError) as exc_info:
            blocker.check_url("not_a_valid_url")

        assert "parse error" in str(exc_info.value).lower()

    def test_get_stats(self, offline_settings):
        """Test blocker statistics"""
        blocker = NetworkBlocker()

        try:
            blocker.check_url("https://example.com")
        except NetworkBlockedError:
            pass

        stats = blocker.get_stats()

        assert stats["offline_mode"] is True
        assert stats["blocked_count"] == 1
        assert "localhost" in stats["allowed_hosts"]


class TestOfflineMode:
    """Test OfflineMode class"""

    def test_is_enabled_when_offline(self, offline_settings):
        """Test is_enabled returns True when offline mode on"""
        offline = OfflineMode()
        assert offline.is_enabled() is True

    def test_is_enabled_when_online(self, online_settings):
        """Test is_enabled returns False when offline mode off"""
        offline = OfflineMode()
        assert offline.is_enabled() is False

    def test_check_network_isolation_when_disabled(self, online_settings):
        """Test network isolation check when offline mode disabled"""
        offline = OfflineMode()
        result = offline.check_network_isolation()

        assert result["offline_mode"] is False
        assert result["isolated"] is False
        assert "not enabled" in result["message"].lower()

    @patch('socket.socket')
    def test_check_network_isolation_with_connectivity(self, mock_socket, offline_settings):
        """Test network isolation check when network is accessible"""
        # Mock successful connection
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 0  # Connection successful
        mock_socket.return_value = mock_sock

        offline = OfflineMode()
        result = offline.check_network_isolation()

        assert result["offline_mode"] is True
        assert len(result["warnings"]) > 0  # Should have warnings about connectivity

    @patch('socket.socket')
    @patch('socket.gethostbyname')
    def test_check_network_isolation_fully_isolated(self, mock_gethostbyname, mock_socket, offline_settings):
        """Test network isolation check when fully isolated"""
        # Mock failed connection (timeout)
        mock_sock = MagicMock()
        mock_sock.connect_ex.side_effect = OSError("Network unreachable")
        mock_socket.return_value = mock_sock

        # Mock DNS failure
        mock_gethostbyname.side_effect = socket.gaierror("DNS failed")

        offline = OfflineMode()
        result = offline.check_network_isolation()

        assert result["offline_mode"] is True
        # Should be isolated or have minimal warnings

    def test_get_degraded_features(self, offline_settings):
        """Test degraded features list when offline"""
        offline = OfflineMode()
        features = offline.get_degraded_features()

        assert len(features) > 0
        assert any("API" in f for f in features)
        assert any("telemetry" in f.lower() for f in features)

    def test_get_degraded_features_when_online(self, online_settings):
        """Test no degraded features when online"""
        offline = OfflineMode()
        features = offline.get_degraded_features()

        assert len(features) == 0

    def test_get_status_when_offline(self, offline_settings):
        """Test status when offline mode enabled"""
        offline = OfflineMode()
        status = offline.get_status()

        assert status["offline_mode"] is True
        assert status["status"] in ["isolated", "partial"]
        assert "blocked_requests" in status
        assert "degraded_features" in status

    def test_get_status_when_online(self, online_settings):
        """Test status when offline mode disabled"""
        offline = OfflineMode()
        status = offline.get_status()

        assert status["offline_mode"] is False
        assert status["status"] == "online"
        assert "full network access" in status["message"].lower()


class TestGlobalFunctions:
    """Test module-level functions"""

    def test_get_offline_mode_singleton(self):
        """Test that get_offline_mode returns singleton"""
        offline1 = get_offline_mode()
        offline2 = get_offline_mode()

        assert offline1 is offline2

    def test_is_offline_mode_when_enabled(self, offline_settings):
        """Test is_offline_mode returns True when enabled"""
        # Clear singleton first
        import core.privacy.offline
        core.privacy.offline._offline_mode = None

        assert is_offline_mode() is True

    def test_is_offline_mode_when_disabled(self, online_settings):
        """Test is_offline_mode returns False when disabled"""
        # Clear singleton first
        import core.privacy.offline
        core.privacy.offline._offline_mode = None

        assert is_offline_mode() is False

    def test_check_url_allowed_when_offline(self, offline_settings):
        """Test check_url_allowed blocks external URLs"""
        # Clear singleton
        import core.privacy.offline
        core.privacy.offline._offline_mode = None

        # Should not raise for localhost
        check_url_allowed("http://localhost:8000")

        # Should raise for external
        with pytest.raises(NetworkBlockedError):
            check_url_allowed("https://api.example.com")

    def test_check_url_allowed_when_online(self, online_settings):
        """Test check_url_allowed allows all URLs when online"""
        # Clear singleton
        import core.privacy.offline
        core.privacy.offline._offline_mode = None

        # Should not raise for any URL
        check_url_allowed("https://api.example.com")
        check_url_allowed("http://localhost:8000")


class TestOfflineModeDecorator:
    """Test offline mode decorator"""

    def test_decorator_blocks_when_offline(self, offline_settings):
        """Test decorator blocks function when offline"""
        from core.privacy.offline import offline_mode_decorator

        @offline_mode_decorator
        async def fetch_data():
            return "data"

        # Clear singleton
        import core.privacy.offline
        core.privacy.offline._offline_mode = None

        import asyncio
        with pytest.raises(NetworkBlockedError):
            asyncio.run(fetch_data())

    def test_decorator_allows_when_online(self, online_settings):
        """Test decorator allows function when online"""
        from core.privacy.offline import offline_mode_decorator

        @offline_mode_decorator
        async def fetch_data():
            return "data"

        # Clear singleton
        import core.privacy.offline
        core.privacy.offline._offline_mode = None

        import asyncio
        result = asyncio.run(fetch_data())
        assert result == "data"


class TestOfflineModeIntegration:
    """Integration tests for offline mode"""

    def test_multiple_blocked_requests(self, offline_settings):
        """Test multiple blocked requests are tracked"""
        offline = OfflineMode()

        urls_to_test = [
            "https://api.example.com",
            "https://google.com",
            "https://github.com",
            "https://amazon.com",
        ]

        for url in urls_to_test:
            try:
                offline.blocker.check_url(url)
            except NetworkBlockedError:
                pass

        assert offline.blocker.blocked_count == len(urls_to_test)

    def test_localhost_variants_all_allowed(self, offline_settings):
        """Test all localhost variants are allowed"""
        offline = OfflineMode()

        localhost_urls = [
            "http://localhost",
            "http://localhost:8000",
            "http://127.0.0.1",
            "http://127.0.0.1:3000",
            "http://[::1]",
            "http://0.0.0.0:8080",
        ]

        for url in localhost_urls:
            # Should not raise
            try:
                offline.blocker.check_url(url)
            except NetworkBlockedError:
                pytest.fail(f"Localhost URL should be allowed: {url}")


# Import socket for DNS error test
import socket
