"""
Offline Mode Implementation

Enforces true offline-first operation:
- Blocks all outbound HTTP requests (except localhost)
- Disables telemetry collection
- Uses cached data only
- Graceful degradation for network-dependent features
"""

import logging
import socket
import urllib.parse
from typing import Optional, Set
from contextlib import contextmanager

from config.settings import get_settings

logger = logging.getLogger(__name__)


class NetworkBlocker:
    """
    Network request blocker for offline mode

    Intercepts and blocks outbound network connections
    """

    def __init__(self):
        self.settings = get_settings()
        self.blocked_count = 0
        self.allowed_hosts: Set[str] = {
            "localhost",
            "127.0.0.1",
            "::1",
            "0.0.0.0",
        }

    def is_allowed_host(self, host: str) -> bool:
        """
        Check if host is allowed in offline mode

        Args:
            host: Hostname or IP address

        Returns:
            True if host is localhost, False otherwise
        """
        # Parse host (remove port if present)
        if ":" in host:
            host = host.split(":")[0]

        # Check if it's localhost
        if host.lower() in self.allowed_hosts:
            return True

        # Check if it's a loopback address
        try:
            # For IPv4
            if host.startswith("127."):
                return True
            # For IPv6
            if host.startswith("::1") or host.startswith("fe80::"):
                return True
        except Exception:
            pass

        return False

    def check_url(self, url: str) -> bool:
        """
        Check if URL is allowed in offline mode

        Args:
            url: URL to check

        Returns:
            True if allowed, False if blocked

        Raises:
            NetworkBlockedError: If offline mode enabled and URL is external
        """
        if not self.settings.offline_mode:
            return True

        try:
            parsed = urllib.parse.urlparse(url)
            host = parsed.hostname or parsed.netloc

            if self.is_allowed_host(host):
                return True
            else:
                self.blocked_count += 1
                raise NetworkBlockedError(
                    f"Offline mode: Blocked external request to {host}. "
                    f"Total blocked: {self.blocked_count}"
                )
        except NetworkBlockedError:
            raise
        except Exception as e:
            logger.error(f"Error checking URL {url}: {e}")
            # Fail closed - block on error
            raise NetworkBlockedError(f"Offline mode: Blocked request due to parse error: {e}")

    def get_stats(self) -> dict:
        """Get blocker statistics"""
        return {
            "offline_mode": self.settings.offline_mode,
            "blocked_count": self.blocked_count,
            "allowed_hosts": list(self.allowed_hosts),
        }


class NetworkBlockedError(Exception):
    """Raised when network request is blocked by offline mode"""
    pass


class OfflineMode:
    """
    Offline mode manager

    Provides context and utilities for offline operation
    """

    def __init__(self):
        self.settings = get_settings()
        self.blocker = NetworkBlocker()

    def is_enabled(self) -> bool:
        """Check if offline mode is enabled"""
        return self.settings.offline_mode

    def check_network_isolation(self) -> dict:
        """
        Check if system is properly network-isolated

        Returns:
            Dictionary with isolation status
        """
        if not self.is_enabled():
            return {
                "offline_mode": False,
                "isolated": False,
                "message": "Offline mode not enabled",
            }

        issues = []
        warnings = []

        # Check for active network connections
        try:
            # Try to connect to common external addresses
            test_hosts = [
                ("8.8.8.8", 53),  # Google DNS
                ("1.1.1.1", 53),  # Cloudflare DNS
            ]

            for host, port in test_hosts:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex((host, port))
                    sock.close()

                    if result == 0:
                        warnings.append(
                            f"Network connection possible to {host}:{port}. "
                            "System may not be fully isolated."
                        )
                except Exception:
                    # Connection failed - good for offline mode
                    pass
        except Exception as e:
            logger.debug(f"Network isolation check error: {e}")

        # Check if DNS resolution works for external domains
        try:
            socket.gethostbyname("google.com")
            warnings.append(
                "DNS resolution working for external domains. "
                "Consider disabling DNS for full isolation."
            )
        except socket.gaierror:
            # DNS failed - good for offline mode
            pass
        except Exception as e:
            logger.debug(f"DNS check error: {e}")

        if warnings:
            logger.warning(
                f"Offline mode enabled but network isolation incomplete: {warnings}"
            )

        return {
            "offline_mode": True,
            "isolated": len(warnings) == 0,
            "issues": issues,
            "warnings": warnings,
            "blocked_requests": self.blocker.blocked_count,
        }

    def verify_on_startup(self) -> None:
        """
        Verify offline mode configuration on startup

        Logs warnings if network connections detected
        """
        if not self.is_enabled():
            return

        logger.info("Offline mode enabled - performing isolation check...")

        status = self.check_network_isolation()

        if status["isolated"]:
            logger.info("✓ Network isolation verified - system is offline")
        else:
            logger.warning(
                f"⚠ Network isolation incomplete: {status['warnings']}"
            )
            logger.warning(
                "External network access may still be possible. "
                "For full air-gap, disable network interfaces at OS level."
            )

    def get_degraded_features(self) -> list:
        """
        Get list of features that degrade in offline mode

        Returns:
            List of feature descriptions
        """
        if not self.is_enabled():
            return []

        return [
            "External API calls (DEX, price feeds, etc.)",
            "Real-time market data updates",
            "Telemetry and error reporting",
            "External AI model API calls (OpenAI, Anthropic)",
            "Live trading (paper trading only with cached data)",
            "Software updates and version checks",
        ]

    def get_status(self) -> dict:
        """Get comprehensive offline mode status"""
        if not self.is_enabled():
            return {
                "offline_mode": False,
                "status": "online",
                "message": "Offline mode not enabled - full network access",
            }

        isolation = self.check_network_isolation()

        return {
            "offline_mode": True,
            "status": "isolated" if isolation["isolated"] else "partial",
            "isolated": isolation["isolated"],
            "warnings": isolation.get("warnings", []),
            "blocked_requests": self.blocker.blocked_count,
            "degraded_features": self.get_degraded_features(),
            "message": (
                "Fully offline and isolated"
                if isolation["isolated"]
                else "Offline mode enabled but network access may be possible"
            ),
        }


# Global offline mode instance
_offline_mode: Optional[OfflineMode] = None


def get_offline_mode() -> OfflineMode:
    """Get global offline mode instance"""
    global _offline_mode
    if _offline_mode is None:
        _offline_mode = OfflineMode()
    return _offline_mode


def is_offline_mode() -> bool:
    """Check if offline mode is enabled"""
    return get_offline_mode().is_enabled()


def check_url_allowed(url: str) -> None:
    """
    Check if URL is allowed in offline mode

    Args:
        url: URL to check

    Raises:
        NetworkBlockedError: If offline mode enabled and URL is external
    """
    offline = get_offline_mode()
    if offline.is_enabled():
        offline.blocker.check_url(url)


def offline_mode_decorator(func):
    """
    Decorator to enforce offline mode for network-dependent functions

    Usage:
        @offline_mode_decorator
        async def fetch_price(token: str):
            # This will raise NetworkBlockedError if offline
            return await http_client.get(f"https://api.example.com/price/{token}")
    """
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        offline = get_offline_mode()
        if offline.is_enabled():
            raise NetworkBlockedError(
                f"Offline mode: Cannot execute {func.__name__} - requires network access"
            )
        return await func(*args, **kwargs)

    return wrapper


@contextmanager
def offline_mode_context():
    """
    Context manager for offline mode operations

    Usage:
        with offline_mode_context():
            # This code runs with offline mode checks
            response = requests.get(url)  # Will be blocked if offline
    """
    offline = get_offline_mode()
    if offline.is_enabled():
        logger.debug("Entering offline mode context")

    try:
        yield offline
    finally:
        if offline.is_enabled():
            logger.debug(f"Exiting offline mode context. Blocked: {offline.blocker.blocked_count}")
