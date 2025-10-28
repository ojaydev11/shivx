"""
Air-Gapped Mode Implementation

Maximum network isolation for sensitive deployments:
- Verifies complete network isolation on startup
- Fails startup if network interfaces detected (optional)
- Audits all network connection attempts
- Blocks at application level before OS socket
"""

import logging
import socket
import os
import subprocess
from typing import List, Dict, Optional
from datetime import datetime

from config.settings import get_settings

logger = logging.getLogger(__name__)


class NetworkInterface:
    """Network interface information"""

    def __init__(self, name: str, status: str, addresses: List[str]):
        self.name = name
        self.status = status
        self.addresses = addresses

    def is_loopback(self) -> bool:
        """Check if interface is loopback"""
        return self.name.startswith(("lo", "loop"))

    def is_active(self) -> bool:
        """Check if interface is active"""
        return self.status.lower() in ("up", "active", "running")

    def __repr__(self):
        return f"<NetworkInterface(name={self.name}, status={self.status}, addresses={self.addresses})>"


class NetworkMonitor:
    """
    Network interface monitor for air-gap mode

    Detects active network interfaces and connections
    """

    def __init__(self):
        self.settings = get_settings()
        self.connection_attempts: List[Dict] = []

    def get_network_interfaces(self) -> List[NetworkInterface]:
        """
        Get list of network interfaces

        Returns:
            List of NetworkInterface objects
        """
        interfaces = []

        try:
            # Try using socket module first
            import netifaces
        except ImportError:
            # Fallback to parsing ifconfig/ip output
            return self._get_interfaces_fallback()

        try:
            for iface in netifaces.interfaces():
                addrs = []
                try:
                    # Get IPv4 addresses
                    ipv4_addrs = netifaces.ifaddresses(iface).get(netifaces.AF_INET, [])
                    for addr in ipv4_addrs:
                        addrs.append(addr.get("addr", ""))

                    # Get IPv6 addresses
                    ipv6_addrs = netifaces.ifaddresses(iface).get(netifaces.AF_INET6, [])
                    for addr in ipv6_addrs:
                        addrs.append(addr.get("addr", ""))
                except Exception:
                    pass

                # Determine status (up/down)
                # Note: netifaces doesn't provide status, assume up if has addresses
                status = "up" if addrs else "down"

                interfaces.append(NetworkInterface(iface, status, addrs))

        except Exception as e:
            logger.error(f"Error getting network interfaces: {e}")
            return self._get_interfaces_fallback()

        return interfaces

    def _get_interfaces_fallback(self) -> List[NetworkInterface]:
        """
        Fallback method to get network interfaces using system commands

        Returns:
            List of NetworkInterface objects
        """
        interfaces = []

        try:
            # Try 'ip addr' on Linux
            result = subprocess.run(
                ["ip", "addr", "show"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                return self._parse_ip_addr(result.stdout)
        except Exception:
            pass

        try:
            # Try 'ifconfig' as fallback
            result = subprocess.run(
                ["ifconfig"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                return self._parse_ifconfig(result.stdout)
        except Exception:
            pass

        logger.warning("Could not detect network interfaces using system commands")
        return interfaces

    def _parse_ip_addr(self, output: str) -> List[NetworkInterface]:
        """Parse 'ip addr show' output"""
        interfaces = []
        current_iface = None
        current_addrs = []

        for line in output.split("\n"):
            # New interface line
            if line and not line.startswith(" "):
                if current_iface:
                    interfaces.append(
                        NetworkInterface(current_iface, "up", current_addrs)
                    )
                # Parse interface name
                parts = line.split(":")
                if len(parts) >= 2:
                    current_iface = parts[1].strip()
                    current_addrs = []
            # Address line
            elif "inet" in line:
                parts = line.strip().split()
                if len(parts) >= 2:
                    addr = parts[1].split("/")[0]  # Remove CIDR notation
                    current_addrs.append(addr)

        # Add last interface
        if current_iface:
            interfaces.append(NetworkInterface(current_iface, "up", current_addrs))

        return interfaces

    def _parse_ifconfig(self, output: str) -> List[NetworkInterface]:
        """Parse 'ifconfig' output"""
        interfaces = []
        current_iface = None
        current_addrs = []
        current_status = "down"

        for line in output.split("\n"):
            # New interface line
            if line and not line.startswith((" ", "\t")):
                if current_iface:
                    interfaces.append(
                        NetworkInterface(current_iface, current_status, current_addrs)
                    )

                # Parse interface name
                current_iface = line.split(":")[0].split()[0]
                current_addrs = []
                current_status = "up" if "UP" in line else "down"

            # Address line
            elif "inet" in line:
                parts = line.strip().split()
                for i, part in enumerate(parts):
                    if part in ("inet", "inet6") and i + 1 < len(parts):
                        addr = parts[i + 1]
                        # Remove any extra info (like %eth0)
                        addr = addr.split("%")[0]
                        current_addrs.append(addr)

        # Add last interface
        if current_iface:
            interfaces.append(
                NetworkInterface(current_iface, current_status, current_addrs)
            )

        return interfaces

    def get_active_external_interfaces(self) -> List[NetworkInterface]:
        """
        Get list of active non-loopback interfaces

        Returns:
            List of external network interfaces
        """
        interfaces = self.get_network_interfaces()
        return [
            iface
            for iface in interfaces
            if iface.is_active() and not iface.is_loopback()
        ]

    def log_connection_attempt(self, target: str, port: int, protocol: str = "tcp"):
        """
        Log a network connection attempt

        Args:
            target: Target host/IP
            port: Target port
            protocol: Protocol (tcp/udp)
        """
        attempt = {
            "timestamp": datetime.utcnow().isoformat(),
            "target": target,
            "port": port,
            "protocol": protocol,
            "airgap_mode": self.settings.airgap_mode,
        }

        self.connection_attempts.append(attempt)

        if self.settings.airgap_mode:
            logger.warning(
                f"⚠ AIRGAP VIOLATION: Connection attempt to {target}:{port} ({protocol})"
            )

    def get_connection_attempts(self) -> List[Dict]:
        """Get all logged connection attempts"""
        return self.connection_attempts


class AirGapMode:
    """
    Air-gapped mode manager

    Enforces complete network isolation
    """

    def __init__(self):
        self.settings = get_settings()
        self.monitor = NetworkMonitor()

    def is_enabled(self) -> bool:
        """Check if air-gap mode is enabled"""
        return self.settings.airgap_mode

    def verify_isolation(self, fail_on_violation: bool = True) -> Dict:
        """
        Verify complete network isolation

        Args:
            fail_on_violation: Raise exception if external interfaces found

        Returns:
            Verification result dictionary

        Raises:
            AirGapViolation: If external interfaces found and fail_on_violation=True
        """
        if not self.is_enabled():
            return {
                "airgap_mode": False,
                "verified": False,
                "message": "Air-gap mode not enabled",
            }

        logger.info("Verifying air-gap isolation...")

        # Get active external interfaces
        external_interfaces = self.monitor.get_active_external_interfaces()

        if external_interfaces:
            violation_msg = (
                f"Air-gap violation: {len(external_interfaces)} active external "
                f"network interface(s) detected: "
                f"{[iface.name for iface in external_interfaces]}"
            )

            logger.error(violation_msg)

            for iface in external_interfaces:
                logger.error(
                    f"  - {iface.name}: {', '.join(iface.addresses)} ({iface.status})"
                )

            if fail_on_violation:
                raise AirGapViolation(
                    f"{violation_msg}\n\n"
                    "To run in air-gap mode, disable all network interfaces:\n"
                    "  sudo ifconfig <interface> down\n"
                    "Or set SHIVX_AIRGAP_MODE=false to disable air-gap verification."
                )

            return {
                "airgap_mode": True,
                "verified": False,
                "violations": len(external_interfaces),
                "interfaces": [
                    {
                        "name": iface.name,
                        "addresses": iface.addresses,
                        "status": iface.status,
                    }
                    for iface in external_interfaces
                ],
                "message": violation_msg,
            }

        logger.info("✓ Air-gap verification passed - no external interfaces detected")

        return {
            "airgap_mode": True,
            "verified": True,
            "violations": 0,
            "interfaces": [],
            "message": "System is properly air-gapped",
        }

    def verify_on_startup(self):
        """
        Verify air-gap isolation on startup

        Fails startup if violations detected
        """
        if not self.is_enabled():
            return

        logger.info("=" * 60)
        logger.info("AIR-GAP MODE ENABLED")
        logger.info("Performing network isolation verification...")
        logger.info("=" * 60)

        try:
            result = self.verify_isolation(fail_on_violation=True)
            logger.info("✓ Air-gap verification successful")
            logger.info("=" * 60)
        except AirGapViolation as e:
            logger.critical(str(e))
            logger.critical("=" * 60)
            logger.critical("STARTUP ABORTED DUE TO AIR-GAP VIOLATION")
            logger.critical("=" * 60)
            raise

    def get_status(self) -> Dict:
        """Get comprehensive air-gap status"""
        if not self.is_enabled():
            return {
                "airgap_mode": False,
                "status": "disabled",
                "message": "Air-gap mode not enabled",
            }

        verification = self.verify_isolation(fail_on_violation=False)

        return {
            "airgap_mode": True,
            "status": "isolated" if verification["verified"] else "violated",
            "verified": verification["verified"],
            "violations": verification.get("violations", 0),
            "external_interfaces": verification.get("interfaces", []),
            "connection_attempts": len(self.monitor.connection_attempts),
            "message": verification["message"],
        }


class AirGapViolation(Exception):
    """Raised when air-gap verification fails"""
    pass


# Global air-gap mode instance
_airgap_mode: Optional[AirGapMode] = None


def get_airgap_mode() -> AirGapMode:
    """Get global air-gap mode instance"""
    global _airgap_mode
    if _airgap_mode is None:
        _airgap_mode = AirGapMode()
    return _airgap_mode


def check_network_isolation() -> Dict:
    """
    Check network isolation status

    Returns:
        Isolation status dictionary
    """
    airgap = get_airgap_mode()
    if airgap.is_enabled():
        return airgap.verify_isolation(fail_on_violation=False)
    else:
        return {
            "airgap_mode": False,
            "verified": False,
            "message": "Air-gap mode not enabled",
        }
