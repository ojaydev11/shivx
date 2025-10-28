"""
Privacy Controls for ShivX Platform

Comprehensive privacy and data protection features:
- Offline mode enforcement
- Telemetry privacy controls
- Consent management (GDPR compliant)
- Data retention policies
- Air-gapped mode
- Right to access, rectification, and erasure (GDPR)
"""

from core.privacy.consent import (
    ConsentManager,
    ConsentType,
    ConsentStatus,
    UserConsent,
)

from core.privacy.gdpr import (
    GDPRCompliance,
    DataExportFormat,
    DataPurgeResult,
)

from core.privacy.airgap import (
    AirGapMode,
    NetworkMonitor,
    check_network_isolation,
)

from core.privacy.offline import (
    OfflineMode,
    NetworkBlocker,
    is_offline_mode,
)

__all__ = [
    # Consent
    "ConsentManager",
    "ConsentType",
    "ConsentStatus",
    "UserConsent",
    # GDPR
    "GDPRCompliance",
    "DataExportFormat",
    "DataPurgeResult",
    # Air-gap
    "AirGapMode",
    "NetworkMonitor",
    "check_network_isolation",
    # Offline
    "OfflineMode",
    "NetworkBlocker",
    "is_offline_mode",
]
