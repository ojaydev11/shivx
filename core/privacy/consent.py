"""
Consent Management System

GDPR-compliant consent tracking and enforcement:
- Granular consent types (necessary, functional, analytics, marketing)
- Consent lifecycle management (grant, revoke, expire)
- Consent verification before data collection
- Audit trail of all consent changes
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.privacy import (
    UserConsent,
    ConsentType,
    ConsentStatus,
    AuditLog,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ConsentManager:
    """
    Manages user consent for GDPR compliance

    Handles consent tracking, verification, and enforcement
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()

    async def grant_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> UserConsent:
        """
        Grant consent for a specific type

        Args:
            user_id: User identifier
            consent_type: Type of consent
            ip_address: User's IP address
            user_agent: User's user agent
            metadata: Additional context

        Returns:
            UserConsent record
        """
        # Check if consent already exists
        existing = await self._get_consent(user_id, consent_type)

        if existing:
            # Update existing consent
            existing.status = ConsentStatus.GRANTED
            existing.granted_at = datetime.utcnow()
            existing.revoked_at = None
            existing.ip_address = ip_address
            existing.user_agent = user_agent
            if metadata:
                existing.metadata = metadata

            consent = existing
        else:
            # Create new consent
            consent = UserConsent(
                user_id=user_id,
                consent_type=consent_type,
                status=ConsentStatus.GRANTED,
                granted_at=datetime.utcnow(),
                ip_address=ip_address,
                user_agent=user_agent,
                metadata=metadata or {},
            )
            self.db.add(consent)

        await self.db.flush()

        # Log audit trail
        await self._log_audit(
            user_id=user_id,
            action="consent.granted",
            resource_type="consent",
            resource_id=str(consent.id),
            status="success",
            ip_address=ip_address,
            user_agent=user_agent,
            metadata={
                "consent_type": consent_type.value,
            },
        )

        logger.info(f"Consent granted: user={user_id}, type={consent_type.value}")

        return consent

    async def revoke_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[UserConsent]:
        """
        Revoke consent for a specific type

        Args:
            user_id: User identifier
            consent_type: Type of consent
            ip_address: User's IP address
            user_agent: User's user agent

        Returns:
            Updated UserConsent record or None if not found
        """
        consent = await self._get_consent(user_id, consent_type)

        if not consent:
            logger.warning(
                f"Cannot revoke non-existent consent: user={user_id}, type={consent_type.value}"
            )
            return None

        # Cannot revoke necessary consent
        if consent_type == ConsentType.NECESSARY:
            logger.warning(
                f"Cannot revoke necessary consent: user={user_id}"
            )
            return None

        consent.status = ConsentStatus.REVOKED
        consent.revoked_at = datetime.utcnow()

        await self.db.flush()

        # Log audit trail
        await self._log_audit(
            user_id=user_id,
            action="consent.revoked",
            resource_type="consent",
            resource_id=str(consent.id),
            status="success",
            ip_address=ip_address,
            user_agent=user_agent,
            metadata={
                "consent_type": consent_type.value,
            },
        )

        logger.info(f"Consent revoked: user={user_id}, type={consent_type.value}")

        return consent

    async def revoke_all_consent(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> int:
        """
        Revoke all non-necessary consent for a user

        Args:
            user_id: User identifier
            ip_address: User's IP address
            user_agent: User's user agent

        Returns:
            Number of consents revoked
        """
        count = 0

        for consent_type in [ConsentType.FUNCTIONAL, ConsentType.ANALYTICS, ConsentType.MARKETING]:
            consent = await self.revoke_consent(
                user_id=user_id,
                consent_type=consent_type,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            if consent:
                count += 1

        logger.info(f"All consent revoked: user={user_id}, count={count}")

        return count

    async def check_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
    ) -> bool:
        """
        Check if user has granted consent for a specific type

        Args:
            user_id: User identifier
            consent_type: Type of consent

        Returns:
            True if consent granted, False otherwise
        """
        # Necessary consent is always granted
        if consent_type == ConsentType.NECESSARY:
            return True

        consent = await self._get_consent(user_id, consent_type)

        if not consent:
            # No consent record = not granted (except necessary)
            return False

        return consent.status == ConsentStatus.GRANTED

    async def get_all_consent(self, user_id: str) -> Dict[str, bool]:
        """
        Get all consent status for a user

        Args:
            user_id: User identifier

        Returns:
            Dictionary mapping consent type to granted status
        """
        result = await self.db.execute(
            select(UserConsent).where(UserConsent.user_id == user_id)
        )
        consents = result.scalars().all()

        consent_map = {
            consent_type.value: False
            for consent_type in ConsentType
        }

        # Necessary is always granted
        consent_map[ConsentType.NECESSARY.value] = True

        for consent in consents:
            consent_map[consent.consent_type.value] = (
                consent.status == ConsentStatus.GRANTED
            )

        return consent_map

    async def initialize_default_consent(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        """
        Initialize default consent for a new user

        By default:
        - Necessary: Always granted
        - Functional: Granted
        - Analytics: Not granted (requires explicit consent)
        - Marketing: Not granted (requires explicit consent)

        Args:
            user_id: User identifier
            ip_address: User's IP address
            user_agent: User's user agent
        """
        # Grant necessary consent (required for basic operation)
        await self.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.NECESSARY,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata={"auto_granted": True, "reason": "Required for basic functionality"},
        )

        # Grant functional consent by default
        await self.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.FUNCTIONAL,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata={"auto_granted": True, "reason": "Default consent"},
        )

        logger.info(f"Initialized default consent for user: {user_id}")

    async def _get_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
    ) -> Optional[UserConsent]:
        """Get consent record for user and type"""
        result = await self.db.execute(
            select(UserConsent).where(
                and_(
                    UserConsent.user_id == user_id,
                    UserConsent.consent_type == consent_type,
                )
            )
        )
        return result.scalar_one_or_none()

    async def _log_audit(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        status: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """Log consent action to audit log"""
        audit = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
        )
        self.db.add(audit)
        await self.db.flush()


async def check_analytics_consent(
    db: AsyncSession,
    user_id: str,
    dnt_header: Optional[str] = None,
) -> bool:
    """
    Check if analytics/telemetry is allowed for user

    Respects both consent and Do Not Track (DNT) header

    Args:
        db: Database session
        user_id: User identifier
        dnt_header: DNT header value from HTTP request

    Returns:
        True if analytics allowed, False otherwise
    """
    settings = get_settings()

    # Check DNT header if configured to respect it
    if settings.respect_dnt and dnt_header == "1":
        logger.debug(f"Analytics blocked by DNT header: user={user_id}")
        return False

    # Check telemetry mode
    if settings.telemetry_mode == "disabled":
        return False

    # Check user consent
    consent_mgr = ConsentManager(db)
    has_consent = await consent_mgr.check_consent(user_id, ConsentType.ANALYTICS)

    return has_consent


def consent_required(consent_type: ConsentType):
    """
    Decorator to enforce consent checks

    Usage:
        @consent_required(ConsentType.ANALYTICS)
        async def track_user_action(user_id: str, action: str, db: AsyncSession):
            # This will only execute if user has granted analytics consent
            ...
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, user_id: str, db: AsyncSession, **kwargs):
            # Check consent
            consent_mgr = ConsentManager(db)
            has_consent = await consent_mgr.check_consent(user_id, consent_type)

            if not has_consent:
                logger.debug(
                    f"Operation blocked - no consent: "
                    f"user={user_id}, type={consent_type.value}, func={func.__name__}"
                )
                return None  # Silently skip operation

            # Execute function
            return await func(*args, user_id=user_id, db=db, **kwargs)

        return wrapper

    return decorator
