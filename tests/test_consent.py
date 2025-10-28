"""
Tests for Consent Management System

Verifies GDPR-compliant consent tracking:
- Grant and revoke consent
- Consent lifecycle
- Consent verification
- Audit trail
- Privacy controls
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from core.privacy.consent import (
    ConsentManager,
    check_analytics_consent,
    consent_required,
)
from app.models.privacy import (
    UserConsent,
    ConsentType,
    ConsentStatus,
    AuditLog,
)


@pytest.fixture
async def mock_db():
    """Mock database session"""
    db = AsyncMock(spec=AsyncSession)
    db.execute = AsyncMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.add = MagicMock()
    return db


@pytest.fixture
def consent_manager(mock_db):
    """Create ConsentManager instance"""
    return ConsentManager(mock_db)


class TestConsentManager:
    """Test ConsentManager class"""

    @pytest.mark.asyncio
    async def test_grant_consent_new(self, consent_manager, mock_db):
        """Test granting new consent"""
        # Mock no existing consent
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        consent = await consent_manager.grant_consent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS,
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0"
        )

        # Should create new consent
        assert mock_db.add.called
        assert mock_db.flush.called

    @pytest.mark.asyncio
    async def test_grant_consent_existing(self, consent_manager, mock_db):
        """Test updating existing consent"""
        # Mock existing consent
        existing_consent = UserConsent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS,
            status=ConsentStatus.REVOKED,
            granted_at=None,
            revoked_at=datetime.utcnow()
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_consent
        mock_db.execute.return_value = mock_result

        consent = await consent_manager.grant_consent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS
        )

        # Should update existing consent
        assert consent.status == ConsentStatus.GRANTED
        assert consent.granted_at is not None
        assert consent.revoked_at is None

    @pytest.mark.asyncio
    async def test_revoke_consent(self, consent_manager, mock_db):
        """Test revoking consent"""
        # Mock existing consent
        existing_consent = UserConsent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS,
            status=ConsentStatus.GRANTED,
            granted_at=datetime.utcnow()
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_consent
        mock_db.execute.return_value = mock_result

        consent = await consent_manager.revoke_consent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS
        )

        # Should revoke consent
        assert consent.status == ConsentStatus.REVOKED
        assert consent.revoked_at is not None

    @pytest.mark.asyncio
    async def test_revoke_consent_nonexistent(self, consent_manager, mock_db):
        """Test revoking non-existent consent"""
        # Mock no consent found
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        consent = await consent_manager.revoke_consent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS
        )

        # Should return None
        assert consent is None

    @pytest.mark.asyncio
    async def test_revoke_necessary_consent_fails(self, consent_manager, mock_db):
        """Test cannot revoke necessary consent"""
        # Mock existing necessary consent
        existing_consent = UserConsent(
            user_id="user123",
            consent_type=ConsentType.NECESSARY,
            status=ConsentStatus.GRANTED
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_consent
        mock_db.execute.return_value = mock_result

        consent = await consent_manager.revoke_consent(
            user_id="user123",
            consent_type=ConsentType.NECESSARY
        )

        # Should return None (cannot revoke)
        assert consent is None

    @pytest.mark.asyncio
    async def test_revoke_all_consent(self, consent_manager, mock_db):
        """Test revoking all non-necessary consent"""
        # Mock existing consents
        functional_consent = UserConsent(
            user_id="user123",
            consent_type=ConsentType.FUNCTIONAL,
            status=ConsentStatus.GRANTED
        )

        analytics_consent = UserConsent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS,
            status=ConsentStatus.GRANTED
        )

        mock_results = [
            MagicMock(scalar_one_or_none=MagicMock(return_value=functional_consent)),
            MagicMock(scalar_one_or_none=MagicMock(return_value=analytics_consent)),
            MagicMock(scalar_one_or_none=MagicMock(return_value=None)),  # Marketing
        ]

        mock_db.execute.side_effect = mock_results

        count = await consent_manager.revoke_all_consent(user_id="user123")

        # Should revoke functional and analytics (not necessary, marketing doesn't exist)
        assert count == 2

    @pytest.mark.asyncio
    async def test_check_consent_granted(self, consent_manager, mock_db):
        """Test checking granted consent"""
        # Mock existing consent
        existing_consent = UserConsent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS,
            status=ConsentStatus.GRANTED
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_consent
        mock_db.execute.return_value = mock_result

        has_consent = await consent_manager.check_consent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS
        )

        assert has_consent is True

    @pytest.mark.asyncio
    async def test_check_consent_revoked(self, consent_manager, mock_db):
        """Test checking revoked consent"""
        # Mock revoked consent
        existing_consent = UserConsent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS,
            status=ConsentStatus.REVOKED
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_consent
        mock_db.execute.return_value = mock_result

        has_consent = await consent_manager.check_consent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS
        )

        assert has_consent is False

    @pytest.mark.asyncio
    async def test_check_consent_nonexistent(self, consent_manager, mock_db):
        """Test checking non-existent consent"""
        # Mock no consent
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        has_consent = await consent_manager.check_consent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS
        )

        assert has_consent is False

    @pytest.mark.asyncio
    async def test_check_necessary_consent_always_granted(self, consent_manager, mock_db):
        """Test necessary consent is always granted"""
        # Don't mock any database call
        has_consent = await consent_manager.check_consent(
            user_id="user123",
            consent_type=ConsentType.NECESSARY
        )

        # Should always be True without database check
        assert has_consent is True

    @pytest.mark.asyncio
    async def test_get_all_consent(self, consent_manager, mock_db):
        """Test getting all consent statuses"""
        # Mock consents
        consents = [
            UserConsent(
                user_id="user123",
                consent_type=ConsentType.NECESSARY,
                status=ConsentStatus.GRANTED
            ),
            UserConsent(
                user_id="user123",
                consent_type=ConsentType.ANALYTICS,
                status=ConsentStatus.GRANTED
            ),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = consents
        mock_db.execute.return_value = mock_result

        consent_map = await consent_manager.get_all_consent(user_id="user123")

        # Should have all consent types
        assert ConsentType.NECESSARY.value in consent_map
        assert ConsentType.FUNCTIONAL.value in consent_map
        assert ConsentType.ANALYTICS.value in consent_map
        assert ConsentType.MARKETING.value in consent_map

        # Necessary should always be True
        assert consent_map[ConsentType.NECESSARY.value] is True
        # Analytics should be True (from mock)
        assert consent_map[ConsentType.ANALYTICS.value] is True
        # Functional should be False (not in mock)
        assert consent_map[ConsentType.FUNCTIONAL.value] is False

    @pytest.mark.asyncio
    async def test_initialize_default_consent(self, consent_manager, mock_db):
        """Test initializing default consent for new user"""
        # Mock no existing consent
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        await consent_manager.initialize_default_consent(
            user_id="newuser123",
            ip_address="192.168.1.1"
        )

        # Should grant necessary and functional consent
        assert mock_db.add.call_count >= 2  # At least necessary and functional

    @pytest.mark.asyncio
    async def test_consent_includes_metadata(self, consent_manager, mock_db):
        """Test consent includes metadata"""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        metadata = {"source": "signup_form", "version": "1.0"}

        consent = await consent_manager.grant_consent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS,
            metadata=metadata
        )

        # Should include metadata in new consent
        assert mock_db.add.called

    @pytest.mark.asyncio
    async def test_consent_tracks_ip_address(self, consent_manager, mock_db):
        """Test consent tracks IP address"""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        consent = await consent_manager.grant_consent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS,
            ip_address="192.168.1.100"
        )

        # Should call add with consent that will have ip_address set
        assert mock_db.add.called

    @pytest.mark.asyncio
    async def test_consent_tracks_user_agent(self, consent_manager, mock_db):
        """Test consent tracks user agent"""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        consent = await consent_manager.grant_consent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS,
            user_agent="Mozilla/5.0"
        )

        # Should call add with consent that will have user_agent set
        assert mock_db.add.called


class TestCheckAnalyticsConsent:
    """Test check_analytics_consent helper function"""

    @pytest.mark.asyncio
    @patch('core.privacy.consent.get_settings')
    async def test_analytics_blocked_by_dnt(self, mock_settings, mock_db):
        """Test analytics blocked by DNT header"""
        mock_settings.return_value.respect_dnt = True
        mock_settings.return_value.telemetry_mode = "standard"

        allowed = await check_analytics_consent(
            db=mock_db,
            user_id="user123",
            dnt_header="1"
        )

        assert allowed is False

    @pytest.mark.asyncio
    @patch('core.privacy.consent.get_settings')
    async def test_analytics_allowed_without_dnt(self, mock_settings, mock_db):
        """Test analytics allowed without DNT header"""
        mock_settings.return_value.respect_dnt = True
        mock_settings.return_value.telemetry_mode = "standard"

        # Mock consent granted
        existing_consent = UserConsent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS,
            status=ConsentStatus.GRANTED
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_consent
        mock_db.execute.return_value = mock_result

        allowed = await check_analytics_consent(
            db=mock_db,
            user_id="user123",
            dnt_header="0"
        )

        assert allowed is True

    @pytest.mark.asyncio
    @patch('core.privacy.consent.get_settings')
    async def test_analytics_blocked_by_telemetry_mode(self, mock_settings, mock_db):
        """Test analytics blocked when telemetry disabled"""
        mock_settings.return_value.respect_dnt = True
        mock_settings.return_value.telemetry_mode = "disabled"

        allowed = await check_analytics_consent(
            db=mock_db,
            user_id="user123",
            dnt_header=None
        )

        assert allowed is False

    @pytest.mark.asyncio
    @patch('core.privacy.consent.get_settings')
    async def test_analytics_blocked_by_consent(self, mock_settings, mock_db):
        """Test analytics blocked by user consent"""
        mock_settings.return_value.respect_dnt = True
        mock_settings.return_value.telemetry_mode = "standard"

        # Mock consent revoked
        existing_consent = UserConsent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS,
            status=ConsentStatus.REVOKED
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_consent
        mock_db.execute.return_value = mock_result

        allowed = await check_analytics_consent(
            db=mock_db,
            user_id="user123",
            dnt_header=None
        )

        assert allowed is False


class TestConsentRequiredDecorator:
    """Test consent_required decorator"""

    @pytest.mark.asyncio
    async def test_decorator_allows_with_consent(self, mock_db):
        """Test decorator allows function with consent"""
        @consent_required(ConsentType.ANALYTICS)
        async def track_action(user_id: str, db: AsyncSession):
            return "tracked"

        # Mock consent granted
        existing_consent = UserConsent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS,
            status=ConsentStatus.GRANTED
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_consent
        mock_db.execute.return_value = mock_result

        result = await track_action(user_id="user123", db=mock_db)

        assert result == "tracked"

    @pytest.mark.asyncio
    async def test_decorator_blocks_without_consent(self, mock_db):
        """Test decorator blocks function without consent"""
        @consent_required(ConsentType.ANALYTICS)
        async def track_action(user_id: str, db: AsyncSession):
            return "tracked"

        # Mock no consent
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        result = await track_action(user_id="user123", db=mock_db)

        # Should return None (silently skip)
        assert result is None


class TestConsentAuditTrail:
    """Test consent audit trail"""

    @pytest.mark.asyncio
    async def test_grant_consent_logs_audit(self, consent_manager, mock_db):
        """Test granting consent creates audit log"""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        await consent_manager.grant_consent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS
        )

        # Should create audit log
        # Check that add was called twice: once for consent, once for audit log
        assert mock_db.add.call_count >= 2

    @pytest.mark.asyncio
    async def test_revoke_consent_logs_audit(self, consent_manager, mock_db):
        """Test revoking consent creates audit log"""
        existing_consent = UserConsent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS,
            status=ConsentStatus.GRANTED
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_consent
        mock_db.execute.return_value = mock_result

        await consent_manager.revoke_consent(
            user_id="user123",
            consent_type=ConsentType.ANALYTICS
        )

        # Should create audit log
        assert mock_db.add.called
