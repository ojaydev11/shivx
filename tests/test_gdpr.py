"""
Tests for GDPR Compliance Features

Verifies all GDPR rights:
- Right to Access (data export)
- Right to Erasure (forget-me)
- Right to Rectification (correct data)
- Right to Data Portability
- Complete data purge
"""

import pytest
import hashlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from core.privacy.gdpr import GDPRCompliance, DataExportFormat, DataPurgeResult
from app.models.privacy import UserConsent, TelemetryPreference, DataRetention


@pytest.fixture
async def mock_db():
    """Mock database session"""
    db = AsyncMock(spec=AsyncSession)
    db.execute = AsyncMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.add = MagicMock()
    return db


@pytest.fixture
def gdpr_compliance(mock_db):
    """Create GDPRCompliance instance"""
    return GDPRCompliance(mock_db)


class TestDataExport:
    """Test data export functionality"""

    @pytest.mark.asyncio
    async def test_export_user_data_json(self, gdpr_compliance, mock_db):
        """Test exporting user data in JSON format"""
        # Mock user profile
        mock_user_result = MagicMock()
        mock_user_result.scalar_one_or_none.return_value = MagicMock(
            id="user123",
            username="testuser",
            email="test@example.com",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        # Mock consents
        mock_consent_result = MagicMock()
        mock_consent_result.scalars.return_value.all.return_value = []

        # Mock telemetry prefs
        mock_telemetry_result = MagicMock()
        mock_telemetry_result.scalar_one_or_none.return_value = None

        mock_db.execute.side_effect = [
            mock_user_result,
            mock_consent_result,
            mock_telemetry_result,
        ]

        data = await gdpr_compliance.export_user_data(
            user_id="user123",
            format=DataExportFormat.JSON
        )

        assert data["user_id"] == "user123"
        assert "export_date" in data
        assert "data" in data
        assert "profile" in data["data"]

    @pytest.mark.asyncio
    async def test_export_includes_all_data_types(self, gdpr_compliance, mock_db):
        """Test export includes all data types"""
        # Mock minimal data
        mock_db.execute.return_value.scalar_one_or_none.return_value = None
        mock_db.execute.return_value.scalars.return_value.all.return_value = []

        data = await gdpr_compliance.export_user_data(user_id="user123")

        # Should have all required sections
        assert "profile" in data["data"]
        assert "consents" in data["data"]
        assert "telemetry_preferences" in data["data"]
        assert "data_retention" in data["data"]
        assert "conversations" in data["data"]
        assert "memory" in data["data"]
        assert "audit_logs" in data["data"]
        assert "trading" in data["data"]

    @pytest.mark.asyncio
    async def test_export_includes_metadata(self, gdpr_compliance, mock_db):
        """Test export includes metadata when requested"""
        mock_db.execute.return_value.scalar_one_or_none.return_value = None
        mock_db.execute.return_value.scalars.return_value.all.return_value = []

        data = await gdpr_compliance.export_user_data(
            user_id="user123",
            include_metadata=True
        )

        assert "metadata" in data
        assert data["metadata"]["platform"] == "shivx"
        assert data["metadata"]["re_importable"] is True

    @pytest.mark.asyncio
    async def test_export_creates_audit_log(self, gdpr_compliance, mock_db):
        """Test export creates audit log entry"""
        mock_db.execute.return_value.scalar_one_or_none.return_value = None
        mock_db.execute.return_value.scalars.return_value.all.return_value = []

        await gdpr_compliance.export_user_data(user_id="user123")

        # Should create audit log
        assert mock_db.add.called

    @pytest.mark.asyncio
    async def test_export_error_logged(self, gdpr_compliance, mock_db):
        """Test export errors are logged"""
        mock_db.execute.side_effect = Exception("Database error")

        with pytest.raises(Exception):
            await gdpr_compliance.export_user_data(user_id="user123")

        # Should create error audit log
        assert mock_db.add.called


class TestForgetMe:
    """Test forget-me / data erasure functionality"""

    @pytest.mark.asyncio
    async def test_forget_user_requires_confirmation(self, gdpr_compliance, mock_db):
        """Test forget-me requires valid confirmation token"""
        with pytest.raises(ValueError, match="Invalid confirmation"):
            await gdpr_compliance.forget_user(
                user_id="user123",
                confirmation_token="wrong_token"
            )

    @pytest.mark.asyncio
    async def test_forget_user_with_valid_confirmation(self, gdpr_compliance, mock_db):
        """Test forget-me with valid confirmation"""
        # Generate correct token
        user_id = "user123"
        token = hashlib.sha256(user_id.encode()).hexdigest()[:16]

        # Mock delete operations
        mock_delete_result = MagicMock()
        mock_delete_result.rowcount = 1
        mock_db.execute.return_value = mock_delete_result

        # Mock no sessions/messages
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        result = await gdpr_compliance.forget_user(
            user_id=user_id,
            confirmation_token=token
        )

        assert isinstance(result, DataPurgeResult)
        assert result.end_time is not None
        assert mock_db.commit.called

    @pytest.mark.asyncio
    async def test_forget_user_deletes_consents(self, gdpr_compliance, mock_db):
        """Test forget-me deletes consent records"""
        user_id = "user123"
        token = hashlib.sha256(user_id.encode()).hexdigest()[:16]

        mock_delete_result = MagicMock()
        mock_delete_result.rowcount = 3  # 3 consents deleted
        mock_db.execute.return_value = mock_delete_result

        # Mock empty results for other queries
        mock_empty = MagicMock()
        mock_empty.scalars.return_value.all.return_value = []
        mock_db.execute.side_effect = [
            mock_delete_result,  # delete consents
            mock_delete_result,  # delete telemetry
            mock_delete_result,  # delete retention
            mock_empty,  # get sessions
            mock_empty,  # get memories
            mock_empty,  # get positions
            mock_empty,  # get orders
            mock_empty,  # get api keys
            mock_empty,  # get audit logs
            mock_delete_result,  # delete user
        ]

        result = await gdpr_compliance.forget_user(user_id=user_id, confirmation_token=token)

        assert result.tables_purged["user_consents"] == 3

    @pytest.mark.asyncio
    async def test_forget_user_anonymizes_audit_logs(self, gdpr_compliance, mock_db):
        """Test forget-me anonymizes audit logs instead of deleting"""
        user_id = "user123"
        token = hashlib.sha256(user_id.encode()).hexdigest()[:16]

        # Mock audit logs
        audit_log1 = MagicMock()
        audit_log1.user_id = user_id
        audit_log1.ip_address = "192.168.1.1"
        audit_log1.user_agent = "TestAgent"

        mock_logs_result = MagicMock()
        mock_logs_result.scalars.return_value.all.return_value = [audit_log1]

        mock_delete_result = MagicMock()
        mock_delete_result.rowcount = 0

        mock_empty = MagicMock()
        mock_empty.scalars.return_value.all.return_value = []

        mock_db.execute.side_effect = [
            mock_delete_result,  # delete consents
            mock_delete_result,  # delete telemetry
            mock_delete_result,  # delete retention
            mock_empty,  # get sessions
            mock_empty,  # get memories
            mock_empty,  # get positions
            mock_empty,  # get orders
            mock_empty,  # get api keys
            mock_logs_result,  # get audit logs
            mock_delete_result,  # delete user
        ]

        result = await gdpr_compliance.forget_user(user_id=user_id, confirmation_token=token)

        # Audit log should be anonymized, not deleted
        assert audit_log1.user_id.startswith("deleted_")
        assert audit_log1.ip_address == "0.0.0.0"
        assert audit_log1.user_agent == "anonymized"

    @pytest.mark.asyncio
    async def test_forget_user_creates_audit_trail(self, gdpr_compliance, mock_db):
        """Test forget-me creates final audit log"""
        user_id = "user123"
        token = hashlib.sha256(user_id.encode()).hexdigest()[:16]

        mock_delete_result = MagicMock()
        mock_delete_result.rowcount = 0
        mock_db.execute.return_value = mock_delete_result

        mock_empty = MagicMock()
        mock_empty.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_empty

        await gdpr_compliance.forget_user(user_id=user_id, confirmation_token=token)

        # Should create final audit log
        assert mock_db.add.called

    @pytest.mark.asyncio
    async def test_forget_user_rollback_on_error(self, gdpr_compliance, mock_db):
        """Test forget-me rolls back on error"""
        user_id = "user123"
        token = hashlib.sha256(user_id.encode()).hexdigest()[:16]

        # Simulate error during deletion
        mock_db.execute.side_effect = Exception("Database error")

        with pytest.raises(Exception):
            await gdpr_compliance.forget_user(user_id=user_id, confirmation_token=token)

        # Should rollback
        assert mock_db.rollback.called

    @pytest.mark.asyncio
    async def test_forget_user_tracks_files_deleted(self, gdpr_compliance, mock_db):
        """Test forget-me tracks deleted files"""
        user_id = "user123"
        token = hashlib.sha256(user_id.encode()).hexdigest()[:16]

        mock_delete_result = MagicMock()
        mock_delete_result.rowcount = 0
        mock_db.execute.return_value = mock_delete_result

        mock_empty = MagicMock()
        mock_empty.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_empty

        # Mock file deletion
        with patch.object(gdpr_compliance, '_delete_user_files', return_value=["file1.txt", "file2.json"]):
            result = await gdpr_compliance.forget_user(user_id=user_id, confirmation_token=token)

        assert len(result.files_deleted) == 2


class TestDataRectification:
    """Test data rectification functionality"""

    @pytest.mark.asyncio
    async def test_rectify_user_profile(self, gdpr_compliance, mock_db):
        """Test rectifying user profile data"""
        mock_user = MagicMock()
        mock_user.email = "old@example.com"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result

        corrections = {
            "profile": {"email": "new@example.com"}
        }

        result = await gdpr_compliance.rectify_user_data(
            user_id="user123",
            corrections=corrections
        )

        assert "profile" in result["corrections_applied"]
        assert mock_db.commit.called

    @pytest.mark.asyncio
    async def test_rectify_creates_audit_log(self, gdpr_compliance, mock_db):
        """Test rectification creates audit log"""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = MagicMock()
        mock_db.execute.return_value = mock_result

        corrections = {"profile": {"email": "new@example.com"}}

        await gdpr_compliance.rectify_user_data(
            user_id="user123",
            corrections=corrections
        )

        # Should create audit log
        assert mock_db.add.called

    @pytest.mark.asyncio
    async def test_rectify_rollback_on_error(self, gdpr_compliance, mock_db):
        """Test rectification rolls back on error"""
        mock_db.execute.side_effect = Exception("Error")

        corrections = {"profile": {"email": "new@example.com"}}

        with pytest.raises(Exception):
            await gdpr_compliance.rectify_user_data(
                user_id="user123",
                corrections=corrections
            )

        assert mock_db.rollback.called


class TestDataExportHelpers:
    """Test data export helper methods"""

    @pytest.mark.asyncio
    async def test_export_user_consents(self, gdpr_compliance, mock_db):
        """Test exporting user consents"""
        consent1 = MagicMock()
        consent1.consent_type.value = "analytics"
        consent1.status.value = "granted"
        consent1.granted_at = datetime.utcnow()
        consent1.revoked_at = None

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [consent1]
        mock_db.execute.return_value = mock_result

        consents = await gdpr_compliance._export_user_consents("user123")

        assert len(consents) == 1
        assert consents[0]["consent_type"] == "analytics"
        assert consents[0]["status"] == "granted"

    @pytest.mark.asyncio
    async def test_export_telemetry_prefs(self, gdpr_compliance, mock_db):
        """Test exporting telemetry preferences"""
        pref = MagicMock()
        pref.telemetry_mode.value = "standard"
        pref.do_not_track = False
        pref.collect_errors = True
        pref.collect_performance = True
        pref.collect_usage = False

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = pref
        mock_db.execute.return_value = mock_result

        prefs = await gdpr_compliance._export_telemetry_prefs("user123")

        assert prefs["telemetry_mode"] == "standard"
        assert prefs["do_not_track"] is False

    @pytest.mark.asyncio
    async def test_export_audit_logs(self, gdpr_compliance, mock_db):
        """Test exporting audit logs"""
        log1 = MagicMock()
        log1.action = "login"
        log1.resource_type = "session"
        log1.resource_id = "session123"
        log1.status = "success"
        log1.performed_at = datetime.utcnow()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [log1]
        mock_db.execute.return_value = mock_result

        logs = await gdpr_compliance._export_audit_logs("user123")

        assert len(logs) == 1
        assert logs[0]["action"] == "login"
        assert logs[0]["status"] == "success"


class TestDataDeletionHelpers:
    """Test data deletion helper methods"""

    @pytest.mark.asyncio
    async def test_delete_user_consents(self, gdpr_compliance, mock_db):
        """Test deleting user consents"""
        mock_result = MagicMock()
        mock_result.rowcount = 3
        mock_db.execute.return_value = mock_result

        count = await gdpr_compliance._delete_user_consents("user123")

        assert count == 3

    @pytest.mark.asyncio
    async def test_delete_conversations(self, gdpr_compliance, mock_db):
        """Test deleting conversations"""
        session1 = MagicMock()
        session1.id = "session1"

        mock_session_result = MagicMock()
        mock_session_result.scalars.return_value.all.return_value = [session1]

        mock_delete_result = MagicMock()
        mock_delete_result.rowcount = 5

        mock_db.execute.side_effect = [
            mock_session_result,  # Get sessions
            mock_delete_result,  # Delete messages
            mock_delete_result,  # Delete sessions
        ]

        count = await gdpr_compliance._delete_conversations("user123")

        assert count == 10  # 5 messages + 5 sessions


class TestDataPurgeResult:
    """Test DataPurgeResult tracking"""

    def test_purge_result_initialization(self):
        """Test DataPurgeResult initialization"""
        result = DataPurgeResult()

        assert result.tables_purged == {}
        assert result.files_deleted == []
        assert result.errors == []
        assert result.start_time is not None
        assert result.end_time is None

    def test_add_table_purge(self):
        """Test adding table purge record"""
        result = DataPurgeResult()
        result.add_table_purge("users", 1)
        result.add_table_purge("consents", 3)

        assert result.tables_purged["users"] == 1
        assert result.tables_purged["consents"] == 3

    def test_add_file_deletion(self):
        """Test adding file deletion record"""
        result = DataPurgeResult()
        result.add_file_deletion("file1.txt")
        result.add_file_deletion("file2.json")

        assert len(result.files_deleted) == 2
        assert "file1.txt" in result.files_deleted

    def test_add_error(self):
        """Test adding error record"""
        result = DataPurgeResult()
        result.add_error("Error 1")
        result.add_error("Error 2")

        assert len(result.errors) == 2
        assert "Error 1" in result.errors

    def test_finalize(self):
        """Test finalizing purge result"""
        result = DataPurgeResult()
        result.finalize()

        assert result.end_time is not None

    def test_to_dict(self):
        """Test converting to dictionary"""
        result = DataPurgeResult()
        result.add_table_purge("users", 1)
        result.add_file_deletion("file.txt")
        result.add_error("error")
        result.finalize()

        data = result.to_dict()

        assert "tables_purged" in data
        assert "total_records_deleted" in data
        assert "files_deleted" in data
        assert "errors" in data
        assert data["total_records_deleted"] == 1
        assert data["total_files_deleted"] == 1
        assert data["has_errors"] is True
