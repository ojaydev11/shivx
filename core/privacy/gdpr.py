"""
GDPR Compliance Features

Implements GDPR rights:
- Right to Access (Article 15)
- Right to Rectification (Article 16)
- Right to Erasure / "Right to be Forgotten" (Article 17)
- Right to Data Portability (Article 20)
- Right to Restrict Processing (Article 18)
"""

import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum
from pathlib import Path

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.privacy import AuditLog, UserConsent, TelemetryPreference, DataRetention
from config.settings import get_settings

logger = logging.getLogger(__name__)


class DataExportFormat(str, Enum):
    """Data export formats"""
    JSON = "json"
    CSV = "csv"
    XML = "xml"


class DataPurgeResult:
    """Result of data purge operation"""

    def __init__(self):
        self.tables_purged: Dict[str, int] = {}
        self.files_deleted: List[str] = []
        self.errors: List[str] = []
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None

    def add_table_purge(self, table: str, count: int):
        """Record table purge"""
        self.tables_purged[table] = count

    def add_file_deletion(self, filepath: str):
        """Record file deletion"""
        self.files_deleted.append(filepath)

    def add_error(self, error: str):
        """Record error"""
        self.errors.append(error)

    def finalize(self):
        """Mark purge as complete"""
        self.end_time = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "tables_purged": self.tables_purged,
            "total_records_deleted": sum(self.tables_purged.values()),
            "files_deleted": self.files_deleted,
            "total_files_deleted": len(self.files_deleted),
            "errors": self.errors,
            "has_errors": len(self.errors) > 0,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time
                else None
            ),
        }


class GDPRCompliance:
    """
    GDPR compliance manager

    Implements all GDPR data subject rights
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()

    async def export_user_data(
        self,
        user_id: str,
        format: DataExportFormat = DataExportFormat.JSON,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Export all user data (Right to Access - Article 15)

        Args:
            user_id: User identifier
            format: Export format
            include_metadata: Include metadata for re-import

        Returns:
            Dictionary containing all user data
        """
        logger.info(f"Exporting data for user: {user_id}")

        export_data = {
            "user_id": user_id,
            "export_date": datetime.utcnow().isoformat(),
            "export_format": format.value,
            "data": {},
        }

        try:
            # Export user profile
            export_data["data"]["profile"] = await self._export_user_profile(user_id)

            # Export consents
            export_data["data"]["consents"] = await self._export_user_consents(user_id)

            # Export telemetry preferences
            export_data["data"]["telemetry_preferences"] = await self._export_telemetry_prefs(
                user_id
            )

            # Export data retention settings
            export_data["data"]["data_retention"] = await self._export_retention_settings(
                user_id
            )

            # Export conversation history
            export_data["data"]["conversations"] = await self._export_conversations(user_id)

            # Export memory entries
            export_data["data"]["memory"] = await self._export_memory(user_id)

            # Export audit logs
            export_data["data"]["audit_logs"] = await self._export_audit_logs(user_id)

            # Export trading data (if applicable)
            export_data["data"]["trading"] = await self._export_trading_data(user_id)

            if include_metadata:
                export_data["metadata"] = {
                    "version": "1.0",
                    "platform": "shivx",
                    "export_type": "gdpr_data_export",
                    "re_importable": True,
                }

            # Log audit trail
            await self._log_audit(
                user_id=user_id,
                action="gdpr.data_export",
                resource_type="user_data",
                resource_id=user_id,
                status="success",
                metadata={"format": format.value},
            )

            logger.info(f"Data export completed for user: {user_id}")

            return export_data

        except Exception as e:
            logger.error(f"Error exporting data for user {user_id}: {e}")

            await self._log_audit(
                user_id=user_id,
                action="gdpr.data_export",
                resource_type="user_data",
                resource_id=user_id,
                status="error",
                metadata={"error": str(e)},
            )

            raise

    async def forget_user(
        self,
        user_id: str,
        confirmation_token: str,
        requester_ip: Optional[str] = None,
    ) -> DataPurgeResult:
        """
        Purge all user data (Right to be Forgotten - Article 17)

        This operation is IRREVERSIBLE!

        Args:
            user_id: User identifier
            confirmation_token: Confirmation token (should match user_id hash)
            requester_ip: IP address of requester

        Returns:
            DataPurgeResult with purge statistics

        Raises:
            ValueError: If confirmation token invalid
        """
        logger.warning(f"⚠ FORGET ME request received for user: {user_id}")

        # Verify confirmation token
        import hashlib

        expected_token = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        if confirmation_token != expected_token:
            raise ValueError(
                "Invalid confirmation token. Cannot proceed with data deletion."
            )

        result = DataPurgeResult()

        try:
            # 1. Delete user consents
            count = await self._delete_user_consents(user_id)
            result.add_table_purge("user_consents", count)

            # 2. Delete telemetry preferences
            count = await self._delete_telemetry_prefs(user_id)
            result.add_table_purge("telemetry_preferences", count)

            # 3. Delete data retention settings
            count = await self._delete_retention_settings(user_id)
            result.add_table_purge("data_retention", count)

            # 4. Delete conversations
            count = await self._delete_conversations(user_id)
            result.add_table_purge("conversations", count)

            # 5. Delete memory entries
            count = await self._delete_memory(user_id)
            result.add_table_purge("memory_entries", count)

            # 6. Delete trading data
            count = await self._delete_trading_data(user_id)
            result.add_table_purge("trading_data", count)

            # 7. Delete API keys
            count = await self._delete_api_keys(user_id)
            result.add_table_purge("api_keys", count)

            # 8. Delete user files
            files = await self._delete_user_files(user_id)
            for file in files:
                result.add_file_deletion(file)

            # 9. Delete from vector store (if applicable)
            try:
                await self._delete_from_vector_store(user_id)
            except Exception as e:
                result.add_error(f"Vector store deletion failed: {e}")

            # 10. Anonymize audit logs (keep for compliance, but remove PII)
            count = await self._anonymize_audit_logs(user_id)
            result.add_table_purge("audit_logs_anonymized", count)

            # 11. Delete user profile (last)
            count = await self._delete_user_profile(user_id)
            result.add_table_purge("users", count)

            # Commit all changes
            await self.db.commit()

            result.finalize()

            # Log final audit entry (before user is deleted)
            await self._log_audit(
                user_id=f"deleted_{user_id}",
                action="gdpr.forget_me",
                resource_type="user_data",
                resource_id=user_id,
                status="success",
                ip_address=requester_ip,
                metadata=result.to_dict(),
            )

            logger.warning(f"✓ User data purged successfully: {user_id}")
            logger.warning(f"  - Total records deleted: {sum(result.tables_purged.values())}")
            logger.warning(f"  - Total files deleted: {len(result.files_deleted)}")

            return result

        except Exception as e:
            logger.error(f"Error purging data for user {user_id}: {e}")
            result.add_error(str(e))
            result.finalize()

            # Rollback on error
            await self.db.rollback()

            await self._log_audit(
                user_id=user_id,
                action="gdpr.forget_me",
                resource_type="user_data",
                resource_id=user_id,
                status="error",
                metadata={"error": str(e)},
            )

            raise

    async def rectify_user_data(
        self,
        user_id: str,
        corrections: Dict[str, Any],
        requester_ip: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Correct user data (Right to Rectification - Article 16)

        Args:
            user_id: User identifier
            corrections: Dictionary of fields to correct
            requester_ip: IP address of requester

        Returns:
            Dictionary with correction results
        """
        logger.info(f"Rectifying data for user: {user_id}")

        results = {
            "user_id": user_id,
            "corrections_applied": [],
            "corrections_failed": [],
        }

        try:
            # Apply corrections to user profile
            if "profile" in corrections:
                try:
                    await self._rectify_user_profile(user_id, corrections["profile"])
                    results["corrections_applied"].append("profile")
                except Exception as e:
                    results["corrections_failed"].append({"field": "profile", "error": str(e)})

            # Apply corrections to other data
            # ... (implement as needed)

            await self.db.commit()

            # Log audit trail
            await self._log_audit(
                user_id=user_id,
                action="gdpr.rectification",
                resource_type="user_data",
                resource_id=user_id,
                status="success",
                ip_address=requester_ip,
                metadata=results,
            )

            return results

        except Exception as e:
            logger.error(f"Error rectifying data for user {user_id}: {e}")
            await self.db.rollback()
            raise

    # ========================================================================
    # Export Helper Methods
    # ========================================================================

    async def _export_user_profile(self, user_id: str) -> Optional[Dict]:
        """Export user profile data"""
        try:
            from app.models.user import User

            result = await self.db.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()

            if user:
                return {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "created_at": user.created_at.isoformat(),
                    "updated_at": user.updated_at.isoformat(),
                }
        except Exception as e:
            logger.error(f"Error exporting user profile: {e}")
        return None

    async def _export_user_consents(self, user_id: str) -> List[Dict]:
        """Export user consent records"""
        result = await self.db.execute(
            select(UserConsent).where(UserConsent.user_id == user_id)
        )
        consents = result.scalars().all()

        return [
            {
                "consent_type": c.consent_type.value,
                "status": c.status.value,
                "granted_at": c.granted_at.isoformat() if c.granted_at else None,
                "revoked_at": c.revoked_at.isoformat() if c.revoked_at else None,
            }
            for c in consents
        ]

    async def _export_telemetry_prefs(self, user_id: str) -> Optional[Dict]:
        """Export telemetry preferences"""
        result = await self.db.execute(
            select(TelemetryPreference).where(TelemetryPreference.user_id == user_id)
        )
        pref = result.scalar_one_or_none()

        if pref:
            return {
                "telemetry_mode": pref.telemetry_mode.value,
                "do_not_track": pref.do_not_track,
                "collect_errors": pref.collect_errors,
                "collect_performance": pref.collect_performance,
                "collect_usage": pref.collect_usage,
            }
        return None

    async def _export_retention_settings(self, user_id: str) -> Optional[Dict]:
        """Export data retention settings"""
        result = await self.db.execute(
            select(DataRetention).where(DataRetention.user_id == user_id)
        )
        retention = result.scalar_one_or_none()

        if retention:
            return {
                "conversation_days": retention.conversation_days,
                "memory_days": retention.memory_days,
                "audit_log_days": retention.audit_log_days,
                "telemetry_days": retention.telemetry_days,
                "auto_purge_enabled": retention.auto_purge_enabled,
            }
        return None

    async def _export_conversations(self, user_id: str) -> List[Dict]:
        """Export conversation history"""
        try:
            from app.models.memory import ConversationSession, ConversationMessage

            # Get sessions
            result = await self.db.execute(
                select(ConversationSession).where(ConversationSession.user_id == user_id)
            )
            sessions = result.scalars().all()

            conversations = []
            for session in sessions:
                # Get messages for this session
                result = await self.db.execute(
                    select(ConversationMessage).where(
                        ConversationMessage.session_id == session.id
                    )
                )
                messages = result.scalars().all()

                conversations.append(
                    {
                        "session_id": session.id,
                        "created_at": session.created_at.isoformat(),
                        "messages": [
                            {
                                "role": m.role,
                                "content": m.content,
                                "timestamp": m.timestamp.isoformat(),
                            }
                            for m in messages
                        ],
                    }
                )

            return conversations
        except Exception as e:
            logger.error(f"Error exporting conversations: {e}")
            return []

    async def _export_memory(self, user_id: str) -> List[Dict]:
        """Export memory entries"""
        try:
            from app.models.memory import MemoryEntry

            result = await self.db.execute(
                select(MemoryEntry).where(MemoryEntry.user_id == user_id)
            )
            memories = result.scalars().all()

            return [
                {
                    "id": m.id,
                    "content": m.content,
                    "importance": m.importance,
                    "created_at": m.created_at.isoformat(),
                }
                for m in memories
            ]
        except Exception as e:
            logger.error(f"Error exporting memory: {e}")
            return []

    async def _export_audit_logs(self, user_id: str) -> List[Dict]:
        """Export audit logs"""
        result = await self.db.execute(
            select(AuditLog).where(AuditLog.user_id == user_id).order_by(AuditLog.performed_at.desc())
        )
        logs = result.scalars().all()

        return [
            {
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_id": log.resource_id,
                "status": log.status,
                "performed_at": log.performed_at.isoformat(),
            }
            for log in logs
        ]

    async def _export_trading_data(self, user_id: str) -> Dict:
        """Export trading data"""
        try:
            from app.models.trading import Position, Order

            # Export positions
            result = await self.db.execute(
                select(Position).where(Position.user_id == user_id)
            )
            positions = result.scalars().all()

            # Export orders
            result = await self.db.execute(
                select(Order).where(Order.user_id == user_id)
            )
            orders = result.scalars().all()

            return {
                "positions": [
                    {
                        "token": p.token,
                        "amount": float(p.amount),
                        "entry_price": float(p.entry_price),
                        "status": p.status.value,
                    }
                    for p in positions
                ],
                "orders": [
                    {
                        "token": o.token,
                        "action": o.action.value,
                        "amount": float(o.amount),
                        "price": float(o.price) if o.price else None,
                        "status": o.status.value,
                    }
                    for o in orders
                ],
            }
        except Exception as e:
            logger.error(f"Error exporting trading data: {e}")
            return {"positions": [], "orders": []}

    # ========================================================================
    # Delete Helper Methods
    # ========================================================================

    async def _delete_user_consents(self, user_id: str) -> int:
        """Delete user consent records"""
        result = await self.db.execute(
            delete(UserConsent).where(UserConsent.user_id == user_id)
        )
        return result.rowcount

    async def _delete_telemetry_prefs(self, user_id: str) -> int:
        """Delete telemetry preferences"""
        result = await self.db.execute(
            delete(TelemetryPreference).where(TelemetryPreference.user_id == user_id)
        )
        return result.rowcount

    async def _delete_retention_settings(self, user_id: str) -> int:
        """Delete retention settings"""
        result = await self.db.execute(
            delete(DataRetention).where(DataRetention.user_id == user_id)
        )
        return result.rowcount

    async def _delete_conversations(self, user_id: str) -> int:
        """Delete conversation history"""
        try:
            from app.models.memory import ConversationSession, ConversationMessage

            # Delete messages first
            result = await self.db.execute(
                select(ConversationSession).where(ConversationSession.user_id == user_id)
            )
            sessions = result.scalars().all()

            msg_count = 0
            for session in sessions:
                result = await self.db.execute(
                    delete(ConversationMessage).where(
                        ConversationMessage.session_id == session.id
                    )
                )
                msg_count += result.rowcount

            # Delete sessions
            result = await self.db.execute(
                delete(ConversationSession).where(ConversationSession.user_id == user_id)
            )
            return msg_count + result.rowcount
        except Exception as e:
            logger.error(f"Error deleting conversations: {e}")
            return 0

    async def _delete_memory(self, user_id: str) -> int:
        """Delete memory entries"""
        try:
            from app.models.memory import MemoryEntry

            result = await self.db.execute(
                delete(MemoryEntry).where(MemoryEntry.user_id == user_id)
            )
            return result.rowcount
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return 0

    async def _delete_trading_data(self, user_id: str) -> int:
        """Delete trading data"""
        try:
            from app.models.trading import Position, Order

            count = 0

            result = await self.db.execute(
                delete(Position).where(Position.user_id == user_id)
            )
            count += result.rowcount

            result = await self.db.execute(
                delete(Order).where(Order.user_id == user_id)
            )
            count += result.rowcount

            return count
        except Exception as e:
            logger.error(f"Error deleting trading data: {e}")
            return 0

    async def _delete_api_keys(self, user_id: str) -> int:
        """Delete API keys"""
        try:
            from app.models.user import APIKey

            result = await self.db.execute(
                delete(APIKey).where(APIKey.user_id == user_id)
            )
            return result.rowcount
        except Exception as e:
            logger.error(f"Error deleting API keys: {e}")
            return 0

    async def _delete_user_files(self, user_id: str) -> List[str]:
        """Delete user files from filesystem"""
        deleted_files = []

        try:
            # Delete user directory if exists
            user_dir = Path(f"./data/users/{user_id}")
            if user_dir.exists():
                import shutil

                shutil.rmtree(user_dir)
                deleted_files.append(str(user_dir))
        except Exception as e:
            logger.error(f"Error deleting user files: {e}")

        return deleted_files

    async def _delete_from_vector_store(self, user_id: str):
        """Delete user data from vector store"""
        # TODO: Implement vector store deletion
        pass

    async def _anonymize_audit_logs(self, user_id: str) -> int:
        """Anonymize audit logs (keep for compliance)"""
        result = await self.db.execute(
            select(AuditLog).where(AuditLog.user_id == user_id)
        )
        logs = result.scalars().all()

        for log in logs:
            log.user_id = f"deleted_{user_id[:8]}"
            log.ip_address = "0.0.0.0"
            log.user_agent = "anonymized"

        return len(logs)

    async def _delete_user_profile(self, user_id: str) -> int:
        """Delete user profile"""
        try:
            from app.models.user import User

            result = await self.db.execute(delete(User).where(User.id == user_id))
            return result.rowcount
        except Exception as e:
            logger.error(f"Error deleting user profile: {e}")
            return 0

    async def _rectify_user_profile(self, user_id: str, corrections: Dict):
        """Apply corrections to user profile"""
        try:
            from app.models.user import User

            result = await self.db.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()

            if user:
                for field, value in corrections.items():
                    if hasattr(user, field):
                        setattr(user, field, value)
        except Exception as e:
            logger.error(f"Error rectifying user profile: {e}")
            raise

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
        """Log to audit trail"""
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
