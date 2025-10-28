"""
Session Storage in Redis
Manages user sessions with Redis backend, session expiration, and concurrent session limits
"""

import json
import logging
import secrets
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

import redis.asyncio as aioredis
from prometheus_client import Counter, Gauge

from app.cache import make_cache_key, redis_operation_duration


logger = logging.getLogger(__name__)


# ============================================================================
# Prometheus Metrics
# ============================================================================

session_operations = Counter(
    "session_operations_total",
    "Total number of session operations",
    ["operation", "status"]
)

active_sessions = Gauge(
    "active_sessions_total",
    "Total number of active sessions"
)

session_creation_time = Counter(
    "session_creation_time_seconds",
    "Session creation timestamp"
)


# ============================================================================
# Session Manager
# ============================================================================

class SessionManager:
    """
    Redis-backed session management

    Features:
    - Session storage in Redis (moved from in-memory)
    - Automatic session expiration
    - Session refresh mechanism
    - Session revocation (logout)
    - Concurrent session limits per user
    - Session metadata tracking (IP, user agent, last activity)
    """

    # Default settings
    DEFAULT_SESSION_TTL = 86400  # 24 hours
    DEFAULT_MAX_SESSIONS_PER_USER = 5  # Maximum concurrent sessions
    CACHE_VERSION = 1

    def __init__(
        self,
        redis: Optional[aioredis.Redis] = None,
        session_ttl: int = DEFAULT_SESSION_TTL,
        max_sessions_per_user: int = DEFAULT_MAX_SESSIONS_PER_USER
    ):
        self.redis = redis
        self.session_ttl = session_ttl
        self.max_sessions_per_user = max_sessions_per_user

    # ========================================================================
    # Session Creation
    # ========================================================================

    async def create_session(
        self,
        user_id: str,
        user_data: Dict[str, Any],
        ip_address: str,
        user_agent: str,
        ttl: Optional[int] = None
    ) -> Optional[str]:
        """
        Create a new session

        Args:
            user_id: User ID
            user_data: User data to store in session
            ip_address: Client IP address
            user_agent: Client user agent
            ttl: Custom TTL in seconds (default: DEFAULT_SESSION_TTL)

        Returns:
            Session token or None if failed
        """
        if not self.redis:
            logger.warning("Redis unavailable, cannot create session")
            return None

        # Generate secure session token
        session_token = secrets.token_urlsafe(32)
        ttl = ttl or self.session_ttl

        # Check concurrent session limit
        existing_sessions = await self.get_user_sessions(user_id)
        if len(existing_sessions) >= self.max_sessions_per_user:
            # Remove oldest session
            oldest_session = min(existing_sessions, key=lambda s: s.get("created_at", ""))
            await self.revoke_session(oldest_session["session_token"])
            logger.info(
                f"Removed oldest session for user {user_id} due to session limit"
            )

        # Create session data
        session_data = {
            "session_token": session_token,
            "user_id": user_id,
            "user_data": user_data,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(seconds=ttl)).isoformat(),
        }

        try:
            # Store session
            session_key = make_cache_key("session", "token", session_token, f"v{self.CACHE_VERSION}")
            with redis_operation_duration.labels(operation="set").time():
                await self.redis.setex(
                    session_key,
                    ttl,
                    json.dumps(session_data)
                )

            # Add to user's session index
            user_sessions_key = make_cache_key("session", "user", user_id, f"v{self.CACHE_VERSION}")
            await self.redis.sadd(user_sessions_key, session_token)
            await self.redis.expire(user_sessions_key, ttl)

            session_operations.labels(operation="create", status="success").inc()
            session_creation_time.inc()
            logger.info(f"Created session for user {user_id} from IP {ip_address}")

            return session_token

        except Exception as e:
            logger.error(f"Error creating session: {e}")
            session_operations.labels(operation="create", status="error").inc()
            return None

    # ========================================================================
    # Session Retrieval
    # ========================================================================

    async def get_session(
        self,
        session_token: str,
        update_activity: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get session data by token

        Args:
            session_token: Session token
            update_activity: Update last activity timestamp

        Returns:
            Session data or None if not found
        """
        if not self.redis:
            return None

        session_key = make_cache_key("session", "token", session_token, f"v{self.CACHE_VERSION}")

        try:
            with redis_operation_duration.labels(operation="get").time():
                cached = await self.redis.get(session_key)

            if not cached:
                session_operations.labels(operation="get", status="not_found").inc()
                return None

            session_data = json.loads(cached)

            # Update last activity if requested
            if update_activity:
                session_data["last_activity"] = datetime.utcnow().isoformat()
                await self.redis.setex(
                    session_key,
                    self.session_ttl,
                    json.dumps(session_data)
                )

            session_operations.labels(operation="get", status="success").inc()
            return session_data

        except Exception as e:
            logger.error(f"Error getting session: {e}")
            session_operations.labels(operation="get", status="error").inc()
            return None

    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all active sessions for a user

        Args:
            user_id: User ID

        Returns:
            List of session data
        """
        if not self.redis:
            return []

        user_sessions_key = make_cache_key("session", "user", user_id, f"v{self.CACHE_VERSION}")

        try:
            # Get all session tokens for user
            session_tokens = await self.redis.smembers(user_sessions_key)

            # Get session data for each token
            sessions = []
            for token in session_tokens:
                session_data = await self.get_session(token, update_activity=False)
                if session_data:
                    sessions.append(session_data)
                else:
                    # Remove invalid token from index
                    await self.redis.srem(user_sessions_key, token)

            return sessions

        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []

    # ========================================================================
    # Session Refresh
    # ========================================================================

    async def refresh_session(
        self,
        session_token: str,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Refresh session expiration

        Args:
            session_token: Session token
            ttl: New TTL in seconds (default: DEFAULT_SESSION_TTL)

        Returns:
            True if refreshed successfully
        """
        if not self.redis:
            return False

        session_data = await self.get_session(session_token, update_activity=False)
        if not session_data:
            return False

        ttl = ttl or self.session_ttl
        session_key = make_cache_key("session", "token", session_token, f"v{self.CACHE_VERSION}")

        try:
            # Update session data
            session_data["last_activity"] = datetime.utcnow().isoformat()
            session_data["expires_at"] = (datetime.utcnow() + timedelta(seconds=ttl)).isoformat()

            # Refresh TTL
            await self.redis.setex(
                session_key,
                ttl,
                json.dumps(session_data)
            )

            session_operations.labels(operation="refresh", status="success").inc()
            return True

        except Exception as e:
            logger.error(f"Error refreshing session: {e}")
            session_operations.labels(operation="refresh", status="error").inc()
            return False

    # ========================================================================
    # Session Revocation
    # ========================================================================

    async def revoke_session(self, session_token: str) -> bool:
        """
        Revoke (logout) a session

        Args:
            session_token: Session token

        Returns:
            True if revoked successfully
        """
        if not self.redis:
            return False

        # Get session data to find user
        session_data = await self.get_session(session_token, update_activity=False)
        if not session_data:
            return False

        user_id = session_data["user_id"]
        session_key = make_cache_key("session", "token", session_token, f"v{self.CACHE_VERSION}")
        user_sessions_key = make_cache_key("session", "user", user_id, f"v{self.CACHE_VERSION}")

        try:
            # Delete session
            await self.redis.delete(session_key)

            # Remove from user's session index
            await self.redis.srem(user_sessions_key, session_token)

            session_operations.labels(operation="revoke", status="success").inc()
            logger.info(f"Revoked session for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error revoking session: {e}")
            session_operations.labels(operation="revoke", status="error").inc()
            return False

    async def revoke_all_user_sessions(self, user_id: str) -> int:
        """
        Revoke all sessions for a user (e.g., on password change)

        Args:
            user_id: User ID

        Returns:
            Number of sessions revoked
        """
        if not self.redis:
            return 0

        sessions = await self.get_user_sessions(user_id)
        revoked_count = 0

        for session in sessions:
            if await self.revoke_session(session["session_token"]):
                revoked_count += 1

        logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
        return revoked_count

    # ========================================================================
    # Session Validation
    # ========================================================================

    async def validate_session(
        self,
        session_token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate session and optionally check IP/user agent

        Args:
            session_token: Session token
            ip_address: Expected IP address (optional)
            user_agent: Expected user agent (optional)

        Returns:
            Tuple of (valid, session_data)
        """
        session_data = await self.get_session(session_token)

        if not session_data:
            return False, None

        # Check IP address if provided
        if ip_address and session_data.get("ip_address") != ip_address:
            logger.warning(
                f"Session IP mismatch: expected {session_data.get('ip_address')}, got {ip_address}"
            )
            # Don't fail, just log (IP can change for mobile users)

        # Check user agent if provided
        if user_agent and session_data.get("user_agent") != user_agent:
            logger.warning(
                f"Session user agent mismatch for user {session_data.get('user_id')}"
            )
            # Don't fail, just log

        return True, session_data

    # ========================================================================
    # Session Statistics
    # ========================================================================

    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics

        Returns:
            Dictionary with session stats
        """
        if not self.redis:
            return {"status": "unavailable"}

        try:
            # Count all sessions
            pattern = make_cache_key("session", "token", "*")
            session_count = 0
            unique_users = set()

            async for key in self.redis.scan_iter(match=pattern):
                session_count += 1
                cached = await self.redis.get(key)
                if cached:
                    session_data = json.loads(cached)
                    unique_users.add(session_data.get("user_id"))

            # Update gauge
            active_sessions.set(session_count)

            return {
                "total_sessions": session_count,
                "unique_users": len(unique_users),
                "max_sessions_per_user": self.max_sessions_per_user,
                "default_ttl": self.session_ttl,
            }

        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {"status": "error", "error": str(e)}

    # ========================================================================
    # Session Cleanup
    # ========================================================================

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions (Redis handles this automatically, but this
        is useful for cleaning up user session indexes)

        Returns:
            Number of cleaned up sessions
        """
        if not self.redis:
            return 0

        cleaned = 0

        try:
            # Find all user session indexes
            user_pattern = make_cache_key("session", "user", "*")

            async for key in self.redis.scan_iter(match=user_pattern):
                # Get all session tokens for this user
                session_tokens = await self.redis.smembers(key)

                for token in session_tokens:
                    # Check if session still exists
                    session_key = make_cache_key("session", "token", token, f"v{self.CACHE_VERSION}")
                    exists = await self.redis.exists(session_key)

                    if not exists:
                        # Remove from user index
                        await self.redis.srem(key, token)
                        cleaned += 1

            logger.info(f"Cleaned up {cleaned} expired session references")
            return cleaned

        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
            return 0


# ============================================================================
# Session Middleware Helper
# ============================================================================

class SessionMiddleware:
    """
    Helper class for session middleware integration

    Usage in FastAPI:
        session_manager = SessionManager(redis)
        app.add_middleware(SessionMiddleware, session_manager=session_manager)
    """

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    async def get_session_from_request(self, request) -> Optional[Dict[str, Any]]:
        """
        Extract and validate session from request

        Args:
            request: FastAPI request

        Returns:
            Session data or None
        """
        # Get session token from header or cookie
        token = request.headers.get("X-Session-Token")
        if not token:
            token = request.cookies.get("session_token")

        if not token:
            return None

        # Validate session
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("User-Agent")

        valid, session_data = await self.session_manager.validate_session(
            token, ip_address, user_agent
        )

        if valid:
            return session_data

        return None
