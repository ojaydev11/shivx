"""
Authentication Dependencies
JWT-based authentication for API endpoints
"""

import os
from typing import Optional
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from config.settings import Settings, get_settings
from core.security.hardening import Permission

security = HTTPBearer(auto_error=False)


class TokenData:
    """JWT token data"""
    def __init__(self, user_id: str, permissions: set):
        self.user_id = user_id
        self.permissions = permissions


def create_access_token(user_id: str, permissions: set, settings: Settings) -> str:
    """
    Create JWT access token

    Args:
        user_id: User identifier
        permissions: Set of Permission enums
        settings: Application settings

    Returns:
        JWT token string
    """
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expiration_minutes)

    to_encode = {
        "sub": user_id,
        "permissions": [p.value for p in permissions],
        "exp": expire,
        "iat": datetime.utcnow()
    }

    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm
    )

    return encoded_jwt


def decode_access_token(token: str, settings: Settings) -> TokenData:
    """
    Decode and validate JWT token

    Args:
        token: JWT token string
        settings: Application settings

    Returns:
        TokenData with user_id and permissions

    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm]
        )

        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception

        permissions_str = payload.get("permissions", [])
        permissions = {Permission(p) for p in permissions_str}

        return TokenData(user_id=user_id, permissions=permissions)

    except JWTError:
        raise credentials_exception


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    settings: Settings = Depends(get_settings)
) -> TokenData:
    """
    Get current authenticated user from JWT token.

    CRITICAL SECURITY: This function enforces authentication for all protected endpoints.
    The skip_auth bypass is ONLY available in local/development environments and is
    automatically blocked in production/staging by the Settings validator.

    Args:
        credentials: HTTP Bearer token from Authorization header
        settings: Application settings (with security validation)

    Returns:
        TokenData with user_id and permissions

    Raises:
        HTTPException: If not authenticated or token is invalid

    Security Notes:
        - skip_auth bypass is logged with WARNING level
        - skip_auth is blocked in production/staging by Settings validator
        - Even if skip_auth is somehow True, we check environment again
        - All authentication events should be audited
    """
    import logging

    logger = logging.getLogger(__name__)

    # CRITICAL SECURITY: Check if authentication bypass is enabled
    # This should NEVER be True in production (blocked by Settings validator)
    if settings.skip_auth:
        # Double-check environment as defense in depth
        # This should never trigger because Settings validator blocks it,
        # but we add this as an additional safety layer
        if settings.env in ("production", "staging"):
            logger.critical(
                "SECURITY VIOLATION: skip_auth is True in %s environment! "
                "This should be impossible due to Settings validator. "
                "Possible configuration override or validator bypass.",
                settings.env
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Security configuration error - authentication required",
            )

        # Log warning about authentication bypass (even in dev)
        logger.warning(
            "AUTHENTICATION BYPASS: skip_auth is enabled. "
            "All requests granted ADMIN access without authentication. "
            "Environment: %s. This is ONLY safe for local development.",
            settings.env
        )

        # Return mock user for development
        return TokenData(
            user_id="dev_user",
            permissions={Permission.ADMIN}  # Full access in dev mode
        )

    # Normal authentication flow - require credentials
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return decode_access_token(credentials.credentials, settings)


def require_permission(*required_permissions: Permission):
    """
    Dependency factory to require specific permissions

    Usage:
        @app.get("/admin", dependencies=[Depends(require_permission(Permission.ADMIN))])
        async def admin_endpoint():
            return {"message": "Admin access"}

    Args:
        *required_permissions: Permissions required for access

    Returns:
        Dependency function
    """
    async def permission_checker(
        current_user: TokenData = Depends(get_current_user)
    ):
        # Admin has all permissions
        if Permission.ADMIN in current_user.permissions:
            return current_user

        # Check if user has all required permissions
        missing_permissions = set(required_permissions) - current_user.permissions

        if missing_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permissions: {[p.value for p in missing_permissions]}"
            )

        return current_user

    return permission_checker


async def get_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings)
) -> Optional[str]:
    """
    Get API key from header

    Args:
        x_api_key: API key from X-API-Key header
        settings: Application settings

    Returns:
        API key if provided

    Raises:
        HTTPException: If API key is invalid
    """
    if x_api_key is None:
        return None

    # Import here to avoid circular dependency
    from app.database import get_db
    from app.models.user import APIKey
    from sqlalchemy import select
    import hashlib
    import logging

    logger = logging.getLogger(__name__)

    # Validate API key against database
    validated_key = await validate_api_key_against_db(x_api_key, settings)
    if validated_key is None:
        logger.warning(f"Invalid API key attempted: {x_api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return validated_key


async def validate_api_key_against_db(api_key: str, settings: Settings) -> Optional[str]:
    """
    Validate API key against database

    Args:
        api_key: API key to validate
        settings: Application settings

    Returns:
        API key if valid, None otherwise
    """
    import hashlib
    from datetime import datetime, timezone
    from app.database import get_db
    from app.models.user import APIKey, User
    from sqlalchemy import select
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Hash the API key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Query database for matching key
        async for db in get_db():
            result = await db.execute(
                select(APIKey, User)
                .join(User, APIKey.user_id == User.user_id)
                .where(APIKey.key_hash == key_hash)
            )
            row = result.first()

            if row is None:
                logger.warning(f"API key not found in database: {api_key[:8]}...")
                return None

            api_key_obj, user = row

            # Check if key is active
            if not api_key_obj.is_active:
                logger.warning(f"API key is inactive: {api_key_obj.key_id}")
                return None

            # Check if key is expired
            if api_key_obj.expires_at and datetime.now(timezone.utc) > api_key_obj.expires_at:
                logger.warning(f"API key is expired: {api_key_obj.key_id}")
                return None

            # Check if user is active
            if not user.is_active:
                logger.warning(f"User is inactive for API key: {user.user_id}")
                return None

            # Update last_used_at
            api_key_obj.last_used_at = datetime.now(timezone.utc)
            await db.commit()

            logger.info(f"API key validated successfully: {api_key_obj.key_id}")
            return api_key

    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        return None

    return None
