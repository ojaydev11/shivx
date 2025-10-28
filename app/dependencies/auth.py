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
    Get current authenticated user from JWT token

    Args:
        credentials: HTTP Bearer token
        settings: Application settings

    Returns:
        TokenData with user info

    Raises:
        HTTPException: If not authenticated
    """
    # Check if authentication is disabled (dev only)
    if settings.skip_auth:
        # Return mock user for development
        return TokenData(
            user_id="dev_user",
            permissions={Permission.ADMIN}  # Full access in dev mode
        )

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

    # TODO: Validate API key against database
    # For now, just return it
    return x_api_key
