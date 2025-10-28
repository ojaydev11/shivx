"""
Authentication API Router
Endpoints for authentication, API key management, and session handling
"""

from typing import List, Optional
from datetime import datetime, timezone
import hashlib
import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.dependencies import get_current_user, require_permission
from app.dependencies.auth import TokenData
from app.dependencies.database import get_db
from app.models.user import APIKey, User
from core.security.hardening import Permission
import logging

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/auth",
    tags=["auth"],
)


# ============================================================================
# Pydantic Models
# ============================================================================

class APIKeyCreate(BaseModel):
    """API key creation request"""
    name: str = Field(..., description="Friendly name for the API key")
    permissions: dict = Field(default_factory=dict, description="Permissions for this key")
    expires_in_days: Optional[int] = Field(None, description="Expiration in days (None = never)")


class APIKeyResponse(BaseModel):
    """API key response"""
    key_id: str
    name: str
    key: Optional[str] = None  # Only returned on creation
    permissions: dict
    is_active: bool
    expires_at: Optional[datetime]
    created_at: datetime
    last_used_at: Optional[datetime]

    class Config:
        from_attributes = True


class APIKeyList(BaseModel):
    """List of API keys"""
    keys: List[APIKeyResponse]
    total: int


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/keys", response_model=APIKeyList)
async def list_api_keys(
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all API keys for the current user

    Requires: Authentication
    """
    # Get user from database
    user = db.query(User).filter(User.user_id == current_user.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Get all API keys for this user
    api_keys = db.query(APIKey).filter(APIKey.user_id == user.user_id).all()

    return APIKeyList(
        keys=[
            APIKeyResponse(
                key_id=str(key.key_id),
                name=key.name,
                permissions=key.permissions,
                is_active=key.is_active,
                expires_at=key.expires_at,
                created_at=key.created_at,
                last_used_at=key.last_used_at
            )
            for key in api_keys
        ],
        total=len(api_keys)
    )


@router.post("/keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: APIKeyCreate,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new API key

    Requires: Authentication

    Returns:
        APIKeyResponse with the generated key (ONLY shown once!)
    """
    # Get user from database
    user = db.query(User).filter(User.user_id == current_user.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Generate secure API key
    raw_key = f"shivx_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    # Calculate expiration
    expires_at = None
    if request.expires_in_days:
        from datetime import timedelta
        expires_at = datetime.now(timezone.utc) + timedelta(days=request.expires_in_days)

    # Create API key
    api_key = APIKey(
        key_hash=key_hash,
        name=request.name,
        permissions=request.permissions,
        user_id=user.user_id,
        is_active=True,
        expires_at=expires_at
    )

    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    logger.info(f"API key created: {api_key.key_id} for user {user.user_id}")

    return APIKeyResponse(
        key_id=str(api_key.key_id),
        name=api_key.name,
        key=raw_key,  # ONLY shown on creation!
        permissions=api_key.permissions,
        is_active=api_key.is_active,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
        last_used_at=api_key.last_used_at
    )


@router.post("/keys/{key_id}/revoke")
async def revoke_api_key(
    key_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Revoke an API key

    Requires: Authentication (key owner or admin)

    Security:
        - Users can only revoke their own keys
        - Admins can revoke any key
    """
    # Get the API key
    api_key = db.query(APIKey).filter(APIKey.key_id == key_id).first()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    # Check authorization
    if str(api_key.user_id) != current_user.user_id and Permission.ADMIN not in current_user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to revoke this key"
        )

    # Revoke the key (soft delete by marking inactive)
    api_key.is_active = False
    db.commit()

    logger.info(f"API key revoked: {key_id} by user {current_user.user_id}")

    return {
        "key_id": key_id,
        "status": "revoked",
        "revoked_at": datetime.now(timezone.utc)
    }


@router.delete("/keys/{key_id}")
async def delete_api_key(
    key_id: str,
    current_user: TokenData = Depends(require_permission(Permission.ADMIN)),
    db: Session = Depends(get_db)
):
    """
    Permanently delete an API key

    Requires: ADMIN permission

    Warning: This is permanent! Use revoke for soft delete.
    """
    # Get the API key
    api_key = db.query(APIKey).filter(APIKey.key_id == key_id).first()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    # Delete the key
    db.delete(api_key)
    db.commit()

    logger.warning(f"API key permanently deleted: {key_id} by admin {current_user.user_id}")

    return {
        "key_id": key_id,
        "status": "deleted",
        "deleted_at": datetime.now(timezone.utc)
    }


@router.get("/keys/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get details of a specific API key

    Requires: Authentication (key owner or admin)
    """
    # Get the API key
    api_key = db.query(APIKey).filter(APIKey.key_id == key_id).first()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    # Check authorization
    if str(api_key.user_id) != current_user.user_id and Permission.ADMIN not in current_user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this key"
        )

    return APIKeyResponse(
        key_id=str(api_key.key_id),
        name=api_key.name,
        key=None,  # Never return the actual key after creation
        permissions=api_key.permissions,
        is_active=api_key.is_active,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
        last_used_at=api_key.last_used_at
    )
