"""
User and Authentication Models
"""

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Boolean, DateTime
from datetime import datetime
from typing import Optional

from app.models.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    """User model"""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Profile
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Timestamps
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class APIKey(Base, TimestampMixin):
    """API Key model for authentication"""
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(primary_key=True)
    key_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    user_id: Mapped[Optional[int]] = mapped_column(nullable=True, index=True)

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Permissions (JSON would be better, but keeping simple for now)
    permissions: Mapped[str] = mapped_column(String(500), nullable=False, default="READ")

    # Usage tracking
    last_used: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    usage_count: Mapped[int] = mapped_column(nullable=False, default=0)

    # Expiration
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
