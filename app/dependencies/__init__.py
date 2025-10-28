"""
Dependency Injection for FastAPI
Provides reusable dependencies for routes
"""

from .auth import get_current_user, require_permission, get_api_key
from .database import get_db
from .config import get_settings

__all__ = [
    "get_current_user",
    "require_permission",
    "get_api_key",
    "get_db",
    "get_settings",
]
