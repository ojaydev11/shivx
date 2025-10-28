"""
Configuration dependency
Provides access to application settings
"""

from fastapi import Depends
from config.settings import Settings, get_settings as _get_settings


def get_settings() -> Settings:
    """Get application settings"""
    return _get_settings()
