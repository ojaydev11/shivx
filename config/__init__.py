"""
Configuration Management for ShivX
Environment-specific configurations using Pydantic Settings
"""

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
