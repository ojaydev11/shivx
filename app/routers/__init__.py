"""
API Routers
Organized by domain: trading, analytics, AI/ML
"""

from .trading import router as trading_router
from .analytics import router as analytics_router
from .ai import router as ai_router

__all__ = ["trading_router", "analytics_router", "ai_router"]
