"""
ShivX Integrations Package
===========================
Safe external service integrations with comprehensive security controls.

Integrations:
- GitHub Operations (read/write with approval)
- Gmail/Calendar (OAuth2 with scopes)
- Telegram Bot (command handling and notifications)
- Browser Automation (sandboxed with Playwright)
- LLM Bridges (Claude, ChatGPT with safety filters)

All integrations:
- Route through Guardian Defense for safety checks
- Log all operations to Audit Chain
- Require appropriate permissions from Policy Guard
- Implement rate limiting and resource controls
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Version info
__version__ = "1.0.0"
__author__ = "ShivX Team"

# Integration status tracking
_integration_status = {
    "github": {"enabled": False, "authenticated": False},
    "google": {"enabled": False, "authenticated": False},
    "telegram": {"enabled": False, "authenticated": False},
    "browser": {"enabled": False, "ready": False},
    "llm": {"enabled": False, "providers": []},
}


def get_integration_status() -> dict:
    """Get status of all integrations"""
    return _integration_status.copy()


def set_integration_status(integration: str, status: dict) -> None:
    """Update integration status"""
    if integration in _integration_status:
        _integration_status[integration].update(status)
        logger.info(f"Integration status updated: {integration} -> {status}")


# Import integration clients (lazy loading to avoid import errors)
def get_github_client():
    """Lazy load GitHub client"""
    from integrations.github_client import GitHubClient
    return GitHubClient


def get_google_client():
    """Lazy load Google client"""
    from integrations.google_client import GoogleClient
    return GoogleClient


def get_telegram_bot():
    """Lazy load Telegram bot"""
    from integrations.telegram_bot import TelegramBot
    return TelegramBot


def get_browser_automation():
    """Lazy load browser automation"""
    from integrations.browser_automation import BrowserAutomation
    return BrowserAutomation


def get_llm_client():
    """Lazy load LLM client"""
    from integrations.llm_client import LLMClient
    return LLMClient


__all__ = [
    "get_github_client",
    "get_google_client",
    "get_telegram_bot",
    "get_browser_automation",
    "get_llm_client",
    "get_integration_status",
    "set_integration_status",
]
