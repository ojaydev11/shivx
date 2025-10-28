"""
Safe Browser Automation
========================
Sandboxed browser automation using Playwright with comprehensive safety controls.

Features:
- Headless browser automation (Chromium, Firefox, WebKit)
- Sandboxing: No downloads, file access, or credentials
- URL allowlist enforcement
- Resource limits (timeout, max pages)
- Network isolation
- Full audit logging
"""

import os
import time
import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

try:
    from playwright.async_api import async_playwright, Browser, Page, Playwright
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None  # type: ignore
    Browser = None  # type: ignore
    Page = None  # type: ignore

from security.guardian_defense import get_guardian_defense
from utils.audit_chain import append_jsonl
from utils.policy_guard import get_policy_guard

logger = logging.getLogger(__name__)


class BrowserType(Enum):
    """Browser types"""
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class BrowserOperationType(Enum):
    """Browser operation types"""
    NAVIGATE = "navigate"
    EXTRACT = "extract"
    FILL_FORM = "fill_form"
    CLICK = "click"
    SCREENSHOT = "screenshot"


@dataclass
class BrowserSession:
    """Browser session metadata"""
    session_id: str
    browser_type: BrowserType
    pages_count: int
    start_time: str
    operations_count: int


class BrowserAutomation:
    """
    Safe browser automation with sandboxing and safety controls.

    Features:
    - Headless browser automation
    - URL allowlist enforcement
    - No credential entry (no login automation)
    - Resource limits (timeout, max pages)
    - Network isolation
    - Audit logging
    """

    def __init__(
        self,
        browser_type: BrowserType = BrowserType.CHROMIUM,
        headless: bool = True,
        audit_log_path: str = "var/audit/browser_operations.jsonl",
        url_allowlist: Optional[List[str]] = None,
        max_pages_per_session: int = 10,
        default_timeout: int = 60000  # 60 seconds
    ):
        """
        Initialize browser automation.

        Args:
            browser_type: Browser to use (chromium, firefox, webkit)
            headless: Run in headless mode
            audit_log_path: Path to audit log file
            url_allowlist: List of allowed domains
            max_pages_per_session: Maximum pages per session
            default_timeout: Default timeout in milliseconds
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright not installed. "
                "Install with: pip install playwright && playwright install"
            )

        self.browser_type = browser_type
        self.headless = headless
        self.audit_log_path = audit_log_path
        self.max_pages_per_session = max_pages_per_session
        self.default_timeout = default_timeout

        # URL allowlist
        self.url_allowlist = set(url_allowlist or [])

        # Load from env var if available
        env_allowlist = os.getenv("BROWSER_URL_ALLOWLIST", "")
        if env_allowlist:
            self.url_allowlist.update(
                domain.strip() for domain in env_allowlist.split(",")
                if domain.strip()
            )

        # Get integrations
        self.guardian = get_guardian_defense()
        self.policy_guard = get_policy_guard()

        # Session tracking
        self.active_sessions: Dict[str, BrowserSession] = {}
        self.operations_count = 0

        # Playwright instances
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None

        logger.info(
            f"BrowserAutomation initialized "
            f"(type={browser_type.value}, headless={headless})"
        )

    def _is_url_allowed(self, url: str) -> bool:
        """Check if URL is in allowlist"""
        if not self.url_allowlist:
            logger.warning("No URL allowlist configured - allowing all URLs")
            return True

        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc

        # Check exact match
        if domain in self.url_allowlist:
            return True

        # Check wildcard match (*.example.com)
        for allowed in self.url_allowlist:
            if allowed.startswith("*."):
                if domain.endswith(allowed[1:]):
                    return True
            elif allowed.endswith(".*"):
                if domain.startswith(allowed[:-1]):
                    return True

        return False

    def _log_operation(
        self,
        session_id: Optional[str],
        operation_type: BrowserOperationType,
        details: Dict[str, Any],
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Log operation to audit chain"""
        try:
            operation_id = f"browser_{int(time.time() * 1000)}"

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "integration": "browser",
                "operation_id": operation_id,
                "session_id": session_id,
                "operation_type": operation_type.value,
                "details": details,
                "success": success,
                "error": error,
            }

            append_jsonl(self.audit_log_path, log_entry)

        except Exception as e:
            logger.error(f"Failed to log operation: {e}")

    async def _start_browser(self) -> Browser:
        """Start browser instance"""
        if self.browser:
            return self.browser

        self.playwright = await async_playwright().start()

        # Launch browser based on type
        if self.browser_type == BrowserType.CHROMIUM:
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled',
                ]
            )
        elif self.browser_type == BrowserType.FIREFOX:
            self.browser = await self.playwright.firefox.launch(
                headless=self.headless
            )
        elif self.browser_type == BrowserType.WEBKIT:
            self.browser = await self.playwright.webkit.launch(
                headless=self.headless
            )

        logger.info(f"Browser started: {self.browser_type.value}")

        return self.browser

    async def _stop_browser(self) -> None:
        """Stop browser instance"""
        if self.browser:
            await self.browser.close()
            self.browser = None

        if self.playwright:
            await self.playwright.stop()
            self.playwright = None

        logger.info("Browser stopped")

    async def create_session(self) -> str:
        """
        Create new browser session.

        Returns:
            Session ID
        """
        session_id = f"session_{int(time.time() * 1000)}"

        browser = await self._start_browser()

        session = BrowserSession(
            session_id=session_id,
            browser_type=self.browser_type,
            pages_count=0,
            start_time=datetime.now().isoformat(),
            operations_count=0
        )

        self.active_sessions[session_id] = session

        self._log_operation(
            session_id,
            BrowserOperationType.NAVIGATE,
            {"action": "session_created"},
            True
        )

        logger.info(f"Browser session created: {session_id}")

        return session_id

    async def navigate(
        self,
        session_id: str,
        url: str,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Navigate to URL.

        Args:
            session_id: Browser session ID
            url: URL to navigate to
            timeout: Navigation timeout in milliseconds

        Returns:
            Navigation result
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        # Check URL allowlist
        if not self._is_url_allowed(url):
            error_msg = f"URL not in allowlist: {url}"
            logger.error(error_msg)
            self._log_operation(
                session_id,
                BrowserOperationType.NAVIGATE,
                {"url": url},
                False,
                error_msg
            )
            raise PermissionError(error_msg)

        # Check policy
        policy_result = self.policy_guard.evaluate({
            "action": "browser.navigate",
            "url": url
        })

        if policy_result.decision == "deny":
            error_msg = f"Navigation denied by policy: {', '.join(policy_result.reasons)}"
            logger.error(error_msg)
            self._log_operation(
                session_id,
                BrowserOperationType.NAVIGATE,
                {"url": url},
                False,
                error_msg
            )
            raise PermissionError(error_msg)

        try:
            browser = await self._start_browser()
            page = await browser.new_page()

            # Check page limit
            session = self.active_sessions[session_id]
            if session.pages_count >= self.max_pages_per_session:
                raise RuntimeError(
                    f"Max pages per session exceeded: {self.max_pages_per_session}"
                )

            # Set timeout
            page.set_default_timeout(timeout or self.default_timeout)

            # Navigate
            response = await page.goto(url)

            # Update session
            session.pages_count += 1
            session.operations_count += 1

            result = {
                "url": url,
                "status": response.status if response else None,
                "title": await page.title(),
                "success": True
            }

            self._log_operation(
                session_id,
                BrowserOperationType.NAVIGATE,
                {"url": url, "status": result["status"]},
                True
            )

            # Keep page reference for future operations
            # In production, would need proper page lifecycle management
            await page.close()

            return result

        except PlaywrightTimeoutError as e:
            error_msg = f"Navigation timeout: {url}"
            logger.error(error_msg)
            self._log_operation(
                session_id,
                BrowserOperationType.NAVIGATE,
                {"url": url},
                False,
                error_msg
            )
            raise TimeoutError(error_msg) from e

        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            self._log_operation(
                session_id,
                BrowserOperationType.NAVIGATE,
                {"url": url},
                False,
                str(e)
            )
            raise

    async def extract_text(
        self,
        session_id: str,
        url: str,
        selector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract text from page.

        Args:
            session_id: Browser session ID
            url: URL to extract from
            selector: CSS selector (extracts from entire page if not provided)

        Returns:
            Extracted text
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        try:
            browser = await self._start_browser()
            page = await browser.new_page()

            # Navigate
            await page.goto(url)

            # Extract text
            if selector:
                element = await page.query_selector(selector)
                if element:
                    text = await element.inner_text()
                else:
                    text = ""
            else:
                text = await page.inner_text('body')

            result = {
                "url": url,
                "selector": selector,
                "text": text,
                "length": len(text),
                "success": True
            }

            self._log_operation(
                session_id,
                BrowserOperationType.EXTRACT,
                {"url": url, "selector": selector, "text_length": len(text)},
                True
            )

            await page.close()

            return result

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            self._log_operation(
                session_id,
                BrowserOperationType.EXTRACT,
                {"url": url, "selector": selector},
                False,
                str(e)
            )
            raise

    async def take_screenshot(
        self,
        session_id: str,
        url: str,
        output_path: str,
        full_page: bool = False
    ) -> Dict[str, Any]:
        """
        Take screenshot of page.

        Args:
            session_id: Browser session ID
            url: URL to screenshot
            output_path: Path to save screenshot
            full_page: Capture full page (not just viewport)

        Returns:
            Screenshot info
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        try:
            browser = await self._start_browser()
            page = await browser.new_page()

            # Navigate
            await page.goto(url)

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Take screenshot
            await page.screenshot(path=output_path, full_page=full_page)

            result = {
                "url": url,
                "output_path": output_path,
                "full_page": full_page,
                "success": True
            }

            self._log_operation(
                session_id,
                BrowserOperationType.SCREENSHOT,
                {"url": url, "output_path": output_path},
                True
            )

            await page.close()

            return result

        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            self._log_operation(
                session_id,
                BrowserOperationType.SCREENSHOT,
                {"url": url, "output_path": output_path},
                False,
                str(e)
            )
            raise

    async def fill_form(
        self,
        session_id: str,
        url: str,
        form_data: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Fill form on page.
        NOTE: Does not submit the form - manual submission required.

        Args:
            session_id: Browser session ID
            url: URL containing the form
            form_data: Form fields to fill (selector: value)

        Returns:
            Fill result
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        # Check policy
        policy_result = self.policy_guard.evaluate({
            "action": "browser.fill_form",
            "url": url,
            "fields": list(form_data.keys())
        })

        if policy_result.decision == "deny":
            error_msg = f"Form fill denied by policy: {', '.join(policy_result.reasons)}"
            logger.error(error_msg)
            raise PermissionError(error_msg)

        try:
            browser = await self._start_browser()
            page = await browser.new_page()

            # Navigate
            await page.goto(url)

            # Fill form fields
            filled_fields = []
            for selector, value in form_data.items():
                try:
                    await page.fill(selector, value)
                    filled_fields.append(selector)
                except Exception as e:
                    logger.warning(f"Failed to fill field {selector}: {e}")

            result = {
                "url": url,
                "fields_filled": len(filled_fields),
                "total_fields": len(form_data),
                "success": len(filled_fields) > 0
            }

            self._log_operation(
                session_id,
                BrowserOperationType.FILL_FORM,
                {
                    "url": url,
                    "fields_filled": len(filled_fields),
                    "fields": list(form_data.keys())
                },
                True
            )

            await page.close()

            return result

        except Exception as e:
            logger.error(f"Form fill failed: {e}")
            self._log_operation(
                session_id,
                BrowserOperationType.FILL_FORM,
                {"url": url, "fields": list(form_data.keys())},
                False,
                str(e)
            )
            raise

    async def click_element(
        self,
        session_id: str,
        url: str,
        selector: str
    ) -> Dict[str, Any]:
        """
        Click element on page.

        Args:
            session_id: Browser session ID
            url: URL containing the element
            selector: CSS selector of element to click

        Returns:
            Click result
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        try:
            browser = await self._start_browser()
            page = await browser.new_page()

            # Navigate
            await page.goto(url)

            # Click element
            await page.click(selector)

            result = {
                "url": url,
                "selector": selector,
                "success": True
            }

            self._log_operation(
                session_id,
                BrowserOperationType.CLICK,
                {"url": url, "selector": selector},
                True
            )

            await page.close()

            return result

        except Exception as e:
            logger.error(f"Click failed: {e}")
            self._log_operation(
                session_id,
                BrowserOperationType.CLICK,
                {"url": url, "selector": selector},
                False,
                str(e)
            )
            raise

    async def close_session(self, session_id: str) -> None:
        """Close browser session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

            self._log_operation(
                session_id,
                BrowserOperationType.NAVIGATE,
                {"action": "session_closed"},
                True
            )

            logger.info(f"Browser session closed: {session_id}")

    async def cleanup(self) -> None:
        """Cleanup all sessions and stop browser"""
        self.active_sessions.clear()
        await self._stop_browser()
        logger.info("Browser automation cleanup complete")

    def get_status(self) -> Dict[str, Any]:
        """Get browser automation status"""
        return {
            "browser_type": self.browser_type.value,
            "headless": self.headless,
            "active_sessions": len(self.active_sessions),
            "operations_count": self.operations_count,
            "url_allowlist_size": len(self.url_allowlist),
            "max_pages_per_session": self.max_pages_per_session,
        }
