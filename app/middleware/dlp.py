"""
DLP Middleware for ShivX API

Scans API responses for sensitive data before returning to clients.
Automatically redacts PII and secrets from responses.
"""

import logging
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from utils.dlp import get_dlp_filter, DetectionResult

logger = logging.getLogger(__name__)


class DLPMiddleware(BaseHTTPMiddleware):
    """
    Middleware to scan API responses for sensitive data

    Scans all API responses and redacts:
    - PII (SSN, email, phone, credit cards)
    - API keys and secrets
    - Passwords
    - Private keys
    """

    def __init__(
        self,
        app: ASGIApp,
        enabled: bool = True,
        log_detections: bool = True,
        block_on_detection: bool = False
    ):
        """
        Initialize DLP middleware

        Args:
            app: ASGI application
            enabled: Whether DLP scanning is enabled
            log_detections: Whether to log detections
            block_on_detection: Whether to block responses with sensitive data
        """
        super().__init__(app)
        self.enabled = enabled
        self.log_detections = log_detections
        self.block_on_detection = block_on_detection
        self.dlp = get_dlp_filter(enable_logging=log_detections)

        logger.info(
            f"DLP Middleware initialized (enabled={enabled}, "
            f"block_on_detection={block_on_detection})"
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and scan response

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response (potentially redacted)
        """
        if not self.enabled:
            return await call_next(request)

        # Get response
        response = await call_next(request)

        # Only scan successful responses with content
        if response.status_code >= 400:
            return response

        # Check if response is JSON (most API responses)
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return response

        # Read response body
        body_bytes = b""
        async for chunk in response.body_iterator:
            body_bytes += chunk

        if not body_bytes:
            return response

        try:
            # Try to decode as JSON
            body_text = body_bytes.decode("utf-8")

            # Scan for sensitive data
            scan_result = self.dlp.scan(body_text)

            if scan_result.found_sensitive_data:
                # Log detection
                logger.warning(
                    f"DLP: Sensitive data detected in response to {request.url.path} - "
                    f"Types: {scan_result.metadata.get('detection_types', [])}, "
                    f"Count: {scan_result.redacted_count}"
                )

                # Log to audit trail
                self._log_detection(request, scan_result)

                if self.block_on_detection:
                    # Block response entirely
                    return Response(
                        content=json.dumps({
                            "error": "Response blocked by DLP",
                            "reason": "Sensitive data detected",
                            "detection_types": scan_result.metadata.get('detection_types', [])
                        }),
                        status_code=403,
                        media_type="application/json"
                    )
                else:
                    # Return redacted response
                    body_text = scan_result.redacted_text

            # Return response (original or redacted)
            return Response(
                content=body_text,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type="application/json"
            )

        except Exception as e:
            logger.error(f"DLP middleware error: {e}")
            # On error, return original response
            return Response(
                content=body_bytes,
                status_code=response.status_code,
                headers=dict(response.headers)
            )

    def _log_detection(self, request: Request, scan_result: DetectionResult):
        """
        Log DLP detection to audit trail

        Args:
            request: Request that triggered detection
            scan_result: DLP scan result
        """
        try:
            # Log detailed detection info
            logger.info(
                "DLP_DETECTION",
                extra={
                    "event_type": "dlp_detection",
                    "path": request.url.path,
                    "method": request.method,
                    "detection_count": scan_result.redacted_count,
                    "detection_types": scan_result.metadata.get('detection_types', []),
                    "client_ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent"),
                }
            )

            # TODO: Write to security audit log table
            # from app.models.security import SecurityAuditLog
            # SecurityAuditLog.create(...)

        except Exception as e:
            logger.error(f"Failed to log DLP detection: {e}")


def create_dlp_middleware(
    enabled: bool = True,
    log_detections: bool = True,
    block_on_detection: bool = False
) -> type:
    """
    Factory function to create DLP middleware with configuration

    Args:
        enabled: Whether DLP scanning is enabled
        log_detections: Whether to log detections
        block_on_detection: Whether to block responses with sensitive data

    Returns:
        Configured DLP middleware class
    """
    class ConfiguredDLPMiddleware(DLPMiddleware):
        def __init__(self, app: ASGIApp):
            super().__init__(
                app,
                enabled=enabled,
                log_detections=log_detections,
                block_on_detection=block_on_detection
            )

    return ConfiguredDLPMiddleware
