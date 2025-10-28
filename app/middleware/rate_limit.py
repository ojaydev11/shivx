"""
Rate Limiting Middleware using Redis
Implements sliding window rate limiting with Redis-backed counters
"""

import logging
import time
from typing import Optional, Callable
from datetime import datetime, timedelta

import redis.asyncio as aioredis
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram

from config.settings import Settings
from app.cache import make_cache_key, redis_operation_duration


logger = logging.getLogger(__name__)


# ============================================================================
# Prometheus Metrics
# ============================================================================

rate_limit_hits = Counter(
    "rate_limit_hits_total",
    "Total number of rate limit hits (blocked requests)",
    ["limit_type"]
)

rate_limit_requests = Counter(
    "rate_limit_requests_total",
    "Total number of rate limited requests",
    ["limit_type", "status"]
)

rate_limit_duration = Histogram(
    "rate_limit_check_duration_seconds",
    "Rate limit check duration"
)


# ============================================================================
# Rate Limiter
# ============================================================================

class RedisRateLimiter:
    """
    Sliding window rate limiter using Redis

    Features:
    - Sliding window algorithm for accurate rate limiting
    - Per-IP and per-API-key rate limiting
    - Configurable time windows and limits
    - Rate limit headers in responses
    - Admin bypass with audit logging
    """

    def __init__(
        self,
        redis: Optional[aioredis.Redis],
        settings: Settings
    ):
        self.redis = redis
        self.settings = settings

        # Default rate limits
        self.default_limit = settings.rate_limit_per_minute
        self.window_seconds = 60  # 1 minute window

        # Rate limit tiers
        self.rate_limits = {
            "default": (60, 60),  # 60 requests per minute
            "authenticated": (120, 60),  # 120 requests per minute
            "premium": (300, 60),  # 300 requests per minute
            "admin": (1000, 60),  # 1000 requests per minute
        }

    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window: int
    ) -> tuple[bool, int, int]:
        """
        Check if request is within rate limit using sliding window

        Args:
            identifier: Unique identifier (IP, API key, etc.)
            limit: Maximum requests allowed
            window: Time window in seconds

        Returns:
            Tuple of (allowed, remaining, reset_time)
        """
        if not self.redis:
            # If Redis is unavailable, allow all requests (graceful degradation)
            return True, limit, int(time.time() + window)

        cache_key = make_cache_key("rate_limit", identifier)
        current_time = time.time()
        window_start = current_time - window

        try:
            with rate_limit_duration.time():
                # Use Redis Lua script for atomic operations
                lua_script = """
                local key = KEYS[1]
                local current_time = tonumber(ARGV[1])
                local window_start = tonumber(ARGV[2])
                local limit = tonumber(ARGV[3])
                local window = tonumber(ARGV[4])

                -- Remove old entries outside the window
                redis.call('ZREMRANGEBYSCORE', key, 0, window_start)

                -- Count current requests
                local current_count = redis.call('ZCARD', key)

                if current_count < limit then
                    -- Add new request
                    redis.call('ZADD', key, current_time, current_time)
                    redis.call('EXPIRE', key, window)
                    return {1, limit - current_count - 1, current_time + window}
                else
                    -- Rate limit exceeded
                    -- Get oldest request to calculate reset time
                    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
                    local reset_time = oldest[2] + window
                    return {0, 0, reset_time}
                end
                """

                result = await self.redis.eval(
                    lua_script,
                    1,
                    cache_key,
                    current_time,
                    window_start,
                    limit,
                    window
                )

                allowed = bool(result[0])
                remaining = int(result[1])
                reset_time = int(result[2])

                return allowed, remaining, reset_time

        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            # On error, allow request (fail open)
            return True, limit, int(current_time + window)

    async def get_rate_limit_info(
        self,
        identifier: str,
        limit: int,
        window: int
    ) -> dict:
        """
        Get current rate limit info without incrementing counter

        Args:
            identifier: Unique identifier
            limit: Maximum requests allowed
            window: Time window in seconds

        Returns:
            Dictionary with rate limit info
        """
        if not self.redis:
            return {
                "limit": limit,
                "remaining": limit,
                "reset": int(time.time() + window),
            }

        cache_key = make_cache_key("rate_limit", identifier)
        current_time = time.time()
        window_start = current_time - window

        try:
            # Remove old entries
            await self.redis.zremrangebyscore(cache_key, 0, window_start)

            # Count current requests
            current_count = await self.redis.zcard(cache_key)
            remaining = max(0, limit - current_count)

            # Get reset time
            if current_count > 0:
                oldest = await self.redis.zrange(cache_key, 0, 0, withscores=True)
                if oldest:
                    reset_time = int(oldest[0][1] + window)
                else:
                    reset_time = int(current_time + window)
            else:
                reset_time = int(current_time + window)

            return {
                "limit": limit,
                "remaining": remaining,
                "reset": reset_time,
                "current_count": current_count,
            }

        except Exception as e:
            logger.error(f"Error getting rate limit info: {e}")
            return {
                "limit": limit,
                "remaining": limit,
                "reset": int(current_time + window),
            }

    def get_rate_limit_tier(self, request: Request) -> tuple[int, int]:
        """
        Determine rate limit tier based on request

        Args:
            request: FastAPI request

        Returns:
            Tuple of (limit, window)
        """
        # Check for admin user (with audit logging)
        user = getattr(request.state, "user", None)
        if user and getattr(user, "is_admin", False):
            logger.info(
                f"Admin user {user.username} bypassing standard rate limit",
                extra={
                    "user_id": user.id,
                    "ip": self.get_client_ip(request),
                    "path": request.url.path,
                }
            )
            return self.rate_limits["admin"]

        # Check for authenticated user
        if user:
            # Check if premium user
            if getattr(user, "is_premium", False):
                return self.rate_limits["premium"]
            return self.rate_limits["authenticated"]

        # Default tier
        return self.rate_limits["default"]

    @staticmethod
    def get_client_ip(request: Request) -> str:
        """
        Get client IP address from request

        Args:
            request: FastAPI request

        Returns:
            Client IP address
        """
        # Check X-Forwarded-For header (proxy/load balancer)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct client
        return request.client.host if request.client else "unknown"

    @staticmethod
    def get_api_key(request: Request) -> Optional[str]:
        """
        Get API key from request

        Args:
            request: FastAPI request

        Returns:
            API key or None
        """
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]

        # Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key

        return None


# ============================================================================
# Rate Limiting Middleware
# ============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting

    Features:
    - Automatic rate limiting for all endpoints
    - Rate limit headers in responses
    - Per-IP and per-API-key tracking
    - Integration with Guardian Defense System
    - Prometheus metrics
    """

    def __init__(self, app, redis: Optional[aioredis.Redis], settings: Settings):
        super().__init__(app)
        self.limiter = RedisRateLimiter(redis, settings)
        self.settings = settings

        # Paths to exempt from rate limiting
        self.exempt_paths = {
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
        }

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request with rate limiting

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response
        """
        # Skip rate limiting for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Get identifier (prefer API key over IP)
        api_key = self.limiter.get_api_key(request)
        if api_key:
            identifier = f"apikey:{api_key}"
            limit_type = "api_key"
        else:
            ip = self.limiter.get_client_ip(request)
            identifier = f"ip:{ip}"
            limit_type = "ip"

        # Get rate limit tier
        limit, window = self.limiter.get_rate_limit_tier(request)

        # Check rate limit
        allowed, remaining, reset_time = await self.limiter.check_rate_limit(
            identifier, limit, window
        )

        # Add rate limit headers to response
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
        }

        if not allowed:
            # Rate limit exceeded
            rate_limit_hits.labels(limit_type=limit_type).inc()
            rate_limit_requests.labels(limit_type=limit_type, status="blocked").inc()

            logger.warning(
                f"Rate limit exceeded for {identifier}",
                extra={
                    "identifier": identifier,
                    "limit_type": limit_type,
                    "path": request.url.path,
                    "limit": limit,
                }
            )

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Try again in {reset_time - int(time.time())} seconds.",
                    "limit": limit,
                    "window": window,
                    "reset": reset_time,
                },
                headers=headers,
            )

        # Request allowed
        rate_limit_requests.labels(limit_type=limit_type, status="allowed").inc()

        # Process request
        response = await call_next(request)

        # Add rate limit headers to successful response
        for key, value in headers.items():
            response.headers[key] = value

        return response


# ============================================================================
# Rate Limit Dependency
# ============================================================================

async def check_rate_limit(
    request: Request,
    redis: Optional[aioredis.Redis] = None,
    settings: Settings = None
):
    """
    FastAPI dependency for manual rate limit checking

    Usage:
        @app.post("/expensive-operation")
        async def expensive_op(
            _: None = Depends(check_rate_limit)
        ):
            # This endpoint has additional rate limiting
            return {"status": "ok"}

    Args:
        request: FastAPI request
        redis: Redis client
        settings: App settings

    Raises:
        HTTPException: If rate limit exceeded
    """
    if not redis or not settings:
        return

    limiter = RedisRateLimiter(redis, settings)

    # Get identifier
    api_key = limiter.get_api_key(request)
    if api_key:
        identifier = f"apikey:{api_key}:expensive"
    else:
        ip = limiter.get_client_ip(request)
        identifier = f"ip:{ip}:expensive"

    # Stricter limit for expensive operations
    limit = 10
    window = 60

    allowed, remaining, reset_time = await limiter.check_rate_limit(
        identifier, limit, window
    )

    if not allowed:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit for this endpoint exceeded. Try again in {reset_time - int(time.time())} seconds.",
                "limit": limit,
                "reset": reset_time,
            },
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(reset_time),
            }
        )
