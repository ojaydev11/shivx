"""
HTTP Response Caching Middleware
Caches GET requests based on URL and query params with ETag support
"""

import hashlib
import json
import logging
from typing import Optional, Callable, Dict, Set
from datetime import datetime

import redis.asyncio as aioredis
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import MutableHeaders
from prometheus_client import Counter, Histogram

from app.cache import make_cache_key, redis_operation_duration


logger = logging.getLogger(__name__)


# ============================================================================
# Prometheus Metrics
# ============================================================================

http_cache_hits = Counter(
    "http_cache_hits_total",
    "Total number of HTTP cache hits"
)

http_cache_misses = Counter(
    "http_cache_misses_total",
    "Total number of HTTP cache misses"
)

http_cache_bypassed = Counter(
    "http_cache_bypassed_total",
    "Total number of HTTP cache bypasses",
    ["reason"]
)

http_cache_duration = Histogram(
    "http_cache_operation_duration_seconds",
    "HTTP cache operation duration",
    ["operation"]
)


# ============================================================================
# Cache Configuration
# ============================================================================

class CacheConfig:
    """
    Per-endpoint cache configuration

    Attributes:
        ttl: Cache TTL in seconds
        cache_authenticated: Whether to cache authenticated requests
        vary_on_headers: HTTP headers to include in cache key
        cache_control: Cache-Control header value
    """

    def __init__(
        self,
        ttl: int = 60,
        cache_authenticated: bool = False,
        vary_on_headers: Optional[Set[str]] = None,
        cache_control: Optional[str] = None
    ):
        self.ttl = ttl
        self.cache_authenticated = cache_authenticated
        self.vary_on_headers = vary_on_headers or set()
        self.cache_control = cache_control or f"public, max-age={ttl}"


# Default cache configurations per endpoint pattern
DEFAULT_CACHE_CONFIGS: Dict[str, CacheConfig] = {
    "/api/market/prices": CacheConfig(ttl=5, cache_control="public, max-age=5"),
    "/api/market/orderbook": CacheConfig(ttl=10, cache_control="public, max-age=10"),
    "/api/market/ohlcv": CacheConfig(ttl=3600, cache_control="public, max-age=3600"),
    "/api/analytics": CacheConfig(ttl=300, cache_control="public, max-age=300"),
    "/api/indicators": CacheConfig(ttl=60, cache_control="public, max-age=60"),
}


# ============================================================================
# HTTP Cache Manager
# ============================================================================

class HTTPCacheManager:
    """
    Manages HTTP response caching with ETags and Cache-Control headers
    """

    CACHE_VERSION = 1

    def __init__(
        self,
        redis: Optional[aioredis.Redis],
        cache_configs: Optional[Dict[str, CacheConfig]] = None
    ):
        self.redis = redis
        self.cache_configs = cache_configs or DEFAULT_CACHE_CONFIGS

    def _generate_cache_key(
        self,
        request: Request,
        config: CacheConfig
    ) -> str:
        """
        Generate cache key from request

        Args:
            request: FastAPI request
            config: Cache configuration

        Returns:
            Cache key string
        """
        # Base key from path
        path = request.url.path

        # Include query parameters (sorted for consistency)
        query_params = sorted(request.query_params.items())
        query_string = "&".join(f"{k}={v}" for k, v in query_params)

        # Include specified headers
        header_values = []
        for header_name in config.vary_on_headers:
            header_value = request.headers.get(header_name, "")
            header_values.append(f"{header_name}:{header_value}")

        # Combine all parts
        key_parts = [path]
        if query_string:
            key_parts.append(query_string)
        if header_values:
            key_parts.extend(header_values)

        return make_cache_key("http", *key_parts, f"v{self.CACHE_VERSION}")

    def _generate_etag(self, content: bytes) -> str:
        """
        Generate ETag from content

        Args:
            content: Response content

        Returns:
            ETag string
        """
        content_hash = hashlib.sha256(content).hexdigest()[:16]
        return f'"{content_hash}"'

    def _should_cache(
        self,
        request: Request,
        config: CacheConfig
    ) -> tuple[bool, Optional[str]]:
        """
        Determine if request should be cached

        Args:
            request: FastAPI request
            config: Cache configuration

        Returns:
            Tuple of (should_cache, reason_if_not)
        """
        # Only cache GET requests
        if request.method != "GET":
            return False, "non_get_method"

        # Check if authenticated request
        if not config.cache_authenticated:
            auth_header = request.headers.get("Authorization")
            if auth_header:
                return False, "authenticated"

        # Check Cache-Control request header
        cache_control = request.headers.get("Cache-Control", "")
        if "no-cache" in cache_control or "no-store" in cache_control:
            return False, "client_no_cache"

        return True, None

    async def get_cached_response(
        self,
        cache_key: str
    ) -> Optional[Dict[str, any]]:
        """
        Get cached response

        Args:
            cache_key: Cache key

        Returns:
            Cached response data or None
        """
        if not self.redis:
            return None

        try:
            with http_cache_duration.labels(operation="get").time():
                cached = await self.redis.get(cache_key)

            if cached:
                http_cache_hits.inc()
                return json.loads(cached)
            else:
                http_cache_misses.inc()
                return None

        except Exception as e:
            logger.error(f"Error getting cached response: {e}")
            http_cache_misses.inc()
            return None

    async def set_cached_response(
        self,
        cache_key: str,
        status_code: int,
        headers: dict,
        content: bytes,
        ttl: int
    ) -> bool:
        """
        Cache response

        Args:
            cache_key: Cache key
            status_code: HTTP status code
            headers: Response headers
            content: Response content
            ttl: TTL in seconds

        Returns:
            True if cached successfully
        """
        if not self.redis:
            return False

        try:
            # Prepare cached data
            cached_data = {
                "status_code": status_code,
                "headers": dict(headers),
                "content": content.decode("utf-8") if isinstance(content, bytes) else content,
                "cached_at": datetime.utcnow().isoformat(),
            }

            with http_cache_duration.labels(operation="set").time():
                await self.redis.setex(
                    cache_key,
                    ttl,
                    json.dumps(cached_data)
                )

            return True

        except Exception as e:
            logger.error(f"Error caching response: {e}")
            return False

    def get_cache_config(self, path: str) -> Optional[CacheConfig]:
        """
        Get cache configuration for a path

        Args:
            path: Request path

        Returns:
            CacheConfig or None
        """
        # Exact match
        if path in self.cache_configs:
            return self.cache_configs[path]

        # Prefix match
        for pattern, config in self.cache_configs.items():
            if path.startswith(pattern):
                return config

        return None


# ============================================================================
# Caching Middleware
# ============================================================================

class CacheMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for HTTP response caching

    Features:
    - Caches GET requests by URL + query params
    - Supports Cache-Control headers
    - Generates ETags for conditional requests
    - Per-endpoint cache configuration
    - Prometheus metrics
    """

    def __init__(
        self,
        app,
        redis: Optional[aioredis.Redis],
        cache_configs: Optional[Dict[str, CacheConfig]] = None
    ):
        super().__init__(app)
        self.cache_manager = HTTPCacheManager(redis, cache_configs)

        # Paths to never cache
        self.never_cache_paths = {
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
        Process request with caching

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response
        """
        path = request.url.path

        # Skip caching for certain paths
        if path in self.never_cache_paths:
            return await call_next(request)

        # Get cache configuration for this endpoint
        config = self.cache_manager.get_cache_config(path)
        if not config:
            # No cache config, pass through
            return await call_next(request)

        # Check if should cache
        should_cache, reason = self.cache_manager._should_cache(request, config)
        if not should_cache:
            http_cache_bypassed.labels(reason=reason).inc()
            return await call_next(request)

        # Generate cache key
        cache_key = self.cache_manager._generate_cache_key(request, config)

        # Try to get cached response
        cached_response = await self.cache_manager.get_cached_response(cache_key)

        if cached_response:
            # Check ETag for conditional request
            if_none_match = request.headers.get("If-None-Match")
            cached_etag = cached_response.get("headers", {}).get("etag")

            if if_none_match and cached_etag and if_none_match == cached_etag:
                # Return 304 Not Modified
                return Response(
                    status_code=304,
                    headers={
                        "ETag": cached_etag,
                        "Cache-Control": config.cache_control,
                    }
                )

            # Return cached response
            content = cached_response["content"]
            headers = cached_response.get("headers", {})

            # Add cache headers
            headers["X-Cache"] = "HIT"
            headers["Cache-Control"] = config.cache_control

            return JSONResponse(
                status_code=cached_response["status_code"],
                content=content if isinstance(content, dict) else {"data": content},
                headers=headers
            )

        # Cache miss - process request
        response = await call_next(request)

        # Only cache successful responses (2xx)
        if 200 <= response.status_code < 300:
            # Read response body
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk

            # Generate ETag
            etag = self.cache_manager._generate_etag(response_body)

            # Prepare headers
            headers = MutableHeaders(response.headers)
            headers["ETag"] = etag
            headers["Cache-Control"] = config.cache_control
            headers["X-Cache"] = "MISS"

            # Cache the response
            await self.cache_manager.set_cached_response(
                cache_key,
                response.status_code,
                dict(headers),
                response_body,
                config.ttl
            )

            # Return response with cached body
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(headers),
                media_type=response.media_type
            )

        # Don't cache error responses
        return response


# ============================================================================
# Cache Decorator
# ============================================================================

def cache_response(ttl: int = 60, cache_authenticated: bool = False):
    """
    Decorator to cache endpoint responses

    Usage:
        @app.get("/api/data")
        @cache_response(ttl=300)
        async def get_data():
            return {"data": "expensive computation"}

    Args:
        ttl: Cache TTL in seconds
        cache_authenticated: Whether to cache authenticated requests

    Returns:
        Decorator function
    """
    def decorator(func):
        # Store cache config on function
        func._cache_config = CacheConfig(
            ttl=ttl,
            cache_authenticated=cache_authenticated
        )
        return func

    return decorator


# ============================================================================
# Cache Invalidation Helper
# ============================================================================

async def invalidate_http_cache(
    redis: Optional[aioredis.Redis],
    path_pattern: str
) -> int:
    """
    Invalidate HTTP cache for a path pattern

    Args:
        redis: Redis client
        path_pattern: Path pattern to invalidate (e.g., "/api/market/*")

    Returns:
        Number of cache entries invalidated
    """
    if not redis:
        return 0

    cache_key_pattern = make_cache_key("http", path_pattern, "*")

    try:
        keys = []
        async for key in redis.scan_iter(match=cache_key_pattern):
            keys.append(key)

        if keys:
            deleted = await redis.delete(*keys)
            logger.info(f"Invalidated {deleted} HTTP cache entries for pattern {path_pattern}")
            return deleted

        return 0

    except Exception as e:
        logger.error(f"Error invalidating HTTP cache: {e}")
        return 0
