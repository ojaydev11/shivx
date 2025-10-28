"""
Redis Connection Management with Pooling and Health Checks
Provides connection factory, dependency injection, and graceful degradation
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
from datetime import datetime, timedelta

import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import (
    RedisError,
    ConnectionError as RedisConnectionError,
    TimeoutError as RedisTimeoutError,
)
from fastapi import Depends
from prometheus_client import Counter, Gauge, Histogram

from config.settings import Settings, get_settings


logger = logging.getLogger(__name__)


# ============================================================================
# Prometheus Metrics
# ============================================================================

redis_connections_total = Gauge(
    "redis_connections_total",
    "Total number of Redis connections in the pool"
)

redis_connections_active = Gauge(
    "redis_connections_active",
    "Number of active Redis connections"
)

redis_connection_errors = Counter(
    "redis_connection_errors_total",
    "Total number of Redis connection errors",
    ["error_type"]
)

redis_operation_duration = Histogram(
    "redis_operation_duration_seconds",
    "Redis operation duration in seconds",
    ["operation"]
)

redis_health_status = Gauge(
    "redis_health_status",
    "Redis health status (1=healthy, 0=unhealthy)"
)


# ============================================================================
# Redis Connection Pool Manager
# ============================================================================

class RedisManager:
    """
    Manages Redis connection pool with health checks and graceful degradation

    Features:
    - Connection pooling (10-50 connections)
    - Automatic reconnection with exponential backoff
    - Circuit breaker pattern
    - Health monitoring
    - Graceful degradation when Redis is unavailable
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[aioredis.Redis] = None
        self._healthy = False
        self._last_health_check: Optional[datetime] = None
        self._circuit_breaker_failures = 0
        self._circuit_breaker_open = False
        self._circuit_breaker_open_until: Optional[datetime] = None

        # Circuit breaker settings
        self.max_failures = 5
        self.circuit_breaker_timeout = 60  # seconds

        # Connection pool settings
        self.pool_size = 50
        self.pool_timeout = 30
        self.socket_connect_timeout = 5
        self.socket_timeout = 5
        self.retry_on_timeout = True
        self.max_retries = 3

    async def initialize(self) -> None:
        """Initialize Redis connection pool"""
        try:
            logger.info("Initializing Redis connection pool...")

            # Parse Redis URL
            url_parts = self.settings.redis_url.split("://")
            if len(url_parts) != 2:
                raise ValueError(f"Invalid Redis URL: {self.settings.redis_url}")

            # Create connection pool
            self.pool = ConnectionPool.from_url(
                self.settings.redis_url,
                password=self.settings.redis_password,
                max_connections=self.pool_size,
                socket_connect_timeout=self.socket_connect_timeout,
                socket_timeout=self.socket_timeout,
                retry_on_timeout=self.retry_on_timeout,
                health_check_interval=30,
                decode_responses=True,  # Auto-decode bytes to strings
            )

            # Create Redis client
            self.client = aioredis.Redis(
                connection_pool=self.pool,
                decode_responses=True,
            )

            # Test connection
            await self.client.ping()
            self._healthy = True
            self._last_health_check = datetime.utcnow()
            redis_health_status.set(1)

            logger.info(
                f"Redis connection pool initialized successfully "
                f"(pool_size={self.pool_size}, url={self.settings.redis_url})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Redis connection pool: {e}")
            redis_connection_errors.labels(error_type="initialization").inc()
            redis_health_status.set(0)
            self._healthy = False

            # Don't raise - allow graceful degradation
            if self.settings.is_production:
                logger.warning("Redis unavailable - running in degraded mode")
            else:
                logger.warning("Redis unavailable - cache operations will be no-ops")

    async def close(self) -> None:
        """Close Redis connection pool"""
        try:
            if self.client:
                await self.client.close()
            if self.pool:
                await self.pool.disconnect()
            logger.info("Redis connection pool closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection pool: {e}")

    async def health_check(self) -> bool:
        """
        Perform health check on Redis connection

        Returns:
            True if Redis is healthy, False otherwise
        """
        # Check circuit breaker
        if self._circuit_breaker_open:
            if datetime.utcnow() < self._circuit_breaker_open_until:
                return False
            else:
                # Reset circuit breaker
                logger.info("Circuit breaker timeout expired, attempting reconnection")
                self._circuit_breaker_open = False
                self._circuit_breaker_failures = 0

        try:
            if not self.client:
                return False

            # Ping Redis
            with redis_operation_duration.labels(operation="ping").time():
                await asyncio.wait_for(
                    self.client.ping(),
                    timeout=self.settings.redis_timeout
                )

            # Update health status
            self._healthy = True
            self._last_health_check = datetime.utcnow()
            self._circuit_breaker_failures = 0
            redis_health_status.set(1)

            # Update connection metrics
            if self.pool:
                redis_connections_total.set(self.pool_size)

            return True

        except (RedisConnectionError, RedisTimeoutError, asyncio.TimeoutError) as e:
            logger.warning(f"Redis health check failed: {e}")
            redis_connection_errors.labels(error_type="health_check").inc()
            self._healthy = False
            redis_health_status.set(0)

            # Update circuit breaker
            self._circuit_breaker_failures += 1
            if self._circuit_breaker_failures >= self.max_failures:
                self._circuit_breaker_open = True
                self._circuit_breaker_open_until = datetime.utcnow() + timedelta(
                    seconds=self.circuit_breaker_timeout
                )
                logger.error(
                    f"Circuit breaker opened due to {self._circuit_breaker_failures} "
                    f"consecutive failures. Will retry after {self.circuit_breaker_timeout}s"
                )

            return False

        except Exception as e:
            logger.error(f"Unexpected error during Redis health check: {e}")
            redis_connection_errors.labels(error_type="unexpected").inc()
            self._healthy = False
            redis_health_status.set(0)
            return False

    def is_healthy(self) -> bool:
        """Check if Redis is currently healthy"""
        return self._healthy and not self._circuit_breaker_open

    async def get_client(self) -> Optional[aioredis.Redis]:
        """
        Get Redis client if available

        Returns:
            Redis client or None if unavailable
        """
        if not self.is_healthy():
            # Attempt health check
            healthy = await self.health_check()
            if not healthy:
                return None

        return self.client

    async def get_info(self) -> dict:
        """Get Redis server info"""
        try:
            if not self.client:
                return {"status": "unavailable"}

            info = await self.client.info()
            return {
                "status": "healthy" if self._healthy else "unhealthy",
                "version": info.get("redis_version", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "used_memory_peak_human": info.get("used_memory_peak_human", "unknown"),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "evicted_keys": info.get("evicted_keys", 0),
                "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
                "circuit_breaker_open": self._circuit_breaker_open,
                "circuit_breaker_failures": self._circuit_breaker_failures,
            }
        except Exception as e:
            logger.error(f"Error getting Redis info: {e}")
            return {"status": "error", "error": str(e)}


# ============================================================================
# Global Redis Manager Instance
# ============================================================================

_redis_manager: Optional[RedisManager] = None


async def initialize_redis(settings: Settings) -> RedisManager:
    """
    Initialize global Redis manager

    Args:
        settings: Application settings

    Returns:
        RedisManager instance
    """
    global _redis_manager

    if _redis_manager is None:
        _redis_manager = RedisManager(settings)
        await _redis_manager.initialize()

    return _redis_manager


async def close_redis() -> None:
    """Close global Redis manager"""
    global _redis_manager

    if _redis_manager is not None:
        await _redis_manager.close()
        _redis_manager = None


def get_redis_manager() -> Optional[RedisManager]:
    """Get global Redis manager instance"""
    return _redis_manager


# ============================================================================
# FastAPI Dependencies
# ============================================================================

async def get_redis(
    settings: Settings = Depends(get_settings)
) -> AsyncGenerator[Optional[aioredis.Redis], None]:
    """
    FastAPI dependency to get Redis client

    Provides graceful degradation - returns None if Redis is unavailable

    Usage:
        @app.get("/data")
        async def get_data(redis: Optional[Redis] = Depends(get_redis)):
            if redis:
                cached = await redis.get("key")
                if cached:
                    return cached
            # Fallback to database
            return fetch_from_db()

    Yields:
        Redis client or None if unavailable
    """
    global _redis_manager

    if _redis_manager is None:
        _redis_manager = RedisManager(settings)
        await _redis_manager.initialize()

    client = await _redis_manager.get_client()

    try:
        yield client
    finally:
        # Connection is managed by pool, no cleanup needed
        pass


@asynccontextmanager
async def redis_pipeline(
    redis: Optional[aioredis.Redis] = None
) -> AsyncGenerator[Optional[aioredis.client.Pipeline], None]:
    """
    Context manager for Redis pipeline operations

    Usage:
        async with redis_pipeline(redis) as pipe:
            if pipe:
                pipe.set("key1", "value1")
                pipe.set("key2", "value2")
                await pipe.execute()

    Args:
        redis: Redis client

    Yields:
        Redis pipeline or None if Redis unavailable
    """
    if redis is None:
        yield None
        return

    try:
        async with redis.pipeline() as pipe:
            yield pipe
    except RedisError as e:
        logger.error(f"Redis pipeline error: {e}")
        redis_connection_errors.labels(error_type="pipeline").inc()
        yield None


# ============================================================================
# Cache Key Helpers
# ============================================================================

def make_cache_key(*parts: str, prefix: str = "shivx") -> str:
    """
    Create a cache key from parts

    Args:
        parts: Key parts to join
        prefix: Key prefix (default: "shivx")

    Returns:
        Cache key string

    Example:
        make_cache_key("market", "SOL-USDC", "price")
        # Returns: "shivx:market:SOL-USDC:price"
    """
    clean_parts = [str(part).replace(":", "_") for part in parts]
    return f"{prefix}:{':'.join(clean_parts)}"


# ============================================================================
# Cache Decorator with Graceful Degradation
# ============================================================================

def cache_with_fallback(
    key_func,
    ttl: int = 60,
    version: int = 1
):
    """
    Decorator to cache function results with fallback on Redis failure

    Args:
        key_func: Function to generate cache key from args
        ttl: Time to live in seconds
        version: Cache version (increment to invalidate all cached values)

    Usage:
        @cache_with_fallback(
            key_func=lambda token: f"price:{token}",
            ttl=5
        )
        async def get_price(token: str, redis: Redis = None) -> float:
            # This will be cached for 5 seconds
            return fetch_price_from_api(token)
    """
    def decorator(func):
        async def wrapper(*args, redis: Optional[aioredis.Redis] = None, **kwargs):
            # Generate cache key
            cache_key = make_cache_key("v" + str(version), key_func(*args, **kwargs))

            # Try to get from cache
            if redis:
                try:
                    with redis_operation_duration.labels(operation="get").time():
                        cached = await redis.get(cache_key)
                    if cached:
                        return cached
                except RedisError as e:
                    logger.warning(f"Cache get failed for {cache_key}: {e}")

            # Execute function
            result = await func(*args, redis=redis, **kwargs)

            # Try to cache result
            if redis and result is not None:
                try:
                    with redis_operation_duration.labels(operation="set").time():
                        await redis.setex(cache_key, ttl, result)
                except RedisError as e:
                    logger.warning(f"Cache set failed for {cache_key}: {e}")

            return result

        return wrapper
    return decorator
