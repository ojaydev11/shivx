"""
Cache Invalidation Strategy
Smart cache invalidation with pub/sub, event-driven invalidation, and bulk operations
"""

import json
import logging
from typing import Optional, Dict, Any, List, Set, Callable
from datetime import datetime
from enum import Enum

import redis.asyncio as aioredis
from prometheus_client import Counter

from app.cache import make_cache_key


logger = logging.getLogger(__name__)


# ============================================================================
# Prometheus Metrics
# ============================================================================

cache_invalidations = Counter(
    "cache_invalidations_total",
    "Total number of cache invalidations",
    ["invalidation_type", "reason"]
)


# ============================================================================
# Invalidation Events
# ============================================================================

class InvalidationEvent(str, Enum):
    """Cache invalidation event types"""
    TRADE_EXECUTED = "trade_executed"
    ORDER_PLACED = "order_placed"
    ORDER_CANCELLED = "order_cancelled"
    PRICE_UPDATE = "price_update"
    MODEL_RETRAINED = "model_retrained"
    USER_LOGOUT = "user_logout"
    MANUAL_FLUSH = "manual_flush"
    SYSTEM_UPDATE = "system_update"


# ============================================================================
# Cache Invalidation Manager
# ============================================================================

class CacheInvalidationManager:
    """
    Manages cache invalidation with smart patterns

    Features:
    - Event-driven invalidation (trades, orders)
    - Pub/sub for distributed invalidation
    - Cache tagging for bulk invalidation
    - Pattern-based invalidation
    - Manual flush endpoint (admin only)
    - Invalidation history tracking
    """

    CACHE_VERSION = 1
    PUBSUB_CHANNEL = "cache:invalidation"

    def __init__(self, redis: Optional[aioredis.Redis]):
        self.redis = redis
        self.pubsub = None
        self._invalidation_handlers: Dict[InvalidationEvent, List[Callable]] = {}
        self._invalidation_history: List[Dict[str, Any]] = []

    # ========================================================================
    # Pub/Sub for Distributed Invalidation
    # ========================================================================

    async def initialize_pubsub(self):
        """Initialize Redis pub/sub for cache invalidation"""
        if not self.redis:
            logger.warning("Redis unavailable, pub/sub not initialized")
            return

        try:
            self.pubsub = self.redis.pubsub()
            await self.pubsub.subscribe(self.PUBSUB_CHANNEL)
            logger.info(f"Subscribed to cache invalidation channel: {self.PUBSUB_CHANNEL}")
        except Exception as e:
            logger.error(f"Error initializing pub/sub: {e}")

    async def close_pubsub(self):
        """Close pub/sub connection"""
        if self.pubsub:
            await self.pubsub.unsubscribe(self.PUBSUB_CHANNEL)
            await self.pubsub.close()
            logger.info("Closed pub/sub connection")

    async def publish_invalidation(
        self,
        event: InvalidationEvent,
        data: Dict[str, Any]
    ) -> bool:
        """
        Publish cache invalidation event

        Args:
            event: Invalidation event type
            data: Event data

        Returns:
            True if published successfully
        """
        if not self.redis:
            return False

        try:
            message = {
                "event": event.value,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
            }

            await self.redis.publish(
                self.PUBSUB_CHANNEL,
                json.dumps(message)
            )

            logger.info(f"Published invalidation event: {event.value}")
            return True

        except Exception as e:
            logger.error(f"Error publishing invalidation event: {e}")
            return False

    async def listen_for_invalidations(self):
        """
        Listen for invalidation events from pub/sub

        This should run in a background task
        """
        if not self.pubsub:
            logger.warning("Pub/sub not initialized")
            return

        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        event = InvalidationEvent(data["event"])
                        event_data = data["data"]

                        logger.info(f"Received invalidation event: {event.value}")

                        # Execute registered handlers
                        await self._execute_handlers(event, event_data)

                    except Exception as e:
                        logger.error(f"Error processing invalidation message: {e}")

        except Exception as e:
            logger.error(f"Error listening for invalidations: {e}")

    # ========================================================================
    # Event Handlers
    # ========================================================================

    def register_handler(
        self,
        event: InvalidationEvent,
        handler: Callable
    ):
        """
        Register a handler for an invalidation event

        Args:
            event: Invalidation event type
            handler: Async handler function

        Example:
            async def on_trade_executed(data):
                # Invalidate price cache
                pass

            manager.register_handler(
                InvalidationEvent.TRADE_EXECUTED,
                on_trade_executed
            )
        """
        if event not in self._invalidation_handlers:
            self._invalidation_handlers[event] = []

        self._invalidation_handlers[event].append(handler)
        logger.info(f"Registered handler for event: {event.value}")

    async def _execute_handlers(
        self,
        event: InvalidationEvent,
        data: Dict[str, Any]
    ):
        """Execute all registered handlers for an event"""
        handlers = self._invalidation_handlers.get(event, [])

        for handler in handlers:
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Error executing handler for {event.value}: {e}")

    # ========================================================================
    # Event-Driven Invalidation
    # ========================================================================

    async def on_trade_executed(
        self,
        token_pair: str,
        trade_data: Dict[str, Any]
    ) -> int:
        """
        Invalidate caches when a trade is executed

        Args:
            token_pair: Token pair
            trade_data: Trade data

        Returns:
            Number of cache entries invalidated
        """
        if not self.redis:
            return 0

        invalidated = 0

        try:
            # Invalidate price cache
            price_key = make_cache_key("market", "price", token_pair, f"v{self.CACHE_VERSION}")
            await self.redis.delete(price_key)
            invalidated += 1

            # Invalidate order book
            orderbook_key = make_cache_key("market", "orderbook", token_pair, f"v{self.CACHE_VERSION}")
            await self.redis.delete(orderbook_key)
            invalidated += 1

            # Invalidate trade history
            trades_pattern = make_cache_key("market", "trades", token_pair, "*")
            async for key in self.redis.scan_iter(match=trades_pattern):
                await self.redis.delete(key)
                invalidated += 1

            # Invalidate indicators (they depend on price data)
            indicator_pattern = make_cache_key("indicator", "*", token_pair, "*")
            async for key in self.redis.scan_iter(match=indicator_pattern):
                await self.redis.delete(key)
                invalidated += 1

            # Invalidate ML predictions (they depend on features which depend on prices)
            ml_pattern = make_cache_key("ml", "*", "*", "*", token_pair, "*")
            async for key in self.redis.scan_iter(match=ml_pattern):
                await self.redis.delete(key)
                invalidated += 1

            # Track invalidation
            self._record_invalidation(
                InvalidationEvent.TRADE_EXECUTED,
                {"token_pair": token_pair, "keys_invalidated": invalidated}
            )

            cache_invalidations.labels(
                invalidation_type="trade",
                reason="trade_executed"
            ).inc()

            logger.info(f"Invalidated {invalidated} cache entries for trade on {token_pair}")
            return invalidated

        except Exception as e:
            logger.error(f"Error invalidating caches on trade: {e}")
            return invalidated

    async def on_order_placed(
        self,
        token_pair: str,
        order_data: Dict[str, Any]
    ) -> int:
        """
        Invalidate caches when an order is placed

        Args:
            token_pair: Token pair
            order_data: Order data

        Returns:
            Number of cache entries invalidated
        """
        if not self.redis:
            return 0

        try:
            # Invalidate order book
            orderbook_key = make_cache_key("market", "orderbook", token_pair, f"v{self.CACHE_VERSION}")
            await self.redis.delete(orderbook_key)

            cache_invalidations.labels(
                invalidation_type="order",
                reason="order_placed"
            ).inc()

            logger.info(f"Invalidated order book cache for {token_pair}")
            return 1

        except Exception as e:
            logger.error(f"Error invalidating caches on order placement: {e}")
            return 0

    async def on_model_retrained(
        self,
        model_name: str,
        model_version: str
    ) -> int:
        """
        Invalidate ML prediction caches when a model is retrained

        Args:
            model_name: Name of model
            model_version: Model version

        Returns:
            Number of cache entries invalidated
        """
        if not self.redis:
            return 0

        try:
            # Invalidate all predictions for this model
            pattern = make_cache_key("ml", "prediction", model_name, model_version, "*")

            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                invalidated = await self.redis.delete(*keys)
            else:
                invalidated = 0

            cache_invalidations.labels(
                invalidation_type="model",
                reason="model_retrained"
            ).inc()

            logger.info(f"Invalidated {invalidated} predictions for model {model_name}")
            return invalidated

        except Exception as e:
            logger.error(f"Error invalidating ML caches on model retrain: {e}")
            return 0

    # ========================================================================
    # Cache Tagging for Bulk Invalidation
    # ========================================================================

    async def tag_cache_entry(
        self,
        cache_key: str,
        tags: List[str]
    ) -> bool:
        """
        Tag a cache entry for bulk invalidation

        Args:
            cache_key: Cache key
            tags: List of tags

        Returns:
            True if tagged successfully
        """
        if not self.redis:
            return False

        try:
            for tag in tags:
                tag_key = make_cache_key("tag", tag)
                await self.redis.sadd(tag_key, cache_key)

            return True

        except Exception as e:
            logger.error(f"Error tagging cache entry: {e}")
            return False

    async def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalidate all cache entries with a specific tag

        Args:
            tag: Tag name

        Returns:
            Number of cache entries invalidated
        """
        if not self.redis:
            return 0

        try:
            tag_key = make_cache_key("tag", tag)

            # Get all cache keys with this tag
            cache_keys = await self.redis.smembers(tag_key)

            if cache_keys:
                # Delete all tagged cache entries
                invalidated = await self.redis.delete(*cache_keys)

                # Delete tag set
                await self.redis.delete(tag_key)

                cache_invalidations.labels(
                    invalidation_type="tag",
                    reason=f"tag_{tag}"
                ).inc()

                logger.info(f"Invalidated {invalidated} cache entries with tag '{tag}'")
                return invalidated

            return 0

        except Exception as e:
            logger.error(f"Error invalidating by tag: {e}")
            return 0

    # ========================================================================
    # Pattern-Based Invalidation
    # ========================================================================

    async def invalidate_by_pattern(
        self,
        pattern: str,
        reason: str = "pattern_match"
    ) -> int:
        """
        Invalidate cache entries matching a pattern

        Args:
            pattern: Redis key pattern (e.g., "market:price:*")
            reason: Reason for invalidation

        Returns:
            Number of cache entries invalidated
        """
        if not self.redis:
            return 0

        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                invalidated = await self.redis.delete(*keys)
            else:
                invalidated = 0

            cache_invalidations.labels(
                invalidation_type="pattern",
                reason=reason
            ).inc()

            logger.info(f"Invalidated {invalidated} cache entries matching pattern '{pattern}'")
            return invalidated

        except Exception as e:
            logger.error(f"Error invalidating by pattern: {e}")
            return 0

    # ========================================================================
    # Manual Cache Flush (Admin Only)
    # ========================================================================

    async def flush_all_cache(
        self,
        admin_user_id: str,
        reason: str = "manual_flush"
    ) -> Dict[str, Any]:
        """
        Flush all cache (admin only, with audit logging)

        Args:
            admin_user_id: ID of admin user performing flush
            reason: Reason for flush

        Returns:
            Dictionary with flush result
        """
        if not self.redis:
            return {"status": "error", "message": "Redis unavailable"}

        try:
            # Get key count before flush
            info = await self.redis.info()
            keys_before = info.get("db0", {}).get("keys", 0)

            # Flush all keys with shivx prefix
            pattern = make_cache_key("*")
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self.redis.delete(*keys)
            else:
                deleted = 0

            # Audit log
            audit_entry = {
                "action": "cache_flush_all",
                "admin_user_id": admin_user_id,
                "reason": reason,
                "keys_deleted": deleted,
                "timestamp": datetime.utcnow().isoformat(),
            }

            logger.warning(
                f"CACHE FLUSH: Admin {admin_user_id} flushed all cache "
                f"({deleted} keys, reason: {reason})",
                extra=audit_entry
            )

            cache_invalidations.labels(
                invalidation_type="flush_all",
                reason=reason
            ).inc()

            # Record in history
            self._record_invalidation(
                InvalidationEvent.MANUAL_FLUSH,
                audit_entry
            )

            return {
                "status": "success",
                "keys_deleted": deleted,
                "admin_user_id": admin_user_id,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return {"status": "error", "error": str(e)}

    async def flush_cache_type(
        self,
        cache_type: str,
        admin_user_id: str,
        reason: str = "manual_flush"
    ) -> Dict[str, Any]:
        """
        Flush cache of a specific type (e.g., "market", "ml")

        Args:
            cache_type: Cache type to flush
            admin_user_id: ID of admin user
            reason: Reason for flush

        Returns:
            Dictionary with flush result
        """
        pattern = make_cache_key(cache_type, "*")
        deleted = await self.invalidate_by_pattern(pattern, reason)

        audit_entry = {
            "action": "cache_flush_type",
            "cache_type": cache_type,
            "admin_user_id": admin_user_id,
            "reason": reason,
            "keys_deleted": deleted,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.warning(
            f"CACHE FLUSH: Admin {admin_user_id} flushed {cache_type} cache "
            f"({deleted} keys, reason: {reason})",
            extra=audit_entry
        )

        return {
            "status": "success",
            "cache_type": cache_type,
            "keys_deleted": deleted,
            "admin_user_id": admin_user_id,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ========================================================================
    # Invalidation History
    # ========================================================================

    def _record_invalidation(
        self,
        event: InvalidationEvent,
        data: Dict[str, Any]
    ):
        """Record invalidation in history"""
        self._invalidation_history.append({
            "event": event.value,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Keep only last 1000 entries
        if len(self._invalidation_history) > 1000:
            self._invalidation_history = self._invalidation_history[-1000:]

    def get_invalidation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get invalidation history

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of invalidation history entries
        """
        return self._invalidation_history[-limit:]

    # ========================================================================
    # Cache Warming After Invalidation
    # ========================================================================

    async def invalidate_and_warm(
        self,
        cache_key: str,
        warm_func: Callable,
        warm_args: tuple = ()
    ) -> bool:
        """
        Invalidate cache and immediately warm it with new data

        Args:
            cache_key: Cache key to invalidate
            warm_func: Function to fetch fresh data
            warm_args: Arguments for warm function

        Returns:
            True if successful
        """
        if not self.redis:
            return False

        try:
            # Invalidate
            await self.redis.delete(cache_key)

            # Warm with fresh data
            fresh_data = await warm_func(*warm_args)

            # Cache is warmed by the warm_func itself
            return True

        except Exception as e:
            logger.error(f"Error in invalidate_and_warm: {e}")
            return False
