"""
Market Data Caching Service
Caches market prices, order books, OHLCV data, and token metadata
"""

import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

import redis.asyncio as aioredis
from prometheus_client import Counter, Histogram

from app.cache import make_cache_key, redis_operation_duration


logger = logging.getLogger(__name__)


# ============================================================================
# Prometheus Metrics
# ============================================================================

cache_hits = Counter(
    "market_cache_hits_total",
    "Total number of market cache hits",
    ["cache_type"]
)

cache_misses = Counter(
    "market_cache_misses_total",
    "Total number of market cache misses",
    ["cache_type"]
)

cache_invalidations = Counter(
    "market_cache_invalidations_total",
    "Total number of market cache invalidations",
    ["cache_type"]
)


# ============================================================================
# Market Data Cache Service
# ============================================================================

class MarketDataCache:
    """
    Caches market data with different TTLs based on data type

    TTL Strategy:
    - Market prices: 5 seconds (high frequency)
    - Order book data: 10 seconds (medium frequency)
    - Historical OHLCV: 1 hour (low frequency)
    - Token metadata: 24 hours (static data)
    """

    # Cache TTLs (seconds)
    TTL_PRICE = 5
    TTL_ORDERBOOK = 10
    TTL_OHLCV = 3600  # 1 hour
    TTL_TOKEN_METADATA = 86400  # 24 hours
    TTL_TRADE_HISTORY = 60  # 1 minute

    # Cache version (increment to invalidate all caches)
    CACHE_VERSION = 1

    def __init__(self, redis: Optional[aioredis.Redis] = None):
        self.redis = redis

    # ========================================================================
    # Market Price Caching
    # ========================================================================

    async def get_price(self, token_pair: str) -> Optional[Dict[str, Any]]:
        """
        Get cached market price

        Args:
            token_pair: Token pair (e.g., "SOL-USDC")

        Returns:
            Price data or None if not cached
        """
        if not self.redis:
            return None

        cache_key = make_cache_key("market", "price", token_pair, f"v{self.CACHE_VERSION}")

        try:
            with redis_operation_duration.labels(operation="get").time():
                cached = await self.redis.get(cache_key)

            if cached:
                cache_hits.labels(cache_type="price").inc()
                return json.loads(cached)
            else:
                cache_misses.labels(cache_type="price").inc()
                return None

        except Exception as e:
            logger.error(f"Error getting cached price for {token_pair}: {e}")
            cache_misses.labels(cache_type="price").inc()
            return None

    async def set_price(
        self,
        token_pair: str,
        price_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache market price

        Args:
            token_pair: Token pair (e.g., "SOL-USDC")
            price_data: Price data to cache
            ttl: Custom TTL in seconds (default: TTL_PRICE)

        Returns:
            True if cached successfully
        """
        if not self.redis:
            return False

        cache_key = make_cache_key("market", "price", token_pair, f"v{self.CACHE_VERSION}")
        ttl = ttl or self.TTL_PRICE

        try:
            # Add timestamp to cached data
            price_data["cached_at"] = datetime.utcnow().isoformat()

            with redis_operation_duration.labels(operation="set").time():
                await self.redis.setex(
                    cache_key,
                    ttl,
                    json.dumps(price_data)
                )
            return True

        except Exception as e:
            logger.error(f"Error caching price for {token_pair}: {e}")
            return False

    async def get_prices_batch(self, token_pairs: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get multiple prices in a single batch operation

        Args:
            token_pairs: List of token pairs

        Returns:
            Dictionary of token_pair -> price_data
        """
        if not self.redis or not token_pairs:
            return {}

        cache_keys = [
            make_cache_key("market", "price", pair, f"v{self.CACHE_VERSION}")
            for pair in token_pairs
        ]

        try:
            with redis_operation_duration.labels(operation="mget").time():
                cached_values = await self.redis.mget(cache_keys)

            results = {}
            for pair, cached in zip(token_pairs, cached_values):
                if cached:
                    cache_hits.labels(cache_type="price").inc()
                    results[pair] = json.loads(cached)
                else:
                    cache_misses.labels(cache_type="price").inc()

            return results

        except Exception as e:
            logger.error(f"Error getting batch prices: {e}")
            return {}

    # ========================================================================
    # Order Book Caching
    # ========================================================================

    async def get_orderbook(self, token_pair: str) -> Optional[Dict[str, Any]]:
        """
        Get cached order book

        Args:
            token_pair: Token pair (e.g., "SOL-USDC")

        Returns:
            Order book data or None if not cached
        """
        if not self.redis:
            return None

        cache_key = make_cache_key("market", "orderbook", token_pair, f"v{self.CACHE_VERSION}")

        try:
            with redis_operation_duration.labels(operation="get").time():
                cached = await self.redis.get(cache_key)

            if cached:
                cache_hits.labels(cache_type="orderbook").inc()
                return json.loads(cached)
            else:
                cache_misses.labels(cache_type="orderbook").inc()
                return None

        except Exception as e:
            logger.error(f"Error getting cached orderbook for {token_pair}: {e}")
            cache_misses.labels(cache_type="orderbook").inc()
            return None

    async def set_orderbook(
        self,
        token_pair: str,
        orderbook_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache order book

        Args:
            token_pair: Token pair (e.g., "SOL-USDC")
            orderbook_data: Order book data (bids, asks)
            ttl: Custom TTL in seconds (default: TTL_ORDERBOOK)

        Returns:
            True if cached successfully
        """
        if not self.redis:
            return False

        cache_key = make_cache_key("market", "orderbook", token_pair, f"v{self.CACHE_VERSION}")
        ttl = ttl or self.TTL_ORDERBOOK

        try:
            orderbook_data["cached_at"] = datetime.utcnow().isoformat()

            with redis_operation_duration.labels(operation="set").time():
                await self.redis.setex(
                    cache_key,
                    ttl,
                    json.dumps(orderbook_data)
                )
            return True

        except Exception as e:
            logger.error(f"Error caching orderbook for {token_pair}: {e}")
            return False

    # ========================================================================
    # OHLCV Data Caching
    # ========================================================================

    async def get_ohlcv(
        self,
        token_pair: str,
        timeframe: str,
        limit: int = 100
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached OHLCV (candlestick) data

        Args:
            token_pair: Token pair (e.g., "SOL-USDC")
            timeframe: Timeframe (e.g., "1m", "5m", "1h", "1d")
            limit: Number of candles

        Returns:
            List of OHLCV candles or None if not cached
        """
        if not self.redis:
            return None

        cache_key = make_cache_key(
            "market", "ohlcv", token_pair, timeframe, str(limit), f"v{self.CACHE_VERSION}"
        )

        try:
            with redis_operation_duration.labels(operation="get").time():
                cached = await self.redis.get(cache_key)

            if cached:
                cache_hits.labels(cache_type="ohlcv").inc()
                return json.loads(cached)
            else:
                cache_misses.labels(cache_type="ohlcv").inc()
                return None

        except Exception as e:
            logger.error(f"Error getting cached OHLCV for {token_pair}: {e}")
            cache_misses.labels(cache_type="ohlcv").inc()
            return None

    async def set_ohlcv(
        self,
        token_pair: str,
        timeframe: str,
        limit: int,
        ohlcv_data: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache OHLCV data

        Args:
            token_pair: Token pair
            timeframe: Timeframe
            limit: Number of candles
            ohlcv_data: OHLCV candle data
            ttl: Custom TTL in seconds (default: TTL_OHLCV)

        Returns:
            True if cached successfully
        """
        if not self.redis:
            return False

        cache_key = make_cache_key(
            "market", "ohlcv", token_pair, timeframe, str(limit), f"v{self.CACHE_VERSION}"
        )
        ttl = ttl or self.TTL_OHLCV

        try:
            with redis_operation_duration.labels(operation="set").time():
                await self.redis.setex(
                    cache_key,
                    ttl,
                    json.dumps({
                        "data": ohlcv_data,
                        "cached_at": datetime.utcnow().isoformat()
                    })
                )
            return True

        except Exception as e:
            logger.error(f"Error caching OHLCV for {token_pair}: {e}")
            return False

    # ========================================================================
    # Token Metadata Caching
    # ========================================================================

    async def get_token_metadata(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Get cached token metadata

        Args:
            token_address: Token contract address

        Returns:
            Token metadata or None if not cached
        """
        if not self.redis:
            return None

        cache_key = make_cache_key("market", "token", token_address, f"v{self.CACHE_VERSION}")

        try:
            with redis_operation_duration.labels(operation="get").time():
                cached = await self.redis.get(cache_key)

            if cached:
                cache_hits.labels(cache_type="token_metadata").inc()
                return json.loads(cached)
            else:
                cache_misses.labels(cache_type="token_metadata").inc()
                return None

        except Exception as e:
            logger.error(f"Error getting cached token metadata for {token_address}: {e}")
            cache_misses.labels(cache_type="token_metadata").inc()
            return None

    async def set_token_metadata(
        self,
        token_address: str,
        metadata: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache token metadata

        Args:
            token_address: Token contract address
            metadata: Token metadata (name, symbol, decimals, etc.)
            ttl: Custom TTL in seconds (default: TTL_TOKEN_METADATA)

        Returns:
            True if cached successfully
        """
        if not self.redis:
            return False

        cache_key = make_cache_key("market", "token", token_address, f"v{self.CACHE_VERSION}")
        ttl = ttl or self.TTL_TOKEN_METADATA

        try:
            metadata["cached_at"] = datetime.utcnow().isoformat()

            with redis_operation_duration.labels(operation="set").time():
                await self.redis.setex(
                    cache_key,
                    ttl,
                    json.dumps(metadata)
                )
            return True

        except Exception as e:
            logger.error(f"Error caching token metadata for {token_address}: {e}")
            return False

    # ========================================================================
    # Trade History Caching
    # ========================================================================

    async def get_trade_history(
        self,
        token_pair: str,
        limit: int = 100
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached trade history

        Args:
            token_pair: Token pair
            limit: Number of trades

        Returns:
            Trade history or None if not cached
        """
        if not self.redis:
            return None

        cache_key = make_cache_key(
            "market", "trades", token_pair, str(limit), f"v{self.CACHE_VERSION}"
        )

        try:
            with redis_operation_duration.labels(operation="get").time():
                cached = await self.redis.get(cache_key)

            if cached:
                cache_hits.labels(cache_type="trade_history").inc()
                return json.loads(cached)
            else:
                cache_misses.labels(cache_type="trade_history").inc()
                return None

        except Exception as e:
            logger.error(f"Error getting cached trade history for {token_pair}: {e}")
            cache_misses.labels(cache_type="trade_history").inc()
            return None

    async def set_trade_history(
        self,
        token_pair: str,
        limit: int,
        trades: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache trade history

        Args:
            token_pair: Token pair
            limit: Number of trades
            trades: Trade data
            ttl: Custom TTL in seconds (default: TTL_TRADE_HISTORY)

        Returns:
            True if cached successfully
        """
        if not self.redis:
            return False

        cache_key = make_cache_key(
            "market", "trades", token_pair, str(limit), f"v{self.CACHE_VERSION}"
        )
        ttl = ttl or self.TTL_TRADE_HISTORY

        try:
            with redis_operation_duration.labels(operation="set").time():
                await self.redis.setex(
                    cache_key,
                    ttl,
                    json.dumps({
                        "data": trades,
                        "cached_at": datetime.utcnow().isoformat()
                    })
                )
            return True

        except Exception as e:
            logger.error(f"Error caching trade history for {token_pair}: {e}")
            return False

    # ========================================================================
    # Cache Invalidation
    # ========================================================================

    async def invalidate_price(self, token_pair: str) -> bool:
        """Invalidate cached price for a token pair"""
        if not self.redis:
            return False

        cache_key = make_cache_key("market", "price", token_pair, f"v{self.CACHE_VERSION}")

        try:
            await self.redis.delete(cache_key)
            cache_invalidations.labels(cache_type="price").inc()
            return True
        except Exception as e:
            logger.error(f"Error invalidating price cache for {token_pair}: {e}")
            return False

    async def invalidate_orderbook(self, token_pair: str) -> bool:
        """Invalidate cached order book for a token pair"""
        if not self.redis:
            return False

        cache_key = make_cache_key("market", "orderbook", token_pair, f"v{self.CACHE_VERSION}")

        try:
            await self.redis.delete(cache_key)
            cache_invalidations.labels(cache_type="orderbook").inc()
            return True
        except Exception as e:
            logger.error(f"Error invalidating orderbook cache for {token_pair}: {e}")
            return False

    async def invalidate_all_for_pair(self, token_pair: str) -> int:
        """
        Invalidate all cached data for a token pair

        Args:
            token_pair: Token pair

        Returns:
            Number of keys invalidated
        """
        if not self.redis:
            return 0

        pattern = make_cache_key("market", "*", token_pair, "*")

        try:
            # Find all matching keys
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            # Delete all keys
            if keys:
                deleted = await self.redis.delete(*keys)
                cache_invalidations.labels(cache_type="all").inc()
                logger.info(f"Invalidated {deleted} cache keys for {token_pair}")
                return deleted
            return 0

        except Exception as e:
            logger.error(f"Error invalidating all cache for {token_pair}: {e}")
            return 0

    # ========================================================================
    # Cache Warming
    # ========================================================================

    async def warm_cache(
        self,
        popular_pairs: List[str],
        fetch_price_func,
        fetch_orderbook_func
    ) -> Dict[str, bool]:
        """
        Warm cache on startup with popular trading pairs

        Args:
            popular_pairs: List of popular token pairs
            fetch_price_func: Function to fetch price data
            fetch_orderbook_func: Function to fetch orderbook data

        Returns:
            Dictionary of pair -> success status
        """
        results = {}

        for pair in popular_pairs:
            try:
                # Fetch and cache price
                price_data = await fetch_price_func(pair)
                if price_data:
                    await self.set_price(pair, price_data)

                # Fetch and cache orderbook
                orderbook_data = await fetch_orderbook_func(pair)
                if orderbook_data:
                    await self.set_orderbook(pair, orderbook_data)

                results[pair] = True
                logger.info(f"Cache warmed for {pair}")

            except Exception as e:
                logger.error(f"Error warming cache for {pair}: {e}")
                results[pair] = False

        return results
