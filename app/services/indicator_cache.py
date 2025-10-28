"""
Technical Indicators Caching Service
Caches RSI, MACD, Bollinger Bands, sentiment scores, and correlation matrices
"""

import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import redis.asyncio as aioredis
from prometheus_client import Counter, Histogram

from app.cache import make_cache_key, redis_operation_duration, redis_pipeline


logger = logging.getLogger(__name__)


# ============================================================================
# Prometheus Metrics
# ============================================================================

indicator_cache_hits = Counter(
    "indicator_cache_hits_total",
    "Total number of indicator cache hits",
    ["indicator_type"]
)

indicator_cache_misses = Counter(
    "indicator_cache_misses_total",
    "Total number of indicator cache misses",
    ["indicator_type"]
)

indicator_cache_computations = Counter(
    "indicator_cache_computations_total",
    "Total number of indicator computations",
    ["indicator_type"]
)


# ============================================================================
# Technical Indicator Cache Service
# ============================================================================

class IndicatorCache:
    """
    Caches technical indicators with composite keys (token + timeframe + indicator)

    TTL Strategy:
    - RSI, MACD, Bollinger Bands: 1 minute
    - Sentiment scores: 5 minutes
    - Correlation matrices: 1 hour
    - Volume profiles: 30 minutes
    """

    # Cache TTLs (seconds)
    TTL_RSI = 60  # 1 minute
    TTL_MACD = 60  # 1 minute
    TTL_BOLLINGER = 60  # 1 minute
    TTL_SENTIMENT = 300  # 5 minutes
    TTL_CORRELATION = 3600  # 1 hour
    TTL_VOLUME_PROFILE = 1800  # 30 minutes
    TTL_MOVING_AVERAGE = 60  # 1 minute
    TTL_STOCHASTIC = 60  # 1 minute

    # Cache version
    CACHE_VERSION = 1

    def __init__(self, redis: Optional[aioredis.Redis] = None):
        self.redis = redis

    # ========================================================================
    # RSI (Relative Strength Index) Caching
    # ========================================================================

    async def get_rsi(
        self,
        token_pair: str,
        timeframe: str,
        period: int = 14
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached RSI value

        Args:
            token_pair: Token pair (e.g., "SOL-USDC")
            timeframe: Timeframe (e.g., "1m", "5m", "1h")
            period: RSI period (default: 14)

        Returns:
            RSI data or None if not cached
        """
        if not self.redis:
            return None

        cache_key = make_cache_key(
            "indicator", "rsi", token_pair, timeframe, str(period), f"v{self.CACHE_VERSION}"
        )

        try:
            with redis_operation_duration.labels(operation="get").time():
                cached = await self.redis.get(cache_key)

            if cached:
                indicator_cache_hits.labels(indicator_type="rsi").inc()
                return json.loads(cached)
            else:
                indicator_cache_misses.labels(indicator_type="rsi").inc()
                return None

        except Exception as e:
            logger.error(f"Error getting cached RSI: {e}")
            indicator_cache_misses.labels(indicator_type="rsi").inc()
            return None

    async def set_rsi(
        self,
        token_pair: str,
        timeframe: str,
        period: int,
        rsi_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache RSI value

        Args:
            token_pair: Token pair
            timeframe: Timeframe
            period: RSI period
            rsi_data: RSI data (value, signal, etc.)
            ttl: Custom TTL (default: TTL_RSI)

        Returns:
            True if cached successfully
        """
        if not self.redis:
            return False

        cache_key = make_cache_key(
            "indicator", "rsi", token_pair, timeframe, str(period), f"v{self.CACHE_VERSION}"
        )
        ttl = ttl or self.TTL_RSI

        try:
            rsi_data["cached_at"] = datetime.utcnow().isoformat()
            rsi_data["timeframe"] = timeframe
            rsi_data["period"] = period

            with redis_operation_duration.labels(operation="set").time():
                await self.redis.setex(cache_key, ttl, json.dumps(rsi_data))

            indicator_cache_computations.labels(indicator_type="rsi").inc()
            return True

        except Exception as e:
            logger.error(f"Error caching RSI: {e}")
            return False

    # ========================================================================
    # MACD (Moving Average Convergence Divergence) Caching
    # ========================================================================

    async def get_macd(
        self,
        token_pair: str,
        timeframe: str,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached MACD values

        Args:
            token_pair: Token pair
            timeframe: Timeframe
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            MACD data or None if not cached
        """
        if not self.redis:
            return None

        cache_key = make_cache_key(
            "indicator", "macd", token_pair, timeframe,
            f"{fast_period}_{slow_period}_{signal_period}", f"v{self.CACHE_VERSION}"
        )

        try:
            with redis_operation_duration.labels(operation="get").time():
                cached = await self.redis.get(cache_key)

            if cached:
                indicator_cache_hits.labels(indicator_type="macd").inc()
                return json.loads(cached)
            else:
                indicator_cache_misses.labels(indicator_type="macd").inc()
                return None

        except Exception as e:
            logger.error(f"Error getting cached MACD: {e}")
            indicator_cache_misses.labels(indicator_type="macd").inc()
            return None

    async def set_macd(
        self,
        token_pair: str,
        timeframe: str,
        fast_period: int,
        slow_period: int,
        signal_period: int,
        macd_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache MACD values

        Args:
            token_pair: Token pair
            timeframe: Timeframe
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            macd_data: MACD data (macd, signal, histogram)
            ttl: Custom TTL (default: TTL_MACD)

        Returns:
            True if cached successfully
        """
        if not self.redis:
            return False

        cache_key = make_cache_key(
            "indicator", "macd", token_pair, timeframe,
            f"{fast_period}_{slow_period}_{signal_period}", f"v{self.CACHE_VERSION}"
        )
        ttl = ttl or self.TTL_MACD

        try:
            macd_data["cached_at"] = datetime.utcnow().isoformat()
            macd_data["timeframe"] = timeframe
            macd_data["parameters"] = {
                "fast": fast_period,
                "slow": slow_period,
                "signal": signal_period
            }

            with redis_operation_duration.labels(operation="set").time():
                await self.redis.setex(cache_key, ttl, json.dumps(macd_data))

            indicator_cache_computations.labels(indicator_type="macd").inc()
            return True

        except Exception as e:
            logger.error(f"Error caching MACD: {e}")
            return False

    # ========================================================================
    # Bollinger Bands Caching
    # ========================================================================

    async def get_bollinger_bands(
        self,
        token_pair: str,
        timeframe: str,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached Bollinger Bands

        Args:
            token_pair: Token pair
            timeframe: Timeframe
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            Bollinger Bands data or None if not cached
        """
        if not self.redis:
            return None

        cache_key = make_cache_key(
            "indicator", "bb", token_pair, timeframe,
            f"{period}_{std_dev}", f"v{self.CACHE_VERSION}"
        )

        try:
            with redis_operation_duration.labels(operation="get").time():
                cached = await self.redis.get(cache_key)

            if cached:
                indicator_cache_hits.labels(indicator_type="bollinger").inc()
                return json.loads(cached)
            else:
                indicator_cache_misses.labels(indicator_type="bollinger").inc()
                return None

        except Exception as e:
            logger.error(f"Error getting cached Bollinger Bands: {e}")
            indicator_cache_misses.labels(indicator_type="bollinger").inc()
            return None

    async def set_bollinger_bands(
        self,
        token_pair: str,
        timeframe: str,
        period: int,
        std_dev: float,
        bb_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache Bollinger Bands

        Args:
            token_pair: Token pair
            timeframe: Timeframe
            period: Moving average period
            std_dev: Standard deviation multiplier
            bb_data: Bollinger Bands data (upper, middle, lower)
            ttl: Custom TTL (default: TTL_BOLLINGER)

        Returns:
            True if cached successfully
        """
        if not self.redis:
            return False

        cache_key = make_cache_key(
            "indicator", "bb", token_pair, timeframe,
            f"{period}_{std_dev}", f"v{self.CACHE_VERSION}"
        )
        ttl = ttl or self.TTL_BOLLINGER

        try:
            bb_data["cached_at"] = datetime.utcnow().isoformat()
            bb_data["timeframe"] = timeframe
            bb_data["parameters"] = {"period": period, "std_dev": std_dev}

            with redis_operation_duration.labels(operation="set").time():
                await self.redis.setex(cache_key, ttl, json.dumps(bb_data))

            indicator_cache_computations.labels(indicator_type="bollinger").inc()
            return True

        except Exception as e:
            logger.error(f"Error caching Bollinger Bands: {e}")
            return False

    # ========================================================================
    # Sentiment Score Caching
    # ========================================================================

    async def get_sentiment(
        self,
        token_pair: str,
        source: str = "all"
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached sentiment score

        Args:
            token_pair: Token pair
            source: Sentiment source (e.g., "twitter", "reddit", "all")

        Returns:
            Sentiment data or None if not cached
        """
        if not self.redis:
            return None

        cache_key = make_cache_key(
            "indicator", "sentiment", token_pair, source, f"v{self.CACHE_VERSION}"
        )

        try:
            with redis_operation_duration.labels(operation="get").time():
                cached = await self.redis.get(cache_key)

            if cached:
                indicator_cache_hits.labels(indicator_type="sentiment").inc()
                return json.loads(cached)
            else:
                indicator_cache_misses.labels(indicator_type="sentiment").inc()
                return None

        except Exception as e:
            logger.error(f"Error getting cached sentiment: {e}")
            indicator_cache_misses.labels(indicator_type="sentiment").inc()
            return None

    async def set_sentiment(
        self,
        token_pair: str,
        source: str,
        sentiment_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache sentiment score

        Args:
            token_pair: Token pair
            source: Sentiment source
            sentiment_data: Sentiment data (score, confidence, etc.)
            ttl: Custom TTL (default: TTL_SENTIMENT)

        Returns:
            True if cached successfully
        """
        if not self.redis:
            return False

        cache_key = make_cache_key(
            "indicator", "sentiment", token_pair, source, f"v{self.CACHE_VERSION}"
        )
        ttl = ttl or self.TTL_SENTIMENT

        try:
            sentiment_data["cached_at"] = datetime.utcnow().isoformat()
            sentiment_data["source"] = source

            with redis_operation_duration.labels(operation="set").time():
                await self.redis.setex(cache_key, ttl, json.dumps(sentiment_data))

            indicator_cache_computations.labels(indicator_type="sentiment").inc()
            return True

        except Exception as e:
            logger.error(f"Error caching sentiment: {e}")
            return False

    # ========================================================================
    # Correlation Matrix Caching
    # ========================================================================

    async def get_correlation_matrix(
        self,
        token_pairs: List[str],
        timeframe: str,
        period: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached correlation matrix

        Args:
            token_pairs: List of token pairs
            timeframe: Timeframe
            period: Correlation period (days)

        Returns:
            Correlation matrix or None if not cached
        """
        if not self.redis:
            return None

        # Sort pairs for consistent key
        sorted_pairs = "_".join(sorted(token_pairs))
        cache_key = make_cache_key(
            "indicator", "correlation", sorted_pairs, timeframe,
            str(period), f"v{self.CACHE_VERSION}"
        )

        try:
            with redis_operation_duration.labels(operation="get").time():
                cached = await self.redis.get(cache_key)

            if cached:
                indicator_cache_hits.labels(indicator_type="correlation").inc()
                return json.loads(cached)
            else:
                indicator_cache_misses.labels(indicator_type="correlation").inc()
                return None

        except Exception as e:
            logger.error(f"Error getting cached correlation matrix: {e}")
            indicator_cache_misses.labels(indicator_type="correlation").inc()
            return None

    async def set_correlation_matrix(
        self,
        token_pairs: List[str],
        timeframe: str,
        period: int,
        correlation_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache correlation matrix

        Args:
            token_pairs: List of token pairs
            timeframe: Timeframe
            period: Correlation period
            correlation_data: Correlation matrix data
            ttl: Custom TTL (default: TTL_CORRELATION)

        Returns:
            True if cached successfully
        """
        if not self.redis:
            return False

        sorted_pairs = "_".join(sorted(token_pairs))
        cache_key = make_cache_key(
            "indicator", "correlation", sorted_pairs, timeframe,
            str(period), f"v{self.CACHE_VERSION}"
        )
        ttl = ttl or self.TTL_CORRELATION

        try:
            correlation_data["cached_at"] = datetime.utcnow().isoformat()
            correlation_data["timeframe"] = timeframe
            correlation_data["period"] = period
            correlation_data["pairs"] = token_pairs

            with redis_operation_duration.labels(operation="set").time():
                await self.redis.setex(cache_key, ttl, json.dumps(correlation_data))

            indicator_cache_computations.labels(indicator_type="correlation").inc()
            return True

        except Exception as e:
            logger.error(f"Error caching correlation matrix: {e}")
            return False

    # ========================================================================
    # Batch Operations
    # ========================================================================

    async def get_indicators_batch(
        self,
        token_pair: str,
        timeframe: str,
        indicators: List[str]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get multiple indicators in a single batch operation

        Args:
            token_pair: Token pair
            timeframe: Timeframe
            indicators: List of indicator names (e.g., ["rsi", "macd", "bollinger"])

        Returns:
            Dictionary of indicator_name -> data
        """
        if not self.redis:
            return {ind: None for ind in indicators}

        results = {}

        try:
            # Build cache keys for all indicators
            cache_keys = []
            for indicator in indicators:
                if indicator == "rsi":
                    key = make_cache_key("indicator", "rsi", token_pair, timeframe, "14", f"v{self.CACHE_VERSION}")
                elif indicator == "macd":
                    key = make_cache_key("indicator", "macd", token_pair, timeframe, "12_26_9", f"v{self.CACHE_VERSION}")
                elif indicator == "bollinger":
                    key = make_cache_key("indicator", "bb", token_pair, timeframe, "20_2.0", f"v{self.CACHE_VERSION}")
                else:
                    key = None
                cache_keys.append(key)

            # Batch get
            with redis_operation_duration.labels(operation="mget").time():
                cached_values = await self.redis.mget([k for k in cache_keys if k])

            # Parse results
            for indicator, cached in zip(indicators, cached_values):
                if cached:
                    results[indicator] = json.loads(cached)
                    indicator_cache_hits.labels(indicator_type=indicator).inc()
                else:
                    results[indicator] = None
                    indicator_cache_misses.labels(indicator_type=indicator).inc()

            return results

        except Exception as e:
            logger.error(f"Error getting batch indicators: {e}")
            return {ind: None for ind in indicators}

    async def set_indicators_batch(
        self,
        token_pair: str,
        timeframe: str,
        indicators_data: Dict[str, Tuple[Dict[str, Any], int]]
    ) -> Dict[str, bool]:
        """
        Set multiple indicators in a single pipeline operation

        Args:
            token_pair: Token pair
            timeframe: Timeframe
            indicators_data: Dict of indicator_name -> (data, ttl)

        Returns:
            Dictionary of indicator_name -> success status
        """
        if not self.redis:
            return {ind: False for ind in indicators_data}

        results = {}

        try:
            async with redis_pipeline(self.redis) as pipe:
                if pipe:
                    for indicator_name, (data, ttl) in indicators_data.items():
                        data["cached_at"] = datetime.utcnow().isoformat()

                        if indicator_name == "rsi":
                            key = make_cache_key("indicator", "rsi", token_pair, timeframe, "14", f"v{self.CACHE_VERSION}")
                        elif indicator_name == "macd":
                            key = make_cache_key("indicator", "macd", token_pair, timeframe, "12_26_9", f"v{self.CACHE_VERSION}")
                        elif indicator_name == "bollinger":
                            key = make_cache_key("indicator", "bb", token_pair, timeframe, "20_2.0", f"v{self.CACHE_VERSION}")
                        else:
                            continue

                        pipe.setex(key, ttl, json.dumps(data))
                        indicator_cache_computations.labels(indicator_type=indicator_name).inc()

                    await pipe.execute()
                    results = {ind: True for ind in indicators_data}
                else:
                    results = {ind: False for ind in indicators_data}

        except Exception as e:
            logger.error(f"Error setting batch indicators: {e}")
            results = {ind: False for ind in indicators_data}

        return results

    # ========================================================================
    # Cache Precomputation
    # ========================================================================

    async def precompute_popular_pairs(
        self,
        popular_pairs: List[str],
        timeframes: List[str],
        compute_func
    ) -> Dict[str, Dict[str, bool]]:
        """
        Precompute indicators for popular trading pairs

        Args:
            popular_pairs: List of popular token pairs
            timeframes: List of timeframes to compute
            compute_func: Function to compute indicators

        Returns:
            Dictionary of pair -> timeframe -> success status
        """
        results = {}

        for pair in popular_pairs:
            results[pair] = {}
            for timeframe in timeframes:
                try:
                    # Compute all indicators
                    indicators = await compute_func(pair, timeframe)

                    # Cache them
                    if indicators:
                        await self.set_indicators_batch(pair, timeframe, {
                            "rsi": (indicators.get("rsi", {}), self.TTL_RSI),
                            "macd": (indicators.get("macd", {}), self.TTL_MACD),
                            "bollinger": (indicators.get("bollinger", {}), self.TTL_BOLLINGER),
                        })
                        results[pair][timeframe] = True
                    else:
                        results[pair][timeframe] = False

                except Exception as e:
                    logger.error(f"Error precomputing indicators for {pair} {timeframe}: {e}")
                    results[pair][timeframe] = False

        return results
