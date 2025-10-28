"""
Cache Monitoring Service
Tracks cache metrics, hit rates, memory usage, and eviction rates
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

import redis.asyncio as aioredis
from prometheus_client import Counter, Gauge, Histogram, generate_latest

from app.cache import make_cache_key


logger = logging.getLogger(__name__)


# ============================================================================
# Prometheus Metrics
# ============================================================================

# Cache hit/miss metrics (already defined in other modules, aggregated here)
cache_hit_rate = Gauge(
    "cache_hit_rate_percentage",
    "Cache hit rate percentage",
    ["cache_type"]
)

cache_memory_usage = Gauge(
    "cache_memory_usage_bytes",
    "Cache memory usage in bytes",
    ["cache_type"]
)

cache_key_count = Gauge(
    "cache_key_count_total",
    "Total number of keys in cache",
    ["cache_type"]
)

cache_evictions = Counter(
    "cache_evictions_total",
    "Total number of cache evictions",
    ["cache_type"]
)

cache_ttl_histogram = Histogram(
    "cache_ttl_seconds",
    "Distribution of cache TTL values",
    ["cache_type"]
)


# ============================================================================
# Cache Monitor
# ============================================================================

class CacheMonitor:
    """
    Monitors cache performance and health

    Features:
    - Track cache hit rate metrics
    - Monitor cache memory usage
    - Track cache eviction rates
    - Prometheus metrics export
    - Alert on low hit rate (<70%)
    - Performance analytics
    """

    # Alert thresholds
    LOW_HIT_RATE_THRESHOLD = 0.70  # 70%
    HIGH_MEMORY_THRESHOLD = 0.90  # 90%

    def __init__(self, redis: Optional[aioredis.Redis]):
        self.redis = redis
        self._last_stats = {}
        self._alerts = []

    # ========================================================================
    # Cache Statistics
    # ========================================================================

    async def get_redis_info(self) -> Dict[str, Any]:
        """
        Get Redis server information

        Returns:
            Dictionary with Redis info
        """
        if not self.redis:
            return {"status": "unavailable"}

        try:
            info = await self.redis.info()

            stats = {
                "status": "healthy",
                "redis_version": info.get("redis_version", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "used_memory_peak": info.get("used_memory_peak", 0),
                "used_memory_peak_human": info.get("used_memory_peak_human", "0B"),
                "maxmemory": info.get("maxmemory", 0),
                "maxmemory_human": info.get("maxmemory_human", "0B"),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "evicted_keys": info.get("evicted_keys", 0),
                "expired_keys": info.get("expired_keys", 0),
                "total_system_memory": info.get("total_system_memory", 0),
                "used_memory_rss": info.get("used_memory_rss", 0),
            }

            # Calculate hit rate
            hits = stats["keyspace_hits"]
            misses = stats["keyspace_misses"]
            total_requests = hits + misses

            if total_requests > 0:
                hit_rate = (hits / total_requests) * 100
                stats["hit_rate_percentage"] = round(hit_rate, 2)
            else:
                stats["hit_rate_percentage"] = 0.0

            # Calculate memory usage percentage
            if stats["maxmemory"] > 0:
                memory_usage_pct = (stats["used_memory"] / stats["maxmemory"]) * 100
                stats["memory_usage_percentage"] = round(memory_usage_pct, 2)
            else:
                stats["memory_usage_percentage"] = 0.0

            # Update Prometheus metrics
            cache_memory_usage.labels(cache_type="redis_total").set(stats["used_memory"])

            return stats

        except Exception as e:
            logger.error(f"Error getting Redis info: {e}")
            return {"status": "error", "error": str(e)}

    async def get_cache_type_stats(self, cache_type: str) -> Dict[str, Any]:
        """
        Get statistics for a specific cache type

        Args:
            cache_type: Cache type (e.g., "market", "indicator", "ml")

        Returns:
            Dictionary with cache type stats
        """
        if not self.redis:
            return {"status": "unavailable"}

        pattern = make_cache_key(cache_type, "*")

        try:
            # Count keys
            key_count = 0
            total_ttl = 0
            ttls = []

            async for key in self.redis.scan_iter(match=pattern):
                key_count += 1

                # Get TTL
                ttl = await self.redis.ttl(key)
                if ttl > 0:
                    ttls.append(ttl)
                    total_ttl += ttl

            # Calculate average TTL
            avg_ttl = total_ttl / key_count if key_count > 0 else 0

            stats = {
                "cache_type": cache_type,
                "key_count": key_count,
                "avg_ttl_seconds": round(avg_ttl, 2),
                "min_ttl_seconds": min(ttls) if ttls else 0,
                "max_ttl_seconds": max(ttls) if ttls else 0,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Update Prometheus metrics
            cache_key_count.labels(cache_type=cache_type).set(key_count)
            if ttls:
                for ttl in ttls:
                    cache_ttl_histogram.labels(cache_type=cache_type).observe(ttl)

            return stats

        except Exception as e:
            logger.error(f"Error getting cache type stats for {cache_type}: {e}")
            return {"status": "error", "error": str(e)}

    async def get_all_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics

        Returns:
            Dictionary with all cache stats
        """
        cache_types = ["market", "indicator", "ml", "http", "session"]

        stats = {
            "redis": await self.get_redis_info(),
            "cache_types": {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        for cache_type in cache_types:
            stats["cache_types"][cache_type] = await self.get_cache_type_stats(cache_type)

        return stats

    # ========================================================================
    # Hit Rate Tracking
    # ========================================================================

    async def calculate_hit_rate(
        self,
        cache_type: str,
        time_window: int = 3600
    ) -> Dict[str, Any]:
        """
        Calculate hit rate for a cache type over a time window

        Args:
            cache_type: Cache type
            time_window: Time window in seconds

        Returns:
            Dictionary with hit rate stats
        """
        # Note: This requires tracking hits/misses in Redis
        # For now, we'll use Redis INFO stats

        redis_info = await self.get_redis_info()

        if redis_info.get("status") != "healthy":
            return {"status": "unavailable"}

        hits = redis_info.get("keyspace_hits", 0)
        misses = redis_info.get("keyspace_misses", 0)
        total_requests = hits + misses

        if total_requests > 0:
            hit_rate = (hits / total_requests) * 100
        else:
            hit_rate = 0.0

        # Update Prometheus metric
        cache_hit_rate.labels(cache_type=cache_type).set(hit_rate)

        # Check alert threshold
        if hit_rate < self.LOW_HIT_RATE_THRESHOLD * 100:
            alert = {
                "type": "low_hit_rate",
                "cache_type": cache_type,
                "hit_rate": hit_rate,
                "threshold": self.LOW_HIT_RATE_THRESHOLD * 100,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._alerts.append(alert)
            logger.warning(
                f"Low cache hit rate for {cache_type}: {hit_rate:.2f}% "
                f"(threshold: {self.LOW_HIT_RATE_THRESHOLD * 100}%)"
            )

        return {
            "cache_type": cache_type,
            "hits": hits,
            "misses": misses,
            "total_requests": total_requests,
            "hit_rate_percentage": round(hit_rate, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ========================================================================
    # Memory Monitoring
    # ========================================================================

    async def check_memory_usage(self) -> Dict[str, Any]:
        """
        Check cache memory usage and alert if high

        Returns:
            Dictionary with memory usage stats
        """
        redis_info = await self.get_redis_info()

        if redis_info.get("status") != "healthy":
            return {"status": "unavailable"}

        memory_usage_pct = redis_info.get("memory_usage_percentage", 0.0)

        # Check alert threshold
        if memory_usage_pct > self.HIGH_MEMORY_THRESHOLD * 100:
            alert = {
                "type": "high_memory_usage",
                "memory_usage_percentage": memory_usage_pct,
                "threshold": self.HIGH_MEMORY_THRESHOLD * 100,
                "used_memory": redis_info.get("used_memory"),
                "maxmemory": redis_info.get("maxmemory"),
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._alerts.append(alert)
            logger.warning(
                f"High cache memory usage: {memory_usage_pct:.2f}% "
                f"(threshold: {self.HIGH_MEMORY_THRESHOLD * 100}%)"
            )

        return {
            "used_memory": redis_info.get("used_memory"),
            "used_memory_human": redis_info.get("used_memory_human"),
            "maxmemory": redis_info.get("maxmemory"),
            "maxmemory_human": redis_info.get("maxmemory_human"),
            "memory_usage_percentage": memory_usage_pct,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ========================================================================
    # Eviction Tracking
    # ========================================================================

    async def track_evictions(self) -> Dict[str, Any]:
        """
        Track cache eviction rates

        Returns:
            Dictionary with eviction stats
        """
        redis_info = await self.get_redis_info()

        if redis_info.get("status") != "healthy":
            return {"status": "unavailable"}

        evicted_keys = redis_info.get("evicted_keys", 0)
        expired_keys = redis_info.get("expired_keys", 0)

        # Calculate eviction rate if we have previous stats
        eviction_rate = 0.0
        if "evicted_keys" in self._last_stats:
            time_diff = (datetime.utcnow() - self._last_stats["timestamp"]).total_seconds()
            if time_diff > 0:
                evicted_diff = evicted_keys - self._last_stats["evicted_keys"]
                eviction_rate = evicted_diff / time_diff

        # Update last stats
        self._last_stats["evicted_keys"] = evicted_keys
        self._last_stats["timestamp"] = datetime.utcnow()

        return {
            "evicted_keys": evicted_keys,
            "expired_keys": expired_keys,
            "eviction_rate_per_second": round(eviction_rate, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ========================================================================
    # Alerts
    # ========================================================================

    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent alerts

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of alerts
        """
        return self._alerts[-limit:]

    def clear_alerts(self):
        """Clear all alerts"""
        self._alerts = []

    # ========================================================================
    # Performance Analytics
    # ========================================================================

    async def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report

        Returns:
            Dictionary with performance report
        """
        redis_info = await self.get_redis_info()
        all_stats = await self.get_all_cache_stats()
        memory_usage = await self.check_memory_usage()
        evictions = await self.track_evictions()

        # Calculate overall metrics
        cache_types_stats = all_stats.get("cache_types", {})
        total_keys = sum(
            stats.get("key_count", 0)
            for stats in cache_types_stats.values()
        )

        report = {
            "summary": {
                "status": redis_info.get("status"),
                "hit_rate_percentage": redis_info.get("hit_rate_percentage", 0.0),
                "total_keys": total_keys,
                "memory_usage_percentage": memory_usage.get("memory_usage_percentage", 0.0),
                "ops_per_sec": redis_info.get("instantaneous_ops_per_sec", 0),
            },
            "redis": redis_info,
            "cache_types": cache_types_stats,
            "memory": memory_usage,
            "evictions": evictions,
            "alerts": self.get_alerts(10),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Add recommendations
        recommendations = []

        if redis_info.get("hit_rate_percentage", 0) < self.LOW_HIT_RATE_THRESHOLD * 100:
            recommendations.append({
                "type": "low_hit_rate",
                "message": "Cache hit rate is below 70%. Consider increasing TTLs or implementing cache warming.",
                "priority": "high"
            })

        if memory_usage.get("memory_usage_percentage", 0) > self.HIGH_MEMORY_THRESHOLD * 100:
            recommendations.append({
                "type": "high_memory",
                "message": "Memory usage is above 90%. Consider increasing maxmemory or implementing better eviction policies.",
                "priority": "high"
            })

        if evictions.get("eviction_rate_per_second", 0) > 10:
            recommendations.append({
                "type": "high_evictions",
                "message": "High eviction rate detected. Consider increasing cache size or reducing TTLs.",
                "priority": "medium"
            })

        report["recommendations"] = recommendations

        return report

    # ========================================================================
    # Prometheus Metrics Export
    # ========================================================================

    async def export_prometheus_metrics(self) -> str:
        """
        Export Prometheus metrics

        Returns:
            Prometheus metrics in text format
        """
        # Update metrics
        await self.get_all_cache_stats()

        # Generate and return metrics
        return generate_latest().decode("utf-8")

    # ========================================================================
    # Health Check
    # ========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform cache health check

        Returns:
            Health check result
        """
        redis_info = await self.get_redis_info()

        if redis_info.get("status") != "healthy":
            return {
                "status": "unhealthy",
                "message": "Redis is unavailable",
                "timestamp": datetime.utcnow().isoformat(),
            }

        hit_rate = redis_info.get("hit_rate_percentage", 0.0)
        memory_usage = redis_info.get("memory_usage_percentage", 0.0)

        # Determine overall health
        if hit_rate < 50 or memory_usage > 95:
            status = "degraded"
            message = "Cache is operating in degraded state"
        else:
            status = "healthy"
            message = "Cache is operating normally"

        return {
            "status": status,
            "message": message,
            "hit_rate_percentage": hit_rate,
            "memory_usage_percentage": memory_usage,
            "timestamp": datetime.utcnow().isoformat(),
        }


# ============================================================================
# Grafana Dashboard Template
# ============================================================================

GRAFANA_DASHBOARD_TEMPLATE = {
    "dashboard": {
        "title": "ShivX Cache Performance",
        "panels": [
            {
                "title": "Cache Hit Rate",
                "targets": [
                    {"expr": "cache_hit_rate_percentage"}
                ],
                "type": "graph"
            },
            {
                "title": "Cache Memory Usage",
                "targets": [
                    {"expr": "cache_memory_usage_bytes"}
                ],
                "type": "graph"
            },
            {
                "title": "Cache Operations per Second",
                "targets": [
                    {"expr": "rate(redis_operations_total[5m])"}
                ],
                "type": "graph"
            },
            {
                "title": "Cache Evictions",
                "targets": [
                    {"expr": "rate(cache_evictions_total[5m])"}
                ],
                "type": "graph"
            },
        ]
    }
}
