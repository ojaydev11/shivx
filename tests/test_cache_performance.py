"""
Cache Performance Tests
Tests cache hit rates, latency, load handling, and failure scenarios
"""

import asyncio
import time
import statistics
from typing import Dict, Any, List

import pytest
import redis.asyncio as aioredis

from app.cache import initialize_redis, close_redis, get_redis_manager
from app.services.market_cache import MarketDataCache
from app.services.indicator_cache import IndicatorCache
from app.services.ml_cache import MLPredictionCache
from app.services.session_cache import SessionManager
from app.services.cache_monitor import CacheMonitor
from config.settings import get_settings


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def redis_client():
    """Initialize Redis client for testing"""
    settings = get_settings()
    manager = await initialize_redis(settings)
    client = await manager.get_client()

    yield client

    await close_redis()


@pytest.fixture
async def market_cache(redis_client):
    """Market data cache instance"""
    return MarketDataCache(redis_client)


@pytest.fixture
async def indicator_cache(redis_client):
    """Indicator cache instance"""
    return IndicatorCache(redis_client)


@pytest.fixture
async def ml_cache(redis_client):
    """ML prediction cache instance"""
    return MLPredictionCache(redis_client)


@pytest.fixture
async def session_manager(redis_client):
    """Session manager instance"""
    return SessionManager(redis_client)


@pytest.fixture
async def cache_monitor(redis_client):
    """Cache monitor instance"""
    return CacheMonitor(redis_client)


# ============================================================================
# Performance Test Helpers
# ============================================================================

class PerformanceMetrics:
    """Track performance metrics"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.latencies = []
        self.errors = 0

    def record_hit(self, latency: float):
        """Record cache hit"""
        self.hits += 1
        self.latencies.append(latency)

    def record_miss(self, latency: float):
        """Record cache miss"""
        self.misses += 1
        self.latencies.append(latency)

    def record_error(self):
        """Record error"""
        self.errors += 1

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        """Calculate average latency in milliseconds"""
        return statistics.mean(self.latencies) * 1000 if self.latencies else 0.0

    @property
    def p95_latency(self) -> float:
        """Calculate 95th percentile latency in milliseconds"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[index] * 1000

    @property
    def p99_latency(self) -> float:
        """Calculate 99th percentile latency in milliseconds"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[index] * 1000

    def summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate_percentage": round(self.hit_rate, 2),
            "avg_latency_ms": round(self.avg_latency, 2),
            "p95_latency_ms": round(self.p95_latency, 2),
            "p99_latency_ms": round(self.p99_latency, 2),
            "total_requests": self.hits + self.misses,
        }


# ============================================================================
# Test 1: Cache Hit Rate (Target >80%)
# ============================================================================

@pytest.mark.asyncio
async def test_cache_hit_rate(market_cache):
    """Test cache hit rate with repeated requests"""

    metrics = PerformanceMetrics()
    token_pairs = ["SOL-USDC", "BTC-USDC", "ETH-USDC"]
    num_iterations = 100

    # Warm cache
    for pair in token_pairs:
        price_data = {"price": 100.0, "volume": 1000}
        await market_cache.set_price(pair, price_data)

    # Test hit rate with repeated requests
    for _ in range(num_iterations):
        for pair in token_pairs:
            start_time = time.time()
            cached_price = await market_cache.get_price(pair)
            latency = time.time() - start_time

            if cached_price:
                metrics.record_hit(latency)
            else:
                metrics.record_miss(latency)

    summary = metrics.summary()
    print(f"\n=== Cache Hit Rate Test ===")
    print(f"Hit Rate: {summary['hit_rate_percentage']}%")
    print(f"Hits: {summary['hits']}")
    print(f"Misses: {summary['misses']}")
    print(f"Average Latency: {summary['avg_latency_ms']}ms")

    # Assert hit rate > 80%
    assert summary["hit_rate_percentage"] > 80.0, \
        f"Cache hit rate {summary['hit_rate_percentage']}% is below 80%"


# ============================================================================
# Test 2: Cache Latency (Target <5ms)
# ============================================================================

@pytest.mark.asyncio
async def test_cache_latency(market_cache):
    """Test cache operation latency"""

    token_pair = "SOL-USDC"
    num_operations = 1000

    set_latencies = []
    get_latencies = []

    # Test set latency
    for i in range(num_operations):
        price_data = {"price": 100.0 + i, "volume": 1000}

        start_time = time.time()
        await market_cache.set_price(token_pair, price_data)
        set_latency = (time.time() - start_time) * 1000  # Convert to ms
        set_latencies.append(set_latency)

    # Test get latency
    for i in range(num_operations):
        start_time = time.time()
        await market_cache.get_price(token_pair)
        get_latency = (time.time() - start_time) * 1000  # Convert to ms
        get_latencies.append(get_latency)

    avg_set_latency = statistics.mean(set_latencies)
    avg_get_latency = statistics.mean(get_latencies)
    p95_set_latency = sorted(set_latencies)[int(len(set_latencies) * 0.95)]
    p95_get_latency = sorted(get_latencies)[int(len(get_latencies) * 0.95)]

    print(f"\n=== Cache Latency Test ===")
    print(f"Average SET latency: {avg_set_latency:.2f}ms")
    print(f"Average GET latency: {avg_get_latency:.2f}ms")
    print(f"P95 SET latency: {p95_set_latency:.2f}ms")
    print(f"P95 GET latency: {p95_get_latency:.2f}ms")

    # Assert average latency < 5ms
    assert avg_get_latency < 5.0, \
        f"Average GET latency {avg_get_latency:.2f}ms exceeds 5ms target"


# ============================================================================
# Test 3: Cache Under Load (1000 req/s)
# ============================================================================

@pytest.mark.asyncio
async def test_cache_under_load(market_cache):
    """Test cache performance under high load"""

    token_pairs = [f"TOKEN{i}-USDC" for i in range(10)]
    target_rps = 1000  # requests per second
    duration_seconds = 5

    # Warm cache
    for pair in token_pairs:
        price_data = {"price": 100.0, "volume": 1000}
        await market_cache.set_price(pair, price_data)

    metrics = PerformanceMetrics()
    start_time = time.time()
    request_count = 0

    async def make_request():
        """Make a single cache request"""
        nonlocal request_count
        pair = token_pairs[request_count % len(token_pairs)]
        request_count += 1

        req_start = time.time()
        try:
            result = await market_cache.get_price(pair)
            latency = time.time() - req_start

            if result:
                metrics.record_hit(latency)
            else:
                metrics.record_miss(latency)
        except Exception:
            metrics.record_error()

    # Generate load
    while time.time() - start_time < duration_seconds:
        batch_size = target_rps // 10  # 10 batches per second
        tasks = [make_request() for _ in range(batch_size)]
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.1)  # 100ms between batches

    elapsed = time.time() - start_time
    actual_rps = request_count / elapsed
    summary = metrics.summary()

    print(f"\n=== Cache Load Test ===")
    print(f"Duration: {elapsed:.2f}s")
    print(f"Total Requests: {request_count}")
    print(f"Target RPS: {target_rps}")
    print(f"Actual RPS: {actual_rps:.2f}")
    print(f"Hit Rate: {summary['hit_rate_percentage']}%")
    print(f"Average Latency: {summary['avg_latency_ms']}ms")
    print(f"P95 Latency: {summary['p95_latency_ms']}ms")
    print(f"Errors: {summary['errors']}")

    # Assert performance targets
    assert actual_rps >= target_rps * 0.8, \
        f"Actual RPS {actual_rps:.2f} is less than 80% of target {target_rps}"
    assert summary["p95_latency_ms"] < 10.0, \
        f"P95 latency {summary['p95_latency_ms']}ms exceeds 10ms under load"
    assert summary["errors"] == 0, \
        f"Encountered {summary['errors']} errors under load"


# ============================================================================
# Test 4: Performance with vs without Cache
# ============================================================================

@pytest.mark.asyncio
async def test_cache_performance_comparison():
    """Compare performance with and without cache"""

    # Simulate expensive operation
    async def expensive_operation(token_pair: str) -> Dict[str, Any]:
        """Simulate expensive database/API call"""
        await asyncio.sleep(0.05)  # 50ms delay
        return {"price": 100.0, "volume": 1000, "token_pair": token_pair}

    token_pair = "SOL-USDC"
    num_requests = 100

    # Test WITHOUT cache
    start_time = time.time()
    for _ in range(num_requests):
        await expensive_operation(token_pair)
    no_cache_time = time.time() - start_time

    # Test WITH cache
    settings = get_settings()
    manager = await initialize_redis(settings)
    redis_client = await manager.get_client()
    market_cache = MarketDataCache(redis_client)

    # Warm cache
    data = await expensive_operation(token_pair)
    await market_cache.set_price(token_pair, data)

    start_time = time.time()
    for _ in range(num_requests):
        cached = await market_cache.get_price(token_pair)
        if not cached:
            data = await expensive_operation(token_pair)
            await market_cache.set_price(token_pair, data)
    with_cache_time = time.time() - start_time

    await close_redis()

    improvement = ((no_cache_time - with_cache_time) / no_cache_time) * 100

    print(f"\n=== Cache Performance Comparison ===")
    print(f"Time WITHOUT cache: {no_cache_time:.2f}s")
    print(f"Time WITH cache: {with_cache_time:.2f}s")
    print(f"Performance improvement: {improvement:.2f}%")

    # Assert significant improvement
    assert improvement > 90.0, \
        f"Cache improvement {improvement:.2f}% is less than expected (>90%)"


# ============================================================================
# Test 5: Cache Failure Scenarios (Graceful Degradation)
# ============================================================================

@pytest.mark.asyncio
async def test_cache_failure_graceful_degradation():
    """Test system works when Redis is unavailable"""

    # Test with None redis client (simulating Redis down)
    market_cache = MarketDataCache(redis=None)

    token_pair = "SOL-USDC"
    price_data = {"price": 100.0, "volume": 1000}

    # These should not raise exceptions
    result = await market_cache.set_price(token_pair, price_data)
    assert result is False, "Set should return False when Redis is unavailable"

    cached = await market_cache.get_price(token_pair)
    assert cached is None, "Get should return None when Redis is unavailable"

    print(f"\n=== Cache Failure Test ===")
    print("✓ System operates correctly when Redis is unavailable")
    print("✓ Graceful degradation working as expected")


# ============================================================================
# Test 6: TTL Expiration
# ============================================================================

@pytest.mark.asyncio
async def test_cache_ttl_expiration(market_cache):
    """Test cache entries expire correctly"""

    token_pair = "SOL-USDC"
    price_data = {"price": 100.0, "volume": 1000}

    # Set with short TTL
    await market_cache.set_price(token_pair, price_data, ttl=1)

    # Should be cached immediately
    cached = await market_cache.get_price(token_pair)
    assert cached is not None, "Cache should contain data immediately after set"

    # Wait for expiration
    await asyncio.sleep(2)

    # Should be expired
    cached = await market_cache.get_price(token_pair)
    assert cached is None, "Cache should be expired after TTL"

    print(f"\n=== TTL Expiration Test ===")
    print("✓ Cache entries expire correctly")


# ============================================================================
# Test 7: Concurrent Access
# ============================================================================

@pytest.mark.asyncio
async def test_cache_concurrent_access(market_cache):
    """Test cache handles concurrent access correctly"""

    token_pair = "SOL-USDC"
    price_data = {"price": 100.0, "volume": 1000}
    num_concurrent = 100

    # Warm cache
    await market_cache.set_price(token_pair, price_data)

    # Concurrent reads
    tasks = [market_cache.get_price(token_pair) for _ in range(num_concurrent)]
    results = await asyncio.gather(*tasks)

    # All should succeed
    assert all(r is not None for r in results), "All concurrent reads should succeed"

    # Concurrent writes
    tasks = [
        market_cache.set_price(token_pair, {**price_data, "price": 100.0 + i})
        for i in range(num_concurrent)
    ]
    results = await asyncio.gather(*tasks)

    # All should succeed
    assert all(r is True for r in results), "All concurrent writes should succeed"

    print(f"\n=== Concurrent Access Test ===")
    print(f"✓ {num_concurrent} concurrent reads succeeded")
    print(f"✓ {num_concurrent} concurrent writes succeeded")


# ============================================================================
# Test 8: Cache Memory Efficiency
# ============================================================================

@pytest.mark.asyncio
async def test_cache_memory_efficiency(cache_monitor):
    """Test cache memory usage is reasonable"""

    # Get initial memory
    redis_info = await cache_monitor.get_redis_info()

    if redis_info.get("status") != "healthy":
        pytest.skip("Redis not available for memory test")

    initial_memory = redis_info.get("used_memory", 0)

    print(f"\n=== Cache Memory Efficiency Test ===")
    print(f"Initial memory: {redis_info.get('used_memory_human', 'N/A')}")
    print(f"Memory usage: {redis_info.get('memory_usage_percentage', 0)}%")

    # Memory should be reasonable
    memory_usage_pct = redis_info.get("memory_usage_percentage", 0)
    assert memory_usage_pct < 90.0, \
        f"Memory usage {memory_usage_pct}% is too high (>90%)"


# ============================================================================
# Test 9: Cache Invalidation Performance
# ============================================================================

@pytest.mark.asyncio
async def test_cache_invalidation_performance(market_cache):
    """Test cache invalidation is fast"""

    token_pairs = [f"TOKEN{i}-USDC" for i in range(100)]

    # Populate cache
    for pair in token_pairs:
        await market_cache.set_price(pair, {"price": 100.0, "volume": 1000})

    # Time invalidation
    start_time = time.time()
    for pair in token_pairs:
        await market_cache.invalidate_price(pair)
    invalidation_time = time.time() - start_time

    avg_invalidation = (invalidation_time / len(token_pairs)) * 1000  # ms

    print(f"\n=== Cache Invalidation Performance ===")
    print(f"Total invalidation time: {invalidation_time:.3f}s")
    print(f"Average per key: {avg_invalidation:.2f}ms")

    # Invalidation should be fast
    assert avg_invalidation < 10.0, \
        f"Average invalidation time {avg_invalidation:.2f}ms exceeds 10ms"


# ============================================================================
# Test 10: Batch Operations Performance
# ============================================================================

@pytest.mark.asyncio
async def test_batch_operations_performance(market_cache):
    """Test batch operations are faster than individual operations"""

    token_pairs = [f"TOKEN{i}-USDC" for i in range(50)]

    # Populate cache
    for pair in token_pairs:
        await market_cache.set_price(pair, {"price": 100.0, "volume": 1000})

    # Time individual operations
    start_time = time.time()
    for pair in token_pairs:
        await market_cache.get_price(pair)
    individual_time = time.time() - start_time

    # Time batch operation
    start_time = time.time()
    await market_cache.get_prices_batch(token_pairs)
    batch_time = time.time() - start_time

    speedup = individual_time / batch_time

    print(f"\n=== Batch Operations Performance ===")
    print(f"Individual operations: {individual_time:.3f}s")
    print(f"Batch operation: {batch_time:.3f}s")
    print(f"Speedup: {speedup:.2f}x")

    # Batch should be significantly faster
    assert speedup > 2.0, \
        f"Batch operation speedup {speedup:.2f}x is less than expected (>2x)"


# ============================================================================
# Run All Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
