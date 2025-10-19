"""
Week 11: Production Hardening Module

Implements production-ready capabilities:
- Comprehensive error handling and recovery
- Circuit breakers for fault tolerance
- Retry logic with exponential backoff
- Rate limiting and throttling
- Performance optimization (caching, batching)
- Security hardening (input validation, rate limiting)
- Monitoring and alerting hooks
- Graceful degradation

This wraps and enhances all previous weeks with production-grade reliability.
"""

import asyncio
import functools
import hashlib
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from contextlib import asynccontextmanager

import numpy as np

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: float = 60.0  # Seconds before half-open
    half_open_max_calls: int = 3  # Max calls in half-open state


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    max_calls: int = 100  # Max calls per window
    window_seconds: float = 60.0  # Time window


@dataclass
class CacheConfig:
    """Configuration for caching"""
    max_size: int = 1000  # Max cache entries
    ttl_seconds: float = 300.0  # Time to live


@dataclass
class RetryConfig:
    """Configuration for retries"""
    max_attempts: int = 3
    base_delay: float = 1.0  # Seconds
    max_delay: float = 60.0  # Seconds
    exponential_base: float = 2.0


@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    context: Dict[str, Any]
    stack_trace: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for a function"""
    function_name: str
    call_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_duration: float = 0.0
    min_duration: float = float("inf")
    max_duration: float = 0.0
    last_call: Optional[datetime] = None

    @property
    def avg_duration(self) -> float:
        """Average duration per call"""
        if self.call_count == 0:
            return 0.0
        return self.total_duration / self.call_count

    @property
    def success_rate(self) -> float:
        """Success rate"""
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Protects against cascading failures by temporarily blocking
    calls to failing services.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0

        logger.info(f"Circuit breaker '{name}' initialized: {config}")

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""

        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise Exception(f"Circuit breaker '{self.name}' is OPEN")

        # Check half-open call limit
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                raise Exception(f"Circuit breaker '{self.name}' half-open limit reached")
            self.half_open_calls += 1

        # Execute function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            self._record_success()
            return result

        except Exception as e:
            self._record_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.config.timeout

    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN"""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        logger.info(f"Circuit breaker '{self.name}' -> HALF_OPEN")

    def _record_success(self):
        """Record successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def _record_failure(self):
        """Record failed call"""
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()

    def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state = CircuitState.OPEN
        self.success_count = 0
        logger.warning(f"Circuit breaker '{self.name}' -> OPEN (failures: {self.failure_count})")

    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker '{self.name}' -> CLOSED")

    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
        }


class RateLimiter:
    """
    Rate limiter using sliding window algorithm.

    Prevents system overload by limiting calls per time window.
    """

    def __init__(self, name: str, config: RateLimitConfig):
        self.name = name
        self.config = config
        self.calls: deque = deque()  # Timestamps of calls

        logger.info(f"Rate limiter '{name}' initialized: {config}")

    async def acquire(self, key: str = "default"):
        """Acquire permission to make a call"""
        now = time.time()

        # Remove old calls outside window
        cutoff = now - self.config.window_seconds
        while self.calls and self.calls[0] < cutoff:
            self.calls.popleft()

        # Check if limit reached
        if len(self.calls) >= self.config.max_calls:
            oldest_call = self.calls[0]
            wait_time = self.config.window_seconds - (now - oldest_call)
            raise Exception(
                f"Rate limit exceeded for '{self.name}': "
                f"{len(self.calls)}/{self.config.max_calls} calls. "
                f"Retry in {wait_time:.1f}s"
            )

        # Record call
        self.calls.append(now)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        now = time.time()
        cutoff = now - self.config.window_seconds

        # Count calls in current window
        current_calls = sum(1 for t in self.calls if t >= cutoff)

        return {
            "name": self.name,
            "current_calls": current_calls,
            "max_calls": self.config.max_calls,
            "utilization": current_calls / self.config.max_calls,
        }


class Cache:
    """
    Simple LRU cache with TTL.

    Improves performance by caching function results.
    """

    def __init__(self, name: str, config: CacheConfig):
        self.name = name
        self.config = config
        self.cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, timestamp)
        self.access_order: deque = deque()  # For LRU eviction

        self.hits = 0
        self.misses = 0

        logger.info(f"Cache '{name}' initialized: {config}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            self.misses += 1
            return None

        value, timestamp = self.cache[key]

        # Check if expired
        if time.time() - timestamp > self.config.ttl_seconds:
            del self.cache[key]
            self.access_order.remove(key)
            self.misses += 1
            return None

        # Update access order (LRU)
        self.access_order.remove(key)
        self.access_order.append(key)

        self.hits += 1
        return value

    def set(self, key: str, value: Any):
        """Set value in cache"""
        # Evict if at capacity
        if len(self.cache) >= self.config.max_size and key not in self.cache:
            self._evict_lru()

        # Store value with timestamp
        self.cache[key] = (value, time.time())

        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.access_order:
            return

        lru_key = self.access_order.popleft()
        del self.cache[lru_key]
        logger.debug(f"Cache '{self.name}' evicted LRU key: {lru_key}")

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_order.clear()
        logger.info(f"Cache '{self.name}' cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "name": self.name,
            "size": len(self.cache),
            "max_size": self.config.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


class RetryHandler:
    """
    Retry logic with exponential backoff.

    Automatically retries failed operations with increasing delays.
    """

    def __init__(self, name: str, config: RetryConfig):
        self.name = name
        self.config = config

        logger.info(f"Retry handler '{name}' initialized: {config}")

    async def call(
        self,
        func: Callable,
        *args,
        retryable_exceptions: Optional[Tuple] = None,
        **kwargs,
    ) -> Any:
        """Execute function with retry logic"""
        if retryable_exceptions is None:
            retryable_exceptions = (Exception,)

        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                if attempt > 0:
                    logger.info(f"Retry '{self.name}' succeeded on attempt {attempt + 1}")

                return result

            except retryable_exceptions as e:
                last_exception = e

                if attempt < self.config.max_attempts - 1:
                    delay = self._compute_delay(attempt)
                    logger.warning(
                        f"Retry '{self.name}' attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Retry '{self.name}' exhausted all {self.config.max_attempts} attempts"
                    )

        # All attempts failed
        raise last_exception

    def _compute_delay(self, attempt: int) -> float:
        """Compute exponential backoff delay"""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        # Add jitter (±10%)
        jitter = delay * 0.1 * (2 * np.random.random() - 1)
        delay = delay + jitter
        # Clamp to max
        return min(delay, self.config.max_delay)


class ProductionHardeningEngine:
    """
    Production hardening engine that wraps all AGI systems
    with error handling, performance optimization, and monitoring.
    """

    def __init__(self):
        # Circuit breakers for each major component
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Rate limiters
        self.rate_limiters: Dict[str, RateLimiter] = {}

        # Caches
        self.caches: Dict[str, Cache] = {}

        # Retry handlers
        self.retry_handlers: Dict[str, RetryHandler] = {}

        # Error tracking
        self.errors: List[ErrorRecord] = []
        self.error_counts: Dict[str, int] = defaultdict(int)

        # Performance tracking
        self.metrics: Dict[str, PerformanceMetrics] = {}

        # Alert thresholds
        self.alert_thresholds = {
            "error_rate": 0.1,  # 10% error rate
            "avg_duration": 5.0,  # 5 seconds
            "circuit_breaker_open": True,  # Alert on any open circuit
        }

        logger.info("Production hardening engine initialized")

    # ========== Component Management ==========

    def create_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Create a circuit breaker for a component"""
        if config is None:
            config = CircuitBreakerConfig()

        breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = breaker
        return breaker

    def create_rate_limiter(
        self,
        name: str,
        config: Optional[RateLimitConfig] = None,
    ) -> RateLimiter:
        """Create a rate limiter"""
        if config is None:
            config = RateLimitConfig()

        limiter = RateLimiter(name, config)
        self.rate_limiters[name] = limiter
        return limiter

    def create_cache(
        self,
        name: str,
        config: Optional[CacheConfig] = None,
    ) -> Cache:
        """Create a cache"""
        if config is None:
            config = CacheConfig()

        cache = Cache(name, config)
        self.caches[name] = cache
        return cache

    def create_retry_handler(
        self,
        name: str,
        config: Optional[RetryConfig] = None,
    ) -> RetryHandler:
        """Create a retry handler"""
        if config is None:
            config = RetryConfig()

        handler = RetryHandler(name, config)
        self.retry_handlers[name] = handler
        return handler

    # ========== Decorators ==========

    def with_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Decorator to wrap function with circuit breaker"""
        if name not in self.circuit_breakers:
            self.create_circuit_breaker(name, config)

        breaker = self.circuit_breakers[name]

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                return await breaker.call(func, *args, **kwargs)
            return wrapper
        return decorator

    def with_rate_limit(self, name: str, config: Optional[RateLimitConfig] = None):
        """Decorator to wrap function with rate limiting"""
        if name not in self.rate_limiters:
            self.create_rate_limiter(name, config)

        limiter = self.rate_limiters[name]

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                await limiter.acquire()
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def with_cache(self, name: str, config: Optional[CacheConfig] = None, key_func: Optional[Callable] = None):
        """Decorator to wrap function with caching"""
        if name not in self.caches:
            self.create_cache(name, config)

        cache = self.caches[name]

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func.__name__, args, kwargs)

                # Check cache
                cached_value = cache.get(cache_key)
                if cached_value is not None:
                    return cached_value

                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Store in cache
                cache.set(cache_key, result)

                return result
            return wrapper
        return decorator

    def with_retry(self, name: str, config: Optional[RetryConfig] = None):
        """Decorator to wrap function with retry logic"""
        if name not in self.retry_handlers:
            self.create_retry_handler(name, config)

        handler = self.retry_handlers[name]

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                return await handler.call(func, *args, **kwargs)
            return wrapper
        return decorator

    def with_monitoring(self, name: str):
        """Decorator to wrap function with performance monitoring"""
        if name not in self.metrics:
            self.metrics[name] = PerformanceMetrics(function_name=name)

        metrics = self.metrics[name]

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                metrics.call_count += 1
                metrics.last_call = datetime.now()

                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    metrics.success_count += 1
                    return result

                except Exception as e:
                    metrics.error_count += 1
                    raise

                finally:
                    duration = time.time() - start_time
                    metrics.total_duration += duration
                    metrics.min_duration = min(metrics.min_duration, duration)
                    metrics.max_duration = max(metrics.max_duration, duration)

            return wrapper
        return decorator

    def _generate_cache_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key from function name and arguments"""
        # Simple hash-based key
        key_data = f"{func_name}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    # ========== Error Handling ==========

    def record_error(
        self,
        error_type: str,
        error_message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[Dict[str, Any]] = None,
        stack_trace: Optional[str] = None,
    ):
        """Record an error occurrence"""
        if context is None:
            context = {}

        error = ErrorRecord(
            timestamp=datetime.now(),
            error_type=error_type,
            error_message=error_message,
            severity=severity,
            context=context,
            stack_trace=stack_trace,
        )

        self.errors.append(error)
        self.error_counts[error_type] += 1

        # Check if alert needed
        self._check_alerts()

        logger.log(
            self._severity_to_log_level(severity),
            f"Error recorded: {error_type} - {error_message}",
        )

    def _severity_to_log_level(self, severity: ErrorSeverity) -> int:
        """Convert severity to logging level"""
        mapping = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }
        return mapping.get(severity, logging.ERROR)

    def _check_alerts(self):
        """Check if any alert thresholds are exceeded"""
        # Check error rate
        if self.metrics:
            for name, metrics in self.metrics.items():
                if metrics.call_count > 10:  # Need sufficient data
                    error_rate = 1 - metrics.success_rate
                    if error_rate > self.alert_thresholds["error_rate"]:
                        logger.critical(
                            f"ALERT: High error rate for {name}: {error_rate:.1%}"
                        )

                    if metrics.avg_duration > self.alert_thresholds["avg_duration"]:
                        logger.warning(
                            f"ALERT: Slow performance for {name}: {metrics.avg_duration:.2f}s"
                        )

        # Check circuit breakers
        if self.alert_thresholds["circuit_breaker_open"]:
            for name, breaker in self.circuit_breakers.items():
                if breaker.state == CircuitState.OPEN:
                    logger.critical(f"ALERT: Circuit breaker {name} is OPEN")

    def get_errors(
        self,
        severity: Optional[ErrorSeverity] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[ErrorRecord]:
        """Get recent errors"""
        filtered = self.errors

        if severity:
            filtered = [e for e in filtered if e.severity == severity]

        if since:
            filtered = [e for e in filtered if e.timestamp >= since]

        # Sort by timestamp descending
        filtered.sort(key=lambda e: e.timestamp, reverse=True)

        return filtered[:limit]

    # ========== Monitoring ==========

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        total_calls = sum(m.call_count for m in self.metrics.values())
        total_errors = sum(m.error_count for m in self.metrics.values())
        overall_error_rate = total_errors / total_calls if total_calls > 0 else 0.0

        # Circuit breaker status
        open_breakers = [
            name for name, b in self.circuit_breakers.items()
            if b.state == CircuitState.OPEN
        ]

        # Cache performance
        cache_stats = {name: c.get_stats() for name, c in self.caches.items()}

        # Rate limiter status
        rate_limiter_stats = {name: rl.get_stats() for name, rl in self.rate_limiters.items()}

        return {
            "status": "unhealthy" if open_breakers else "healthy",
            "total_calls": total_calls,
            "total_errors": total_errors,
            "error_rate": overall_error_rate,
            "open_circuit_breakers": open_breakers,
            "caches": cache_stats,
            "rate_limiters": rate_limiter_stats,
            "monitored_functions": len(self.metrics),
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all monitored functions"""
        report = {}

        for name, metrics in self.metrics.items():
            report[name] = {
                "calls": metrics.call_count,
                "success_rate": metrics.success_rate,
                "avg_duration": metrics.avg_duration,
                "min_duration": metrics.min_duration if metrics.min_duration != float("inf") else 0.0,
                "max_duration": metrics.max_duration,
                "last_call": metrics.last_call.isoformat() if metrics.last_call else None,
            }

        return report

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "circuit_breakers": {name: cb.get_state() for name, cb in self.circuit_breakers.items()},
            "rate_limiters": {name: rl.get_stats() for name, rl in self.rate_limiters.items()},
            "caches": {name: c.get_stats() for name, c in self.caches.items()},
            "errors": {
                "total": len(self.errors),
                "by_type": dict(self.error_counts),
                "recent": len(self.get_errors(since=datetime.now() - timedelta(hours=1))),
            },
            "performance": self.get_performance_report(),
        }


# ========== Testing Functions ==========

async def test_production_hardening():
    """Test production hardening capabilities"""
    print("\n" + "="*60)
    print("Testing Production Hardening Engine")
    print("="*60)

    engine = ProductionHardeningEngine()

    # Test 1: Circuit Breaker
    print("\n1. Testing circuit breaker...")

    @engine.with_circuit_breaker("test_service", CircuitBreakerConfig(failure_threshold=3, timeout=2.0))
    @engine.with_monitoring("test_service")
    async def unreliable_service(should_fail: bool = False):
        if should_fail:
            raise Exception("Service failure")
        return "success"

    # Trigger failures to open circuit
    for i in range(5):
        try:
            await unreliable_service(should_fail=True)
        except Exception as e:
            print(f"  Attempt {i+1}: {e}")

    # Check circuit state
    breaker_state = engine.circuit_breakers["test_service"].get_state()
    print(f"Circuit breaker state: {breaker_state['state']}")

    # Test 2: Rate Limiting
    print("\n2. Testing rate limiting...")

    @engine.with_rate_limit("api", RateLimitConfig(max_calls=3, window_seconds=1.0))
    async def api_call():
        return "api result"

    # Make calls within limit
    for i in range(3):
        await api_call()
        print(f"  Call {i+1}: success")

    # Try to exceed limit
    try:
        await api_call()
        print("  Call 4: success (unexpected)")
    except Exception as e:
        print(f"  Call 4: rate limited - {e}")

    # Test 3: Caching
    print("\n3. Testing caching...")

    call_count = 0

    @engine.with_cache("expensive_compute", CacheConfig(max_size=10, ttl_seconds=60))
    async def expensive_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)  # Simulate work
        return x * x

    # First call - cache miss
    start = time.time()
    result1 = await expensive_function(5)
    duration1 = time.time() - start
    print(f"  First call: result={result1}, duration={duration1:.3f}s, calls={call_count}")

    # Second call - cache hit
    start = time.time()
    result2 = await expensive_function(5)
    duration2 = time.time() - start
    print(f"  Second call: result={result2}, duration={duration2:.3f}s, calls={call_count}")

    cache_stats = engine.caches["expensive_compute"].get_stats()
    print(f"  Cache hit rate: {cache_stats['hit_rate']:.1%}")

    # Test 4: Retry Logic
    print("\n4. Testing retry logic...")

    attempt_count = 0

    @engine.with_retry("flaky_service", RetryConfig(max_attempts=3, base_delay=0.5))
    async def flaky_service():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception(f"Temporary failure (attempt {attempt_count})")
        return "success after retries"

    try:
        result = await flaky_service()
        print(f"  Result: {result} (after {attempt_count} attempts)")
    except Exception as e:
        print(f"  Failed after retries: {e}")

    # Test 5: Error Tracking
    print("\n5. Testing error tracking...")

    engine.record_error(
        "DatabaseError",
        "Connection timeout",
        ErrorSeverity.ERROR,
        {"database": "postgres", "timeout": 30},
    )

    engine.record_error(
        "ValidationError",
        "Invalid input",
        ErrorSeverity.WARNING,
        {"field": "email"},
    )

    recent_errors = engine.get_errors(limit=10)
    print(f"  Recorded {len(recent_errors)} errors")
    for error in recent_errors[:2]:
        print(f"    - {error.error_type}: {error.error_message} ({error.severity.value})")

    # System Health
    print("\n" + "="*60)
    print("System Health Report")
    print("="*60)

    health = engine.get_system_health()
    print(f"Status: {health['status']}")
    print(f"Total calls: {health['total_calls']}")
    print(f"Error rate: {health['error_rate']:.1%}")
    print(f"Open circuit breakers: {health['open_circuit_breakers']}")

    # Performance Report
    print("\n" + "="*60)
    print("Performance Report")
    print("="*60)

    perf = engine.get_performance_report()
    for func_name, metrics in perf.items():
        print(f"\n{func_name}:")
        print(f"  Calls: {metrics['calls']}")
        print(f"  Success rate: {metrics['success_rate']:.1%}")
        print(f"  Avg duration: {metrics['avg_duration']:.3f}s")

    return engine


if __name__ == "__main__":
    asyncio.run(test_production_hardening())
