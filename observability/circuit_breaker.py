"""
Circuit Breaker Pattern Implementation
Protects external services from cascading failures
"""

import time
import asyncio
from enum import Enum
from typing import Callable, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerException(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation

    Protects services from cascading failures by:
    1. Monitoring failures
    2. Opening circuit after threshold
    3. Testing recovery periodically
    4. Closing circuit when service recovers

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, requests rejected immediately
    - HALF_OPEN: Testing if service recovered

    Usage:
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            success_threshold=2
        )

        @breaker
        async def call_external_api():
            # Your code here
            pass
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
        timeout: float = 30.0,
    ):
        """
        Initialize circuit breaker

        Args:
            name: Circuit breaker name (for logging/metrics)
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            success_threshold: Consecutive successes needed to close circuit
            timeout: Request timeout in seconds
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.timeout = timeout

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.opened_at: Optional[datetime] = None

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator for circuit breaker

        Usage:
            @circuit_breaker
            async def my_function():
                pass
        """
        if asyncio.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                return await self.call_async(func, *args, **kwargs)
        else:
            def wrapper(*args, **kwargs):
                return self.call(func, *args, **kwargs)

        return wrapper

    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function through circuit breaker

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerException: If circuit is open
            TimeoutError: If function exceeds timeout
        """
        # Check circuit state
        if not self.can_proceed():
            raise CircuitBreakerException(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Service unavailable for {self.recovery_timeout}s."
            )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.timeout
            )

            # Record success
            self.on_success()
            return result

        except asyncio.TimeoutError as e:
            logger.error(f"Circuit breaker '{self.name}': Timeout after {self.timeout}s")
            self.on_failure()
            raise TimeoutError(f"Request timeout after {self.timeout}s") from e

        except Exception as e:
            logger.error(f"Circuit breaker '{self.name}': Failure - {e}")
            self.on_failure()
            raise

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute sync function through circuit breaker

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerException: If circuit is open
        """
        if not self.can_proceed():
            raise CircuitBreakerException(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Service unavailable for {self.recovery_timeout}s."
            )

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result

        except Exception as e:
            logger.error(f"Circuit breaker '{self.name}': Failure - {e}")
            self.on_failure()
            raise

    def can_proceed(self) -> bool:
        """
        Check if request can proceed

        Returns:
            True if request should be allowed
        """
        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout elapsed
            if self.opened_at and \
               (datetime.now() - self.opened_at).total_seconds() >= self.recovery_timeout:
                # Try recovery
                logger.info(f"Circuit breaker '{self.name}': Attempting recovery (HALF_OPEN)")
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                return True
            return False

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Allow requests in half-open state to test recovery
            return True

        return False

    def on_success(self):
        """Record successful request"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            logger.info(
                f"Circuit breaker '{self.name}': Success in HALF_OPEN "
                f"({self.success_count}/{self.success_threshold})"
            )

            if self.success_count >= self.success_threshold:
                # Service recovered
                logger.info(f"Circuit breaker '{self.name}': CLOSED (service recovered)")
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.opened_at = None

        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            if self.failure_count > 0:
                self.failure_count = 0

    def on_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Failure during recovery - reopen circuit
            logger.warning(
                f"Circuit breaker '{self.name}': OPEN (recovery failed)"
            )
            self.state = CircuitBreakerState.OPEN
            self.opened_at = datetime.now()
            self.failure_count = 0
            self.success_count = 0

        elif self.state == CircuitBreakerState.CLOSED:
            # Check if threshold exceeded
            if self.failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker '{self.name}': OPEN "
                    f"(threshold {self.failure_threshold} exceeded)"
                )
                self.state = CircuitBreakerState.OPEN
                self.opened_at = datetime.now()
                self.success_count = 0

    def get_state(self) -> CircuitBreakerState:
        """Get current state"""
        return self.state

    def get_metrics(self) -> dict:
        """Get circuit breaker metrics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None
        }

    def reset(self):
        """Manually reset circuit breaker"""
        logger.info(f"Circuit breaker '{self.name}': Manual reset to CLOSED")
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.opened_at = None
