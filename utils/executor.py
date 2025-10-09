"""
Bounded Executor for ShivX with Performance Controls
"""

import threading
import time
import queue
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RejectPolicy(Enum):
    """Rejection policy for when queue is full."""
    DROP = "drop"           # Drop task silently
    BLOCK = "block"         # Block until space available
    EXCEPTION = "exception" # Raise exception


@dataclass
class ExecutorMetrics:
    """Metrics for executor performance."""
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_rejected: int = 0
    tasks_queued: int = 0
    queue_size: int = 0
    active_threads: int = 0
    total_execution_time: float = 0.0
    last_metrics_reset: float = field(default_factory=time.time)
    
    def reset(self):
        """Reset metrics."""
        self.tasks_submitted = 0
        self.tasks_completed = 0
        self.tasks_rejected = 0
        self.tasks_queued = 0
        self.queue_size = 0
        self.active_threads = 0
        self.total_execution_time = 0.0
        self.last_metrics_reset = time.time()


class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    """Thread pool executor with bounded queue and performance controls."""
    
    def __init__(
        self,
        max_workers: int = 4,
        queue_size: int = 50,
        reject_policy: RejectPolicy = RejectPolicy.DROP,
        thread_name_prefix: str = "shivx-worker",
        enable_metrics: bool = False
    ):
        """
        Initialize bounded executor.
        
        Args:
            max_workers: Maximum number of worker threads
            queue_size: Maximum size of task queue
            reject_policy: What to do when queue is full
            thread_name_prefix: Prefix for worker thread names
            enable_metrics: Whether to collect performance metrics
        """
        # Create bounded queue
        self._task_queue = queue.Queue(maxsize=queue_size)
        self._reject_policy = reject_policy
        self._enable_metrics = enable_metrics
        
        # Initialize metrics if enabled
        if self._enable_metrics:
            self._metrics = ExecutorMetrics()
        else:
            self._metrics = None
        
        # Call parent constructor with our queue
        super().__init__(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )
        
        # Track active threads
        self._active_threads = 0
        self._active_threads_lock = threading.Lock()
        
        logger.info(f"Bounded executor initialized: {max_workers} workers, {queue_size} queue size, {reject_policy.value} policy")
    
    def submit(self, fn: Callable, *args, **kwargs) -> Optional[Future]:
        """
        Submit a task to the executor.
        
        Args:
            fn: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Future object if task accepted, None if rejected
        """
        try:
            # Check if we can accept the task
            if self._reject_policy == RejectPolicy.BLOCK:
                # Block until space available
                future = super().submit(fn, *args, **kwargs)
            elif self._reject_policy == RejectPolicy.DROP:
                # Try to submit without blocking
                if self._task_queue.full():
                    self._record_rejection()
                    return None
                future = super().submit(fn, *args, **kwargs)
            elif self._reject_policy == RejectPolicy.EXCEPTION:
                # Raise exception if queue full
                if self._task_queue.full():
                    raise queue.Full("Task queue is full")
                future = super().submit(fn, *args, **kwargs)
            else:
                raise ValueError(f"Unknown reject policy: {self._reject_policy}")
            
            # Record submission
            self._record_submission()
            
            # Add completion callback for metrics
            if self._enable_metrics:
                future.add_done_callback(self._on_task_completion)
            
            return future
            
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            self._record_rejection()
            return None
    
    def _record_submission(self):
        """Record task submission."""
        if self._enable_metrics:
            with threading.Lock():
                self._metrics.tasks_submitted += 1
                self._metrics.queue_size = self._task_queue.qsize()
                self._metrics.tasks_queued += 1
    
    def _record_rejection(self):
        """Record task rejection."""
        if self._enable_metrics:
            with threading.Lock():
                self._metrics.tasks_rejected += 1
        
        logger.warning(f"Task rejected due to {self._reject_policy.value} policy")
    
    def _on_task_completion(self, future: Future):
        """Callback when task completes."""
        if not self._enable_metrics:
            return
        
        try:
            # Record completion
            with threading.Lock():
                self._metrics.tasks_completed += 1
                self._metrics.queue_size = self._task_queue.qsize()
                self._metrics.tasks_queued -= 1
            
            # Record execution time if available
            if hasattr(future, '_start_time'):
                execution_time = time.time() - future._start_time
                with threading.Lock():
                    self._metrics.total_execution_time += execution_time
                    
        except Exception as e:
            logger.error(f"Error in task completion callback: {e}")
    
    def _adjust_thread_count(self):
        """Adjust thread count based on queue size."""
        if not self._enable_metrics:
            return
        
        with self._active_threads_lock:
            current_active = self._active_threads
            queue_size = self._task_queue.qsize()
            
            # Update active thread count
            self._metrics.active_threads = current_active
            
            # Log if queue is getting full
            if queue_size > self._task_queue.maxsize * 0.8:
                logger.warning(f"Task queue is {queue_size}/{self._task_queue.maxsize} full")
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current performance metrics."""
        if not self._enable_metrics:
            return None
        
        with threading.Lock():
            metrics = {
                'tasks_submitted': self._metrics.tasks_submitted,
                'tasks_completed': self._metrics.tasks_completed,
                'tasks_rejected': self._metrics.tasks_rejected,
                'tasks_queued': self._metrics.tasks_queued,
                'queue_size': self._task_queue.qsize(),
                'queue_capacity': self._task_queue.maxsize,
                'active_threads': self._metrics.active_threads,
                'max_workers': self._max_workers,
                'reject_policy': self._reject_policy.value,
                'uptime_seconds': time.time() - self._metrics.last_metrics_reset
            }
            
            # Calculate averages
            if self._metrics.tasks_completed > 0:
                metrics['avg_execution_time'] = (
                    self._metrics.total_execution_time / self._metrics.tasks_completed
                )
            else:
                metrics['avg_execution_time'] = 0.0
            
            # Calculate queue utilization
            metrics['queue_utilization'] = (
                self._task_queue.qsize() / self._task_queue.maxsize
            )
            
            return metrics
    
    def reset_metrics(self):
        """Reset performance metrics."""
        if self._enable_metrics:
            with threading.Lock():
                self._metrics.reset()
            logger.info("Executor metrics reset")
    
    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        """Shutdown the executor."""
        logger.info("Shutting down bounded executor")
        super().shutdown(wait, cancel_futures=cancel_futures)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


class PerformanceExecutor:
    """High-level performance executor with automatic configuration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance executor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Get settings
        self._load_settings()
        
        # Create executor if enabled
        self.executor = None
        if self.enabled:
            self.executor = BoundedThreadPoolExecutor(
                max_workers=self.max_workers,
                queue_size=self.queue_size,
                reject_policy=self.reject_policy,
                enable_metrics=self.enable_metrics
            )
            logger.info(f"Performance executor enabled: {self.max_workers} workers, {self.queue_size} queue")
        else:
            logger.info("Performance executor disabled")
    
    def _load_settings(self):
        """Load executor settings from configuration."""
        try:
            from config.settings import get_settings
            settings = get_settings()
            
            if settings.get('features', {}).get('performance_plus', {}).get('enabled', False):
                performance_config = settings.get('performance', {})
                
                self.enabled = True
                self.max_workers = performance_config.get('max_workers', 4)
                self.queue_size = performance_config.get('queue_size', 50)
                self.reject_policy = RejectPolicy(
                    performance_config.get('reject_policy', 'drop')
                )
                self.enable_metrics = performance_config.get('enable_metrics', True)
            else:
                self.enabled = False
                self.max_workers = 4
                self.queue_size = 50
                self.reject_policy = RejectPolicy.DROP
                self.enable_metrics = False
                
        except Exception as e:
            logger.warning(f"Failed to load performance settings: {e}")
            self.enabled = False
            self.max_workers = 4
            self.queue_size = 50
            self.reject_policy = RejectPolicy.DROP
            self.enable_metrics = False
    
    def submit(self, fn: Callable, *args, **kwargs) -> Optional[Future]:
        """Submit task to executor."""
        if not self.enabled or not self.executor:
            # Fallback to direct execution
            return self._direct_execute(fn, *args, **kwargs)
        
        return self.executor.submit(fn, *args, **kwargs)
    
    def _direct_execute(self, fn: Callable, *args, **kwargs) -> Future:
        """Execute task directly (fallback)."""
        # Create a simple future
        future = Future()
        
        def execute():
            try:
                result = fn(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        
        # Run in background thread
        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        
        return future
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get executor metrics."""
        if self.executor:
            return self.executor.get_metrics()
        return None
    
    def shutdown(self):
        """Shutdown the executor."""
        if self.executor:
            self.executor.shutdown()


# Global performance executor instance
_performance_executor = None


def get_performance_executor() -> PerformanceExecutor:
    """Get global performance executor instance."""
    global _performance_executor
    if _performance_executor is None:
        _performance_executor = PerformanceExecutor()
    return _performance_executor


def submit_task(fn: Callable, *args, **kwargs) -> Optional[Future]:
    """Submit task to performance executor."""
    executor = get_performance_executor()
    return executor.submit(fn, *args, **kwargs)


def get_executor_metrics() -> Optional[Dict[str, Any]]:
    """Get performance executor metrics."""
    executor = get_performance_executor()
    return executor.get_metrics()


def shutdown_executor():
    """Shutdown performance executor."""
    global _performance_executor
    if _performance_executor:
        _performance_executor.shutdown()
        _performance_executor = None
