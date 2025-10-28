"""
Prometheus Metrics Collection
Comprehensive metrics for trading, ML, and system performance
"""

from typing import Optional
from functools import lru_cache

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
    CollectorRegistry,
    make_asgi_app,
)


class MetricsCollector:
    """
    Centralized metrics collector for ShivX
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector

        Args:
            registry: Prometheus registry (creates new if None)
        """
        self.registry = registry or CollectorRegistry()

        # ====================================================================
        # HTTP Metrics
        # ====================================================================

        self.http_requests_total = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=self.registry
        )

        self.http_request_duration = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"],
            registry=self.registry
        )

        self.http_request_size = Summary(
            "http_request_size_bytes",
            "HTTP request size",
            ["method", "endpoint"],
            registry=self.registry
        )

        self.http_response_size = Summary(
            "http_response_size_bytes",
            "HTTP response size",
            ["method", "endpoint"],
            registry=self.registry
        )

        # ====================================================================
        # Trading Metrics
        # ====================================================================

        self.trades_total = Counter(
            "trades_total",
            "Total trades executed",
            ["token", "action", "strategy"],
            registry=self.registry
        )

        self.trade_pnl = Histogram(
            "trade_pnl_usd",
            "Trade profit/loss in USD",
            ["token", "strategy"],
            buckets=[-1000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 1000, 5000],
            registry=self.registry
        )

        self.position_size = Gauge(
            "position_size_usd",
            "Current position size in USD",
            ["token"],
            registry=self.registry
        )

        self.portfolio_value = Gauge(
            "portfolio_value_usd",
            "Total portfolio value in USD",
            registry=self.registry
        )

        self.trading_signals = Counter(
            "trading_signals_total",
            "Trading signals generated",
            ["token", "action", "strategy", "executed"],
            registry=self.registry
        )

        # ====================================================================
        # ML Model Metrics
        # ====================================================================

        self.ml_predictions = Counter(
            "ml_predictions_total",
            "ML model predictions",
            ["model_id", "model_type"],
            registry=self.registry
        )

        self.ml_prediction_latency = Histogram(
            "ml_prediction_latency_seconds",
            "ML prediction latency",
            ["model_id"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )

        self.ml_model_accuracy = Gauge(
            "ml_model_accuracy",
            "ML model accuracy",
            ["model_id", "model_type"],
            registry=self.registry
        )

        self.ml_training_epochs = Gauge(
            "ml_training_epochs",
            "ML training epochs completed",
            ["job_id", "model_name"],
            registry=self.registry
        )

        self.ml_training_loss = Gauge(
            "ml_training_loss",
            "ML training loss",
            ["job_id", "model_name"],
            registry=self.registry
        )

        # ====================================================================
        # Security Metrics
        # ====================================================================

        self.auth_attempts = Counter(
            "auth_attempts_total",
            "Authentication attempts",
            ["method", "success"],
            registry=self.registry
        )

        self.api_key_usage = Counter(
            "api_key_usage_total",
            "API key usage",
            ["key_name", "endpoint"],
            registry=self.registry
        )

        self.rate_limit_hits = Counter(
            "rate_limit_hits_total",
            "Rate limit exceeded events",
            ["endpoint", "ip"],
            registry=self.registry
        )

        self.security_events = Counter(
            "security_events_total",
            "Security events",
            ["event_type", "severity"],
            registry=self.registry
        )

        # ====================================================================
        # System Metrics
        # ====================================================================

        self.system_cpu_usage = Gauge(
            "system_cpu_usage_percent",
            "System CPU usage percentage",
            registry=self.registry
        )

        self.system_memory_usage = Gauge(
            "system_memory_usage_bytes",
            "System memory usage in bytes",
            registry=self.registry
        )

        self.system_disk_usage = Gauge(
            "system_disk_usage_bytes",
            "System disk usage in bytes",
            registry=self.registry
        )

        self.active_connections = Gauge(
            "active_connections",
            "Number of active connections",
            ["type"],  # db, redis, websocket
            registry=self.registry
        )

        # ====================================================================
        # External Service Metrics
        # ====================================================================

        self.external_api_calls = Counter(
            "external_api_calls_total",
            "External API calls",
            ["service", "endpoint", "status"],
            registry=self.registry
        )

        self.external_api_latency = Histogram(
            "external_api_latency_seconds",
            "External API latency",
            ["service", "endpoint"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )

        self.circuit_breaker_state = Gauge(
            "circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half-open)",
            ["service"],
            registry=self.registry
        )

        # ====================================================================
        # Application Info
        # ====================================================================

        self.app_info = Info(
            "shivx_app",
            "ShivX application information",
            registry=self.registry
        )

    def set_app_info(self, version: str, env: str, git_sha: str):
        """Set application information"""
        self.app_info.info({
            "version": version,
            "environment": env,
            "git_sha": git_sha
        })

    def update_system_metrics(self):
        """Update system resource metrics"""
        import psutil

        # CPU
        self.system_cpu_usage.set(psutil.cpu_percent(interval=0.1))

        # Memory
        mem = psutil.virtual_memory()
        self.system_memory_usage.set(mem.used)

        # Disk
        disk = psutil.disk_usage('/')
        self.system_disk_usage.set(disk.used)


@lru_cache()
def get_metrics_collector() -> MetricsCollector:
    """Get singleton metrics collector"""
    return MetricsCollector()


def get_metrics_app():
    """Get Prometheus metrics ASGI app"""
    collector = get_metrics_collector()
    return make_asgi_app(registry=collector.registry)
