from __future__ import annotations
"""
Lightweight metrics wrapper for ShivX.
Exposes Prometheus counters and a tiny API that avoids importing
prometheus_client everywhere.
"""
from typing import Optional
from prometheus_client import Counter, Histogram, CollectorRegistry, REGISTRY, generate_latest, CONTENT_TYPE_LATEST, Gauge
import os

# Use the default global registry so existing /metrics endpoint (if any)
# automatically sees these metrics.
_registry: CollectorRegistry = REGISTRY

def get_registry() -> CollectorRegistry:
    """Get the metrics registry for external use."""
    return _registry

# ---- Build info metric -----------------------------------------------------
BUILD_INFO = Gauge(
    "shivx_build_info",
    "Build metadata",
    labelnames=("version", "git", "env"),
    registry=_registry,
)

# Initialize build info metric
try:
    BUILD_INFO.labels(
        version=os.getenv("SHIVX_VERSION", "dev"),
        git=os.getenv("SHIVX_GIT_SHA", "unknown"),
        env=os.getenv("SHIVX_ENV", "local"),
    ).set(1)
except Exception:
    pass

# ---- Policy metrics ---------------------------------------------------------
# decision: "allow" | "warn" | "deny"
# action:   e.g. "subprocess.exec", "fs.write", "net.http", etc.
POLICY_DECISIONS = Counter(
    "shivx_policy_decisions_total",
    "Count of policy decisions by action and decision",
    labelnames=("action", "decision"),
    registry=_registry,
)

# Testpack results
POLICY_TESTS = Counter(
    "shivx_policy_test_results_total",
    "Count of policy testpack results by status",
    labelnames=("status",),  # "pass" | "fail"
    registry=_registry,
)

# Latency (seconds) for policy evaluations
POLICY_EVAL_LATENCY = Histogram(
    "shivx_policy_evaluate_seconds",
    "Policy evaluation latency (seconds) by action",
    labelnames=("action",),
    registry=_registry,
)

# Errors during evaluation
POLICY_EVAL_ERRORS = Counter(
    "shivx_policy_evaluate_errors_total",
    "Count of exceptions in policy evaluation by action",
    labelnames=("action",),
    registry=_registry,
)

# CLI command latency (seconds)
CLI_COMMAND_LATENCY = Histogram(
    "shivx_cli_command_latency_seconds",
    "CLI command execution latency (seconds) by command",
    labelnames=("cmd",),
    registry=_registry,
)

# Log suppression counter
LOG_SUPPRESSED = Counter(
    "shivx_log_suppressed_total",
    "Count of suppressed log messages by component",
    labelnames=("component",),
    registry=_registry,
)

def inc_policy_decision(action: str, decision: str) -> None:
    """Increment the policy decision counter."""
    try:
        POLICY_DECISIONS.labels(action=action, decision=decision).inc()
    except Exception:
        # Metrics must never crash business logic
        pass

def inc_policy_test(status: str) -> None:
    """Increment the policy test result counter ('pass'|'fail')."""
    try:
        POLICY_TESTS.labels(status=status).inc()
    except Exception:
        pass

def record_cli_latency(cmd: str, duration: float) -> None:
    """Record CLI command execution latency."""
    try:
        CLI_COMMAND_LATENCY.labels(cmd=cmd).observe(duration)
    except Exception:
        pass

def inc_log_suppressed(component: str) -> None:
    """Increment the log suppression counter."""
    try:
        LOG_SUPPRESSED.labels(component=component).inc()
    except Exception:
        pass

# --- Feature usage counter ----------------------------------------------------
FEATURE_USAGE = Counter(
    "shivx_feature_usage_total",
    "Count of feature usage by feature and enabled status",
    labelnames=("feature", "enabled"),
    registry=_registry,
)

def inc_feature_usage(feature: str, enabled: bool) -> None:
    try:
        FEATURE_USAGE.labels(feature=feature, enabled=str(bool(enabled)).lower()).inc()
    except Exception:
        pass

# Optional: expose plaintext metrics payload if you want to wire into FastAPI
def prometheus_payload() -> tuple[bytes, str]:
    return generate_latest(_registry), CONTENT_TYPE_LATEST
