"""
Distributed Tracing with OpenTelemetry
Provides end-to-end request tracing
"""

import logging
from functools import wraps
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

logger = logging.getLogger(__name__)


def setup_tracing(
    service_name: str = "shivx",
    service_version: str = "2.0.0",
    environment: str = "local",
    otlp_endpoint: Optional[str] = None,
    enable_console_export: bool = False
):
    """
    Setup OpenTelemetry distributed tracing

    Args:
        service_name: Service name
        service_version: Service version
        environment: Environment (local, staging, production)
        otlp_endpoint: OTLP collector endpoint (e.g., http://localhost:4317)
        enable_console_export: Also export to console (for debugging)

    Returns:
        TracerProvider instance
    """
    # Create resource with service information
    resource = Resource.create({
        "service.name": service_name,
        "service.version": service_version,
        "deployment.environment": environment,
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add OTLP exporter if endpoint provided
    if otlp_endpoint:
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"✓ OTLP tracing enabled (endpoint: {otlp_endpoint})")
        except Exception as e:
            logger.error(f"Failed to setup OTLP exporter: {e}")

    # Add console exporter for debugging
    if enable_console_export:
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))
        logger.info("✓ Console tracing enabled")

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    logger.info("✓ Distributed tracing initialized")
    return provider


def instrument_fastapi_app(app):
    """
    Instrument FastAPI app with OpenTelemetry

    Args:
        app: FastAPI application instance
    """
    FastAPIInstrumentor.instrument_app(app)
    logger.info("✓ FastAPI instrumented for tracing")


def trace_function(name: Optional[str] = None, attributes: Optional[dict] = None):
    """
    Decorator to trace a function

    Usage:
        @trace_function("my_function", {"custom_attr": "value"})
        async def my_function():
            pass

    Args:
        name: Span name (defaults to function name)
        attributes: Custom attributes to add to span
    """
    def decorator(func):
        span_name = name or func.__name__
        tracer = trace.get_tracer(__name__)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name) as span:
                # Add custom attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Add function metadata
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, str(e))
                    )
                    span.record_exception(e)
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                try:
                    result = func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, str(e))
                    )
                    span.record_exception(e)
                    raise

        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def get_current_trace_id() -> Optional[str]:
    """Get current trace ID"""
    span = trace.get_current_span()
    if span:
        return format(span.get_span_context().trace_id, '032x')
    return None


def get_current_span_id() -> Optional[str]:
    """Get current span ID"""
    span = trace.get_current_span()
    if span:
        return format(span.get_span_context().span_id, '016x')
    return None
