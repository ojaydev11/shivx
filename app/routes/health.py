"""
Health Check API Routes
Provides liveness and readiness endpoints for Kubernetes/Docker health checks
"""

import logging
from fastapi import APIRouter
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("/live")
async def live():
    """Liveness check - returns OK if the process is alive"""
    return {"ok": True}


@router.get("/status")
async def status():
    """Alias for liveness check"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@router.get("/ready")
async def ready():
    """
    Readiness check - returns OK if the service is ready to accept requests

    Checks:
    - Database connectivity (if applicable)
    - Required services availability
    - System resources
    """
    try:
        # Get readiness status from services
        from app.services import readiness

        r = await readiness.check_readiness()

        # Fix: Handle case where 'components' key might not exist
        components = r.get("components", {})

        # Check if all components are ready
        all_ready = True
        failing_components = []
        warnings = []

        for name, c in components.items():
            if not c.get("ready", False):
                all_ready = False
                failing_components.append(name)
            if c.get("warning"):
                warnings.append({name: c.get("warning")})

        if all_ready:
            response = {
                "ready": True,
                "status": "ok",
                "components": components,
                "timestamp": datetime.now().isoformat()
            }
            if warnings:
                response["warnings"] = warnings
            return response
        else:
            return {
                "ready": False,
                "status": "degraded",
                "components": components,
                "failing": failing_components,
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Readiness check failed: {e}", exc_info=True)
        # Return degraded status instead of 500 error
        return {
            "ready": False,
            "status": "degraded",
            "error": str(e),
            "components": {},
            "timestamp": datetime.now().isoformat()
        }


@router.get("/metrics")
async def metrics():
    """
    Health metrics endpoint for Prometheus scraping

    Returns basic health metrics in a format Prometheus can understand
    """
    try:
        from app.services import readiness
        import psutil

        r = await readiness.check_readiness()
        components = r.get("components", {})

        # Generate Prometheus-compatible metrics
        lines = [
            "# HELP health_ready Whether the service is ready",
            "# TYPE health_ready gauge",
            f"health_ready {int(r.get('ready', False))}",
            "",
            "# HELP health_component_ready Component readiness status",
            "# TYPE health_component_ready gauge",
        ]

        for name, comp in components.items():
            lines.append(f'health_component_ready{{component="{name}"}} {int(comp.get("ready", False))}')

        # Add system metrics
        lines.extend([
            "",
            "# HELP system_memory_usage_percent Memory usage percentage",
            "# TYPE system_memory_usage_percent gauge",
            f"system_memory_usage_percent {psutil.virtual_memory().percent}",
            "",
            "# HELP system_disk_usage_percent Disk usage percentage",
            "# TYPE system_disk_usage_percent gauge",
            f"system_disk_usage_percent {psutil.disk_usage('/').percent}",
        ])

        return "\n".join(lines) + "\n"

    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}", exc_info=True)
        return "# Error generating metrics\n"
