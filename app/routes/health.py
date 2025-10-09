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

        for name, c in components.items():
            if not c.get("ready", False):
                all_ready = False
                failing_components.append(name)

        if all_ready:
            return {
                "ready": True,
                "status": "ok",
                "components": components,
                "timestamp": datetime.now().isoformat()
            }
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
