"""
Readiness Check Service
Checks if the application and its dependencies are ready to serve traffic
"""

import logging
import asyncio
import psutil
import os
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


async def check_disk_space() -> Dict[str, Any]:
    """Check available disk space"""
    try:
        disk = psutil.disk_usage('/')
        percent_free = (disk.free / disk.total) * 100

        if percent_free < 10:
            return {
                "ready": False,
                "message": f"Low disk space: {percent_free:.1f}% free",
                "details": {
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent_free": percent_free
                }
            }
        elif percent_free < 20:
            return {
                "ready": True,
                "message": f"Disk space adequate: {percent_free:.1f}% free",
                "warning": "Approaching low disk space threshold",
                "details": {
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent_free": percent_free
                }
            }
        else:
            return {
                "ready": True,
                "message": f"Disk space healthy: {percent_free:.1f}% free",
                "details": {
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent_free": percent_free
                }
            }
    except Exception as e:
        logger.error(f"Disk space check failed: {e}")
        return {
            "ready": False,
            "message": f"Disk space check failed: {str(e)}"
        }


async def check_memory() -> Dict[str, Any]:
    """Check available memory"""
    try:
        memory = psutil.virtual_memory()
        percent_used = memory.percent

        if percent_used > 90:
            return {
                "ready": False,
                "message": f"Critical memory usage: {percent_used}%",
                "details": {
                    "total_mb": memory.total / (1024**2),
                    "available_mb": memory.available / (1024**2),
                    "percent_used": percent_used
                }
            }
        elif percent_used > 80:
            return {
                "ready": True,
                "message": f"High memory usage: {percent_used}%",
                "warning": "Memory usage is high",
                "details": {
                    "total_mb": memory.total / (1024**2),
                    "available_mb": memory.available / (1024**2),
                    "percent_used": percent_used
                }
            }
        else:
            return {
                "ready": True,
                "message": f"Memory healthy: {percent_used}% used",
                "details": {
                    "total_mb": memory.total / (1024**2),
                    "available_mb": memory.available / (1024**2),
                    "percent_used": percent_used
                }
            }
    except Exception as e:
        logger.error(f"Memory check failed: {e}")
        return {
            "ready": False,
            "message": f"Memory check failed: {str(e)}"
        }


async def check_database() -> Dict[str, Any]:
    """Check database connectivity"""
    try:
        # Import here to avoid circular dependencies
        from sqlalchemy import create_engine, text

        db_url = os.getenv("SHIVX_DATABASE_URL", "")
        if not db_url:
            return {
                "ready": False,
                "message": "Database URL not configured"
            }

        # Create engine with short timeout
        engine = create_engine(db_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})

        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            if result == 1:
                return {
                    "ready": True,
                    "message": "Database connection OK"
                }
            else:
                return {
                    "ready": False,
                    "message": "Database query returned unexpected result"
                }
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return {
            "ready": False,
            "message": f"Database connection failed: {str(e)}"
        }


async def check_redis() -> Dict[str, Any]:
    """Check Redis connectivity"""
    try:
        import redis

        redis_url = os.getenv("SHIVX_REDIS_URL", "redis://localhost:6379/0")

        # Parse Redis URL and create client
        r = redis.from_url(redis_url, socket_connect_timeout=5, socket_timeout=5)

        # Test connection
        r.ping()

        # Get info
        info = r.info("memory")
        used_memory_mb = info.get("used_memory", 0) / (1024**2)

        return {
            "ready": True,
            "message": "Redis connection OK",
            "details": {
                "used_memory_mb": used_memory_mb
            }
        }
    except Exception as e:
        logger.error(f"Redis check failed: {e}")
        return {
            "ready": False,
            "message": f"Redis connection failed: {str(e)}"
        }


async def check_readiness() -> Dict[str, Any]:
    """
    Check readiness of all application components

    Returns:
        Dictionary with readiness status and component details
    """
    components = {}

    # Check 1: Application Health
    components["application"] = {
        "ready": True,
        "message": "Application is running",
        "checked_at": datetime.now().isoformat()
    }

    # Check 2: Database connectivity
    components["database"] = await check_database()
    components["database"]["checked_at"] = datetime.now().isoformat()

    # Check 3: Redis connectivity
    components["redis"] = await check_redis()
    components["redis"]["checked_at"] = datetime.now().isoformat()

    # Check 4: Disk space
    components["disk_space"] = await check_disk_space()
    components["disk_space"]["checked_at"] = datetime.now().isoformat()

    # Check 5: Memory usage
    components["memory"] = await check_memory()
    components["memory"]["checked_at"] = datetime.now().isoformat()

    # Determine overall readiness
    all_ready = all(component["ready"] for component in components.values())

    return {
        "ready": all_ready,
        "components": components,
        "timestamp": datetime.now().isoformat()
    }


async def check_liveness() -> Dict[str, Any]:
    """
    Simple liveness check - just confirms the process is alive

    Returns:
        Dictionary with liveness status
    """
    return {
        "alive": True,
        "timestamp": datetime.now().isoformat()
    }
