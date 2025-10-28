"""
ShivX AI Trading System - Main Application Entry Point
Production-ready FastAPI application with comprehensive security

Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# Import logging setup first
from utils.logging_setup import setup_logging

# Setup logging
logger = setup_logging(__name__)

# Import routes
from app.routes.health import router as health_router

# Import security components
from core.security.hardening import SecurityHardeningEngine

# Version info
VERSION = os.getenv("SHIVX_VERSION", "dev")
GIT_SHA = os.getenv("SHIVX_GIT_SHA", "unknown")
ENV = os.getenv("SHIVX_ENV", "local")


# ============================================================================
# Application Lifecycle Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("=" * 70)
    logger.info(f"ShivX AI Trading System v{VERSION} ({ENV})")
    logger.info(f"Git SHA: {GIT_SHA}")
    logger.info("=" * 70)

    # Initialize security engine
    app.state.security = SecurityHardeningEngine()
    logger.info("✓ Security hardening engine initialized")

    # Initialize other components here
    # app.state.database = await init_database()
    # app.state.redis = await init_redis()
    # app.state.trading_engine = await init_trading_engine()

    logger.info("✓ Application startup complete")
    logger.info("=" * 70)

    yield

    # Shutdown
    logger.info("Shutting down ShivX AI Trading System...")

    # Cleanup resources
    # await app.state.database.close()
    # await app.state.redis.close()

    logger.info("✓ Application shutdown complete")


# ============================================================================
# FastAPI Application Factory
# ============================================================================

def create_app() -> FastAPI:
    """
    Create and configure FastAPI application with security best practices
    """

    # CORS configuration
    cors_origins_str = os.getenv("SHIVX_CORS_ORIGINS", "http://localhost:3000")
    cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

    # Create FastAPI app
    app = FastAPI(
        title="ShivX AI Trading System",
        description="Advanced autonomous AI trading platform with reinforcement learning and multi-strategy execution",
        version=VERSION,
        lifespan=lifespan,
        docs_url="/api/docs" if ENV != "production" else None,  # Disable docs in production
        redoc_url="/api/redoc" if ENV != "production" else None,
        openapi_url="/api/openapi.json" if ENV != "production" else None,
    )

    # ========================================================================
    # Security Middleware
    # ========================================================================

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )
    logger.info(f"✓ CORS configured for origins: {cors_origins}")

    # 2. Trusted Host Middleware (prevent host header attacks)
    if ENV == "production":
        trusted_hosts = os.getenv("SHIVX_TRUSTED_HOSTS", "*").split(",")
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=trusted_hosts
        )
        logger.info(f"✓ Trusted hosts configured: {trusted_hosts}")

    # 3. Security Headers Middleware (custom)
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add security headers to all responses"""
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # Custom headers
        response.headers["X-ShivX-Version"] = VERSION
        response.headers["X-Environment"] = ENV

        return response

    # 4. Request ID Middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Add unique request ID to all requests"""
        import uuid
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response

    # ========================================================================
    # Exception Handlers
    # ========================================================================

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "request_id": getattr(request.state, "request_id", "unknown")
                }
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": 422,
                    "message": "Validation error",
                    "details": exc.errors(),
                    "request_id": getattr(request.state, "request_id", "unknown")
                }
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": 500,
                    "message": "Internal server error" if ENV == "production" else str(exc),
                    "request_id": getattr(request.state, "request_id", "unknown")
                }
            }
        )

    # ========================================================================
    # Routes
    # ========================================================================

    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information"""
        return {
            "service": "ShivX AI Trading System",
            "version": VERSION,
            "environment": ENV,
            "status": "operational",
            "docs": "/api/docs" if ENV != "production" else "disabled",
            "health": "/api/health/live"
        }

    # Include routers
    app.include_router(health_router)

    # Additional routers would be included here:
    # from app.routes.trading import router as trading_router
    # from app.routes.analytics import router as analytics_router
    # app.include_router(trading_router)
    # app.include_router(analytics_router)

    logger.info("✓ Routes configured")

    return app


# ============================================================================
# Application Instance
# ============================================================================

app = create_app()


# ============================================================================
# Main Entry Point (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Configuration
    host = os.getenv("SHIVX_HOST", "0.0.0.0")
    port = int(os.getenv("SHIVX_PORT", "8000"))
    reload = os.getenv("SHIVX_DEV", "false").lower() == "true"

    logger.info(f"Starting development server on {host}:{port}")
    logger.info(f"Reload enabled: {reload}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True,
    )
