"""
ShivX AI Trading System - Main Application (v2.0)
Production-ready FastAPI application with:
- Centralized configuration (Pydantic Settings)
- JWT authentication
- IP-based rate limiting
- Comprehensive routers (Trading, Analytics, AI)
- Dependency injection
- Security hardening
- Monitoring & observability

Run with: uvicorn main_v2:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configuration
from config.settings import get_settings

# Import logging setup
from utils.logging_setup import setup_logging

# Setup logging
logger = setup_logging(__name__)

# Import routes
from app.routes.health import router as health_router
from app.routers.trading import router as trading_router
from app.routers.analytics import router as analytics_router
from app.routers.ai import router as ai_router

# Import security
from core.security.hardening import SecurityHardeningEngine

# Get settings
settings = get_settings()


# ============================================================================
# Rate Limiting Setup
# ============================================================================

limiter = Limiter(key_func=get_remote_address)


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
    logger.info(f"ShivX AI Trading System v{settings.version} ({settings.env.value})")
    logger.info(f"Git SHA: {settings.git_sha}")
    logger.info("=" * 70)

    # Initialize security engine
    app.state.security = SecurityHardeningEngine()
    logger.info("✓ Security hardening engine initialized")

    # Initialize settings
    app.state.settings = settings
    logger.info(f"✓ Configuration loaded (env: {settings.env.value})")

    # Log feature flags
    features = settings.get_feature_flags()
    enabled_features = [k for k, v in features.items() if v]
    logger.info(f"✓ Feature flags: {len(enabled_features)}/{len(features)} enabled")

    # Trading mode warning
    if settings.trading_mode.value == "live":
        logger.warning("⚠️  LIVE TRADING MODE ENABLED - Real funds at risk!")
    else:
        logger.info(f"✓ Trading mode: {settings.trading_mode.value} (safe)")

    # Initialize other components
    # app.state.database = await init_database(settings)
    # app.state.redis = await init_redis(settings)
    # app.state.trading_engine = await init_trading_engine(settings)
    # app.state.ml_models = await load_ml_models(settings)

    logger.info("✓ Application startup complete")
    logger.info("=" * 70)

    yield

    # Shutdown
    logger.info("Shutting down ShivX AI Trading System...")

    # Cleanup resources
    # await app.state.database.close()
    # await app.state.redis.close()
    # await app.state.trading_engine.shutdown()

    logger.info("✓ Application shutdown complete")


# ============================================================================
# FastAPI Application Factory
# ============================================================================

def create_app() -> FastAPI:
    """
    Create and configure FastAPI application
    """

    # Create FastAPI app
    app = FastAPI(
        title="ShivX AI Trading System",
        description="Advanced autonomous AI trading platform with reinforcement learning and multi-strategy execution",
        version=settings.version,
        lifespan=lifespan,
        docs_url="/api/docs" if not settings.is_production else None,
        redoc_url="/api/redoc" if not settings.is_production else None,
        openapi_url="/api/openapi.json" if not settings.is_production else None,
    )

    # Add rate limiter state
    app.state.limiter = limiter

    # ========================================================================
    # Security Middleware
    # ========================================================================

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )
    logger.info(f"✓ CORS configured for origins: {settings.cors_origins}")

    # 2. Trusted Host Middleware (prevent host header attacks)
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.trusted_hosts
        )
        logger.info(f"✓ Trusted hosts configured: {settings.trusted_hosts}")

    # 3. Security Headers Middleware
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
        response.headers["X-ShivX-Version"] = settings.version
        response.headers["X-Environment"] = settings.env.value

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

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        """Handle rate limit exceeded"""
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": {
                    "code": 429,
                    "message": "Rate limit exceeded. Please try again later.",
                    "request_id": getattr(request.state, "request_id", "unknown")
                }
            }
        )

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
                    "message": "Internal server error" if settings.is_production else str(exc),
                    "request_id": getattr(request.state, "request_id", "unknown")
                }
            }
        )

    # ========================================================================
    # Routes
    # ========================================================================

    # Root endpoint
    @app.get("/", tags=["root"])
    @limiter.limit("30/minute")
    async def root(request: Request):
        """Root endpoint with API information"""
        return {
            "service": "ShivX AI Trading System",
            "version": settings.version,
            "environment": settings.env.value,
            "status": "operational",
            "trading_mode": settings.trading_mode.value,
            "docs": "/api/docs" if not settings.is_production else "disabled",
            "health": "/api/health/live",
            "features": settings.get_feature_flags()
        }

    # Include routers
    app.include_router(health_router)
    app.include_router(trading_router)
    app.include_router(analytics_router)
    app.include_router(ai_router)

    logger.info("✓ Routes configured")

    return app


# ============================================================================
# Application Instance
# ============================================================================

app = create_app()

# Add rate limit handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ============================================================================
# Main Entry Point (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting development server on {settings.host}:{settings.port}")
    logger.info(f"Reload enabled: {settings.reload}")

    uvicorn.run(
        "main_v2:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.value.lower(),
        access_log=True,
    )
