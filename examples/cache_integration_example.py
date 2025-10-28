"""
ShivX Caching System Integration Example
Demonstrates how to integrate and use the caching system in your application
"""

import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status
from redis.asyncio import Redis

from app.cache import initialize_redis, close_redis, get_redis, get_redis_manager
from app.middleware.rate_limit import RateLimitMiddleware, check_rate_limit
from app.middleware.cache import CacheMiddleware, CacheConfig, cache_response
from app.services.market_cache import MarketDataCache
from app.services.indicator_cache import IndicatorCache
from app.services.ml_cache import MLPredictionCache
from app.services.session_cache import SessionManager
from app.services.cache_monitor import CacheMonitor
from app.services.cache_invalidation import CacheInvalidationManager, InvalidationEvent
from config.settings import Settings, get_settings


# ============================================================================
# Application Lifespan with Cache Initialization
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager with cache initialization

    This handles:
    - Redis connection initialization
    - Cache warming on startup
    - Cache invalidation pub/sub setup
    - Cleanup on shutdown
    """
    settings = get_settings()

    # Initialize Redis connection pool
    print("Initializing Redis connection pool...")
    await initialize_redis(settings)

    redis_manager = get_redis_manager()
    redis = await redis_manager.get_client() if redis_manager else None

    if redis:
        # Initialize cache services
        app.state.market_cache = MarketDataCache(redis)
        app.state.indicator_cache = IndicatorCache(redis)
        app.state.ml_cache = MLPredictionCache(redis)
        app.state.session_manager = SessionManager(redis, session_ttl=settings.cache_session_ttl)
        app.state.cache_monitor = CacheMonitor(redis)
        app.state.cache_invalidation = CacheInvalidationManager(redis)

        # Initialize pub/sub for distributed invalidation
        if settings.cache_invalidation_pubsub:
            await app.state.cache_invalidation.initialize_pubsub()

        # Warm cache on startup
        if settings.cache_warming_enabled:
            print("Warming cache with popular trading pairs...")
            await warm_cache_on_startup(app)

        print("✓ Caching system initialized successfully")
    else:
        print("⚠ Redis unavailable - running without caching")

    yield

    # Cleanup on shutdown
    print("Shutting down caching system...")
    if hasattr(app.state, 'cache_invalidation'):
        await app.state.cache_invalidation.close_pubsub()
    await close_redis()
    print("✓ Caching system shut down")


# ============================================================================
# Cache Warming
# ============================================================================

async def warm_cache_on_startup(app: FastAPI):
    """
    Warm cache on startup with popular trading pairs and indicators

    This proactively loads frequently accessed data into cache
    """
    market_cache = app.state.market_cache
    indicator_cache = app.state.indicator_cache

    # Popular trading pairs to warm
    popular_pairs = ["SOL-USDC", "BTC-USDC", "ETH-USDC", "BONK-USDC"]
    timeframes = ["1m", "5m", "1h"]

    for pair in popular_pairs:
        # In a real implementation, fetch actual data
        # For example purposes, we use mock data
        mock_price_data = {
            "price": 100.0,
            "volume_24h": 1000000,
            "change_24h": 5.2,
        }
        await market_cache.set_price(pair, mock_price_data)

        # Warm indicator cache
        for timeframe in timeframes:
            mock_indicators = {
                "rsi": {"value": 65.0, "signal": "neutral"},
                "macd": {"macd": 1.2, "signal": 0.8, "histogram": 0.4},
                "bollinger": {"upper": 105.0, "middle": 100.0, "lower": 95.0},
            }

            # Cache indicators
            await indicator_cache.set_indicators_batch(
                pair,
                timeframe,
                {
                    "rsi": (mock_indicators["rsi"], indicator_cache.TTL_RSI),
                    "macd": (mock_indicators["macd"], indicator_cache.TTL_MACD),
                    "bollinger": (mock_indicators["bollinger"], indicator_cache.TTL_BOLLINGER),
                }
            )

    print(f"✓ Cache warmed with {len(popular_pairs)} pairs across {len(timeframes)} timeframes")


# ============================================================================
# Create FastAPI Application
# ============================================================================

app = FastAPI(
    title="ShivX AI Trading Platform",
    description="High-performance AI-driven trading platform with Redis caching",
    version="1.0.0",
    lifespan=lifespan
)

# Get settings
settings = get_settings()


# ============================================================================
# Add Middleware (Order matters!)
# ============================================================================

# Add caching middleware (applied first, closest to routes)
@app.on_event("startup")
async def setup_middleware():
    """Setup middleware after Redis is initialized"""
    redis_manager = get_redis_manager()
    redis = await redis_manager.get_client() if redis_manager else None

    # HTTP caching middleware
    app.add_middleware(CacheMiddleware, redis=redis)

    # Rate limiting middleware
    app.add_middleware(RateLimitMiddleware, redis=redis, settings=settings)


# ============================================================================
# Example 1: Market Data Endpoint with Caching
# ============================================================================

@app.get("/api/market/price/{token_pair}")
async def get_market_price(
    token_pair: str,
    redis: Optional[Redis] = Depends(get_redis)
):
    """
    Get market price with caching

    Cache Strategy:
    - TTL: 5 seconds
    - Fallback to API if cache miss
    - Automatic invalidation on trades
    """
    market_cache = MarketDataCache(redis)

    # Try cache first
    cached_price = await market_cache.get_price(token_pair)
    if cached_price:
        return {
            "token_pair": token_pair,
            "source": "cache",
            **cached_price
        }

    # Cache miss - fetch from API (simulated)
    print(f"Cache miss for {token_pair}, fetching from API...")
    price_data = await fetch_price_from_api(token_pair)

    # Cache for next request
    await market_cache.set_price(token_pair, price_data)

    return {
        "token_pair": token_pair,
        "source": "api",
        **price_data
    }


async def fetch_price_from_api(token_pair: str) -> Dict[str, Any]:
    """Simulate fetching price from external API"""
    await asyncio.sleep(0.05)  # Simulate network delay
    return {
        "price": 100.0,
        "volume_24h": 1000000,
        "change_24h": 5.2,
        "timestamp": "2025-10-28T00:00:00Z"
    }


# ============================================================================
# Example 2: Technical Indicators with Batch Caching
# ============================================================================

@app.get("/api/indicators/{token_pair}")
async def get_indicators(
    token_pair: str,
    timeframe: str = "1h",
    redis: Optional[Redis] = Depends(get_redis)
):
    """
    Get technical indicators with batch caching

    Returns: RSI, MACD, Bollinger Bands
    Cache: 1 minute TTL
    """
    indicator_cache = IndicatorCache(redis)

    # Try batch cache fetch
    cached_indicators = await indicator_cache.get_indicators_batch(
        token_pair,
        timeframe,
        ["rsi", "macd", "bollinger"]
    )

    # Compute missing indicators
    if not all(cached_indicators.values()):
        print(f"Computing indicators for {token_pair} {timeframe}...")
        computed = await compute_indicators(token_pair, timeframe)

        # Cache computed indicators
        await indicator_cache.set_indicators_batch(
            token_pair,
            timeframe,
            {
                "rsi": (computed["rsi"], indicator_cache.TTL_RSI),
                "macd": (computed["macd"], indicator_cache.TTL_MACD),
                "bollinger": (computed["bollinger"], indicator_cache.TTL_BOLLINGER),
            }
        )

        return {
            "token_pair": token_pair,
            "timeframe": timeframe,
            "source": "computed",
            "indicators": computed
        }

    return {
        "token_pair": token_pair,
        "timeframe": timeframe,
        "source": "cache",
        "indicators": cached_indicators
    }


async def compute_indicators(token_pair: str, timeframe: str) -> Dict[str, Any]:
    """Simulate computing technical indicators"""
    await asyncio.sleep(0.1)  # Simulate computation time
    return {
        "rsi": {"value": 65.0, "signal": "neutral"},
        "macd": {"macd": 1.2, "signal": 0.8, "histogram": 0.4},
        "bollinger": {"upper": 105.0, "middle": 100.0, "lower": 95.0}
    }


# ============================================================================
# Example 3: ML Predictions with Confidence Filtering
# ============================================================================

@app.get("/api/ml/predict/{token_pair}")
async def get_ml_prediction(
    token_pair: str,
    redis: Optional[Redis] = Depends(get_redis)
):
    """
    Get ML prediction with caching

    Only caches predictions with confidence >70%
    TTL: 30 seconds
    """
    ml_cache = MLPredictionCache(redis)

    model_name = "lstm_v1"
    model_version = "1.0.0"

    # Compute features
    features = await compute_features(token_pair)
    features_hash = ml_cache._hash_features(features)

    # Try cache
    cached_prediction = await ml_cache.get_prediction(
        model_name,
        model_version,
        token_pair,
        features_hash
    )

    if cached_prediction:
        return {
            "token_pair": token_pair,
            "source": "cache",
            **cached_prediction
        }

    # Run inference
    print(f"Running ML inference for {token_pair}...")
    prediction = await run_ml_inference(model_name, features)

    # Cache if confidence is high enough
    await ml_cache.set_prediction(
        model_name,
        model_version,
        token_pair,
        features,
        prediction
    )

    return {
        "token_pair": token_pair,
        "source": "inference",
        **prediction
    }


async def compute_features(token_pair: str) -> Dict[str, Any]:
    """Compute ML features"""
    return {
        "price": 100.0,
        "volume": 1000000,
        "rsi": 65.0,
        "macd": 1.2,
    }


async def run_ml_inference(model_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate ML inference"""
    await asyncio.sleep(0.2)  # Simulate inference time
    return {
        "prediction": "bullish",
        "confidence": 0.85,
        "price_target": 110.0,
    }


# ============================================================================
# Example 4: Session Management
# ============================================================================

@app.post("/api/auth/login")
async def login(
    username: str,
    password: str,
    redis: Optional[Redis] = Depends(get_redis)
):
    """
    Login endpoint with session creation

    Creates session in Redis with:
    - 24 hour TTL
    - IP tracking
    - User agent tracking
    - Concurrent session limits
    """
    # Authenticate user (simulated)
    user_id = "user123"
    user_data = {"username": username, "role": "trader"}

    session_manager = SessionManager(redis)

    # Create session
    session_token = await session_manager.create_session(
        user_id=user_id,
        user_data=user_data,
        ip_address="192.168.1.1",
        user_agent="Mozilla/5.0"
    )

    if not session_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session"
        )

    return {
        "success": True,
        "session_token": session_token,
        "user": user_data
    }


@app.post("/api/auth/logout")
async def logout(
    session_token: str,
    redis: Optional[Redis] = Depends(get_redis)
):
    """
    Logout endpoint with session revocation

    Immediately invalidates session in Redis
    """
    session_manager = SessionManager(redis)

    success = await session_manager.revoke_session(session_token)

    return {
        "success": success,
        "message": "Logged out successfully" if success else "Session not found"
    }


# ============================================================================
# Example 5: Cache Invalidation on Trade Execution
# ============================================================================

@app.post("/api/trading/execute")
async def execute_trade(
    token_pair: str,
    amount: float,
    redis: Optional[Redis] = Depends(get_redis)
):
    """
    Execute trade and invalidate related caches

    Automatically invalidates:
    - Price cache
    - Order book
    - Indicators
    - ML predictions
    """
    # Execute trade (simulated)
    trade_data = {
        "trade_id": "trade123",
        "token_pair": token_pair,
        "amount": amount,
        "price": 100.0,
        "timestamp": "2025-10-28T00:00:00Z"
    }

    # Invalidate affected caches
    invalidation_manager = CacheInvalidationManager(redis)
    keys_invalidated = await invalidation_manager.on_trade_executed(
        token_pair,
        trade_data
    )

    # Publish invalidation event for other instances
    await invalidation_manager.publish_invalidation(
        InvalidationEvent.TRADE_EXECUTED,
        {"token_pair": token_pair, "trade_id": trade_data["trade_id"]}
    )

    return {
        "success": True,
        "trade": trade_data,
        "cache_invalidated": keys_invalidated
    }


# ============================================================================
# Example 6: Cache Monitoring Dashboard
# ============================================================================

@app.get("/api/admin/cache/stats")
async def get_cache_stats():
    """
    Get comprehensive cache statistics

    Returns:
    - Hit rates by cache type
    - Memory usage
    - Key counts
    - Performance metrics
    - Recommendations
    """
    cache_monitor: CacheMonitor = app.state.cache_monitor
    return await cache_monitor.get_performance_report()


@app.get("/api/admin/cache/health")
async def cache_health_check():
    """Cache health check endpoint"""
    cache_monitor: CacheMonitor = app.state.cache_monitor
    return await cache_monitor.health_check()


# ============================================================================
# Example 7: Manual Cache Flush (Admin Only)
# ============================================================================

@app.post("/api/admin/cache/flush")
async def flush_cache(
    cache_type: Optional[str] = None,
    admin_user_id: str = "admin",
    reason: str = "manual_flush"
):
    """
    Manually flush cache (admin only)

    Args:
        cache_type: Specific cache type to flush (or None for all)
        admin_user_id: ID of admin performing flush
        reason: Reason for flush (audit logging)

    Note: This should have proper authentication in production
    """
    invalidation_manager: CacheInvalidationManager = app.state.cache_invalidation

    if cache_type:
        result = await invalidation_manager.flush_cache_type(
            cache_type,
            admin_user_id,
            reason
        )
    else:
        result = await invalidation_manager.flush_all_cache(
            admin_user_id,
            reason
        )

    return result


# ============================================================================
# Example 8: Expensive Endpoint with Additional Rate Limiting
# ============================================================================

@app.post("/api/trading/backtest")
async def run_backtest(
    strategy: str,
    _: None = Depends(check_rate_limit)  # Additional rate limiting
):
    """
    Expensive backtest endpoint with strict rate limiting

    Rate Limit: 10 requests per minute (stricter than normal)
    """
    await asyncio.sleep(2)  # Simulate expensive computation

    return {
        "success": True,
        "strategy": strategy,
        "results": {"profit": 15.2, "sharpe": 1.8}
    }


# ============================================================================
# Example 9: Cached Endpoint with Custom Decorator
# ============================================================================

@app.get("/api/analytics/summary")
@cache_response(ttl=300, cache_authenticated=False)
async def get_analytics_summary():
    """
    Analytics summary with 5-minute cache

    Using @cache_response decorator for automatic caching
    """
    await asyncio.sleep(1)  # Simulate expensive computation

    return {
        "total_trades": 1000,
        "total_volume": 5000000,
        "top_pairs": ["SOL-USDC", "BTC-USDC", "ETH-USDC"],
        "avg_profit": 12.5
    }


# ============================================================================
# Example 10: Prometheus Metrics Endpoint
# ============================================================================

@app.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint

    Exports all cache-related metrics
    """
    cache_monitor: CacheMonitor = app.state.cache_monitor
    metrics = await cache_monitor.export_prometheus_metrics()

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=metrics)


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("""
╔══════════════════════════════════════════════════════════════╗
║         ShivX AI Trading Platform - Cache Example          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Redis caching system with:                                 ║
║  ✓ Market data caching                                      ║
║  ✓ Technical indicators caching                             ║
║  ✓ ML predictions caching                                   ║
║  ✓ Session management                                       ║
║  ✓ Rate limiting                                            ║
║  ✓ HTTP response caching                                    ║
║  ✓ Cache monitoring & metrics                               ║
║                                                              ║
║  Endpoints:                                                  ║
║  • GET  /api/market/price/{pair}    - Market prices        ║
║  • GET  /api/indicators/{pair}      - Indicators           ║
║  • GET  /api/ml/predict/{pair}      - ML predictions       ║
║  • POST /api/auth/login             - Login                ║
║  • POST /api/trading/execute        - Execute trade        ║
║  • GET  /api/admin/cache/stats      - Cache stats          ║
║  • GET  /metrics                     - Prometheus metrics   ║
║                                                              ║
║  Documentation: http://localhost:8000/docs                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "cache_integration_example:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
