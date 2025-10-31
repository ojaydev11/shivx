"""
Analytics API Router
Endpoints for market data, analysis, and reporting
"""

from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, require_permission, get_settings
from app.dependencies.auth import TokenData
from app.database import get_db
from app.services.analytics_service import AnalyticsService
from config.settings import Settings
from core.security.hardening import Permission


router = APIRouter(
    prefix="/api/analytics",
    tags=["analytics"],
)


# ============================================================================
# Pydantic Models
# ============================================================================

class MarketData(BaseModel):
    """Market data point"""
    token: str
    price: float
    volume_24h: float
    market_cap: Optional[float] = None
    price_change_24h: float
    timestamp: datetime


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators"""
    token: str
    rsi: float = Field(..., ge=0, le=100, description="Relative Strength Index")
    macd: float
    macd_signal: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    timestamp: datetime


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result"""
    token: str
    sentiment_score: float = Field(..., ge=-1, le=1, description="Sentiment score (-1 to 1)")
    sentiment_label: str  # very_negative, negative, neutral, positive, very_positive
    confidence: float = Field(..., ge=0, le=1)
    sources: int
    keywords: List[str]
    analyzed_at: datetime


class PerformanceReport(BaseModel):
    """Performance report"""
    period: str
    start_date: datetime
    end_date: datetime
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_factor: float
    total_trades: int
    best_trade: float
    worst_trade: float
    average_trade: float


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/market-data", response_model=List[MarketData])
async def get_market_data(
    tokens: Optional[str] = Query(None, description="Comma-separated token symbols"),
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    Get current market data for tokens

    Requires: READ permission

    Args:
        tokens: Comma-separated token symbols (e.g., "SOL,ETH,BTC")
    """
    # Parse tokens
    token_list = tokens.split(",") if tokens else None

    service = AnalyticsService(settings)
    market_data = await service.get_market_data(tokens=token_list)

    return [
        MarketData(
            token=data["token"],
            price=data["price"],
            volume_24h=data["volume_24h"],
            market_cap=data.get("market_cap"),
            price_change_24h=data["price_change_24h"],
            timestamp=data["timestamp"]
        )
        for data in market_data
    ]


@router.get("/technical-indicators/{token}", response_model=TechnicalIndicators)
async def get_technical_indicators(
    token: str,
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    Get technical indicators for a token

    Requires: READ permission

    Args:
        token: Token symbol (e.g., SOL)
    """
    service = AnalyticsService(settings)
    indicators = await service.get_technical_indicators(token=token)

    return TechnicalIndicators(**indicators)


@router.get("/sentiment/{token}", response_model=SentimentAnalysis)
async def get_sentiment(
    token: str,
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    Get sentiment analysis for a token

    Requires: READ permission

    Args:
        token: Token symbol (e.g., SOL)
    """
    service = AnalyticsService(settings)
    sentiment = await service.get_sentiment(token=token)

    return SentimentAnalysis(**sentiment)


@router.get("/reports/performance", response_model=PerformanceReport)
async def get_performance_report(
    period: str = Query("30d", description="Period (7d, 30d, 90d, 1y)"),
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    Get comprehensive performance report

    Requires: READ permission

    Args:
        period: Time period for report
    """
    service = AnalyticsService(settings)
    report = await service.get_performance_report(db=db, period=period)

    return PerformanceReport(**report)


@router.get("/price-history/{token}")
async def get_price_history(
    token: str,
    interval: str = Query("1h", description="Interval (1m, 5m, 15m, 1h, 4h, 1d)"),
    limit: int = Query(100, ge=1, le=1000, description="Number of candles"),
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    Get historical price data (OHLCV)

    Requires: READ permission

    Args:
        token: Token symbol
        interval: Candlestick interval
        limit: Number of candles to return
    """
    service = AnalyticsService(settings)
    history = await service.get_price_history(
        token=token,
        interval=interval,
        limit=limit
    )

    return history


@router.get("/portfolio")
async def get_portfolio_analytics(
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    Get portfolio analytics

    Requires: READ permission

    Returns:
        Comprehensive portfolio metrics
    """
    service = AnalyticsService(settings)
    portfolio = await service.get_portfolio_analytics(
        db=db,
        user_id=current_user.username
    )

    return portfolio


@router.get("/market-overview")
async def get_market_overview(
    settings: Settings = Depends(get_settings)
):
    """
    Get general market overview (public endpoint)

    No authentication required
    """
    service = AnalyticsService(settings)
    overview = await service.get_market_overview()

    return overview
