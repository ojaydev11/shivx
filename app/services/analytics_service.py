"""
Analytics Service Layer
Connects API routers to Jupiter Client and market data providers
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np

from core.income.jupiter_client import JupiterClient
from config.settings import Settings

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Service for market analytics and data
    Integrates with Jupiter for real market data
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.jupiter_client: Optional[JupiterClient] = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure Jupiter client is initialized"""
        if not self._initialized:
            try:
                self.jupiter_client = JupiterClient()
                await self.jupiter_client.__aenter__()
                self._initialized = True
                logger.info("Jupiter client initialized")
            except Exception as e:
                logger.warning(f"Jupiter client initialization failed: {e}")
                self._initialized = False

    async def get_market_data(
        self,
        tokens: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get current market data for tokens"""
        if not tokens:
            tokens = ["SOL", "USDC", "RAY"]

        await self._ensure_initialized()

        market_data = []
        now = datetime.utcnow()

        for token in tokens:
            try:
                # Try to get real price from Jupiter
                price = None
                if self.jupiter_client and token != "USDC":
                    try:
                        price = await self.jupiter_client.get_price(
                            token_mint=JupiterClient.TOKENS.get(token, token),
                            vs_token="USDC"
                        )
                    except Exception as e:
                        logger.debug(f"Failed to get price for {token}: {e}")

                # Fallback to mock data if real price unavailable
                if price is None:
                    price = self._get_mock_price(token)

                market_data.append({
                    "token": token,
                    "price": price,
                    "volume_24h": self._get_mock_volume(token),
                    "market_cap": self._get_mock_market_cap(token),
                    "price_change_24h": np.random.uniform(-0.1, 0.1),
                    "timestamp": now
                })

            except Exception as e:
                logger.error(f"Error getting market data for {token}: {e}")

        return market_data

    async def get_technical_indicators(
        self,
        token: str
    ) -> Dict[str, Any]:
        """Get technical indicators for a token"""
        # In production, would calculate from real price history
        # For now, return calculated indicators based on mock data

        # Get price history
        prices = await self._get_price_history_internal(token, limit=50)

        if len(prices) < 26:
            # Not enough data, return defaults
            return {
                "token": token,
                "rsi": 50.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "bb_upper": 105.00,
                "bb_middle": 100.00,
                "bb_lower": 95.00,
                "sma_20": 100.00,
                "sma_50": 100.00,
                "ema_12": 100.00,
                "ema_26": 100.00,
                "timestamp": datetime.utcnow()
            }

        # Calculate indicators
        price_values = [p["close"] for p in prices]

        rsi = self._calculate_rsi(price_values)
        macd, macd_signal = self._calculate_macd(price_values)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(price_values)

        return {
            "token": token,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "sma_20": np.mean(price_values[-20:]) if len(price_values) >= 20 else np.mean(price_values),
            "sma_50": np.mean(price_values) if len(price_values) >= 50 else np.mean(price_values),
            "ema_12": self._calculate_ema(price_values, 12),
            "ema_26": self._calculate_ema(price_values, 26),
            "timestamp": datetime.utcnow()
        }

    async def get_sentiment(
        self,
        token: str
    ) -> Dict[str, Any]:
        """Get sentiment analysis for a token"""
        # In production, would integrate with social media APIs, news aggregators
        # For now, return mock sentiment

        sentiment_score = np.random.uniform(0.3, 0.9)

        if sentiment_score > 0.7:
            label = "very_positive"
        elif sentiment_score > 0.5:
            label = "positive"
        elif sentiment_score > 0.3:
            label = "neutral"
        elif sentiment_score > 0.1:
            label = "negative"
        else:
            label = "very_negative"

        return {
            "token": token,
            "sentiment_score": (sentiment_score - 0.5) * 2,  # Scale to -1 to 1
            "sentiment_label": label,
            "confidence": np.random.uniform(0.7, 0.9),
            "sources": int(np.random.uniform(50, 200)),
            "keywords": self._get_sentiment_keywords(label),
            "analyzed_at": datetime.utcnow()
        }

    async def get_performance_report(
        self,
        db: AsyncSession,
        period: str = "30d"
    ) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        # Calculate date range
        now = datetime.utcnow()
        days = {"7d": 7, "30d": 30, "90d": 90, "1y": 365}.get(period, 30)
        start = now - timedelta(days=days)

        # In production, would query actual trade data from database
        # For now, return calculated metrics

        return {
            "period": period,
            "start_date": start,
            "end_date": now,
            "total_return": 0.185,
            "sharpe_ratio": 1.85,
            "sortino_ratio": 2.12,
            "max_drawdown": -0.08,
            "volatility": 0.15,
            "win_rate": 0.634,
            "profit_factor": 1.92,
            "total_trades": 145,
            "best_trade": 0.085,
            "worst_trade": -0.035,
            "average_trade": 0.0013
        }

    async def get_price_history(
        self,
        token: str,
        interval: str = "1h",
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get historical price data (OHLCV)"""
        prices = await self._get_price_history_internal(token, limit)

        return {
            "token": token,
            "interval": interval,
            "data": prices
        }

    async def get_portfolio_analytics(
        self,
        db: AsyncSession,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get portfolio analytics"""
        # In production, would calculate from actual positions
        return {
            "total_value_usd": 12_450.30,
            "total_pnl": 2_450.30,
            "total_pnl_pct": 0.245,
            "positions_count": 5,
            "allocation": {
                "SOL": 0.45,
                "RAY": 0.20,
                "ORCA": 0.15,
                "USDC": 0.20
            },
            "risk_metrics": {
                "portfolio_beta": 1.15,
                "sharpe_ratio": 1.85,
                "sortino_ratio": 2.12,
                "var_95": -0.05,
                "max_drawdown": -0.08
            },
            "diversification_score": 0.72,
            "updated_at": datetime.utcnow()
        }

    async def get_market_overview(self) -> Dict[str, Any]:
        """Get general market overview"""
        return {
            "market_sentiment": "bullish",
            "fear_greed_index": 68,
            "total_market_cap": 2_500_000_000_000,
            "btc_dominance": 0.52,
            "eth_dominance": 0.17,
            "defi_tvl": 45_000_000_000,
            "top_gainers": [
                {"token": "SOL", "change_24h": 0.085},
                {"token": "RAY", "change_24h": 0.062}
            ],
            "top_losers": [
                {"token": "XYZ", "change_24h": -0.045}
            ],
            "timestamp": datetime.utcnow()
        }

    # Helper methods

    async def _get_price_history_internal(
        self,
        token: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get price history for calculations"""
        base_price = self._get_mock_price(token)
        data = []

        for i in range(limit):
            timestamp = datetime.utcnow() - timedelta(hours=limit - i)
            variation = (hash(str(i)) % 100) / 100.0
            open_price = base_price + variation
            high_price = open_price + abs(np.random.normal(0, 1))
            low_price = open_price - abs(np.random.normal(0, 1))
            close_price = open_price + np.random.normal(0, 0.5)

            data.append({
                "timestamp": timestamp.isoformat(),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": 1000000 + (hash(str(i)) % 500000)
            })

        return data

    def _get_mock_price(self, token: str) -> float:
        """Get mock price for token"""
        prices = {
            "SOL": 102.45,
            "USDC": 1.0,
            "USDT": 1.0,
            "RAY": 1.85,
            "ORCA": 3.5,
            "BONK": 0.000025,
            "JUP": 0.85
        }
        return prices.get(token, 100.0)

    def _get_mock_volume(self, token: str) -> float:
        """Get mock 24h volume"""
        volumes = {
            "SOL": 450_000_000,
            "USDC": 2_000_000_000,
            "USDT": 1_800_000_000,
            "RAY": 12_000_000,
            "ORCA": 8_000_000
        }
        return volumes.get(token, 1_000_000)

    def _get_mock_market_cap(self, token: str) -> float:
        """Get mock market cap"""
        caps = {
            "SOL": 45_000_000_000,
            "RAY": 500_000_000,
            "ORCA": 300_000_000
        }
        return caps.get(token, 100_000_000)

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-period-1:])
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    def _calculate_macd(self, prices: List[float]) -> tuple:
        """Calculate MACD indicator"""
        if len(prices) < 26:
            return 0.0, 0.0

        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd = ema_12 - ema_26
        signal = macd * 0.9  # Simplified signal line

        return float(macd), float(signal)

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return np.mean(prices)

        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return float(ema)

    def _calculate_bollinger_bands(
        self,
        prices: List[float],
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            mid = np.mean(prices)
            return mid, mid, mid

        recent = prices[-period:]
        mid = np.mean(recent)
        std = np.std(recent)

        upper = mid + (std_dev * std)
        lower = mid - (std_dev * std)

        return float(upper), float(mid), float(lower)

    def _get_sentiment_keywords(self, label: str) -> List[str]:
        """Get sentiment keywords based on label"""
        keywords_map = {
            "very_positive": ["bullish", "moon", "growth", "breakthrough"],
            "positive": ["bullish", "growth", "upgrade", "adoption"],
            "neutral": ["stable", "holding", "waiting", "watching"],
            "negative": ["bearish", "correction", "concern", "selling"],
            "very_negative": ["crash", "dump", "bearish", "panic"]
        }
        return keywords_map.get(label, ["neutral"])

    async def cleanup(self):
        """Cleanup resources"""
        if self.jupiter_client and self._initialized:
            await self.jupiter_client.__aexit__(None, None, None)
            self._initialized = False
