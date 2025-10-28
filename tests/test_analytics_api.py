"""
Analytics API Integration Tests
Coverage Target: 100% of app/routers/analytics.py

Tests all analytics endpoints including market data, technical indicators,
sentiment analysis, and performance reports
"""

import pytest
from fastapi import status
from datetime import datetime


@pytest.mark.unit
class TestMarketData:
    """Test GET /api/analytics/market-data endpoint"""

    def test_get_market_data_default(self, client, test_token):
        """Test getting market data with default tokens"""
        response = client.get(
            "/api/analytics/market-data",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Verify structure
        item = data[0]
        assert "token" in item
        assert "price" in item
        assert "volume_24h" in item
        assert "price_change_24h" in item
        assert "timestamp" in item

    def test_get_market_data_specific_tokens(self, client, test_token):
        """Test with specific token symbols"""
        response = client.get(
            "/api/analytics/market-data?tokens=SOL,RAY,ORCA",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

    def test_get_market_data_single_token(self, client, test_token):
        """Test with single token"""
        response = client.get(
            "/api/analytics/market-data?tokens=SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

    def test_get_market_data_without_auth(self, client):
        """Test without authentication"""
        response = client.get("/api/analytics/market-data")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_market_data_readonly_permission(self, client, readonly_token):
        """Test with READ permission"""
        response = client.get(
            "/api/analytics/market-data",
            headers={"Authorization": f"Bearer {readonly_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

    def test_market_data_values_valid(self, client, test_token):
        """Test that returned values are valid numbers"""
        response = client.get(
            "/api/analytics/market-data",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()
        for item in data:
            assert isinstance(item["price"], (int, float))
            assert item["price"] >= 0
            assert isinstance(item["volume_24h"], (int, float))
            assert item["volume_24h"] >= 0
            assert isinstance(item["price_change_24h"], (int, float))


@pytest.mark.unit
class TestTechnicalIndicators:
    """Test GET /api/analytics/technical-indicators/{token} endpoint"""

    def test_get_technical_indicators(self, client, test_token):
        """Test getting technical indicators for a token"""
        response = client.get(
            "/api/analytics/technical-indicators/SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify all required indicators
        assert "token" in data
        assert data["token"] == "SOL"
        assert "rsi" in data
        assert "macd" in data
        assert "macd_signal" in data
        assert "bb_upper" in data
        assert "bb_middle" in data
        assert "bb_lower" in data
        assert "sma_20" in data
        assert "sma_50" in data
        assert "ema_12" in data
        assert "ema_26" in data
        assert "timestamp" in data

    def test_get_technical_indicators_different_tokens(self, client, test_token):
        """Test with different token symbols"""
        tokens = ["SOL", "RAY", "ORCA", "BTC", "ETH"]

        for token in tokens:
            response = client.get(
                f"/api/analytics/technical-indicators/{token}",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["token"] == token

    def test_get_technical_indicators_without_auth(self, client):
        """Test without authentication"""
        response = client.get("/api/analytics/technical-indicators/SOL")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_technical_indicators_rsi_range(self, client, test_token):
        """Test that RSI is in valid range (0-100)"""
        response = client.get(
            "/api/analytics/technical-indicators/SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()
        assert 0 <= data["rsi"] <= 100

    def test_technical_indicators_bollinger_bands(self, client, test_token):
        """Test Bollinger Bands logical ordering"""
        response = client.get(
            "/api/analytics/technical-indicators/SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()
        # Upper should be >= middle >= lower
        assert data["bb_upper"] >= data["bb_middle"]
        assert data["bb_middle"] >= data["bb_lower"]


@pytest.mark.unit
class TestSentimentAnalysis:
    """Test GET /api/analytics/sentiment/{token} endpoint"""

    def test_get_sentiment(self, client, test_token):
        """Test getting sentiment analysis"""
        response = client.get(
            "/api/analytics/sentiment/SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "token" in data
        assert data["token"] == "SOL"
        assert "sentiment_score" in data
        assert "sentiment_label" in data
        assert "confidence" in data
        assert "sources" in data
        assert "keywords" in data
        assert "analyzed_at" in data

    def test_sentiment_score_range(self, client, test_token):
        """Test sentiment score is in valid range (-1 to 1)"""
        response = client.get(
            "/api/analytics/sentiment/SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()
        assert -1 <= data["sentiment_score"] <= 1

    def test_sentiment_confidence_range(self, client, test_token):
        """Test confidence is in valid range (0 to 1)"""
        response = client.get(
            "/api/analytics/sentiment/SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()
        assert 0 <= data["confidence"] <= 1

    def test_sentiment_label_values(self, client, test_token):
        """Test sentiment label is valid"""
        response = client.get(
            "/api/analytics/sentiment/SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()
        valid_labels = ["very_negative", "negative", "neutral", "positive", "very_positive"]
        assert data["sentiment_label"] in valid_labels

    def test_sentiment_keywords_is_list(self, client, test_token):
        """Test keywords is a list"""
        response = client.get(
            "/api/analytics/sentiment/SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()
        assert isinstance(data["keywords"], list)

    def test_get_sentiment_without_auth(self, client):
        """Test without authentication"""
        response = client.get("/api/analytics/sentiment/SOL")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.unit
class TestPerformanceReport:
    """Test GET /api/analytics/reports/performance endpoint"""

    def test_get_performance_report_default(self, client, test_token):
        """Test getting performance report with default period"""
        response = client.get(
            "/api/analytics/reports/performance",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify all required fields
        assert "period" in data
        assert "start_date" in data
        assert "end_date" in data
        assert "total_return" in data
        assert "sharpe_ratio" in data
        assert "sortino_ratio" in data
        assert "max_drawdown" in data
        assert "volatility" in data
        assert "win_rate" in data
        assert "profit_factor" in data
        assert "total_trades" in data
        assert "best_trade" in data
        assert "worst_trade" in data
        assert "average_trade" in data

    def test_get_performance_report_different_periods(self, client, test_token):
        """Test different time periods"""
        periods = ["7d", "30d", "90d", "1y"]

        for period in periods:
            response = client.get(
                f"/api/analytics/reports/performance?period={period}",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["period"] == period

    def test_performance_report_metrics_valid(self, client, test_token):
        """Test that metrics are valid values"""
        response = client.get(
            "/api/analytics/reports/performance",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()

        # Win rate should be between 0 and 1
        assert 0 <= data["win_rate"] <= 1

        # Total trades should be non-negative integer
        assert isinstance(data["total_trades"], int)
        assert data["total_trades"] >= 0

        # Max drawdown should be negative or zero
        assert data["max_drawdown"] <= 0

    def test_get_performance_report_without_auth(self, client):
        """Test without authentication"""
        response = client.get("/api/analytics/reports/performance")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.unit
class TestPriceHistory:
    """Test GET /api/analytics/price-history/{token} endpoint"""

    def test_get_price_history_default(self, client, test_token):
        """Test getting price history with defaults"""
        response = client.get(
            "/api/analytics/price-history/SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "token" in data
        assert data["token"] == "SOL"
        assert "interval" in data
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_price_history_candle_structure(self, client, test_token):
        """Test that candles have correct structure"""
        response = client.get(
            "/api/analytics/price-history/SOL?limit=10",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()
        candles = data["data"]

        assert len(candles) > 0
        candle = candles[0]

        assert "timestamp" in candle
        assert "open" in candle
        assert "high" in candle
        assert "low" in candle
        assert "close" in candle
        assert "volume" in candle

    def test_price_history_different_intervals(self, client, test_token):
        """Test different candlestick intervals"""
        intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]

        for interval in intervals:
            response = client.get(
                f"/api/analytics/price-history/SOL?interval={interval}",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["interval"] == interval

    def test_price_history_limit_parameter(self, client, test_token):
        """Test limit parameter"""
        limits = [10, 50, 100, 500]

        for limit in limits:
            response = client.get(
                f"/api/analytics/price-history/SOL?limit={limit}",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data["data"]) <= limit

    def test_price_history_limit_validation(self, client, test_token):
        """Test limit validation (min 1, max 1000)"""
        # Too low
        response = client.get(
            "/api/analytics/price-history/SOL?limit=0",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Too high
        response = client.get(
            "/api/analytics/price-history/SOL?limit=2000",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_price_history_without_auth(self, client):
        """Test without authentication"""
        response = client.get("/api/analytics/price-history/SOL")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_price_history_ohlc_logic(self, client, test_token):
        """Test OHLC logic (high >= low, etc.)"""
        response = client.get(
            "/api/analytics/price-history/SOL?limit=10",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()
        for candle in data["data"]:
            # High should be >= Low
            assert candle["high"] >= candle["low"]


@pytest.mark.unit
class TestPortfolioAnalytics:
    """Test GET /api/analytics/portfolio endpoint"""

    def test_get_portfolio_analytics(self, client, test_token):
        """Test getting portfolio analytics"""
        response = client.get(
            "/api/analytics/portfolio",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "total_value_usd" in data
        assert "total_pnl" in data
        assert "total_pnl_pct" in data
        assert "positions_count" in data
        assert "allocation" in data
        assert "risk_metrics" in data
        assert "diversification_score" in data
        assert "updated_at" in data

    def test_portfolio_allocation_structure(self, client, test_token):
        """Test portfolio allocation is a dictionary"""
        response = client.get(
            "/api/analytics/portfolio",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()
        assert isinstance(data["allocation"], dict)

    def test_portfolio_risk_metrics(self, client, test_token):
        """Test risk metrics structure"""
        response = client.get(
            "/api/analytics/portfolio",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()
        risk_metrics = data["risk_metrics"]

        assert "portfolio_beta" in risk_metrics
        assert "sharpe_ratio" in risk_metrics
        assert "sortino_ratio" in risk_metrics
        assert "var_95" in risk_metrics
        assert "max_drawdown" in risk_metrics

    def test_portfolio_diversification_score(self, client, test_token):
        """Test diversification score is in valid range (0-1)"""
        response = client.get(
            "/api/analytics/portfolio",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()
        assert 0 <= data["diversification_score"] <= 1

    def test_portfolio_without_auth(self, client):
        """Test without authentication"""
        response = client.get("/api/analytics/portfolio")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.unit
class TestMarketOverview:
    """Test GET /api/analytics/market-overview endpoint"""

    def test_get_market_overview(self, client):
        """Test getting market overview - public endpoint"""
        response = client.get("/api/analytics/market-overview")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "market_sentiment" in data
        assert "fear_greed_index" in data
        assert "total_market_cap" in data
        assert "btc_dominance" in data
        assert "eth_dominance" in data
        assert "defi_tvl" in data
        assert "top_gainers" in data
        assert "top_losers" in data
        assert "timestamp" in data

    def test_market_overview_with_auth(self, client, test_token):
        """Test that endpoint works with auth too"""
        response = client.get(
            "/api/analytics/market-overview",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

    def test_market_overview_fear_greed_range(self, client):
        """Test fear/greed index is in valid range (0-100)"""
        response = client.get("/api/analytics/market-overview")
        data = response.json()

        assert 0 <= data["fear_greed_index"] <= 100

    def test_market_overview_dominance_values(self, client):
        """Test dominance percentages are valid"""
        response = client.get("/api/analytics/market-overview")
        data = response.json()

        assert 0 <= data["btc_dominance"] <= 1
        assert 0 <= data["eth_dominance"] <= 1

    def test_market_overview_top_gainers_structure(self, client):
        """Test top gainers is a list"""
        response = client.get("/api/analytics/market-overview")
        data = response.json()

        assert isinstance(data["top_gainers"], list)
        if len(data["top_gainers"]) > 0:
            gainer = data["top_gainers"][0]
            assert "token" in gainer
            assert "change_24h" in gainer

    def test_market_overview_top_losers_structure(self, client):
        """Test top losers is a list"""
        response = client.get("/api/analytics/market-overview")
        data = response.json()

        assert isinstance(data["top_losers"], list)


@pytest.mark.integration
class TestAnalyticsWorkflows:
    """Integration tests for analytics workflows"""

    def test_complete_analytics_flow(self, client, test_token):
        """
        Test complete analytics workflow:
        1. Get market data
        2. Get technical indicators
        3. Get sentiment
        4. Get portfolio analytics
        """
        # 1. Market data
        market_response = client.get(
            "/api/analytics/market-data?tokens=SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert market_response.status_code == status.HTTP_200_OK

        # 2. Technical indicators
        tech_response = client.get(
            "/api/analytics/technical-indicators/SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert tech_response.status_code == status.HTTP_200_OK

        # 3. Sentiment
        sentiment_response = client.get(
            "/api/analytics/sentiment/SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert sentiment_response.status_code == status.HTTP_200_OK

        # 4. Portfolio
        portfolio_response = client.get(
            "/api/analytics/portfolio",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert portfolio_response.status_code == status.HTTP_200_OK

    def test_multi_token_analysis(self, client, test_token):
        """Test analyzing multiple tokens"""
        tokens = ["SOL", "RAY", "ORCA"]

        for token in tokens:
            # Get technical indicators
            tech_response = client.get(
                f"/api/analytics/technical-indicators/{token}",
                headers={"Authorization": f"Bearer {test_token}"}
            )
            assert tech_response.status_code == status.HTTP_200_OK

            # Get sentiment
            sentiment_response = client.get(
                f"/api/analytics/sentiment/{token}",
                headers={"Authorization": f"Bearer {test_token}"}
            )
            assert sentiment_response.status_code == status.HTTP_200_OK


@pytest.mark.security
class TestAnalyticsSecurity:
    """Security tests for analytics API"""

    def test_sql_injection_in_token_parameter(self, client, test_token):
        """Test SQL injection in token parameter"""
        malicious_tokens = [
            "SOL'; DROP TABLE--",
            "SOL OR 1=1",
            "SOL UNION SELECT * FROM"
        ]

        for token in malicious_tokens:
            response = client.get(
                f"/api/analytics/technical-indicators/{token}",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            # Should not crash
            assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_xss_in_parameters(self, client, test_token):
        """Test XSS in query parameters"""
        xss_token = "<script>alert('xss')</script>"

        response = client.get(
            f"/api/analytics/sentiment/{xss_token}",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        # Should handle safely
        assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_path_traversal_attempt(self, client, test_token):
        """Test path traversal in token parameter"""
        response = client.get(
            "/api/analytics/technical-indicators/../../../etc/passwd",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        # Should not expose filesystem
        assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.performance
class TestAnalyticsPerformance:
    """Performance tests for analytics API"""

    def test_market_data_performance(self, client, test_token, benchmark):
        """Benchmark market data retrieval"""
        def get_market_data():
            response = client.get(
                "/api/analytics/market-data",
                headers={"Authorization": f"Bearer {test_token}"}
            )
            assert response.status_code == status.HTTP_200_OK
            return response

        result = benchmark(get_market_data)
        assert result.status_code == status.HTTP_200_OK

    def test_price_history_large_dataset(self, client, test_token):
        """Test retrieving large price history dataset"""
        response = client.get(
            "/api/analytics/price-history/SOL?limit=1000",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["data"]) <= 1000

    @pytest.mark.slow
    def test_concurrent_analytics_requests(self, client, test_token):
        """Test concurrent analytics requests"""
        import concurrent.futures

        def make_request(endpoint):
            return client.get(
                endpoint,
                headers={"Authorization": f"Bearer {test_token}"}
            )

        endpoints = [
            "/api/analytics/market-data",
            "/api/analytics/technical-indicators/SOL",
            "/api/analytics/sentiment/SOL",
            "/api/analytics/portfolio",
            "/api/analytics/reports/performance"
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, ep) for ep in endpoints * 10]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert all(r.status_code == status.HTTP_200_OK for r in results)
