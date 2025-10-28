"""
Trading API Integration Tests
Coverage Target: 100% of app/routers/trading.py

Tests all trading endpoints with authentication, permissions, and error cases
"""

import pytest
from fastapi import status
from datetime import datetime


@pytest.mark.unit
class TestTradingStrategies:
    """Test GET /api/trading/strategies endpoint"""

    def test_list_strategies_with_auth(self, client, test_token):
        """Test listing strategies with valid authentication"""
        response = client.get(
            "/api/trading/strategies",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Verify strategy structure
        strategy = data[0]
        assert "name" in strategy
        assert "enabled" in strategy
        assert "max_position_size" in strategy
        assert "stop_loss_pct" in strategy
        assert "take_profit_pct" in strategy
        assert "risk_tolerance" in strategy

    def test_list_strategies_without_auth(self, client):
        """Test listing strategies without authentication - should fail"""
        response = client.get("/api/trading/strategies")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_list_strategies_with_invalid_token(self, client, invalid_token):
        """Test with invalid token"""
        response = client.get(
            "/api/trading/strategies",
            headers={"Authorization": f"Bearer {invalid_token}"}
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_list_strategies_with_expired_token(self, client, expired_token):
        """Test with expired token"""
        response = client.get(
            "/api/trading/strategies",
            headers={"Authorization": f"Bearer {expired_token}"}
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_list_strategies_readonly_permission(self, client, readonly_token):
        """Test with READ permission - should succeed"""
        response = client.get(
            "/api/trading/strategies",
            headers={"Authorization": f"Bearer {readonly_token}"}
        )

        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
class TestTradingPositions:
    """Test GET /api/trading/positions endpoint"""

    def test_list_positions_with_auth(self, client, test_token):
        """Test listing positions with authentication"""
        response = client.get(
            "/api/trading/positions",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

    def test_list_positions_without_auth(self, client):
        """Test without authentication"""
        response = client.get("/api/trading/positions")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_list_positions_with_status_filter(self, client, test_token):
        """Test filtering by status parameter"""
        response = client.get(
            "/api/trading/positions?status=open",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Verify positions match filter (if any returned)
        for position in data:
            assert position["status"] == "open"

    def test_list_positions_invalid_readonly(self, client, readonly_token):
        """Test with READ permission - should succeed"""
        response = client.get(
            "/api/trading/positions",
            headers={"Authorization": f"Bearer {readonly_token}"}
        )

        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
class TestTradingSignals:
    """Test GET /api/trading/signals endpoint"""

    def test_get_signals_with_auth(self, client, test_token):
        """Test getting trading signals"""
        response = client.get(
            "/api/trading/signals",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

        if len(data) > 0:
            signal = data[0]
            assert "signal_id" in signal
            assert "token" in signal
            assert "action" in signal
            assert "confidence" in signal
            assert "strategy" in signal
            assert "reasoning" in signal

    def test_get_signals_without_auth(self, client):
        """Test without authentication"""
        response = client.get("/api/trading/signals")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_signals_with_token_filter(self, client, test_token):
        """Test filtering by token"""
        response = client.get(
            "/api/trading/signals?token=SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

    def test_get_signals_with_strategy_filter(self, client, test_token):
        """Test filtering by strategy"""
        response = client.get(
            "/api/trading/signals?strategy=RL+Trading",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

    def test_get_signals_feature_disabled(self, client, test_token, test_settings):
        """Test when advanced trading feature is disabled"""
        # Temporarily disable feature
        test_settings.feature_advanced_trading = False

        response = client.get(
            "/api/trading/signals",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        # Should return 503 Service Unavailable
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "disabled" in response.json()["detail"].lower()

        # Re-enable
        test_settings.feature_advanced_trading = True


@pytest.mark.unit
class TestTradeExecution:
    """Test POST /api/trading/execute endpoint"""

    def test_execute_trade_paper_mode(self, client, test_token):
        """Test executing trade in paper mode"""
        trade_data = {
            "token": "SOL",
            "action": "buy",
            "amount": 100.0,
            "slippage_bps": 50
        }

        response = client.post(
            "/api/trading/execute",
            json=trade_data,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        result = response.json()

        assert "trade_id" in result
        assert result["token"] == "SOL"
        assert result["action"] == "buy"
        assert result["status"] == "success"
        assert "executed_at" in result

    def test_execute_trade_without_auth(self, client):
        """Test without authentication"""
        trade_data = {
            "token": "SOL",
            "action": "buy",
            "amount": 100.0
        }

        response = client.post("/api/trading/execute", json=trade_data)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_execute_trade_insufficient_permission(self, client, readonly_token):
        """Test with READ permission - should fail (needs EXECUTE)"""
        trade_data = {
            "token": "SOL",
            "action": "buy",
            "amount": 100.0
        }

        response = client.post(
            "/api/trading/execute",
            json=trade_data,
            headers={"Authorization": f"Bearer {readonly_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_execute_trade_invalid_data(self, client, test_token):
        """Test with invalid trade data"""
        # Missing required fields
        trade_data = {
            "token": "SOL"
            # Missing action and amount
        }

        response = client.post(
            "/api/trading/execute",
            json=trade_data,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_execute_trade_invalid_amount(self, client, test_token):
        """Test with invalid amount (negative)"""
        trade_data = {
            "token": "SOL",
            "action": "buy",
            "amount": -100.0  # Invalid
        }

        response = client.post(
            "/api/trading/execute",
            json=trade_data,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_execute_trade_invalid_slippage(self, client, test_token):
        """Test with invalid slippage (too high)"""
        trade_data = {
            "token": "SOL",
            "action": "buy",
            "amount": 100.0,
            "slippage_bps": 2000  # Above max 1000
        }

        response = client.post(
            "/api/trading/execute",
            json=trade_data,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_execute_trade_sell_action(self, client, test_token):
        """Test sell action"""
        trade_data = {
            "token": "SOL",
            "action": "sell",
            "amount": 50.0
        }

        response = client.post(
            "/api/trading/execute",
            json=trade_data,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert result["action"] == "sell"

    def test_execute_trade_live_mode_not_implemented(self, client, test_token, test_settings):
        """Test live mode returns not implemented"""
        # Switch to live mode
        from config.settings import TradingMode
        original_mode = test_settings.trading_mode
        test_settings.trading_mode = TradingMode.LIVE

        trade_data = {
            "token": "SOL",
            "action": "buy",
            "amount": 100.0
        }

        response = client.post(
            "/api/trading/execute",
            json=trade_data,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED

        # Restore
        test_settings.trading_mode = original_mode


@pytest.mark.unit
class TestTradingPerformance:
    """Test GET /api/trading/performance endpoint"""

    def test_get_performance_default(self, client, test_token):
        """Test getting performance metrics with default period"""
        response = client.get(
            "/api/trading/performance",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "period" in data
        assert "total_trades" in data
        assert "winning_trades" in data
        assert "losing_trades" in data
        assert "win_rate" in data
        assert "total_pnl" in data
        assert "sharpe_ratio" in data
        assert "max_drawdown" in data

    def test_get_performance_different_periods(self, client, test_token):
        """Test different time periods"""
        periods = ["1h", "24h", "7d", "30d", "all"]

        for period in periods:
            response = client.get(
                f"/api/trading/performance?period={period}",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["period"] == period

    def test_get_performance_without_auth(self, client):
        """Test without authentication"""
        response = client.get("/api/trading/performance")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_performance_readonly_permission(self, client, readonly_token):
        """Test with READ permission"""
        response = client.get(
            "/api/trading/performance",
            headers={"Authorization": f"Bearer {readonly_token}"}
        )

        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
class TestStrategyManagement:
    """Test strategy enable/disable endpoints"""

    def test_enable_strategy(self, client, test_token):
        """Test POST /api/trading/strategies/{name}/enable"""
        response = client.post(
            "/api/trading/strategies/test_strategy/enable",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["enabled"] is True
        assert "strategy" in data

    def test_disable_strategy(self, client, test_token):
        """Test POST /api/trading/strategies/{name}/disable"""
        response = client.post(
            "/api/trading/strategies/test_strategy/disable",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["enabled"] is False

    def test_enable_strategy_without_auth(self, client):
        """Test without authentication"""
        response = client.post("/api/trading/strategies/test/enable")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_enable_strategy_insufficient_permission(self, client, readonly_token):
        """Test with READ permission - should fail (needs WRITE)"""
        response = client.post(
            "/api/trading/strategies/test/enable",
            headers={"Authorization": f"Bearer {readonly_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_disable_strategy_without_auth(self, client):
        """Test disable without authentication"""
        response = client.post("/api/trading/strategies/test/disable")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_enable_strategy_with_special_chars(self, client, test_token):
        """Test with strategy name containing special characters"""
        response = client.post(
            "/api/trading/strategies/RL%20Trading%20(PPO)/enable",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
class TestTradingMode:
    """Test GET /api/trading/mode endpoint"""

    def test_get_trading_mode(self, client):
        """Test getting trading mode - public endpoint (no auth)"""
        response = client.get("/api/trading/mode")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "mode" in data
        assert data["mode"] in ["paper", "live"]
        assert "max_position_size" in data
        assert "stop_loss_pct" in data
        assert "take_profit_pct" in data

    def test_get_trading_mode_with_auth(self, client, test_token):
        """Test that endpoint works with auth too"""
        response = client.get(
            "/api/trading/mode",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

    def test_trading_mode_values(self, client):
        """Test that returned values are valid"""
        response = client.get("/api/trading/mode")
        data = response.json()

        assert isinstance(data["max_position_size"], (int, float))
        assert data["max_position_size"] >= 0
        assert isinstance(data["stop_loss_pct"], (int, float))
        assert 0 <= data["stop_loss_pct"] <= 1
        assert isinstance(data["take_profit_pct"], (int, float))
        assert 0 <= data["take_profit_pct"] <= 1


@pytest.mark.integration
class TestTradingWorkflow:
    """Integration tests for complete trading workflows"""

    def test_complete_trading_flow(self, client, test_token):
        """
        Test complete workflow:
        1. Get strategies
        2. Get signals
        3. Execute trade
        4. Check performance
        """
        # 1. Get strategies
        strategies_response = client.get(
            "/api/trading/strategies",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert strategies_response.status_code == status.HTTP_200_OK
        strategies = strategies_response.json()
        assert len(strategies) > 0

        # 2. Get signals
        signals_response = client.get(
            "/api/trading/signals",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert signals_response.status_code == status.HTTP_200_OK

        # 3. Execute trade
        trade_data = {
            "token": "SOL",
            "action": "buy",
            "amount": 100.0
        }
        execute_response = client.post(
            "/api/trading/execute",
            json=trade_data,
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert execute_response.status_code == status.HTTP_200_OK

        # 4. Check performance
        perf_response = client.get(
            "/api/trading/performance",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert perf_response.status_code == status.HTTP_200_OK

    def test_strategy_enable_disable_flow(self, client, test_token):
        """Test enabling and disabling strategy"""
        strategy_name = "test_strategy"

        # Enable
        enable_response = client.post(
            f"/api/trading/strategies/{strategy_name}/enable",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert enable_response.status_code == status.HTTP_200_OK
        assert enable_response.json()["enabled"] is True

        # Disable
        disable_response = client.post(
            f"/api/trading/strategies/{strategy_name}/disable",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert disable_response.status_code == status.HTTP_200_OK
        assert disable_response.json()["enabled"] is False


@pytest.mark.unit
class TestTradingAPIErrorHandling:
    """Test error handling across trading endpoints"""

    def test_malformed_json(self, client, test_token):
        """Test with malformed JSON"""
        response = client.post(
            "/api/trading/execute",
            data="not valid json",
            headers={
                "Authorization": f"Bearer {test_token}",
                "Content-Type": "application/json"
            }
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_content_type(self, client, test_token):
        """Test POST without content-type header"""
        response = client.post(
            "/api/trading/execute",
            data='{"token": "SOL", "action": "buy", "amount": 100}',
            headers={"Authorization": f"Bearer {test_token}"}
        )

        # FastAPI should handle this gracefully
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]

    def test_extra_fields_ignored(self, client, test_token):
        """Test that extra fields in request are ignored"""
        trade_data = {
            "token": "SOL",
            "action": "buy",
            "amount": 100.0,
            "extra_field": "should be ignored"
        }

        response = client.post(
            "/api/trading/execute",
            json=trade_data,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        # Should succeed (Pydantic ignores extra fields by default)
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.security
class TestTradingAPISecurity:
    """Security tests for trading API"""

    def test_sql_injection_in_parameters(self, client, test_token):
        """Test SQL injection attempts in query parameters"""
        malicious_inputs = [
            "SOL'; DROP TABLE positions--",
            "SOL OR 1=1",
            "SOL UNION SELECT * FROM users"
        ]

        for malicious_input in malicious_inputs:
            response = client.get(
                f"/api/trading/signals?token={malicious_input}",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            # Should not cause server error
            assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST]

    def test_xss_in_strategy_name(self, client, test_token):
        """Test XSS attempts in strategy names"""
        xss_attempts = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ]

        for xss in xss_attempts:
            response = client.post(
                f"/api/trading/strategies/{xss}/enable",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            # Should handle safely
            assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_permission_escalation_attempt(self, client, readonly_token):
        """Test attempting to execute trades with READ-only permission"""
        trade_data = {
            "token": "SOL",
            "action": "buy",
            "amount": 100.0
        }

        response = client.post(
            "/api/trading/execute",
            json=trade_data,
            headers={"Authorization": f"Bearer {readonly_token}"}
        )

        # Should be forbidden
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_token_in_error_messages(self, client):
        """Ensure tokens are not leaked in error messages"""
        fake_token = "supersecrettoken123"
        response = client.get(
            "/api/trading/strategies",
            headers={"Authorization": f"Bearer {fake_token}"}
        )

        # Should not include token in response
        assert fake_token not in response.text


@pytest.mark.performance
class TestTradingAPIPerformance:
    """Performance tests for trading API"""

    def test_list_strategies_performance(self, client, test_token, benchmark):
        """Benchmark strategy listing performance"""
        def list_strategies():
            response = client.get(
                "/api/trading/strategies",
                headers={"Authorization": f"Bearer {test_token}"}
            )
            assert response.status_code == status.HTTP_200_OK
            return response

        result = benchmark(list_strategies)
        assert result.status_code == status.HTTP_200_OK

    def test_execute_trade_performance(self, client, test_token, benchmark):
        """Benchmark trade execution performance"""
        trade_data = {
            "token": "SOL",
            "action": "buy",
            "amount": 100.0
        }

        def execute_trade():
            response = client.post(
                "/api/trading/execute",
                json=trade_data,
                headers={"Authorization": f"Bearer {test_token}"}
            )
            assert response.status_code == status.HTTP_200_OK
            return response

        result = benchmark(execute_trade)
        assert result.status_code == status.HTTP_200_OK

    @pytest.mark.slow
    def test_concurrent_requests(self, client, test_token):
        """Test handling of concurrent requests"""
        import concurrent.futures

        def make_request(_):
            return client.get(
                "/api/trading/strategies",
                headers={"Authorization": f"Bearer {test_token}"}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert all(r.status_code == status.HTTP_200_OK for r in results)
