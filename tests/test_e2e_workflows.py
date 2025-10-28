"""
End-to-End Integration Tests
Tests complete user workflows across multiple system components
"""

import pytest
from fastapi import status


@pytest.mark.e2e
class TestCompleteUserJourney:
    """Test complete user journeys from signup to trading"""

    def test_user_workflow_market_analysis_to_trade(self, client, test_token):
        """
        Complete workflow:
        1. Check trading mode
        2. Get market data
        3. Get technical indicators
        4. Get AI prediction
        5. Get trading signal
        6. Execute trade
        7. Check position
        8. Review performance
        """
        # 1. Check trading mode
        mode_response = client.get("/api/trading/mode")
        assert mode_response.status_code == status.HTTP_200_OK
        assert mode_response.json()["mode"] == "paper"

        # 2. Get market data
        market_response = client.get(
            "/api/analytics/market-data?tokens=SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert market_response.status_code == status.HTTP_200_OK

        # 3. Get technical indicators
        tech_response = client.get(
            "/api/analytics/technical-indicators/SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert tech_response.status_code == status.HTTP_200_OK

        # 4. Get AI prediction
        pred_request = {
            "model_id": "rl_ppo_v1",
            "features": {"rsi": 65.5, "macd": 1.25}
        }
        pred_response = client.post(
            "/api/ai/predict",
            json=pred_request,
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert pred_response.status_code == status.HTTP_200_OK

        # 5. Get trading signal
        signal_response = client.get(
            "/api/trading/signals?token=SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert signal_response.status_code == status.HTTP_200_OK

        # 6. Execute trade
        trade_request = {
            "token": "SOL",
            "action": "buy",
            "amount": 100.0
        }
        trade_response = client.post(
            "/api/trading/execute",
            json=trade_request,
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert trade_response.status_code == status.HTTP_200_OK

        # 7. Check positions
        positions_response = client.get(
            "/api/trading/positions",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert positions_response.status_code == status.HTTP_200_OK

        # 8. Review performance
        perf_response = client.get(
            "/api/trading/performance",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert perf_response.status_code == status.HTTP_200_OK


    def test_analytics_comprehensive_workflow(self, client, test_token):
        """
        Analytics workflow:
        1. Get market overview
        2. Get market data for multiple tokens
        3. Analyze technical indicators
        4. Check sentiment
        5. Review portfolio
        6. Generate performance report
        """
        # 1. Market overview (public)
        overview_response = client.get("/api/analytics/market-overview")
        assert overview_response.status_code == status.HTTP_200_OK

        # 2. Multi-token market data
        tokens_response = client.get(
            "/api/analytics/market-data?tokens=SOL,RAY,ORCA",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert tokens_response.status_code == status.HTTP_200_OK

        # 3. Technical indicators for each
        for token in ["SOL", "RAY", "ORCA"]:
            tech_response = client.get(
                f"/api/analytics/technical-indicators/{token}",
                headers={"Authorization": f"Bearer {test_token}"}
            )
            assert tech_response.status_code == status.HTTP_200_OK

        # 4. Sentiment analysis
        sentiment_response = client.get(
            "/api/analytics/sentiment/SOL",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert sentiment_response.status_code == status.HTTP_200_OK

        # 5. Portfolio analytics
        portfolio_response = client.get(
            "/api/analytics/portfolio",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert portfolio_response.status_code == status.HTTP_200_OK

        # 6. Performance report
        report_response = client.get(
            "/api/analytics/reports/performance?period=30d",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert report_response.status_code == status.HTTP_200_OK


    def test_ai_ml_complete_workflow(self, client, test_token, admin_token):
        """
        AI/ML workflow:
        1. List available models
        2. Get model details
        3. Make prediction
        4. Get explainability
        5. List training jobs
        6. Start new training (admin)
        7. Check training status
        """
        # 1. List models
        models_response = client.get(
            "/api/ai/models",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert models_response.status_code == status.HTTP_200_OK

        # 2. Get specific model
        model_response = client.get(
            "/api/ai/models/rl_ppo_v1",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert model_response.status_code == status.HTTP_200_OK

        # 3. Make prediction
        pred_response = client.post(
            "/api/ai/predict",
            json={"model_id": "rl_ppo_v1", "features": {"rsi": 65.5}, "explain": True},
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert pred_response.status_code == status.HTTP_200_OK
        prediction_id = pred_response.json()["prediction_id"]

        # 4. Get explainability
        explain_response = client.get(
            f"/api/ai/explainability/{prediction_id}",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert explain_response.status_code == status.HTTP_200_OK

        # 5. List training jobs
        jobs_response = client.get(
            "/api/ai/training-jobs",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert jobs_response.status_code == status.HTTP_200_OK

        # 6. Start training (admin only)
        train_response = client.post(
            "/api/ai/train",
            json={
                "model_name": "Test Model",
                "model_type": "supervised",
                "dataset_id": "test",
                "epochs": 10
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert train_response.status_code == status.HTTP_200_OK
        job_id = train_response.json()["job_id"]

        # 7. Check training status
        status_response = client.get(
            f"/api/ai/training-jobs/{job_id}",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert status_response.status_code == status.HTTP_200_OK


@pytest.mark.e2e
class TestPermissionBasedWorkflows:
    """Test workflows with different permission levels"""

    def test_readonly_user_workflow(self, client, readonly_token):
        """Test that readonly user can view but not modify"""
        # Can view
        assert client.get(
            "/api/trading/strategies",
            headers={"Authorization": f"Bearer {readonly_token}"}
        ).status_code == status.HTTP_200_OK

        # Cannot modify
        assert client.post(
            "/api/trading/strategies/test/enable",
            headers={"Authorization": f"Bearer {readonly_token}"}
        ).status_code == status.HTTP_403_FORBIDDEN

        # Cannot execute
        assert client.post(
            "/api/trading/execute",
            json={"token": "SOL", "action": "buy", "amount": 100},
            headers={"Authorization": f"Bearer {readonly_token}"}
        ).status_code == status.HTTP_403_FORBIDDEN


    def test_admin_workflow(self, client, admin_token):
        """Test admin has full access"""
        # Can view
        assert client.get(
            "/api/ai/models",
            headers={"Authorization": f"Bearer {admin_token}"}
        ).status_code == status.HTTP_200_OK

        # Can deploy
        assert client.post(
            "/api/ai/models/test/deploy",
            headers={"Authorization": f"Bearer {admin_token}"}
        ).status_code == status.HTTP_200_OK

        # Can start training
        assert client.post(
            "/api/ai/train",
            json={
                "model_name": "Admin Test",
                "model_type": "supervised",
                "dataset_id": "test"
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        ).status_code == status.HTTP_200_OK


@pytest.mark.e2e
@pytest.mark.slow
class TestHighVolumeWorkflows:
    """Test workflows under high volume"""

    def test_bulk_trade_execution(self, client, test_token):
        """Test executing multiple trades in sequence"""
        trades = [
            {"token": "SOL", "action": "buy", "amount": 100},
            {"token": "RAY", "action": "buy", "amount": 50},
            {"token": "ORCA", "action": "buy", "amount": 75},
            {"token": "SOL", "action": "sell", "amount": 50},
            {"token": "RAY", "action": "sell", "amount": 25},
        ]

        results = []
        for trade in trades:
            response = client.post(
                "/api/trading/execute",
                json=trade,
                headers={"Authorization": f"Bearer {test_token}"}
            )
            results.append(response.status_code)

        assert all(status == status.HTTP_200_OK for status in results)


    def test_concurrent_analytics_queries(self, client, test_token):
        """Test concurrent analytics queries"""
        import concurrent.futures

        endpoints = [
            "/api/analytics/market-data",
            "/api/analytics/technical-indicators/SOL",
            "/api/analytics/sentiment/SOL",
            "/api/analytics/portfolio",
            "/api/analytics/reports/performance"
        ]

        def query(endpoint):
            return client.get(
                endpoint,
                headers={"Authorization": f"Bearer {test_token}"}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(query, ep) for ep in endpoints]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(r.status_code == status.HTTP_200_OK for r in results)


@pytest.mark.e2e
class TestErrorRecoveryWorkflows:
    """Test system behavior during and after errors"""

    def test_invalid_trade_recovery(self, client, test_token):
        """Test that invalid trade doesn't break subsequent trades"""
        # Invalid trade (negative amount)
        invalid_response = client.post(
            "/api/trading/execute",
            json={"token": "SOL", "action": "buy", "amount": -100},
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert invalid_response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Valid trade should still work
        valid_response = client.post(
            "/api/trading/execute",
            json={"token": "SOL", "action": "buy", "amount": 100},
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert valid_response.status_code == status.HTTP_200_OK


    def test_auth_failure_recovery(self, client, test_token):
        """Test recovery from auth failures"""
        # Failed auth
        invalid_auth = client.get(
            "/api/trading/strategies",
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert invalid_auth.status_code == status.HTTP_401_UNAUTHORIZED

        # Valid auth should work immediately after
        valid_auth = client.get(
            "/api/trading/strategies",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert valid_auth.status_code == status.HTTP_200_OK
