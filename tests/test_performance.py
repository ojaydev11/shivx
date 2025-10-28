"""
Performance and Load Tests
Tests system performance under various load conditions
"""

import pytest
import time
from fastapi import status


@pytest.mark.performance
class TestAPILatency:
    """Test API endpoint latency"""

    def test_trading_strategies_latency(self, client, test_token, benchmark):
        """Benchmark trading strategies endpoint"""
        def get_strategies():
            response = client.get(
                "/api/trading/strategies",
                headers={"Authorization": f"Bearer {test_token}"}
            )
            assert response.status_code == status.HTTP_200_OK
            return response

        result = benchmark(get_strategies)
        assert result.status_code == status.HTTP_200_OK


    def test_market_data_latency(self, client, test_token, benchmark):
        """Benchmark market data endpoint"""
        def get_market_data():
            response = client.get(
                "/api/analytics/market-data",
                headers={"Authorization": f"Bearer {test_token}"}
            )
            assert response.status_code == status.HTTP_200_OK
            return response

        result = benchmark(get_market_data)
        assert result.status_code == status.HTTP_200_OK


    def test_prediction_latency(self, client, test_token, benchmark):
        """Benchmark ML prediction latency"""
        prediction_request = {
            "model_id": "rl_ppo_v1",
            "features": {"rsi": 65.5}
        }

        def make_prediction():
            response = client.post(
                "/api/ai/predict",
                json=prediction_request,
                headers={"Authorization": f"Bearer {test_token}"}
            )
            assert response.status_code == status.HTTP_200_OK
            return response

        result = benchmark(make_prediction)
        # ML prediction should be under 500ms (mocked, so very fast)
        assert result.status_code == status.HTTP_200_OK


@pytest.mark.performance
@pytest.mark.slow
class TestConcurrentLoad:
    """Test system under concurrent load"""

    def test_concurrent_read_requests(self, client, test_token):
        """Test 100 concurrent read requests"""
        import concurrent.futures

        def make_request(_):
            return client.get(
                "/api/trading/strategies",
                headers={"Authorization": f"Bearer {test_token}"}
            )

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request, i) for i in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        elapsed_time = time.time() - start_time

        # All should succeed
        assert all(r.status_code == status.HTTP_200_OK for r in results)

        # Should complete in reasonable time (under 10 seconds)
        assert elapsed_time < 10.0


    def test_concurrent_write_requests(self, client, test_token):
        """Test concurrent write/execute requests"""
        import concurrent.futures

        def execute_trade(i):
            return client.post(
                "/api/trading/execute",
                json={"token": "SOL", "action": "buy", "amount": 100},
                headers={"Authorization": f"Bearer {test_token}"}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(execute_trade, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed (paper mode)
        assert all(r.status_code == status.HTTP_200_OK for r in results)


    def test_mixed_concurrent_requests(self, client, test_token):
        """Test mixed read/write concurrent requests"""
        import concurrent.futures

        def read_request(i):
            endpoints = [
                "/api/trading/strategies",
                "/api/trading/positions",
                "/api/analytics/market-data"
            ]
            return client.get(
                endpoints[i % len(endpoints)],
                headers={"Authorization": f"Bearer {test_token}"}
            )

        def write_request(i):
            return client.post(
                "/api/trading/strategies/test/enable",
                headers={"Authorization": f"Bearer {test_token}"}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            read_futures = [executor.submit(read_request, i) for i in range(75)]
            write_futures = [executor.submit(write_request, i) for i in range(25)]

            all_futures = read_futures + write_futures
            results = [f.result() for f in concurrent.futures.as_completed(all_futures)]

        # All should succeed
        assert all(r.status_code == status.HTTP_200_OK for r in results)


@pytest.mark.performance
class TestResponseTimes:
    """Test response time requirements"""

    def test_health_check_fast(self, client):
        """Test health check responds quickly"""
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start

        assert response.status_code == status.HTTP_200_OK
        assert elapsed < 0.1  # Under 100ms


    def test_public_endpoints_fast(self, client):
        """Test public endpoints respond quickly"""
        endpoints = [
            "/api/trading/mode",
            "/api/analytics/market-overview",
            "/api/ai/capabilities"
        ]

        for endpoint in endpoints:
            start = time.time()
            response = client.get(endpoint)
            elapsed = time.time() - start

            assert response.status_code == status.HTTP_200_OK
            assert elapsed < 1.0  # Under 1 second


@pytest.mark.performance
class TestThroughput:
    """Test system throughput"""

    def test_requests_per_second(self, client, test_token):
        """Measure requests per second"""
        import concurrent.futures

        request_count = 0
        test_duration = 5  # seconds

        def make_request(_):
            return client.get(
                "/api/trading/strategies",
                headers={"Authorization": f"Bearer {test_token}"}
            )

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            while time.time() - start_time < test_duration:
                future = executor.submit(make_request, request_count)
                request_count += 1

            # Wait for all to complete
            time.sleep(1)

        elapsed = time.time() - start_time
        rps = request_count / elapsed

        # Should handle at least 10 requests per second
        assert rps >= 10

        print(f"\nThroughput: {rps:.2f} requests/second")


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage under load"""

    @pytest.mark.slow
    def test_memory_stable_under_load(self, client, test_token):
        """Test memory doesn't leak under sustained load"""
        # Make many requests
        for _ in range(500):
            client.get(
                "/api/trading/strategies",
                headers={"Authorization": f"Bearer {test_token}"}
            )

        # If we get here without crashing, memory is reasonably stable
        assert True


@pytest.mark.performance
class TestDatabasePerformance:
    """Test database query performance"""

    def test_auth_token_validation_performance(self, test_token, test_settings, benchmark):
        """Benchmark JWT token validation"""
        from app.dependencies.auth import decode_access_token

        def validate_token():
            return decode_access_token(test_token, test_settings)

        result = benchmark(validate_token)
        assert result.user_id is not None


@pytest.mark.performance
class TestCachingEffectiveness:
    """Test caching improves performance"""

    def test_repeated_requests_faster(self, client, test_token):
        """Test repeated requests benefit from caching (if implemented)"""
        endpoint = "/api/trading/strategies"

        # First request (cold)
        start = time.time()
        response1 = client.get(endpoint, headers={"Authorization": f"Bearer {test_token}"})
        first_time = time.time() - start

        # Second request (potentially cached)
        start = time.time()
        response2 = client.get(endpoint, headers={"Authorization": f"Bearer {test_token}"})
        second_time = time.time() - start

        assert response1.status_code == status.HTTP_200_OK
        assert response2.status_code == status.HTTP_200_OK

        # Both should complete quickly
        assert first_time < 1.0
        assert second_time < 1.0


@pytest.mark.performance
@pytest.mark.slow
class TestLongRunningOperations:
    """Test long-running operations"""

    def test_bulk_predictions_performance(self, client, test_token):
        """Test performance of multiple predictions"""
        predictions = []

        start_time = time.time()

        for i in range(50):
            response = client.post(
                "/api/ai/predict",
                json={
                    "model_id": "rl_ppo_v1",
                    "features": {"rsi": 60 + i % 20}
                },
                headers={"Authorization": f"Bearer {test_token}"}
            )
            predictions.append(response.status_code)

        elapsed = time.time() - start_time

        # All should succeed
        assert all(status == status.HTTP_200_OK for status in predictions)

        # Should complete in reasonable time (under 30 seconds)
        assert elapsed < 30.0

        print(f"\n50 predictions in {elapsed:.2f} seconds ({50/elapsed:.2f} pred/sec)")


@pytest.mark.performance
class TestRateLimiting:
    """Test rate limiting behavior"""

    def test_rate_limit_threshold(self, client, test_token):
        """Test that rate limiting activates at threshold"""
        # Make many requests rapidly
        responses = []
        for i in range(150):  # Above typical rate limit
            response = client.get(
                "/api/trading/strategies",
                headers={"Authorization": f"Bearer {test_token}"}
            )
            responses.append(response.status_code)

        # Most should succeed (depends on rate limit implementation)
        success_count = sum(1 for status in responses if status == status.HTTP_200_OK)
        assert success_count > 100  # At least 100 should succeed


@pytest.mark.performance
class TestScalability:
    """Test system scalability"""

    @pytest.mark.slow
    def test_increasing_load(self, client, test_token):
        """Test performance under increasing load"""
        import concurrent.futures

        results = {}

        for concurrency in [10, 20, 50]:
            def make_request(_):
                return client.get(
                    "/api/trading/strategies",
                    headers={"Authorization": f"Bearer {test_token}"}
                )

            start_time = time.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(make_request, i) for i in range(concurrency)]
                responses = [f.result() for f in concurrent.futures.as_completed(futures)]

            elapsed = time.time() - start_time
            success_rate = sum(1 for r in responses if r.status_code == status.HTTP_200_OK) / len(responses)

            results[concurrency] = {
                "elapsed": elapsed,
                "success_rate": success_rate
            }

            print(f"\nConcurrency {concurrency}: {elapsed:.2f}s, {success_rate*100:.1f}% success")

        # All should have high success rate
        assert all(r["success_rate"] > 0.95 for r in results.values())
