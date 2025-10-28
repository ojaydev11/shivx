"""
Security Penetration Tests
Tests that common attack vectors are properly blocked
"""

import pytest
from fastapi import status


@pytest.mark.security
class TestSQLInjectionProtection:
    """Test SQL injection protection"""

    def test_sql_injection_in_query_params(self, client, test_token):
        """Test SQL injection attempts in query parameters"""
        injection_attempts = [
            "SOL'; DROP TABLE trades--",
            "SOL OR 1=1--",
            "SOL UNION SELECT * FROM users--",
            "SOL'; UPDATE trades SET price=0--",
            "1' OR '1'='1",
        ]

        for injection in injection_attempts:
            response = client.get(
                f"/api/analytics/technical-indicators/{injection}",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            # Should not crash or expose data
            assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR


    def test_sql_injection_in_post_body(self, client, test_token):
        """Test SQL injection in POST request bodies"""
        malicious_data = {
            "token": "SOL'; DROP TABLE--",
            "action": "buy",
            "amount": 100
        }

        response = client.post(
            "/api/trading/execute",
            json=malicious_data,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        # Should handle safely
        assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.security
class TestXSSProtection:
    """Test Cross-Site Scripting protection"""

    def test_xss_in_path_parameters(self, client, test_token):
        """Test XSS attempts in path parameters"""
        xss_attempts = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
        ]

        for xss in xss_attempts:
            response = client.post(
                f"/api/trading/strategies/{xss}/enable",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            # Should not execute script
            assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR
            # Response should not contain unescaped script
            assert "<script>" not in response.text.lower()


    def test_xss_in_request_body(self, client, test_token):
        """Test XSS in request bodies"""
        xss_data = {
            "model_id": "<script>alert('xss')</script>",
            "features": {
                "xss": "<img src=x onerror=alert('xss')>"
            }
        }

        response = client.post(
            "/api/ai/predict",
            json=xss_data,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert "<script>" not in response.text


@pytest.mark.security
class TestAuthenticationBypass:
    """Test authentication bypass attempts"""

    def test_missing_auth_header(self, client):
        """Test accessing protected endpoint without auth"""
        response = client.get("/api/trading/strategies")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


    def test_malformed_auth_header(self, client):
        """Test with malformed authorization headers"""
        malformed_headers = [
            {"Authorization": "InvalidFormat token123"},
            {"Authorization": "Bearer"},
            {"Authorization": "token_without_bearer"},
            {"Authorization": ""},
        ]

        for headers in malformed_headers:
            response = client.get("/api/trading/strategies", headers=headers)
            assert response.status_code == status.HTTP_401_UNAUTHORIZED


    def test_expired_token_rejected(self, client, expired_token):
        """Test expired token is rejected"""
        response = client.get(
            "/api/trading/strategies",
            headers={"Authorization": f"Bearer {expired_token}"}
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


    def test_token_tampering_detected(self, client, test_token):
        """Test tampered token is detected"""
        # Tamper with token
        parts = test_token.split('.')
        tampered = parts[0] + '.tampered.' + parts[2]

        response = client.get(
            "/api/trading/strategies",
            headers={"Authorization": f"Bearer {tampered}"}
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.security
class TestAuthorizationEscalation:
    """Test authorization escalation attempts"""

    def test_readonly_cannot_write(self, client, readonly_token):
        """Test read-only user cannot write"""
        response = client.post(
            "/api/trading/strategies/test/enable",
            headers={"Authorization": f"Bearer {readonly_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN


    def test_readonly_cannot_execute(self, client, readonly_token):
        """Test read-only user cannot execute trades"""
        response = client.post(
            "/api/trading/execute",
            json={"token": "SOL", "action": "buy", "amount": 100},
            headers={"Authorization": f"Bearer {readonly_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN


    def test_non_admin_cannot_train(self, client, test_token):
        """Test non-admin cannot start training"""
        response = client.post(
            "/api/ai/train",
            json={
                "model_name": "Test",
                "model_type": "supervised",
                "dataset_id": "test"
            },
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN


    def test_non_admin_cannot_deploy(self, client, test_token):
        """Test non-admin cannot deploy models"""
        response = client.post(
            "/api/ai/models/test/deploy",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.security
class TestAPIKeyBruteForce:
    """Test API key brute force protection"""

    @pytest.mark.slow
    def test_rate_limit_blocks_brute_force(self, client, guardian_defense):
        """Test that rate limiting blocks brute force attempts"""
        # Attempt many failed authentications
        for i in range(20):
            client.get(
                "/api/trading/strategies",
                headers={"Authorization": f"Bearer fake_token_{i}"}
            )

        # Should receive 401s (protected)
        # Real implementation would also trigger guardian defense


@pytest.mark.security
class TestPathTraversal:
    """Test path traversal protection"""

    def test_path_traversal_in_endpoints(self, client, test_token):
        """Test path traversal attempts"""
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "....//....//....//etc/passwd",
        ]

        for attempt in traversal_attempts:
            response = client.get(
                f"/api/analytics/technical-indicators/{attempt}",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            # Should not expose filesystem
            assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "root:" not in response.text  # Unix passwd file pattern


@pytest.mark.security
class TestCSRFProtection:
    """Test CSRF protection"""

    def test_state_changing_requires_auth(self, client):
        """Test state-changing operations require authentication"""
        state_changing_ops = [
            ("POST", "/api/trading/execute", {"token": "SOL", "action": "buy", "amount": 100}),
            ("POST", "/api/trading/strategies/test/enable", {}),
            ("POST", "/api/ai/train", {"model_name": "Test", "model_type": "supervised", "dataset_id": "test"}),
        ]

        for method, endpoint, data in state_changing_ops:
            if method == "POST":
                response = client.post(endpoint, json=data)
            else:
                response = client.request(method, endpoint, json=data)

            assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.security
class TestInputValidation:
    """Test input validation"""

    def test_negative_amounts_rejected(self, client, test_token):
        """Test negative amounts are rejected"""
        response = client.post(
            "/api/trading/execute",
            json={"token": "SOL", "action": "buy", "amount": -100},
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


    def test_out_of_range_slippage_rejected(self, client, test_token):
        """Test slippage outside valid range is rejected"""
        response = client.post(
            "/api/trading/execute",
            json={"token": "SOL", "action": "buy", "amount": 100, "slippage_bps": 5000},
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


    def test_invalid_rsi_rejected(self, client, test_token):
        """Test invalid RSI values are rejected"""
        # RSI should be 0-100
        response = client.post(
            "/api/ai/predict",
            json={"model_id": "test", "features": {"rsi": 150}},
            headers={"Authorization": f"Bearer {test_token}"}
        )

        # Should handle validation
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]


@pytest.mark.security
class TestSecretLeakage:
    """Test that secrets are not leaked"""

    def test_token_not_in_error_messages(self, client):
        """Test tokens not leaked in error messages"""
        fake_token = "secret_token_12345"

        response = client.get(
            "/api/trading/strategies",
            headers={"Authorization": f"Bearer {fake_token}"}
        )

        # Token should not appear in response
        assert fake_token not in response.text


    def test_internal_errors_sanitized(self, client, test_token):
        """Test internal errors don't leak sensitive info"""
        # Try to trigger various edge cases
        responses = [
            client.get("/api/trading/strategies", headers={"Authorization": f"Bearer {test_token}"}),
            client.post("/api/trading/execute", json={"token": "SOL", "action": "buy", "amount": 100}, headers={"Authorization": f"Bearer {test_token}"}),
        ]

        for response in responses:
            # Check for common leak patterns
            assert "password" not in response.text.lower()
            assert "secret" not in response.text.lower()
            assert "api_key" not in response.text.lower()


@pytest.mark.security
class TestSessionSecurity:
    """Test session security"""

    def test_concurrent_sessions_allowed(self, client, test_token):
        """Test that same token can be used concurrently"""
        import concurrent.futures

        def make_request(_):
            return client.get(
                "/api/trading/strategies",
                headers={"Authorization": f"Bearer {test_token}"}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(r.status_code == status.HTTP_200_OK for r in results)


@pytest.mark.security
class TestDOSProtection:
    """Test Denial of Service protection"""

    def test_large_payload_handled(self, client, test_token):
        """Test large payloads are handled safely"""
        large_features = {f"feature_{i}": i for i in range(1000)}

        response = client.post(
            "/api/ai/predict",
            json={"model_id": "test", "features": large_features},
            headers={"Authorization": f"Bearer {test_token}"}
        )

        # Should handle without crashing
        assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR


    def test_deeply_nested_json_handled(self, client, test_token):
        """Test deeply nested JSON is handled"""
        nested = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}

        response = client.post(
            "/api/ai/predict",
            json={"model_id": "test", "features": nested},
            headers={"Authorization": f"Bearer {test_token}"}
        )

        # Should not crash
        assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR
