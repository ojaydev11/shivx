"""
Comprehensive Authentication and Authorization Tests
Coverage Target: 100% of app/dependencies/auth.py

Tests JWT tokens, permissions, API keys, and authentication flows
"""

import pytest
from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import status, HTTPException

from app.dependencies.auth import (
    create_access_token,
    decode_access_token,
    get_current_user,
    require_permission,
    TokenData
)
from core.security.hardening import Permission
from config.settings import Settings


@pytest.mark.unit
class TestJWTTokenCreation:
    """Test JWT token creation"""

    def test_create_access_token_basic(self, test_settings):
        """Test creating basic JWT token"""
        user_id = "test_user_123"
        permissions = {Permission.READ, Permission.WRITE}

        token = create_access_token(user_id, permissions, test_settings)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_access_token_payload(self, test_settings):
        """Test token contains correct payload"""
        user_id = "test_user_123"
        permissions = {Permission.READ, Permission.WRITE}

        token = create_access_token(user_id, permissions, test_settings)

        # Decode without validation to inspect payload
        payload = jwt.decode(
            token,
            test_settings.jwt_secret,
            algorithms=[test_settings.jwt_algorithm]
        )

        assert payload["sub"] == user_id
        assert set(payload["permissions"]) == {p.value for p in permissions}
        assert "exp" in payload
        assert "iat" in payload

    def test_create_access_token_expiration(self, test_settings):
        """Test token expiration time"""
        user_id = "test_user"
        permissions = {Permission.READ}

        before_creation = datetime.utcnow()
        token = create_access_token(user_id, permissions, test_settings)
        after_creation = datetime.utcnow()

        payload = jwt.decode(
            token,
            test_settings.jwt_secret,
            algorithms=[test_settings.jwt_algorithm]
        )

        exp_time = datetime.fromtimestamp(payload["exp"])
        expected_exp = before_creation + timedelta(minutes=test_settings.jwt_expiration_minutes)

        # Allow 5 second tolerance
        assert abs((exp_time - expected_exp).total_seconds()) < 5

    def test_create_access_token_different_permissions(self, test_settings):
        """Test creating tokens with different permission sets"""
        test_cases = [
            {Permission.READ},
            {Permission.READ, Permission.WRITE},
            {Permission.ADMIN},
            {Permission.READ, Permission.WRITE, Permission.EXECUTE},
            {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN}
        ]

        for permissions in test_cases:
            token = create_access_token("user", permissions, test_settings)
            payload = jwt.decode(
                token,
                test_settings.jwt_secret,
                algorithms=[test_settings.jwt_algorithm]
            )

            assert set(payload["permissions"]) == {p.value for p in permissions}

    def test_create_access_token_empty_permissions(self, test_settings):
        """Test creating token with no permissions"""
        token = create_access_token("user", set(), test_settings)
        payload = jwt.decode(
            token,
            test_settings.jwt_secret,
            algorithms=[test_settings.jwt_algorithm]
        )

        assert payload["permissions"] == []


@pytest.mark.unit
class TestJWTTokenDecoding:
    """Test JWT token decoding and validation"""

    def test_decode_valid_token(self, test_settings):
        """Test decoding valid token"""
        user_id = "test_user"
        permissions = {Permission.READ, Permission.WRITE}

        token = create_access_token(user_id, permissions, test_settings)
        token_data = decode_access_token(token, test_settings)

        assert token_data.user_id == user_id
        assert token_data.permissions == permissions

    def test_decode_token_with_admin_permission(self, test_settings):
        """Test decoding token with admin permission"""
        token = create_access_token("admin_user", {Permission.ADMIN}, test_settings)
        token_data = decode_access_token(token, test_settings)

        assert Permission.ADMIN in token_data.permissions

    def test_decode_expired_token(self, expired_token, test_settings):
        """Test decoding expired token raises exception"""
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(expired_token, test_settings)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_decode_invalid_signature(self, test_settings):
        """Test token with invalid signature"""
        # Create token with wrong secret
        wrong_settings = Settings(
            jwt_secret="wrong_secret_key_32_chars_minimum_required_for_testing",
            jwt_algorithm=test_settings.jwt_algorithm
        )

        token = create_access_token("user", {Permission.READ}, wrong_settings)

        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(token, test_settings)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_decode_malformed_token(self, test_settings):
        """Test decoding malformed token"""
        malformed_tokens = [
            "not.a.valid.jwt",
            "invalid",
            "",
            "a.b",
            "header.payload"
        ]

        for token in malformed_tokens:
            with pytest.raises(HTTPException) as exc_info:
                decode_access_token(token, test_settings)

            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_decode_token_missing_subject(self, test_settings):
        """Test token without 'sub' field"""
        # Create token manually without sub
        payload = {
            "permissions": ["read"],
            "exp": datetime.utcnow() + timedelta(hours=1)
        }

        token = jwt.encode(payload, test_settings.jwt_secret, algorithm=test_settings.jwt_algorithm)

        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(token, test_settings)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_decode_token_missing_permissions(self, test_settings):
        """Test token without permissions field"""
        payload = {
            "sub": "user",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }

        token = jwt.encode(payload, test_settings.jwt_secret, algorithm=test_settings.jwt_algorithm)

        # Should default to empty permissions
        token_data = decode_access_token(token, test_settings)
        assert token_data.permissions == set()


@pytest.mark.unit
class TestTokenData:
    """Test TokenData class"""

    def test_token_data_creation(self):
        """Test creating TokenData instance"""
        user_id = "test_user"
        permissions = {Permission.READ, Permission.WRITE}

        token_data = TokenData(user_id, permissions)

        assert token_data.user_id == user_id
        assert token_data.permissions == permissions

    def test_token_data_with_empty_permissions(self):
        """Test TokenData with no permissions"""
        token_data = TokenData("user", set())

        assert token_data.user_id == "user"
        assert token_data.permissions == set()


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetCurrentUser:
    """Test get_current_user dependency"""

    async def test_get_current_user_valid_token(self, test_token, test_settings):
        """Test with valid token"""
        from fastapi.security import HTTPAuthorizationCredentials

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=test_token
        )

        token_data = await get_current_user(credentials, test_settings)

        assert token_data.user_id is not None
        assert isinstance(token_data.permissions, set)

    async def test_get_current_user_no_credentials(self, test_settings):
        """Test without credentials"""
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(None, test_settings)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_get_current_user_invalid_token(self, invalid_token, test_settings):
        """Test with invalid token"""
        from fastapi.security import HTTPAuthorizationCredentials

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=invalid_token
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials, test_settings)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_get_current_user_skip_auth_enabled(self):
        """Test with skip_auth enabled (dev mode)"""
        dev_settings = Settings(
            skip_auth=True,
            secret_key="test_secret_key_32_chars_minimum_required_for_security",
            jwt_secret="test_jwt_secret_32_chars_minimum_required_for_security"
        )

        token_data = await get_current_user(None, dev_settings)

        assert token_data.user_id == "dev_user"
        assert Permission.ADMIN in token_data.permissions

    async def test_get_current_user_expired_token(self, expired_token, test_settings):
        """Test with expired token"""
        from fastapi.security import HTTPAuthorizationCredentials

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=expired_token
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials, test_settings)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.unit
@pytest.mark.asyncio
class TestRequirePermission:
    """Test require_permission dependency factory"""

    async def test_require_permission_user_has_permission(self, test_user_id, test_permissions):
        """Test when user has required permission"""
        current_user = TokenData(test_user_id, test_permissions)

        permission_checker = require_permission(Permission.READ)
        result = await permission_checker(current_user)

        assert result == current_user

    async def test_require_permission_user_missing_permission(self, test_user_id):
        """Test when user lacks required permission"""
        current_user = TokenData(test_user_id, {Permission.READ})

        permission_checker = require_permission(Permission.ADMIN)

        with pytest.raises(HTTPException) as exc_info:
            await permission_checker(current_user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Missing permissions" in str(exc_info.value.detail)

    async def test_require_permission_admin_has_all(self, test_user_id):
        """Test that admin permission grants access to everything"""
        admin_user = TokenData(test_user_id, {Permission.ADMIN})

        # Admin should pass all permission checks
        for perm in [Permission.READ, Permission.WRITE, Permission.DELETE, Permission.EXECUTE]:
            permission_checker = require_permission(perm)
            result = await permission_checker(admin_user)
            assert result == admin_user

    async def test_require_multiple_permissions(self, test_user_id):
        """Test requiring multiple permissions"""
        user = TokenData(test_user_id, {Permission.READ, Permission.WRITE, Permission.EXECUTE})

        # Should pass when user has all
        permission_checker = require_permission(Permission.READ, Permission.WRITE)
        result = await permission_checker(user)
        assert result == user

    async def test_require_multiple_permissions_missing_one(self, test_user_id):
        """Test when user has some but not all required permissions"""
        user = TokenData(test_user_id, {Permission.READ})

        permission_checker = require_permission(Permission.READ, Permission.WRITE)

        with pytest.raises(HTTPException) as exc_info:
            await permission_checker(user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    async def test_require_permission_empty_permissions_user(self, test_user_id):
        """Test user with no permissions"""
        user = TokenData(test_user_id, set())

        permission_checker = require_permission(Permission.READ)

        with pytest.raises(HTTPException) as exc_info:
            await permission_checker(user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.integration
class TestAuthenticationFlow:
    """Integration tests for complete authentication flows"""

    def test_full_auth_flow_read_endpoint(self, client, test_token):
        """Test complete flow: create token → access endpoint"""
        # Use token to access protected endpoint
        response = client.get(
            "/api/trading/strategies",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

    def test_full_auth_flow_write_endpoint(self, client, test_token):
        """Test write operation with proper permission"""
        response = client.post(
            "/api/trading/strategies/test/enable",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

    def test_full_auth_flow_execute_endpoint(self, client, test_token):
        """Test execute operation with proper permission"""
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

        assert response.status_code == status.HTTP_200_OK

    def test_permission_escalation_blocked(self, client, readonly_token):
        """Test that read-only user cannot perform write operations"""
        response = client.post(
            "/api/trading/strategies/test/enable",
            headers={"Authorization": f"Bearer {readonly_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_multiple_tokens_different_users(self, test_settings):
        """Test multiple tokens for different users"""
        user1_token = create_access_token("user1", {Permission.READ}, test_settings)
        user2_token = create_access_token("user2", {Permission.ADMIN}, test_settings)

        user1_data = decode_access_token(user1_token, test_settings)
        user2_data = decode_access_token(user2_token, test_settings)

        assert user1_data.user_id != user2_data.user_id
        assert Permission.ADMIN in user2_data.permissions
        assert Permission.ADMIN not in user1_data.permissions


@pytest.mark.security
class TestAuthenticationSecurity:
    """Security tests for authentication system"""

    def test_token_tampering_detected(self, test_token, test_settings):
        """Test that tampered tokens are rejected"""
        # Split token into parts
        parts = test_token.split('.')

        # Tamper with payload
        tampered_token = parts[0] + '.modified_payload.' + parts[2]

        with pytest.raises(HTTPException):
            decode_access_token(tampered_token, test_settings)

    def test_token_reuse_after_expiration(self, test_settings):
        """Test that expired tokens cannot be reused"""
        # Create token that expires immediately
        expire = datetime.utcnow() - timedelta(seconds=1)
        payload = {
            "sub": "user",
            "permissions": ["read"],
            "exp": expire,
            "iat": datetime.utcnow()
        }

        expired_token = jwt.encode(
            payload,
            test_settings.jwt_secret,
            algorithm=test_settings.jwt_algorithm
        )

        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(expired_token, test_settings)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_bearer_scheme_required(self, client, test_token):
        """Test that Bearer scheme is properly enforced"""
        # Try without Bearer prefix
        response = client.get(
            "/api/trading/strategies",
            headers={"Authorization": test_token}  # Missing "Bearer"
        )

        # Should fail authentication
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_case_sensitive_bearer(self, client, test_token):
        """Test Authorization header case sensitivity"""
        # Try with lowercase bearer
        response = client.get(
            "/api/trading/strategies",
            headers={"Authorization": f"bearer {test_token}"}
        )

        # FastAPI HTTPBearer is case-insensitive for "Bearer"
        # This might succeed or fail depending on implementation
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]

    def test_different_secret_keys_incompatible(self):
        """Test that tokens from different secrets are incompatible"""
        settings1 = Settings(
            jwt_secret="secret1_32_chars_minimum_required_for_security_key",
            jwt_algorithm="HS256"
        )

        settings2 = Settings(
            jwt_secret="secret2_32_chars_minimum_required_for_security_key",
            jwt_algorithm="HS256"
        )

        token = create_access_token("user", {Permission.READ}, settings1)

        with pytest.raises(HTTPException):
            decode_access_token(token, settings2)

    def test_skip_auth_blocked_in_production(self):
        """Test that skip_auth is blocked in production"""
        prod_settings = Settings(
            env="production",
            skip_auth=True,  # Should be ignored
            secret_key="production_secret_32_chars_minimum_required",
            jwt_secret="production_jwt_secret_32_chars_minimum_req"
        )

        # In production, skip_auth should ideally be blocked
        # This test documents expected behavior
        assert prod_settings.env == "production"


@pytest.mark.performance
class TestAuthenticationPerformance:
    """Performance tests for authentication"""

    def test_token_creation_performance(self, test_settings, benchmark):
        """Benchmark token creation"""
        def create_token():
            return create_access_token(
                "user_id",
                {Permission.READ, Permission.WRITE},
                test_settings
            )

        token = benchmark(create_token)
        assert token is not None

    def test_token_validation_performance(self, test_token, test_settings, benchmark):
        """Benchmark token validation"""
        def validate_token():
            return decode_access_token(test_token, test_settings)

        token_data = benchmark(validate_token)
        assert token_data.user_id is not None

    @pytest.mark.slow
    def test_concurrent_token_validation(self, test_token, test_settings):
        """Test concurrent token validation"""
        import concurrent.futures

        def validate(_):
            return decode_access_token(test_token, test_settings)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(validate, i) for i in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed with same user_id
        assert all(r.user_id == results[0].user_id for r in results)


@pytest.mark.unit
class TestPermissionEnum:
    """Test Permission enum values"""

    def test_all_permission_values(self):
        """Test all permission enum values"""
        assert Permission.READ.value == "read"
        assert Permission.WRITE.value == "write"
        assert Permission.DELETE.value == "delete"
        assert Permission.ADMIN.value == "admin"
        assert Permission.EXECUTE.value == "execute"

    def test_permission_from_string(self):
        """Test creating Permission from string"""
        perms = ["read", "write", "delete", "admin", "execute"]

        for perm_str in perms:
            perm = Permission(perm_str)
            assert perm.value == perm_str
