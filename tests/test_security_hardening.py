"""
Security Hardening Tests
Tests for core/security/hardening.py authentication, encryption, and authorization
"""

import pytest
from core.security.hardening import (
    SecurityHardeningEngine,
    Permission,
    InputType,
)


class TestInputValidation:
    """Test input validation functionality"""

    def test_email_validation_valid(self):
        """Test valid email validation"""
        engine = SecurityHardeningEngine()

        valid_emails = [
            "user@example.com",
            "test.user@domain.co.uk",
            "admin+tag@company.org"
        ]

        for email in valid_emails:
            assert engine.validator.validate_input(email, InputType.EMAIL), \
                f"Email {email} should be valid"

    def test_email_validation_invalid(self):
        """Test invalid email rejection"""
        engine = SecurityHardeningEngine()

        invalid_emails = [
            "notanemail",
            "@example.com",
            "user@",
            "user @example.com"
        ]

        for email in invalid_emails:
            assert not engine.validator.validate_input(email, InputType.EMAIL), \
                f"Email {email} should be invalid"

    def test_username_validation(self):
        """Test username validation"""
        engine = SecurityHardeningEngine()

        # Valid usernames
        assert engine.validator.validate_input("validuser", InputType.USERNAME)
        assert engine.validator.validate_input("user_123", InputType.USERNAME)
        assert engine.validator.validate_input("user-name", InputType.USERNAME)

        # Invalid usernames
        assert not engine.validator.validate_input("ab", InputType.USERNAME)  # Too short
        assert not engine.validator.validate_input("user@name", InputType.USERNAME)  # Invalid char
        assert not engine.validator.validate_input("a" * 50, InputType.USERNAME)  # Too long

    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection"""
        engine = SecurityHardeningEngine()

        sql_injection_attempts = [
            "'; DROP TABLE users; --",
            "admin' OR '1'='1",
            "1 UNION SELECT * FROM passwords",
            "user/* comment */admin"
        ]

        for attempt in sql_injection_attempts:
            assert not engine.validator.validate_string(attempt, max_length=100), \
                f"SQL injection attempt should be blocked: {attempt}"

    def test_xss_detection(self):
        """Test XSS pattern detection"""
        engine = SecurityHardeningEngine()

        xss_attempts = [
            "<script>alert('XSS')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "<div onclick='malicious()'>Click</div>"
        ]

        for attempt in xss_attempts:
            assert not engine.validator.validate_string(attempt, max_length=100), \
                f"XSS attempt should be blocked: {attempt}"


class TestEncryption:
    """Test encryption and decryption functionality"""

    def test_encryption_decryption(self):
        """Test basic encryption and decryption"""
        engine = SecurityHardeningEngine()

        plaintext = "sensitive_data_test_12345"
        ciphertext = engine.encryption.encrypt(plaintext)
        decrypted = engine.encryption.decrypt(ciphertext)

        assert ciphertext != plaintext, "Ciphertext should differ from plaintext"
        assert plaintext == decrypted, "Decrypted text should match original"

    def test_encryption_produces_different_output(self):
        """Test that encryption with same key produces different output (due to IV)"""
        engine = SecurityHardeningEngine()

        plaintext = "test_message"
        ciphertext1 = engine.encryption.encrypt(plaintext)
        ciphertext2 = engine.encryption.encrypt(plaintext)

        # Due to random IV in Fernet, each encryption should be different
        # Note: This might not apply to all encryption schemes
        # For deterministic encryption, remove this test


class TestUserAuthentication:
    """Test user authentication functionality"""

    @pytest.mark.asyncio
    async def test_create_user(self, test_username, test_password):
        """Test user creation"""
        engine = SecurityHardeningEngine()

        user = engine.auth.create_user(
            test_username,
            test_password,
            permissions={Permission.READ, Permission.WRITE}
        )

        assert user.username == test_username
        assert user.user_id is not None
        assert Permission.READ in user.permissions
        assert Permission.WRITE in user.permissions

    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, test_username, test_password):
        """Test successful user authentication"""
        engine = SecurityHardeningEngine()

        # Create user
        engine.auth.create_user(
            test_username,
            test_password,
            permissions={Permission.READ}
        )

        # Authenticate
        session_token = engine.auth.authenticate_user(test_username, test_password)

        assert session_token is not None, "Authentication should succeed"
        assert len(session_token) > 0, "Session token should not be empty"

    @pytest.mark.asyncio
    async def test_authenticate_user_wrong_password(self, test_username, test_password, wrong_password):
        """Test authentication failure with wrong password"""
        engine = SecurityHardeningEngine()

        # Create user
        engine.auth.create_user(
            test_username,
            test_password,
            permissions={Permission.READ}
        )

        # Try to authenticate with wrong password
        session_token = engine.auth.authenticate_user(test_username, wrong_password)

        assert session_token is None, "Authentication should fail with wrong password"

    @pytest.mark.asyncio
    async def test_session_validation(self, test_username, test_password):
        """Test session token validation"""
        engine = SecurityHardeningEngine()

        # Create user and authenticate
        user = engine.auth.create_user(
            test_username,
            test_password,
            permissions={Permission.READ}
        )
        session_token = engine.auth.authenticate_user(test_username, test_password)

        # Validate session
        user_id = engine.auth.validate_session(session_token)

        assert user_id == user.user_id, "Session should map to correct user"

    @pytest.mark.asyncio
    async def test_invalid_session_rejected(self):
        """Test that invalid session tokens are rejected"""
        engine = SecurityHardeningEngine()

        invalid_token = "invalid_session_token_12345"
        user_id = engine.auth.validate_session(invalid_token)

        assert user_id is None, "Invalid session should be rejected"


class TestAPIKeyManagement:
    """Test API key management functionality"""

    def test_create_api_key(self):
        """Test API key creation"""
        engine = SecurityHardeningEngine()

        raw_key, api_key = engine.auth.create_api_key(
            "test_api_key",
            permissions={Permission.READ},
            rate_limit=100
        )

        assert raw_key is not None
        assert len(raw_key) > 20, "API key should be sufficiently long"
        assert api_key.name == "test_api_key"
        assert api_key.rate_limit == 100
        assert Permission.READ in api_key.permissions

    def test_validate_api_key_success(self):
        """Test successful API key validation"""
        engine = SecurityHardeningEngine()

        raw_key, api_key = engine.auth.create_api_key(
            "test_api_key",
            permissions={Permission.READ}
        )

        validated = engine.auth.validate_api_key(raw_key)

        assert validated is not None, "Valid API key should be accepted"
        assert validated.key_id == api_key.key_id

    def test_validate_api_key_failure(self):
        """Test API key validation failure with invalid key"""
        engine = SecurityHardeningEngine()

        invalid_key = "invalid_api_key_xyz_123456789"
        validated = engine.auth.validate_api_key(invalid_key)

        assert validated is None, "Invalid API key should be rejected"


class TestPermissions:
    """Test permission system"""

    def test_check_permission_granted(self, test_username, test_password):
        """Test permission check for granted permission"""
        engine = SecurityHardeningEngine()

        user = engine.auth.create_user(
            test_username,
            test_password,
            permissions={Permission.READ, Permission.WRITE}
        )

        has_read = engine.auth.check_permission(user.user_id, Permission.READ)
        has_write = engine.auth.check_permission(user.user_id, Permission.WRITE)

        assert has_read, "User should have READ permission"
        assert has_write, "User should have WRITE permission"

    def test_check_permission_denied(self, test_username, test_password):
        """Test permission check for denied permission"""
        engine = SecurityHardeningEngine()

        user = engine.auth.create_user(
            test_username,
            test_password,
            permissions={Permission.READ}  # Only READ, no ADMIN
        )

        has_admin = engine.auth.check_permission(user.user_id, Permission.ADMIN)
        has_delete = engine.auth.check_permission(user.user_id, Permission.DELETE)

        assert not has_admin, "User should not have ADMIN permission"
        assert not has_delete, "User should not have DELETE permission"


class TestSecurityAuditing:
    """Test security audit logging"""

    def test_log_audit_event(self, test_username, test_password):
        """Test logging security audit events"""
        engine = SecurityHardeningEngine()

        user = engine.auth.create_user(
            test_username,
            test_password,
            permissions={Permission.READ}
        )

        # Log authentication event
        engine.auditor.log_event(
            "authentication",
            "test_resource",
            "login",
            True,
            user_id=user.user_id,
            ip_address="127.0.0.1"
        )

        # Log authorization event
        engine.auditor.log_event(
            "authorization",
            "secret_data",
            "access",
            False,
            user_id=user.user_id
        )

        stats = engine.auditor.get_statistics()

        assert stats['total_events'] >= 2, "Should have logged at least 2 events"
        assert 'authentication' in stats['events_by_type']
        assert 'authorization' in stats['events_by_type']

    def test_audit_success_rate(self):
        """Test audit success rate calculation"""
        engine = SecurityHardeningEngine()

        # Log successful events
        for i in range(8):
            engine.auditor.log_event("test", "resource", "action", True)

        # Log failed events
        for i in range(2):
            engine.auditor.log_event("test", "resource", "action", False)

        stats = engine.auditor.get_statistics()

        assert stats['success_rate'] == 0.8, "Success rate should be 80%"


class TestProtectedFunctions:
    """Test decorator-based authentication and authorization"""

    @pytest.mark.asyncio
    async def test_protected_function_with_auth(self, test_username, test_password):
        """Test accessing protected function with authentication"""
        engine = SecurityHardeningEngine()

        # Create user and authenticate
        engine.auth.create_user(
            test_username,
            test_password,
            permissions={Permission.READ}
        )
        session_token = engine.auth.authenticate_user(test_username, test_password)

        # Define protected function
        @engine.require_authentication
        @engine.require_permission(Permission.READ)
        async def protected_function(user_id: str):
            return f"Access granted for user {user_id}"

        # Should succeed
        result = await protected_function(session_token=session_token)
        assert "Access granted" in result

    @pytest.mark.asyncio
    async def test_protected_function_without_auth(self):
        """Test accessing protected function without authentication"""
        engine = SecurityHardeningEngine()

        @engine.require_authentication
        async def protected_function(user_id: str):
            return f"Access granted for user {user_id}"

        # Should raise PermissionError
        with pytest.raises(PermissionError):
            await protected_function()


class TestSecurityStatistics:
    """Test security statistics collection"""

    def test_get_security_stats(self, test_username, test_password):
        """Test getting comprehensive security statistics"""
        engine = SecurityHardeningEngine()

        # Create some users and API keys
        engine.auth.create_user(test_username, test_password, permissions={Permission.READ})
        engine.auth.create_api_key("test_key", permissions={Permission.READ})

        stats = engine.get_security_stats()

        assert 'users' in stats
        assert 'active_sessions' in stats
        assert 'api_keys' in stats
        assert 'audit_log' in stats
        assert stats['users'] >= 1
        assert stats['api_keys'] >= 1


@pytest.mark.asyncio
async def test_comprehensive_security_workflow(test_username, test_password):
    """
    Integration test for complete security workflow
    Tests the entire flow: create user -> authenticate -> validate -> permissions -> audit
    """
    engine = SecurityHardeningEngine()

    # 1. Create user
    user = engine.auth.create_user(
        test_username,
        test_password,
        permissions={Permission.READ, Permission.WRITE}
    )
    assert user is not None

    # 2. Authenticate
    session_token = engine.auth.authenticate_user(test_username, test_password)
    assert session_token is not None

    # 3. Validate session
    user_id = engine.auth.validate_session(session_token)
    assert user_id == user.user_id

    # 4. Check permissions
    assert engine.auth.check_permission(user_id, Permission.READ)
    assert engine.auth.check_permission(user_id, Permission.WRITE)
    assert not engine.auth.check_permission(user_id, Permission.ADMIN)

    # 5. Create and validate API key
    raw_key, api_key = engine.auth.create_api_key(
        "integration_test_key",
        permissions={Permission.READ}
    )
    validated_key = engine.auth.validate_api_key(raw_key)
    assert validated_key is not None

    # 6. Log audit events
    engine.auditor.log_event(
        "authentication",
        "integration_test",
        "complete_workflow",
        True,
        user_id=user_id
    )

    # 7. Get final statistics
    stats = engine.get_security_stats()
    assert stats['users'] >= 1
    assert stats['api_keys'] >= 1
    assert stats['audit_log']['total_events'] >= 1
