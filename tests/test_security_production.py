"""
CRITICAL SECURITY TESTS - Production Ready Security Implementation
Tests for Task 1-4: Secret keys, skip_auth blocking, and password validation
These tests MUST pass before production deployment
"""

import pytest
import os
from unittest.mock import patch
from pydantic import ValidationError

from config.settings import Settings, Environment
from core.security.hardening import (
    SecurityHardeningEngine,
    PasswordValidator,
    Permission,
)


# ============================================================================
# Task 1 & 2: Secret Key Validation Tests
# ============================================================================

class TestSecretKeyValidation:
    """Test secret key validation - CRITICAL for production security"""

    def test_secret_key_rejects_insecure_defaults(self):
        """CRITICAL: Secret key must reject all insecure default values"""
        insecure_values = [
            "INSECURE_CHANGE_IN_PRODUCTION",
            "insecure_anything",
            "changeme_password",
            "secret_key_123",
            "default_value",
        ]

        for insecure_value in insecure_values:
            with pytest.raises(ValidationError, match="SECURITY VIOLATION"):
                with patch.dict(os.environ, {"SHIVX_SECRET_KEY": insecure_value}):
                    Settings()

    def test_secret_key_minimum_length_enforced(self):
        """CRITICAL: Secret key must be at least 32 characters"""
        short_key = "short_key_only_20char"  # 21 chars

        with pytest.raises(ValidationError, match="too short"):
            with patch.dict(os.environ, {"SHIVX_SECRET_KEY": short_key}):
                Settings()

    def test_secret_key_entropy_check(self):
        """CRITICAL: Secret key must have sufficient character diversity"""
        weak_key = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"  # 34 chars but low entropy

        with pytest.raises(ValidationError, match="insufficient entropy"):
            with patch.dict(os.environ, {"SHIVX_SECRET_KEY": weak_key}):
                Settings()

    def test_secret_key_production_length_requirement(self):
        """CRITICAL: In production, secret key must be at least 48 characters"""
        # 40 character key (would pass in dev but not production)
        medium_key = "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8s9T0"  # 40 chars

        with pytest.raises(ValidationError, match="must be at least 48 chars"):
            with patch.dict(os.environ, {
                "SHIVX_ENV": "production",
                "SHIVX_SECRET_KEY": medium_key
            }):
                Settings()

    def test_secret_key_accepts_valid_cryptographic_key(self):
        """Verify valid cryptographic keys are accepted"""
        valid_key = "zZi3aYpv7w-zA2dIvXCCUJUhIu9YpULFXO3R9f2St71tFfAl1xn5dR0Re7xO09aw"

        with patch.dict(os.environ, {"SHIVX_SECRET_KEY": valid_key}):
            settings = Settings()
            assert settings.secret_key == valid_key

    def test_jwt_secret_rejects_insecure_defaults(self):
        """CRITICAL: JWT secret must reject insecure defaults"""
        insecure_values = [
            "INSECURE_JWT_CHANGE_IN_PRODUCTION",
            "insecure_jwt_secret",
            "jwt_changeme",
            "default_jwt_secret",
        ]

        for insecure_value in insecure_values:
            with pytest.raises(ValidationError, match="SECURITY VIOLATION"):
                with patch.dict(os.environ, {"SHIVX_JWT_SECRET": insecure_value}):
                    Settings()

    def test_jwt_secret_must_differ_from_secret_key(self):
        """CRITICAL: JWT secret must be different from main secret key"""
        same_secret = "ValidSecret123WithGoodEntropyAndLength456789"

        with pytest.raises(ValidationError, match="must be different from secret_key"):
            with patch.dict(os.environ, {
                "SHIVX_SECRET_KEY": same_secret,
                "SHIVX_JWT_SECRET": same_secret
            }):
                Settings()

    def test_jwt_secret_accepts_valid_different_key(self):
        """Verify JWT secret accepts valid key different from secret_key"""
        secret_key = "zZi3aYpv7w-zA2dIvXCCUJUhIu9YpULFXO3R9f2St71tFfAl1xn5dR0Re7xO09aw"
        jwt_secret = "-M09hJ0D1THK8JvYG9BwfCT2kb7OnR3ihcy44oke4Loaqc_utvzEFCNEkEO4MJl-"

        with patch.dict(os.environ, {
            "SHIVX_SECRET_KEY": secret_key,
            "SHIVX_JWT_SECRET": jwt_secret
        }):
            settings = Settings()
            assert settings.secret_key != settings.jwt_secret
            assert len(settings.secret_key) >= 32
            assert len(settings.jwt_secret) >= 32


# ============================================================================
# Task 3: skip_auth Protection Tests
# ============================================================================

class TestSkipAuthProtection:
    """Test skip_auth blocking - CRITICAL for production security"""

    def test_skip_auth_blocked_in_production(self):
        """CRITICAL: skip_auth MUST be blocked in production environment"""
        with pytest.raises(ValidationError, match="skip_auth cannot be enabled in production"):
            with patch.dict(os.environ, {
                "SHIVX_ENV": "production",
                "SHIVX_SKIP_AUTH": "true"
            }):
                Settings()

    def test_skip_auth_blocked_in_staging(self):
        """CRITICAL: skip_auth MUST be blocked in staging environment"""
        with pytest.raises(ValidationError, match="skip_auth cannot be enabled in staging"):
            with patch.dict(os.environ, {
                "SHIVX_ENV": "staging",
                "SHIVX_SKIP_AUTH": "true"
            }):
                Settings()

    def test_skip_auth_allowed_in_local(self):
        """Verify skip_auth is allowed in local environment (for dev)"""
        with patch.dict(os.environ, {
            "SHIVX_ENV": "local",
            "SHIVX_SKIP_AUTH": "true"
        }):
            settings = Settings()
            assert settings.skip_auth is True

    def test_skip_auth_allowed_in_development(self):
        """Verify skip_auth is allowed in development environment"""
        with patch.dict(os.environ, {
            "SHIVX_ENV": "development",
            "SHIVX_SKIP_AUTH": "true"
        }):
            settings = Settings()
            assert settings.skip_auth is True

    def test_skip_auth_default_is_false(self):
        """Verify skip_auth defaults to False (secure by default)"""
        settings = Settings()
        assert settings.skip_auth is False

    @pytest.mark.asyncio
    async def test_get_current_user_blocks_skip_auth_in_production(self):
        """CRITICAL: Even if skip_auth somehow becomes True, auth.py blocks it"""
        from app.dependencies.auth import get_current_user
        from fastapi import HTTPException

        # Create settings with skip_auth (in dev mode where it's allowed)
        with patch.dict(os.environ, {"SHIVX_ENV": "local", "SHIVX_SKIP_AUTH": "true"}):
            settings = Settings()
            assert settings.skip_auth is True

            # Now manually override env to production (simulating bypass attempt)
            settings.env = Environment.PRODUCTION

            # Should raise security error, not allow access
            with pytest.raises(HTTPException, match="Security configuration error"):
                await get_current_user(credentials=None, settings=settings)


# ============================================================================
# Task 4: Password Validation Tests
# ============================================================================

class TestPasswordValidation:
    """Test password validation - CRITICAL for user security"""

    def test_password_minimum_length_12_chars(self):
        """CRITICAL: Password must be at least 12 characters"""
        short_passwords = [
            "Short1!",  # 7 chars
            "Password1!",  # 11 chars
        ]

        for pwd in short_passwords:
            is_valid, error = PasswordValidator.validate_password(pwd)
            assert not is_valid
            assert "at least 12 characters" in error

    def test_password_requires_uppercase(self):
        """CRITICAL: Password must contain uppercase letter"""
        pwd = "lowercase123!@#"
        is_valid, error = PasswordValidator.validate_password(pwd)
        assert not is_valid
        assert "uppercase letter" in error

    def test_password_requires_lowercase(self):
        """CRITICAL: Password must contain lowercase letter"""
        pwd = "UPPERCASE123!@#"
        is_valid, error = PasswordValidator.validate_password(pwd)
        assert not is_valid
        assert "lowercase letter" in error

    def test_password_requires_digit(self):
        """CRITICAL: Password must contain digit"""
        pwd = "NoDigitsHere!@#"
        is_valid, error = PasswordValidator.validate_password(pwd)
        assert not is_valid
        assert "digit" in error

    def test_password_requires_special_char(self):
        """CRITICAL: Password must contain special character"""
        pwd = "NoSpecialChar123"
        is_valid, error = PasswordValidator.validate_password(pwd)
        assert not is_valid
        assert "special character" in error

    def test_password_rejects_common_weak_passwords(self):
        """CRITICAL: Reject common weak passwords"""
        weak_passwords = [
            "Password123!",  # Contains "password"
            "Admin123456!",  # Contains "admin"
            "Welcome123!@",  # Contains "welcome"
            "Qwerty123456!",  # Contains "qwerty"
        ]

        for pwd in weak_passwords:
            is_valid, error = PasswordValidator.validate_password(pwd)
            assert not is_valid
            assert "weak pattern" in error.lower()

    def test_password_rejects_sequential_chars(self):
        """CRITICAL: Reject passwords with sequential characters"""
        sequential_passwords = [
            "Abc123456!@#",  # "abc" sequence
            "Password987!",  # "987" sequence
        ]

        for pwd in sequential_passwords:
            is_valid, error = PasswordValidator.validate_password(pwd)
            assert not is_valid
            assert "sequential" in error.lower()

    def test_password_rejects_repeated_chars(self):
        """CRITICAL: Reject passwords with too many repeated characters"""
        pwd = "Paaaassword123!"  # "aaa" repeated
        is_valid, error = PasswordValidator.validate_password(pwd)
        assert not is_valid
        assert "repeated" in error.lower()

    def test_password_requires_character_diversity(self):
        """CRITICAL: Password must have minimum character diversity"""
        pwd = "Aa1!Aa1!Aa1!Aa1!"  # Only 5 unique chars
        is_valid, error = PasswordValidator.validate_password(pwd)
        assert not is_valid
        assert "8 different characters" in error

    def test_password_accepts_strong_password(self):
        """Verify strong passwords are accepted"""
        strong_passwords = [
            "MyStr0ng!P@ssw0rd",
            "C0mpl3x!Pass#2024",
            "Secur3$Tr@ding!Key",
            "Bitt3r$weet#Coffee99",
        ]

        for pwd in strong_passwords:
            is_valid, error = PasswordValidator.validate_password(pwd)
            assert is_valid, f"Password {pwd} should be valid, got error: {error}"
            assert error == ""

    def test_password_strength_scoring(self):
        """Test password strength scoring"""
        # Weak password
        weak = "Weak123!"
        weak_score = PasswordValidator.get_password_strength_score(weak)
        assert weak_score < 50

        # Strong password
        strong = "MyV3ry$tr0ng!P@ssw0rd2024"
        strong_score = PasswordValidator.get_password_strength_score(strong)
        assert strong_score >= 70


# ============================================================================
# Integration Tests
# ============================================================================

class TestUserCreationWithPasswordValidation:
    """Test that user creation enforces password validation"""

    def test_create_user_rejects_weak_password(self):
        """CRITICAL: User creation must reject weak passwords"""
        engine = SecurityHardeningEngine()

        with pytest.raises(ValueError, match="Password validation failed"):
            engine.auth.create_user(
                "testuser",
                "weak",  # Too short, no uppercase, no special char
                permissions={Permission.READ}
            )

    def test_create_user_accepts_strong_password(self):
        """Verify user creation works with strong password"""
        engine = SecurityHardeningEngine()

        user = engine.auth.create_user(
            "testuser",
            "MyStr0ng!P@ssw0rd",
            permissions={Permission.READ}
        )

        assert user is not None
        assert user.username == "testuser"

    def test_create_user_password_not_stored_plaintext(self):
        """CRITICAL: Ensure password is hashed, not stored as plaintext"""
        engine = SecurityHardeningEngine()

        password = "MyStr0ng!P@ssw0rd"
        user = engine.auth.create_user(
            "testuser",
            password,
            permissions={Permission.READ}
        )

        # Password hash should not equal plaintext password
        assert password not in user.password_hash
        assert ":" in user.password_hash  # Should have hash:salt format

    @pytest.mark.asyncio
    async def test_authentication_with_strong_password(self):
        """Test full authentication flow with strong password"""
        engine = SecurityHardeningEngine()

        password = "MyStr0ng!P@ssw0rd"
        user = engine.auth.create_user(
            "testuser",
            password,
            permissions={Permission.READ}
        )

        # Should authenticate successfully
        token = engine.auth.authenticate_user("testuser", password)
        assert token is not None

        # Should reject wrong password
        wrong_token = engine.auth.authenticate_user("testuser", "WrongP@ssw0rd123")
        assert wrong_token is None


# ============================================================================
# Regression Tests
# ============================================================================

class TestBackwardCompatibility:
    """Ensure changes don't break existing functionality"""

    def test_settings_can_still_be_instantiated(self):
        """Verify Settings class can be instantiated with valid values"""
        settings = Settings()
        assert settings is not None
        assert settings.env == Environment.LOCAL

    def test_jwt_token_creation_still_works(self):
        """Verify JWT token creation still functions"""
        from app.dependencies.auth import create_access_token

        settings = Settings()
        token = create_access_token(
            "test_user",
            {Permission.READ, Permission.WRITE},
            settings
        )

        assert token is not None
        assert len(token) > 0

    def test_encryption_manager_still_works(self):
        """Verify encryption functionality still works"""
        from core.security.hardening import EncryptionManager

        manager = EncryptionManager()
        plaintext = "sensitive_data"
        encrypted = manager.encrypt(plaintext)
        decrypted = manager.decrypt(encrypted)

        assert plaintext == decrypted


# ============================================================================
# Production Environment Validation
# ============================================================================

class TestProductionEnvironmentValidation:
    """Comprehensive production environment security validation"""

    def test_production_environment_full_validation(self):
        """CRITICAL: Validate all security settings work in production"""
        valid_secret = "zZi3aYpv7w-zA2dIvXCCUJUhIu9YpULFXO3R9f2St71tFfAl1xn5dR0Re7xO09aw"
        valid_jwt = "-M09hJ0D1THK8JvYG9BwfCT2kb7OnR3ihcy44oke4Loaqc_utvzEFCNEkEO4MJl-"

        with patch.dict(os.environ, {
            "SHIVX_ENV": "production",
            "SHIVX_SECRET_KEY": valid_secret,
            "SHIVX_JWT_SECRET": valid_jwt,
            "SHIVX_SKIP_AUTH": "false"
        }):
            settings = Settings()
            assert settings.env == Environment.PRODUCTION
            assert settings.skip_auth is False
            assert len(settings.secret_key) >= 48
            assert len(settings.jwt_secret) >= 48
            assert settings.secret_key != settings.jwt_secret

    def test_staging_environment_full_validation(self):
        """Validate all security settings work in staging"""
        valid_secret = "zZi3aYpv7w-zA2dIvXCCUJUhIu9YpULFXO3R9f2St71tFfAl1xn5dR0Re7xO09aw"
        valid_jwt = "-M09hJ0D1THK8JvYG9BwfCT2kb7OnR3ihcy44oke4Loaqc_utvzEFCNEkEO4MJl-"

        with patch.dict(os.environ, {
            "SHIVX_ENV": "staging",
            "SHIVX_SECRET_KEY": valid_secret,
            "SHIVX_JWT_SECRET": valid_jwt,
            "SHIVX_SKIP_AUTH": "false"
        }):
            settings = Settings()
            assert settings.env == Environment.STAGING
            assert settings.skip_auth is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
